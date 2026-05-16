from __future__ import annotations

import json
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any, Protocol

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent, Tool

from process_trace import NoopProcessTracer, ProcessTracer
from robot_control.call_validation import (
    AGENT_TO_LEGACY_MCP_TOOL_NAMES,
    ALLOWED_ROBOT_TOOLS,
    RobotCallValidationError,
    agent_tool_description,
    structured_robot_call_error,
    validate_robot_tool_call,
)

LEGACY_TO_AGENT_TOOL_NAMES = {legacy: agent for agent, legacy in AGENT_TO_LEGACY_MCP_TOOL_NAMES.items()}
TASK_SOLUTION_EXECUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "task_solution_id": {"type": "string"},
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "task_solution_id"],
    "additionalProperties": False,
}
TASK_PLAN_EXECUTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "task_solution_id": {"type": "string"},
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "task_solution_id"],
    "additionalProperties": False,
}
AGENT_TOOL_ORDER = {
    name: index
    for index, name in enumerate(
        [
            "moveit_get_current_pose",
            "moveit_get_robot_state",
            "moveit_list_scene_objects",
            "moveit_get_object_context",
            "moveit_plan_pick_task",
            "moveit_plan_place_task",
            "moveit_execute_task_plan",
            "moveit_execute_task_solution",
            "moveit_plan_pick",
            "moveit_plan_place",
            "moveit_plan_free_motion",
            "moveit_plan_cartesian_motion",
            "moveit_execute_plan",
            "moveit_explain_motion_failure",
            "moveit_verify_attached_object",
            "moveit_open_gripper",
            "moveit_close_gripper",
            "moveit_attach_object",
        ]
    )
}


class RobotMCPError(RuntimeError):
    """Raised when robot MCP tool setup or execution fails."""


class MCPServerLike(Protocol):
    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult: ...


class StreamableHttpMCPServer:
    """Small MCP Streamable HTTP client for robot tool discovery and calls."""

    def __init__(self, url: str, *, client_session_timeout_seconds: float = 30):
        self._url = url
        self._timeout = timedelta(seconds=client_session_timeout_seconds)
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        try:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self._url)
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._timeout,
                )
            )
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._session = session

    async def cleanup(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    async def list_tools(self) -> list[Tool]:
        if self._session is None:
            raise RobotMCPError("Robot MCP server is not connected")
        return list((await self._session.list_tools()).tools)

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        if self._session is None:
            raise RobotMCPError("Robot MCP server is not connected")
        return await self._session.call_tool(tool_name, arguments)


class RobotMCPBridge:
    """Converts robot MCP tools to LangChain function tools and executes validated calls."""

    def __init__(
        self,
        mcp_server_url: str,
        *,
        server: MCPServerLike | None = None,
        tracer: ProcessTracer | NoopProcessTracer | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._server = server or StreamableHttpMCPServer(mcp_server_url)
        self._tracer = tracer or NoopProcessTracer()
        self._tools: list[Tool] = []
        self._backing_tool_names: dict[str, str] = {}
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        async with self._tracer.span(
            "robot.mcp.connect",
            "robot_control",
            attributes={"mcp.url": self._mcp_server_url},
        ):
            await self._server.connect()
        async with self._tracer.span("robot.mcp.list_tools", "robot_control"):
            selected_tools: dict[str, Tool] = {}
            for tool in await self._server.list_tools():
                agent_name = self._agent_tool_name(tool.name)
                if agent_name is None:
                    continue
                existing = selected_tools.get(agent_name)
                if existing is None or tool.name == agent_name:
                    selected_tools[agent_name] = tool
        ordered_tools = sorted(selected_tools.items(), key=lambda item: _agent_tool_order(item[0]))
        self._tools = [tool for _, tool in ordered_tools]
        self._backing_tool_names = {agent_name: tool.name for agent_name, tool in ordered_tools}
        self._connected = True

    async def disconnect(self) -> None:
        await self._server.cleanup()
        self._connected = False
        self._tools = []
        self._backing_tool_names = {}

    def function_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        for tool in self._tools:
            agent_name = self._agent_tool_name(tool.name)
            if agent_name is None:
                continue
            tools.append(
                {
                    "type": "function",
                    "name": agent_name,
                    "description": agent_tool_description(agent_name),
                    "parameters": _agent_tool_schema(agent_name, tool.inputSchema),
                    "strict": None,
                }
            )
        if self._should_advertise_task_plan_execution():
            tools.append(
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
                    "description": agent_tool_description("moveit_execute_task_plan"),
                    "parameters": _agent_tool_schema(
                        "moveit_execute_task_plan",
                        TASK_PLAN_EXECUTION_SCHEMA,
                    ),
                    "strict": None,
                }
            )
        return sorted(tools, key=lambda tool: _agent_tool_order(str(tool.get("name") or "")))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        normalized_arguments = _normalize_agent_arguments(name, arguments)
        validation_attributes = self._tool_attributes(name, normalized_arguments)
        async with self._tracer.span(
            "robot.call_validation",
            "robot_control",
            attributes=validation_attributes,
        ):
            try:
                validate_robot_tool_call(name, normalized_arguments)
            except RobotCallValidationError as exc:
                blocked_attributes: dict[str, Any] = {
                    "tool.name": name,
                    "reason": str(exc),
                    "correction": exc.correction,
                }
                if self._tracer.options.include_tool_payloads:
                    blocked_attributes["tool.arguments"] = normalized_arguments
                self._tracer.event(
                    "robot.call_validation.blocked",
                    "robot_control",
                    attributes=blocked_attributes,
                )
                return _serialize_validation_failure(exc)

        backing_tool_name = self._backing_tool_names.get(name)
        if backing_tool_name is None:
            raise RobotMCPError(f"Tool is not allowed: {name}")
        call_attributes = self._tool_attributes(name, normalized_arguments)
        call_attributes["mcp.tool.name"] = backing_tool_name
        async with self._tracer.span(
            "robot.mcp.call_tool",
            "robot_control",
            attributes=call_attributes,
        ) as span:
            result = await self._server.call_tool(backing_tool_name, normalized_arguments)
            serialized_output = _serialize_tool_result(result)
            result_attributes = _tool_result_trace_attributes(result)
            span.update_attributes(result_attributes)
            if _tool_result_failed(result):
                span.set_status("failed-result")
            if self._tracer.options.include_tool_payloads:
                self._tracer.event(
                    "robot.mcp.tool_result",
                    "robot_control",
                    attributes={
                        "tool.name": name,
                        "mcp.tool.name": backing_tool_name,
                        "tool.result": serialized_output,
                    },
                )
            return serialized_output

    @staticmethod
    def _agent_tool_name(tool_name: str) -> str | None:
        if tool_name in ALLOWED_ROBOT_TOOLS:
            return tool_name
        return LEGACY_TO_AGENT_TOOL_NAMES.get(tool_name)

    def _tool_attributes(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        attributes: dict[str, Any] = {"tool.name": name}
        if self._tracer.options.include_tool_payloads:
            attributes["tool.arguments"] = arguments
        return attributes

    def _should_advertise_task_plan_execution(self) -> bool:
        return "moveit_plan_pick_task" in self._backing_tool_names


def _normalize_agent_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    if name != "moveit_plan_cartesian_motion":
        return arguments
    normalized = {key: value for key, value in arguments.items() if key not in {"points", "positions"}}
    if "waypoints" in normalized:
        return normalized
    points = arguments.get("points", arguments.get("positions"))
    if points is None:
        return arguments
    normalized["waypoints"] = points
    return normalized


def _agent_tool_schema(name: str, input_schema: dict[str, Any]) -> dict[str, Any]:
    if name == "moveit_execute_task_solution":
        return {
            "type": "object",
            "properties": dict(TASK_SOLUTION_EXECUTION_SCHEMA["properties"]),
            "required": list(TASK_SOLUTION_EXECUTION_SCHEMA["required"]),
            "additionalProperties": False,
        }
    if name == "moveit_execute_task_plan":
        return {
            "type": "object",
            "properties": dict(TASK_PLAN_EXECUTION_SCHEMA["properties"]),
            "required": list(TASK_PLAN_EXECUTION_SCHEMA["required"]),
            "additionalProperties": False,
        }
    return input_schema


def _agent_tool_order(name: str) -> int:
    return AGENT_TOOL_ORDER.get(name, len(AGENT_TOOL_ORDER))


def _serialize_validation_failure(exc: RobotCallValidationError) -> str:
    return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)


def _serialize_tool_result(result: CallToolResult) -> str:
    content: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            content.append(item.text)
        else:
            content.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    return json.dumps(
        {
            "content": content,
            "structured_content": result.structuredContent,
            "is_error": result.isError,
        },
        ensure_ascii=False,
    )


def _tool_result_failed(result: CallToolResult) -> bool:
    structured_content = result.structuredContent
    if isinstance(structured_content, dict) and structured_content.get("ok") is False:
        return True
    return result.isError is True


def _tool_result_trace_attributes(result: CallToolResult) -> dict[str, Any]:
    structured_content = result.structuredContent
    attributes: dict[str, Any] = {
        "mcp.transport.ok": True,
        "tool.result.is_error": result.isError,
    }
    if isinstance(structured_content, dict) and "ok" in structured_content:
        attributes["tool.result.ok"] = structured_content["ok"]
    if isinstance(structured_content, dict) and isinstance(structured_content.get("error"), str):
        attributes["tool.result.error"] = structured_content["error"]
    return attributes
