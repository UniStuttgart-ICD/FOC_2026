from __future__ import annotations

import json
from typing import Any, Protocol

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent, Tool

from voice_runtime.robot_safety import (
    AGENT_TO_LEGACY_MCP_TOOL_NAMES,
    RobotSafetyError,
    canonical_mcp_tool_name,
    validate_robot_tool_call,
)

ALLOWED_ROBOT_TOOLS = set(AGENT_TO_LEGACY_MCP_TOOL_NAMES)
LEGACY_TO_AGENT_TOOL_NAMES = {legacy: agent for agent, legacy in AGENT_TO_LEGACY_MCP_TOOL_NAMES.items()}


class RobotMCPError(RuntimeError):
    """Raised when robot MCP tool setup or execution fails."""


class MCPServerLike(Protocol):
    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult: ...


class RobotMCPBridge:
    """Converts robot MCP tools to Codex function tools and executes safe calls."""

    def __init__(self, mcp_server_url: str, *, server: MCPServerLike | None = None):
        self._server = server or MCPServerStreamableHttp(
            {"url": mcp_server_url},
            name="robot",
            cache_tools_list=True,
        )
        self._tools: list[Tool] = []
        self._connected = False

    async def connect(self) -> None:
        if self._connected:
            return
        await self._server.connect()
        self._tools = [tool for tool in await self._server.list_tools() if self._agent_tool_name(tool.name) is not None]
        self._connected = True

    async def disconnect(self) -> None:
        await self._server.cleanup()
        self._connected = False
        self._tools = []

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
                    "description": tool.description or "",
                    "parameters": tool.inputSchema,
                    "strict": None,
                }
            )
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            validate_robot_tool_call(name, arguments)
            mcp_tool_name = canonical_mcp_tool_name(name)
        except RobotSafetyError as exc:
            return _serialize_validation_failure(exc)

        if mcp_tool_name not in {tool.name for tool in self._tools}:
            raise RobotMCPError(f"Tool is not allowed: {name}")
        result = await self._server.call_tool(mcp_tool_name, arguments)
        return _serialize_tool_result(result)

    @staticmethod
    def _agent_tool_name(tool_name: str) -> str | None:
        if tool_name in ALLOWED_ROBOT_TOOLS:
            return tool_name
        return LEGACY_TO_AGENT_TOOL_NAMES.get(tool_name)


def _serialize_validation_failure(exc: RobotSafetyError) -> str:
    return json.dumps(
        {"error": str(exc), "correction": exc.correction},
        ensure_ascii=False,
    )


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
