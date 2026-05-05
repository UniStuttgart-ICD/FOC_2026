"""LangGraph orchestration for Codex robot agent turns."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from loguru import logger

from codex_backend_client import CodexResponseResult
from prompts import SYSTEM_PROMPT
from robot_control.task_policy import structured_task_policy_error, validate_task_step
from robot_mcp_bridge import RobotMCPError
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.robot_context import RobotContextStore
from voice_runtime.robot_safety import (
    RobotSafetyError,
    executable_plan_name,
    execution_result_text,
    structured_robot_error,
)

MAX_CODEX_TOOL_TURNS = 3
VIZOR_ROBOT_NAME = "UR10"
PLAN_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
OBSERVE_TOOL_NAMES = ("moveit_get_current_pose",)
FREE_MOTION_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_and_execute_free_motion"}
CARTESIAN_MOTION_TOOL_NAMES = {
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_cartesian_motion",
}
NO_TEXT_RESPONSE = "I completed the action but have nothing to report."


class RobotAgentState(TypedDict):
    user_text: str
    messages: list[Mapping[str, Any]]
    input_items: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    instructions: str
    pending_tool_calls: list[dict[str, Any]]
    codex_output_items: list[dict[str, Any]]
    codex_text: str
    tool_turns: int
    final_text: str
    error_text: str | None


class LangGraphRobotAgent:
    """Runs a Codex robot dialogue turn through a LangGraph state machine."""

    def __init__(
        self,
        *,
        model: str,
        credential_store: Any,
        backend_client: Any,
        tool_bridge: Any,
        robot_context: RobotContextStore,
        thread_id: str | None = None,
    ) -> None:
        self._model = model
        self._credential_store = credential_store
        self._backend_client = backend_client
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._thread_id = thread_id or f"codex-robot-agent-{uuid.uuid4()}"
        self._turn_credentials: Any | None = None
        self._latest_state: dict[str, Any] | None = None
        self._graph = self._compile_graph()

    async def run_turn(self, turn: AgentTurnInput, *, credentials: Any | None = None) -> str:
        turn_credentials = credentials or self._credential_store.get_credentials()
        previous_credentials = getattr(self, "_turn_credentials", None)
        self._turn_credentials = turn_credentials
        state: RobotAgentState = {
            "user_text": turn.user_text,
            "messages": turn.messages,
            "input_items": [],
            "tools": [],
            "instructions": "",
            "pending_tool_calls": [],
            "codex_output_items": [],
            "codex_text": "",
            "tool_turns": 0,
            "final_text": "",
            "error_text": None,
        }
        try:
            result = await self._graph.ainvoke(
                state,
                {"configurable": {"thread_id": self._thread_id}},
            )
        finally:
            self._turn_credentials = previous_credentials
        self._latest_state = result
        return str(result.get("final_text") or NO_TEXT_RESPONSE)

    def latest_state(self) -> dict[str, Any]:
        return dict(self._latest_state or {})

    def _compile_graph(self):
        builder = StateGraph(RobotAgentState)
        builder.add_node("observe_current_pose", self._observe_current_pose)
        builder.add_node("call_codex", self._call_codex)
        builder.add_node("repair_tool_arguments", self._repair_tool_arguments)
        builder.add_node("execute_robot_tool", self._execute_robot_tool)
        builder.add_node("final_response", self._final_response)
        builder.add_edge(START, "observe_current_pose")
        builder.add_edge("observe_current_pose", "call_codex")
        builder.add_conditional_edges("call_codex", self._route_after_codex)
        builder.add_edge("repair_tool_arguments", "execute_robot_tool")
        builder.add_edge("execute_robot_tool", "observe_current_pose")
        builder.add_edge("final_response", END)
        return builder.compile(checkpointer=InMemorySaver())

    async def _observe_current_pose(self, state: RobotAgentState) -> dict[str, Any]:
        tools = self._tool_bridge.function_tools()
        observe_tool_name = _first_available_tool(tools, OBSERVE_TOOL_NAMES)
        if observe_tool_name is None:
            return {"tools": tools}
        logger.info(f"Refreshing robot observation before Codex request with {observe_tool_name}")
        await self._execute_tool(observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME})
        return {"tools": tools}

    async def _call_codex(self, state: RobotAgentState) -> dict[str, Any]:
        input_items = state["input_items"] or _input_items_from_messages(state["messages"]) or [
            _user_input_item(state["user_text"])
        ]
        tools = state["tools"] or self._tool_bridge.function_tools()
        instructions = self._instructions()
        credentials = self._turn_credentials
        result: CodexResponseResult = await self._backend_client.create_response(
            credentials,
            model=self._model,
            instructions=instructions,
            input_items=input_items,
            tools=tools,
        )
        pending = [_tool_call_to_state(tool_call) for tool_call in result.tool_calls]
        return {
            "input_items": input_items,
            "tools": tools,
            "instructions": instructions,
            "pending_tool_calls": pending,
            "codex_output_items": result.output_items,
            "codex_text": result.text,
        }

    def _route_after_codex(
        self, state: RobotAgentState
    ) -> Literal["repair_tool_arguments", "final_response"]:
        if state["error_text"]:
            return "final_response"
        if not state["pending_tool_calls"]:
            return "final_response"
        if state["tool_turns"] >= MAX_CODEX_TOOL_TURNS:
            return "final_response"
        return "repair_tool_arguments"

    def _repair_tool_arguments(self, state: RobotAgentState) -> dict[str, Any]:
        repaired: list[dict[str, Any]] = []
        for tool_call in state["pending_tool_calls"]:
            repaired.append(
                {
                    **tool_call,
                    "arguments": self._repaired_tool_arguments(
                        tool_call["name"], tool_call["arguments"], state["user_text"]
                    ),
                }
            )
        return {"pending_tool_calls": repaired}

    async def _execute_robot_tool(self, state: RobotAgentState) -> dict[str, Any]:
        input_items = [*state["input_items"], *state["codex_output_items"]]
        for tool_call in state["pending_tool_calls"]:
            output = await self._execute_tool_call(tool_call)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_call["call_id"],
                    "output": output,
                }
            )
        return {
            "input_items": input_items,
            "pending_tool_calls": [],
            "codex_output_items": [],
            "tool_turns": state["tool_turns"] + 1,
        }

    def _final_response(self, state: RobotAgentState) -> dict[str, Any]:
        return {"final_text": state["error_text"] or state["codex_text"] or NO_TEXT_RESPONSE}

    def _instructions(self) -> str:
        return f"{SYSTEM_PROMPT}\n\n{self._robot_context.render_instruction_block()}"

    async def _execute_tool_call(self, tool_call: dict[str, Any]) -> str:
        return await self._execute_tool(tool_call["name"], tool_call["arguments"])

    async def _call_policy_checked_tool(self, name: str, arguments: dict[str, Any]) -> str:
        decision = validate_task_step(name, arguments, self._robot_context)
        if not decision.ok:
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
        return output

    async def _execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            output = await self._call_policy_checked_tool(name, arguments)
            plan_name = executable_plan_name(output)
            if name in PLAN_TOOL_NAMES and plan_name:
                execution_output = await self._call_policy_checked_tool(
                    "moveit_execute_plan",
                    {"robot_name": VIZOR_ROBOT_NAME, "plan_name": plan_name},
                )
                return json.dumps(
                    {
                        "planned": json.loads(output),
                        "execution": json.loads(execution_output),
                        "execution_text": execution_result_text(execution_output),
                    },
                    ensure_ascii=False,
                )
            return output
        except RobotMCPError as exc:
            safety_error = RobotSafetyError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_error(safety_error), ensure_ascii=False)

    def _repaired_tool_arguments(
        self, name: str, arguments: dict[str, Any], user_text: str
    ) -> dict[str, Any]:
        if name in FREE_MOTION_TOOL_NAMES and not _has_any_argument(
            arguments, ("target_pose", "position")
        ):
            target_pose = self._relative_target_pose(user_text)
            if target_pose is not None:
                return {**arguments, "target_pose": target_pose}
        if name in CARTESIAN_MOTION_TOOL_NAMES and not _has_any_argument(
            arguments, ("waypoints", "positions")
        ):
            target_pose = self._relative_target_pose(user_text)
            if target_pose is not None:
                return {**arguments, "waypoints": [target_pose]}
        return arguments

    def _relative_target_pose(self, user_text: str) -> dict[str, Any] | None:
        pose = self._robot_context.latest_tcp_pose()
        if pose is None:
            return None
        position = pose.get("position") if isinstance(pose.get("position"), dict) else pose
        if not isinstance(position, dict):
            return None
        try:
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        except (KeyError, TypeError, ValueError):
            return None

        delta = _relative_delta(user_text)
        if delta is None:
            return None
        dx, dy, dz = delta
        target_position = {"x": round(x + dx, 4), "y": round(y + dy, 4), "z": round(z + dz, 4)}
        orientation = pose.get("orientation")
        if isinstance(orientation, dict):
            return {"position": target_position, "orientation": dict(orientation)}
        return target_position


def _tool_call_to_state(tool_call: Any) -> dict[str, Any]:
    return {
        "call_id": tool_call.call_id,
        "item_id": tool_call.item_id,
        "name": tool_call.name,
        "arguments": dict(tool_call.arguments),
        "raw_arguments": tool_call.raw_arguments,
    }


def _first_available_tool(tools: list[dict[str, Any]], names: tuple[str, ...]) -> str | None:
    tool_names = {tool.get("name") for tool in tools}
    for name in names:
        if name in tool_names:
            return name
    return None


def _has_any_argument(arguments: dict[str, Any], names: tuple[str, ...]) -> bool:
    return any(name in arguments and arguments[name] is not None for name in names)


def _relative_delta(text: str) -> tuple[float, float, float] | None:
    words = set(re.findall(r"[a-zA-Z']+", text.lower()))
    distance = 0.05 if words & {"bit", "slightly"} else 0.10
    if words & {"lot", "far"}:
        distance = 0.30
    if "back" in words or "backward" in words:
        return (-distance, 0.0, 0.0)
    if "forward" in words:
        return (distance, 0.0, 0.0)
    if "left" in words:
        return (0.0, distance, 0.0)
    if "right" in words:
        return (0.0, -distance, 0.0)
    if "up" in words or "raise" in words:
        return (0.0, 0.0, distance)
    if "down" in words or "lower" in words:
        return (0.0, 0.0, -distance)
    return None


def _input_items_from_messages(messages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        if role not in {"user", "assistant"}:
            continue
        text = _message_text(msg)
        if not text:
            continue
        if role == "user":
            items.append(_user_input_item(text))
        else:
            items.append(_assistant_input_item(text))
    return items


def _user_input_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant_input_item(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def _message_text(msg: Mapping[str, Any]) -> str | None:
    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping):
                continue
            text = part.get("text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts) if parts else None
    return None
