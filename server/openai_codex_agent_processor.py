"""OpenAI Codex OAuth backend Adapter for Agent Turn processing."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from loguru import logger

from codex_auth import CodexAuthError, PiCodexCredentialStore
from codex_backend_client import CodexBackendClient, CodexBackendError, CodexResponseResult
from prompts import SYSTEM_PROMPT
from robot_mcp_bridge import RobotMCPBridge, RobotMCPError
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.robot_context import RobotContextStore
from voice_runtime.robot_safety import executable_plan_name, execution_result_text

MAX_CODEX_TOOL_TURNS = 3
VIZOR_ROBOT_NAME = "UR10"
PLAN_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
OBSERVE_TOOL_NAMES = ("moveit_get_current_pose", "moveit_get_robot_status")
FREE_MOTION_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_and_execute_free_motion"}
CARTESIAN_MOTION_TOOL_NAMES = {"moveit_plan_cartesian_motion", "moveit_plan_and_execute_cartesian_motion"}


class OpenAICodexAgentProcessor:
    """Runs Agent Turns through ChatGPT's Codex backend with Pi OAuth credentials."""

    def __init__(
        self,
        mcp_server_url: str,
        model: str,
        *,
        credential_store: Any | None = None,
        backend_client: Any | None = None,
        tool_bridge: Any | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._credential_store = credential_store or PiCodexCredentialStore()
        self._backend_client = backend_client
        self._tool_bridge = tool_bridge
        self._owns_backend_client = backend_client is None
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        if self._backend_client is not None and (self._connected or not self._owns_backend_client):
            await self._backend_client.close()
        self._backend_client = None
        self._tool_bridge = None
        self._connected = False
        logger.info("OpenAI Codex backend agent disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        logger.info(f"User said: {turn.user_text}")
        try:
            await self._ensure_connected()
            credentials = self._credential_store.get_credentials()
        except CodexAuthError as exc:
            logger.error(f"OpenAI Codex OAuth error: {exc}")
            yield str(exc)
            return
        except Exception as exc:
            logger.error(f"OpenAI Codex agent connection error: {exc}")
            yield "I can't reach the robot control server right now."
            return

        backend_client = self._backend_client
        tool_bridge = self._tool_bridge
        if backend_client is None or tool_bridge is None:
            yield "I can't reach the robot control server right now."
            return

        if not self._model_logged:
            logger.info(f"OpenAI Codex model: {self._model}")
            self._model_logged = True

        input_items = _input_items_from_messages(turn.messages) or [_user_input_item(turn.user_text)]
        tools = tool_bridge.function_tools()

        try:
            await self._refresh_robot_observation(tool_bridge, tools)
            result = await backend_client.create_response(
                credentials,
                model=self._model,
                instructions=self._instructions(),
                input_items=input_items,
                tools=tools,
            )
            result = await self._run_tool_loop(
                result=result,
                input_items=input_items,
                credentials=credentials,
                backend_client=backend_client,
                tool_bridge=tool_bridge,
                tools=tools,
                user_text=turn.user_text,
            )
            yield result.text or "I completed the action but have nothing to report."
        except CodexBackendError as exc:
            logger.error(f"OpenAI Codex backend error: {exc}")
            yield "I encountered an error. Please try again."
        except Exception as exc:
            logger.error(f"OpenAI Codex agent error: {exc}")
            yield "I encountered an error. Please try again."

    def _instructions(self) -> str:
        return f"{SYSTEM_PROMPT}\n\n{self._robot_context.render_instruction_block()}"

    async def _refresh_robot_observation(self, tool_bridge: RobotMCPBridge, tools: list[dict[str, Any]]) -> None:
        observe_tool_name = _first_available_tool(tools, OBSERVE_TOOL_NAMES)
        if observe_tool_name is None:
            return
        logger.info(f"Refreshing robot observation before Codex request with {observe_tool_name}")
        await self._call_robot_tool(tool_bridge, observe_tool_name, {"robot_name": VIZOR_ROBOT_NAME})

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._backend_client is None:
            self._backend_client = CodexBackendClient()
        if self._tool_bridge is None:
            self._tool_bridge = RobotMCPBridge(self._mcp_server_url)
        await self._tool_bridge.connect()
        self._connected = True
        logger.info("OpenAI Codex backend agent connected")

    async def _run_tool_loop(
        self,
        *,
        result: CodexResponseResult,
        input_items: list[dict[str, Any]],
        credentials: Any,
        backend_client: CodexBackendClient,
        tool_bridge: RobotMCPBridge,
        tools: list[dict[str, Any]],
        user_text: str,
    ) -> CodexResponseResult:
        turns = 0
        while result.tool_calls and turns < MAX_CODEX_TOOL_TURNS:
            turns += 1
            input_items.extend(result.output_items)
            for tool_call in result.tool_calls:
                output = await self._call_robot_tool(
                    tool_bridge,
                    tool_call.name,
                    self._repaired_tool_arguments(tool_call.name, tool_call.arguments, user_text),
                )
                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": output,
                    }
                )
            result = await backend_client.create_response(
                credentials,
                model=self._model,
                instructions=self._instructions(),
                input_items=input_items,
                tools=tools,
            )
        return result

    async def _call_robot_tool(self, tool_bridge: RobotMCPBridge, name: str, arguments: dict[str, Any]) -> str:
        try:
            output = await tool_bridge.call_tool(name, arguments)
            self._robot_context.update_from_tool_result(name, output)
            plan_name = executable_plan_name(output)
            if name in PLAN_TOOL_NAMES and plan_name:
                execution_output = await tool_bridge.call_tool(
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
            return json.dumps({"error": str(exc)}, ensure_ascii=False)

    def _repaired_tool_arguments(self, name: str, arguments: dict[str, Any], user_text: str) -> dict[str, Any]:
        if name in FREE_MOTION_TOOL_NAMES and not _has_any_argument(arguments, ("target_pose", "position")):
            target_pose = self._relative_target_pose(user_text)
            if target_pose is not None:
                return {**arguments, "target_pose": target_pose}
        if name in CARTESIAN_MOTION_TOOL_NAMES and not _has_any_argument(arguments, ("waypoints", "positions")):
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
    assistant_index = 0
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
            assistant_index += 1
            items.append(_assistant_output_item(text, assistant_index))
    return items


def _user_input_item(text: str) -> dict[str, Any]:
    return {"role": "user", "content": [{"type": "input_text", "text": text}]}


def _assistant_output_item(text: str, index: int) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
        "status": "completed",
        "id": f"history-assistant-{index}",
    }


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
