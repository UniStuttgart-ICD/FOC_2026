from __future__ import annotations

import json
import math
from typing import Any, Protocol

from agents.mcp import MCPServerStreamableHttp
from mcp.types import CallToolResult, TextContent, Tool

ALLOWED_ROBOT_TOOLS = {
    "connect_robot",
    "disconnect_robot",
    "get_robot_status",
    "get_joints",
    "get_tcp_pose",
    "move_to_position",
    "move_to_pose",
    "move_linear",
    "move_joints",
    "stop",
    "pause",
    "resume",
    "control_gripper",
    "control_gripper_position",
    "get_gripper_status",
}

SIMULATION_ROBOT_IPS = {"127.0.0.1", "localhost"}
WORKSPACE_ABS_LIMIT_M = 1.5
ROTATION_ABS_LIMIT_RAD = 6.5
JOINT_ABS_LIMIT_RAD = 6.3
GRIPPER_POSITION_MIN = 0.0
GRIPPER_POSITION_MAX = 255.0

_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "connect_robot": {"robot_ip", "robot_type", "robot_port", "skip_gripper", "use_simulated_gripper"},
    "disconnect_robot": set(),
    "get_robot_status": set(),
    "get_joints": set(),
    "get_tcp_pose": set(),
    "move_to_position": {"positions"},
    "move_to_pose": {"poses"},
    "move_linear": {"poses"},
    "move_joints": {"positions"},
    "stop": set(),
    "pause": set(),
    "resume": set(),
    "control_gripper": {"action"},
    "control_gripper_position": {"position"},
    "get_gripper_status": set(),
}


class RobotMCPError(RuntimeError):
    """Raised when robot MCP tool setup or execution fails."""


class MCPServerLike(Protocol):
    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...

    async def list_tools(self) -> list[Tool]: ...

    async def call_tool(self, tool_name: str, arguments: dict[str, Any] | None) -> CallToolResult: ...


class RobotMCPBridge:
    """Converts robot MCP tools to Codex function tools and executes allowed calls."""

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
        self._tools = [tool for tool in await self._server.list_tools() if tool.name in ALLOWED_ROBOT_TOOLS]
        self._connected = True

    async def disconnect(self) -> None:
        await self._server.cleanup()
        self._connected = False
        self._tools = []

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
                "strict": None,
            }
            for tool in self._tools
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in ALLOWED_ROBOT_TOOLS or name not in {tool.name for tool in self._tools}:
            raise RobotMCPError(f"Tool is not allowed: {name}")
        _validate_tool_arguments(name, arguments)
        result = await self._server.call_tool(name, arguments)
        return _serialize_tool_result(result)


def _validate_tool_arguments(name: str, arguments: dict[str, Any]) -> None:
    allowed = _ALLOWED_ARGUMENTS.get(name)
    if allowed is not None:
        unexpected = set(arguments) - allowed
        if unexpected:
            raise RobotMCPError(f"Unexpected argument for {name}: {sorted(unexpected)[0]}")

    if name == "connect_robot":
        robot_ip = arguments.get("robot_ip")
        if robot_ip not in SIMULATION_ROBOT_IPS:
            raise RobotMCPError("Only simulation robot IP 127.0.0.1 is allowed")
        return

    if name == "move_to_position":
        _validate_vectors(arguments.get("positions"), dimensions=3, limit=WORKSPACE_ABS_LIMIT_M)
        return

    if name in {"move_to_pose", "move_linear"}:
        _validate_pose_vectors(arguments.get("poses"))
        return

    if name == "move_joints":
        _validate_vectors(arguments.get("positions"), dimensions=6, limit=JOINT_ABS_LIMIT_RAD, label="joint")
        return

    if name == "control_gripper":
        action = arguments.get("action")
        if action not in {"open", "close"}:
            raise RobotMCPError("control_gripper action must be open or close")
        return

    if name == "control_gripper_position":
        position = arguments.get("position")
        if not isinstance(position, (int, float)) or isinstance(position, bool):
            raise RobotMCPError("control_gripper_position position is outside safe range")
        numeric_position = float(position)
        if not math.isfinite(numeric_position) or not GRIPPER_POSITION_MIN <= numeric_position <= GRIPPER_POSITION_MAX:
            raise RobotMCPError("control_gripper_position position is outside safe range")


def _validate_vectors(value: Any, *, dimensions: int, limit: float, label: str = "coordinate") -> None:
    if not isinstance(value, list) or not value:
        raise RobotMCPError("Expected a non-empty outer list of target vectors")
    for vector in value:
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise RobotMCPError(f"Expected target vectors with {dimensions} values")
        for number in vector:
            if not _finite_number(number):
                raise RobotMCPError("Expected finite numeric target values")
            if abs(float(number)) > limit:
                if label == "joint":
                    raise RobotMCPError("Target is outside joint limit")
                raise RobotMCPError("Target is outside simulation workspace")


def _validate_pose_vectors(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise RobotMCPError("Expected a non-empty outer list of pose vectors")
    for pose in value:
        if not isinstance(pose, list) or len(pose) != 6:
            raise RobotMCPError("Expected pose vectors with 6 values")
        for coordinate in pose[:3]:
            if not _finite_number(coordinate) or abs(float(coordinate)) > WORKSPACE_ABS_LIMIT_M:
                raise RobotMCPError("Target is outside simulation workspace")
        for rotation in pose[3:]:
            if not _finite_number(rotation) or abs(float(rotation)) > ROTATION_ABS_LIMIT_RAD:
                raise RobotMCPError("Target rotation is outside safe range")


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


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
