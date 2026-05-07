from __future__ import annotations

import copy
import json
from typing import Any

from robot_control.call_validation import agent_tool_description

ROBOT_NAME = "UR10"
INITIAL_POSE: dict[str, Any] = {
    "position": {"x": 0.4, "y": 0.1, "z": 0.3},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}

_SUPPORTED_TOOLS = (
    "moveit_get_current_pose",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_plan_and_execute_named_pose",
    "moveit_plan_and_execute_joint_goal",
    "moveit_list_available_robots",
)

_DESCRIPTIONS = {
    "moveit_plan_and_execute_named_pose": "Move the UR10 to a deterministic named pose.",
    "moveit_plan_and_execute_joint_goal": "Move the UR10 to a deterministic joint goal.",
    "moveit_list_available_robots": "List robots available in the simulated MoveIt scene.",
}


class SimulatedMoveItAdapter:
    """Deterministic offline Robot Tool Adapter for model evaluation."""

    def __init__(self) -> None:
        self._pose = copy.deepcopy(INITIAL_POSE)
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": name,
                "description": _tool_description(name),
                "parameters": _tool_parameters(name),
                "strict": None,
            }
            for name in _SUPPORTED_TOOLS
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_get_current_pose":
            return _tool_output(
                content=[f"{ROBOT_NAME} pose is x={self._pose['position']['x']}, y={self._pose['position']['y']}, z={self._pose['position']['z']}."],
                structured_content={
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "planning_frame": "base_link",
                    "raw": {"pose": copy.deepcopy(self._pose)},
                },
            )
        if name == "moveit_plan_and_execute_cartesian_motion":
            return self._execute_cartesian(arguments)
        if name == "moveit_plan_and_execute_named_pose":
            return self._execute_named_pose(arguments)
        if name == "moveit_plan_and_execute_joint_goal":
            return self._execute_joint_goal(arguments)
        if name == "moveit_list_available_robots":
            return _tool_output(
                content=[f"Available robots: {ROBOT_NAME}."],
                structured_content={
                    "ok": True,
                    "robots": [{"name": ROBOT_NAME, "planning_frame": "base_link"}],
                    "raw": {"robots": [ROBOT_NAME]},
                },
            )
        return _error_output(f"Tool is not supported by simulated MoveIt adapter: {name}")

    def _execute_cartesian(self, arguments: dict[str, Any]) -> str:
        waypoints = arguments.get("waypoints", arguments.get("positions", arguments.get("points")))
        if not isinstance(waypoints, list) or not waypoints:
            return _error_output("Expected one or more Cartesian waypoints.")
        final_pose = _pose_from_value(waypoints[-1], fallback_orientation=self._pose["orientation"])
        if final_pose is None:
            return _error_output("Expected final waypoint with finite x/y/z position.")
        self._pose = final_pose
        return _motion_output(
            content=["Cartesian motion executed and verified."],
            raw={
                "plan_name": arguments.get("plan_name", "simulated_cartesian_plan"),
                "waypoint_count": len(waypoints),
                "pose": copy.deepcopy(self._pose),
            },
        )

    def _execute_named_pose(self, arguments: dict[str, Any]) -> str:
        named_pose = str(arguments.get("named_pose", arguments.get("pose_name", "home")))
        if named_pose in {"home", "ready"}:
            self._pose = copy.deepcopy(INITIAL_POSE)
        return _motion_output(
            content=[f"Named pose {named_pose} executed and verified."],
            raw={"named_pose": named_pose, "pose": copy.deepcopy(self._pose)},
        )

    def _execute_joint_goal(self, arguments: dict[str, Any]) -> str:
        joint_goal = arguments.get("joint_goal", arguments.get("joints", {}))
        return _motion_output(
            content=["Joint goal executed and verified."],
            raw={"joint_goal": joint_goal, "pose": copy.deepcopy(self._pose)},
        )


def _tool_description(name: str) -> str:
    if name in _DESCRIPTIONS:
        return _DESCRIPTIONS[name]
    return agent_tool_description(name)


def _tool_parameters(name: str) -> dict[str, Any]:
    if name == "moveit_get_current_pose":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "timeout_s": {"type": "number"},
            },
        }
    if name == "moveit_plan_and_execute_cartesian_motion":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "waypoints": {"type": "array", "items": {"type": "object"}},
                "plan_name": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["waypoints"],
        }
    if name == "moveit_plan_and_execute_named_pose":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "named_pose": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["named_pose"],
        }
    if name == "moveit_plan_and_execute_joint_goal":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "joint_goal": {"type": "object"},
                "timeout_s": {"type": "number"},
            },
            "required": ["joint_goal"],
        }
    return {"type": "object", "properties": {}}


def _pose_from_value(value: Any, *, fallback_orientation: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    position = value.get("position") if isinstance(value.get("position"), dict) else value
    if not isinstance(position, dict):
        return None
    coordinates: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        coordinate = position.get(axis)
        if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
            return None
        coordinates[axis] = float(coordinate)
    orientation = value.get("orientation")
    if not isinstance(orientation, dict):
        orientation = fallback_orientation
    return {"position": coordinates, "orientation": copy.deepcopy(orientation)}


def _motion_output(*, content: list[str], raw: dict[str, Any]) -> str:
    return _tool_output(
        content=content,
        structured_content={
            "ok": True,
            "feedback": {"can_execute": True},
            "verification": {"result": "pass"},
            "execution": {"verification_result": "pass"},
            "raw": raw,
        },
    )


def _error_output(error: str) -> str:
    return _tool_output(
        content=[error],
        structured_content={"ok": False, "error": error, "retryable": False},
        is_error=True,
    )


def _tool_output(
    *,
    content: list[str],
    structured_content: dict[str, Any],
    is_error: bool = False,
) -> str:
    return json.dumps(
        {
            "content": content,
            "structured_content": structured_content,
            "is_error": is_error,
        },
        ensure_ascii=False,
    )
