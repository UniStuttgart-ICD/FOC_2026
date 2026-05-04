from __future__ import annotations

import json
import math
from typing import Any

VIZOR_ROBOT_NAME = "UR10"
WORKSPACE_ABS_LIMIT_M = 1.5
DEFAULT_TIMEOUT_MAX_S = 60.0

AGENT_TO_LEGACY_MCP_TOOL_NAMES = {
    "moveit_plan_free_motion": "plan_free_motion",
    "moveit_plan_linear_motion": "plan_linear_motion",
    "moveit_execute_plan": "execute_plan",
    "moveit_open_gripper": "open_gripper",
    "moveit_close_gripper": "close_gripper",
    "moveit_get_robot_status": "get_robot_status",
}
ALLOWED_ROBOT_TOOLS = frozenset(AGENT_TO_LEGACY_MCP_TOOL_NAMES)

_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "moveit_plan_free_motion": {"robot_name", "position", "timeout_s"},
    "moveit_plan_linear_motion": {"robot_name", "position", "timeout_s"},
    "moveit_execute_plan": {"robot_name", "plan_name", "timeout_s"},
    "moveit_open_gripper": {"robot_name", "timeout_s"},
    "moveit_close_gripper": {"robot_name", "timeout_s"},
    "moveit_get_robot_status": {"robot_name"},
}


class RobotSafetyError(ValueError):
    """Raised when a robot tool call violates local safety policy."""

    def __init__(self, message: str, *, correction: str):
        super().__init__(message)
        self.correction = correction


def canonical_mcp_tool_name(agent_tool_name: str) -> str:
    try:
        return AGENT_TO_LEGACY_MCP_TOOL_NAMES[agent_tool_name]
    except KeyError as exc:
        raise RobotSafetyError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from exc


def validate_robot_tool_call(name: str, arguments: dict[str, Any]) -> None:
    if name not in ALLOWED_ROBOT_TOOLS:
        raise RobotSafetyError(
            f"Tool is not allowed: {name}",
            correction="Use one of the allowed MoveIt robot tools.",
        )

    allowed = _ALLOWED_ARGUMENTS[name]
    unexpected = set(arguments) - allowed
    if unexpected:
        raise RobotSafetyError(
            f"Unexpected argument for {name}: {sorted(unexpected)[0]}",
            correction="Remove unsupported arguments and retry.",
        )

    _validate_robot_name(arguments.get("robot_name"))

    if name in {"moveit_plan_free_motion", "moveit_plan_linear_motion"}:
        _validate_pose(arguments.get("position"))
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_execute_plan":
        plan_name = arguments.get("plan_name")
        if not isinstance(plan_name, str) or not plan_name:
            raise RobotSafetyError(
                "Expected a non-empty plan_name",
                correction="Plan first, then retry with the returned plan_name.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name in {"moveit_open_gripper", "moveit_close_gripper"}:
        _validate_timeout(arguments.get("timeout_s"))


def executable_plan_name(output: str) -> str | None:
    structured_content = _structured_content(output)
    if not isinstance(structured_content, dict):
        return None
    if structured_content.get("ok") is not True:
        return None

    feedback = structured_content.get("feedback")
    if not isinstance(feedback, dict) or feedback.get("can_execute") is not True:
        return None

    raw = structured_content.get("raw")
    if not isinstance(raw, dict):
        return None

    plan_name = raw.get("plan_name")
    if isinstance(plan_name, str) and plan_name:
        return plan_name
    return None


def execution_result_text(output: str) -> str:
    structured_content = _structured_content(output)
    if isinstance(structured_content, dict):
        verification = structured_content.get("verification")
        if structured_content.get("ok") is True and isinstance(verification, dict):
            if verification.get("result") == "pass":
                return "Motion completed."
    return "I planned the motion, but execution could not be verified."


def _validate_robot_name(robot_name: Any) -> None:
    if robot_name != VIZOR_ROBOT_NAME:
        raise RobotSafetyError(
            "Only Vizor robot UR10 is allowed",
            correction='Retry with robot_name="UR10".',
        )


def _validate_timeout(timeout_s: Any) -> None:
    if timeout_s is None:
        return
    if not _finite_number(timeout_s):
        raise RobotSafetyError(
            "timeout_s must be a finite number",
            correction=f"Retry with timeout_s less than or equal to {DEFAULT_TIMEOUT_MAX_S}.",
        )
    numeric_timeout_s = float(timeout_s)
    if numeric_timeout_s <= 0.0 or numeric_timeout_s > DEFAULT_TIMEOUT_MAX_S:
        raise RobotSafetyError(
            "timeout_s is outside safe range",
            correction=f"Retry with timeout_s less than or equal to {DEFAULT_TIMEOUT_MAX_S}.",
        )


def _validate_pose(value: Any) -> None:
    if not isinstance(value, dict):
        raise RobotSafetyError(
            "Expected position with position and orientation fields",
            correction="Retry with a MoveIt pose target inside the simulation workspace.",
        )

    position = value.get("position")
    if not isinstance(position, dict):
        raise RobotSafetyError(
            "Expected position coordinates",
            correction="Retry with x, y, and z coordinates inside the simulation workspace.",
        )
    for axis in ("x", "y", "z"):
        coordinate = _finite_float(position.get(axis))
        if coordinate is None or abs(coordinate) > WORKSPACE_ABS_LIMIT_M:
            raise RobotSafetyError(
                "Target is outside simulation workspace",
                correction=f"Retry with x/y/z coordinates within +/-{WORKSPACE_ABS_LIMIT_M} m.",
            )

    orientation = value.get("orientation")
    if not isinstance(orientation, dict):
        raise RobotSafetyError(
            "Expected orientation quaternion",
            correction="Retry with finite x, y, z, and w quaternion values.",
        )
    for component in ("x", "y", "z", "w"):
        rotation = orientation.get(component)
        if not _finite_number(rotation):
            raise RobotSafetyError(
                "Expected finite orientation values",
                correction="Retry with finite x, y, z, and w quaternion values.",
            )


def _finite_number(value: Any) -> bool:
    return _finite_float(value) is not None


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")
