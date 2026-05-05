from __future__ import annotations

import json
import math
from typing import Any

VIZOR_ROBOT_NAME = "UR10"
WORKSPACE_ABS_LIMIT_M = 1.5
DEFAULT_TIMEOUT_MAX_S = 60.0

AGENT_TO_LEGACY_MCP_TOOL_NAMES = {
    "moveit_get_current_pose": "get_current_pose",
    "moveit_plan_free_motion": "plan_free_motion",
    "moveit_plan_cartesian_motion": "plan_cartesian_motion",
    "moveit_execute_plan": "execute_plan",
    "moveit_open_gripper": "open_gripper",
    "moveit_close_gripper": "close_gripper",
    "moveit_attach_object": "attach_object",
}
CANONICAL_ONLY_MCP_TOOL_NAMES = {
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
}
ALLOWED_ROBOT_TOOLS = frozenset(AGENT_TO_LEGACY_MCP_TOOL_NAMES) | CANONICAL_ONLY_MCP_TOOL_NAMES

_AGENT_TOOL_DESCRIPTIONS = {
    "moveit_get_current_pose": "Observe the UR10 current end-effector pose and planning frame. Call before relative, vague, repeated, or safety-sensitive movement.",
    "moveit_plan_free_motion": "Plan a collision-aware free-space motion to one absolute target pose in base_link. Use the returned plan_name with moveit_execute_plan.",
    "moveit_plan_cartesian_motion": "Plan a Cartesian end-effector path through waypoints in base_link. Use for straight-line or waypoint-following motion.",
    "moveit_plan_and_execute_free_motion": "High-level workflow to plan, execute, and verify one free-space target pose. Use for simple voice moves when no separate plan review is needed.",
    "moveit_plan_and_execute_cartesian_motion": "High-level workflow to plan, execute, and verify a Cartesian waypoint sequence. Use for straight-line or multi-waypoint voice moves.",
    "moveit_execute_plan": "Execute a returned plan_name from a successful planning tool. Do not invent plan names.",
    "moveit_open_gripper": "Open the simulated UR10 gripper state.",
    "moveit_close_gripper": "Close the simulated UR10 gripper state.",
    "moveit_attach_object": "Attach an object to the simulated gripper after the gripper has been closed.",
}

_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "moveit_get_current_pose": {"robot_name", "timeout_s"},
    "moveit_plan_free_motion": {"robot_name", "target_pose", "position", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_plan_cartesian_motion": {"robot_name", "waypoints", "positions", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_plan_and_execute_free_motion": {"robot_name", "target_pose", "plan_name", "timeout_s"},
    "moveit_plan_and_execute_cartesian_motion": {"robot_name", "waypoints", "plan_name", "timeout_s"},
    "moveit_execute_plan": {"robot_name", "plan_name", "timeout_s"},
    "moveit_open_gripper": {"robot_name"},
    "moveit_close_gripper": {"robot_name"},
    "moveit_attach_object": {"robot_name", "object_name"},
}


class RobotSafetyError(ValueError):
    """Raised when a robot tool call violates local validation policy."""

    def __init__(self, message: str, *, correction: str):
        super().__init__(message)
        self.correction = correction


def canonical_mcp_tool_name(agent_tool_name: str) -> str:
    if agent_tool_name in CANONICAL_ONLY_MCP_TOOL_NAMES:
        return agent_tool_name
    try:
        return AGENT_TO_LEGACY_MCP_TOOL_NAMES[agent_tool_name]
    except KeyError as exc:
        raise RobotSafetyError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from exc


def agent_tool_description(agent_tool_name: str) -> str:
    try:
        return _AGENT_TOOL_DESCRIPTIONS[agent_tool_name]
    except KeyError as exc:
        raise RobotSafetyError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from exc


def structured_robot_error(
    exc: RobotSafetyError,
    *,
    retryable: bool = True,
    suggested_next_tool: str | None = "moveit_get_current_pose",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": str(exc),
        "correction": exc.correction,
        "retryable": retryable,
    }
    if suggested_next_tool is not None:
        payload["suggested_next_tool"] = suggested_next_tool
    return payload


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

    _validate_robot_name(arguments.get("robot_name", VIZOR_ROBOT_NAME))

    if name == "moveit_get_current_pose":
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name in {"moveit_plan_free_motion", "moveit_plan_and_execute_free_motion"}:
        pose = arguments.get("target_pose", arguments.get("position"))
        _validate_pose(pose)
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name in {"moveit_plan_cartesian_motion", "moveit_plan_and_execute_cartesian_motion"}:
        waypoints = arguments.get("waypoints", arguments.get("positions"))
        _validate_waypoints(waypoints)
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

    if name == "moveit_attach_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotSafetyError(
                "Expected a non-empty object_name",
                correction="Retry with the object name to attach.",
            )


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
            "Expected target_pose with position fields",
            correction="Retry with a MoveIt target pose inside the simulation workspace.",
        )

    position = value.get("position") if isinstance(value.get("position"), dict) else value
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
    if orientation is None:
        return
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


def _validate_waypoints(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise RobotSafetyError(
            "Expected at least one waypoint",
            correction="Retry with one or more target poses inside the simulation workspace.",
        )
    for waypoint in value:
        _validate_pose(waypoint)


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
