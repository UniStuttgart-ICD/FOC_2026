from __future__ import annotations

import json
import math
from typing import Any

from robot_control.context import RobotContextStore
from robot_control.manipulation_plans import executable_plan_name_from_output

VIZOR_ROBOT_NAME = "UR10"
WORKSPACE_ABS_LIMIT_M = 1.5
DEFAULT_TIMEOUT_MAX_S = 60.0
PLANNING_STRATEGIES = {"auto", "cartesian", "sampled_approach"}

AGENT_TO_LEGACY_MCP_TOOL_NAMES = {
    "moveit_get_current_pose": "get_current_pose",
    "moveit_plan_free_motion": "plan_free_motion",
    "moveit_plan_cartesian_motion": "plan_cartesian_motion",
    "moveit_execute_plan": "execute_plan",
    "moveit_open_gripper": "open_gripper",
    "moveit_close_gripper": "close_gripper",
    "moveit_attach_object": "attach_object",
}
CANONICAL_ONLY_MCP_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "moveit_get_robot_state",
        "moveit_list_scene_objects",
        "moveit_get_object_context",
        "moveit_plan_pick",
        "moveit_plan_place",
        "moveit_plan_pick_task",
        "moveit_plan_place_task",
        "moveit_execute_task_solution",
        "moveit_execute_task_plan",
        "moveit_explain_motion_failure",
        "moveit_verify_attached_object",
    }
)
ALLOWED_ROBOT_TOOLS = frozenset(AGENT_TO_LEGACY_MCP_TOOL_NAMES) | CANONICAL_ONLY_MCP_TOOL_NAMES

_AGENT_TOOL_DESCRIPTIONS = {
    "moveit_get_current_pose": (
        "Observe the UR10 current end-effector pose, orientation, and planning frame. "
        "Use it to ground gestures before relative, vague, repeated, or state-dependent movement."
    ),
    "moveit_get_robot_state": (
        "Observe the UR10 current pose, planning frame, physical-mode flag, and latest "
        "fake-controller joint state. Use it to diagnose readiness or motion failures; "
        "use moveit_get_current_pose for ordinary relative motion grounding."
    ),
    "moveit_list_scene_objects": (
        "Observe MoveIt planning-scene objects, including names, frames, poses, shape "
        "summaries, bounds, colors when available, and attached/free state. Use before "
        "object-relative or pick tasks."
    ),
    "moveit_get_object_context": (
        "Observe one MoveIt planning-scene object's pose, bounds, shape summaries, "
        "grasp-relevant faces, clearance when available, planning frame, and attached/free state. "
        "Call moveit_list_scene_objects first and pass one returned object_name."
    ),
    "moveit_plan_pick": (
        "Legacy fallback pick planner for one MoveIt planning-scene object. Use only when "
        "moveit_plan_pick_task is unavailable or the user explicitly asks for a legacy "
        "executable plan. Use after moveit_list_scene_objects and moveit_get_object_context. "
        "It returns raw.plan_name, "
        "feedback.can_execute, selected grasp face, waypoints, raw.candidate_attempts, object context, "
        "and workflow metadata. Optional planning_strategy values are auto, cartesian, and "
        "sampled_approach. "
        "It uses the same executable-plan result shape as other planning tools: ok, "
        "feedback.can_execute, raw.plan_name, robot/robot_name, and optional "
        "raw.next_action.after_success. "
        "It derives approach, pre-grasp, close-gripper, attach, and lift workflow steps, plans "
        "motion waypoints through the existing Cartesian planner, and does not execute motion "
        "or gripper actions."
    ),
    "moveit_plan_place": (
        "Legacy fallback place planner for an attached MoveIt planning-scene object. Use only "
        "when moveit_plan_place_task is unavailable or the user explicitly asks for a legacy "
        "executable plan. Use target_pose or target_position to describe the target pose of the object in "
        "base_link, plus orientation_mode such as keep, horizontal, vertical, or explicit. It returns "
        "raw.plan_name, feedback.can_execute, release TCP pose, waypoints, object context, "
        "and workflow metadata. It uses the same executable-plan result shape as other planning "
        "tools: ok, feedback.can_execute, raw.plan_name, robot/robot_name, and optional "
        "raw.next_action.after_success. It derives carry, rotate, descend, release, detach, and retreat "
        "workflow steps through the existing motion planners, and does not execute motion or "
        "gripper actions."
    ),
    "moveit_plan_pick_task": (
        "Primary tool for ordinary pick requests. Plan a task solution for picking one "
        "MoveIt planning-scene object. Use after "
        "moveit_list_scene_objects, moveit_get_object_context, and moveit_get_current_pose. "
        "It returns a task_solution_id, stage evidence, scene snapshot evidence, and approval "
        "payload. It does not execute motion or gripper actions."
    ),
    "moveit_plan_place_task": (
        "Primary tool for ordinary place requests. Plan a task solution for placing one "
        "attached MoveIt planning-scene object. Use a target_pose or target_position plus "
        "orientation_mode. It returns a task_solution_id, "
        "stage evidence, scene snapshot evidence, and approval payload. It does not execute "
        "motion or gripper actions."
    ),
    "moveit_execute_task_solution": (
        "Execute a returned task_solution_id from moveit_plan_pick_task or moveit_plan_place_task. "
        "Use only for sim/emulated task-solution execution after explicit user intent is bound "
        "to that exact task solution."
    ),
    "moveit_execute_task_plan": (
        "Execute a returned pick task_solution_id through Verified Real Robot Execution by "
        "planning concrete motion stages, executing each returned plan_name, closing the gripper, "
        "attaching the object, and verifying attachment. Use only after explicit user intent is "
        "bound to that exact task_solution_id. Use timeout_s around 30 for real-robot execution "
        "unless the user asks for a shorter supervised timeout."
    ),
    "moveit_explain_motion_failure": (
        "Explain one failed planner or executor result for the UR10. Use after a MoveIt tool "
        "returns ok=false or verification fails. Pass failed_tool_name, failed_tool_result, and "
        "failed_tool_arguments when available. It returns a compact error category, correction, "
        "retry guidance, retryable flag, and suggested next tool; it does not plan or execute motion."
    ),
    "moveit_verify_attached_object": (
        "Verify that one planning-scene object is attached; this attached object check confirms "
        "it is attached to the gripper and moved with the gripper after executing a pick or place "
        "plan. Use after moveit_execute_plan for pick/place workflows before claiming the object "
        "was picked, carried, placed, or released. Do not use it to execute motion or attach objects."
    ),
    "moveit_plan_free_motion": (
        "Plan collision-aware free-space point-to-point motion to one target pose in base_link. "
        "Use for a single destination, not for drawing shapes or expressive paths. "
        "Use the returned plan_name with moveit_execute_plan."
    ),
    "moveit_plan_cartesian_motion": (
        "Plan Cartesian expressive TCP paths through ordered waypoints in base_link. "
        "Use for waving, tracing, drawing simple shapes, sweeping, multi-point motion, "
        "straight-line motion, or waypoint-following from a fresh current pose. Preserve "
        "orientation unless the task asks to rotate; when preserving orientation, copy the current raw.pose.orientation "
        "into every waypoint."
    ),
    "moveit_execute_plan": (
        "Execute a returned plan_name from a successful free/cartesian or legacy pick/place "
        "planning tool. Do not use it for task_solution_id values and do not invent plan names."
    ),
    "moveit_open_gripper": "Open the UR10 gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback.",
    "moveit_close_gripper": "Close the UR10 gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback.",
    "moveit_attach_object": "Attach an object to the simulated gripper after the gripper has been closed.",
}

_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "moveit_get_current_pose": {"robot_name", "timeout_s"},
    "moveit_get_robot_state": {"robot_name", "timeout_s"},
    "moveit_list_scene_objects": {"robot_name", "timeout_s"},
    "moveit_get_object_context": {"robot_name", "object_name", "timeout_s"},
    "moveit_plan_pick": {
        "robot_name",
        "object_name",
        "plan_name",
        "grasp_face",
        "approach_distance_m",
        "grasp_standoff_m",
        "lift_distance_m",
        "planning_strategy",
        "timeout_s",
    },
    "moveit_plan_place": {
        "robot_name",
        "object_name",
        "plan_name",
        "target_pose",
        "target_position",
        "orientation_mode",
        "place_face",
        "support_face",
        "approach_distance_m",
        "place_standoff_m",
        "retreat_distance_m",
        "timeout_s",
    },
    "moveit_plan_pick_task": {
        "robot_name",
        "object_name",
        "grasp_face",
        "timeout_s",
    },
    "moveit_plan_place_task": {
        "robot_name",
        "object_name",
        "target_pose",
        "target_position",
        "orientation_mode",
        "timeout_s",
    },
    "moveit_execute_task_solution": {"robot_name", "task_solution_id", "timeout_s"},
    "moveit_execute_task_plan": {"robot_name", "task_solution_id", "timeout_s"},
    "moveit_explain_motion_failure": {
        "robot_name",
        "failed_tool_name",
        "failed_tool_arguments",
        "failed_tool_result",
        "user_intent",
        "timeout_s",
    },
    "moveit_verify_attached_object": {"robot_name", "object_name", "timeout_s"},
    "moveit_plan_free_motion": {"robot_name", "target_pose", "position", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_plan_cartesian_motion": {"robot_name", "waypoints", "positions", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_execute_plan": {"robot_name", "plan_name", "timeout_s"},
    "moveit_open_gripper": {"robot_name", "timeout_s"},
    "moveit_close_gripper": {"robot_name", "timeout_s"},
    "moveit_attach_object": {"robot_name", "object_name"},
}


class RobotCallValidationError(ValueError):
    """Raised when a robot tool call violates local validation policy."""

    def __init__(self, message: str, *, correction: str):
        super().__init__(message)
        self.correction = correction


def canonical_mcp_tool_name(agent_tool_name: str) -> str:
    try:
        return AGENT_TO_LEGACY_MCP_TOOL_NAMES[agent_tool_name]
    except KeyError:
        if agent_tool_name in CANONICAL_ONLY_MCP_TOOL_NAMES:
            return agent_tool_name
        if agent_tool_name.startswith("moveit_plan_and_execute_"):
            raise RobotCallValidationError(
                f"Tool is not allowed: {agent_tool_name}",
                correction=(
                    "Plan with moveit_plan_free_motion or moveit_plan_cartesian_motion, "
                    "then execute the returned plan_name with moveit_execute_plan."
                ),
            ) from None
        raise RobotCallValidationError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from None


def agent_tool_description(agent_tool_name: str) -> str:
    try:
        return _AGENT_TOOL_DESCRIPTIONS[agent_tool_name]
    except KeyError as exc:
        raise RobotCallValidationError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from exc


def structured_robot_call_error(
    exc: RobotCallValidationError,
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
        if name.startswith("moveit_plan_and_execute_"):
            raise RobotCallValidationError(
                f"Tool is not allowed: {name}",
                correction=(
                    "Plan with moveit_plan_free_motion or moveit_plan_cartesian_motion, "
                    "then execute the returned plan_name with moveit_execute_plan."
                ),
            )
        raise RobotCallValidationError(
            f"Tool is not allowed: {name}",
            correction="Use one of the allowed MoveIt robot tools.",
        )

    allowed = _ALLOWED_ARGUMENTS[name]
    unexpected = set(arguments) - allowed
    if unexpected:
        raise RobotCallValidationError(
            f"Unexpected argument for {name}: {sorted(unexpected)[0]}",
            correction="Remove unsupported arguments and retry.",
        )

    _validate_robot_name(arguments.get("robot_name", VIZOR_ROBOT_NAME))

    if name in {"moveit_get_current_pose", "moveit_get_robot_state", "moveit_list_scene_objects"}:
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_get_object_context":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_explain_motion_failure":
        failed_tool_name = arguments.get("failed_tool_name")
        if not isinstance(failed_tool_name, str) or not failed_tool_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty failed_tool_name",
                correction="Retry with the MoveIt tool name that returned the failed result.",
            )
        failed_tool_result = arguments.get("failed_tool_result")
        if not isinstance(failed_tool_result, (dict, str)):
            raise RobotCallValidationError(
                "Expected failed_tool_result",
                correction="Retry with the failed planner or executor output as failed_tool_result.",
            )
        failed_tool_arguments = arguments.get("failed_tool_arguments")
        if failed_tool_arguments is not None and not isinstance(failed_tool_arguments, dict):
            raise RobotCallValidationError(
                "Expected failed_tool_arguments object",
                correction="Omit failed_tool_arguments or retry with the original tool arguments object.",
            )
        user_intent = arguments.get("user_intent")
        if user_intent is not None and not isinstance(user_intent, str):
            raise RobotCallValidationError(
                "Expected user_intent text",
                correction="Omit user_intent or retry with the user's request as text.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_verify_attached_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Retry with the object to verify.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_pick":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name.",
            )
        plan_name = arguments.get("plan_name")
        if plan_name is not None and (not isinstance(plan_name, str) or not plan_name.strip()):
            raise RobotCallValidationError(
                "Expected a non-empty plan_name",
                correction="Omit plan_name or retry with a non-empty plan label.",
            )
        grasp_face = arguments.get("grasp_face")
        if grasp_face is not None and (not isinstance(grasp_face, str) or not grasp_face.strip()):
            raise RobotCallValidationError(
                "Expected a non-empty grasp_face",
                correction="Omit grasp_face or retry with one raw.object.grasp_faces name.",
            )
        planning_strategy = arguments.get("planning_strategy")
        if planning_strategy is not None and planning_strategy not in PLANNING_STRATEGIES:
            raise RobotCallValidationError(
                "Expected planning_strategy to be auto, cartesian, or sampled_approach",
                correction='Use planning_strategy="auto", "cartesian", or "sampled_approach".',
            )
        for distance_name in ("approach_distance_m", "grasp_standoff_m", "lift_distance_m"):
            _validate_pick_distance(arguments.get(distance_name))
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_pick_task":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name.",
            )
        grasp_face = arguments.get("grasp_face")
        if grasp_face is not None and (not isinstance(grasp_face, str) or not grasp_face.strip()):
            raise RobotCallValidationError(
                "Expected a non-empty grasp_face",
                correction="Omit grasp_face or retry with one raw.object.grasp_faces name.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_place":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name.",
            )
        plan_name = arguments.get("plan_name")
        if plan_name is not None and (not isinstance(plan_name, str) or not plan_name.strip()):
            raise RobotCallValidationError(
                "Expected a non-empty plan_name",
                correction="Omit plan_name or retry with a non-empty plan label.",
            )
        target = arguments.get("target_pose", arguments.get("target_position"))
        if target is None:
            raise RobotCallValidationError(
                "Expected target_pose or target_position",
                correction="Retry with an object placement target in base_link.",
            )
        _validate_pose(target)
        orientation_mode = arguments.get("orientation_mode")
        if orientation_mode is not None and orientation_mode not in {
            "keep",
            "horizontal",
            "vertical",
            "explicit",
        }:
            raise RobotCallValidationError(
                "Unsupported orientation_mode",
                correction='Retry with orientation_mode "keep", "horizontal", "vertical", or "explicit".',
            )
        for face_name in ("place_face", "support_face"):
            face = arguments.get(face_name)
            if face is not None and (not isinstance(face, str) or not face.strip()):
                raise RobotCallValidationError(
                    f"Expected a non-empty {face_name}",
                    correction=f"Omit {face_name} or retry with a non-empty face name.",
                )
        for distance_name in ("approach_distance_m", "place_standoff_m", "retreat_distance_m"):
            _validate_place_distance(arguments.get(distance_name))
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_place_task":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name.",
            )
        target = arguments.get("target_pose", arguments.get("target_position"))
        if target is None:
            raise RobotCallValidationError(
                "Expected target_pose or target_position",
                correction="Retry with an object placement target in base_link.",
            )
        _validate_pose(target)
        orientation_mode = arguments.get("orientation_mode")
        if orientation_mode is not None and orientation_mode not in {
            "keep",
            "horizontal",
            "vertical",
            "explicit",
        }:
            raise RobotCallValidationError(
                "Unsupported orientation_mode",
                correction='Retry with orientation_mode "keep", "horizontal", "vertical", or "explicit".',
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_free_motion":
        pose = arguments.get("target_pose", arguments.get("position"))
        _validate_pose(pose)
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_cartesian_motion":
        waypoints = arguments.get("waypoints", arguments.get("positions"))
        _validate_waypoints(waypoints)
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_execute_plan":
        plan_name = arguments.get("plan_name")
        if not isinstance(plan_name, str) or not plan_name:
            raise RobotCallValidationError(
                "Expected a non-empty plan_name",
                correction="Plan first, then retry with the returned plan_name.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name in {"moveit_execute_task_solution", "moveit_execute_task_plan"}:
        task_solution_id = arguments.get("task_solution_id")
        if not isinstance(task_solution_id, str) or not task_solution_id.strip():
            raise RobotCallValidationError(
                "Expected a non-empty task_solution_id",
                correction="Plan a pick/place task first, then retry with the returned task_solution_id.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name in {"moveit_open_gripper", "moveit_close_gripper"}:
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_attach_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Retry with the object name to attach.",
            )


def executable_plan_name(output: str) -> str | None:
    return executable_plan_name_from_output(output)


def ensure_task_solution_execution_allowed(
    context: RobotContextStore,
    arguments: dict[str, Any],
) -> None:
    task_solution_id = arguments.get("task_solution_id")
    if not isinstance(task_solution_id, str) or not task_solution_id.strip():
        raise RobotCallValidationError(
            "Expected a non-empty task_solution_id",
            correction="Plan a pick/place task first, then retry with the returned task_solution_id.",
        )
    status = context.task_solution_execution_approval_status(
        task_solution_id,
    )
    if status.ok:
        return
    if status.reason == "approval_for_different_task_solution":
        raise RobotCallValidationError(
            "Task solution approval points to a different task solution",
            correction="Ask for explicit approval for this returned task_solution_id before executing.",
        )
    if status.reason == "scene_snapshot_changed":
        raise RobotCallValidationError(
            "Task solution scene snapshot changed",
            correction="Re-observe the scene and plan the pick/place task again before executing.",
        )
    if status.reason == "approval_stale_after_new_user_intent":
        raise RobotCallValidationError(
            "Task solution approval is stale after newer user intent",
            correction="Ask for explicit approval for the current task_solution_id before executing.",
        )
    raise RobotCallValidationError(
        "Task solution execution requires explicit approval",
        correction="Ask for explicit approval for the returned task_solution_id before executing.",
    )


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
        raise RobotCallValidationError(
            "Only Vizor robot UR10 is allowed",
            correction='Retry with robot_name="UR10".',
        )


def _validate_timeout(timeout_s: Any) -> None:
    if timeout_s is None:
        return
    if not _finite_number(timeout_s):
        raise RobotCallValidationError(
            "timeout_s must be a finite number",
            correction=f"Retry with timeout_s less than or equal to {DEFAULT_TIMEOUT_MAX_S}.",
        )
    numeric_timeout_s = float(timeout_s)
    if numeric_timeout_s <= 0.0 or numeric_timeout_s > DEFAULT_TIMEOUT_MAX_S:
        raise RobotCallValidationError(
            "timeout_s is outside safe range",
            correction=f"Retry with timeout_s less than or equal to {DEFAULT_TIMEOUT_MAX_S}.",
        )


def _validate_pose(value: Any) -> None:
    if not isinstance(value, dict):
        raise RobotCallValidationError(
            "Expected target_pose with position fields",
            correction="Retry with a MoveIt target pose inside the simulation workspace.",
        )

    position = value.get("position") if isinstance(value.get("position"), dict) else value
    if not isinstance(position, dict):
        raise RobotCallValidationError(
            "Expected position coordinates",
            correction="Retry with x, y, and z coordinates inside the simulation workspace.",
        )
    for axis in ("x", "y", "z"):
        coordinate = _finite_float(position.get(axis))
        if coordinate is None or abs(coordinate) > WORKSPACE_ABS_LIMIT_M:
            raise RobotCallValidationError(
                "Target is outside simulation workspace",
                correction=f"Retry with x/y/z coordinates within +/-{WORKSPACE_ABS_LIMIT_M} m.",
            )

    orientation = value.get("orientation")
    if orientation is None:
        return
    if not isinstance(orientation, dict):
        raise RobotCallValidationError(
            "Expected orientation quaternion",
            correction="Retry with finite x, y, z, and w quaternion values.",
        )
    for component in ("x", "y", "z", "w"):
        rotation = orientation.get(component)
        if not _finite_number(rotation):
            raise RobotCallValidationError(
                "Expected finite orientation values",
                correction="Retry with finite x, y, z, and w quaternion values.",
            )


def _validate_waypoints(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise RobotCallValidationError(
            "Expected at least one waypoint",
            correction="Retry with one or more target poses inside the simulation workspace.",
        )
    for waypoint in value:
        _validate_pose(waypoint)


def _validate_pick_distance(value: Any) -> None:
    if value is None:
        return
    if not _finite_number(value) or float(value) <= 0.0:
        raise RobotCallValidationError(
            "Pick distances must be positive finite numbers",
            correction="Retry with positive approach_distance_m, grasp_standoff_m, and lift_distance_m values.",
        )


def _validate_place_distance(value: Any) -> None:
    if value is None:
        return
    if not _finite_number(value) or float(value) <= 0.0:
        raise RobotCallValidationError(
            "Place distances must be positive finite numbers",
            correction="Retry with positive approach_distance_m, place_standoff_m, and retreat_distance_m values.",
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
