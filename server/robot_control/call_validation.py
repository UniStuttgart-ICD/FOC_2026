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
HOLD_LIFT_DISTANCE_MIN_M = 0.03
HOLD_LIFT_DISTANCE_MAX_M = 0.20
MANIPULATION_TASK_GOAL_VALUES = (
    "hold",
    "place",
    "release",
    "move_and_release",
    "pick_place",
)
MANIPULATION_TASK_GOALS = set(MANIPULATION_TASK_GOAL_VALUES)
MANIPULATION_TASK_GOALS_REQUIRING_TARGET = {
    "place",
    "move_and_release",
    "pick_place",
}
COMPOUND_TASK_GOAL_VALUES = (
    "hold",
    "release",
    "move_and_release",
    "pick_place",
)
COMPOUND_TASK_GOALS = set(COMPOUND_TASK_GOAL_VALUES)
COMPOUND_TASK_GOALS_REQUIRING_TARGET = {
    "move_and_release",
    "pick_place",
}
COMPOUND_TASK_KINDS_REQUIRING_EXECUTION_CONTRACT = MANIPULATION_TASK_GOALS | COMPOUND_TASK_GOALS
SUPPORTED_TASK_PLAN_HANDLERS = {
    "motion",
    "close_gripper",
    "open_gripper",
    "attach_object",
    "release_object",
    "verify_attached_object",
    "verify_released_object",
}
SUPPORTED_TASK_PLAN_REQUIRED_PROOFS = {
    "plan_execution_verified",
    "verified_motion_plan",
    "emulated_motion_plan",
    "verified_gripper_closed",
    "verified_gripper_open",
    "planning_scene_attached",
    "planning_scene_update",
    "attachment_check",
    "attachment_verified",
    "attached_object",
    "release_check",
}
COMPOUND_STAGE_INTENTS = {
    "observe_current_state",
    "approach_object",
    "close_gripper",
    "verify_attached",
    "lift",
    "move_to_pose",
    "adjust_pose",
    "open_gripper",
    "release_object",
    "verify_released",
}
COMPOUND_UNSAFE_STAGE_HINTS = {
    "slide",
    "push",
    "code",
    "raw_code",
    "script",
    "raw_script",
    "waypoint",
    "waypoints",
    "raw_waypoint",
    "raw_waypoints",
}

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
        "moveit_plan_manipulation_task",
        "moveit_plan_compound_task",
        "moveit_execute_task_solution",
        "moveit_execute_task_plan",
        "moveit_go_home",
        "moveit_sync_real_robot_state",
        "moveit_explain_motion_failure",
        "moveit_verify_attached_object",
        "moveit_release_object",
        "moveit_verify_released_object",
        "moveit_remove_scene_object",
    }
)
ALLOWED_ROBOT_TOOLS = frozenset(AGENT_TO_LEGACY_MCP_TOOL_NAMES) | CANONICAL_ONLY_MCP_TOOL_NAMES
CONTRACT_INTERNAL_TOOL_NAMES = frozenset(
    {
        "moveit_release_object",
        "moveit_verify_released_object",
        "moveit_remove_scene_object",
    }
)

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
        "Primary tool for ordinary pick and pick-hold requests involving one MoveIt "
        "planning-scene object. Plan a task solution after "
        "moveit_list_scene_objects, moveit_get_object_context, and moveit_get_current_pose. "
        "It returns a task_solution_id, stage evidence, scene snapshot evidence, and approval "
        "payload. It does not execute motion or gripper actions."
    ),
    "moveit_plan_place_task": (
        "Primary tool for ordinary place/release requests involving a held or attached "
        "object. Plan a task solution for placing one attached MoveIt planning-scene "
        "object. Use a target_pose or target_position plus orientation_mode. It returns a task_solution_id, "
        "stage evidence, scene snapshot evidence, and approval payload. It does not execute "
        "motion or gripper actions."
    ),
    "moveit_plan_compound_task": (
        "Plan a supported compound manipulation task from requirements through the MTC "
        'backend only. Use backend="mtc" with requirements.goal and requirements.object_name. '
        "Use preferences as non-executable planner hints; optional stage_intents are only "
        "stage-intent hints, not trusted executable steps. The backend compiles and solves "
        "the executable task solution. It returns task_solution_id, execution_contract, stage "
        "evidence, scene snapshot evidence, and approval payload. It does not execute motion "
        "or gripper actions."
    ),
    "moveit_plan_manipulation_task": (
        "Plan a staged MoveIt manipulation task from requirements. Use requirements.goal "
        "hold, place, release, move_and_release, or pick_place with requirements.object_name "
        "unless release can use the current held object. Use requirements.target_pose or "
        "requirements.target_position for place, move_and_release, and pick_place. "
        "Optional preferences are non-executable planner hints. It returns task_solution_id, "
        "execution_contract, preview evidence, scene snapshot evidence, and approval payload. "
        "It does not execute motion or gripper actions."
    ),
    "moveit_execute_task_solution": (
        "Execute a returned task_solution_id from moveit_plan_pick_task or moveit_plan_place_task. "
        "Use only for sim/emulated task-solution execution after explicit user intent is bound "
        "to that exact task solution."
    ),
    "moveit_execute_task_plan": (
        "Execute a returned supported task_solution_id with a supported execution_contract "
        "through Verified Real Robot Execution by planning concrete motion stages, executing "
        "each returned plan_name, running verified gripper actions, attaching or releasing "
        "the object as directed, and verifying required proof. Use only after explicit user "
        "intent is bound to that exact task_solution_id. Use timeout_s around 30 for real-robot "
        "execution unless the user asks for a shorter supervised timeout."
    ),
    "moveit_go_home": (
        "Send the real UR10 home through Verified Real Robot Execution, then sync the RViz/MoveIt "
        "fake-controller joint state. Use only when the user or operator explicitly asks to go home."
    ),
    "moveit_sync_real_robot_state": (
        "Sync RViz/MoveIt to the real UR10 by reading real joint state through Verified Real Robot "
        "Execution and publishing the fake-controller joint state. It observes and aligns state; "
        "it does not move the physical robot."
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
    "moveit_release_object": (
        "Release/detach one planning-scene object after the verified executor has opened the "
        "gripper. Requires verified_gripper_open=true and an object_pose supplied by the task "
        "execution_contract; do not call it as a standalone release shortcut."
    ),
    "moveit_verify_released_object": (
        "Verify that one planning-scene object is released/free after an execution_contract "
        "open-gripper and release_object step. Use this proof before claiming a place or release "
        "completed."
    ),
    "moveit_remove_scene_object": (
        "Explicit cleanup tool that removes one free MoveIt planning-scene object and verifies "
        "readback. Use only after direct user/operator cleanup intent; attached objects must be "
        "released and verified before removal."
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
        "into every waypoint. Do not use for compound manipulation tasks involving pick, "
        "place, held objects, gripper, attach, detach, or release; use "
        "moveit_plan_manipulation_task."
    ),
    "moveit_execute_plan": (
        "Execute a returned plan_name from a successful free/cartesian or legacy pick/place "
        "planning tool. Do not use it for task_solution_id values and do not invent plan names."
    ),
    "moveit_open_gripper": "Open the UR10 gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback.",
    "moveit_close_gripper": "Close the UR10 gripper through Vizor and verify /Robot/gripper plus /Robot/status feedback.",
    "moveit_attach_object": (
        "Attach an object in the MoveIt planning scene after the gripper has been closed; "
        "moveit_execute_task_plan uses this only after Verified Real Robot Execution closes the physical gripper."
    ),
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
    "moveit_plan_compound_task": {
        "robot_name",
        "object_name",
        "task_goal",
        "requirements",
        "preferences",
        "stage_intents",
        "target_pose",
        "target_position",
        "backend",
        "timeout_s",
    },
    "moveit_plan_manipulation_task": {
        "robot_name",
        "requirements",
        "preferences",
        "timeout_s",
    },
    "moveit_execute_task_solution": {"robot_name", "task_solution_id", "timeout_s"},
    "moveit_execute_task_plan": {"robot_name", "task_solution_id", "timeout_s"},
    "moveit_go_home": {"robot_name", "timeout_s"},
    "moveit_sync_real_robot_state": {"robot_name", "timeout_s"},
    "moveit_explain_motion_failure": {
        "robot_name",
        "failed_tool_name",
        "failed_tool_arguments",
        "failed_tool_result",
        "user_intent",
        "timeout_s",
    },
    "moveit_verify_attached_object": {"robot_name", "object_name", "timeout_s"},
    "moveit_release_object": {
        "robot_name",
        "object_name",
        "object_pose",
        "verified_gripper_open",
        "timeout_s",
    },
    "moveit_verify_released_object": {"robot_name", "object_name", "timeout_s"},
    "moveit_remove_scene_object": {"robot_name", "object_name", "timeout_s"},
    "moveit_plan_free_motion": {"robot_name", "target_pose", "position", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_plan_cartesian_motion": {"robot_name", "waypoints", "positions", "plan_name", "timeout_s", "allow_existing_name"},
    "moveit_execute_plan": {"robot_name", "plan_name", "timeout_s"},
    "moveit_open_gripper": {"robot_name", "timeout_s"},
    "moveit_close_gripper": {"robot_name", "timeout_s"},
    "moveit_attach_object": {"robot_name", "object_name", "verified_gripper_closed"},
}


class RobotCallValidationError(ValueError):
    """Raised when a robot tool call violates local validation policy."""

    _SUGGESTED_TOOL_UNSET = object()

    def __init__(
        self,
        message: str,
        *,
        correction: str,
        code: str | None = None,
        retryable: bool | None = None,
        suggested_next_tool: str | None | object = _SUGGESTED_TOOL_UNSET,
    ):
        super().__init__(message)
        self.correction = correction
        self.code = code
        self.retryable = retryable
        self.suggested_next_tool = suggested_next_tool


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
        "retryable": exc.retryable if exc.retryable is not None else retryable,
    }
    if exc.code is not None:
        payload["code"] = exc.code
    suggested = (
        exc.suggested_next_tool
        if exc.suggested_next_tool is not RobotCallValidationError._SUGGESTED_TOOL_UNSET
        else suggested_next_tool
    )
    if suggested is not None:
        payload["suggested_next_tool"] = suggested
    return payload


def validate_robot_tool_call(
    name: str,
    arguments: dict[str, Any],
    *,
    allow_contract_internal: bool = False,
) -> None:
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

    if name in CONTRACT_INTERNAL_TOOL_NAMES and not allow_contract_internal:
        raise RobotCallValidationError(
            f"{name} is reserved for cached execution_contract steps",
            correction=(
                "Run moveit_execute_task_plan for the cached execution_contract; "
                "do not call this tool directly."
            ),
            code="contract_internal_tool",
            retryable=False,
            suggested_next_tool="moveit_execute_task_plan",
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

    if name in {"moveit_go_home", "moveit_sync_real_robot_state"}:
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_remove_scene_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Retry with the free planning-scene object to remove.",
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

    if name in {"moveit_verify_attached_object", "moveit_verify_released_object"}:
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Retry with the object to verify.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_release_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            raise RobotCallValidationError(
                "Expected a non-empty object_name",
                correction="Retry with the object to release.",
            )
        if arguments.get("verified_gripper_open") is not True:
            raise RobotCallValidationError(
                "Release requires verified_gripper_open=true",
                correction="Open the gripper through verified execution before releasing the object.",
            )
        object_pose = arguments.get("object_pose")
        if object_pose is None:
            raise RobotCallValidationError(
                "Expected object_pose for release",
                correction="Use the backend execution_contract release step with object_pose evidence.",
            )
        _validate_pose(object_pose)
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

    if name == "moveit_plan_compound_task":
        backend = arguments.get("backend")
        if backend != "mtc":
            raise RobotCallValidationError(
                'moveit_plan_compound_task requires backend="mtc"',
                correction='Retry with backend="mtc"; unsupported compound tasks must fail at planning.',
            )
        requirements = arguments.get("requirements")
        if not isinstance(requirements, dict):
            raise RobotCallValidationError(
                "Expected requirements object",
                correction=(
                    "Retry with requirements.goal and requirements.object_name; use preferences "
                    "and stage_intents only as optional planner hints."
                ),
            )
        goal = requirements.get("goal")
        if goal not in COMPOUND_TASK_GOALS:
            raise RobotCallValidationError(
                "Unsupported compound requirements.goal",
                correction=(
                    "Use requirements.goal hold, release, move_and_release, or pick_place."
                ),
            )
        object_name = requirements.get("object_name")
        if goal != "release" and (not isinstance(object_name, str) or not object_name.strip()):
            raise RobotCallValidationError(
                "Expected requirements.object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name in requirements.object_name.",
                code="object_not_found",
                suggested_next_tool="moveit_list_scene_objects",
            )
        if goal == "hold":
            _validate_hold_lift_distance(requirements.get("lift_distance_m"))
        target = requirements.get("target_pose", requirements.get("target_position"))
        if goal in COMPOUND_TASK_GOALS_REQUIRING_TARGET and target is None:
            raise RobotCallValidationError(
                "Expected requirements.target_pose or requirements.target_position",
                correction="Retry with the compound task target inside requirements.",
            )
        if target is not None:
            _validate_pose(target)
        preferences = arguments.get("preferences")
        if preferences is not None:
            if not isinstance(preferences, dict):
                raise RobotCallValidationError(
                    "Expected preferences object",
                    correction="Omit preferences or retry with non-executable planner hints.",
                )
            for key in preferences:
                if not isinstance(key, str):
                    raise RobotCallValidationError(
                        "Expected preference hint names",
                        correction="Use string preference hint names only.",
                    )
                normalized_key = _normalize_compound_hint_name(key)
                if normalized_key in COMPOUND_UNSAFE_STAGE_HINTS:
                    raise RobotCallValidationError(
                        f"Unsupported compound preference hint: {normalized_key}",
                        correction="Preferences are non-executable planner hints; slide/push/code/raw waypoints are unsupported.",
                    )
        stage_intents = arguments.get("stage_intents")
        if stage_intents is not None:
            if not isinstance(stage_intents, list):
                raise RobotCallValidationError(
                    "Expected stage_intents list",
                    correction="Omit stage_intents or retry with supported stage-intent hint names.",
                )
            for intent in stage_intents:
                if not isinstance(intent, str) or not intent.strip():
                    raise RobotCallValidationError(
                        "Expected stage_intents to contain hint names",
                        correction="Use supported stage-intent hint names only.",
                    )
                normalized_intent = _normalize_compound_hint_name(intent)
                if (
                    normalized_intent in COMPOUND_UNSAFE_STAGE_HINTS
                    or normalized_intent not in COMPOUND_STAGE_INTENTS
                ):
                    raise RobotCallValidationError(
                        f"Unsupported compound stage intent: {normalized_intent}",
                        correction="Use supported stage-intent hints; slide/push/code/raw waypoints are unsupported.",
                    )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_manipulation_task":
        requirements = arguments.get("requirements")
        if not isinstance(requirements, dict):
            raise RobotCallValidationError(
                "Expected requirements object",
                correction=(
                    "Retry with requirements.goal and requirements.object_name; use preferences "
                    "only as optional planner hints."
                ),
            )
        goal = requirements.get("goal")
        if goal not in MANIPULATION_TASK_GOALS:
            raise RobotCallValidationError(
                "Unsupported manipulation requirements.goal",
                correction=(
                    "Use requirements.goal hold, place, release, move_and_release, or pick_place."
                ),
            )
        object_name = requirements.get("object_name")
        if goal != "release" and (not isinstance(object_name, str) or not object_name.strip()):
            raise RobotCallValidationError(
                "Expected requirements.object_name",
                correction="Call moveit_list_scene_objects, then retry with one returned object_name in requirements.object_name.",
                code="object_not_found",
                suggested_next_tool="moveit_list_scene_objects",
            )
        _validate_manipulation_lift_distance(requirements.get("lift_distance_m"))
        target = requirements.get("target_pose", requirements.get("target_position"))
        if goal in MANIPULATION_TASK_GOALS_REQUIRING_TARGET and target is None:
            raise RobotCallValidationError(
                "Expected requirements.target_pose or requirements.target_position",
                correction="Retry with the manipulation task target inside requirements.",
            )
        if target is not None:
            _validate_pose(target)
        preferences = arguments.get("preferences")
        if preferences is not None:
            if not isinstance(preferences, dict):
                raise RobotCallValidationError(
                    "Expected preferences object",
                    correction="Omit preferences or retry with non-executable planner hints.",
                )
            for key in preferences:
                if not isinstance(key, str):
                    raise RobotCallValidationError(
                        "Expected preference hint names",
                        correction="Use string preference hint names only.",
                    )
                normalized_key = _normalize_compound_hint_name(key)
                if normalized_key in COMPOUND_UNSAFE_STAGE_HINTS:
                    raise RobotCallValidationError(
                        f"Unsupported manipulation preference hint: {normalized_key}",
                        correction="Preferences are non-executable planner hints; slide/push/code/raw waypoints are unsupported.",
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
    if not status.ok and status.reason == "approval_for_different_task_solution":
        raise RobotCallValidationError(
            "Task solution approval points to a different task solution",
            correction="Ask for explicit approval for this returned task_solution_id before executing.",
            code="approval_for_different_task_solution",
            suggested_next_tool=None,
        )
    if not status.ok and status.reason == "scene_snapshot_changed":
        raise RobotCallValidationError(
            "Task solution scene snapshot changed",
            correction="Re-observe the scene and plan the pick/place task again before executing.",
            code="stale_scene",
            suggested_next_tool=None,
        )
    if not status.ok and status.reason == "approval_stale_after_new_user_intent":
        raise RobotCallValidationError(
            "Task solution approval is stale after newer user intent",
            correction="Ask for explicit approval for the current task_solution_id before executing.",
            code="approval_stale_after_new_user_intent",
            suggested_next_tool=None,
        )
    if not status.ok and status.reason == "approval_expired":
        raise RobotCallValidationError(
            "Task solution approval expired",
            correction="Ask for explicit approval for the current task_solution_id before executing.",
            code="approval_expired",
            suggested_next_tool=None,
        )
    if not status.ok:
        raise RobotCallValidationError(
            "Task solution execution requires explicit approval",
            correction="Ask for explicit approval for the returned task_solution_id before executing.",
            code=status.reason or "approval_missing",
            suggested_next_tool=None,
        )
    _validate_recent_task_solution_execution_evidence(context, task_solution_id)


def _validate_recent_task_solution_execution_evidence(
    context: RobotContextStore,
    task_solution_id: str,
) -> None:
    recent = context.recent_task_solution
    if recent is None or recent.task_solution_id != task_solution_id:
        raise RobotCallValidationError(
            "Task solution execution requires recent task solution evidence",
            correction="Plan the compound task again, then retry with the returned task_solution_id.",
        )
    if recent.backend != "mtc" and recent.task_kind not in COMPOUND_TASK_KINDS_REQUIRING_EXECUTION_CONTRACT:
        return
    raw = recent.raw
    if not isinstance(raw, dict):
        raise RobotCallValidationError(
            "Task plan execution requires the recent raw task solution",
            correction="Plan the compound task again, then retry with the returned task_solution_id.",
            retryable=False,
            suggested_next_tool=None,
        )
    contract_steps = _task_plan_contract_steps(raw.get("execution_contract"))
    if not contract_steps:
        raise RobotCallValidationError(
            "Task plan execution requires a backend execution_contract",
            correction="Plan the compound task again with a supported execution contract.",
            retryable=False,
            suggested_next_tool=None,
        )
    for step in contract_steps:
        if not isinstance(step, dict):
            raise RobotCallValidationError(
                "Task plan execution_contract contains an unsupported step",
                correction="Plan the compound task again with typed execution steps.",
                retryable=False,
                suggested_next_tool=None,
            )
        handler = step.get("handler")
        if not isinstance(handler, str) or handler not in SUPPORTED_TASK_PLAN_HANDLERS:
            raise RobotCallValidationError(
                "Task plan execution_contract contains an unsupported step handler",
                correction="Plan the compound task again with supported typed handlers.",
                retryable=False,
                suggested_next_tool=None,
            )
        source_stage = step.get("source_stage")
        if not isinstance(source_stage, str) or not source_stage.strip():
            raise RobotCallValidationError(
                "Task plan execution_contract step is missing source_stage",
                correction="Plan the compound task again with source stage metadata.",
                retryable=False,
                suggested_next_tool=None,
            )
        required_proof = step.get("required_proof")
        if not isinstance(required_proof, str) or not required_proof.strip():
            raise RobotCallValidationError(
                "Task plan execution_contract step is missing required_proof",
                correction="Plan the compound task again with proof metadata.",
                retryable=False,
                suggested_next_tool=None,
            )
        if required_proof.strip() not in SUPPORTED_TASK_PLAN_REQUIRED_PROOFS:
            raise RobotCallValidationError(
                "Task plan execution_contract step has unsupported required_proof",
                correction="Plan the compound task again with supported proof metadata.",
                retryable=False,
                suggested_next_tool=None,
            )


def _task_plan_contract_steps(execution_contract: Any) -> list[Any] | None:
    if isinstance(execution_contract, list):
        return execution_contract
    if not isinstance(execution_contract, dict):
        return None
    steps = execution_contract.get("steps")
    return steps if isinstance(steps, list) else None


def execution_result_text(output: str) -> str:
    structured_content = _structured_content(output)
    if isinstance(structured_content, dict):
        verification = structured_content.get("verification")
        if structured_content.get("ok") is True and isinstance(verification, dict):
            if verification.get("result") == "pass":
                return "Motion completed."
    return "I planned the motion, but execution could not be verified."


def _normalize_compound_hint_name(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


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


def _validate_hold_lift_distance(value: Any) -> None:
    if value is None:
        return
    if (
        not _finite_number(value)
        or float(value) < HOLD_LIFT_DISTANCE_MIN_M
        or float(value) > HOLD_LIFT_DISTANCE_MAX_M
    ):
        raise RobotCallValidationError(
            "requirements.lift_distance_m is outside supported hold range",
            correction=(
                "Retry hold with requirements.lift_distance_m between "
                f"{HOLD_LIFT_DISTANCE_MIN_M:.2f} m and {HOLD_LIFT_DISTANCE_MAX_M:.2f} m."
            ),
        )


def _validate_manipulation_lift_distance(value: Any) -> None:
    if value is None:
        return
    if (
        not _finite_number(value)
        or float(value) < HOLD_LIFT_DISTANCE_MIN_M
        or float(value) > HOLD_LIFT_DISTANCE_MAX_M
    ):
        raise RobotCallValidationError(
            "requirements.lift_distance_m is outside supported manipulation range",
            correction=(
                "Retry with requirements.lift_distance_m between "
                f"{HOLD_LIFT_DISTANCE_MIN_M:.2f} m and {HOLD_LIFT_DISTANCE_MAX_M:.2f} m."
            ),
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
