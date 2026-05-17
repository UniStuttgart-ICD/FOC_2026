from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_FRESH_OBSERVATION_MAX_AGE_S = 15.0
DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S = 120.0
DEFAULT_GRIPPER_STATE_MAX_AGE_S = 30.0

MOTION_TOOL_NAMES = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_place",
        "moveit_execute_plan",
    }
)
NON_TASK_MANIPULATION_PLANNING_TOOLS = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_pick",
        "moveit_plan_place",
    }
)
PICK_INTENT_TERMS = ("pick", "pick up", "grab", "grasp")
PLACE_INTENT_TERMS = ("place", "release", "let go", "drop")
GRIPPER_INTENT_TERMS = ("gripper", "attach", "detach")
MOTION_INTENT_TERMS = ("move", "carry", "bring", "take", "put")
COMPOUND_CONNECTOR_TERMS = ("and", "then", "after", "before", "followed by")
TASK_MANIPULATION_PLANNER = "moveit_plan_manipulation_task"
COMPOUND_GOALS_REQUIRING_FRESH_POSE = frozenset({"hold", "place", "move_and_release", "pick_place"})
CONTRACT_INTERNAL_TOOL_NAMES = frozenset(
    {
        "moveit_release_object",
        "moveit_verify_released_object",
        "moveit_remove_scene_object",
    }
)


class TaskPolicyContext(Protocol):
    def has_recent_robot_observation(self, *, max_age_s: float) -> bool: ...

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool: ...

    def gripper_state(self) -> str | None: ...

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool: ...

    def held_object_name(self) -> str | None: ...

    def has_recent_held_object(self, object_name: str, *, max_age_s: float) -> bool: ...


@dataclass(frozen=True)
class TaskPolicyDecision:
    ok: bool
    error: str | None = None
    correction: str | None = None
    retryable: bool = True
    suggested_next_tool: str | None = None
    code: str | None = None


def validate_task_step(
    name: str,
    arguments: dict[str, Any],
    context: TaskPolicyContext,
    *,
    user_text: str | None = None,
    fresh_observation_max_age_s: float = DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    executable_plan_max_age_s: float = DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    gripper_state_max_age_s: float = DEFAULT_GRIPPER_STATE_MAX_AGE_S,
    explicit_execute_requested: bool = False,
) -> TaskPolicyDecision:
    if name in CONTRACT_INTERNAL_TOOL_NAMES:
        return TaskPolicyDecision(
            ok=False,
            error=f"{name} is reserved for cached execution_contract steps.",
            correction=(
                "Run moveit_execute_task_plan for the cached execution_contract; "
                "do not call this tool directly."
            ),
            retryable=False,
            suggested_next_tool="moveit_execute_task_plan",
            code="contract_internal_tool",
        )

    if name in MOTION_TOOL_NAMES and not context.has_recent_robot_observation(
        max_age_s=fresh_observation_max_age_s
    ):
        return TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before motion.",
            correction="Call moveit_get_current_pose, then retry the motion.",
            suggested_next_tool="moveit_get_current_pose",
        )

    suggested_task_tool = _suggested_task_tool_for_compound_intent(name, user_text)
    if suggested_task_tool is not None:
        return TaskPolicyDecision(
            ok=False,
            error="Compound manipulation tasks must use task planning tools.",
            correction=(
                "Use moveit_plan_manipulation_task for pick, hold, place, release, or other "
                "multi-stage manipulation tasks."
            ),
            suggested_next_tool=suggested_task_tool,
        )

    release_decision = _validate_compound_release_preconditions(
        name,
        arguments,
        context,
        fresh_observation_max_age_s=fresh_observation_max_age_s,
        held_object_max_age_s=gripper_state_max_age_s,
    )
    if release_decision is not None:
        return release_decision

    compound_motion_decision = _validate_compound_motion_preconditions(
        name,
        arguments,
        context,
        fresh_observation_max_age_s=fresh_observation_max_age_s,
    )
    if compound_motion_decision is not None:
        return compound_motion_decision

    if name == "moveit_execute_plan":
        plan_name = arguments.get("plan_name")
        if not isinstance(plan_name, str) or not context.has_recent_executable_plan(
            plan_name,
            max_age_s=executable_plan_max_age_s,
        ):
            return TaskPolicyDecision(
                ok=False,
                error="Cannot execute an unknown or stale plan.",
                correction="Plan first, then execute the returned plan_name.",
                suggested_next_tool="moveit_plan_free_motion",
            )
        if not explicit_execute_requested:
            return TaskPolicyDecision(
                ok=False,
                error="Execution requires an explicit user request.",
                correction="Ask the user to explicitly confirm execution, then retry moveit_execute_plan.",
                suggested_next_tool=None,
            )

    if name == "moveit_attach_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            return TaskPolicyDecision(
                ok=False,
                error="Cannot attach an unnamed object.",
                correction="Retry with the object_name to attach.",
                suggested_next_tool=None,
            )
        if not context.has_recent_gripper_state("closed", max_age_s=gripper_state_max_age_s):
            return TaskPolicyDecision(
                ok=False,
                error="Cannot attach object before the gripper is known closed.",
                correction="Close the gripper or observe gripper state before attaching.",
                suggested_next_tool="moveit_close_gripper",
            )

    return TaskPolicyDecision(ok=True)


def _suggested_task_tool_for_compound_intent(name: str, user_text: str | None) -> str | None:
    if name not in NON_TASK_MANIPULATION_PLANNING_TOOLS or not user_text:
        return None
    text = _normalized_text(user_text)
    has_pick_intent = any(_contains_phrase(text, term) for term in PICK_INTENT_TERMS)
    has_place_intent = any(_contains_phrase(text, term) for term in PLACE_INTENT_TERMS)
    has_gripper_intent = any(_contains_phrase(text, term) for term in GRIPPER_INTENT_TERMS)
    has_motion_intent = any(_contains_phrase(text, term) for term in MOTION_INTENT_TERMS)
    has_compound_connector = any(
        _contains_phrase(text, term) for term in COMPOUND_CONNECTOR_TERMS
    )
    if has_pick_intent:
        return TASK_MANIPULATION_PLANNER
    if has_place_intent:
        return TASK_MANIPULATION_PLANNER
    if has_gripper_intent and has_motion_intent and has_compound_connector:
        return TASK_MANIPULATION_PLANNER
    return None


def _validate_compound_release_preconditions(
    name: str,
    arguments: dict[str, Any],
    context: TaskPolicyContext,
    *,
    fresh_observation_max_age_s: float,
    held_object_max_age_s: float,
) -> TaskPolicyDecision | None:
    if name not in {"moveit_plan_compound_task", TASK_MANIPULATION_PLANNER}:
        return None
    requirements = arguments.get("requirements")
    if not isinstance(requirements, dict):
        return None
    goal = requirements.get("goal")
    if goal not in {"release", "move_and_release"}:
        return None
    object_name = requirements.get("object_name")
    held_object_name = context.held_object_name()
    if (
        not isinstance(object_name, str)
        or not object_name.strip()
        or held_object_name != object_name.strip()
    ):
        return TaskPolicyDecision(
            ok=False,
            error="Cannot release an object that is not currently held.",
            correction="Verify the held object or plan a hold task before release.",
            suggested_next_tool="moveit_verify_attached_object",
            code="not_holding_object",
        )
    if not context.has_recent_held_object(object_name.strip(), max_age_s=held_object_max_age_s):
        return TaskPolicyDecision(
            ok=False,
            error="Cannot release an object without recent held-object evidence.",
            correction="Verify the held object before planning release.",
            suggested_next_tool="moveit_verify_attached_object",
            code="stale_held_object",
        )
    if not context.has_recent_robot_observation(max_age_s=fresh_observation_max_age_s):
        return TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before release.",
            correction="Call moveit_get_current_pose, then retry the release plan.",
            suggested_next_tool="moveit_get_current_pose",
        )
    return None


def _validate_compound_motion_preconditions(
    name: str,
    arguments: dict[str, Any],
    context: TaskPolicyContext,
    *,
    fresh_observation_max_age_s: float,
) -> TaskPolicyDecision | None:
    if name not in {"moveit_plan_compound_task", TASK_MANIPULATION_PLANNER}:
        return None
    requirements = arguments.get("requirements")
    if not isinstance(requirements, dict):
        return None
    if requirements.get("goal") not in COMPOUND_GOALS_REQUIRING_FRESH_POSE:
        return None
    if context.has_recent_robot_observation(max_age_s=fresh_observation_max_age_s):
        return None
    return TaskPolicyDecision(
        ok=False,
        error="Fresh robot pose is required before compound task planning.",
        correction="Call moveit_get_current_pose, then retry the compound task plan.",
        suggested_next_tool="moveit_get_current_pose",
    )


def _normalized_text(text: str) -> str:
    return " ".join(text.lower().split())


def _contains_phrase(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def structured_task_policy_error(decision: TaskPolicyDecision) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": decision.error or "Task policy rejected the robot step.",
        "correction": decision.correction or "Revise the robot step and retry.",
        "retryable": decision.retryable,
    }
    if decision.suggested_next_tool is not None:
        payload["suggested_next_tool"] = decision.suggested_next_tool
    if decision.code is not None:
        payload["code"] = decision.code
    return payload
