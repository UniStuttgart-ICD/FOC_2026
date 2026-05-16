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


class TaskPolicyContext(Protocol):
    def has_recent_robot_observation(self, *, max_age_s: float) -> bool: ...

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool: ...

    def gripper_state(self) -> str | None: ...

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool: ...


@dataclass(frozen=True)
class TaskPolicyDecision:
    ok: bool
    error: str | None = None
    correction: str | None = None
    retryable: bool = True
    suggested_next_tool: str | None = None


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
                "Use moveit_plan_pick_task or moveit_plan_place_task for requests that combine "
                "motion with gripper, attach, detach, pick, place, or release actions."
            ),
            suggested_next_tool=suggested_task_tool,
        )

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
    if any(_contains_phrase(text, term) for term in PICK_INTENT_TERMS):
        return "moveit_plan_pick_task"
    has_place_intent = any(_contains_phrase(text, term) for term in PLACE_INTENT_TERMS)
    has_gripper_intent = any(_contains_phrase(text, term) for term in GRIPPER_INTENT_TERMS)
    has_motion_intent = any(_contains_phrase(text, term) for term in MOTION_INTENT_TERMS)
    has_compound_connector = any(
        _contains_phrase(text, term) for term in COMPOUND_CONNECTOR_TERMS
    )
    if has_place_intent and (has_motion_intent or has_gripper_intent or has_compound_connector):
        return "moveit_plan_place_task"
    if has_gripper_intent and has_motion_intent and has_compound_connector:
        return "moveit_plan_place_task"
    return None


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
    return payload
