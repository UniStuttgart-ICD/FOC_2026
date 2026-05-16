from __future__ import annotations

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
