from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_FRESH_OBSERVATION_MAX_AGE_S = 15.0

MOTION_TOOL_NAMES = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_and_execute_free_motion",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_execute_plan",
    }
)


class TaskPolicyContext(Protocol):
    def has_recent_robot_observation(self, *, max_age_s: float) -> bool: ...


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
) -> TaskPolicyDecision:
    del arguments
    if name in MOTION_TOOL_NAMES and not context.has_recent_robot_observation(
        max_age_s=fresh_observation_max_age_s
    ):
        return TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before motion.",
            correction="Call moveit_get_current_pose, then retry the motion.",
            suggested_next_tool="moveit_get_current_pose",
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
