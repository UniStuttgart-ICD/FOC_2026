from dataclasses import dataclass

from robot_control.task_policy import (
    DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    TaskPolicyDecision,
    structured_task_policy_error,
    validate_task_step,
)

VALID_TARGET_POSE = {
    "position": {"x": 0.1, "y": 0.2, "z": 0.3},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


@dataclass
class FakeTaskPolicyContext:
    recent_pose: bool = False
    seen_max_age_s: float | None = None

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        self.seen_max_age_s = max_age_s
        return self.recent_pose


def test_policy_allows_observation_without_existing_context() -> None:
    decision = validate_task_step(
        "moveit_get_current_pose",
        {"robot_name": "UR10"},
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_motion_without_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(),
    )

    assert decision.ok is False
    assert decision.error == "Fresh robot pose is required before motion."
    assert decision.correction == "Call moveit_get_current_pose, then retry the motion."
    assert decision.suggested_next_tool == "moveit_get_current_pose"


def test_policy_allows_motion_after_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(recent_pose=True),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_uses_configured_pose_freshness_window() -> None:
    context = FakeTaskPolicyContext(recent_pose=True)

    decision = validate_task_step(
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "waypoints": [VALID_TARGET_POSE]},
        context,
        fresh_observation_max_age_s=DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    )

    assert decision.ok is True
    assert context.seen_max_age_s == DEFAULT_FRESH_OBSERVATION_MAX_AGE_S


def test_structured_task_policy_error_shape() -> None:
    payload = structured_task_policy_error(
        TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before motion.",
            correction="Call moveit_get_current_pose, then retry the motion.",
            suggested_next_tool="moveit_get_current_pose",
        )
    )

    assert payload == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }
