from dataclasses import dataclass

from robot_control.task_policy import (
    DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    DEFAULT_GRIPPER_STATE_MAX_AGE_S,
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
    executable_plans: set[str] | None = None
    gripper: str | None = None
    recent_gripper: bool = False
    seen_pose_max_age_s: float | None = None
    seen_plan_max_age_s: float | None = None
    seen_gripper_max_age_s: float | None = None

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        self.seen_pose_max_age_s = max_age_s
        return self.recent_pose

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
        self.seen_plan_max_age_s = max_age_s
        return plan_name in (self.executable_plans or set())

    def gripper_state(self) -> str | None:
        return self.gripper

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool:
        self.seen_gripper_max_age_s = max_age_s
        return self.gripper == state and self.recent_gripper


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
    assert context.seen_pose_max_age_s == DEFAULT_FRESH_OBSERVATION_MAX_AGE_S


def test_policy_rejects_execute_plan_when_plan_was_not_returned_by_planning() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "invented-plan"},
        FakeTaskPolicyContext(recent_pose=True),
    )

    assert decision.ok is False
    assert decision.error == "Cannot execute an unknown or stale plan."
    assert decision.correction == "Plan first, then execute the returned plan_name."
    assert decision.suggested_next_tool == "moveit_plan_free_motion"


def test_policy_rejects_execute_plan_without_plan_name() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10"},
        FakeTaskPolicyContext(recent_pose=True),
    )

    assert decision.ok is False
    assert decision.error == "Cannot execute an unknown or stale plan."


def test_policy_allows_execute_plan_when_plan_was_recently_recorded() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        FakeTaskPolicyContext(recent_pose=True, executable_plans={"plan-1"}),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_passes_executable_plan_freshness_window_to_context() -> None:
    context = FakeTaskPolicyContext(recent_pose=True, executable_plans={"plan-1"})

    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        context,
        executable_plan_max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    )

    assert decision.ok is True
    assert context.seen_plan_max_age_s == DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S


def test_policy_rejects_attach_when_gripper_state_is_unknown() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach object before the gripper is known closed."
    assert decision.correction == "Close the gripper or observe gripper state before attaching."
    assert decision.suggested_next_tool == "moveit_close_gripper"


def test_policy_rejects_attach_when_gripper_state_is_stale() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=False),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach object before the gripper is known closed."


def test_policy_allows_attach_when_gripper_is_recently_closed() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=True),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_passes_gripper_freshness_window_to_context() -> None:
    context = FakeTaskPolicyContext(gripper="closed", recent_gripper=True)

    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        context,
        gripper_state_max_age_s=DEFAULT_GRIPPER_STATE_MAX_AGE_S,
    )

    assert decision.ok is True
    assert context.seen_gripper_max_age_s == DEFAULT_GRIPPER_STATE_MAX_AGE_S


def test_policy_rejects_attach_without_object_name() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": ""},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=True),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach an unnamed object."
    assert decision.correction == "Retry with the object_name to attach."
    assert decision.suggested_next_tool is None


def test_structured_execute_plan_policy_error_shape() -> None:
    payload = structured_task_policy_error(
        TaskPolicyDecision(
            ok=False,
            error="Cannot execute an unknown or stale plan.",
            correction="Plan first, then execute the returned plan_name.",
            suggested_next_tool="moveit_plan_free_motion",
        )
    )

    assert payload == {
        "ok": False,
        "error": "Cannot execute an unknown or stale plan.",
        "correction": "Plan first, then execute the returned plan_name.",
        "retryable": True,
        "suggested_next_tool": "moveit_plan_free_motion",
    }


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
