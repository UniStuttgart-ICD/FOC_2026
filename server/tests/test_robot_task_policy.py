from dataclasses import dataclass

from robot_control.task_policy import (
    DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
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
    executable_plans: set[str] | None = None
    gripper: str | None = None
    recent_gripper: bool = False
    held_object: str | None = None
    recent_held_object: bool = True
    seen_plan_max_age_s: float | None = None
    seen_gripper_max_age_s: float | None = None
    seen_held_object_max_age_s: float | None = None

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
        self.seen_plan_max_age_s = max_age_s
        return plan_name in (self.executable_plans or set())

    def gripper_state(self) -> str | None:
        return self.gripper

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool:
        self.seen_gripper_max_age_s = max_age_s
        return self.gripper == state and self.recent_gripper

    def held_object_name(self) -> str | None:
        return self.held_object

    def has_recent_held_object(self, object_name: str, *, max_age_s: float) -> bool:
        self.seen_held_object_max_age_s = max_age_s
        return self.held_object == object_name and self.recent_held_object


def test_policy_allows_observation_without_existing_context() -> None:
    decision = validate_task_step(
        "moveit_get_current_pose",
        {"robot_name": "UR10"},
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_contract_internal_scene_tools_as_standalone_calls() -> None:
    decision = validate_task_step(
        "moveit_verify_released_object",
        {"robot_name": "UR10", "object_name": "dynamic_5"},
        FakeTaskPolicyContext(held_object="dynamic_5"),
        user_text="verify the release",
    )

    assert decision.ok is False
    assert decision.code == "contract_internal_tool"
    assert decision.suggested_next_tool == "moveit_execute_task"
    assert structured_task_policy_error(decision)["code"] == "contract_internal_tool"


def test_policy_allows_motion_without_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_cartesian_motion_without_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "waypoints": [VALID_TARGET_POSE]},
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_cartesian_for_move_then_release_compound_task() -> None:
    decision = validate_task_step(
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "waypoints": [VALID_TARGET_POSE]},
        FakeTaskPolicyContext(),
        user_text="move it 30cm in robot left side and then release the gripper",
    )

    assert decision.ok is False
    assert decision.error == "Compound manipulation tasks must use task planning tools."
    assert decision.correction == (
        "Use moveit_plan_manipulation_task for pick, hold, place, release, or other "
        "multi-stage manipulation tasks."
    )
    assert decision.suggested_next_tool == "moveit_plan_manipulation_task"


def test_policy_rejects_free_motion_for_pick_compound_task() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(),
        user_text="pick up dynamic_5",
    )

    assert decision.ok is False
    assert decision.suggested_next_tool == "moveit_plan_manipulation_task"


def test_policy_allows_release_goal_without_held_object_evidence() -> None:
    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "release", "object_name": "dynamic_5"},
        },
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_release_goal_with_matching_held_object_evidence() -> None:
    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "release", "object_name": "dynamic_5"},
        },
        FakeTaskPolicyContext(held_object="dynamic_5"),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_release_goal_with_stale_held_object_evidence() -> None:
    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "release", "object_name": "dynamic_5"},
        },
        FakeTaskPolicyContext(
            held_object="dynamic_5",
            recent_held_object=False,
        ),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_manipulation_hold_goal_without_prior_robot_state() -> None:
    context = FakeTaskPolicyContext()

    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "hold", "object_name": "dynamic_5"},
        },
        context,
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_manipulation_pick_place_goal_without_prior_robot_state() -> None:
    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "pick_place",
                "object_name": "dynamic_5",
                "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
            },
        },
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_manipulation_release_goal_without_prior_robot_state() -> None:
    decision = validate_task_step(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "release", "object_name": "dynamic_5"},
        },
        FakeTaskPolicyContext(held_object="dynamic_5"),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_allows_cartesian_for_non_manipulation_multi_point_motion() -> None:
    decision = validate_task_step(
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "waypoints": [VALID_TARGET_POSE]},
        FakeTaskPolicyContext(),
        user_text="draw a square and wave",
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_treats_place_planning_as_motion() -> None:
    decision = validate_task_step(
        "moveit_plan_place",
        {
            "robot_name": "UR10",
            "object_name": "beam_001",
            "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
        },
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_execute_plan_when_plan_was_not_returned_by_planning() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "invented-plan"},
        FakeTaskPolicyContext(),
        explicit_execute_requested=True,
    )

    assert decision.ok is False
    assert decision.error == "Cannot execute an unknown or stale plan."
    assert decision.correction == "Plan first, then execute the returned plan_name."
    assert decision.suggested_next_tool == "moveit_plan_free_motion"


def test_policy_rejects_execute_plan_without_plan_name() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10"},
        FakeTaskPolicyContext(),
        explicit_execute_requested=True,
    )

    assert decision.ok is False
    assert decision.error == "Cannot execute an unknown or stale plan."


def test_policy_allows_execute_plan_when_plan_was_recently_recorded() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        FakeTaskPolicyContext(executable_plans={"plan-1"}),
        explicit_execute_requested=True,
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_execute_plan_without_explicit_user_request() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        FakeTaskPolicyContext(executable_plans={"plan-1"}),
    )

    assert decision.ok is False
    assert decision.error == "Execution requires an explicit user request."
    assert decision.correction == "Ask the user to explicitly confirm execution, then retry moveit_execute_plan."
    assert decision.suggested_next_tool is None


def test_policy_passes_executable_plan_freshness_window_to_context() -> None:
    context = FakeTaskPolicyContext(executable_plans={"plan-1"})

    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        context,
        executable_plan_max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
        explicit_execute_requested=True,
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
            error="Cannot attach object before the gripper is known closed.",
            correction="Close the gripper or observe gripper state before attaching.",
            suggested_next_tool="moveit_close_gripper",
        )
    )

    assert payload == {
        "ok": False,
        "error": "Cannot attach object before the gripper is known closed.",
        "correction": "Close the gripper or observe gripper state before attaching.",
        "retryable": True,
        "suggested_next_tool": "moveit_close_gripper",
    }
