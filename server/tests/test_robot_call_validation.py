import json

import pytest

from robot_control.call_validation import (
    RobotCallValidationError,
    agent_tool_description,
    canonical_mcp_tool_name,
    ensure_task_solution_execution_allowed,
    executable_plan_name,
    execution_result_text,
    structured_robot_call_error,
    validate_robot_tool_call,
)
from robot_control.context import RobotContextStore

VALID_POSE = {
    "position": {"x": 0.57, "y": 0.39, "z": 0.62},
    "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
}


def test_accepts_safe_free_motion_arguments():
    validate_robot_tool_call(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_POSE, "timeout_s": 25.0},
    )


def test_accepts_legacy_free_motion_position_argument():
    validate_robot_tool_call(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "position": VALID_POSE, "timeout_s": 25.0},
    )


def test_accepts_current_pose_observation_arguments():
    validate_robot_tool_call("moveit_get_current_pose", {"robot_name": "UR10", "timeout_s": 2.0})


def test_accepts_robot_state_observation_arguments():
    validate_robot_tool_call("moveit_get_robot_state", {"robot_name": "UR10", "timeout_s": 2.0})
    assert canonical_mcp_tool_name("moveit_get_robot_state") == "moveit_get_robot_state"
    assert "pose" in agent_tool_description("moveit_get_robot_state").lower()
    assert "physical" in agent_tool_description("moveit_get_robot_state").lower()


def test_accepts_scene_object_observation_arguments():
    validate_robot_tool_call("moveit_list_scene_objects", {"robot_name": "UR10", "timeout_s": 2.0})
    validate_robot_tool_call(
        "moveit_get_object_context",
        {"robot_name": "UR10", "object_name": "beam_001", "timeout_s": 2.0},
    )
    assert canonical_mcp_tool_name("moveit_list_scene_objects") == "moveit_list_scene_objects"
    assert canonical_mcp_tool_name("moveit_get_object_context") == "moveit_get_object_context"
    assert "planning-scene object" in agent_tool_description("moveit_list_scene_objects")
    assert "grasp-relevant faces" in agent_tool_description("moveit_get_object_context")


def test_accepts_recovery_home_and_sync_arguments() -> None:
    validate_robot_tool_call("moveit_go_home", {"robot_name": "UR10", "timeout_s": 30.0})
    validate_robot_tool_call(
        "moveit_sync_real_robot_state",
        {"robot_name": "UR10", "timeout_s": 10.0},
    )

    assert canonical_mcp_tool_name("moveit_remove_scene_object") == "moveit_remove_scene_object"
    assert "home" in agent_tool_description("moveit_go_home").lower()
    assert "sync" in agent_tool_description("moveit_sync_real_robot_state").lower()
    assert "explicit" in agent_tool_description("moveit_remove_scene_object").lower()


def test_rejects_standalone_contract_internal_scene_tools() -> None:
    cases = [
        (
            "moveit_release_object",
            {
                "robot_name": "UR10",
                "object_name": "dynamic_5",
                "object_pose": VALID_POSE,
                "verified_gripper_open": True,
            },
        ),
        (
            "moveit_verify_released_object",
            {"robot_name": "UR10", "object_name": "dynamic_5"},
        ),
        (
            "moveit_remove_scene_object",
            {"robot_name": "UR10", "object_name": "dynamic_5"},
        ),
    ]

    for tool_name, arguments in cases:
        with pytest.raises(RobotCallValidationError) as exc_info:
            validate_robot_tool_call(tool_name, arguments)
        assert exc_info.value.code == "contract_internal_tool"
        assert "execution_contract" in exc_info.value.correction


def test_accepts_contract_internal_scene_tools_for_execution_contract_steps() -> None:
    validate_robot_tool_call(
        "moveit_release_object",
        {
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "object_pose": VALID_POSE,
            "verified_gripper_open": True,
        },
        allow_contract_internal=True,
    )
    validate_robot_tool_call(
        "moveit_verify_released_object",
        {"robot_name": "UR10", "object_name": "dynamic_5"},
        allow_contract_internal=True,
    )
    validate_robot_tool_call(
        "moveit_remove_scene_object",
        {"robot_name": "UR10", "object_name": "dynamic_5"},
        allow_contract_internal=True,
    )


def test_accepts_pick_planning_arguments():
    validate_robot_tool_call(
        "moveit_plan_pick",
        {
            "robot_name": "UR10",
            "object_name": "beam_001",
            "grasp_face": "top",
            "approach_distance_m": 0.08,
            "grasp_standoff_m": 0.01,
            "lift_distance_m": 0.1,
            "timeout_s": 10.0,
        },
    )
    assert canonical_mcp_tool_name("moveit_plan_pick") == "moveit_plan_pick"
    description = agent_tool_description("moveit_plan_pick")
    assert "pick" in description.lower()
    assert "raw.plan_name" in description
    assert "feedback.can_execute" in description
    assert "selected grasp face" in description
    assert "approach, pre-grasp, close-gripper, attach, and lift workflow steps" in description
    assert "existing Cartesian planner" in description
    assert "object context" in description
    assert "workflow metadata" in description
    assert "Legacy fallback" in description
    assert "same executable-plan result shape" in description
    assert "does not execute" in description


def test_accepts_task_solution_pick_planning_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_pick_task",
        {
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "grasp_face": "top",
            "timeout_s": 10.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_plan_pick_task") == "moveit_plan_pick_task"
    description = agent_tool_description("moveit_plan_pick_task")
    assert "Primary tool for ordinary pick and pick-hold requests" in description
    assert "task solution" in description
    assert "does not execute" in description


def test_rejects_task_solution_pick_planning_without_object_name() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_pick_task",
            {"robot_name": "UR10", "object_name": ""},
        )

    assert str(exc.value) == "Expected a non-empty object_name"


def test_accepts_task_solution_place_planning_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_place_task",
        {
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
            "orientation_mode": "keep",
            "timeout_s": 10.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_plan_place_task") == "moveit_plan_place_task"
    description = agent_tool_description("moveit_plan_place_task")
    assert "Primary tool for ordinary place/release requests" in description
    assert "task solution" in description


def test_task_tool_descriptions_route_staged_manipulation_tasks() -> None:
    manipulation_task = agent_tool_description("moveit_plan_manipulation_task").lower()
    cartesian = agent_tool_description("moveit_plan_cartesian_motion").lower()

    assert "staged moveit manipulation task" in manipulation_task
    assert "requirements" in manipulation_task
    assert "preferences" in manipulation_task
    assert "hold" in manipulation_task
    assert "place" in manipulation_task
    assert "move_and_release" in manipulation_task
    assert "pick_place" in manipulation_task
    assert "object target pose, not tcp pose" in manipulation_task
    assert "geometry world context" in manipulation_task
    assert "vizor user position" in manipulation_task
    assert "execution_contract" in manipulation_task
    assert "do not use for compound manipulation tasks" in cartesian
    assert "moveit_plan_manipulation_task" in cartesian


def test_accepts_requirements_only_manipulation_task_planning_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "hold", "object_name": "dynamic_5"},
            "timeout_s": 10.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_plan_manipulation_task") == "moveit_plan_manipulation_task"


def test_accepts_manipulation_required_grasp_face() -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "hold",
                "object_name": "dynamic_1",
                "grasp_face": "top",
            },
        },
    )


def test_rejects_empty_manipulation_required_grasp_face() -> None:
    with pytest.raises(RobotCallValidationError) as exc_info:
        validate_robot_tool_call(
            "moveit_plan_manipulation_task",
            {
                "robot_name": "UR10",
                "requirements": {
                    "goal": "hold",
                    "object_name": "dynamic_1",
                    "grasp_face": "",
                },
            },
        )

    assert "requirements.grasp_face" in str(exc_info.value)


@pytest.mark.parametrize("goal", ["hold", "release"])
def test_accepts_untargeted_final_manipulation_task_goals(goal: str) -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": goal, "object_name": "dynamic_5"},
        },
    )


def test_accepts_release_manipulation_without_object_name_for_current_held_object() -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": "release"},
        },
    )


@pytest.mark.parametrize("goal", ["place", "move_and_release", "pick_place"])
def test_accepts_targeted_final_manipulation_task_goals(goal: str) -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": goal,
                "object_name": "dynamic_5",
                "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
            },
        },
    )


@pytest.mark.parametrize("goal", ["pick", "approach_hold_adjust_release"])
def test_rejects_removed_model_visible_manipulation_task_goals(goal: str) -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_manipulation_task",
            {
                "robot_name": "UR10",
                "requirements": {"goal": goal, "object_name": "dynamic_5"},
            },
        )

    structured = structured_robot_call_error(exc.value)
    assert structured["ok"] is False
    assert "hold, place, release, move_and_release, or pick_place" in structured["correction"]
    assert "approach_hold_adjust_release" not in structured["correction"]


@pytest.mark.parametrize("lift_distance_m", [0.0, 0.03, 0.1, 0.2])
def test_accepts_lift_distance_within_manipulation_bounds(lift_distance_m: float) -> None:
    validate_robot_tool_call(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "hold",
                "object_name": "dynamic_5",
                "lift_distance_m": lift_distance_m,
            },
        },
    )


@pytest.mark.parametrize("lift_distance_m", [-0.001, 0.201, "0.1", True, float("nan"), float("inf")])
def test_rejects_lift_distance_outside_manipulation_bounds(lift_distance_m: object) -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_manipulation_task",
            {
                "robot_name": "UR10",
                "requirements": {
                    "goal": "hold",
                    "object_name": "dynamic_5",
                    "lift_distance_m": lift_distance_m,
                },
            },
        )

    assert str(exc.value) == "requirements.lift_distance_m is outside supported manipulation range"
    assert "0.00 m and 0.20 m" in exc.value.correction


def test_accepts_compound_task_preferences_as_non_executable_hints() -> None:
    validate_robot_tool_call(
        "moveit_plan_compound_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "move_and_release",
                "object_name": "dynamic_5",
                "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
            },
            "preferences": {
                "grasp_face": "top",
                "orientation_mode": "keep",
                "retreat_distance_m": 0.08,
                "release_behavior": "open_then_detach",
            },
            "backend": "mtc",
        },
    )


def test_accepts_optional_valid_stage_intents_as_compound_task_hints() -> None:
    validate_robot_tool_call(
        "moveit_plan_compound_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "move_and_release",
                "object_name": "dynamic_5",
                "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
            },
            "stage_intents": [
                "observe_current_state",
                "move_to_pose",
                "open_gripper",
                "release_object",
                "verify_released",
            ],
            "backend": "mtc",
        },
    )

    description = agent_tool_description("moveit_plan_compound_task").lower()
    assert "stage_intents" in description
    assert "optional" in description
    assert "hints" in description


def test_rejects_compound_task_planning_without_requirements() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {"robot_name": "UR10", "backend": "mtc"},
        )

    assert "requirements" in str(exc.value)
    structured = structured_robot_call_error(exc.value)
    assert structured["ok"] is False
    assert structured["correction"]
    assert "requirements.goal" in structured["correction"]
    assert "requirements.object_name" in structured["correction"]


@pytest.mark.parametrize(
    ("requirements", "message"),
    [
        ({"object_name": "dynamic_5"}, "requirements.goal"),
        ({"goal": "hold"}, "requirements.object_name"),
        ({"goal": "hold", "object_name": " "}, "requirements.object_name"),
    ],
)
def test_rejects_compound_task_planning_without_required_requirement_fields(
    requirements: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {"robot_name": "UR10", "requirements": requirements, "backend": "mtc"},
        )

    assert message in str(exc.value)


@pytest.mark.parametrize("goal", ["move_and_release", "pick_place"])
def test_rejects_targeted_compound_goals_without_target_pose_or_position(goal: str) -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "requirements": {"goal": goal, "object_name": "dynamic_5"},
                "backend": "mtc",
            },
        )

    assert str(exc.value) == "Expected requirements.target_pose or requirements.target_position"


@pytest.mark.parametrize("goal", ["hold", "release"])
def test_untargeted_compound_goals_do_not_require_target_pose_or_position(goal: str) -> None:
    validate_robot_tool_call(
        "moveit_plan_compound_task",
        {
            "robot_name": "UR10",
            "requirements": {"goal": goal, "object_name": "dynamic_5"},
            "backend": "mtc",
        },
    )


def test_rejects_unsupported_compound_task_goal() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "requirements": {"goal": "slide", "object_name": "dynamic_5"},
                "backend": "mtc",
            },
        )

    assert "Unsupported" in str(exc.value)
    assert "requirements.goal" in str(exc.value)


def test_structured_compound_object_failure_includes_stable_code() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "requirements": {"goal": "hold", "object_name": " "},
                "backend": "mtc",
            },
        )

    structured = structured_robot_call_error(exc.value)
    assert structured["code"] == "object_not_found"
    assert structured["suggested_next_tool"] == "moveit_list_scene_objects"


def test_rejects_compound_task_planning_without_mtc_backend() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "requirements": {
                    "goal": "move_and_release",
                    "object_name": "dynamic_5",
                    "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
                },
                "backend": "emulated",
            },
        )

    assert str(exc.value) == 'moveit_plan_compound_task requires backend="mtc"'


@pytest.mark.parametrize(
    "hint",
    ["slide", "push", "run_python_code", "script", "raw_waypoints", "waypoints", "unknown_hint"],
)
def test_rejects_unsafe_or_unknown_compound_stage_intent_hint(hint: str) -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "requirements": {
                    "goal": "move_and_release",
                    "object_name": "dynamic_5",
                    "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
                },
                "stage_intents": [hint],
                "backend": "mtc",
            },
        )

    assert "Unsupported compound stage intent" in str(exc.value)
    assert hint in str(exc.value)


def test_accepts_task_solution_execution_arguments() -> None:
    validate_robot_tool_call(
        "moveit_execute_task_solution",
        {
            "robot_name": "UR10",
            "task_solution_id": "pick_task_dynamic_5_001",
            "timeout_s": 30.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_execute_task_solution") == "moveit_execute_task_solution"
    assert "task_solution_id" in agent_tool_description("moveit_execute_task_solution")


def test_accepts_unified_task_execution_arguments() -> None:
    validate_robot_tool_call(
        "moveit_execute_task",
        {
            "robot_name": "UR10",
            "task_solution_id": "pick_task_dynamic_5_001",
            "timeout_s": 30.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_execute_task") == "moveit_execute_task"
    description = agent_tool_description("moveit_execute_task")
    assert "RViz" in description
    assert "real robot" in description
    assert "task_solution_id" in description


def test_rejects_unified_task_execution_plan_name_argument() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_execute_task",
            {
                "robot_name": "UR10",
                "task_solution_id": "pick_task_dynamic_5_001",
                "plan_name": "pick-plan-1",
            },
        )

    assert str(exc.value) == "Unexpected argument for moveit_execute_task: plan_name"


def test_accepts_task_plan_execution_arguments() -> None:
    validate_robot_tool_call(
        "moveit_execute_task_plan",
        {
            "robot_name": "UR10",
            "task_solution_id": "pick_task_dynamic_5_001",
            "timeout_s": 30.0,
        },
    )

    assert canonical_mcp_tool_name("moveit_execute_task_plan") == "moveit_execute_task_plan"
    description = agent_tool_description("moveit_execute_task_plan")
    assert "Verified Real Robot Execution" in description
    assert "task_solution_id" in description
    assert "supported" in description
    assert "execution_contract" in description
    assert "around 60" in description
    assert "pick-only" not in description.lower()
    assert "pick only" not in description.lower()


def test_rejects_task_solution_execution_without_id() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_execute_task_solution",
            {"robot_name": "UR10", "task_solution_id": " "},
        )

    assert str(exc.value) == "Expected a non-empty task_solution_id"


def test_rejects_task_solution_execution_public_scene_snapshot_argument() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_execute_task_solution",
            {
                "robot_name": "UR10",
                "task_solution_id": "pick_task_1",
                "scene_snapshot_id": "scene_1",
            },
        )

    assert str(exc.value) == "Unexpected argument for moveit_execute_task_solution: scene_snapshot_id"


def test_blocks_task_solution_execution_without_matching_approval() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {"task_solution_id": "pick_task_dynamic_5_001"},
        )

    assert str(exc.value) == "Task solution execution requires explicit approval"
    structured = structured_robot_call_error(exc.value)
    assert structured["code"] == "approval_missing"
    assert "suggested_next_tool" not in structured


def test_allows_task_solution_execution_with_matching_current_approval() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="approach grasp, close gripper, attach object, lift object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id="pick_task_dynamic_5_001",
        task_kind="pick",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
    )

    ensure_task_solution_execution_allowed(
        store,
        {
            "robot_name": "UR10",
            "task_solution_id": "pick_task_dynamic_5_001",
            "timeout_s": 30.0,
        },
    )


def test_blocks_task_solution_execution_when_approval_expires_after_60_seconds() -> None:
    now = 100.0
    store = RobotContextStore(time_fn=lambda: now)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="compound_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "compound_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id="compound_task_dynamic_5_001",
        task_kind="hold",
        object_name="dynamic_5",
        backend="mtc",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw={
            "execution_contract": {
                "steps": [
                    {
                        "handler": "verify_attached_object",
                        "source_stage": "verify_attached_object",
                        "object_name": "dynamic_5",
                        "required_proof": "attachment_check",
                    }
                ]
            }
        },
    )

    now = 161.0

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {
                "robot_name": "UR10",
                "task_solution_id": "compound_task_dynamic_5_001",
                "timeout_s": 30.0,
            },
        )

    assert "expired" in str(exc.value)
    structured = structured_robot_call_error(exc.value)
    assert structured["code"] == "approval_expired"
    assert "suggested_next_tool" not in structured


def test_blocks_task_solution_execution_when_approval_is_stale_after_new_user_intent() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="compound_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "compound_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.mark_new_user_intent()

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {
                "robot_name": "UR10",
                "task_solution_id": "compound_task_dynamic_5_001",
                "timeout_s": 30.0,
            },
        )

    structured = structured_robot_call_error(exc.value)
    assert structured["code"] == "approval_stale_after_new_user_intent"
    assert "suggested_next_tool" not in structured


def test_blocks_task_solution_execution_when_scene_snapshot_changed() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="approach grasp, close gripper, attach object, lift object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id="pick_task_dynamic_5_001",
        task_kind="pick",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_002",
        approval_required=True,
    )

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {
                "robot_name": "UR10",
                "task_solution_id": "pick_task_dynamic_5_001",
                "timeout_s": 30.0,
            },
        )

    assert "scene snapshot changed" in str(exc.value)
    structured = structured_robot_call_error(exc.value)
    assert structured["code"] == "stale_scene"
    assert "suggested_next_tool" not in structured


def test_blocks_compound_task_plan_execution_without_recent_solution_evidence() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="compound_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "compound_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {
                "robot_name": "UR10",
                "task_solution_id": "compound_task_dynamic_5_001",
            },
        )

    assert str(exc.value) == "Task solution execution requires recent task solution evidence"


def test_allows_compound_task_plan_execution_with_supported_cached_contract() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="compound_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "compound_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id="compound_task_dynamic_5_001",
        task_kind="hold",
        object_name="dynamic_5",
        backend="mtc",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw={
            "execution_contract": {
                "steps": [
                    {
                        "handler": "motion",
                        "name": "approach",
                        "source_stage": "approach",
                        "object_name": "dynamic_5",
                        "target_pose": VALID_POSE,
                        "required_proof": "plan_execution_verified",
                    },
                    {
                        "handler": "verify_attached_object",
                        "source_stage": "verify_attached_object",
                        "object_name": "dynamic_5",
                        "required_proof": "attachment_verified",
                    },
                ]
            }
        },
    )

    ensure_task_solution_execution_allowed(
        store,
        {
            "robot_name": "UR10",
            "task_solution_id": "compound_task_dynamic_5_001",
        },
    )


def test_allows_long_hybrid_task_plan_execution_contract() -> None:
    task_solution_id = "hybrid_pick_place_dynamic_5_001"
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id=task_solution_id,
        source_tool="moveit_plan_manipulation_task",
        object_name="dynamic_5",
        expected_movement="pick and place object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        task_solution_id,
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id=task_solution_id,
        task_kind="pick_place",
        object_name="dynamic_5",
        backend="staged_moveit",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw={
            "execution_contract": {
                "steps": [
                    {
                        "handler": "motion",
                        "name": "connect_to_pre_grasp",
                        "source_stage": "connect_to_pre_grasp",
                        "target_pose": VALID_POSE,
                        "required_proof": "emulated_motion_plan",
                    },
                    {
                        "handler": "motion",
                        "name": "approach_to_pre_grasp",
                        "source_stage": "approach_to_pre_grasp",
                        "target_pose": VALID_POSE,
                        "required_proof": "emulated_motion_plan",
                    },
                    {
                        "handler": "close_gripper",
                        "name": "close_gripper",
                        "source_stage": "close_gripper",
                        "required_proof": "verified_gripper_closed",
                    },
                    {
                        "handler": "attach_object",
                        "name": "attach_object",
                        "source_stage": "attach_object",
                        "object_name": "dynamic_5",
                        "required_proof": "planning_scene_attached",
                    },
                    {
                        "handler": "motion",
                        "name": "post_grasp_lift",
                        "source_stage": "post_grasp_lift",
                        "target_pose": VALID_POSE,
                        "required_proof": "plan_execution_verified",
                    },
                    {
                        "handler": "motion",
                        "name": "connect_to_place",
                        "source_stage": "connect_to_place",
                        "target_pose": VALID_POSE,
                        "required_proof": "emulated_motion_plan",
                    },
                    {
                        "handler": "motion",
                        "name": "approach_place",
                        "source_stage": "approach_place",
                        "target_pose": VALID_POSE,
                        "required_proof": "emulated_motion_plan",
                    },
                    {
                        "handler": "open_gripper",
                        "name": "open_gripper",
                        "source_stage": "open_gripper",
                        "required_proof": "verified_gripper_open",
                    },
                    {
                        "handler": "release_object",
                        "name": "release_object",
                        "source_stage": "release_object",
                        "object_name": "dynamic_5",
                        "required_proof": "planning_scene_update",
                    },
                    {
                        "handler": "motion",
                        "name": "retreat",
                        "source_stage": "retreat",
                        "target_pose": VALID_POSE,
                        "required_proof": "emulated_motion_plan",
                    },
                    {
                        "handler": "verify_released_object",
                        "name": "verify_released_object",
                        "source_stage": "verify_released_object",
                        "object_name": "dynamic_5",
                        "required_proof": "release_check",
                    },
                ]
            }
        },
    )

    ensure_task_solution_execution_allowed(
        store,
        {
            "robot_name": "UR10",
            "task_solution_id": task_solution_id,
        },
    )


def test_blocks_verified_motion_contract_without_plan_handle() -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    task_solution_id = "hold_task_dynamic_1_003"
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id=task_solution_id,
        source_tool="moveit_plan_manipulation_task",
        object_name="dynamic_1",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260518_003",
    )
    store.record_task_solution_approval(
        task_solution_id,
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id=task_solution_id,
        task_kind="hold",
        object_name="dynamic_1",
        backend="staged_moveit",
        scene_snapshot_id="scene_20260518_003",
        approval_required=True,
        raw={
            "execution_contract": {
                "can_execute": True,
                "steps": [
                    {
                        "handler": "motion",
                        "source_stage": "connect_to_pre_grasp",
                        "required_proof": "verified_motion_plan",
                        "waypoint_index": 0,
                    }
                ],
            }
        },
    )

    with pytest.raises(RobotCallValidationError) as exc_info:
        ensure_task_solution_execution_allowed(
            store,
            {"robot_name": "UR10", "task_solution_id": task_solution_id},
        )

    assert "plan_handle" in str(exc_info.value)


@pytest.mark.parametrize(
    "execution_contract",
    [
        {"steps": [{"handler": "execute_plan", "source_stage": "approach", "required_proof": "plan"}]},
        {"steps": [{"handler": "motion", "source_stage": "approach", "required_proof": "plan"}]},
        {"steps": [{"handler": "motion", "source_stage": "approach"}]},
        {"steps": [{"handler": "motion", "required_proof": "plan_execution_verified"}]},
        {"steps": []},
    ],
)
def test_blocks_compound_task_plan_execution_with_invalid_cached_contract(
    execution_contract: dict[str, object],
) -> None:
    store = RobotContextStore(time_fn=lambda: 100.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="compound_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "compound_task_dynamic_5_001",
        approval_turn_id="turn-1",
        approved_at=100.0,
    )
    store.remember_task_solution(
        task_solution_id="compound_task_dynamic_5_001",
        task_kind="hold",
        object_name="dynamic_5",
        backend="mtc",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw={"execution_contract": execution_contract},
    )

    with pytest.raises(RobotCallValidationError) as exc:
        ensure_task_solution_execution_allowed(
            store,
            {
                "robot_name": "UR10",
                "task_solution_id": "compound_task_dynamic_5_001",
            },
        )

    structured = structured_robot_call_error(exc.value)
    assert structured["ok"] is False
    assert structured["retryable"] is False
    assert structured.get("suggested_next_tool") is None


@pytest.mark.parametrize("strategy", ["auto", "cartesian", "sampled_approach"])
def test_moveit_plan_pick_accepts_planning_strategy(strategy: str) -> None:
    validate_robot_tool_call(
        "moveit_plan_pick",
        {"robot_name": "UR10", "object_name": "beam_001", "planning_strategy": strategy},
    )


@pytest.mark.parametrize("strategy", ["sampled", "ptp", "", 123])
def test_moveit_plan_pick_rejects_invalid_planning_strategy(strategy: object) -> None:
    with pytest.raises(RobotCallValidationError):
        validate_robot_tool_call(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "beam_001", "planning_strategy": strategy},
        )


def test_accepts_semantic_place_planning_arguments():
    validate_robot_tool_call(
        "moveit_plan_place",
        {
            "robot_name": "UR10",
            "object_name": "beam_001",
            "target_pose": {
                "position": {"x": 0.75, "y": 0.2, "z": 0.28},
                "orientation": {"x": 0.0, "y": 0.70710678, "z": 0.0, "w": 0.70710678},
            },
            "orientation_mode": "vertical",
            "place_face": "side",
            "support_face": "table",
            "approach_distance_m": 0.08,
            "place_standoff_m": 0.01,
            "retreat_distance_m": 0.1,
            "timeout_s": 10.0,
        },
    )
    assert canonical_mcp_tool_name("moveit_plan_place") == "moveit_plan_place"
    description = agent_tool_description("moveit_plan_place")
    assert "place" in description.lower()
    assert "object" in description.lower()
    assert "target pose" in description.lower()
    assert "orientation_mode" in description
    assert "raw.plan_name" in description
    assert "same executable-plan result shape" in description
    assert "does not execute" in description


def test_accepts_failure_explanation_arguments():
    validate_robot_tool_call(
        "moveit_explain_motion_failure",
        {
            "robot_name": "UR10",
            "failed_tool_name": "moveit_plan_pick",
            "failed_tool_arguments": {"robot_name": "UR10", "object_name": "beam_001"},
            "failed_tool_result": {
                "ok": False,
                "feedback": {"status": "incomplete path"},
                "verification": {"checks": [{"name": "trajectory_observed", "passed": True}]},
            },
            "user_intent": "pick up the beam",
            "timeout_s": 5.0,
        },
    )
    assert canonical_mcp_tool_name("moveit_explain_motion_failure") == "moveit_explain_motion_failure"
    description = agent_tool_description("moveit_explain_motion_failure")
    assert "failed planner or executor result" in description
    assert "retry guidance" in description
    assert "suggested next tool" in description


def test_accepts_attached_object_verification_arguments():
    validate_robot_tool_call(
        "moveit_verify_attached_object",
        {"robot_name": "UR10", "object_name": "beam_001", "timeout_s": 5.0},
    )
    assert canonical_mcp_tool_name("moveit_verify_attached_object") == "moveit_verify_attached_object"
    description = agent_tool_description("moveit_verify_attached_object")
    assert "Verify that one planning-scene object is attached" in description
    assert "moved with the gripper" in description
    assert "Do not use it to execute" in description


def test_rejects_failure_explanation_without_failed_result():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_explain_motion_failure",
            {"robot_name": "UR10", "failed_tool_name": "moveit_plan_pick"},
        )

    assert str(exc.value) == "Expected failed_tool_result"
    assert "failed planner or executor output" in exc.value.correction


def test_rejects_attached_object_verification_without_object_name():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call("moveit_verify_attached_object", {"robot_name": "UR10"})

    assert str(exc.value) == "Expected a non-empty object_name"
    assert "object to verify" in exc.value.correction


def test_rejects_place_planning_without_target_pose_or_position():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_place",
            {"robot_name": "UR10", "object_name": "beam_001"},
        )

    assert str(exc.value) == "Expected target_pose or target_position"
    assert "object placement target" in exc.value.correction


def test_rejects_empty_object_context_name():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_get_object_context",
            {"robot_name": "UR10", "object_name": " "},
        )

    assert str(exc.value) == "Expected a non-empty object_name"
    assert "moveit_list_scene_objects" in exc.value.correction


def test_rejects_empty_pick_object_name():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": ""},
        )

    assert str(exc.value) == "Expected a non-empty object_name"
    assert "moveit_list_scene_objects" in exc.value.correction


def test_rejects_invalid_pick_distance():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "beam_001", "lift_distance_m": -0.1},
        )

    assert str(exc.value) == "Pick distances must be positive finite numbers"
    assert "positive" in exc.value.correction


def test_accepts_gripper_timeout_arguments() -> None:
    validate_robot_tool_call("moveit_open_gripper", {"robot_name": "UR10", "timeout_s": 5.0})
    validate_robot_tool_call("moveit_close_gripper", {"robot_name": "UR10", "timeout_s": 5.0})


def test_accepts_attach_after_verified_gripper_close_argument() -> None:
    validate_robot_tool_call(
        "moveit_attach_object",
        {
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "verified_gripper_closed": True,
        },
    )


def test_rejects_unknown_tool():
    with pytest.raises(RobotCallValidationError, match="Tool is not allowed"):
        validate_robot_tool_call("move_to_position", {"robot_name": "UR10"})


def test_rejects_non_ur10_robot_name():
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call("moveit_open_gripper", {"robot_name": "UR5"})

    assert str(exc.value) == "Only Vizor robot UR10 is allowed"
    assert exc.value.correction == 'Retry with robot_name="UR10".'


def test_rejects_workspace_escape():
    unsafe_pose = {
        "position": {"x": 99.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": unsafe_pose},
        )

    assert str(exc.value) == "Target is outside simulation workspace"
    assert "within +/-1.5 m" in exc.value.correction


def test_maps_canonical_agent_tool_to_legacy_mcp_tool_name():
    assert canonical_mcp_tool_name("moveit_get_current_pose") == "get_current_pose"
    assert canonical_mcp_tool_name("moveit_plan_free_motion") == "plan_free_motion"
    assert canonical_mcp_tool_name("moveit_open_gripper") == "open_gripper"


def test_extracts_executable_plan_name_from_structured_tool_output():
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "plan-1"},
            }
        }
    )

    assert executable_plan_name(output) == "plan-1"


def test_extracts_pick_executable_plan_name_from_structured_tool_output() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "pick-plan-1", "selected_grasp_face": "top"},
            }
        }
    )

    assert executable_plan_name(output) == "pick-plan-1"


def test_does_not_extract_non_executable_place_plan_name() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": False},
                "raw": {"plan_name": "place-plan-1"},
            }
        }
    )

    assert executable_plan_name(output) is None


def test_execution_result_text_requires_passed_verification():
    success = json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
    failure = json.dumps({"structured_content": {"ok": True, "verification": {"result": "fail"}}})

    assert execution_result_text(success) == "Motion completed."
    assert execution_result_text(failure) == "I planned the motion, but execution could not be verified."


def test_accepts_cartesian_motion_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_cartesian_motion",
        {
            "robot_name": "UR10",
            "waypoints": [VALID_POSE, {**VALID_POSE, "position": {"x": 0.57, "y": 0.39, "z": 0.67}}],
            "timeout_s": 10.0,
        },
    )


def test_rejects_empty_cartesian_waypoints() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call("moveit_plan_cartesian_motion", {"robot_name": "UR10", "waypoints": []})

    assert str(exc.value) == "Expected at least one waypoint"


def test_rejects_combined_plan_and_execute_tools() -> None:
    with pytest.raises(RobotCallValidationError) as exc:
        validate_robot_tool_call(
            "moveit_plan_and_execute_free_motion",
            {"robot_name": "UR10", "target_pose": VALID_POSE, "timeout_s": 10.0},
        )

    assert str(exc.value) == "Tool is not allowed: moveit_plan_and_execute_free_motion"
    assert "Plan with moveit_plan_free_motion or moveit_plan_cartesian_motion" in exc.value.correction


def test_structured_robot_call_error_shape() -> None:
    err = RobotCallValidationError("bad target", correction="Use a safe target.")

    assert structured_robot_call_error(err) == {
        "ok": False,
        "error": "bad target",
        "correction": "Use a safe target.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


def test_agent_tool_descriptions_are_high_signal() -> None:
    assert "current end-effector pose" in agent_tool_description("moveit_get_current_pose")
    assert "target pose" in agent_tool_description("moveit_plan_free_motion")
    assert "Cartesian" in agent_tool_description("moveit_plan_cartesian_motion")
    assert "retry guidance" in agent_tool_description("moveit_explain_motion_failure")
    assert "attached object" in agent_tool_description("moveit_verify_attached_object").lower()


def test_gripper_tool_descriptions_match_vizor_feedback_contract() -> None:
    for tool_name in ("moveit_open_gripper", "moveit_close_gripper"):
        description = agent_tool_description(tool_name).lower()

        assert "vizor" in description
        assert "/robot/gripper" in description
        assert "/robot/status" in description
        assert "simulated" not in description


def test_cartesian_tool_descriptions_enable_improvisational_tcp_paths() -> None:
    cartesian = agent_tool_description("moveit_plan_cartesian_motion")

    lowered = cartesian.lower()
    assert "expressive tcp paths" in lowered
    assert "waving" in lowered
    assert "drawing" in lowered
    assert "multi-point motion" in lowered
    assert "ordered waypoints" in lowered
    assert "preserve orientation" in lowered
    assert "copy" in lowered
    assert "raw.pose.orientation" in lowered
    assert "fresh current pose" in lowered
    assert "bounded" not in lowered
    assert "workspace" not in lowered


def test_free_motion_tool_description_distinguishes_point_to_point_from_paths() -> None:
    description = agent_tool_description("moveit_plan_free_motion").lower()

    assert "one target pose" in description
    assert "point-to-point" in description
    assert "not for drawing shapes" in description
