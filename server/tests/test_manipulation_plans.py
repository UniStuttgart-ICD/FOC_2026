import json

from robot_control.manipulation_plans import (
    ManipulationFollowUpAction,
    parse_executable_plan_result,
    parse_task_solution_result,
)


def test_parse_executable_pick_plan_result() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {
                    "plan_name": "pick-plan-1",
                    "selected_grasp_face": "top",
                    "candidate_attempts": [{"face": "top", "ok": True}],
                },
            }
        }
    )

    result = parse_executable_plan_result("moveit_plan_pick", output)

    assert result is not None
    assert result.tool_name == "moveit_plan_pick"
    assert result.plan_name == "pick-plan-1"
    assert result.robot_name == "UR10"
    assert result.can_execute is True
    assert result.raw["selected_grasp_face"] == "top"


def test_parse_executable_place_plan_result_from_robot_name() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot_name": "UR10",
                "feedback": {"can_execute": True},
                "raw": {
                    "plan_name": "place-plan-1",
                    "release_tcp_pose": {"position": {"x": 0.7, "y": 0.1, "z": 0.3}},
                },
            }
        }
    )

    result = parse_executable_plan_result("moveit_plan_place", output)

    assert result is not None
    assert result.plan_name == "place-plan-1"
    assert result.robot_name == "UR10"
    assert result.raw["release_tcp_pose"]["position"]["z"] == 0.3


def test_parse_plan_result_preserves_after_success_action() -> None:
    follow_up_args = {
        "robot_name": "UR10",
        "object_name": "beam_001",
        "target_position": {"x": 0.7, "y": 0.1, "z": 0.3},
    }
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {
                    "plan_name": "pick-plan-1",
                    "next_action": {
                        "after_success": {
                            "tool": "moveit_plan_place",
                            "arguments": follow_up_args,
                        }
                    },
                },
            }
        }
    )

    result = parse_executable_plan_result("moveit_plan_pick", output)

    assert result is not None
    assert result.after_success == ManipulationFollowUpAction(
        tool="moveit_plan_place",
        arguments=follow_up_args,
    )


def test_parse_plan_result_rejects_non_executable_plan() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": False},
                "raw": {"plan_name": "blocked-plan"},
            }
        }
    )

    assert parse_executable_plan_result("moveit_plan_pick", output) is None


def test_parse_pick_plan_result_rejects_partial_legacy_diagnostic() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "error": "pick_segment_planning_failed",
                "failed_segment": "local_cartesian_pick",
                "feedback": {"can_execute": False},
                "raw": {
                    "partial_plan": {
                        "kind": "preposition",
                        "plan_name": "pick_dynamic_5_preposition",
                    }
                },
            }
        }
    )

    assert parse_executable_plan_result("moveit_plan_pick", output) is None


def test_parse_plan_result_rejects_malformed_json() -> None:
    assert parse_executable_plan_result("moveit_plan_place", "not-json") is None


def test_parse_task_solution_result_from_task_planner_output() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True, "execution_target": "task_solution"},
                "raw": {
                    "task_solution_id": "pick_task_dynamic_5_001",
                    "task_kind": "pick",
                    "backend": "emulated",
                    "object_name": "dynamic_5",
                    "robot_name": "UR10",
                    "created_from_tool": "moveit_plan_pick_task",
                    "scene_snapshot_id": "scene_20260515_001",
                    "approval": {
                        "required": True,
                        "target_kind": "task_solution",
                        "task_solution_id": "pick_task_dynamic_5_001",
                        "source_tool": "moveit_plan_pick_task",
                        "object_name": "dynamic_5",
                        "expected_movement": "approach grasp, close gripper, attach object, lift object",
                        "scene_snapshot_id": "scene_20260515_001",
                    },
                },
            }
        }
    )

    solution = parse_task_solution_result("moveit_plan_pick_task", output)

    assert solution is not None
    assert solution.task_solution_id == "pick_task_dynamic_5_001"
    assert solution.task_kind == "pick"
    assert solution.backend == "emulated"
    assert solution.object_name == "dynamic_5"
    assert solution.scene_snapshot_id == "scene_20260515_001"
    assert solution.approval_required is True


def test_parse_task_solution_result_rejects_ordinary_plan_result() -> None:
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "pick-plan-1"},
            }
        }
    )

    assert parse_task_solution_result("moveit_plan_pick_task", output) is None
