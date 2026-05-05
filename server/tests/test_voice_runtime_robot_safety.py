import json

import pytest

from voice_runtime.robot_safety import (
    RobotSafetyError,
    agent_tool_description,
    canonical_mcp_tool_name,
    executable_plan_name,
    execution_result_text,
    structured_robot_error,
    validate_robot_tool_call,
)

VALID_POSE = {
    "position": {"x": 0.57, "y": 0.39, "z": 0.62},
    "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
}


def test_accepts_safe_free_motion_arguments():
    validate_robot_tool_call(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "position": VALID_POSE, "timeout_s": 25.0},
    )


def test_rejects_unknown_tool():
    with pytest.raises(RobotSafetyError, match="Tool is not allowed"):
        validate_robot_tool_call("move_to_position", {"robot_name": "UR10"})


def test_rejects_non_ur10_robot_name():
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call("moveit_open_gripper", {"robot_name": "UR5"})

    assert str(exc.value) == "Only Vizor robot UR10 is allowed"
    assert exc.value.correction == 'Retry with robot_name="UR10".'


def test_rejects_workspace_escape():
    unsafe_pose = {
        "position": {"x": 99.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "position": unsafe_pose},
        )

    assert str(exc.value) == "Target is outside simulation workspace"
    assert "within +/-1.5 m" in exc.value.correction


def test_maps_canonical_agent_tool_to_legacy_mcp_tool_name():
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


def test_execution_result_text_requires_passed_verification():
    success = json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
    failure = json.dumps({"structured_content": {"ok": True, "verification": {"result": "fail"}}})

    assert execution_result_text(success) == "Motion completed."
    assert execution_result_text(failure) == "I planned the motion, but execution could not be verified."


def test_accepts_relative_motion_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_relative_motion",
        {
            "robot_name": "UR10",
            "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
            "motion_type": "free",
            "timeout_s": 10.0,
        },
    )


def test_rejects_relative_motion_outside_delta_limit() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call(
            "moveit_plan_relative_motion",
            {
                "robot_name": "UR10",
                "delta": {"x": 0.0, "y": 0.0, "z": 2.0},
                "motion_type": "free",
            },
        )

    assert str(exc.value) == "Relative motion is outside safe delta range"
    assert "within +/-0.30 m" in exc.value.correction


def test_rejects_unknown_relative_motion_type() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call(
            "moveit_plan_relative_motion",
            {
                "robot_name": "UR10",
                "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
                "motion_type": "diagonal",
            },
        )

    assert str(exc.value) == "motion_type must be free or linear"


def test_accepts_named_pose_tools() -> None:
    validate_robot_tool_call("moveit_list_named_poses", {"robot_name": "UR10"})
    validate_robot_tool_call(
        "moveit_plan_named_pose",
        {"robot_name": "UR10", "pose_name": "home", "timeout_s": 10.0},
    )


def test_rejects_empty_named_pose() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call("moveit_plan_named_pose", {"robot_name": "UR10", "pose_name": ""})

    assert str(exc.value) == "Expected a non-empty pose_name"


def test_structured_robot_error_shape() -> None:
    err = RobotSafetyError("bad target", correction="Use a safe target.")

    assert structured_robot_error(err) == {
        "ok": False,
        "error": "bad target",
        "correction": "Use a safe target.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_robot_status",
    }


def test_agent_tool_descriptions_are_high_signal() -> None:
    assert "fresh robot state" in agent_tool_description("moveit_get_robot_status")
    assert "relative" in agent_tool_description("moveit_plan_relative_motion")
    assert "named" in agent_tool_description("moveit_plan_named_pose")
