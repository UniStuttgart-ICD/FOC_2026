import json

import pytest

from robot_control.call_validation import (
    RobotCallValidationError,
    agent_tool_description,
    canonical_mcp_tool_name,
    executable_plan_name,
    execution_result_text,
    structured_robot_call_error,
    validate_robot_tool_call,
)

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
    assert canonical_mcp_tool_name("moveit_plan_and_execute_free_motion") == "plan_and_execute_free_motion"
    assert canonical_mcp_tool_name("moveit_plan_and_execute_cartesian_motion") == "plan_and_execute_cartesian_motion"
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


def test_accepts_high_level_plan_and_execute_tool() -> None:
    validate_robot_tool_call(
        "moveit_plan_and_execute_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_POSE, "timeout_s": 10.0},
    )


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
    assert "plan, execute, and verify" in agent_tool_description("moveit_plan_and_execute_free_motion")


def test_cartesian_tool_descriptions_enable_improvisational_tcp_paths() -> None:
    cartesian = agent_tool_description("moveit_plan_cartesian_motion")
    compound = agent_tool_description("moveit_plan_and_execute_cartesian_motion")

    for description in (cartesian, compound):
        lowered = description.lower()
        assert "expressive tcp paths" in lowered
        assert "waving" in lowered
        assert "drawing" in lowered
        assert "multi-point motion" in lowered
        assert "ordered waypoints" in lowered
        assert "preserve orientation" in lowered


def test_free_motion_tool_description_distinguishes_point_to_point_from_paths() -> None:
    description = agent_tool_description("moveit_plan_and_execute_free_motion").lower()

    assert "one target pose" in description
    assert "point-to-point" in description
    assert "not for drawing shapes" in description
