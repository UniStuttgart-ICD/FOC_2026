import json

import pytest

from voice_runtime.robot_safety import (
    RobotSafetyError,
    canonical_mcp_tool_name,
    executable_plan_name,
    execution_result_text,
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
