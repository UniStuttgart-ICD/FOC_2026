from __future__ import annotations

import json

import pytest

from model_eval.simulated_moveit import SimulatedMoveItAdapter


@pytest.mark.asyncio
async def test_current_pose_returns_deterministic_ur10_pose() -> None:
    adapter = SimulatedMoveItAdapter()

    await adapter.connect()
    output = json.loads(await adapter.call_tool("moveit_get_current_pose", {"robot_name": "UR10"}))

    assert output["is_error"] is False
    assert output["structured_content"]["ok"] is True
    pose = output["structured_content"]["raw"]["pose"]
    assert pose["position"] == {"x": 0.4, "y": 0.1, "z": 0.3}
    assert pose["orientation"]["w"] == 1.0


@pytest.mark.asyncio
async def test_cartesian_execution_updates_pose_and_marks_verified() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    output = json.loads(
        await adapter.call_tool(
            "moveit_plan_and_execute_cartesian_motion",
            {
                "robot_name": "UR10",
                "waypoints": [
                    {"position": {"x": 0.4, "y": 0.1, "z": 0.35}},
                    {"position": {"x": 0.4, "y": 0.1, "z": 0.36}},
                ],
            },
        )
    )
    pose_output = json.loads(await adapter.call_tool("moveit_get_current_pose", {"robot_name": "UR10"}))

    assert output["structured_content"]["verification"]["result"] == "pass"
    assert output["structured_content"]["execution"]["verification_result"] == "pass"
    assert pose_output["structured_content"]["raw"]["pose"]["position"]["z"] == 0.36


@pytest.mark.asyncio
async def test_named_pose_execution_includes_live_smoke_verification_fields() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    output = json.loads(
        await adapter.call_tool(
            "moveit_plan_and_execute_named_pose",
            {"robot_name": "UR10", "named_pose": "home"},
        )
    )

    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["verification"]["result"] == "pass"
    assert output["structured_content"]["execution"]["verification_result"] == "pass"


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    output = json.loads(await adapter.call_tool("missing_tool", {}))

    assert output["is_error"] is True
    assert output["structured_content"]["ok"] is False
    assert "missing_tool" in output["structured_content"]["error"]


def test_function_tools_include_core_scenario_tools() -> None:
    adapter = SimulatedMoveItAdapter()

    tool_names = {tool["name"] for tool in adapter.function_tools()}

    assert {
        "moveit_get_current_pose",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_plan_and_execute_named_pose",
        "moveit_plan_and_execute_joint_goal",
        "moveit_list_available_robots",
    } <= tool_names
