from __future__ import annotations

import json

import pytest

import model_eval.simulated_moveit as simulated_moveit
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
async def test_cartesian_plan_then_execute_updates_pose_and_marks_verified() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    plan_output = json.loads(
        await adapter.call_tool(
            "moveit_plan_cartesian_motion",
            {
                "robot_name": "UR10",
                "waypoints": [
                    {"position": {"x": 0.4, "y": 0.1, "z": 0.35}},
                    {"position": {"x": 0.4, "y": 0.1, "z": 0.36}},
                ],
            },
        )
    )
    plan_name = plan_output["structured_content"]["raw"]["plan_name"]
    output = json.loads(
        await adapter.call_tool(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": plan_name},
        )
    )
    pose_output = json.loads(await adapter.call_tool("moveit_get_current_pose", {"robot_name": "UR10"}))

    assert plan_output["structured_content"]["feedback"]["can_execute"] is True
    assert output["structured_content"]["verification"]["result"] == "pass"
    assert output["structured_content"]["execution"]["verification_result"] == "pass"
    assert pose_output["structured_content"]["raw"]["pose"]["position"]["z"] == 0.36


@pytest.mark.asyncio
async def test_execute_unknown_plan_returns_structured_error() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    output = json.loads(
        await adapter.call_tool(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "missing"},
        )
    )

    assert output["is_error"] is True
    assert output["structured_content"]["ok"] is False
    assert output["structured_content"]["error"] == "Unknown simulated plan: missing"


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
        "moveit_plan_cartesian_motion",
        "moveit_execute_plan",
        "moveit_list_available_robots",
        "moveit_plan_compound_task",
        "moveit_execute_task_plan",
    } <= tool_names


@pytest.mark.asyncio
async def test_compound_pick_planner_returns_mtc_hold_task_solution_contract() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()

    output = json.loads(
        await adapter.call_tool(
            "moveit_plan_compound_task",
            {
                "robot_name": "UR10",
                "backend": "mtc",
                "requirements": {
                    "goal": "hold",
                    "object_name": "dynamic_5",
                    "lift_distance_m": 0.10,
                },
            },
        )
    )

    raw = output["structured_content"]["raw"]
    assert output["structured_content"]["ok"] is True
    assert raw["created_from_tool"] == "moveit_plan_compound_task"
    assert raw["backend"] == "mtc"
    assert raw["task_kind"] == "hold"
    assert raw["requirements"] == {
        "goal": "hold",
        "object_name": "dynamic_5",
        "lift_distance_m": 0.10,
    }
    assert raw["execution_contract"]["steps"]


@pytest.mark.asyncio
async def test_execute_task_plan_direct_adapter_call_requires_agent_control_intercept() -> None:
    adapter = SimulatedMoveItAdapter()
    await adapter.connect()
    await adapter.call_tool(
        "moveit_plan_compound_task",
        {
            "robot_name": "UR10",
            "backend": "mtc",
            "requirements": {
                "goal": "hold",
                "object_name": "dynamic_5",
                "lift_distance_m": 0.10,
            },
        },
    )

    output = json.loads(
        await adapter.call_tool(
            "moveit_execute_task_plan",
            {"robot_name": "UR10", "task_solution_id": "compound_hold_dynamic_5_001"},
        )
    )

    assert output["ok"] is False
    assert output["code"] == "agent_control_execution_required"
    assert "Agent Control" in output["correction"]


def test_task_level_pick_replay_records_full_loop_artifact() -> None:
    scenario = simulated_moveit.task_level_pick_replay_scenario()

    assert scenario["utterance"] == "pick up dynamic_5"
    assert scenario["mocked_robot_state"]["robot_name"] == "UR10"
    assert scenario["mocked_scene_objects"][0]["name"] == "dynamic_5"
    assert scenario["expected_tool_sequence"] == [
        "moveit_list_scene_objects",
        "moveit_get_object_context",
        "moveit_get_current_pose",
        "moveit_plan_compound_task",
        "approval_recorded",
        "moveit_execute_task_plan",
        "moveit_verify_attached_object",
    ]
    assert [output["tool_name"] for output in scenario["typed_tool_outputs"]] == [
        "moveit_list_scene_objects",
        "moveit_get_object_context",
        "moveit_get_current_pose",
        "moveit_plan_compound_task",
        "moveit_execute_task_plan",
        "moveit_verify_attached_object",
    ]
    planner_raw = scenario["typed_tool_outputs"][3]["structured_content"]["raw"]
    assert planner_raw["backend"] == "mtc"
    assert planner_raw["task_kind"] == "hold"
    assert planner_raw["requirements"] == {
        "goal": "hold",
        "object_name": "dynamic_5",
        "lift_distance_m": 0.10,
    }
    assert planner_raw["execution_contract"]["steps"]
    assert scenario["typed_tool_outputs"][4]["source"] == "agent_control_intercept"
    assert scenario["approval_payload"]["source_tool"] == "moveit_plan_compound_task"
    assert scenario["policy_decisions"][-1] == {
        "tool_name": "moveit_execute_task_plan",
        "decision": "allow",
    }
    assert scenario["validation_results"][-1] == {
        "tool_name": "moveit_execute_task_plan",
        "ok": True,
    }
    assert scenario["approval_payload"]["task_solution_id"] == "compound_hold_dynamic_5_001"
    assert scenario["approval_payload"]["scene_snapshot_id"] == "scene_20260515_001"
    assert scenario["execution_result"]["ok"] is True
    assert scenario["execution_result"]["tool"] == "moveit_execute_task_plan"
    assert scenario["execution_result"]["source"] == "agent_control_intercept"
    assert scenario["adapter_direct_execution_equivalent"] is False
    assert scenario["execution_result"]["task_solution_id"] == "compound_hold_dynamic_5_001"
    assert scenario["verification_result"]["attached"] is True
    assert scenario["terminal_job_event"] == {
        "event_type": "robot_job_completed",
        "tool_name": "moveit_verify_attached_object",
        "ok": True,
    }


def test_negative_pick_replay_scenarios_cover_no_execution_and_no_success_claims() -> None:
    scenarios = simulated_moveit.negative_pick_replay_scenarios()

    partial = scenarios["partial_legacy_pick"]
    assert partial["utterance"] == "use the legacy pick planner for dynamic_5"
    assert partial["tool_name"] == "moveit_plan_pick_task"
    assert partial["tool_output"]["structured_content"]["failed_stage"] == "local_cartesian_pick"
    assert partial["execution_attempted"] is False

    missing_approval = scenarios["missing_approval"]
    assert missing_approval["tool_name"] == "moveit_execute_task_plan"
    assert missing_approval["policy_decision"]["decision"] == "block"
    assert missing_approval["policy_decision"]["reason"] == "missing_approval"
    assert missing_approval["execution_attempted"] is False

    stale_scene = scenarios["stale_scene_snapshot_id"]
    assert stale_scene["tool_name"] == "moveit_execute_task_plan"
    assert stale_scene["policy_decision"]["decision"] == "block"
    assert stale_scene["policy_decision"]["reason"] == "stale_scene_snapshot_id"
    assert stale_scene["execution_attempted"] is False

    verification_failure = scenarios["attachment_verification_failure"]
    assert verification_failure["verification_result"]["ok"] is False
    assert verification_failure["success_claim_allowed"] is False
