import json

import pytest

from agent_control.robot_job_submission import QUEUEABLE_ROBOT_ACTION_TOOLS, RobotJobSubmitter
from robot_control.job_board import RobotJobBoard, RobotJobStatus


@pytest.mark.asyncio
async def test_submitter_queues_exact_tool_call_and_returns_queued_feedback() -> None:
    board = RobotJobBoard()
    submitter = RobotJobSubmitter(board)
    arguments = {"robot_name": "UR10", "timeout_s": 10}

    output = await submitter.submit_tool(
        "moveit_plan_and_execute_free_motion",
        arguments,
        requested_by_turn_id="turn-1",
    )

    job = await board.claim_next()
    assert job is not None
    assert job.status is RobotJobStatus.RUNNING
    assert job.tool_name == "moveit_plan_and_execute_free_motion"
    assert job.arguments == arguments
    assert job.requested_by_turn_id == "turn-1"
    payload = json.loads(output)
    assert payload["structured_content"] == {
        "ok": True,
        "status": "queued",
        "job_id": job.job_id,
        "tool_name": "moveit_plan_and_execute_free_motion",
    }
    assert payload["is_error"] is False
    assert payload["content"] == [
        f"Queued robot job {job.job_id} for moveit_plan_and_execute_free_motion. "
        "The robot worker will report completion or failure."
    ]


def test_queueable_action_tools_do_not_include_observation() -> None:
    assert "moveit_get_current_pose" not in QUEUEABLE_ROBOT_ACTION_TOOLS
    assert "moveit_plan_and_execute_free_motion" in QUEUEABLE_ROBOT_ACTION_TOOLS
    assert "moveit_open_gripper" in QUEUEABLE_ROBOT_ACTION_TOOLS
