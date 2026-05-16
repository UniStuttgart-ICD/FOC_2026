import pytest

from robot_control.job_board import (
    RobotJobBoard,
    RobotJobEventType,
    RobotJobStatus,
    SubmitRobotJob,
)


@pytest.mark.asyncio
async def test_submit_job_records_queued_event_and_returns_job_id() -> None:
    board = RobotJobBoard()

    job = await board.submit(
        SubmitRobotJob(
            tool_name="moveit_plan_free_motion",
            arguments={"robot_name": "UR10", "timeout_s": 10},
            requested_by_turn_id="turn-1",
        )
    )

    assert job.job_id
    assert job.status is RobotJobStatus.QUEUED
    assert job.tool_name == "moveit_plan_free_motion"
    assert job.arguments == {"robot_name": "UR10", "timeout_s": 10}
    events = board.events_since(0)
    assert [(event.event_type, event.job_id) for event in events] == [
        (RobotJobEventType.QUEUED, job.job_id)
    ]


@pytest.mark.asyncio
async def test_worker_claims_jobs_fifo() -> None:
    board = RobotJobBoard()
    first = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    second = await board.submit(
        SubmitRobotJob("moveit_close_gripper", {"robot_name": "UR10"}, "turn-2")
    )

    claimed_first = await board.claim_next()
    claimed_second = await board.claim_next()

    assert claimed_first is not None
    assert claimed_second is not None
    assert claimed_first.job_id == first.job_id
    assert claimed_second.job_id == second.job_id
    assert claimed_first.status is RobotJobStatus.RUNNING
    assert claimed_second.status is RobotJobStatus.RUNNING


@pytest.mark.asyncio
async def test_complete_and_fail_record_terminal_events() -> None:
    board = RobotJobBoard()
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    claimed = await board.claim_next()
    assert claimed is not None

    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')
    await board.fail(job.job_id, "ignored after completion")

    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert stored.result == '{"structured_content": {"ok": true}}'
    assert [event.event_type for event in board.events_since(0)] == [
        RobotJobEventType.QUEUED,
        RobotJobEventType.STARTED,
        RobotJobEventType.COMPLETED,
    ]
