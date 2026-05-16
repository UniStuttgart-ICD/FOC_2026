import pytest

from process_trace import MemoryTraceWriter, ProcessTracer
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


@pytest.mark.asyncio
async def test_job_board_renders_recent_job_summary_with_salient_args() -> None:
    now = 100.0
    board = RobotJobBoard(time_fn=lambda: now)
    completed = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 10.0},
            "turn-1",
        )
    )
    running = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-2")
    )
    failed = await board.submit(
        SubmitRobotJob("moveit_plan_pick", {"object_name": "dynamic_5"}, "turn-3")
    )
    cancelled = await board.submit(
        SubmitRobotJob(
            "moveit_plan_cartesian_motion",
            {"robot_name": "UR10", "task_solution_id": "task-1"},
            "turn-4",
        )
    )

    await board.claim_next()
    await board.complete(completed.job_id, '{"structured_content": {"ok": true}}')
    await board.claim_next()
    await board.claim_next()
    await board.fail(failed.job_id, "planner failed")
    await board.cancel_queued_for_turn(
        requested_by_turn_id="turn-4",
        tool_names=frozenset({"moveit_plan_cartesian_motion"}),
        reason="newer plan selected",
    )
    completed_sequence = [
        event.sequence
        for event in board.events_since(0)
        if event.job_id == completed.job_id and event.event_type is RobotJobEventType.COMPLETED
    ][0]

    summary = board.render_instruction_block(
        max_age_s=60.0,
        context_recorded_sequences={completed_sequence},
    )

    assert summary is not None
    assert "Robot Job Blackboard:" in summary
    assert "moveit_execute_plan: completed" in summary
    assert "robot_name=UR10" in summary
    assert "plan_name=plan-1" in summary
    assert "result recorded in Robot Context" in summary
    assert "moveit_open_gripper: running" in summary
    assert "moveit_plan_pick: failed" in summary
    assert "object_name=dynamic_5" in summary
    assert "error=planner failed" in summary
    assert "moveit_plan_cartesian_motion: cancelled" in summary
    assert "task_solution_id=task-1" in summary


@pytest.mark.asyncio
async def test_job_board_emits_process_trace_lifecycle_events() -> None:
    writer = MemoryTraceWriter()
    board = RobotJobBoard(tracer=ProcessTracer(writer))

    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1"},
            "turn-1",
        )
    )
    await board.claim_next()
    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')

    records = [record for record in writer.records if record["name"].startswith("robot.job.")]

    assert [record["name"] for record in records] == [
        "robot.job.queued",
        "robot.job.started",
        "robot.job.completed",
    ]
    assert records[-1]["module"] == "robot_control"
    assert records[-1]["attributes"] == {
        "job.id": job.job_id,
        "job.tool_name": "moveit_execute_plan",
        "job.status": "completed",
        "job.requested_by_turn_id": "turn-1",
        "job.arg.robot_name": "UR10",
        "job.arg.plan_name": "plan-1",
        "job.result_present": True,
    }
