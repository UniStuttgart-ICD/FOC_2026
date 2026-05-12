import pytest

from robot_control.job_board import RobotJobBoard, RobotJobStatus, SubmitRobotJob
from robot_control.job_worker import RobotJobWorker


class RecordingToolBridge:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append((tool_name, arguments))
        return self.result


class FailingToolBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, tool_name: str, arguments: dict[str, object]) -> str:
        self.calls.append((tool_name, arguments))
        raise RuntimeError("planning failed")


@pytest.mark.asyncio
async def test_worker_executes_exact_queued_call_and_marks_job_completed() -> None:
    board = RobotJobBoard()
    bridge = RecordingToolBridge('{"structured_content": {"ok": true}}')
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_and_execute_free_motion",
            {"robot_name": "UR10", "timeout_s": 10},
            "turn-1",
        )
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert stored.result == '{"structured_content": {"ok": true}}'
    assert bridge.calls[0][1] is stored.arguments
    assert bridge.calls == [
        (
            "moveit_plan_and_execute_free_motion",
            {"robot_name": "UR10", "timeout_s": 10},
        )
    ]


@pytest.mark.asyncio
async def test_worker_records_tool_failure_without_retrying_or_rewriting_args() -> None:
    board = RobotJobBoard()
    bridge = FailingToolBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert bridge.calls[0][1] is stored.arguments
    assert bridge.calls == [("moveit_open_gripper", {"robot_name": "UR10"})]
    assert stored.status is RobotJobStatus.FAILED
    assert stored.error == "planning failed"
