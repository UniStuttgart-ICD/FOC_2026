import json

import pytest
from fastapi.testclient import TestClient

from robot_control.job_board import RobotJobBoard, RobotJobStatus, SubmitRobotJob
from voice_runtime.agent_turn import AgentTurnProcessor


@pytest.mark.asyncio
async def test_monitor_api_returns_live_robot_job_board_snapshot() -> None:
    from robot_control.job_monitor import create_app

    now = 500.0
    board = RobotJobBoard(time_fn=lambda: now)
    completed = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 10.0},
            "turn-1",
            user_text="execute the plan",
        )
    )
    running = await board.submit(
        SubmitRobotJob("moveit_plan_pick", {"object_name": "dynamic_5"}, "turn-2")
    )
    await board.claim_next()
    await board.complete(completed.job_id, json.dumps({"structured_content": {"ok": True}}))
    await board.claim_next()

    client = TestClient(create_app(board))
    response = client.get("/api/robot-jobs")

    assert response.status_code == 200
    body = response.json()
    assert body["counts"]["completed"] == 1
    assert body["counts"]["running"] == 1
    assert body["counts"]["failed"] == 0
    assert body["queue"] == []
    jobs = {job["job_id"]: job for job in body["jobs"]}
    assert jobs[completed.job_id]["status"] == RobotJobStatus.COMPLETED.value
    assert jobs[completed.job_id]["salient_arguments"] == {
        "robot_name": "UR10",
        "plan_name": "plan-1",
    }
    assert jobs[completed.job_id]["result_present"] is True
    assert jobs[running.job_id]["status"] == RobotJobStatus.RUNNING.value
    assert body["events"][-1]["event_type"] == "robot_job_started"


def test_monitor_index_page_serves_dashboard_shell() -> None:
    from robot_control.job_monitor import create_app

    client = TestClient(create_app(RobotJobBoard()))

    response = client.get("/")

    assert response.status_code == 200
    assert "Robot Job Blackboard" in response.text
    assert "jobGrid" in response.text
    assert "eventStream" in response.text
    assert "/api/robot-jobs" in response.text


class _BackendWithBoard:
    robot_job_board = "board-token"

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def run_turn(self, turn):
        if False:
            yield ""


def test_agent_turn_processor_exposes_backend_robot_job_board() -> None:
    processor = AgentTurnProcessor(backend=_BackendWithBoard())

    assert processor.robot_job_board == "board-token"
