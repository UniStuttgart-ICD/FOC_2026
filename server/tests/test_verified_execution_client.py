from __future__ import annotations

import json

import pytest

from robot_control.verified_execution_client import HttpVerifiedExecutionClient


class RecordingVerifiedExecutionClient(HttpVerifiedExecutionClient):
    def __init__(self, response: dict[str, object]) -> None:
        super().__init__("http://verified.test", timeout_margin_s=2.0)
        self.response = response
        self.posts: list[tuple[str, dict[str, object], float | None]] = []

    async def _post_json(
        self,
        path: str,
        payload: dict[str, object],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        self.posts.append((path, payload, timeout_s))
        return self.response


@pytest.mark.asyncio
async def test_verified_execution_client_converts_http_response_to_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object], float]] = []

    def fake_post_json(url: str, payload: dict[str, object], timeout_s: float) -> dict:
        calls.append((url, payload, timeout_s))
        return {
            "ok": True,
            "robot_name": "UR10",
            "plan_name": "plan-1",
            "status": "executed",
            "trajectory_points": 2,
            "verification_result": "pass",
            "error": None,
            "correction": None,
        }

    monkeypatch.setattr(
        "robot_control.verified_execution_client._post_json",
        fake_post_json,
    )
    client = HttpVerifiedExecutionClient("http://127.0.0.1:8770", request_timeout_s=3.0)

    output = await client.execute_plan(
        robot_name="UR10",
        plan_name="plan-1",
        timeout_s=5.0,
    )

    assert calls == [
        (
            "http://127.0.0.1:8770/execute",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
            7.0,
        )
    ]
    payload = json.loads(output)
    assert payload["structured_content"] == {
        "ok": True,
        "robot": "UR10",
        "tool": "moveit_execute_plan",
        "phase": "executed",
        "status": "executed",
        "feedback": {
            "phase": "executed",
            "status": "executed",
            "message": "Verified execution completed.",
            "can_execute": False,
        },
        "verification": {"result": "pass"},
        "execution": {
            "ok": True,
            "status": "executed",
            "verification_result": "pass",
        },
        "raw": {
            "plan_name": "plan-1",
            "trajectory_points": 2,
        },
    }
    assert payload["is_error"] is False


@pytest.mark.asyncio
async def test_verified_execution_client_reports_transport_errors_as_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post_json(url: str, payload: dict[str, object], timeout_s: float) -> dict:
        raise OSError("connection refused")

    monkeypatch.setattr(
        "robot_control.verified_execution_client._post_json",
        fake_post_json,
    )
    client = HttpVerifiedExecutionClient("http://127.0.0.1:8770")

    output = await client.execute_plan(
        robot_name="UR10",
        plan_name="plan-1",
        timeout_s=5.0,
    )

    payload = json.loads(output)
    assert payload["structured_content"]["ok"] is False
    assert payload["structured_content"]["error"] == "Verified execution server unavailable."
    assert payload["structured_content"]["correction"] == (
        "Start the verified execution server, then retry execution."
    )
    assert payload["is_error"] is True


@pytest.mark.asyncio
async def test_execute_plan_uses_http_timeout_longer_than_robot_execution_timeout() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "plan_name": "plan-1",
            "status": "executed",
            "trajectory_points": 2,
            "verification_result": "pass",
        }
    )

    await client.execute_plan(robot_name="UR10", plan_name="plan-1", timeout_s=15.0)

    assert client.posts == [
        (
            "/execute",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 15.0},
            17.0,
        )
    ]


@pytest.mark.asyncio
async def test_execute_plan_surfaces_execution_sync_metadata_in_feedback() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "plan_name": "plan-1",
            "status": "executed",
            "trajectory_points": 2,
            "verification_result": "pass",
            "target_joint_positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0],
            "final_joint_positions": [0.101, -1.471, 1.481, 0.101, 0.001, -0.001],
            "max_joint_error": 0.001,
            "joint_tolerance_rad": 0.01,
            "state_sync_published": True,
        }
    )

    result = await client.execute_plan(robot_name="UR10", plan_name="plan-1", timeout_s=15.0)

    structured_content = json.loads(result)["structured_content"]
    assert structured_content["ok"] is True
    assert structured_content["verification"] == {"result": "pass"}
    assert structured_content["feedback"] == {
        "phase": "executed",
        "status": "executed",
        "message": "Verified execution completed.",
        "can_execute": False,
        "target_joint_positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0],
        "final_joint_positions": [0.101, -1.471, 1.481, 0.101, 0.001, -0.001],
        "max_joint_error": 0.001,
        "joint_tolerance_rad": 0.01,
        "state_sync_published": True,
    }
    assert structured_content["execution"]["state_sync_published"] is True


@pytest.mark.asyncio
async def test_close_gripper_posts_to_verified_execution_server() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "command": "gripper_close",
            "status": "gripper_closed",
        }
    )

    result = await client.close_gripper(robot_name="UR10", timeout_s=8.0)

    assert client.posts == [
        (
            "/gripper/close",
            {"robot_name": "UR10", "timeout_s": 8.0},
            10.0,
        )
    ]
    structured_content = json.loads(result)["structured_content"]
    assert structured_content == {
        "ok": True,
        "robot": "UR10",
        "tool": "moveit_close_gripper",
        "phase": "gripper",
        "status": "gripper_closed",
        "feedback": {
            "phase": "gripper",
            "status": "gripper_closed",
            "message": "Verified gripper close completed.",
            "can_execute": False,
        },
        "verification": {"result": "pass"},
        "raw": {"command": "gripper_close"},
    }
