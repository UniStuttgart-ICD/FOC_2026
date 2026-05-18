from __future__ import annotations

import json

import pytest

from robot_control.verified_execution_client import HttpVerifiedExecutionClient


class RecordingVerifiedExecutionClient(HttpVerifiedExecutionClient):
    def __init__(
        self,
        response: dict[str, object],
        *,
        get_response: dict[str, object] | None = None,
        get_error: OSError | None = None,
    ) -> None:
        super().__init__("http://verified.test", timeout_margin_s=2.0)
        self.response = response
        self.get_response = get_response if get_response is not None else response
        self.get_error = get_error
        self.posts: list[tuple[str, dict[str, object], float | None]] = []
        self.gets: list[tuple[str, float | None]] = []

    async def _post_json(
        self,
        path: str,
        payload: dict[str, object],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        self.posts.append((path, payload, timeout_s))
        return self.response

    async def _get_json(
        self,
        path: str,
        *,
        timeout_s: float | None = None,
    ) -> dict[str, object]:
        self.gets.append((path, timeout_s))
        if self.get_error is not None:
            raise self.get_error
        return self.get_response


@pytest.mark.asyncio
async def test_get_readiness_reports_healthy_robot_from_health_endpoint() -> None:
    client = RecordingVerifiedExecutionClient(
        {},
        get_response={
            "ok": True,
            "ros_connected": True,
            "cached_plans": 2,
            "robot": {
                "robot_name": "UR10",
                "robot_connected": True,
                "gripper_connected": True,
            },
        },
    )

    readiness = await client.get_readiness(timeout_s=4.0)

    assert client.gets == [("/health", 4.0)]
    assert readiness == {
        "server_available": True,
        "robot_connected": True,
        "gripper_connected": True,
        "error": None,
    }


@pytest.mark.asyncio
async def test_get_readiness_reports_disconnected_robot_without_raising() -> None:
    client = RecordingVerifiedExecutionClient(
        {},
        get_response={
            "ok": True,
            "ros_connected": True,
            "cached_plans": 0,
            "robot": {
                "robot_name": "UR10",
                "robot_connected": False,
                "gripper_connected": True,
            },
        },
    )

    readiness = await client.get_readiness(timeout_s=4.0)

    assert readiness == {
        "server_available": True,
        "robot_connected": False,
        "gripper_connected": True,
        "error": None,
    }


@pytest.mark.asyncio
async def test_get_readiness_treats_missing_robot_payload_as_unknown_physical_state() -> None:
    client = RecordingVerifiedExecutionClient(
        {},
        get_response={
            "ok": True,
            "ros_connected": True,
            "cached_plans": 0,
        },
    )

    readiness = await client.get_readiness(timeout_s=4.0)

    assert readiness == {
        "server_available": True,
        "robot_connected": None,
        "gripper_connected": None,
        "error": None,
    }


@pytest.mark.asyncio
async def test_get_readiness_reports_transport_errors_as_unavailable() -> None:
    client = RecordingVerifiedExecutionClient(
        {},
        get_error=OSError("connection refused"),
    )

    readiness = await client.get_readiness(timeout_s=4.0)

    assert client.gets == [("/health", 4.0)]
    assert readiness == {
        "server_available": False,
        "robot_connected": False,
        "gripper_connected": False,
        "error": "Verified execution server unavailable.",
    }


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


@pytest.mark.asyncio
async def test_open_gripper_posts_to_verified_execution_server() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "command": "gripper_open",
            "status": "gripper_open",
        }
    )

    result = await client.open_gripper(robot_name="UR10", timeout_s=8.0)

    assert client.posts == [
        (
            "/gripper/open",
            {"robot_name": "UR10", "timeout_s": 8.0},
            10.0,
        )
    ]
    structured_content = json.loads(result)["structured_content"]
    assert structured_content == {
        "ok": True,
        "robot": "UR10",
        "tool": "moveit_open_gripper",
        "phase": "gripper",
        "status": "gripper_open",
        "feedback": {
            "phase": "gripper",
            "status": "gripper_open",
            "message": "Verified gripper open completed.",
            "can_execute": False,
        },
        "verification": {"result": "pass"},
        "raw": {"command": "gripper_open"},
    }


@pytest.mark.asyncio
async def test_go_home_posts_to_verified_execution_server_and_surfaces_sync_metadata() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "command": "home",
            "status": "homed",
            "final_joint_positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
            "state_sync_published": True,
        }
    )

    result = await client.go_home(robot_name="UR10", timeout_s=12.0)

    assert client.posts == [
        (
            "/home",
            {"robot_name": "UR10", "timeout_s": 12.0},
            14.0,
        )
    ]
    structured_content = json.loads(result)["structured_content"]
    assert structured_content["ok"] is True
    assert structured_content["tool"] == "moveit_go_home"
    assert structured_content["status"] == "homed"
    assert structured_content["verification"] == {"result": "pass"}
    assert structured_content["feedback"]["final_joint_positions"] == [
        0.0,
        -1.57,
        1.57,
        0.0,
        0.0,
        0.0,
    ]
    assert structured_content["feedback"]["state_sync_published"] is True


@pytest.mark.asyncio
async def test_sync_real_robot_state_posts_to_verified_execution_server() -> None:
    client = RecordingVerifiedExecutionClient(
        {
            "ok": True,
            "robot_name": "UR10",
            "command": "sync_state",
            "status": "state_synced",
            "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
            "actual_tcp_pose": [0.4, -0.2, 0.3, 0.0, 3.14, 0.0],
            "state_sync_published": True,
            "actual_gripper_position": 128,
            "actual_gripper_joint_position": 128.0 / 255.0 * 0.8,
            "gripper_joint_state_published": True,
            "gripper_joint_name": "finger_joint",
            "gripper_joint_state_topic": "/UR10/gripper_joint_states",
            "gripper_open_threshold_position": 10,
            "gripper_considered_open": True,
            "attached_object_release_checked": True,
            "attached_objects_before_release": ["held_part"],
            "attached_objects_released": ["held_part"],
            "attached_object_release_published": True,
            "attached_object_release_verified": True,
            "attached_object_release_topic_or_service": "/UR10/apply_planning_scene",
        }
    )

    result = await client.sync_real_robot_state(robot_name="UR10", timeout_s=6.0)

    assert client.posts == [
        (
            "/sync_state",
            {"robot_name": "UR10", "timeout_s": 6.0},
            8.0,
        )
    ]
    structured_content = json.loads(result)["structured_content"]
    assert structured_content["ok"] is True
    assert structured_content["tool"] == "moveit_sync_real_robot_state"
    assert structured_content["status"] == "state_synced"
    assert structured_content["verification"] == {"result": "pass"}
    assert structured_content["feedback"]["actual_joint_positions"] == [
        0.2,
        -1.4,
        1.3,
        0.1,
        0.0,
        -0.2,
    ]
    assert structured_content["feedback"]["state_sync_published"] is True
    assert structured_content["feedback"]["actual_gripper_position"] == 128
    assert structured_content["feedback"]["actual_gripper_joint_position"] == pytest.approx(
        128.0 / 255.0 * 0.8
    )
    assert structured_content["feedback"]["gripper_joint_state_published"] is True
    assert structured_content["feedback"]["gripper_joint_name"] == "finger_joint"
    assert structured_content["feedback"]["gripper_joint_state_topic"] == "/UR10/gripper_joint_states"
    assert structured_content["feedback"]["gripper_open_threshold_position"] == 10
    assert structured_content["feedback"]["gripper_considered_open"] is True
    assert structured_content["feedback"]["attached_object_release_checked"] is True
    assert structured_content["feedback"]["attached_objects_before_release"] == ["held_part"]
    assert structured_content["feedback"]["attached_objects_released"] == ["held_part"]
    assert structured_content["feedback"]["attached_object_release_published"] is True
    assert structured_content["feedback"]["attached_object_release_verified"] is True
    assert (
        structured_content["feedback"]["attached_object_release_topic_or_service"]
        == "/UR10/apply_planning_scene"
    )
