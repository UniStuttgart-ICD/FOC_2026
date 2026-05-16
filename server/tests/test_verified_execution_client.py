from __future__ import annotations

import json

import pytest

from robot_control.verified_execution_client import HttpVerifiedExecutionClient


@pytest.mark.asyncio
async def test_verified_execution_client_converts_http_response_to_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict, float]] = []

    def fake_post_json(url: str, payload: dict, timeout_s: float) -> dict:
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
            3.0,
        )
    ]
    payload = json.loads(output)
    assert payload["structured_content"] == {
        "ok": True,
        "robot": "UR10",
        "tool": "execute_plan",
        "phase": "executed",
        "status": "executed",
        "feedback": {
            "plan_name": "plan-1",
            "trajectory_points": 2,
        },
        "verification": {"result": "pass"},
    }
    assert payload["is_error"] is False


@pytest.mark.asyncio
async def test_verified_execution_client_reports_transport_errors_as_tool_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_post_json(url: str, payload: dict, timeout_s: float) -> dict:
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
    assert payload["structured_content"]["correction"] == "Start the verified execution server, then retry execution."
    assert payload["is_error"] is True
