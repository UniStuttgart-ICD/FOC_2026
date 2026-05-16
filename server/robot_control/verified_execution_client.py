from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Protocol


class VerifiedExecutionClient(Protocol):
    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str: ...


class HttpVerifiedExecutionClient:
    def __init__(self, base_url: str, *, request_timeout_s: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_timeout_s = request_timeout_s

    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        try:
            response = await asyncio.to_thread(
                _post_json,
                f"{self._base_url}/execute",
                {
                    "robot_name": robot_name,
                    "plan_name": plan_name,
                    "timeout_s": timeout_s,
                },
                self._request_timeout_s,
            )
        except OSError:
            return _tool_output(
                ok=False,
                robot_name=robot_name,
                plan_name=plan_name,
                status="server_unavailable",
                trajectory_points=0,
                verification_result="fail",
                error="Verified execution server unavailable.",
                correction="Start the verified execution server, then retry execution.",
            )
        return _tool_output(
            ok=response.get("ok") is True,
            robot_name=str(response.get("robot_name") or robot_name),
            plan_name=str(response.get("plan_name") or plan_name),
            status=str(response.get("status") or "unknown"),
            trajectory_points=int(response.get("trajectory_points") or 0),
            verification_result=str(response.get("verification_result") or "unknown"),
            error=response.get("error") if isinstance(response.get("error"), str) else None,
            correction=(
                response.get("correction")
                if isinstance(response.get("correction"), str)
                else None
            ),
        )


def _post_json(url: str, payload: dict, timeout_s: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        data = exc.read().decode("utf-8")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise OSError("verified execution server returned non-object JSON")
    return parsed


def _tool_output(
    *,
    ok: bool,
    robot_name: str,
    plan_name: str,
    status: str,
    trajectory_points: int,
    verification_result: str,
    error: str | None = None,
    correction: str | None = None,
) -> str:
    structured: dict[str, Any] = {
        "ok": ok,
        "robot": robot_name,
        "tool": "execute_plan",
        "phase": "executed" if ok else "pre_execute",
        "status": status,
        "feedback": {
            "plan_name": plan_name,
            "trajectory_points": trajectory_points,
        },
        "verification": {"result": verification_result},
    }
    if error is not None:
        structured["error"] = error
    if correction is not None:
        structured["correction"] = correction
    return json.dumps(
        {
            "content": [_content_text(ok=ok, status=status, error=error)],
            "structured_content": structured,
            "is_error": not ok,
        },
        ensure_ascii=False,
    )


def _content_text(*, ok: bool, status: str, error: str | None) -> str:
    if ok:
        return "Verified execution completed."
    return error or f"Verified execution failed: {status}"
