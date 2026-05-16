from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any, Protocol

VerifiedExecutionOutput = dict[str, Any] | str
VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S = 30.0


class VerifiedExecutionClient(Protocol):
    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> VerifiedExecutionOutput: ...

    async def close_gripper(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> VerifiedExecutionOutput: ...


class HttpVerifiedExecutionClient:
    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float = 10.0,
        timeout_margin_s: float = 2.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._request_timeout_s = request_timeout_s
        self._timeout_margin_s = timeout_margin_s

    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        request_timeout_s = max(self._request_timeout_s, timeout_s + self._timeout_margin_s)
        try:
            response = await self._post_json(
                "/execute",
                {
                    "robot_name": robot_name,
                    "plan_name": plan_name,
                    "timeout_s": timeout_s,
                },
                timeout_s=request_timeout_s,
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
            target_joint_positions=_float_list(response.get("target_joint_positions")),
            final_joint_positions=_float_list(response.get("final_joint_positions")),
            max_joint_error=_float_or_none(response.get("max_joint_error")),
            joint_tolerance_rad=_float_or_none(response.get("joint_tolerance_rad")),
            state_sync_published=(
                response.get("state_sync_published")
                if isinstance(response.get("state_sync_published"), bool)
                else None
            ),
        )

    async def _post_json(
        self,
        path: str,
        payload: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            _post_json,
            f"{self._base_url}{path}",
            payload,
            timeout_s if timeout_s is not None else self._request_timeout_s,
        )

    async def close_gripper(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        request_timeout_s = max(self._request_timeout_s, timeout_s + self._timeout_margin_s)
        try:
            response = await self._post_json(
                "/gripper/close",
                {
                    "robot_name": robot_name,
                    "timeout_s": timeout_s,
                },
                timeout_s=request_timeout_s,
            )
        except OSError:
            return _gripper_tool_output(
                ok=False,
                robot_name=robot_name,
                action="close",
                status="server_unavailable",
                error="Verified execution server unavailable.",
                correction="Start the verified execution server, then retry gripper close.",
            )
        ok = bool(response.get("ok"))
        return _gripper_tool_output(
            ok=ok,
            robot_name=str(response.get("robot_name") or robot_name),
            action="close",
            status=str(response.get("status") or ("gripper_closed" if ok else "failed")),
            error=response.get("error") if isinstance(response.get("error"), str) else None,
            correction=(
                response.get("correction")
                if isinstance(response.get("correction"), str)
                else None
            ),
            command=response.get("command") if isinstance(response.get("command"), str) else None,
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
    target_joint_positions: list[float] | None = None,
    final_joint_positions: list[float] | None = None,
    max_joint_error: float | None = None,
    joint_tolerance_rad: float | None = None,
    state_sync_published: bool | None = None,
) -> str:
    metadata: dict[str, Any] = {}
    if target_joint_positions is not None:
        metadata["target_joint_positions"] = target_joint_positions
    if final_joint_positions is not None:
        metadata["final_joint_positions"] = final_joint_positions
    if max_joint_error is not None:
        metadata["max_joint_error"] = max_joint_error
    if joint_tolerance_rad is not None:
        metadata["joint_tolerance_rad"] = joint_tolerance_rad
    if state_sync_published is not None:
        metadata["state_sync_published"] = state_sync_published
    feedback = {
        "phase": "executed" if ok else "pre_execute",
        "status": status,
        "message": _content_text(ok=ok, status=status, error=error),
        "can_execute": False,
        **metadata,
    }
    structured: dict[str, Any] = {
        "ok": ok,
        "robot": robot_name,
        "tool": "moveit_execute_plan",
        "phase": "executed" if ok else "pre_execute",
        "status": status,
        "feedback": feedback,
        "verification": {"result": verification_result},
        "execution": {
            "ok": ok,
            "status": status,
            "verification_result": verification_result,
            **metadata,
        },
        "raw": {
            "plan_name": plan_name,
            "trajectory_points": trajectory_points,
        },
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


def _gripper_tool_output(
    *,
    ok: bool,
    robot_name: str,
    action: str,
    status: str,
    error: str | None = None,
    correction: str | None = None,
    command: str | None = None,
) -> str:
    message = (
        f"Verified gripper {action} completed."
        if ok
        else error or f"Verified gripper {action} failed: {status}"
    )
    structured: dict[str, Any] = {
        "ok": ok,
        "robot": robot_name,
        "tool": f"moveit_{action}_gripper",
        "phase": "gripper",
        "status": status,
        "feedback": {
            "phase": "gripper",
            "status": status,
            "message": message,
            "can_execute": False,
        },
        "verification": {"result": "pass" if ok else "fail"},
        "raw": {"command": command or f"gripper_{action}"},
    }
    if error is not None:
        structured["error"] = error
    if correction is not None:
        structured["correction"] = correction
    return json.dumps(
        {
            "content": [message],
            "structured_content": structured,
            "is_error": not ok,
        },
        ensure_ascii=False,
    )


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def verified_execution_output_to_json(output: VerifiedExecutionOutput) -> str:
    if isinstance(output, str):
        return output
    return json.dumps(output, ensure_ascii=False)
