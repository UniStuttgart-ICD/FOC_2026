from __future__ import annotations

import asyncio
import json
from typing import Any, Protocol

from robot_control.call_validation import RobotCallValidationError, validate_robot_tool_call
from robot_control.execution_intent import should_auto_execute_successful_plan
from robot_control.job_board import RobotJob, RobotJobBoard, SubmitRobotJob
from robot_control.manipulation_plans import parse_executable_plan_result
from robot_control.verified_execution_client import (
    VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S,
    VerifiedExecutionOutput,
    verified_execution_output_to_json,
)

PLAN_JOB_TOOLS = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_pick",
        "moveit_plan_place",
    }
)
AFTER_SUCCESS_JOB_TOOLS = PLAN_JOB_TOOLS | frozenset({"moveit_open_gripper"})


class RobotToolBridgeLike(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class VerifiedExecutionClientLike(Protocol):
    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> VerifiedExecutionOutput: ...


class RobotJobWorker:
    def __init__(
        self,
        *,
        board: RobotJobBoard,
        tool_bridge: RobotToolBridgeLike,
        verified_execution_client: VerifiedExecutionClientLike | None = None,
    ) -> None:
        self._board = board
        self._tool_bridge = tool_bridge
        self._verified_execution_client = verified_execution_client
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is None:
            return
        await self._task
        self._task = None

    async def run_once(self) -> bool:
        job = await self._board.claim_next()
        if job is None:
            return False
        try:
            result = await self._execute_job(job)
        except Exception as exc:
            await self._board.fail(job.job_id, str(exc))
        else:
            error = _tool_result_error(result)
            if error is not None:
                await self._board.fail(job.job_id, error, result=result)
            else:
                await self._board.complete(job.job_id, result)
                await self._queue_after_success_continuation(job)
                await self._queue_execution_for_successful_plan(job, result)
        return True

    async def _execute_job(self, job: RobotJob) -> str:
        if (
            job.tool_name == "moveit_execute_plan"
            and self._verified_execution_client is not None
            and not job.execute_via_mcp
        ):
            return verified_execution_output_to_json(
                await self._verified_execution_client.execute_plan(
                    robot_name=str(job.arguments.get("robot_name") or "UR10"),
                    plan_name=str(job.arguments.get("plan_name") or ""),
                    timeout_s=float(
                        job.arguments.get("timeout_s")
                        or VERIFIED_EXECUTION_DEFAULT_TIMEOUT_S
                    ),
                )
            )
        return await self._tool_bridge.call_tool(job.tool_name, job.arguments)

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            ran = await self.run_once()
            if not ran:
                await asyncio.sleep(0.05)

    async def _queue_execution_for_successful_plan(self, job: RobotJob, result: str) -> None:
        if job.tool_name not in PLAN_JOB_TOOLS:
            return
        if not should_auto_execute_successful_plan(job.user_text):
            return
        plan = parse_executable_plan_result(job.tool_name, result)
        if plan is None:
            return

        await self._board.cancel_queued_for_turn(
            requested_by_turn_id=job.requested_by_turn_id,
            tool_names=PLAN_JOB_TOOLS,
            reason="Skipped after the first successful plan was selected for execution.",
        )
        await self._board.submit(
            SubmitRobotJob(
                "moveit_execute_plan",
                {
                    "robot_name": plan.robot_name or "UR10",
                    "plan_name": plan.plan_name,
                    "timeout_s": 10.0,
                },
                job.requested_by_turn_id,
                user_text=job.user_text,
                after_success_tool=plan.after_success.tool if plan.after_success is not None else None,
                after_success_arguments=(
                    plan.after_success.arguments if plan.after_success is not None else None
                ),
                execute_via_mcp=_requires_mcp_execution(plan.raw),
            ),
            front=True,
        )

    async def _queue_after_success_continuation(self, job: RobotJob) -> None:
        if job.tool_name != "moveit_execute_plan":
            return
        if job.after_success_tool not in AFTER_SUCCESS_JOB_TOOLS:
            return
        if not isinstance(job.after_success_arguments, dict):
            return
        try:
            validate_robot_tool_call(job.after_success_tool, job.after_success_arguments)
        except RobotCallValidationError:
            return
        await self._board.submit(
            SubmitRobotJob(
                job.after_success_tool,
                dict(job.after_success_arguments),
                job.requested_by_turn_id,
                user_text=job.user_text,
            ),
            front=True,
        )


def _tool_result_error(result: str) -> str | None:
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    structured = payload.get("structured_content")
    if isinstance(structured, dict):
        if _structured_result_failed(structured):
            return _structured_error_text(structured)

    if payload.get("ok") is False or payload.get("is_error") is True:
        return _structured_error_text(payload)
    return None


def _structured_result_failed(result: dict[str, Any]) -> bool:
    if result.get("ok") is False:
        return True
    verification = result.get("verification")
    if isinstance(verification, dict) and verification.get("result") == "fail":
        return True
    execution = result.get("execution")
    return isinstance(execution, dict) and execution.get("ok") is False


def _structured_error_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    feedback = result.get("feedback")
    for value in (
        result.get("error"),
        feedback.get("message") if isinstance(feedback, dict) else None,
        result.get("correction"),
        feedback.get("correction") if isinstance(feedback, dict) else None,
    ):
        if isinstance(value, str) and value.strip() and value.strip() not in parts:
            parts.append(value.strip())
    return " ".join(parts) or "Robot action failed."


def _requires_mcp_execution(raw: dict[str, Any]) -> bool:
    workflow_kind = raw.get("workflow_kind")
    if workflow_kind in {"pick", "place"}:
        return True
    next_action = raw.get("next_action")
    if isinstance(next_action, dict) and isinstance(next_action.get("after_success"), dict):
        return True
    return False
