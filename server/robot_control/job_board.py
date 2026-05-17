from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from process_trace import NoopProcessTracer, ProcessTracer

ProcessTracerLike = ProcessTracer | NoopProcessTracer
SALIENT_JOB_ARGUMENTS = ("robot_name", "plan_name", "task_solution_id", "object_name")


class RobotJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RobotJobEventType(str, Enum):
    QUEUED = "robot_job_queued"
    STARTED = "robot_job_started"
    COMPLETED = "robot_job_completed"
    FAILED = "robot_job_failed"
    CANCELLED = "robot_job_cancelled"


@dataclass(frozen=True)
class SubmitRobotJob:
    tool_name: str
    arguments: dict[str, Any]
    requested_by_turn_id: str | None
    user_text: str | None = None
    after_success_tool: str | None = None
    after_success_arguments: dict[str, Any] | None = None
    execute_via_mcp: bool = False


@dataclass(frozen=True)
class RobotJob:
    job_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_by_turn_id: str | None
    user_text: str | None
    status: RobotJobStatus
    created_at: float
    updated_at: float
    result: str | None = None
    error: str | None = None
    after_success_tool: str | None = None
    after_success_arguments: dict[str, Any] | None = None
    execute_via_mcp: bool = False


@dataclass(frozen=True)
class RobotJobEvent:
    sequence: int
    event_type: RobotJobEventType
    job_id: str
    tool_name: str
    status: RobotJobStatus
    created_at: float
    payload: dict[str, Any]


class RobotJobBoard:
    def __init__(
        self,
        *,
        tracer: ProcessTracerLike | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self._condition = asyncio.Condition()
        self._jobs: dict[str, RobotJob] = {}
        self._queue: list[str] = []
        self._events: list[RobotJobEvent] = []
        self._next_sequence = 1
        self._tracer = tracer or NoopProcessTracer()
        self._time_fn = time_fn

    async def submit(self, job: SubmitRobotJob, *, front: bool = False) -> RobotJob:
        async with self._condition:
            now = self._time_fn()
            stored = RobotJob(
                job_id=uuid.uuid4().hex,
                tool_name=job.tool_name,
                arguments=dict(job.arguments),
                requested_by_turn_id=job.requested_by_turn_id,
                user_text=job.user_text,
                status=RobotJobStatus.QUEUED,
                created_at=now,
                updated_at=now,
                after_success_tool=job.after_success_tool,
                after_success_arguments=(
                    dict(job.after_success_arguments)
                    if isinstance(job.after_success_arguments, dict)
                    else None
                ),
                execute_via_mcp=job.execute_via_mcp,
            )
            self._jobs[stored.job_id] = stored
            if front:
                self._queue.insert(0, stored.job_id)
            else:
                self._queue.append(stored.job_id)
            self._record_locked(RobotJobEventType.QUEUED, stored, {})
            self._condition.notify_all()
            return stored

    async def claim_next(self) -> RobotJob | None:
        async with self._condition:
            if not self._queue:
                return None
            job_id = self._queue.pop(0)
            job = self._jobs[job_id]
            running = replace(job, status=RobotJobStatus.RUNNING, updated_at=self._time_fn())
            self._jobs[job_id] = running
            self._record_locked(RobotJobEventType.STARTED, running, {})
            self._condition.notify_all()
            return running

    async def complete(self, job_id: str, result: str) -> None:
        async with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            completed = replace(
                job,
                status=RobotJobStatus.COMPLETED,
                updated_at=self._time_fn(),
                result=result,
                error=None,
            )
            self._jobs[job_id] = completed
            self._record_locked(RobotJobEventType.COMPLETED, completed, {"result": result})
            self._condition.notify_all()

    async def fail(self, job_id: str, error: str, *, result: str | None = None) -> None:
        async with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.status in _TERMINAL_STATUSES:
                return
            failed = replace(
                job,
                status=RobotJobStatus.FAILED,
                updated_at=self._time_fn(),
                result=result,
                error=error,
            )
            self._jobs[job_id] = failed
            payload: dict[str, Any] = {"error": error}
            if result is not None:
                payload["result"] = result
            self._record_locked(RobotJobEventType.FAILED, failed, payload)
            self._condition.notify_all()

    async def cancel_queued_for_turn(
        self,
        *,
        requested_by_turn_id: str | None,
        tool_names: frozenset[str],
        reason: str,
    ) -> list[RobotJob]:
        if requested_by_turn_id is None:
            return []
        async with self._condition:
            cancelled: list[RobotJob] = []
            kept_queue: list[str] = []
            for job_id in self._queue:
                job = self._jobs[job_id]
                if (
                    job.requested_by_turn_id == requested_by_turn_id
                    and job.tool_name in tool_names
                    and job.status is RobotJobStatus.QUEUED
                ):
                    cancelled_job = replace(
                        job,
                        status=RobotJobStatus.CANCELLED,
                        updated_at=self._time_fn(),
                        error=reason,
                    )
                    self._jobs[job_id] = cancelled_job
                    self._record_locked(
                        RobotJobEventType.CANCELLED,
                        cancelled_job,
                        {"reason": reason},
                    )
                    cancelled.append(cancelled_job)
                else:
                    kept_queue.append(job_id)
            if cancelled:
                self._queue = kept_queue
                self._condition.notify_all()
            return cancelled

    def get(self, job_id: str) -> RobotJob | None:
        return self._jobs.get(job_id)

    def events_since(self, sequence: int) -> list[RobotJobEvent]:
        return [event for event in self._events if event.sequence > sequence]

    async def snapshot(self, *, max_events: int = 50) -> dict[str, Any]:
        async with self._condition:
            now = self._time_fn()
            queued_positions = {job_id: index + 1 for index, job_id in enumerate(self._queue)}
            jobs = sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)
            return {
                "now_monotonic_s": now,
                "counts": {
                    status.value: sum(1 for job in self._jobs.values() if job.status is status)
                    for status in RobotJobStatus
                },
                "queue": list(self._queue),
                "jobs": [
                    _job_snapshot(job, now=now, queue_position=queued_positions.get(job.job_id))
                    for job in jobs
                ],
                "events": [
                    _event_snapshot(event)
                    for event in self._events[-max(max_events, 0) :]
                ],
            }

    def render_instruction_block(
        self,
        *,
        max_age_s: float = 120.0,
        max_jobs: int = 5,
        context_recorded_sequences: set[int] | None = None,
    ) -> str | None:
        now = self._time_fn()
        recent_jobs = [
            job for job in self._jobs.values() if now - job.updated_at <= max_age_s
        ]
        if not recent_jobs:
            return None

        recorded = context_recorded_sequences or set()
        lines = ["Robot Job Blackboard:"]
        for job in sorted(recent_jobs, key=lambda item: item.updated_at, reverse=True)[:max_jobs]:
            parts = [
                f"- {job.tool_name}: {job.status.value} ({now - job.updated_at:.1f}s old)"
            ]
            args = _format_salient_arguments(job.arguments)
            if args:
                parts.append(args)
            terminal_sequence = self._terminal_event_sequence(job.job_id)
            if job.status is RobotJobStatus.COMPLETED:
                if terminal_sequence in recorded:
                    parts.append("result recorded in Robot Context")
                else:
                    parts.append("result not yet recorded in Robot Context")
            elif job.status is RobotJobStatus.FAILED and job.error:
                parts.append(f"error={job.error}")
            elif job.status is RobotJobStatus.CANCELLED and job.error:
                parts.append(f"reason={job.error}")
            lines.append("; ".join(parts) + ".")
        return "\n".join(lines)

    def _record_locked(
        self, event_type: RobotJobEventType, job: RobotJob, payload: dict[str, Any]
    ) -> None:
        event = RobotJobEvent(
            sequence=self._next_sequence,
            event_type=event_type,
            job_id=job.job_id,
            tool_name=job.tool_name,
            status=job.status,
            created_at=self._time_fn(),
            payload=dict(payload),
        )
        self._next_sequence += 1
        self._events.append(event)
        self._tracer.event(
            _trace_event_name(event_type),
            "robot_control",
            attributes=robot_job_trace_attributes(job, payload),
        )

    def _terminal_event_sequence(self, job_id: str) -> int | None:
        for event in reversed(self._events):
            if event.job_id == job_id and event.status in _TERMINAL_STATUSES:
                return event.sequence
        return None


def robot_job_trace_attributes(job: RobotJob, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    attributes: dict[str, Any] = {
        "job.id": job.job_id,
        "job.tool_name": job.tool_name,
        "job.status": job.status.value,
        "job.requested_by_turn_id": job.requested_by_turn_id,
    }
    for key, value in salient_job_arguments(job.arguments).items():
        attributes[f"job.arg.{key}"] = value
    payload = payload or {}
    if isinstance(payload.get("error"), str):
        attributes["job.error"] = payload["error"]
    if "result" in payload or job.result is not None:
        attributes["job.result_present"] = True
    return attributes


def salient_job_arguments(arguments: dict[str, Any]) -> dict[str, str]:
    salient: dict[str, str] = {}
    for key in SALIENT_JOB_ARGUMENTS:
        value = arguments.get(key)
        if isinstance(value, str) and value.strip():
            salient[key] = value.strip()
    return salient


def _format_salient_arguments(arguments: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value}" for key, value in salient_job_arguments(arguments).items()
    )


def _trace_event_name(event_type: RobotJobEventType) -> str:
    return f"robot.job.{event_type.value.removeprefix('robot_job_')}"


def _job_snapshot(
    job: RobotJob,
    *,
    now: float,
    queue_position: int | None,
) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "tool_name": job.tool_name,
        "arguments": dict(job.arguments),
        "salient_arguments": salient_job_arguments(job.arguments),
        "requested_by_turn_id": job.requested_by_turn_id,
        "user_text": job.user_text,
        "status": job.status.value,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "age_s": max(now - job.created_at, 0.0),
        "updated_age_s": max(now - job.updated_at, 0.0),
        "queue_position": queue_position,
        "result_present": job.result is not None,
        "error": job.error,
        "after_success_tool": job.after_success_tool,
        "after_success_arguments": (
            dict(job.after_success_arguments)
            if isinstance(job.after_success_arguments, dict)
            else None
        ),
        "execute_via_mcp": job.execute_via_mcp,
    }


def _event_snapshot(event: RobotJobEvent) -> dict[str, Any]:
    return {
        "sequence": event.sequence,
        "event_type": event.event_type.value,
        "job_id": event.job_id,
        "tool_name": event.tool_name,
        "status": event.status.value,
        "created_at": event.created_at,
        "payload": _monitor_payload(event.payload),
    }


def _monitor_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rendered = dict(payload)
    if "result" in rendered:
        rendered["result_present"] = True
        del rendered["result"]
    return rendered


_TERMINAL_STATUSES = {
    RobotJobStatus.COMPLETED,
    RobotJobStatus.FAILED,
    RobotJobStatus.CANCELLED,
}
