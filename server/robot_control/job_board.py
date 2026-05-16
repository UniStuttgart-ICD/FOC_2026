from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any


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
    def __init__(self) -> None:
        self._condition = asyncio.Condition()
        self._jobs: dict[str, RobotJob] = {}
        self._queue: list[str] = []
        self._events: list[RobotJobEvent] = []
        self._next_sequence = 1

    async def submit(self, job: SubmitRobotJob, *, front: bool = False) -> RobotJob:
        async with self._condition:
            now = time.monotonic()
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
            running = replace(job, status=RobotJobStatus.RUNNING, updated_at=time.monotonic())
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
                updated_at=time.monotonic(),
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
                updated_at=time.monotonic(),
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
                        updated_at=time.monotonic(),
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

    def _record_locked(
        self, event_type: RobotJobEventType, job: RobotJob, payload: dict[str, Any]
    ) -> None:
        event = RobotJobEvent(
            sequence=self._next_sequence,
            event_type=event_type,
            job_id=job.job_id,
            tool_name=job.tool_name,
            status=job.status,
            created_at=time.monotonic(),
            payload=dict(payload),
        )
        self._next_sequence += 1
        self._events.append(event)


_TERMINAL_STATUSES = {
    RobotJobStatus.COMPLETED,
    RobotJobStatus.FAILED,
    RobotJobStatus.CANCELLED,
}
