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


class RobotJobEventType(str, Enum):
    QUEUED = "robot_job_queued"
    STARTED = "robot_job_started"
    COMPLETED = "robot_job_completed"
    FAILED = "robot_job_failed"


@dataclass(frozen=True)
class SubmitRobotJob:
    tool_name: str
    arguments: dict[str, Any]
    requested_by_turn_id: str | None


@dataclass(frozen=True)
class RobotJob:
    job_id: str
    tool_name: str
    arguments: dict[str, Any]
    requested_by_turn_id: str | None
    status: RobotJobStatus
    created_at: float
    updated_at: float
    result: str | None = None
    error: str | None = None


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

    async def submit(self, job: SubmitRobotJob) -> RobotJob:
        async with self._condition:
            now = time.monotonic()
            stored = RobotJob(
                job_id=uuid.uuid4().hex,
                tool_name=job.tool_name,
                arguments=dict(job.arguments),
                requested_by_turn_id=job.requested_by_turn_id,
                status=RobotJobStatus.QUEUED,
                created_at=now,
                updated_at=now,
            )
            self._jobs[stored.job_id] = stored
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
            if job is None or job.status in {RobotJobStatus.COMPLETED, RobotJobStatus.FAILED}:
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

    async def fail(self, job_id: str, error: str) -> None:
        async with self._condition:
            job = self._jobs.get(job_id)
            if job is None or job.status in {RobotJobStatus.COMPLETED, RobotJobStatus.FAILED}:
                return
            failed = replace(
                job,
                status=RobotJobStatus.FAILED,
                updated_at=time.monotonic(),
                result=None,
                error=error,
            )
            self._jobs[job_id] = failed
            self._record_locked(RobotJobEventType.FAILED, failed, {"error": error})
            self._condition.notify_all()

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
