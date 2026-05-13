# Blackboard Robot Job Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple slow MoveIt MCP execution from the spoken Agent Turn by submitting typed robot jobs to a blackboard and executing them through a deterministic worker.

**Architecture:** The LLM agent still chooses robot tools and fills exact tool arguments. Agent Control submits long-running robot action calls as blackboard jobs, a deterministic Robot Control worker validates and executes the exact queued call, and Voice Runtime speaks completion/failure notifications from blackboard events. Observation remains synchronous for v1 so movement commands still start from fresh robot context.

**Tech Stack:** Python 3.10-3.12, LangGraph 1.x, LangChain Core 1.x, Pipecat frames/processors, MCP Streamable HTTP, pytest/pytest-asyncio, uv.

---

## Execution Setup

Work from the Git repository root:

```powershell
Set-Location C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
```

Create an isolated worktree before implementation. The current checkout has user changes; do not edit it directly.

```powershell
$repo = "C:\Users\Samuel\Documents\github\pipecat\pipecat-agent"
$branch = "blackboard-robot-jobs"
$worktree = Join-Path $repo ".worktrees\$branch"
Set-Location $repo
git check-ignore -q .worktrees
if ($LASTEXITCODE -ne 0) { throw ".worktrees must be gitignored before creating a worktree" }
git worktree add $worktree -b $branch
Set-Location $worktree
```

Copy environment files into the worktree without printing or inspecting secret contents. Stop if an env file would be tracked.

```powershell
$repo = "C:\Users\Samuel\Documents\github\pipecat\pipecat-agent"
$worktree = Join-Path $repo ".worktrees\blackboard-robot-jobs"
$envRelPaths = @(".env", "server\.env")
foreach ($rel in $envRelPaths) {
    $src = Join-Path $repo $rel
    if (-not (Test-Path -LiteralPath $src)) { continue }
    Set-Location $repo
    git check-ignore -q -- $rel
    if ($LASTEXITCODE -ne 0) { throw "$rel exists but is not gitignored; do not copy secrets" }
    $dest = Join-Path $worktree $rel
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
    Copy-Item -LiteralPath $src -Destination $dest -Force
}
Set-Location (Join-Path $worktree "server")
uv sync
uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_langgraph_robot_agent.py tests/test_voice_runtime_agent_turn.py -q
```

Expected baseline: selected tests pass or any failure is reported before implementation continues.

No changes return to the original checkout until after Samuel runs a real E2E test and approves the branch.

## File Structure

Create:

- `server/robot_control/job_board.py` - typed in-memory blackboard for robot jobs, job events, and advisory robot context snapshots.
- `server/robot_control/job_worker.py` - deterministic async worker that consumes queued jobs and executes the exact MoveIt tool call.
- `server/agent_control/robot_job_submission.py` - adapter that turns selected LangGraph tool calls into blackboard job submissions.
- `server/tests/test_robot_job_board.py` - Robot Control blackboard unit tests.
- `server/tests/test_robot_job_worker.py` - deterministic worker unit tests.
- `server/tests/test_robot_job_submission.py` - Agent Control queued-tool submission tests.

Modify:

- `server/agent_control/langgraph_robot_agent.py` - route queueable action tool calls through job submission and return accepted job feedback to the model.
- `server/agent_control/langchain_agent_processor.py` - own the job board/worker lifecycle and expose a notification source.
- `server/agent_control/factory.py` - pass notification-capable backend through the existing Agent Turn seam.
- `server/voice_runtime/agent_turn.py` - add optional background notification source support and speak notifications in LLM frames.
- `server/pipeline_builder.py` - keep app wiring in the composition root.
- `server/tests/test_langgraph_robot_agent.py` - assert queued action semantics.
- `server/tests/test_langchain_agent_processor.py` - assert worker lifecycle and notification source behavior.
- `server/tests/test_agent_processor_factory.py` - assert factory wiring.
- `server/tests/test_voice_runtime_agent_turn.py` - assert notifications can be spoken outside a user turn.
- `ARCHITECTURE.md` and `CONTEXT.md` - add blackboard/job terminology and ownership once behavior is implemented.

Parallelization:

- Task 1 is prerequisite.
- After Task 1, Task 2, Task 3, and Task 4 can run in parallel with disjoint write sets.
- Task 5 integrates the lanes and must run sequentially.
- Task 6 docs and Task 7 verification run after integration.

## Task 1: Robot Job Blackboard Contract

**Files:**
- Create: `server/robot_control/job_board.py`
- Test: `server/tests/test_robot_job_board.py`

- [ ] **Step 1: Write failing blackboard tests**

Create `server/tests/test_robot_job_board.py`:

```python
import pytest

from robot_control.job_board import (
    RobotJobBoard,
    RobotJobEventType,
    RobotJobStatus,
    SubmitRobotJob,
)


@pytest.mark.asyncio
async def test_submit_job_records_queued_event_and_returns_job_id() -> None:
    board = RobotJobBoard()

    job = await board.submit(
        SubmitRobotJob(
            tool_name="moveit_plan_and_execute_free_motion",
            arguments={"robot_name": "UR10", "timeout_s": 10},
            requested_by_turn_id="turn-1",
        )
    )

    assert job.job_id
    assert job.status is RobotJobStatus.QUEUED
    assert job.tool_name == "moveit_plan_and_execute_free_motion"
    assert job.arguments == {"robot_name": "UR10", "timeout_s": 10}
    events = board.events_since(0)
    assert [(event.event_type, event.job_id) for event in events] == [
        (RobotJobEventType.QUEUED, job.job_id)
    ]


@pytest.mark.asyncio
async def test_worker_claims_jobs_fifo() -> None:
    board = RobotJobBoard()
    first = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    second = await board.submit(
        SubmitRobotJob("moveit_close_gripper", {"robot_name": "UR10"}, "turn-2")
    )

    claimed_first = await board.claim_next()
    claimed_second = await board.claim_next()

    assert claimed_first is not None
    assert claimed_second is not None
    assert claimed_first.job_id == first.job_id
    assert claimed_second.job_id == second.job_id
    assert claimed_first.status is RobotJobStatus.RUNNING
    assert claimed_second.status is RobotJobStatus.RUNNING


@pytest.mark.asyncio
async def test_complete_and_fail_record_terminal_events() -> None:
    board = RobotJobBoard()
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    claimed = await board.claim_next()
    assert claimed is not None

    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')
    await board.fail(job.job_id, "ignored after completion")

    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert stored.result == '{"structured_content": {"ok": true}}'
    assert [event.event_type for event in board.events_since(0)] == [
        RobotJobEventType.QUEUED,
        RobotJobEventType.STARTED,
        RobotJobEventType.COMPLETED,
    ]
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
Set-Location C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\.worktrees\blackboard-robot-jobs\server
uv run pytest tests/test_robot_job_board.py -q
```

Expected: fail because `robot_control.job_board` does not exist.

- [ ] **Step 3: Implement minimal blackboard**

Create `server/robot_control/job_board.py`:

```python
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
```

- [ ] **Step 4: Verify blackboard tests pass**

Run:

```powershell
uv run pytest tests/test_robot_job_board.py -q
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add server/robot_control/job_board.py server/tests/test_robot_job_board.py
git commit -m "feat: add robot job blackboard"
```

## Task 2: Deterministic MoveIt Job Worker

**Files:**
- Create: `server/robot_control/job_worker.py`
- Test: `server/tests/test_robot_job_worker.py`

- [ ] **Step 1: Write failing worker tests**

Create `server/tests/test_robot_job_worker.py`:

```python
import pytest

from robot_control.job_board import RobotJobBoard, RobotJobStatus, SubmitRobotJob
from robot_control.job_worker import RobotJobWorker


class FakeToolBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.result = '{"structured_content": {"ok": true}}'

    async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
        self.calls.append((name, arguments))
        return self.result


@pytest.mark.asyncio
async def test_worker_executes_exact_queued_call() -> None:
    board = RobotJobBoard()
    bridge = FakeToolBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob(
            tool_name="moveit_plan_and_execute_free_motion",
            arguments={"robot_name": "UR10", "timeout_s": 10},
            requested_by_turn_id="turn-1",
        )
    )

    ran = await worker.run_once()

    assert ran is True
    assert bridge.calls == [
        ("moveit_plan_and_execute_free_motion", {"robot_name": "UR10", "timeout_s": 10})
    ]
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED


@pytest.mark.asyncio
async def test_worker_records_tool_failure_without_retrying_or_rewriting_args() -> None:
    class FailingBridge(FakeToolBridge):
        async def call_tool(self, name: str, arguments: dict[str, object]) -> str:
            self.calls.append((name, arguments))
            raise RuntimeError("planning failed")

    board = RobotJobBoard()
    bridge = FailingBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )

    ran = await worker.run_once()

    assert ran is True
    assert bridge.calls == [("moveit_open_gripper", {"robot_name": "UR10"})]
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.FAILED
    assert stored.error == "planning failed"
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
uv run pytest tests/test_robot_job_worker.py -q
```

Expected: fail because `robot_control.job_worker` does not exist.

- [ ] **Step 3: Implement deterministic worker**

Create `server/robot_control/job_worker.py`:

```python
from __future__ import annotations

import asyncio
from typing import Any, Protocol

from robot_control.job_board import RobotJobBoard


class RobotToolBridgeLike(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class RobotJobWorker:
    def __init__(self, *, board: RobotJobBoard, tool_bridge: RobotToolBridgeLike) -> None:
        self._board = board
        self._tool_bridge = tool_bridge
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
            result = await self._tool_bridge.call_tool(job.tool_name, dict(job.arguments))
        except Exception as exc:
            await self._board.fail(job.job_id, str(exc))
        else:
            await self._board.complete(job.job_id, result)
        return True

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            ran = await self.run_once()
            if not ran:
                await asyncio.sleep(0.05)
```

- [ ] **Step 4: Verify worker tests pass**

Run:

```powershell
uv run pytest tests/test_robot_job_worker.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```powershell
git add server/robot_control/job_worker.py server/tests/test_robot_job_worker.py
git commit -m "feat: add deterministic robot job worker"
```

## Task 3: Agent Control Queued Tool Submission

**Files:**
- Create: `server/agent_control/robot_job_submission.py`
- Modify: `server/agent_control/langgraph_robot_agent.py`
- Test: `server/tests/test_robot_job_submission.py`
- Test: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Write failing submission adapter tests**

Create `server/tests/test_robot_job_submission.py`:

```python
import json

import pytest

from agent_control.robot_job_submission import (
    QUEUEABLE_ROBOT_ACTION_TOOLS,
    RobotJobSubmitter,
)
from robot_control.job_board import RobotJobBoard, RobotJobStatus


@pytest.mark.asyncio
async def test_submitter_queues_action_tool_and_returns_structured_tool_feedback() -> None:
    board = RobotJobBoard()
    submitter = RobotJobSubmitter(board)

    output = await submitter.submit(
        "moveit_plan_and_execute_free_motion",
        {"robot_name": "UR10", "timeout_s": 10},
        requested_by_turn_id="turn-1",
    )

    payload = json.loads(output)
    assert payload["structured_content"]["ok"] is True
    assert payload["structured_content"]["status"] == "queued"
    assert payload["structured_content"]["tool_name"] == "moveit_plan_and_execute_free_motion"
    job = board.get(payload["structured_content"]["job_id"])
    assert job is not None
    assert job.status is RobotJobStatus.QUEUED


def test_queueable_action_tools_do_not_include_observation() -> None:
    assert "moveit_get_current_pose" not in QUEUEABLE_ROBOT_ACTION_TOOLS
    assert "moveit_plan_and_execute_free_motion" in QUEUEABLE_ROBOT_ACTION_TOOLS
    assert "moveit_open_gripper" in QUEUEABLE_ROBOT_ACTION_TOOLS
```

- [ ] **Step 2: Verify tests fail**

Run:

```powershell
uv run pytest tests/test_robot_job_submission.py -q
```

Expected: fail because `agent_control.robot_job_submission` does not exist.

- [ ] **Step 3: Implement submission adapter**

Create `server/agent_control/robot_job_submission.py`:

```python
from __future__ import annotations

import json
from typing import Any

from robot_control.job_board import RobotJobBoard, SubmitRobotJob


QUEUEABLE_ROBOT_ACTION_TOOLS = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_and_execute_free_motion",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_execute_plan",
        "moveit_open_gripper",
        "moveit_close_gripper",
        "moveit_attach_object",
    }
)


class RobotJobSubmitter:
    def __init__(self, board: RobotJobBoard) -> None:
        self._board = board

    async def submit(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        requested_by_turn_id: str | None,
    ) -> str:
        job = await self._board.submit(
            SubmitRobotJob(
                tool_name=tool_name,
                arguments=dict(arguments),
                requested_by_turn_id=requested_by_turn_id,
            )
        )
        return json.dumps(
            {
                "content": [
                    f"Queued robot job {job.job_id} for {tool_name}. "
                    "The robot worker will report completion or failure."
                ],
                "structured_content": {
                    "ok": True,
                    "status": "queued",
                    "job_id": job.job_id,
                    "tool_name": tool_name,
                },
                "is_error": False,
            },
            ensure_ascii=False,
        )
```

- [ ] **Step 4: Verify submission adapter tests pass**

Run:

```powershell
uv run pytest tests/test_robot_job_submission.py -q
```

Expected: 2 passed.

- [ ] **Step 5: Add LangGraph queued action tests**

Append to `server/tests/test_langgraph_robot_agent.py`:

```python
@pytest.mark.asyncio
async def test_graph_queues_long_running_action_tool_when_submitter_is_present() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard, RobotJobStatus

    board = RobotJobBoard()
    action_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
        },
        "timeout_s": 10,
    }
    fixture = make_graph(
        [ai_tool_call("moveit_plan_and_execute_free_motion", action_args), ai_text("Started.")],
    )
    fixture.graph._job_submitter = RobotJobSubmitter(board)

    text = await fixture.graph.run_turn(turn("move up a bit"))

    assert text == "Started."
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    queued = [event for event in board.events_since(0) if event.tool_name]
    assert queued[0].tool_name == "moveit_plan_and_execute_free_motion"
    job = board.get(queued[0].job_id)
    assert job is not None
    assert job.status is RobotJobStatus.QUEUED
    tool_output = json.loads(last_tool_content(fixture.model))
    assert tool_output["structured_content"]["status"] == "queued"
```

- [ ] **Step 6: Update LangGraph agent constructor and execution route**

Modify `server/agent_control/langgraph_robot_agent.py`:

```python
from agent_control.robot_job_submission import (
    QUEUEABLE_ROBOT_ACTION_TOOLS,
    RobotJobSubmitter,
)
```

Add constructor parameter and field:

```python
        job_submitter: RobotJobSubmitter | None = None,
```

```python
        self._job_submitter = job_submitter
```

Inside `_execute_robot_tool`, replace the non-observation execution branch with:

```python
            if name in OBSERVE_TOOL_NAMES:
                output, observed_this_turn = await self._execute_observation_tool(name, dict(args))
            elif self._job_submitter is not None and name in QUEUEABLE_ROBOT_ACTION_TOOLS:
                output = await self._job_submitter.submit(
                    name,
                    dict(args),
                    requested_by_turn_id=None,
                )
                action_tool_ran = True
                observed_this_turn = False
            else:
                output = await self._execute_tool(name, dict(args))
                action_tool_ran = action_tool_ran or name in ACTION_TOOL_NAMES
                observed_this_turn = False
```

Do not queue `moveit_get_current_pose`; it must remain synchronous.

- [ ] **Step 7: Verify LangGraph queue behavior**

Run:

```powershell
uv run pytest tests/test_robot_job_submission.py tests/test_langgraph_robot_agent.py::test_graph_queues_long_running_action_tool_when_submitter_is_present -q
```

Expected: selected tests pass.

- [ ] **Step 8: Commit**

```powershell
git add server/agent_control/robot_job_submission.py server/agent_control/langgraph_robot_agent.py server/tests/test_robot_job_submission.py server/tests/test_langgraph_robot_agent.py
git commit -m "feat: queue robot actions from agent orchestration"
```

## Task 4: Voice Runtime Notification Source

**Files:**
- Modify: `server/voice_runtime/agent_turn.py`
- Test: `server/tests/test_voice_runtime_agent_turn.py`

- [ ] **Step 1: Write failing notification tests**

Append to `server/tests/test_voice_runtime_agent_turn.py`:

```python
import asyncio


class NotificationBackend(EchoBackend):
    def __init__(self) -> None:
        super().__init__([])
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()

    async def notifications(self):
        while True:
            text = await self.queue.get()
            if text is None:
                return
            yield text


@pytest.mark.asyncio
async def test_agent_turn_processor_speaks_backend_notifications_outside_user_turn() -> None:
    backend = NotificationBackend()
    processor = CapturingProcessor(backend)

    await processor.connect()
    await backend.queue.put("Robot motion completed.")
    await asyncio.sleep(0)
    await backend.queue.put(None)
    await processor.disconnect()

    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    assert [frame.text for frame in text_frames] == ["Robot motion completed."]
    assert [type(frame) for frame in processor.pushed] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]
```

- [ ] **Step 2: Verify test fails**

Run:

```powershell
uv run pytest tests/test_voice_runtime_agent_turn.py::test_agent_turn_processor_speaks_backend_notifications_outside_user_turn -q
```

Expected: fail because `AgentTurnProcessor` does not pump backend notifications.

- [ ] **Step 3: Add optional notification protocol and pump**

Modify `server/voice_runtime/agent_turn.py`:

```python
import asyncio
```

Add protocol:

```python
class AgentNotificationSource(Protocol):
    def notifications(self) -> AsyncIterator[str]: ...
```

Add field in `AgentTurnProcessor.__init__`:

```python
        self._notification_task: asyncio.Task[None] | None = None
```

Modify `connect`:

```python
    async def connect(self) -> None:
        await self._backend.connect()
        notifications = getattr(self._backend, "notifications", None)
        if notifications is not None and self._notification_task is None:
            self._notification_task = asyncio.create_task(self._pump_notifications(notifications))
```

Modify `disconnect`:

```python
    async def disconnect(self) -> None:
        if self._notification_task is not None:
            self._notification_task.cancel()
            try:
                await self._notification_task
            except asyncio.CancelledError:
                pass
            self._notification_task = None
        await self._backend.disconnect()
```

Add method:

```python
    async def _pump_notifications(self, notifications: Callable[[], AsyncIterator[str]]) -> None:
        async for text in notifications():
            if not text:
                continue
            await self.push_frame(LLMFullResponseStartFrame())
            await self.push_frame(LLMTextFrame(text=text))
            await self.push_frame(LLMFullResponseEndFrame())
```

- [ ] **Step 4: Verify notification test passes**

Run:

```powershell
uv run pytest tests/test_voice_runtime_agent_turn.py::test_agent_turn_processor_speaks_backend_notifications_outside_user_turn -q
```

Expected: 1 passed.

- [ ] **Step 5: Run full Agent Turn tests**

Run:

```powershell
uv run pytest tests/test_voice_runtime_agent_turn.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```powershell
git add server/voice_runtime/agent_turn.py server/tests/test_voice_runtime_agent_turn.py
git commit -m "feat: speak backend robot job notifications"
```

## Task 5: Backend Integration And Job Notifications

**Files:**
- Modify: `server/agent_control/langchain_agent_processor.py`
- Modify: `server/agent_control/factory.py`
- Modify: `server/pipeline_builder.py`
- Test: `server/tests/test_langchain_agent_processor.py`
- Test: `server/tests/test_agent_processor_factory.py`
- Test: `server/tests/test_pipeline_builder.py`

- [ ] **Step 1: Add failing backend lifecycle tests**

Append to `server/tests/test_langchain_agent_processor.py`:

```python
@pytest.mark.asyncio
async def test_langchain_processor_starts_and_stops_robot_job_worker() -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    worker = FakeWorker()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_worker=worker,
    )

    await processor.connect()
    await processor.disconnect()

    assert worker.started is True
    assert worker.stopped is True


@pytest.mark.asyncio
async def test_langchain_processor_notifications_report_terminal_job_events() -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    await board.claim_next()
    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert "Robot job" in text
    assert "completed" in text
```

- [ ] **Step 2: Verify backend tests fail**

Run:

```powershell
uv run pytest tests/test_langchain_agent_processor.py::test_langchain_processor_starts_and_stops_robot_job_worker tests/test_langchain_agent_processor.py::test_langchain_processor_notifications_report_terminal_job_events -q
```

Expected: fail because constructor injection and notifications are missing.

- [ ] **Step 3: Wire board, submitter, worker, and notifications**

Modify `server/agent_control/langchain_agent_processor.py` imports:

```python
import asyncio

from agent_control.robot_job_submission import RobotJobSubmitter
from robot_control.job_board import RobotJobBoard, RobotJobEventType
from robot_control.job_worker import RobotJobWorker
```

Add constructor parameters:

```python
        robot_job_board: RobotJobBoard | None = None,
        robot_job_worker: Any | None = None,
```

Add fields:

```python
        self._robot_job_board = robot_job_board or RobotJobBoard()
        self._robot_job_submitter = RobotJobSubmitter(self._robot_job_board)
        self._robot_job_worker = robot_job_worker
        self._owns_robot_job_worker = robot_job_worker is None
```

After tool bridge connection in `_ensure_connected`, create and start worker:

```python
        if self._robot_job_worker is None and self._tool_bridge is not None:
            self._robot_job_worker = RobotJobWorker(
                board=self._robot_job_board,
                tool_bridge=self._tool_bridge,
            )
        if self._robot_job_worker is not None:
            await self._robot_job_worker.start()
```

In `disconnect`, stop worker before disconnecting the tool bridge:

```python
        if self._robot_job_worker is not None:
            stop = getattr(self._robot_job_worker, "stop", None)
            if stop is not None:
                await stop()
```

Pass submitter into graph construction:

```python
                "job_submitter": self._robot_job_submitter,
```

Add notification stream:

```python
    async def notifications(self):
        sequence = 0
        while True:
            events = self._robot_job_board.events_since(sequence)
            for event in events:
                sequence = max(sequence, event.sequence)
                if event.event_type is RobotJobEventType.COMPLETED:
                    yield f"Robot job {event.job_id} completed."
                elif event.event_type is RobotJobEventType.FAILED:
                    error = event.payload.get("error", "unknown error")
                    yield f"Robot job {event.job_id} failed: {error}"
            await asyncio.sleep(0.05)
```

- [ ] **Step 4: Verify backend integration tests**

Run:

```powershell
uv run pytest tests/test_langchain_agent_processor.py -q
```

Expected: all LangChain processor tests pass.

- [ ] **Step 5: Update factory/pipeline tests if constructor signatures assert kwargs**

Run:

```powershell
uv run pytest tests/test_agent_processor_factory.py tests/test_pipeline_builder.py -q
```

Expected: pass. If tests fail because fake graph constructors assert keyword args, update those fake constructors to accept `job_submitter: Any | None = None`. If factory or pipeline tests assert constructor args, update expectations to include no required blackboard args; defaults should keep current app wiring unchanged.

- [ ] **Step 6: Commit**

```powershell
git add server/agent_control/langchain_agent_processor.py server/agent_control/factory.py server/pipeline_builder.py server/tests/test_langchain_agent_processor.py server/tests/test_agent_processor_factory.py server/tests/test_pipeline_builder.py
git commit -m "feat: wire robot job blackboard into agent backend"
```

## Task 6: Documentation And Domain Language

**Files:**
- Modify: `CONTEXT.md`
- Modify: `ARCHITECTURE.md`
- Modify: `server/agent_control/prompt_parts/robot_contract.md`
- Test: `server/tests/test_prompts.py`

- [ ] **Step 1: Update domain glossary**

Add concise terms to `CONTEXT.md` under Robot Control:

```markdown
**Robot Job Blackboard**:
The shared typed job/event surface for long-running robot action execution. Agent Control writes queued robot jobs; Robot Control workers write started, completed, and failed events.

**Robot Job Worker**:
A deterministic Robot Control worker that validates and executes the exact queued MoveIt tool call. It does not invent new tool calls, repair arguments, or make LLM decisions.
```

- [ ] **Step 2: Update architecture map**

In `ARCHITECTURE.md`, update Robot Control contains-list with:

```markdown
- **Robot Job Blackboard**: stores queued/running/completed/failed robot jobs and terminal events for long-running action execution.
- **Robot Job Worker**: deterministic executor for queued MoveIt jobs; it calls the exact tool and arguments submitted by Agent Control.
```

Add invariant:

```markdown
### Long-running robot execution is blackboarded

Agent Control may queue long-running MoveIt action tools as Robot Jobs. The Robot Job Worker owns deterministic execution and writes terminal events. The LLM may decide what tool call to submit, but the worker must not improvise, repair, or reinterpret the tool arguments.
```

- [ ] **Step 3: Update robot prompt contract**

In `server/agent_control/prompt_parts/robot_contract.md`, add a short rule:

```markdown
- Long-running robot action tools may return a queued job id instead of a completed motion result. When a job is queued, tell the user the action has started and wait for the job completion or failure notification before claiming completion.
```

- [ ] **Step 4: Run prompt/doc tests**

Run:

```powershell
uv run pytest tests/test_prompts.py -q
```

Expected: pass. If prompt tests assert exact text, update only the expected prompt fragment related to queued job semantics.

- [ ] **Step 5: Commit**

```powershell
git add CONTEXT.md ARCHITECTURE.md server/agent_control/prompt_parts/robot_contract.md server/tests/test_prompts.py
git commit -m "docs: document robot job blackboard semantics"
```

## Task 7: Verification, Parallel Safety, And Manual E2E Gate

**Files:**
- Modify: only test/evidence docs if a command reveals a concrete missing assertion.

- [ ] **Step 1: Run targeted test set**

Run:

```powershell
Set-Location C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\.worktrees\blackboard-robot-jobs\server
uv run pytest tests/test_robot_job_board.py tests/test_robot_job_worker.py tests/test_robot_job_submission.py tests/test_langgraph_robot_agent.py tests/test_langchain_agent_processor.py tests/test_voice_runtime_agent_turn.py -q
```

Expected: pass.

- [ ] **Step 2: Run structural and app wiring tests**

Run:

```powershell
uv run pytest tests/test_orthogonal_imports.py tests/test_agent_processor_factory.py tests/test_pipeline_builder.py tests/test_prompts.py -q
```

Expected: pass.

- [ ] **Step 3: Run lint and type checks**

Run:

```powershell
uv run ruff check .
uv run pyright .
```

Expected: no errors.

- [ ] **Step 4: Run full unit suite**

Run:

```powershell
uv run pytest -q
```

Expected: pass.

- [ ] **Step 5: Prepare Samuel's E2E run command**

Run the app from the worktree only:

```powershell
Set-Location C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\.worktrees\blackboard-robot-jobs\server
uv run bot.py
```

Samuel performs the real E2E voice/MoveIt run. Expected behavior:

- Spoken movement request gets an immediate “started/queued” style response.
- The robot action proceeds through MoveIt MCP in the background.
- Completion or failure is spoken as a later notification.
- No changes are merged back before Samuel confirms the E2E behavior.

- [ ] **Step 6: Bring work back only after user approval**

After Samuel confirms E2E success:

```powershell
Set-Location C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\.worktrees\blackboard-robot-jobs
git status --short
git log --oneline --max-count=10
```

Then merge/cherry-pick from the original checkout only with explicit approval.

## Parallel Agent Handoff

Use this dispatch shape after Task 1 lands:

- Agent A owns Task 2 only: `server/robot_control/job_worker.py`, `server/tests/test_robot_job_worker.py`.
- Agent B owns Task 3 only: `server/agent_control/robot_job_submission.py`, `server/agent_control/langgraph_robot_agent.py`, `server/tests/test_robot_job_submission.py`, selected `server/tests/test_langgraph_robot_agent.py`.
- Agent C owns Task 4 only: `server/voice_runtime/agent_turn.py`, `server/tests/test_voice_runtime_agent_turn.py`.

Do not run Agents A, B, and C against the same files. Task 5 is the sequential integration pass after their commits are present.

## Self-Review

- Spec coverage: the plan covers isolated worktree setup, `.env` copy without secret disclosure, deterministic worker, LLM-filled queued tools, blackboard events, voice notifications, docs, automated tests, and Samuel's manual E2E gate.
- Placeholder scan: no deferred implementation placeholders remain; every task names files, test commands, expected results, and concrete code entry points.
- Type consistency: `RobotJobBoard`, `SubmitRobotJob`, `RobotJobWorker`, and `RobotJobSubmitter` names are consistent across tests and implementation steps.
