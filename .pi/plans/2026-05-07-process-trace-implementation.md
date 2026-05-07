# Process Trace Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an always-on, lightweight `process_trace` module that records semantic spans/events across wake, speech capture, STT, agent turn, LangGraph nodes, model calls, policy checks, validation, MCP tool calls, and TTS.

**Architecture:** A reusable local JSONL trace core is injected from the runtime composition root into thin Voice Runtime, Agent Control, and Robot Control adapters. The core owns IDs, parentage, timestamps, JSON safety, redaction, and writer failure handling. Existing modules keep their behavior ownership.

**Tech Stack:** Python 3.10+, pytest, stdlib `contextvars`, `dataclasses`, `json`, `pathlib`, `time`, `uuid`; existing Pipecat observers, LangGraph, LangChain messages, MCP bridge code. No new runtime dependency in v1.

---

## Source Inputs

- Approved design spec: `docs/superpowers/specs/2026-05-07-process-trace-design.md`.
- Vocabulary added to `CONTEXT.md`: Process Trace, Trace Session, Trace Turn, Trace Span, Trace Event.
- Existing example for v2 visualization: `C:/Users/Samuel/Documents/github/DF2025_CLEAN/docs/diagrams/voice_robot_concurrency_process.html`.
- Research decision: custom local JSONL core in v1; OpenTelemetry, LangSmith, Langfuse, Perfetto, and Chrome Trace remain future adapters/export targets.

## Parallel Execution Map

Run Task 1 first. After Task 1 lands, Tasks 2, 3, 4, and 5 can run in parallel because their write sets are disjoint. Task 6 runs after those worker branches are integrated.

| Task | Lane | Owner Write Set | Can Run |
| --- | --- | --- | --- |
| 1 | Core gate | `server/process_trace/*`, `server/tests/test_process_trace_core.py`, `server/tests/test_orthogonal_imports.py` | Serial first |
| 2 | Runtime config/wiring | `server/voice_runtime/profiles.py`, `server/config.py`, `server/runtime_profiles.toml`, `server/pipeline_builder.py`, `server/agent_processor_factory.py`, profile/config/pipeline/factory tests | Parallel after Task 1 |
| 3 | Voice/Pipecat | `server/process_trace/pipecat_observer.py`, `server/voice_runtime/agent_turn.py`, voice observer and agent-turn tests | Parallel after Task 1 |
| 4 | Agent/LangGraph | `server/langchain_agent_processor.py`, `server/langgraph_robot_agent.py`, LangChain/LangGraph tests | Parallel after Task 1 |
| 5 | Robot/MCP | `server/robot_control/mcp_bridge.py`, MCP bridge tests | Parallel after Task 1 |
| 6 | Integration/docs | `ARCHITECTURE.md`, `docs/architecture.md`, final verification only | Serial final |

Do not split `server/langgraph_robot_agent.py` between workers. Do not let the Voice/Pipecat worker edit `server/pipeline_builder.py`; Task 2 owns composition wiring.

## Naming Contract

Use these exact trace names in v1:

```text
trace.session_start
trace.turn_start
voice.wake
voice.speech_capture
voice.stt
voice.agent_turn
voice.tts
agent.backend_turn
agent.graph_turn
agent.langgraph_node.observe_current_pose
agent.langgraph_node.call_model
agent.langgraph_node.execute_robot_tool
agent.langgraph_node.repair_missing_action
agent.langgraph_node.execute_supported_action
agent.langgraph_node.stop_after_tool_limit
agent.langgraph_node.final_response
agent.model_call
robot.task_policy
robot.call_validation
robot.mcp.connect
robot.mcp.list_tools
robot.mcp.call_tool
robot.context_update
```

Status values are exactly:

```text
ok
error
cancelled
```

## Task 1: Core `process_trace` Package

**Owner:** core worker.  
**Depends on:** nothing.  
**Blocks:** Tasks 2 through 5.

**Files**

- Add: `server/process_trace/__init__.py`
- Add: `server/process_trace/context.py`
- Add: `server/process_trace/jsonl.py`
- Add: `server/process_trace/records.py`
- Add: `server/process_trace/trace.py`
- Add: `server/tests/test_process_trace_core.py`
- Modify: `server/tests/test_orthogonal_imports.py`

**Steps**

- [ ] Add tests first for session/turn/span ID creation, parent nesting, event records, JSONL writing, redaction, non-JSON-safe values, exception status, cancellation status, and write-failure disablement.
- [ ] Add structural import tests that pure core modules do not import Pipecat, LangGraph, LangChain, MCP, Voice Runtime, Agent Control, or Robot Control.
- [ ] Implement records and sanitizer.
- [ ] Implement contextvars, explicit context activation, and tracer-owned current session/turn fallback for Pipecat task boundaries.
- [ ] Implement tracer, sync/async span scope, manual span recording, no-op tracer, and memory writer for tests.
- [ ] Implement append-only JSONL writer that creates the parent directory and disables itself after the first write failure.
- [ ] Export the public API from `server/process_trace/__init__.py`.
- [ ] Run `python -m pytest server/tests/test_process_trace_core.py server/tests/test_orthogonal_imports.py`.
- [ ] Commit with `git add server/process_trace server/tests/test_process_trace_core.py server/tests/test_orthogonal_imports.py` and `git commit -m "Add process trace core"`.

**Core API**

Implement this public shape:

```python
# server/process_trace/__init__.py
from .context import TraceContext, current_trace_context, use_trace_context
from .jsonl import JsonlTraceWriter
from .trace import (
    MemoryTraceWriter,
    NoopProcessTracer,
    ProcessTracer,
    TraceOptions,
    TraceWriter,
)

__all__ = [
    "JsonlTraceWriter",
    "MemoryTraceWriter",
    "NoopProcessTracer",
    "ProcessTracer",
    "TraceContext",
    "TraceOptions",
    "TraceWriter",
    "current_trace_context",
    "use_trace_context",
]
```

```python
# server/process_trace/context.py
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TraceContext:
    trace_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    parent_span_id: str | None = None


_trace_id: ContextVar[str | None] = ContextVar("process_trace_id", default=None)
_session_id: ContextVar[str | None] = ContextVar("process_trace_session_id", default=None)
_turn_id: ContextVar[str | None] = ContextVar("process_trace_turn_id", default=None)
_span_stack: ContextVar[tuple[str, ...]] = ContextVar("process_trace_span_stack", default=())


def current_trace_context() -> TraceContext:
    stack = _span_stack.get()
    return TraceContext(
        trace_id=_trace_id.get(),
        session_id=_session_id.get(),
        turn_id=_turn_id.get(),
        parent_span_id=stack[-1] if stack else None,
    )


@contextmanager
def use_trace_context(context: TraceContext) -> Iterator[None]:
    trace_token: Token[str | None] = _trace_id.set(context.trace_id)
    session_token: Token[str | None] = _session_id.set(context.session_id)
    turn_token: Token[str | None] = _turn_id.set(context.turn_id)
    stack_token: Token[tuple[str, ...]] = _span_stack.set(
        (context.parent_span_id,) if context.parent_span_id else ()
    )
    try:
        yield
    finally:
        _span_stack.reset(stack_token)
        _turn_id.reset(turn_token)
        _session_id.reset(session_token)
        _trace_id.reset(trace_token)
```

```python
# server/process_trace/records.py
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

SCHEMA_VERSION = 1
TraceStatus = Literal["ok", "error", "cancelled"]
RecordType = Literal["span", "event"]
SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)
REDACTED = "[redacted]"


def sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            if _is_secret_key(key):
                result[key] = REDACTED
            else:
                result[key] = sanitize_value(raw_value)
        return result
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray | str):
        return [sanitize_value(item) for item in value]
    if isinstance(value, bytes | bytearray):
        return f"<{type(value).__name__} len={len(value)}>"
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def sanitize_attributes(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    return sanitize_value(dict(attributes or {}))


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(part in lowered for part in SECRET_KEY_PARTS)
```

```python
# server/process_trace/trace.py
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass
from types import TracebackType
from typing import Any, Protocol

from .context import TraceContext, current_trace_context
from .records import TraceStatus, sanitize_attributes


class TraceWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None:
        pass


@dataclass(frozen=True)
class TraceOptions:
    include_text: bool = True
    include_tool_payloads: bool = True


class MemoryTraceWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


class ProcessTracer:
    def __init__(self, writer: TraceWriter, options: TraceOptions | None = None) -> None:
        self._writer = writer
        self._options = options or TraceOptions()
        self._session_context: TraceContext | None = None
        self._turn_context: TraceContext | None = None

    @property
    def options(self) -> TraceOptions:
        return self._options

    def start_session(self, *, profile: str, category: str) -> TraceContext:
        trace_id = _new_id("tr")
        session_id = _new_id("session")
        context = TraceContext(trace_id=trace_id, session_id=session_id)
        self._session_context = context
        self._turn_context = None
        self.event(
            "trace.session_start",
            module="process_trace",
            attributes={"profile": profile, "category": category},
            context=context,
        )
        return context

    def start_turn(
        self,
        *,
        input_text: str | None = None,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        base = context or current_trace_context()
        if base.trace_id is None and self._session_context is not None:
            base = self._session_context
        turn_context = TraceContext(
            trace_id=base.trace_id or _new_id("tr"),
            session_id=base.session_id or _new_id("session"),
            turn_id=_new_id("turn"),
        )
        self._turn_context = turn_context
        event_attributes = dict(attributes or {})
        if self.options.include_text and input_text is not None:
            event_attributes["input_text"] = input_text
        self.event("trace.turn_start", module="process_trace", attributes=event_attributes, context=turn_context)
        return turn_context

    def span(
        self,
        name: str,
        *,
        module: str,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> "_TraceSpan":
        return _TraceSpan(self, name=name, module=module, attributes=attributes, context=context)

    def current_context(self) -> TraceContext:
        active = current_trace_context()
        if active.trace_id is not None or active.session_id is not None or active.turn_id is not None:
            return active
        if self._turn_context is not None:
            return self._turn_context
        if self._session_context is not None:
            return self._session_context
        return TraceContext()

    def record_span(
        self,
        name: str,
        *,
        module: str,
        started_at_unix_ns: int,
        ended_at_unix_ns: int,
        status: TraceStatus = "ok",
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
        parent_span_id: str | None = None,
    ) -> None:
        active = context or self.current_context()
        self._write(
            {
                "schema_version": 1,
                "record_type": "span",
                "trace_id": active.trace_id,
                "span_id": _new_id("sp"),
                "parent_span_id": parent_span_id or active.parent_span_id,
                "session_id": active.session_id,
                "turn_id": active.turn_id,
                "name": name,
                "module": module,
                "started_at_unix_ns": started_at_unix_ns,
                "ended_at_unix_ns": ended_at_unix_ns,
                "duration_ms": (ended_at_unix_ns - started_at_unix_ns) / 1_000_000,
                "status": status,
                "attributes": sanitize_attributes(attributes),
                "events": [],
            }
        )

    def event(
        self,
        name: str,
        *,
        module: str,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> None:
        active = context or self.current_context()
        now_ns = time.time_ns()
        self._write(
            {
                "schema_version": 1,
                "record_type": "event",
                "trace_id": active.trace_id,
                "span_id": _new_id("ev"),
                "parent_span_id": active.parent_span_id,
                "session_id": active.session_id,
                "turn_id": active.turn_id,
                "name": name,
                "module": module,
                "started_at_unix_ns": now_ns,
                "ended_at_unix_ns": now_ns,
                "duration_ms": 0.0,
                "status": "ok",
                "attributes": sanitize_attributes(attributes),
                "events": [],
            }
        )

    def _write(self, record: dict[str, Any]) -> None:
        self._writer.write(record)


class NoopProcessTracer(ProcessTracer):
    def __init__(self) -> None:
        super().__init__(MemoryTraceWriter())

    def _write(self, record: dict[str, Any]) -> None:
        return None


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"
```

The implementation must add `_TraceSpan.__enter__`, `_TraceSpan.__exit__`, `_TraceSpan.__aenter__`, and `_TraceSpan.__aexit__`. Cancellation uses `asyncio.CancelledError` and records `status="cancelled"`. Other exceptions record `status="error"` and re-raise.

```python
# server/process_trace/jsonl.py
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class JsonlTraceWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._disabled = False

    def write(self, record: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        except OSError:
            self._disabled = True
            logger.warning("process trace writer disabled after write failure", exc_info=True)
```

**Core Tests**

Add tests with these assertions:

```python
def test_nested_spans_record_parent_child_ids() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    session = tracer.start_session(profile="test", category="voice_robot")
    turn = tracer.start_turn(input_text="move home", context=session)
    with use_trace_context(turn):
        with tracer.span("voice.agent_turn", module="voice_runtime"):
            with tracer.span("agent.model_call", module="agent_control"):
                pass
    spans = [record for record in writer.records if record["record_type"] == "span"]
    assert spans[1]["parent_span_id"] == spans[0]["span_id"]
    assert spans[0]["turn_id"] == turn.turn_id
```

```python
async def test_async_cancelled_span_records_cancelled_status() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    session = tracer.start_session(profile="test", category="voice_robot")
    turn = tracer.start_turn(context=session)
    with use_trace_context(turn):
        with pytest.raises(asyncio.CancelledError):
            async with tracer.span("agent.model_call", module="agent_control"):
                raise asyncio.CancelledError()
    span = [record for record in writer.records if record["record_type"] == "span"][-1]
    assert span["status"] == "cancelled"
```

## Task 2: Runtime Profile, Config, and Composition Wiring

**Owner:** runtime wiring worker.  
**Depends on:** Task 1 public API.  
**Can run in parallel with:** Tasks 3, 4, and 5 after Task 1.

**Files**

- Modify: `server/voice_runtime/profiles.py`
- Modify: `server/config.py`
- Modify: `server/runtime_profiles.toml`
- Modify: `server/pipeline_builder.py`
- Modify: `server/agent_processor_factory.py`
- Modify: `server/tests/test_voice_runtime_profiles.py`
- Modify: `server/tests/test_config.py`
- Modify: `server/tests/test_pipeline_builder.py`
- Modify: `server/tests/test_agent_processor_factory.py`

**Steps**

- [ ] Add `ProcessTraceProfile` with always-on defaults.
- [ ] Parse `process_trace` as an optional table so older test fixtures continue to load.
- [ ] Add profile/config propagation.
- [ ] Add `[profiles.<name>.process_trace]` sections to every bundled runtime profile with `enabled = true`.
- [ ] Build the concrete tracer in `pipeline_builder.py`.
- [ ] Start one trace session during `build_pipeline`.
- [ ] Add `ProcessTraceObserver` to Pipecat observers when enabled.
- [ ] Pass the tracer into `create_agent_processor`.
- [ ] Extend `create_agent_processor` to pass the tracer into `LangChainAgentProcessor` and `AgentTurnProcessor`.
- [ ] Keep Pipecat built-in OpenTelemetry tracing disabled in v1; this task wires the custom local process trace only.
- [ ] Run `python -m pytest server/tests/test_voice_runtime_profiles.py server/tests/test_config.py server/tests/test_pipeline_builder.py server/tests/test_agent_processor_factory.py`.
- [ ] Commit with `git add server/voice_runtime/profiles.py server/config.py server/runtime_profiles.toml server/pipeline_builder.py server/agent_processor_factory.py server/tests/test_voice_runtime_profiles.py server/tests/test_config.py server/tests/test_pipeline_builder.py server/tests/test_agent_processor_factory.py` and `git commit -m "Wire process trace runtime config"`.

**Profile Types**

```python
# server/voice_runtime/profiles.py
@dataclass(frozen=True)
class ProcessTraceProfile:
    enabled: bool = True
    path: Path = Path("logs/process_trace.jsonl")
    include_text: bool = True
    include_tool_payloads: bool = True


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    category: Category
    wake: WakeProfile
    emergency_stop: EmergencyStopProfile
    stt: STTProfile
    tts: TTSProfile
    agent: AgentProfile
    mcp_robot_url: str
    metrics: MetricsProfile
    process_trace: ProcessTraceProfile
    server_dir: Path
```

```python
# server/voice_runtime/profiles.py
def _optional_table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ProfileError(f"{key} must be a table")
    return value


def _parse_process_trace(data: dict[str, Any], server_root: Path) -> ProcessTraceProfile:
    enabled = _bool(data, "enabled", default=True)
    path = _path(data, "path", server_root=server_root, default="logs/process_trace.jsonl")
    include_text = _bool(data, "include_text", default=True)
    include_tool_payloads = _bool(data, "include_tool_payloads", default=True)
    return ProcessTraceProfile(
        enabled=enabled,
        path=path,
        include_text=include_text,
        include_tool_payloads=include_tool_payloads,
    )
```

Use this in `load_runtime_profile`:

```python
process_trace = _parse_process_trace(_optional_table(raw_profile, "process_trace"), server_root)
```

**Config Propagation**

```python
# server/config.py
from voice_runtime.profiles import ProcessTraceProfile as ProcessTraceConfig


@dataclass(frozen=True)
class RuntimeConfig:
    profile_name: str
    category: Category
    wake: WakeConfig
    emergency_stop: EmergencyStopConfig
    stt: STTConfig
    tts: TTSConfig
    agent: AgentConfig
    mcp_robot_url: str
    metrics: MetricsConfig
    process_trace: ProcessTraceConfig
    server_dir: Path
```

Update `RuntimeConfig.from_profile()` and `RuntimeConfig.required_env_names()` so `process_trace` is copied and reconstructed.

**Profile TOML**

Add this section beside each profile's `[profiles.<name>.metrics]` section:

```toml
[profiles.hybrid_low_latency.process_trace]
enabled = true
path = "logs/process_trace.jsonl"
include_text = true
include_tool_payloads = true
```

Repeat for:

```text
hybrid_low_latency
hybrid_gemini
hybrid_anthropic
openai_all
deepgram_all
local_current
no_wake_debug
```

**Pipeline Wiring**

```python
# server/pipeline_builder.py
from process_trace import JsonlTraceWriter, NoopProcessTracer, ProcessTracer, TraceOptions
from process_trace.pipecat_observer import ProcessTraceObserver


@dataclass(frozen=True)
class BuiltPipeline:
    pipeline: Pipeline
    task: PipelineTask
    runner: PipelineRunner
    metrics: VoiceMetricsRecorder | None
    process_tracer: ProcessTracer
```

```python
# server/pipeline_builder.py
def _build_process_tracer(config: RuntimeConfig) -> ProcessTracer:
    if not config.process_trace.enabled:
        return NoopProcessTracer()
    return ProcessTracer(
        JsonlTraceWriter(config.process_trace.path),
        options=TraceOptions(
            include_text=config.process_trace.include_text,
            include_tool_payloads=config.process_trace.include_tool_payloads,
        ),
    )
```

Inside `build_pipeline`:

```python
process_tracer = _build_process_tracer(config)
session_context = process_tracer.start_session(
    profile=config.profile_name,
    category=config.category,
)

agent_processor = create_agent_processor(
    config.agent,
    mcp_server_url=config.mcp_robot_url,
    tracer=process_tracer,
    **agent_kwargs,
)

if config.process_trace.enabled:
    observers.append(ProcessTraceObserver(process_tracer, session_context=session_context))
```

Return the tracer:

```python
return BuiltPipeline(
    pipeline=pipeline,
    task=task,
    runner=runner,
    metrics=metrics,
    process_tracer=process_tracer,
)
```

**Agent Factory Wiring**

```python
# server/agent_processor_factory.py
from process_trace import NoopProcessTracer, ProcessTracer


def create_agent_processor(
    config: AgentConfig,
    *,
    mcp_server_url: str,
    on_turn_started: AgentTurnCallback | None = None,
    on_turn_finished: AgentTurnCallback | None = None,
    tracer: ProcessTracer | None = None,
) -> FrameProcessor:
    process_tracer = tracer or NoopProcessTracer()
    backend = LangChainAgentProcessor(
        config=config,
        mcp_server_url=mcp_server_url,
        tracer=process_tracer,
    )
    return AgentTurnProcessor(
        backend=backend,
        on_turn_started=on_turn_started,
        on_turn_finished=on_turn_finished,
        tracer=process_tracer,
    )
```

**Runtime Tests**

Add assertions:

```python
def test_process_trace_profile_defaults_when_section_missing(tmp_path: Path) -> None:
    profile = load_runtime_profile(profiles_path=_write_profiles(tmp_path), server_root=tmp_path)
    assert profile.process_trace.enabled is True
    assert profile.process_trace.path == tmp_path / "logs/process_trace.jsonl"
    assert profile.process_trace.include_text is True
    assert profile.process_trace.include_tool_payloads is True
```

```python
def test_pipeline_wires_process_trace_observer_and_agent_tracer(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_create_agent_processor(config: AgentConfig, **kwargs: Any) -> FrameProcessor:
        captured.update(kwargs)
        return FakeProcessor()

    monkeypatch.setattr(pipeline_builder, "create_agent_processor", fake_create_agent_processor)
    built = build_pipeline(_config(process_trace_enabled=True), FakeTransport())
    assert isinstance(built.process_tracer, ProcessTracer)
    assert captured["tracer"] is built.process_tracer
    assert any(observer.__class__.__name__ == "ProcessTraceObserver" for observer in built.task.observers)
```

## Task 3: Voice Runtime and Pipecat Trace Adapter

**Owner:** Voice/Pipecat worker.  
**Depends on:** Task 1 public API.  
**Can run in parallel with:** Tasks 2, 4, and 5 after Task 1.

**Files**

- Add: `server/process_trace/pipecat_observer.py`
- Modify: `server/voice_runtime/agent_turn.py`
- Add: `server/tests/test_process_trace_pipecat_observer.py`
- Modify: `server/tests/test_voice_runtime_agent_turn.py`

**Steps**

- [ ] Add tests for wake event, speech capture span, STT span, TTS span, and text inclusion toggles.
- [ ] Add tests for `AgentTurnProcessor` success and backend error spans.
- [ ] Implement a thin Pipecat observer that records semantic voice spans/events; do not edit `server/metrics.py`.
- [ ] Start a new trace turn on the first meaningful wake or speech/STT activity for a spoken command.
- [ ] Record `voice.wake` as an instant event because there is no current earlier wake start timestamp.
- [ ] Record `voice.speech_capture` from `UserStartedSpeakingFrame` to `UserStoppedSpeakingFrame`.
- [ ] Record `voice.stt` from `UserStoppedSpeakingFrame` to finalized `TranscriptionFrame`.
- [ ] Record `voice.tts` from first `LLMTextFrame` to `TTSStoppedFrame` or `BotStoppedSpeakingFrame`; record first audio as event attributes when available.
- [ ] Wrap backend execution in `AgentTurnProcessor` with `voice.agent_turn`.
- [ ] Record assistant response text from `AgentTurnProcessor` when `TraceOptions.include_text` is true.
- [ ] Run `python -m pytest server/tests/test_process_trace_pipecat_observer.py server/tests/test_voice_runtime_agent_turn.py`.
- [ ] Commit with `git add server/process_trace/pipecat_observer.py server/voice_runtime/agent_turn.py server/tests/test_process_trace_pipecat_observer.py server/tests/test_voice_runtime_agent_turn.py` and `git commit -m "Trace voice runtime process spans"`.

**Pipecat Observer Shape**

```python
# server/process_trace/pipecat_observer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed

from voice_runtime.wake_command import WakeDetectedFrame

from .context import TraceContext, use_trace_context
from .trace import ProcessTracer


@dataclass
class _OpenSpan:
    started_at_unix_ns: int
    context: TraceContext
    attributes: dict[str, Any]


class ProcessTraceObserver(BaseObserver):
    def __init__(self, tracer: ProcessTracer, *, session_context: TraceContext) -> None:
        self._tracer = tracer
        self._session_context = session_context
        self._turn_context: TraceContext | None = None
        self._speech: _OpenSpan | None = None
        self._stt: _OpenSpan | None = None
        self._tts: _OpenSpan | None = None
        self._tts_first_audio_seen = False

    async def on_push_frame(self, data: FramePushed) -> None:
        frame = data.frame
        if isinstance(frame, WakeDetectedFrame):
            self._on_wake(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._on_user_started_speaking()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._on_user_stopped_speaking()
        elif isinstance(frame, TranscriptionFrame) and getattr(frame, "final", True):
            self._on_transcription(frame)
        elif isinstance(frame, LLMTextFrame):
            self._on_llm_text(frame)
        elif isinstance(frame, TTSAudioRawFrame):
            self._on_tts_audio(frame)
        elif isinstance(frame, TTSStoppedFrame | BotStoppedSpeakingFrame):
            self._on_tts_stopped()
```

Voice timestamps should use `time.time_ns()` at the frame hook. Store the active `TraceContext` with every open span because Pipecat observer callbacks may run in processor tasks where implicit contextvars are not reliable.

```python
def _ensure_turn(self, *, input_text: str | None = None) -> TraceContext:
    if self._turn_context is None:
        self._turn_context = self._tracer.start_turn(
            input_text=input_text,
            context=self._session_context,
        )
    return self._turn_context
```

Do not record raw audio bytes:

```python
def _on_tts_audio(self, frame: TTSAudioRawFrame) -> None:
    context = self._ensure_turn()
    if not self._tts_first_audio_seen:
        self._tts_first_audio_seen = True
        self._tracer.event(
            "voice.tts_first_audio",
            module="voice_runtime",
            attributes={"audio_bytes": len(frame.audio)},
            context=context,
        )
```

**Agent Turn Span**

```python
# server/voice_runtime/agent_turn.py
from process_trace import NoopProcessTracer, ProcessTracer, use_trace_context


class AgentTurnProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        backend: AgentTurnBackend,
        on_turn_started: AgentTurnCallback | None = None,
        on_turn_finished: AgentTurnCallback | None = None,
        tracer: ProcessTracer | None = None,
    ) -> None:
        super().__init__()
        self._backend = backend
        self._on_turn_started = on_turn_started
        self._on_turn_finished = on_turn_finished
        self._tracer = tracer or NoopProcessTracer()
```

Wrap the backend call after `agent_turn_input(frame)` succeeds:

```python
turn_context = self._tracer.current_context()
if turn_context.turn_id is None:
    turn_context = self._tracer.start_turn(input_text=turn.user_text)
with use_trace_context(turn_context):
    async with self._tracer.span(
        "voice.agent_turn",
        module="voice_runtime",
        attributes={
            "input_text": turn.user_text if self._tracer.options.include_text else None,
            "message_count": len(turn.messages),
        },
    ):
        response = await self._backend.run_turn(turn)
        if self._tracer.options.include_text:
            self._tracer.event(
                "voice.agent_turn.response",
                module="voice_runtime",
                attributes={"assistant_text": response},
            )
```

If the observer already started the turn, the implementation should reuse the current turn context when present. If no turn exists, `AgentTurnProcessor` starts one so text-only tests and non-wake profiles still trace correctly.

## Task 4: Agent Control and LangGraph Trace Hooks

**Owner:** Agent/LangGraph worker.  
**Depends on:** Task 1 public API.  
**Can run in parallel with:** Tasks 2, 3, and 5 after Task 1.

**Files**

- Modify: `server/langchain_agent_processor.py`
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langchain_agent_processor.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

**Steps**

- [ ] Add tests for `agent.backend_turn`, `agent.graph_turn`, `agent.langgraph_node.*`, and `agent.model_call`.
- [ ] Add tracer injection to `LangChainAgentProcessor`.
- [ ] Wrap `LangChainAgentProcessor.run_turn` in `agent.backend_turn`.
- [ ] Pass the tracer into `RobotMCPBridge` and `LangGraphRobotAgent`.
- [ ] Add tracer injection to `LangGraphRobotAgent`.
- [ ] Wrap `run_turn` in `agent.graph_turn`.
- [ ] Wrap every LangGraph node registration with a traced async node wrapper.
- [ ] Wrap `model.ainvoke` in `agent.model_call`.
- [ ] Add `robot.task_policy` and `robot.context_update` spans/events at the policy-checked tool call site.
- [ ] Record events when the graph rejects extra model tool calls or takes supported-action fallback without MCP execution.
- [ ] Run `python -m pytest server/tests/test_langchain_agent_processor.py server/tests/test_langgraph_robot_agent.py`.
- [ ] Commit with `git add server/langchain_agent_processor.py server/langgraph_robot_agent.py server/tests/test_langchain_agent_processor.py server/tests/test_langgraph_robot_agent.py` and `git commit -m "Trace agent graph process spans"`.

**LangChain Backend**

```python
# server/langchain_agent_processor.py
from process_trace import NoopProcessTracer, ProcessTracer


class LangChainAgentProcessor(AgentTurnBackend):
    def __init__(
        self,
        config: AgentConfig,
        *,
        mcp_server_url: str,
        tracer: ProcessTracer | None = None,
    ) -> None:
        self._config = config
        self._mcp_server_url = mcp_server_url
        self._tracer = tracer or NoopProcessTracer()
```

```python
async def run_turn(self, turn: AgentTurn) -> str:
    async with self._tracer.span(
        "agent.backend_turn",
        module="agent_control",
        attributes={
            "provider": self._config.provider,
            "model": self._config.model,
            "message_count": len(turn.messages),
        },
    ):
        await self._ensure_connected()
        graph = await self._graph_agent_for(turn.thread_id)
        return await graph.run_turn(turn)
```

When creating dependencies:

```python
bridge = RobotMCPBridge(self._mcp_server_url, tracer=self._tracer)
agent = LangGraphRobotAgent(
    model=model,
    tools=tools,
    robot_context=RobotContextStore(),
    tracer=self._tracer,
)
```

**LangGraph Node Wrapper**

```python
# server/langgraph_robot_agent.py
from collections.abc import Awaitable, Callable
from process_trace import NoopProcessTracer, ProcessTracer

NodeFn = Callable[[RobotAgentState], Awaitable[dict[str, Any]]]


def _traced_node(self, node_name: str, node_fn: NodeFn) -> NodeFn:
    async def wrapped(state: RobotAgentState) -> dict[str, Any]:
        async with self._tracer.span(
            f"agent.langgraph_node.{node_name}",
            module="agent_control",
            attributes={
                "node.name": node_name,
                "tool_turns": state.get("tool_turns", 0),
                "message_count": len(state.get("messages", [])),
            },
        ):
            return await node_fn(state)

    return wrapped
```

Register nodes through the wrapper:

```python
builder.add_node("observe_current_pose", self._traced_node("observe_current_pose", self._observe_current_pose))
builder.add_node("call_model", self._traced_node("call_model", self._call_model))
builder.add_node("execute_robot_tool", self._traced_node("execute_robot_tool", self._execute_robot_tool))
builder.add_node("repair_missing_action", self._traced_node("repair_missing_action", self._repair_missing_action))
builder.add_node("execute_supported_action", self._traced_node("execute_supported_action", self._execute_supported_action))
builder.add_node("stop_after_tool_limit", self._traced_node("stop_after_tool_limit", self._stop_after_tool_limit))
builder.add_node("final_response", self._traced_node("final_response", self._final_response))
```

**Model Call**

```python
async with self._tracer.span(
    "agent.model_call",
    module="agent_control",
    attributes={
        "gen_ai.system": self._model_provider_name(),
        "gen_ai.request.model": self._model_name(),
        "message_count": len(messages),
        "tool_turns": state.get("tool_turns", 0),
    },
):
    response = await model.ainvoke(messages, config=config)
```

If the exact provider/model fields are not currently available, use existing logger metadata fields already present in `_call_model`.

**Policy and Context Events**

Policy belongs in `LangGraphRobotAgent._call_policy_checked_tool`, before MCP validation:

```python
async with self._tracer.span(
    "robot.task_policy",
    module="robot_control",
    attributes={"tool.name": tool_name},
):
    policy_result = validate_task_step(self._robot_context.snapshot(), tool_name, arguments)
```

After a successful MCP tool result is folded into Robot Context:

```python
self._robot_context.update_from_tool_result(tool_name, result)
self._tracer.event(
    "robot.context_update",
    module="robot_control",
    attributes={"tool.name": tool_name},
)
```

Do not move task policy into `RobotMCPBridge`; policy intentionally runs before bridge validation.

## Task 5: Robot Control MCP and Validation Trace Hooks

**Owner:** Robot/MCP worker.  
**Depends on:** Task 1 public API.  
**Can run in parallel with:** Tasks 2, 3, and 4 after Task 1.

**Files**

- Modify: `server/robot_control/mcp_bridge.py`
- Modify: `server/tests/test_robot_mcp_bridge.py`

**Steps**

- [ ] Add tests for `robot.mcp.connect`, `robot.mcp.list_tools`, `robot.call_validation`, and `robot.mcp.call_tool`.
- [ ] Add tracer injection to `RobotMCPBridge`.
- [ ] Wrap server connect in `robot.mcp.connect`.
- [ ] Wrap tool listing/deduping in `robot.mcp.list_tools`.
- [ ] Wrap `validate_robot_tool_call` in `robot.call_validation`; record blocked validation as an expected decision.
- [ ] Wrap the actual MCP server call in `robot.mcp.call_tool`.
- [ ] Include full tool arguments/results only when `tracer.options.include_tool_payloads` is true.
- [ ] Run `python -m pytest server/tests/test_robot_mcp_bridge.py server/tests/test_robot_call_validation.py`.
- [ ] Commit with `git add server/robot_control/mcp_bridge.py server/tests/test_robot_mcp_bridge.py` and `git commit -m "Trace robot MCP process spans"`.

**Bridge Constructor**

```python
# server/robot_control/mcp_bridge.py
from process_trace import NoopProcessTracer, ProcessTracer


class RobotMCPBridge:
    def __init__(
        self,
        server_url: str,
        *,
        server: MCPServerStreamableHTTP | None = None,
        tracer: ProcessTracer | None = None,
    ) -> None:
        self._server_url = server_url
        self._server = server
        self._tracer = tracer or NoopProcessTracer()
```

**Connect and List Tools**

```python
async def connect(self) -> None:
    async with self._tracer.span(
        "robot.mcp.connect",
        module="robot_control",
        attributes={"mcp.server_url": self._server_url},
    ):
        await self._server.connect()
    async with self._tracer.span("robot.mcp.list_tools", module="robot_control"):
        tools = await self._server.list_tools()
        self._tools = _dedupe_tools(tools)
```

**Validation and Tool Call**

```python
async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
    normalized = _normalize_agent_arguments(tool_name, arguments)
    payload_attributes = (
        {"tool.arguments": normalized}
        if self._tracer.options.include_tool_payloads
        else {}
    )
    async with self._tracer.span(
        "robot.call_validation",
        module="robot_control",
        attributes={"tool.name": tool_name, **payload_attributes},
    ):
        validation = validate_robot_tool_call(tool_name, normalized)
    if not validation.ok:
        self._tracer.event(
            "robot.call_validation.blocked",
            module="robot_control",
            attributes={"tool.name": tool_name, "reason": validation.reason},
        )
        return _validation_failure_json(validation)

    backing_name = self._tool_name_mapping[tool_name]
    async with self._tracer.span(
        "robot.mcp.call_tool",
        module="robot_control",
        attributes={"tool.name": tool_name, "mcp.tool.name": backing_name, **payload_attributes},
    ):
        result = await self._server.call_tool(backing_name, normalized)
    if self._tracer.options.include_tool_payloads:
        self._tracer.event(
            "robot.mcp.tool_result",
            module="robot_control",
            attributes={"tool.name": tool_name, "tool.result": result},
        )
    return _serialize_tool_result(result)
```

Keep validation failures as normal returned tool errors, not raised exceptions. The trace span status remains `ok` unless bridge code itself throws.

## Task 6: Final Integration, Documentation, and Verification

**Owner:** integration worker or main agent.  
**Depends on:** Tasks 2, 3, 4, and 5 integrated.

**Files**

- Modify: `ARCHITECTURE.md`
- Modify: `docs/architecture.md`
- Run verification over all changed test surfaces.

**Steps**

- [ ] Resolve import/signature mismatches between parallel workers.
- [ ] Confirm no worker edited another worker's owned file without coordination.
- [ ] Update architecture docs with a short `process_trace` module section and the runtime path.
- [ ] Confirm the trace is always-on by default in bundled profiles.
- [ ] Confirm disabling `process_trace.enabled = false` yields `NoopProcessTracer` and no observer.
- [ ] Confirm writer failures cannot break a voice run.
- [ ] Run focused tests from every task.
- [ ] Run broader server tests if focused tests pass.
- [ ] Inspect one generated JSONL record set from a fake pipeline/agent run and verify required fields are present.
- [ ] Commit with `git add ARCHITECTURE.md docs/architecture.md` and `git commit -m "Document process trace module"`.

**Architecture Doc Text**

Add a concise section like:

```md
## Process Trace

`process_trace` records semantic spans/events for voice robot runs. It is always-on by default and writes local JSONL under `logs/process_trace.jsonl`.

The module is observational. Voice Runtime emits wake, speech, STT, Agent Turn, and TTS spans. Agent Control emits backend, LangGraph node, and model-call spans. Robot Control emits policy, validation, Robot Context, and MCP spans. Visualization and external exporters read the JSONL trace in later work.
```

**Verification Commands**

Run these in order after integration:

```powershell
python -m pytest `
  server/tests/test_process_trace_core.py `
  server/tests/test_voice_runtime_profiles.py `
  server/tests/test_config.py `
  server/tests/test_pipeline_builder.py `
  server/tests/test_agent_processor_factory.py `
  server/tests/test_process_trace_pipecat_observer.py `
  server/tests/test_voice_runtime_agent_turn.py `
  server/tests/test_langchain_agent_processor.py `
  server/tests/test_langgraph_robot_agent.py `
  server/tests/test_robot_mcp_bridge.py `
  server/tests/test_robot_call_validation.py `
  server/tests/test_orthogonal_imports.py `
  server/tests/test_robot_control_imports.py
```

Then run the broader suite if time allows:

```powershell
python -m pytest server/tests
```

## Acceptance Criteria

- [ ] `process_trace` is enabled by default in runtime profiles.
- [ ] V1 writes local append-only JSONL with `schema_version = 1`.
- [ ] Records include `trace_id`, `span_id`, `parent_span_id`, `session_id`, `turn_id`, name, module, timestamps, duration, status, attributes, and events.
- [ ] Full transcript text, assistant text, tool arguments, and tool outputs are recorded when profile options allow them.
- [ ] Raw audio, API keys, auth material, secrets, and environment dumps are not recorded.
- [ ] Wake, speech capture, STT, agent turn, LangGraph node, model call, policy check, validation, MCP tool call, and TTS are represented.
- [ ] Tracing failures never break a robot run.
- [ ] Pure `process_trace` core imports no Pipecat, LangGraph, LangChain, MCP, Voice Runtime, Agent Control, or Robot Control modules.
- [ ] Visualization remains out of scope for this implementation; JSONL is ready for a later visualization/export task.

## Ready-To-Dispatch Worker Prompts

Use these after Task 1 has landed. Tell each worker they are not alone in the codebase and must not revert edits made by others.

**Worker 2: Runtime Config/Wiring**

```text
Implement Task 2 from .pi/plans/2026-05-07-process-trace-implementation.md. You own only server/voice_runtime/profiles.py, server/config.py, server/runtime_profiles.toml, server/pipeline_builder.py, server/agent_processor_factory.py, and their listed tests. Do not edit voice_runtime/agent_turn.py, langchain_agent_processor.py, langgraph_robot_agent.py, or robot_control/mcp_bridge.py. Use the ProcessTracer API from Task 1. You are not alone in the codebase; do not revert edits made by others.
```

**Worker 3: Voice/Pipecat**

```text
Implement Task 3 from .pi/plans/2026-05-07-process-trace-implementation.md. You own only server/process_trace/pipecat_observer.py, server/voice_runtime/agent_turn.py, server/tests/test_process_trace_pipecat_observer.py, and server/tests/test_voice_runtime_agent_turn.py. Do not edit pipeline_builder.py or metrics.py. Use the ProcessTracer API from Task 1. You are not alone in the codebase; do not revert edits made by others.
```

**Worker 4: Agent/LangGraph**

```text
Implement Task 4 from .pi/plans/2026-05-07-process-trace-implementation.md. You own only server/langchain_agent_processor.py, server/langgraph_robot_agent.py, server/tests/test_langchain_agent_processor.py, and server/tests/test_langgraph_robot_agent.py. Do not edit robot_control/mcp_bridge.py or pipeline_builder.py. Use the ProcessTracer API from Task 1. You are not alone in the codebase; do not revert edits made by others.
```

**Worker 5: Robot/MCP**

```text
Implement Task 5 from .pi/plans/2026-05-07-process-trace-implementation.md. You own only server/robot_control/mcp_bridge.py and server/tests/test_robot_mcp_bridge.py. Do not edit langgraph_robot_agent.py or pipeline_builder.py. Use the ProcessTracer API from Task 1. You are not alone in the codebase; do not revert edits made by others.
```

## Notes for Implementers

- Keep v1 local-first. Do not add OpenTelemetry, LangSmith, Langfuse, Phoenix, Logfire, Perfetto, or Chrome Trace dependencies.
- Do not call `PipelineTask(enable_tracing=True)` for this v1 unless a separate OpenTelemetry adapter is explicitly added later.
- Prefer explicit trace context passing where Pipecat or LangGraph crosses task boundaries.
- Keep spans semantic, not frame-by-frame.
- Keep trace attributes JSON-safe and compact enough for local append-only logs while honoring the approved default to include text and tool payloads.
