import asyncio
import json
from pathlib import Path

import pytest

from process_trace import (
    JsonlTraceWriter,
    MemoryTraceWriter,
    NoopProcessTracer,
    ProcessTracer,
    TraceContext,
    current_trace_context,
    use_trace_context,
)


def test_nested_spans_record_parent_child_ids_and_turn_id() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    tracer.start_session(profile="hybrid_low_latency", category="benchmark_streaming")
    turn_context = tracer.start_turn(input_text="move up")

    with tracer.span("outer", "test"):
        with tracer.span("inner", "test"):
            pass

    spans = [record for record in writer.records if record["record_type"] == "span"]
    assert [span["name"] for span in spans] == ["inner", "outer"]
    inner, outer = spans
    assert inner["trace_id"] == turn_context.trace_id
    assert inner["turn_id"] == turn_context.turn_id
    assert outer["turn_id"] == turn_context.turn_id
    assert inner["parent_span_id"] == outer["span_id"]
    assert outer["parent_span_id"] is None
    assert inner["status"] == "ok"
    assert outer["status"] == "ok"


def test_event_records_have_required_shape() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    tracer.start_session(profile="profile", category="category")
    turn_context = tracer.start_turn(attributes={"mode": "test"})

    tracer.event("tool.called", "agent_control", attributes={"tool": "moveit_get_current_pose"})

    event = writer.records[-1]
    assert event["schema_version"] == 1
    assert event["record_type"] == "event"
    assert event["trace_id"] == turn_context.trace_id
    assert event["span_id"]
    assert event["parent_span_id"] is None
    assert event["session_id"] == turn_context.session_id
    assert event["turn_id"] == turn_context.turn_id
    assert event["name"] == "tool.called"
    assert event["module"] == "agent_control"
    assert isinstance(event["started_at_unix_ns"], int)
    assert event["ended_at_unix_ns"] == event["started_at_unix_ns"]
    assert event["duration_ms"] == 0
    assert event["status"] == "ok"
    assert event["attributes"] == {"tool": "moveit_get_current_pose"}
    assert event["events"] == []


def test_start_session_accepts_supplied_session_id() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)

    context = tracer.start_session(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        session_id="session-123",
    )

    session_start = writer.records[-1]
    assert context.session_id == "session-123"
    assert session_start["session_id"] == "session-123"
    assert session_start["name"] == "trace.session_start"


def test_jsonl_trace_writer_writes_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "trace.jsonl"
    writer = JsonlTraceWriter(path)
    record = {
        "schema_version": 1,
        "record_type": "event",
        "trace_id": "trace",
        "span_id": None,
        "parent_span_id": None,
        "session_id": "session",
        "turn_id": "turn",
        "name": "name",
        "module": "module",
        "started_at_unix_ns": 10,
        "ended_at_unix_ns": 10,
        "duration_ms": 0,
        "status": "ok",
        "attributes": {"text": "Mave fährt"},
        "events": [],
    }

    writer.write(record)

    assert json.loads(path.read_text(encoding="utf-8")) == record


def test_sanitizer_redacts_secret_auth_keys_and_converts_unsafe_values() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    tracer.event(
        "sanitize",
        "test",
        attributes={
            "api_key": "sk-test",
            "nested": {"Authorization": "Bearer abc", "safe": "ok"},
            "payload": b"\x00\x01\x02",
            "items": [bytearray(b"abc"), {1, 2}],
        },
    )

    attributes = writer.records[-1]["attributes"]
    assert attributes["api_key"] == "[REDACTED]"
    assert attributes["nested"] == {"Authorization": "[REDACTED]", "safe": "ok"}
    assert attributes["payload"] == "<bytes len=3>"
    assert attributes["items"][0] == "<bytearray len=3>"
    assert attributes["items"][1].startswith("{")
    json.dumps(writer.records[-1])


def test_sanitizer_redacts_hyphenated_api_key_headers() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    tracer.event(
        "sanitize",
        "test",
        attributes={"headers": {"X-API-Key": "sk-live", "Authorization": "bearer"}},
    )

    assert writer.records[-1]["attributes"]["headers"] == {
        "X-API-Key": "[REDACTED]",
        "Authorization": "[REDACTED]",
    }


def test_sync_exception_span_records_error() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)

    with pytest.raises(ValueError, match="boom"):
        with tracer.span("explode", "test"):
            raise ValueError("boom")

    span = writer.records[-1]
    assert span["record_type"] == "span"
    assert span["name"] == "explode"
    assert span["status"] == "error"
    assert span["attributes"]["error_type"] == "ValueError"
    assert span["attributes"]["error"] == "boom"


@pytest.mark.asyncio
async def test_async_cancelled_error_span_records_cancelled() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)

    with pytest.raises(asyncio.CancelledError):
        async with tracer.span("cancel", "test"):
            raise asyncio.CancelledError()

    span = writer.records[-1]
    assert span["record_type"] == "span"
    assert span["name"] == "cancel"
    assert span["status"] == "cancelled"


def test_writer_disables_after_one_write_failure_without_raising(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    writer = JsonlTraceWriter(tmp_path / "trace.jsonl")
    calls = 0

    def fail_open(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise OSError("disk full")

    monkeypatch.setattr(Path, "open", fail_open)

    writer.write({"one": 1})
    writer.write({"two": 2})

    assert calls == 1


def test_process_tracer_suppresses_writer_failure_and_restores_span_context() -> None:
    class FailingWriter:
        def write(self, record: dict[str, object]) -> None:
            raise RuntimeError("writer unavailable")

    tracer = ProcessTracer(FailingWriter())
    parent_context = TraceContext(
        trace_id="trace",
        session_id="session",
        turn_id="turn",
        parent_span_id="parent",
    )

    with use_trace_context(parent_context):
        with tracer.span("span", "test"):
            assert current_trace_context().parent_span_id != parent_context.parent_span_id
        assert current_trace_context() == parent_context

        tracer.event("event", "test")
        tracer.record_span("manual", "test")
        assert current_trace_context() == parent_context


def test_noop_process_tracer_writes_nothing() -> None:
    tracer = NoopProcessTracer()
    assert tracer.start_session(profile="profile", category="category") == TraceContext()
    assert tracer.start_turn(input_text="secret") == TraceContext()
    assert tracer.current_context() == TraceContext()

    with tracer.span("span", "test"):
        pass
    tracer.record_span("manual", "test")
    tracer.event("event", "test")


def test_noop_process_tracer_exposes_default_options() -> None:
    options = NoopProcessTracer().options

    assert options.include_text is True
    assert options.include_tool_payloads is True


def test_event_records_have_truthy_span_id() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)

    tracer.event("instant", "test")

    assert writer.records[-1]["span_id"]


def test_noop_record_methods_return_stable_record_shape() -> None:
    tracer = NoopProcessTracer()

    event = tracer.event("event", "test")
    span = tracer.record_span("span", "test")

    assert event["record_type"] == "event"
    assert event["status"] == "ok"
    assert span["record_type"] == "span"
    assert span["status"] == "ok"
