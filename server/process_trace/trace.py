from __future__ import annotations

import asyncio
import logging
import time
import uuid
from contextvars import Token
from dataclasses import dataclass
from types import TracebackType
from typing import Any

from process_trace.context import (
    TraceContext,
    _active_trace_context,
    _reset_trace_context,
    _set_trace_context,
    current_trace_context,
)
from process_trace.records import TraceOptions, TraceWriter, sanitize_attributes

LOGGER = logging.getLogger(__name__)


def _new_id() -> str:
    return uuid.uuid4().hex


def _duration_ms(started_at_unix_ns: int, ended_at_unix_ns: int) -> float:
    return round((ended_at_unix_ns - started_at_unix_ns) / 1_000_000, 3)


class ProcessTracer:
    def __init__(self, writer: TraceWriter, options: TraceOptions | None = None) -> None:
        self._writer = writer
        self._options = options or TraceOptions()
        self._current_session_context: TraceContext | None = None
        self._current_turn_context: TraceContext | None = None

    @property
    def options(self) -> TraceOptions:
        return self._options

    def start_session(
        self,
        profile: str,
        category: str,
        *,
        session_id: str | None = None,
    ) -> TraceContext:
        context = TraceContext(trace_id=_new_id(), session_id=session_id or _new_id())
        self._current_session_context = context
        self._current_turn_context = None
        self.event(
            "trace.session_start",
            "process_trace",
            attributes={"profile": profile, "category": category},
            context=context,
        )
        return context

    def start_turn(
        self,
        input_text: str | None = None,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        base_context = context if context is not None else current_trace_context()
        if base_context.trace_id is None and self._current_session_context is not None:
            base_context = self._current_session_context
        turn_context = TraceContext(
            trace_id=base_context.trace_id or _new_id(),
            session_id=base_context.session_id or _new_id(),
            turn_id=_new_id(),
            parent_span_id=base_context.parent_span_id,
        )
        self._current_turn_context = turn_context
        turn_attributes = dict(attributes or {})
        if self._options.include_text and input_text is not None:
            turn_attributes["input_text"] = input_text
        self.event(
            "trace.turn_start",
            "process_trace",
            attributes=turn_attributes,
            context=turn_context,
        )
        return turn_context

    def current_context(self) -> TraceContext:
        active_context = _active_trace_context()
        if active_context is not None:
            return active_context
        if self._current_turn_context is not None:
            return self._current_turn_context
        if self._current_session_context is not None:
            return self._current_session_context
        return TraceContext()

    def span(
        self,
        name: str,
        module: str,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> "_TraceSpan":
        return _TraceSpan(self, name, module, attributes, context)

    def record_span(
        self,
        name: str,
        module: str,
        *,
        started_at_unix_ns: int | None = None,
        ended_at_unix_ns: int | None = None,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        start_ns = started_at_unix_ns if started_at_unix_ns is not None else time.time_ns()
        end_ns = ended_at_unix_ns if ended_at_unix_ns is not None else time.time_ns()
        trace_context = context or self.current_context()
        record = self._make_record(
            record_type="span",
            trace_context=trace_context,
            name=name,
            module=module,
            started_at_unix_ns=start_ns,
            ended_at_unix_ns=end_ns,
            status=status,
            attributes=attributes,
            span_id=span_id or _new_id(),
            parent_span_id=trace_context.parent_span_id,
        )
        self._write_record(record)
        return record

    def event(
        self,
        name: str,
        module: str,
        *,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
        status: str = "ok",
    ) -> dict[str, Any]:
        now_ns = time.time_ns()
        trace_context = context or self.current_context()
        record = self._make_record(
            record_type="event",
            trace_context=trace_context,
            name=name,
            module=module,
            started_at_unix_ns=now_ns,
            ended_at_unix_ns=now_ns,
            status=status,
            attributes=attributes,
            span_id=_new_id(),
            parent_span_id=trace_context.parent_span_id,
        )
        self._write_record(record)
        return record

    def _write_record(self, record: dict[str, Any]) -> None:
        try:
            self._writer.write(record)
        except Exception as exc:
            LOGGER.warning("Suppressing process trace writer failure: %s", exc)

    def _make_record(
        self,
        *,
        record_type: str,
        trace_context: TraceContext,
        name: str,
        module: str,
        started_at_unix_ns: int,
        ended_at_unix_ns: int,
        status: str,
        attributes: dict[str, Any] | None,
        span_id: str | None,
        parent_span_id: str | None,
    ) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "record_type": record_type,
            "trace_id": trace_context.trace_id,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "session_id": trace_context.session_id,
            "turn_id": trace_context.turn_id,
            "name": name,
            "module": module,
            "started_at_unix_ns": started_at_unix_ns,
            "ended_at_unix_ns": ended_at_unix_ns,
            "duration_ms": _duration_ms(started_at_unix_ns, ended_at_unix_ns),
            "status": status,
            "attributes": sanitize_attributes(attributes),
            "events": [],
        }


@dataclass
class _TraceSpan:
    _tracer: ProcessTracer
    _name: str
    _module: str
    _attributes: dict[str, Any] | None
    _context: TraceContext | None
    _span_id: str | None = None
    _started_at_unix_ns: int | None = None
    _token: Token[TraceContext | None] | None = None

    def __enter__(self) -> "_TraceSpan":
        parent_context = self._context or self._tracer.current_context()
        self._span_id = _new_id()
        self._started_at_unix_ns = time.time_ns()
        span_context = TraceContext(
            trace_id=parent_context.trace_id,
            session_id=parent_context.session_id,
            turn_id=parent_context.turn_id,
            parent_span_id=self._span_id,
        )
        self._context = parent_context
        self._token = _set_trace_context(span_context)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        ended_at_unix_ns = time.time_ns()
        attributes = dict(self._attributes or {})
        status = "ok"
        if exc is not None:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
            attributes["error_type"] = type(exc).__name__
            attributes["error"] = str(exc)
        try:
            self._tracer.record_span(
                self._name,
                self._module,
                started_at_unix_ns=self._started_at_unix_ns,
                ended_at_unix_ns=ended_at_unix_ns,
                status=status,
                attributes=attributes,
                context=self._context,
                span_id=self._span_id,
            )
        finally:
            if self._token is not None:
                _reset_trace_context(self._token)
        return False

    async def __aenter__(self) -> "_TraceSpan":
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return self.__exit__(exc_type, exc, traceback)


class NoopProcessTracer:
    def __init__(self, options: TraceOptions | None = None) -> None:
        self._options = options or TraceOptions()

    @property
    def options(self) -> TraceOptions:
        return self._options

    def start_session(
        self,
        profile: str,
        category: str,
        *,
        session_id: str | None = None,
    ) -> TraceContext:
        return TraceContext()

    def start_turn(
        self,
        input_text: str | None = None,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> TraceContext:
        return TraceContext()

    def current_context(self) -> TraceContext:
        return TraceContext()

    def span(
        self,
        name: str,
        module: str,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
    ) -> "_NoopSpan":
        return _NoopSpan()

    def record_span(
        self,
        name: str,
        module: str,
        *,
        started_at_unix_ns: int | None = None,
        ended_at_unix_ns: int | None = None,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
        span_id: str | None = None,
    ) -> dict[str, Any]:
        return _noop_record(
            record_type="span",
            name=name,
            module=module,
            status=status,
            attributes=attributes,
            span_id=span_id,
            context=context,
        )

    def event(
        self,
        name: str,
        module: str,
        *,
        attributes: dict[str, Any] | None = None,
        context: TraceContext | None = None,
        status: str = "ok",
    ) -> dict[str, Any]:
        return _noop_record(
            record_type="event",
            name=name,
            module=module,
            status=status,
            attributes=attributes,
            span_id=None,
            context=context,
        )


def _noop_record(
    *,
    record_type: str,
    name: str,
    module: str,
    status: str,
    attributes: dict[str, Any] | None,
    span_id: str | None,
    context: TraceContext | None,
) -> dict[str, Any]:
    trace_context = context or TraceContext()
    return {
        "schema_version": 1,
        "record_type": record_type,
        "trace_id": trace_context.trace_id,
        "span_id": span_id,
        "parent_span_id": trace_context.parent_span_id,
        "session_id": trace_context.session_id,
        "turn_id": trace_context.turn_id,
        "name": name,
        "module": module,
        "started_at_unix_ns": None,
        "ended_at_unix_ns": None,
        "duration_ms": 0,
        "status": status,
        "attributes": sanitize_attributes(attributes),
        "events": [],
    }


class _NoopSpan:
    def __enter__(self) -> "_NoopSpan":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False

    async def __aenter__(self) -> "_NoopSpan":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        return False
