from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class TraceContext:
    trace_id: str | None = None
    session_id: str | None = None
    turn_id: str | None = None
    parent_span_id: str | None = None


_CURRENT_TRACE_CONTEXT: ContextVar[TraceContext | None] = ContextVar(
    "process_trace_current_context", default=None
)


def current_trace_context() -> TraceContext:
    return _CURRENT_TRACE_CONTEXT.get() or TraceContext()


@contextmanager
def use_trace_context(context: TraceContext) -> Iterator[TraceContext]:
    token = _CURRENT_TRACE_CONTEXT.set(context)
    try:
        yield context
    finally:
        _CURRENT_TRACE_CONTEXT.reset(token)


def _set_trace_context(context: TraceContext):
    return _CURRENT_TRACE_CONTEXT.set(context)


def _reset_trace_context(token: object) -> None:
    _CURRENT_TRACE_CONTEXT.reset(token)


def _active_trace_context() -> TraceContext | None:
    return _CURRENT_TRACE_CONTEXT.get()
