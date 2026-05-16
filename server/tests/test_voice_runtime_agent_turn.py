from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping
from typing import Any, cast

import pytest
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from process_trace import MemoryTraceWriter, ProcessTracer, TraceContext, TraceOptions
from process_trace.context import use_trace_context
from voice_runtime.agent_turn import (
    AgentBackend,
    AgentTurnInput,
    AgentTurnProcessor,
    agent_turn_input,
    is_actionable_user_text,
    latest_user_text,
)
from voice_runtime.response_coordination import BotResponseCoordinator


class EchoBackend:
    def __init__(self, chunks: list[str] | None = None, *, raises: bool = False) -> None:
        self.connected = False
        self.disconnected = False
        self.turns: list[AgentTurnInput] = []
        self.chunks = chunks if chunks is not None else []
        self.raises = raises

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        self.turns.append(turn)
        if self.raises:
            raise RuntimeError("boom")
        for chunk in self.chunks:
            yield chunk


class NotifyingBackend(EchoBackend):
    def __init__(self) -> None:
        super().__init__()
        self.notifications_started = asyncio.Event()
        self.notification_queue: asyncio.Queue[str] = asyncio.Queue()

    async def notifications(self) -> AsyncIterator[str]:
        self.notifications_started.set()
        while True:
            yield await self.notification_queue.get()


class CapturingProcessor(AgentTurnProcessor):
    def __init__(self, backend: AgentBackend, **kwargs: Any) -> None:
        super().__init__(backend=backend, **kwargs)
        self.pushed: list[Frame] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append(frame)


def _context_frame(messages: list[Mapping[str, Any]]) -> LLMContextFrame:
    return LLMContextFrame(context=LLMContext(messages=cast(Any, messages)))


def test_latest_user_text_reads_string_and_text_parts() -> None:
    assert latest_user_text(_context_frame([{"role": "user", "content": "move up"}])) == "move up"
    assert (
        latest_user_text(
            _context_frame([{"role": "user", "content": [{"type": "text", "text": "status"}]}])
        )
        == "status"
    )


def test_latest_user_text_reads_latest_user_and_ignores_empty_content() -> None:
    frame = _context_frame(
        [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "reply"},
            {"role": "user", "content": "  "},
            {"role": "user", "content": [{"type": "image", "text": "ignored"}, {"type": "text", "text": "again"}]}, 
        ]
    )

    assert latest_user_text(frame) == "again"


def test_agent_turn_input_copies_context_messages() -> None:
    message = {"role": "user", "content": "move up"}
    frame = _context_frame([message])

    turn = agent_turn_input(frame)

    assert turn == AgentTurnInput(user_text="move up", messages=[message])
    assert turn is not None
    assert turn.messages is not frame.context.messages


def test_agent_turn_input_returns_none_without_user_text() -> None:
    assert agent_turn_input(_context_frame([{"role": "assistant", "content": "hello"}])) is None


@pytest.mark.parametrize("text", ["Mave", "Maeve", "May", "", "  "])
def test_wake_only_or_likely_wake_false_positive_text_is_not_actionable(text: str) -> None:
    assert is_actionable_user_text(text) is False


@pytest.mark.parametrize(
    "text",
    [
        "Names,",
        "Name.",
        "Mail.",
        "Nave.",
        "Base.",
        "up the robot wave.",
    ],
)
def test_is_actionable_user_text_rejects_live_run_wake_junk(text: str) -> None:
    assert is_actionable_user_text(text) is False


def test_is_actionable_user_text_keeps_command_after_wake_variant_cleanup() -> None:
    assert is_actionable_user_text("move robot up.") is True


@pytest.mark.parametrize("text", ["stop", "move up", "what can you do?"])
def test_command_text_is_actionable(text: str) -> None:
    assert is_actionable_user_text(text) is True


@pytest.mark.parametrize("text", ["Mave", "Maeve", "May"])
def test_agent_turn_input_returns_none_for_wake_only_false_positive_text(text: str) -> None:
    assert agent_turn_input(_context_frame([{"role": "user", "content": text}])) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["Mave", "Maeve", "May"])
async def test_agent_turn_does_not_call_backend_for_wake_only_false_positive_text(
    text: str,
) -> None:
    backend = EchoBackend(["unused"])
    processor = CapturingProcessor(backend)
    frame = _context_frame([{"role": "user", "content": text}])

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [frame]
    assert backend.turns == []


@pytest.mark.asyncio
async def test_agent_turn_wraps_backend_output_in_llm_frames() -> None:
    backend = EchoBackend(["echo: move up"])
    processor = CapturingProcessor(backend)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
    )

    assert [type(frame) for frame in processor.pushed] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert isinstance(processor.pushed[1], LLMTextFrame)
    assert processor.pushed[1].text == "echo: move up"
    assert [turn.user_text for turn in backend.turns] == ["move up"]
    assert backend.turns[0].messages == [{"role": "user", "content": "move up"}]


@pytest.mark.asyncio
async def test_agent_turn_pumps_backend_notifications_after_connect() -> None:
    backend = NotifyingBackend()
    processor = CapturingProcessor(backend)

    await processor.connect()
    await asyncio.wait_for(backend.notifications_started.wait(), timeout=1)
    await backend.notification_queue.put("position reached")
    await asyncio.wait_for(_wait_for_pushed_frames(processor, 3), timeout=1)

    assert [type(frame) for frame in processor.pushed] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert isinstance(processor.pushed[1], LLMTextFrame)
    assert processor.pushed[1].text == "position reached"

    await processor.disconnect()


@pytest.mark.asyncio
async def test_agent_turn_waits_for_response_coordinator_before_next_turn() -> None:
    backend = EchoBackend(["done"])
    coordinator = BotResponseCoordinator()
    processor = CapturingProcessor(backend, response_coordinator=coordinator)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]),
        FrameDirection.DOWNSTREAM,
    )
    assert coordinator.is_response_active is True

    second_turn = asyncio.create_task(
        processor.process_frame(
            _context_frame([{"role": "user", "content": "move down"}]),
            FrameDirection.DOWNSTREAM,
        )
    )
    await asyncio.sleep(0)

    assert [turn.user_text for turn in backend.turns] == ["move up"]

    coordinator.finish_response()
    await asyncio.wait_for(second_turn, timeout=1)

    assert [turn.user_text for turn in backend.turns] == ["move up", "move down"]


@pytest.mark.asyncio
async def test_notifications_wait_for_response_coordinator() -> None:
    backend = NotifyingBackend()
    coordinator = BotResponseCoordinator()
    processor = CapturingProcessor(backend, response_coordinator=coordinator)
    await coordinator.begin_response()

    await processor.connect()
    await asyncio.wait_for(backend.notifications_started.wait(), timeout=1)
    await backend.notification_queue.put("position reached")
    await asyncio.sleep(0)

    assert processor.pushed == []

    coordinator.finish_response()
    await asyncio.wait_for(_wait_for_pushed_frames(processor, 3), timeout=1)

    assert isinstance(processor.pushed[1], LLMTextFrame)
    assert processor.pushed[1].text == "position reached"

    await processor.disconnect()


@pytest.mark.asyncio
async def test_agent_turn_processor_emits_voice_agent_turn_on_successful_backend() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = CapturingProcessor(EchoBackend(["done"]), tracer=tracer)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
    )

    agent_span = [record for record in writer.records if record["name"] == "voice.agent_turn"][0]
    response = [
        record for record in writer.records if record["name"] == "voice.agent_turn.response"
    ][0]
    assert agent_span["record_type"] == "span"
    assert agent_span["status"] == "ok"
    assert agent_span["turn_id"]
    assert response["record_type"] == "event"
    assert response["attributes"] == {"text": "done"}


@pytest.mark.asyncio
async def test_agent_turn_processor_reuses_current_trace_turn_context() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = CapturingProcessor(EchoBackend(["done"]), tracer=tracer)
    turn_context = TraceContext(trace_id="trace", session_id="session", turn_id="turn")

    with use_trace_context(turn_context):
        await processor.process_frame(
            _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
        )

    agent_span = [record for record in writer.records if record["name"] == "voice.agent_turn"][0]
    assert agent_span["trace_id"] == "trace"
    assert agent_span["session_id"] == "session"
    assert agent_span["turn_id"] == "turn"
    assert [record["name"] for record in writer.records].count("trace.turn_start") == 0


@pytest.mark.asyncio
async def test_agent_turn_processor_reuses_tracer_owned_observer_turn_context() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = CapturingProcessor(EchoBackend(["done"]), tracer=tracer)
    session_context = tracer.start_session("test", "local_debug")
    turn_context = tracer.start_turn(context=session_context)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
    )

    agent_span = [record for record in writer.records if record["name"] == "voice.agent_turn"][0]
    turn_starts = [record for record in writer.records if record["name"] == "trace.turn_start"]
    assert agent_span["turn_id"] == turn_context.turn_id
    assert len(turn_starts) == 1


@pytest.mark.asyncio
async def test_agent_turn_processor_starts_new_turn_for_independent_context_frames() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = CapturingProcessor(EchoBackend(["done"]), tracer=tracer)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
    )
    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move down"}]), FrameDirection.DOWNSTREAM
    )

    turn_starts = [record for record in writer.records if record["name"] == "trace.turn_start"]
    turn_ids = [record["turn_id"] for record in turn_starts]
    assert len(turn_starts) == 2
    assert len(set(turn_ids)) == 2


@pytest.mark.asyncio
async def test_agent_turn_processor_omits_response_text_when_include_text_false() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer, TraceOptions(include_text=False))
    processor = CapturingProcessor(EchoBackend(["done"]), tracer=tracer)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM
    )

    assert not [record for record in writer.records if record["name"] == "voice.agent_turn.response"]


@pytest.mark.asyncio
async def test_agent_turn_emits_fallback_when_backend_yields_no_text() -> None:
    backend = EchoBackend([])
    processor = CapturingProcessor(backend)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "status"}]), FrameDirection.DOWNSTREAM
    )

    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    assert [frame.text for frame in text_frames] == ["I could not confirm that the action completed."]


@pytest.mark.asyncio
async def test_agent_turn_emits_error_message_when_backend_raises() -> None:
    backend = EchoBackend(raises=True)
    processor = CapturingProcessor(backend)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "status"}]), FrameDirection.DOWNSTREAM
    )

    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    assert [frame.text for frame in text_frames] == ["I encountered an error. Please try again."]


@pytest.mark.asyncio
async def test_agent_turn_processor_records_span_around_backend_failure() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = CapturingProcessor(EchoBackend(raises=True), tracer=tracer)

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "status"}]), FrameDirection.DOWNSTREAM
    )

    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    agent_span = [record for record in writer.records if record["name"] == "voice.agent_turn"][0]
    response = [
        record for record in writer.records if record["name"] == "voice.agent_turn.response"
    ][0]
    assert [frame.text for frame in text_frames] == ["I encountered an error. Please try again."]
    assert agent_span["status"] == "error"
    assert agent_span["attributes"]["error_type"] == "RuntimeError"
    assert response["attributes"] == {"text": "I encountered an error. Please try again."}


@pytest.mark.asyncio
async def test_agent_turn_processor_calls_lifecycle_callbacks() -> None:
    events: list[str] = []
    processor = CapturingProcessor(
        EchoBackend(["done"]),
        on_turn_started=lambda: events.append("started"),
        on_turn_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move robot up."}]),
        FrameDirection.DOWNSTREAM,
    )

    assert events == ["started", "finished"]


@pytest.mark.asyncio
async def test_agent_turn_processor_finishes_lifecycle_after_backend_error() -> None:
    events: list[str] = []
    processor = CapturingProcessor(
        EchoBackend(raises=True),
        on_turn_started=lambda: events.append("started"),
        on_turn_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(
        _context_frame([{"role": "user", "content": "move robot up."}]),
        FrameDirection.DOWNSTREAM,
    )

    assert events == ["started", "finished"]


@pytest.mark.asyncio
async def test_agent_turn_forwards_non_user_context_frame() -> None:
    backend = EchoBackend(["unused"])
    processor = CapturingProcessor(backend)
    frame = _context_frame([{"role": "assistant", "content": "hello"}])

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [frame]
    assert backend.turns == []


@pytest.mark.asyncio
async def test_agent_turn_disconnects_backend_on_cancel() -> None:
    backend = EchoBackend()
    processor = CapturingProcessor(backend)

    await processor.connect()
    await processor.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)

    assert backend.connected is True
    assert backend.disconnected is True
    assert isinstance(processor.pushed[0], CancelFrame)


@pytest.mark.asyncio
async def test_agent_turn_disconnects_backend_on_end() -> None:
    backend = EchoBackend()
    processor = CapturingProcessor(backend)

    await processor.connect()
    await processor.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert backend.connected is True
    assert backend.disconnected is True
    assert isinstance(processor.pushed[0], EndFrame)


async def _wait_for_pushed_frames(processor: CapturingProcessor, count: int) -> None:
    while len(processor.pushed) < count:
        await asyncio.sleep(0)
