from __future__ import annotations

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

from voice_runtime.agent_turn import (
    AgentBackend,
    AgentTurnInput,
    AgentTurnProcessor,
    agent_turn_input,
    is_actionable_user_text,
    latest_user_text,
)


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
