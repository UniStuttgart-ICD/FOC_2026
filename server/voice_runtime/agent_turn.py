from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Protocol, cast

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from process_trace import (
    NoopProcessTracer,
    ProcessTracer,
    TraceContext,
    current_trace_context,
    use_trace_context,
)

logger = logging.getLogger(__name__)

NO_TEXT_RESPONSE = "I could not confirm that the action completed."
ERROR_RESPONSE = "I encountered an error. Please try again."
_WAKE_ONLY_TEXT = {"mave", "maeve", "may", "mail", "nave", "name", "names", "base"}
_PROBABLE_WAKE_JUNK_TEXT = {"up the robot wave"}
_WORD_PATTERN = re.compile(r"[a-z]+", re.IGNORECASE)


@dataclass(frozen=True)
class AgentTurnInput:
    user_text: str
    messages: list[Mapping[str, Any]]


class AgentBackend(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]: ...


def latest_user_text(frame: LLMContextFrame) -> str | None:
    messages = frame.context.messages if frame.context else []
    for msg in reversed(messages):
        if not isinstance(msg, Mapping) or msg.get("role") != "user":
            continue
        text = _message_text(msg)
        if text:
            return text
    return None


def agent_turn_input(frame: LLMContextFrame) -> AgentTurnInput | None:
    user_text = latest_user_text(frame)
    if not user_text or not is_actionable_user_text(user_text):
        return None

    messages = frame.context.messages if frame.context else []
    mapping_messages: list[Mapping[str, Any]] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            mapping_messages.append(cast(Mapping[str, Any], msg))
    return AgentTurnInput(user_text=user_text, messages=mapping_messages)


def is_actionable_user_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False

    words = [word.lower() for word in _WORD_PATTERN.findall(stripped)]
    if len(words) == 1 and words[0] in _WAKE_ONLY_TEXT:
        return False

    normalized = " ".join(words)
    if normalized in _PROBABLE_WAKE_JUNK_TEXT:
        return False

    return True


class AgentTurnProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        backend: AgentBackend,
        tracer: ProcessTracer | NoopProcessTracer | None = None,
        on_turn_started: Callable[[], None] | None = None,
        on_turn_finished: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._backend = backend
        self._tracer = tracer or NoopProcessTracer()
        self._on_turn_started = on_turn_started
        self._on_turn_finished = on_turn_finished
        self._last_agent_turn_id: str | None = None
        self._notification_task: asyncio.Task[None] | None = None

    async def connect(self) -> None:
        await self._backend.connect()
        notifications = getattr(self._backend, "notifications", None)
        if callable(notifications):
            self._notification_task = asyncio.create_task(self._pump_notifications(notifications))

    async def disconnect(self) -> None:
        if self._notification_task is not None:
            self._notification_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._notification_task
            self._notification_task = None
        await self._backend.disconnect()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.disconnect()
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return

        turn = agent_turn_input(frame)
        if turn is None:
            await self.push_frame(frame, direction)
            return

        if self._on_turn_started is not None:
            self._on_turn_started()
        try:
            await self.push_frame(LLMFullResponseStartFrame())
            turn_context = self._trace_turn_context(turn)
            with use_trace_context(turn_context):
                response_text = ""
                try:
                    async with self._tracer.span(
                        "voice.agent_turn",
                        "voice_runtime",
                        context=turn_context,
                    ):
                        response_text = await self._run_turn(turn)
                except Exception:
                    logger.exception("Agent backend turn failed")
                    response_text = ERROR_RESPONSE
                    await self.push_frame(LLMTextFrame(text=ERROR_RESPONSE))

                if self._tracer.options.include_text:
                    self._tracer.event(
                        "voice.agent_turn.response",
                        "voice_runtime",
                        attributes={"text": response_text},
                        context=turn_context,
                    )
            await self.push_frame(LLMFullResponseEndFrame())
        finally:
            if self._on_turn_finished is not None:
                self._on_turn_finished()

    def _trace_turn_context(self, turn: AgentTurnInput) -> TraceContext:
        active_context = current_trace_context()
        if active_context.turn_id is not None:
            self._last_agent_turn_id = active_context.turn_id
            return active_context
        tracer_context = self._tracer.current_context()
        if (
            tracer_context.turn_id is not None
            and tracer_context.turn_id != self._last_agent_turn_id
        ):
            self._last_agent_turn_id = tracer_context.turn_id
            return tracer_context
        turn_context = self._tracer.start_turn(input_text=turn.user_text, context=active_context)
        self._last_agent_turn_id = turn_context.turn_id
        return turn_context

    async def _run_turn(self, turn: AgentTurnInput) -> str:
        has_text = False
        response_parts: list[str] = []
        async for chunk in self._backend.run_turn(turn):
            if not chunk:
                continue
            has_text = True
            response_parts.append(chunk)
            await self.push_frame(LLMTextFrame(text=chunk))

        if not has_text:
            await self.push_frame(LLMTextFrame(text=NO_TEXT_RESPONSE))
            return NO_TEXT_RESPONSE

        return "".join(response_parts)

    async def _pump_notifications(self, notifications: Callable[[], AsyncIterator[str]]) -> None:
        try:
            async for text in notifications():
                await self.push_frame(LLMFullResponseStartFrame())
                await self.push_frame(LLMTextFrame(text=text))
                await self.push_frame(LLMFullResponseEndFrame())
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Agent backend notification pump failed")


def _message_text(msg: Mapping[str, Any]) -> str | None:
    content = msg.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if not isinstance(part, Mapping) or part.get("type") != "text":
                continue
            text = part.get("text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        if parts:
            return "\n".join(parts)

    return None
