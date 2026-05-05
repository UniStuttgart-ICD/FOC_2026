from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Mapping
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

logger = logging.getLogger(__name__)

NO_TEXT_RESPONSE = "I could not confirm that the action completed."
ERROR_RESPONSE = "I encountered an error. Please try again."


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
    if not user_text:
        return None

    messages = frame.context.messages if frame.context else []
    mapping_messages: list[Mapping[str, Any]] = []
    for msg in messages:
        if isinstance(msg, Mapping):
            mapping_messages.append(cast(Mapping[str, Any], msg))
    return AgentTurnInput(user_text=user_text, messages=mapping_messages)


class AgentTurnProcessor(FrameProcessor):
    def __init__(self, *, backend: AgentBackend, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._backend = backend

    async def connect(self) -> None:
        await self._backend.connect()

    async def disconnect(self) -> None:
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

        await self.push_frame(LLMFullResponseStartFrame())
        await self._run_turn(turn)
        await self.push_frame(LLMFullResponseEndFrame())

    async def _run_turn(self, turn: AgentTurnInput) -> None:
        has_text = False
        try:
            async for chunk in self._backend.run_turn(turn):
                if not chunk:
                    continue
                has_text = True
                await self.push_frame(LLMTextFrame(text=chunk))
        except Exception:
            logger.exception("Agent backend turn failed")
            await self.push_frame(LLMTextFrame(text=ERROR_RESPONSE))
            return

        if not has_text:
            await self.push_frame(LLMTextFrame(text=NO_TEXT_RESPONSE))


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
