from __future__ import annotations

from typing import Any, cast

import pytest
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_runtime.rtvi import (
    GeminiLiveConversationRTVIObserver,
    GeminiLiveConversationRTVIProcessor,
)


class CapturingRTVI:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def push_transport_message(self, model: Any, exclude_none: bool = True) -> None:
        self.messages.append(model.model_dump(exclude_none=exclude_none))


@pytest.mark.asyncio
async def test_gemini_live_observer_emits_bot_output_from_llm_text() -> None:
    rtvi = CapturingRTVI()
    observer = GeminiLiveConversationRTVIObserver(cast(Any, rtvi))

    await observer.on_push_frame(_pushed(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_pushed(LLMTextFrame("Hmmmmmm. ")))
    await observer.on_push_frame(_pushed(LLMTextFrame("A tower, yes.")))
    await observer.on_push_frame(_pushed(LLMFullResponseEndFrame()))

    outputs = [message for message in rtvi.messages if message["type"] == "bot-output"]

    assert outputs == [
        {
            "label": "rtvi-ai",
            "type": "bot-output",
            "data": {
                "text": "Hmmmmmm. A tower, yes.",
                "spoken": True,
                "aggregated_by": "sentence",
            },
        }
    ]


@pytest.mark.asyncio
async def test_gemini_live_observer_ignores_empty_bot_output() -> None:
    rtvi = CapturingRTVI()
    observer = GeminiLiveConversationRTVIObserver(cast(Any, rtvi))

    await observer.on_push_frame(_pushed(LLMFullResponseStartFrame()))
    await observer.on_push_frame(_pushed(LLMTextFrame("   ")))
    await observer.on_push_frame(_pushed(LLMFullResponseEndFrame()))

    assert [message for message in rtvi.messages if message["type"] == "bot-output"] == []


def test_gemini_live_processor_creates_conversation_observer() -> None:
    processor = GeminiLiveConversationRTVIProcessor()

    observer = processor.create_rtvi_observer()

    assert isinstance(observer, GeminiLiveConversationRTVIObserver)


def _pushed(frame: Any) -> FramePushed:
    return FramePushed(
        source=FrameProcessor(),
        destination=FrameProcessor(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )
