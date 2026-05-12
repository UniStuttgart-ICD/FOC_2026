from __future__ import annotations

from typing import Any

import pipecat.processors.frameworks.rtvi.models as RTVI
from pipecat.frames.frames import LLMFullResponseEndFrame, LLMFullResponseStartFrame, LLMTextFrame
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.processors.frameworks.rtvi import RTVIObserver, RTVIObserverParams, RTVIProcessor


class GeminiLiveConversationRTVIObserver(RTVIObserver):
    """Emit client-visible bot output for the Gemini Live speech renderer."""

    def __init__(
        self,
        rtvi: RTVIProcessor | None = None,
        *,
        params: RTVIObserverParams | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(rtvi, params=params, **kwargs)
        self._conversation_text = ""
        self._conversation_collecting = False
        self._conversation_seen_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        await super().on_push_frame(data)

        if self._should_skip_conversation_output(data):
            return

        frame = data.frame
        self._conversation_seen_frame_ids.add(frame.id)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._conversation_text = ""
            self._conversation_collecting = True
            return

        if isinstance(frame, LLMTextFrame) and self._conversation_collecting:
            self._conversation_text += frame.text
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            await self._send_conversation_output()

    def _should_skip_conversation_output(self, data: FramePushed) -> bool:
        if self._ignored_sources and data.source in self._ignored_sources:
            return True
        if (
            data.frame.broadcast_sibling_id is not None
            and data.direction != FrameDirection.DOWNSTREAM
        ):
            return True
        if data.frame.id in self._conversation_seen_frame_ids:
            return True
        return not isinstance(
            data.frame,
            (LLMFullResponseStartFrame, LLMTextFrame, LLMFullResponseEndFrame),
        )

    async def _send_conversation_output(self) -> None:
        text = self._conversation_text.strip()
        self._conversation_text = ""
        self._conversation_collecting = False
        if not text:
            return

        await self.send_rtvi_message(
            RTVI.BotOutputMessage(
                data=RTVI.BotOutputMessageData(
                    text=text,
                    spoken=True,
                    aggregated_by="sentence",
                )
            )
        )


class GeminiLiveConversationRTVIProcessor(RTVIProcessor):
    def create_rtvi_observer(
        self,
        *,
        params: RTVIObserverParams | None = None,
        **kwargs: Any,
    ) -> GeminiLiveConversationRTVIObserver:
        return GeminiLiveConversationRTVIObserver(self, params=params, **kwargs)
