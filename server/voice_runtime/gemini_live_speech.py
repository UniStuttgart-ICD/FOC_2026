from __future__ import annotations

import re
from collections.abc import Callable

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_GEMINI_LIVE_VOICE = "Kore"

_SENTENCE_BOUNDARY = re.compile(r"(.+?[.!?])(\s+|$)", re.DOTALL)


def build_strict_speech_prompt(*, transcript: str, instructions: str | None) -> str:
    delivery = instructions or "Use natural, warm delivery."
    return "\n".join(
        [
            "You are only a speech renderer.",
            "Speak the transcript exactly.",
            "Do not add, remove, summarize, or rephrase words.",
            "Treat bracketed tags like [laughs], [sighs], and [whispers] as delivery cues.",
            f"Delivery instructions: {delivery}",
            "TRANSCRIPT TO SPEAK EXACTLY:",
            transcript,
        ]
    )


def pop_speakable_segments(buffer: str, *, flush: bool = False) -> tuple[list[str], str]:
    segments: list[str] = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        segments.append(match.group(1).strip())
        position = match.end(1)
    tail = buffer[position:]
    if flush and tail.strip():
        segments.append(tail.strip())
        tail = ""
    return segments, tail


class GeminiLiveSpeechRendererService(FrameProcessor):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GEMINI_LIVE_MODEL,
        voice: str = DEFAULT_GEMINI_LIVE_VOICE,
        instructions: str | None = None,
        client_factory: Callable[..., object] | None = None,
        connect_on_start: bool = True,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._client_factory = client_factory
        self._connect_on_start = connect_on_start
        self._text_buffer = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._text_buffer += frame.text
            segments, self._text_buffer = pop_speakable_segments(self._text_buffer)
            for segment in segments:
                await self._speak_segment(segment)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            segments, self._text_buffer = pop_speakable_segments(
                self._text_buffer,
                flush=True,
            )
            for segment in segments:
                await self._speak_segment(segment)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _speak_segment(self, transcript: str) -> None:
        prompt = build_strict_speech_prompt(
            transcript=transcript,
            instructions=self.instructions,
        )
        await self._stream_prompt_audio(prompt)

    async def _stream_prompt_audio(self, prompt: str) -> None:
        raise NotImplementedError("Gemini Live streaming is implemented in the next task")
