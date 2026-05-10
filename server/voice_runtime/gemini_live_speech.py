from __future__ import annotations

import re
from collections.abc import Callable

from pipecat.processors.frame_processor import FrameProcessor

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
