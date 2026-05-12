from __future__ import annotations

import re
from collections.abc import Callable

from google import genai
from google.genai import types
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
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
            await self.push_frame(frame, direction)
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
        client = self._client()
        audio_started = False
        async with client.aio.live.connect(
            model=self.model,
            config=self._live_config(),
        ) as session:
            await session.send_client_content(
                turns=[
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    )
                ],
                turn_complete=True,
            )
            async for message in session.receive():
                audio = _extract_audio(message)
                if audio:
                    if not audio_started:
                        audio_started = True
                        await self.push_frame(TTSStartedFrame())
                    await self.push_frame(
                        TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
                    )
                if _message_is_complete(message):
                    break
        if audio_started:
            await self.push_frame(TTSStoppedFrame())

    def _client(self):
        factory = self._client_factory or genai.Client
        return factory(api_key=self.api_key)

    def _live_config(self):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                )
            ),
        )


def _extract_audio(message: object) -> bytes | None:
    data = getattr(message, "data", None)
    if data:
        return data
    server_content = getattr(message, "server_content", None)
    model_turn = getattr(server_content, "model_turn", None)
    parts = getattr(model_turn, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        mime_type = getattr(inline_data, "mime_type", "")
        if mime_type.startswith("audio/pcm"):
            audio = getattr(inline_data, "data", None)
            if audio:
                return audio
    return None


def _message_is_complete(message: object) -> bool:
    server_content = getattr(message, "server_content", None)
    return bool(
        getattr(server_content, "turn_complete", False)
        or getattr(server_content, "generation_complete", False)
    )
