from __future__ import annotations

from collections.abc import Callable

from google import genai
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
from voice_runtime.gemini_live_audio import (
    DEFAULT_GEMINI_LIVE_MODEL,
    DEFAULT_GEMINI_LIVE_VOICE,
    _live_audio_config,
    build_strict_speech_prompt,
    pop_speakable_segments,
    stream_gemini_live_audio,
)


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
        audio_started = False
        async for audio in stream_gemini_live_audio(
            api_key=self.api_key,
            model=self.model,
            voice=self.voice,
            prompt=prompt,
            client_factory=self._client_factory,
        ):
            if not audio_started:
                audio_started = True
                await self.push_frame(TTSStartedFrame())
            await self.push_frame(TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1))
        if audio_started:
            await self.push_frame(TTSStoppedFrame())

    def _client(self):
        factory = self._client_factory or genai.Client
        return factory(api_key=self.api_key)

    def _live_config(self):
        return _live_audio_config(self.voice)
