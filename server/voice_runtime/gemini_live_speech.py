from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

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

from voice_modulation.stream_trace import (
    VOICE_STREAM_CHUNK_SEQ,
    VOICE_STREAM_SOURCE,
    VOICE_STREAM_UTTERANCE_ID,
    VoiceStreamTracerProtocol,
    pcm16_audio_metrics,
)

DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_GEMINI_LIVE_VOICE = "Kore"

_SENTENCE_BOUNDARY = re.compile(r"(.+?[.!?])(\s+|$)", re.DOTALL)


@dataclass(frozen=True)
class GeminiLiveAudioChunk:
    audio: bytes
    message_seq: int
    audio_part_seq: int
    audio_parts_in_message: int
    non_audio_parts_in_message: int
    generation_complete: bool
    turn_complete: bool


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


async def stream_gemini_live_audio(
    *,
    api_key: str,
    model: str,
    voice: str,
    prompt: str,
    client_factory: Callable[..., Any] | None = None,
) -> AsyncIterator[bytes]:
    factory = client_factory or genai.Client
    client = factory(api_key=api_key)
    async with client.aio.live.connect(
        model=model,
        config=_live_audio_config(voice),
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
        message_seq = 0
        async for message in session.receive():
            message_seq += 1
            audio_parts, non_audio_parts = _extract_audio_parts(message)
            complete = _message_completion(message)
            for audio_part_seq, audio in enumerate(audio_parts, start=1):
                yield GeminiLiveAudioChunk(
                    audio=audio,
                    message_seq=message_seq,
                    audio_part_seq=audio_part_seq,
                    audio_parts_in_message=len(audio_parts),
                    non_audio_parts_in_message=non_audio_parts,
                    generation_complete=complete["generation_complete"],
                    turn_complete=complete["turn_complete"],
                )
            if complete["turn_complete"]:
                break


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
        voice_stream_tracer: VoiceStreamTracerProtocol | None = None,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._client_factory = client_factory
        self._connect_on_start = connect_on_start
        self._text_buffer = ""
        self._voice_stream_tracer = voice_stream_tracer
        self._utterance_sequence = 0
        self._current_utterance_id: str | None = None
        self._segment_sequence = 0
        self._chunk_sequence = 0
        self._tts_started = False
        self._tts_stopped = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._reset_response_state(allocate_utterance=True)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._text_buffer += frame.text
            await self.push_frame(frame, direction)
            segments, self._text_buffer = pop_speakable_segments(self._text_buffer)
            for segment in segments:
                await self._speak_segment(segment, direction)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            segments, self._text_buffer = pop_speakable_segments(
                self._text_buffer,
                flush=True,
            )
            for segment in segments:
                await self._speak_segment(segment, direction)
            await self._finish_tts_lifecycle(direction)
            await self.push_frame(frame, direction)
            self._reset_response_state()
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._text_buffer = ""
            await self._finish_tts_lifecycle(direction)
            self._reset_response_state()
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _speak_segment(
        self,
        transcript: str,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        prompt = build_strict_speech_prompt(
            transcript=transcript,
            instructions=self.instructions,
        )
        try:
            await self._stream_segment_audio(prompt, direction)
        except BaseException:
            await self._finish_tts_lifecycle(direction)
            raise

    async def _stream_prompt_audio(
        self,
        prompt: str,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        standalone = self._current_utterance_id is None
        if standalone:
            self._reset_response_state(allocate_utterance=True)
        try:
            await self._stream_segment_audio(prompt, direction)
        except BaseException:
            await self._finish_tts_lifecycle(direction)
            raise
        else:
            if standalone:
                await self._finish_tts_lifecycle(direction)
        finally:
            if standalone:
                self._reset_response_state()

    async def _stream_segment_audio(
        self,
        prompt: str,
        direction: FrameDirection,
    ) -> None:
        utterance_id = self._ensure_response_utterance_id()
        self._segment_sequence += 1
        segment_seq = self._segment_sequence
        segment_chunk_sequence = 0
        self._trace(
            "gemini.segment_start",
            utterance_id=utterance_id,
            segment_seq=segment_seq,
            prompt_chars=len(prompt),
            model=self.model,
            voice=self.voice,
        )
        async for item in stream_gemini_live_audio(
            api_key=self.api_key,
            model=self.model,
            voice=self.voice,
            prompt=prompt,
            client_factory=self._client_factory,
        ):
            audio, chunk_trace = _audio_chunk_payload_and_trace(item)
            segment_chunk_sequence += 1
            self._chunk_sequence += 1
            chunk_sequence = self._chunk_sequence
            if not self._tts_started:
                await self._start_tts_lifecycle(direction)
            self._trace(
                "gemini.audio_chunk",
                utterance_id=utterance_id,
                segment_seq=segment_seq,
                segment_chunk_seq=segment_chunk_sequence,
                chunk_seq=chunk_sequence,
                source="gemini_live",
                **chunk_trace,
                **pcm16_audio_metrics(audio, sample_rate=24000, num_channels=1),
            )
            audio_frame = TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
            self._annotate_trace_frame(
                audio_frame,
                utterance_id=utterance_id,
                chunk_seq=chunk_sequence,
            )
            await self.push_frame(audio_frame, direction)
        if segment_chunk_sequence == 0:
            self._trace(
                "gemini.segment_end",
                utterance_id=utterance_id,
                segment_seq=segment_seq,
                chunks=0,
            )

    async def _start_tts_lifecycle(self, direction: FrameDirection) -> None:
        if self._tts_started:
            return
        utterance_id = self._ensure_response_utterance_id()
        self._tts_started = True
        self._tts_stopped = False
        start_frame = TTSStartedFrame()
        self._annotate_trace_frame(start_frame, utterance_id=utterance_id)
        self._trace("gemini.tts_start", utterance_id=utterance_id)
        await self.push_frame(start_frame, direction)

    async def _finish_tts_lifecycle(self, direction: FrameDirection) -> None:
        if not self._tts_started or self._tts_stopped:
            return
        utterance_id = self._ensure_response_utterance_id()
        self._tts_stopped = True
        stop_frame = TTSStoppedFrame()
        self._annotate_trace_frame(stop_frame, utterance_id=utterance_id)
        self._trace(
            "gemini.tts_stop",
            utterance_id=utterance_id,
            chunks=self._chunk_sequence,
            segments=self._segment_sequence,
        )
        await self.push_frame(stop_frame, direction)

    def _next_utterance_id(self) -> str:
        self._utterance_sequence += 1
        return f"gemini-live-{self._utterance_sequence:04d}"

    def _ensure_response_utterance_id(self) -> str:
        if self._current_utterance_id is None:
            self._current_utterance_id = self._next_utterance_id()
        return self._current_utterance_id

    def _annotate_trace_frame(
        self,
        frame: Frame,
        *,
        utterance_id: str,
        chunk_seq: int | None = None,
    ) -> None:
        frame.metadata[VOICE_STREAM_UTTERANCE_ID] = utterance_id
        frame.metadata[VOICE_STREAM_SOURCE] = "gemini_live"
        if chunk_seq is not None:
            frame.metadata[VOICE_STREAM_CHUNK_SEQ] = chunk_seq

    def _trace(self, event: str, **attributes: Any) -> None:
        if self._voice_stream_tracer is not None:
            self._voice_stream_tracer.event(event, **attributes)

    def _reset_response_state(self, *, allocate_utterance: bool = False) -> None:
        self._text_buffer = ""
        self._current_utterance_id = self._next_utterance_id() if allocate_utterance else None
        self._segment_sequence = 0
        self._chunk_sequence = 0
        self._tts_started = False
        self._tts_stopped = False

    def _client(self):
        factory = self._client_factory or genai.Client
        return factory(api_key=self.api_key)

    def _live_config(self):
        return _live_audio_config(self.voice)


def _live_audio_config(voice: str):
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )


def _audio_chunk_payload_and_trace(
    item: bytes | GeminiLiveAudioChunk,
) -> tuple[bytes, dict[str, Any]]:
    if isinstance(item, GeminiLiveAudioChunk):
        return item.audio, {
            "message_seq": item.message_seq,
            "audio_part_seq": item.audio_part_seq,
            "audio_parts_in_message": item.audio_parts_in_message,
            "non_audio_parts_in_message": item.non_audio_parts_in_message,
            "generation_complete": item.generation_complete,
            "turn_complete": item.turn_complete,
        }
    return item, {}


def _extract_audio_parts(message: object) -> tuple[list[bytes], int]:
    audio_parts: list[bytes] = []
    non_audio_parts = 0
    data = getattr(message, "data", None)
    if data:
        audio_parts.append(data)
    server_content = getattr(message, "server_content", None)
    model_turn = getattr(server_content, "model_turn", None)
    parts = getattr(model_turn, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        mime_type = getattr(inline_data, "mime_type", "") if inline_data is not None else ""
        if mime_type.startswith("audio/pcm"):
            audio = getattr(inline_data, "data", None)
            if audio:
                audio_parts.append(audio)
                continue
        non_audio_parts += 1
    return audio_parts, non_audio_parts


def _message_completion(message: object) -> dict[str, bool]:
    server_content = getattr(message, "server_content", None)
    return {
        "turn_complete": bool(getattr(server_content, "turn_complete", False)),
        "generation_complete": bool(getattr(server_content, "generation_complete", False)),
    }
