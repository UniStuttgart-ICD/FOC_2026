from __future__ import annotations

import asyncio
import math
import time
from dataclasses import replace
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_modulation import dsp
from voice_modulation.dsp import VoiceModulationState
from voice_modulation.settings import VoiceModulationSettings
from voice_modulation.stream_trace import (
    VOICE_STREAM_CHUNK_SEQ,
    VOICE_STREAM_SOURCE,
    VOICE_STREAM_UTTERANCE_ID,
    VoiceStreamTracerProtocol,
    pcm16_audio_metrics,
)

_BLOCK_DURATION_S = 0.02
_TAIL_MAX_DURATION_S = 0.24
_TAIL_SILENCE_RMS = 0.0005
_TAIL_SILENT_BLOCKS = 2


class VoiceModulationProcessor(FrameProcessor):
    def __init__(
        self,
        *,
        settings: VoiceModulationSettings,
        voice_stream_tracer: VoiceStreamTracerProtocol | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._voice_stream_tracer = voice_stream_tracer
        self._dsp_state = VoiceModulationState()
        self._audio_buffer = bytearray()
        self._buffer_template: TTSAudioRawFrame | None = None
        self._tail_template: TTSAudioRawFrame | None = None
        self._stream_sample_rate: int | None = None
        self._stream_num_channels: int | None = None
        self._pending_start_frame: TTSStartedFrame | None = None
        self._startup_frames: list[tuple[TTSAudioRawFrame, str]] = []
        self._startup_released = False
        self._current_utterance_id: str | None = None
        self._generated_utterance_count = 0
        self._audio_frame_sequence = 0
        self._dsp_block_sequence = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            utterance_id = self._ensure_utterance_id(frame)
            self._audio_frame_sequence = 0
            self._dsp_block_sequence = 0
            if self._should_process_audio():
                self._pending_start_frame = frame
                self._audio_buffer.clear()
                self._buffer_template = None
                self._startup_frames.clear()
                self._startup_released = False
                self._trace(
                    "modulation.tts_start",
                    utterance_id=utterance_id,
                    mode="held",
                    audible_effect=True,
                )
                return
            self._trace(
                "modulation.tts_start",
                utterance_id=utterance_id,
                mode="passthrough",
                audible_effect=False,
            )
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStoppedFrame):
            utterance_id = self._ensure_utterance_id(frame)
            self._trace(
                "modulation.tts_stop",
                utterance_id=utterance_id,
                raw_buffer_bytes=len(self._audio_buffer),
                queued_startup_frames=len(self._startup_frames),
                pending_start_frame=self._pending_start_frame is not None,
            )
            if self._should_process_audio():
                await self._flush_buffer(direction)
                await self._emit_tail(direction)
                await self._release_startup_prebuffer(direction, force=True)
            await self.push_frame(frame, direction)
            self._reset_stream(reason="tts_stop")
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._trace(
                "modulation.reset_frame",
                utterance_id=self._current_utterance_id,
                frame_type=type(frame).__name__,
                raw_buffer_bytes=len(self._audio_buffer),
                dropped_start_frame=self._pending_start_frame is not None,
                dropped_startup_frames=len(self._startup_frames),
            )
            self._reset_stream(reason=type(frame).__name__)
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, TTSAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        utterance_id = self._ensure_utterance_id(frame)
        chunk_seq = self._frame_chunk_sequence(frame)
        self._trace(
            "modulation.audio_receive",
            utterance_id=utterance_id,
            chunk_seq=chunk_seq,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            **pcm16_audio_metrics(
                frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            ),
        )
        if not self._should_process_audio():
            self._trace_audio_push(
                frame,
                release_mode="passthrough",
                direction=direction,
                chunk_seq=chunk_seq,
            )
            await self.push_frame(frame, direction)
            return

        await self._process_tts_audio(frame, direction)

    async def _process_tts_audio(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        await self._ensure_stream_format(frame, direction)
        self._tail_template = frame
        if self._pending_start_frame is not None and not self._startup_released:
            self._audio_buffer.extend(frame.audio)
            self._buffer_template = frame
            self._trace_buffer(frame, raw_buffer_bytes=len(self._audio_buffer))
            if len(self._audio_buffer) < self._block_size_bytes(
                frame.sample_rate,
                frame.num_channels,
            ):
                return
            await self._flush_buffer(direction)
            return

        self._trace(
            "modulation.buffer",
            utterance_id=self._ensure_utterance_id(frame),
            chunk_seq=self._frame_chunk_sequence(frame, advance=False),
            raw_buffer_bytes=len(frame.audio),
            block_size_bytes=self._block_size_bytes(frame.sample_rate, frame.num_channels),
        )
        await self._push_processed_audio(frame.audio, frame, direction)

    def _trace_buffer(
        self,
        frame: TTSAudioRawFrame,
        *,
        raw_buffer_bytes: int,
    ) -> None:
        self._trace(
            "modulation.buffer",
            utterance_id=self._ensure_utterance_id(frame),
            chunk_seq=self._frame_chunk_sequence(frame, advance=False),
            raw_buffer_bytes=raw_buffer_bytes,
            block_size_bytes=self._block_size_bytes(frame.sample_rate, frame.num_channels),
        )

    async def _push_processed_audio(
        self,
        audio: bytes,
        template: TTSAudioRawFrame,
        direction: FrameDirection,
        *,
        release_mode: str = "immediate",
    ) -> bytes:
        self._dsp_block_sequence += 1
        block_seq = self._dsp_block_sequence
        self._trace(
            "modulation.dsp_start",
            utterance_id=self._ensure_utterance_id(template),
            block_seq=block_seq,
            release_mode=release_mode,
            input_bytes=len(audio),
        )
        started_at = time.perf_counter()
        processed = await self._process_pcm16(
            audio,
            sample_rate=template.sample_rate,
            num_channels=template.num_channels,
        )
        dsp_ms = (time.perf_counter() - started_at) * 1000.0
        self._trace(
            "modulation.dsp_end",
            utterance_id=self._ensure_utterance_id(template),
            block_seq=block_seq,
            release_mode=release_mode,
            dsp_ms=round(dsp_ms, 3),
            input_bytes=len(audio),
            **pcm16_audio_metrics(
                processed,
                sample_rate=template.sample_rate,
                num_channels=template.num_channels,
            ),
        )
        await self._push_or_buffer_processed_frame(
            self._processed_frame(template, processed),
            direction,
            release_mode=release_mode,
        )
        return processed

    async def _push_or_buffer_processed_frame(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
        *,
        release_mode: str,
    ) -> None:
        if self._pending_start_frame is None or self._startup_released:
            await self._push_audio_frame(frame, direction, release_mode=release_mode)
            return

        self._startup_frames.append((frame, release_mode))
        self._trace(
            "modulation.prebuffer_queue",
            utterance_id=self._ensure_utterance_id(frame),
            queued_frames=len(self._startup_frames),
            queued_audio_bytes=sum(len(item.audio) for item, _ in self._startup_frames),
            release_mode=release_mode,
        )
        await self._release_startup_prebuffer(direction)

    async def _release_startup_prebuffer(
        self,
        direction: FrameDirection,
        *,
        force: bool = False,
    ) -> None:
        if self._startup_released:
            return
        if not force and not self._startup_frames:
            return
        self._trace(
            "modulation.prebuffer_release",
            utterance_id=self._current_utterance_id,
            force=force,
            queued_frames=len(self._startup_frames),
            queued_audio_bytes=sum(len(item.audio) for item, _ in self._startup_frames),
            pending_start_frame=self._pending_start_frame is not None,
        )
        if self._pending_start_frame is not None and self._startup_frames:
            await self.push_frame(self._pending_start_frame, direction)
        self._pending_start_frame = None
        for frame, release_mode in self._startup_frames:
            queued_mode = "prebuffer" if release_mode == "immediate" else release_mode
            await self._push_audio_frame(frame, direction, release_mode=queued_mode)
        self._startup_frames.clear()
        self._startup_released = True

    async def _push_audio_frame(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
        *,
        release_mode: str,
    ) -> None:
        self._trace_audio_push(frame, release_mode=release_mode, direction=direction)
        await self.push_frame(frame, direction)

    def _processed_frame(self, frame: TTSAudioRawFrame, audio: bytes) -> TTSAudioRawFrame:
        processed = replace(frame, audio=audio)
        processed.metadata = dict(frame.metadata)
        processed.pts = frame.pts
        processed.broadcast_sibling_id = frame.broadcast_sibling_id
        processed.transport_source = frame.transport_source
        processed.transport_destination = frame.transport_destination
        return processed

    async def _flush_buffer(self, direction: FrameDirection) -> None:
        if not self._audio_buffer or self._buffer_template is None:
            return
        template = self._buffer_template
        block_size = self._block_size_bytes(template.sample_rate, template.num_channels)
        if len(self._audio_buffer) < block_size:
            self._trace(
                "modulation.buffer_drop",
                utterance_id=self._ensure_utterance_id(template),
                raw_buffer_bytes=len(self._audio_buffer),
                block_size_bytes=block_size,
            )
            self._audio_buffer.clear()
            self._buffer_template = None
            return
        block = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        self._buffer_template = None
        await self._push_processed_audio(block, template, direction)

    async def _emit_tail(self, direction: FrameDirection) -> None:
        if (
            not self._has_tail_effect()
            or self._tail_template is None
            or self._stream_sample_rate is None
            or self._stream_num_channels is None
        ):
            return

        template = self._tail_template
        silence = b"\x00" * self._block_size_bytes(
            self._stream_sample_rate,
            self._stream_num_channels,
        )
        max_blocks = int(_TAIL_MAX_DURATION_S / _BLOCK_DURATION_S)
        grace_blocks = self._tail_grace_blocks(max_blocks)
        silent_blocks = 0

        for index in range(max_blocks):
            processed = await self._push_processed_audio(
                silence,
                template,
                direction,
                release_mode="tail",
            )
            if index + 1 <= grace_blocks:
                continue
            if dsp.pcm16_rms(processed) < _TAIL_SILENCE_RMS:
                silent_blocks += 1
            else:
                silent_blocks = 0
            if silent_blocks >= _TAIL_SILENT_BLOCKS:
                break

    async def _ensure_stream_format(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        if (
            self._stream_sample_rate == frame.sample_rate
            and self._stream_num_channels == frame.num_channels
        ):
            return
        if self._audio_buffer:
            await self._flush_buffer(direction)
        self._dsp_state = VoiceModulationState()
        self._stream_sample_rate = frame.sample_rate
        self._stream_num_channels = frame.num_channels

    async def _process_pcm16(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
    ) -> bytes:
        return await asyncio.to_thread(
            dsp.process_pcm16,
            audio,
            sample_rate=sample_rate,
            num_channels=num_channels,
            settings=self._settings,
            state=self._dsp_state,
        )

    def _should_process_audio(self) -> bool:
        return self._settings.enabled and self._settings.has_audible_effect()

    def _ensure_utterance_id(self, frame: Frame | None = None) -> str:
        if frame is not None:
            value = frame.metadata.get(VOICE_STREAM_UTTERANCE_ID)
            if isinstance(value, str) and value:
                self._current_utterance_id = value
                return value
        if self._current_utterance_id is None:
            self._generated_utterance_count += 1
            self._current_utterance_id = f"voice-modulation-{self._generated_utterance_count:04d}"
        return self._current_utterance_id

    def _frame_chunk_sequence(
        self,
        frame: TTSAudioRawFrame,
        *,
        advance: bool = True,
    ) -> int:
        value = frame.metadata.get(VOICE_STREAM_CHUNK_SEQ)
        if isinstance(value, int):
            self._audio_frame_sequence = max(self._audio_frame_sequence, value)
            return value
        if advance:
            self._audio_frame_sequence += 1
        return self._audio_frame_sequence

    def _trace_audio_push(
        self,
        frame: TTSAudioRawFrame,
        *,
        release_mode: str,
        direction: FrameDirection,
        chunk_seq: int | None = None,
    ) -> None:
        self._trace(
            "modulation.audio_push",
            utterance_id=self._ensure_utterance_id(frame),
            chunk_seq=chunk_seq
            if chunk_seq is not None
            else self._frame_chunk_sequence(frame, advance=False),
            release_mode=release_mode,
            direction=direction.name,
            sample_rate=frame.sample_rate,
            num_channels=frame.num_channels,
            source=frame.metadata.get(VOICE_STREAM_SOURCE),
            **pcm16_audio_metrics(
                frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
            ),
        )

    def _trace(self, event: str, **attributes: Any) -> None:
        if self._voice_stream_tracer is not None:
            self._voice_stream_tracer.event(event, **attributes)

    def _has_tail_effect(self) -> bool:
        return (
            self._settings.echo_mix > 0.0
            and self._settings.echo_delay_ms > 0.0
        ) or (
            self._settings.chorus_mix > 0.0
            and self._settings.chorus_depth_ms > 0.0
        )

    def _tail_grace_blocks(self, max_blocks: int) -> int:
        delay_ms = 0.0
        if self._settings.echo_mix > 0.0 and self._settings.echo_delay_ms > 0.0:
            delay_ms = max(delay_ms, self._settings.echo_delay_ms)
        if self._settings.chorus_mix > 0.0 and self._settings.chorus_depth_ms > 0.0:
            delay_ms = max(delay_ms, 12.0 + self._settings.chorus_depth_ms)
        return min(max_blocks, math.ceil(delay_ms / (_BLOCK_DURATION_S * 1000.0)))

    def _block_size_bytes(self, sample_rate: int, num_channels: int) -> int:
        samples = max(1, int(round(sample_rate * _BLOCK_DURATION_S)))
        return samples * max(1, num_channels) * 2

    def _reset_stream(self, *, reason: str) -> None:
        self._dsp_state = VoiceModulationState()
        self._audio_buffer.clear()
        self._buffer_template = None
        self._tail_template = None
        self._stream_sample_rate = None
        self._stream_num_channels = None
        self._pending_start_frame = None
        self._startup_frames.clear()
        self._startup_released = False
        self._current_utterance_id = None
        self._audio_frame_sequence = 0
        self._dsp_block_sequence = 0
        self._trace("modulation.reset", reason=reason)
