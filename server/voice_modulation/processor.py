from __future__ import annotations

import asyncio
import math
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

_BLOCK_DURATION_S = 0.02
_TAIL_MAX_DURATION_S = 0.24
_TAIL_SILENCE_RMS = 0.0005
_TAIL_SILENT_BLOCKS = 2
_STARTUP_PREBUFFER_BLOCKS = 3


class VoiceModulationProcessor(FrameProcessor):
    def __init__(self, *, settings: VoiceModulationSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._dsp_state = VoiceModulationState()
        self._audio_buffer = bytearray()
        self._buffer_template: TTSAudioRawFrame | None = None
        self._tail_template: TTSAudioRawFrame | None = None
        self._stream_sample_rate: int | None = None
        self._stream_num_channels: int | None = None
        self._pending_start_frame: TTSStartedFrame | None = None
        self._startup_frames: list[TTSAudioRawFrame] = []
        self._startup_released = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSStartedFrame):
            if self._should_process_audio():
                self._pending_start_frame = frame
                self._startup_frames.clear()
                self._startup_released = False
                return
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TTSStoppedFrame):
            if self._should_process_audio():
                await self._flush_buffer(direction)
                await self._emit_tail(direction)
                await self._release_startup_prebuffer(direction, force=True)
            await self.push_frame(frame, direction)
            self._reset_stream()
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._reset_stream()
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, TTSAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        if not self._should_process_audio():
            await self.push_frame(frame, direction)
            return

        await self._process_tts_audio(frame, direction)

    async def _process_tts_audio(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        await self._ensure_stream_format(frame, direction)
        self._audio_buffer.extend(frame.audio)
        self._buffer_template = frame
        self._tail_template = frame
        await self._push_complete_blocks(frame, direction)

    async def _push_complete_blocks(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        block_size = self._block_size_bytes(frame.sample_rate, frame.num_channels)
        while len(self._audio_buffer) >= block_size:
            block = bytes(self._audio_buffer[:block_size])
            del self._audio_buffer[:block_size]
            await self._push_processed_audio(block, frame, direction)
        if not self._audio_buffer:
            self._buffer_template = None

    async def _push_processed_audio(
        self,
        audio: bytes,
        template: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        processed = await self._process_pcm16(
            audio,
            sample_rate=template.sample_rate,
            num_channels=template.num_channels,
        )
        await self._push_or_buffer_processed_frame(
            self._processed_frame(template, processed),
            direction,
        )

    async def _push_or_buffer_processed_frame(
        self,
        frame: TTSAudioRawFrame,
        direction: FrameDirection,
    ) -> None:
        if self._pending_start_frame is None or self._startup_released:
            await self.push_frame(frame, direction)
            return

        self._startup_frames.append(frame)
        if len(self._startup_frames) >= _STARTUP_PREBUFFER_BLOCKS:
            await self._release_startup_prebuffer(direction)

    async def _release_startup_prebuffer(
        self,
        direction: FrameDirection,
        *,
        force: bool = False,
    ) -> None:
        if self._startup_released:
            return
        if (
            not force
            and len(self._startup_frames) < _STARTUP_PREBUFFER_BLOCKS
        ):
            return
        if self._pending_start_frame is not None:
            await self.push_frame(self._pending_start_frame, direction)
            self._pending_start_frame = None
        for frame in self._startup_frames:
            await self.push_frame(frame, direction)
        self._startup_frames.clear()
        self._startup_released = True

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
        block = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        template = self._buffer_template
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
            processed = await self._process_pcm16(
                silence,
                sample_rate=self._stream_sample_rate,
                num_channels=self._stream_num_channels,
            )
            await self._push_or_buffer_processed_frame(
                self._processed_frame(template, processed),
                direction,
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

    def _reset_stream(self) -> None:
        self._dsp_state = VoiceModulationState()
        self._audio_buffer.clear()
        self._buffer_template = None
        self._tail_template = None
        self._stream_sample_rate = None
        self._stream_num_channels = None
        self._pending_start_frame = None
        self._startup_frames.clear()
        self._startup_released = False
