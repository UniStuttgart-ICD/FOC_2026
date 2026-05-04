from __future__ import annotations

from collections import deque

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from wake.openwakeword_detector import OpenWakeWordDetector


class MaveWakeWordGate(FrameProcessor):
    """Blocks user audio until Mave is detected, then allows one command through."""

    def __init__(self, detector: OpenWakeWordDetector, *, pre_buffer_s: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._detector = detector
        self._pre_buffer_s = pre_buffer_s
        self._ring: deque[InputAudioRawFrame] = deque()
        self._ring_samples = 0
        self._awake = False

    @property
    def is_awake(self) -> bool:
        return self._awake

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._process_audio_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _process_audio_frame(
        self, frame: InputAudioRawFrame, direction: FrameDirection
    ) -> None:
        if self._awake:
            await self.push_frame(frame, direction)
            return

        self._append_ring(frame)
        pcm16 = self._to_mono_int16(frame)
        detected, name, score = self._detector.detected(pcm16)
        if not detected:
            return

        logger.info(f"Wake word detected: {name}={score:.3f}")
        self._awake = True
        buffered = list(self._ring)
        self._ring.clear()
        self._ring_samples = 0
        for buffered_frame in buffered:
            await self.push_frame(buffered_frame, direction)

    def _append_ring(self, frame: InputAudioRawFrame) -> None:
        self._ring.append(frame)
        self._ring_samples += len(frame.audio) // 2 // max(frame.num_channels, 1)
        max_samples = int(frame.sample_rate * self._pre_buffer_s)
        while self._ring and self._ring_samples > max_samples:
            old = self._ring.popleft()
            self._ring_samples -= len(old.audio) // 2 // max(old.num_channels, 1)

    def reset(self) -> None:
        self._awake = False
        self._ring.clear()
        self._ring_samples = 0

    @staticmethod
    def _to_mono_int16(frame: InputAudioRawFrame) -> np.ndarray:
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        if frame.num_channels <= 1:
            return pcm
        return pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
