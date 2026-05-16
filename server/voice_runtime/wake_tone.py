from __future__ import annotations

import math
from typing import Any

from pipecat.frames.frames import Frame, OutputAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_runtime.wake_command import WakeDetectedFrame


class WakeToneProcessor(FrameProcessor):
    """Emits a short output-side ding when the wake gate opens."""

    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        duration_s: float = 0.09,
        frequency_hz: float = 880.0,
        volume: float = 0.18,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._sample_rate = sample_rate
        self._tone_audio = _build_ding_pcm16(
            sample_rate=sample_rate,
            duration_s=duration_s,
            frequency_hz=frequency_hz,
            volume=volume,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

        if isinstance(frame, WakeDetectedFrame):
            await self.push_frame(
                OutputAudioRawFrame(
                    audio=self._tone_audio,
                    sample_rate=self._sample_rate,
                    num_channels=1,
                ),
                direction,
            )


def _build_ding_pcm16(
    *,
    sample_rate: int,
    duration_s: float,
    frequency_hz: float,
    volume: float,
) -> bytes:
    sample_count = int(sample_rate * duration_s)
    peak = int(32767 * volume)
    attack_samples = max(1, int(sample_rate * 0.005))
    data = bytearray()

    for index in range(sample_count):
        time_s = index / sample_rate
        fade_out = 1.0 - (index / max(sample_count - 1, 1))
        attack = min(1.0, index / attack_samples)
        envelope = attack * fade_out
        sample = int(math.sin(2.0 * math.pi * frequency_hz * time_s) * peak * envelope)
        data.extend(sample.to_bytes(2, "little", signed=True))

    return bytes(data)
