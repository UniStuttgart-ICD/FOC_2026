from __future__ import annotations

import numpy as np
import pytest
from pipecat.frames.frames import Frame, OutputAudioRawFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.wake_command import WakeDetectedFrame
from voice_runtime.wake_tone import WakeToneProcessor


class CapturingProcessor:
    def __init__(self) -> None:
        self.pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append((frame, direction))


@pytest.mark.asyncio
async def test_wake_tone_forwards_wake_event_and_emits_short_output_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = WakeToneProcessor(sample_rate=16000, duration_s=0.08)
    capture = CapturingProcessor()
    monkeypatch.setattr(processor, "push_frame", capture.push_frame)
    wake = WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91)

    await processor.process_frame(wake, FrameDirection.DOWNSTREAM)

    assert capture.pushed[0] == (wake, FrameDirection.DOWNSTREAM)
    tone = capture.pushed[1][0]
    assert isinstance(tone, OutputAudioRawFrame)
    assert not isinstance(tone, TTSAudioRawFrame)
    assert tone.sample_rate == 16000
    assert tone.num_channels == 1
    assert len(tone.audio) == int(16000 * 0.08) * 2
    assert np.frombuffer(tone.audio, dtype=np.int16).max() > 0


@pytest.mark.asyncio
async def test_wake_tone_forwards_non_wake_frames_without_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = WakeToneProcessor()
    capture = CapturingProcessor()
    monkeypatch.setattr(processor, "push_frame", capture.push_frame)
    frame = Frame()

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert capture.pushed == [(frame, FrameDirection.DOWNSTREAM)]
