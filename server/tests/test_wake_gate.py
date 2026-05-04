from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from wake.wake_gate import MaveWakeWordGate


class CapturingGate(MaveWakeWordGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def _frame(value: int, samples: int = 1600):
    audio = np.full(samples, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)


@pytest.mark.asyncio
async def test_blocks_audio_until_wake_detected():
    detector = Mock()
    detector.detected.return_value = (False, None, 0.0)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)

    assert gate.pushed == []


@pytest.mark.asyncio
async def test_replays_prebuffer_on_wake():
    detector = Mock()
    detector.detected.side_effect = [
        (False, None, 0.0),
        (True, "mave", 0.9),
    ]
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)
    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [item[0] for item in gate.pushed if isinstance(item[0], InputAudioRawFrame)]
    assert len(pushed_audio) == 2
    assert np.frombuffer(pushed_audio[0].audio, dtype=np.int16)[0] == 1
    assert np.frombuffer(pushed_audio[1].audio, dtype=np.int16)[0] == 2


@pytest.mark.asyncio
async def test_strips_wake_phrase_from_transcription_and_resets_to_sleep():
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.9)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)
    await gate.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    transcription = [item[0] for item in gate.pushed if isinstance(item[0], TranscriptionFrame)][0]
    assert transcription.text == "move up"
    assert gate.is_awake is False
