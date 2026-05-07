from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.frames.frames import InputAudioRawFrame, TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.wake_command import (
    MaveVoiceCommandAudioGate,
    MaveVoiceCommandTranscriptAdapter,
    strip_mave_wake_phrase,
)


class CapturingCleaner(MaveVoiceCommandTranscriptAdapter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


class CapturingGate(MaveVoiceCommandAudioGate):
    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        pass


def _audio_frame(value: int, samples: int = 1600):
    audio = np.full(samples, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)


def test_strips_leading_mave():
    assert strip_mave_wake_phrase("Mave, move up a bit") == "move up a bit"


def test_strips_hey_mave():
    assert strip_mave_wake_phrase("hey mave stop") == "stop"


def test_leaves_non_wake_text_unchanged():
    assert strip_mave_wake_phrase("move up a bit") == "move up a bit"


@pytest.mark.asyncio
async def test_cleaner_strips_wake_phrase_from_transcription():
    cleaner = CapturingCleaner()

    await cleaner.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    frame, direction = cleaner.pushed[0]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "move up"
    assert frame.user_id == "u"
    assert frame.timestamp == "t"
    assert direction == FrameDirection.DOWNSTREAM


@pytest.mark.asyncio
async def test_cleaner_pushes_non_transcription_frames_unchanged():
    cleaner = CapturingCleaner()
    frame = TextFrame(text="Mave, not a transcript")

    await cleaner.process_frame(frame, FrameDirection.UPSTREAM)

    assert cleaner.pushed == [(frame, FrameDirection.UPSTREAM)]


@pytest.mark.asyncio
async def test_finalized_transcription_through_cleaner_resets_audio_gate():
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.9)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)
    cleaner = CapturingCleaner(on_finalized_transcription=gate.reset)

    await gate.process_frame(_audio_frame(2), FrameDirection.DOWNSTREAM)
    assert gate.is_awake is True

    await cleaner.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    frame, _ = cleaner.pushed[0]
    assert isinstance(frame, TranscriptionFrame)
    assert frame.text == "move up"
    assert gate.is_awake is False
