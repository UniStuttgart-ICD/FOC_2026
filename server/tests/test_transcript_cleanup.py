import pytest
from pipecat.frames.frames import TextFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from wake.transcript_cleanup import WakePhraseTranscriptCleaner, strip_wake_phrase


class CapturingCleaner(WakePhraseTranscriptCleaner):
    def __init__(self):
        super().__init__()
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def test_strips_leading_mave():
    assert strip_wake_phrase("Mave, move up a bit") == "move up a bit"


def test_strips_hey_mave():
    assert strip_wake_phrase("hey mave stop") == "stop"


def test_leaves_non_wake_text_unchanged():
    assert strip_wake_phrase("move up a bit") == "move up a bit"


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
