from __future__ import annotations

import re

from pipecat.frames.frames import Frame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_WAKE_PATTERN = re.compile(r"^\s*(?:hey\s+)?mave[\s,;:!?.-]*", re.IGNORECASE)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading Mave wake phrase from a transcript."""
    cleaned = _WAKE_PATTERN.sub("", text, count=1).strip()
    return cleaned or text.strip()


class WakePhraseTranscriptCleaner(FrameProcessor):
    """Removes a leading wake phrase from downstream transcription frames."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            frame = TranscriptionFrame(
                text=strip_wake_phrase(frame.text),
                user_id=frame.user_id,
                timestamp=frame.timestamp,
                language=frame.language,
                result=frame.result,
                finalized=frame.finalized,
            )

        await self.push_frame(frame, direction)
