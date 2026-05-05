from __future__ import annotations

from collections.abc import Callable
from typing import Any

from voice_runtime.wake_command import MaveVoiceCommandTranscriptAdapter, strip_mave_wake_phrase


def strip_wake_phrase(text: str) -> str:
    """Compatibility wrapper for Mave wake phrase cleanup."""
    cleaned = strip_mave_wake_phrase(text)
    return cleaned or text.strip()


class WakePhraseTranscriptCleaner(MaveVoiceCommandTranscriptAdapter):
    """Compatibility wrapper for the reusable Mave Voice Command transcript Adapter."""

    def __init__(self, *, on_finalized_transcription: Callable[[], None] | None = None, **kwargs: Any):
        super().__init__(on_finalized_transcription=on_finalized_transcription, **kwargs)
