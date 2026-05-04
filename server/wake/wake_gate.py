from __future__ import annotations

from voice_runtime.wake_command import MaveVoiceCommandAudioGate
from wake.openwakeword_detector import OpenWakeWordDetector


class MaveWakeWordGate(MaveVoiceCommandAudioGate):
    """Compatibility wrapper for the reusable Mave Voice Command audio Adapter."""

    def __init__(self, detector: OpenWakeWordDetector, *, pre_buffer_s: float = 1.5, **kwargs):
        super().__init__(detector=detector, pre_buffer_s=pre_buffer_s, **kwargs)
