from __future__ import annotations

from dataclasses import replace
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_modulation import dsp
from voice_modulation.dsp import VoiceModulationState
from voice_modulation.settings import VoiceModulationSettings


class VoiceModulationProcessor(FrameProcessor):
    def __init__(self, *, settings: VoiceModulationSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._dsp_state = VoiceModulationState()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (TTSStoppedFrame, CancelFrame, EndFrame)):
            self._reset_dsp_state()
            await self.push_frame(frame, direction)
            return

        if not isinstance(frame, TTSAudioRawFrame):
            await self.push_frame(frame, direction)
            return

        if not self._settings.enabled:
            await self.push_frame(frame, direction)
            return

        await self.push_frame(self._process_tts_audio(frame), direction)

    def _process_tts_audio(self, frame: TTSAudioRawFrame) -> TTSAudioRawFrame:
        audio = self._process_pcm16(frame)
        processed = replace(frame, audio=audio)
        processed.metadata = dict(frame.metadata)
        processed.pts = frame.pts
        processed.broadcast_sibling_id = frame.broadcast_sibling_id
        processed.transport_source = frame.transport_source
        processed.transport_destination = frame.transport_destination
        return processed

    def _process_pcm16(self, frame: TTSAudioRawFrame) -> bytes:
        kwargs = {
            "sample_rate": frame.sample_rate,
            "num_channels": frame.num_channels,
            "settings": self._settings,
            "state": self._dsp_state,
        }
        return dsp.process_pcm16(frame.audio, **kwargs)

    def _reset_dsp_state(self) -> None:
        self._dsp_state = VoiceModulationState()
