from __future__ import annotations

from typing import Any

import pytest
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_modulation.dsp import VoiceModulationState
from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings


class CapturingVoiceModulationProcessor(VoiceModulationProcessor):
    def __init__(self, settings: VoiceModulationSettings) -> None:
        super().__init__(settings=settings)
        self.pushed: list[Frame] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append(frame)


@pytest.mark.asyncio
async def test_disabled_settings_push_original_tts_audio_object_unchanged() -> None:
    processor = CapturingVoiceModulationProcessor(VoiceModulationSettings(enabled=False))
    frame = TTSAudioRawFrame(audio=b"\x01\x00\x02\x00", sample_rate=24000, num_channels=1)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [frame]
    assert processor.pushed[0] is frame


@pytest.mark.asyncio
async def test_enabled_settings_replace_tts_audio_and_preserve_metadata(monkeypatch) -> None:
    calls: list[dict[str, Any]] = []

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        calls.append(
            {
                "audio": audio,
                "sample_rate": sample_rate,
                "num_channels": num_channels,
                "settings": settings,
                "state": state,
            }
        )
        return b"\x03\x00\x04\x00"

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    settings = VoiceModulationSettings(enabled=True, gain_db=3.0)
    processor = CapturingVoiceModulationProcessor(settings)
    frame = TTSAudioRawFrame(
        audio=b"\x01\x00\x02\x00",
        sample_rate=24000,
        num_channels=1,
        context_id="tts-context",
    )
    frame.metadata["trace"] = "kept"

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    pushed = processor.pushed[0]
    assert isinstance(pushed, TTSAudioRawFrame)
    assert pushed is not frame
    assert pushed.audio == b"\x03\x00\x04\x00"
    assert len(pushed.audio) == len(frame.audio)
    assert pushed.sample_rate == 24000
    assert pushed.num_channels == 1
    assert pushed.context_id == "tts-context"
    assert pushed.metadata == {"trace": "kept"}
    assert calls == [
        {
            "audio": b"\x01\x00\x02\x00",
            "sample_rate": 24000,
            "num_channels": 1,
            "settings": settings,
            "state": calls[0]["state"],
        }
    ]


@pytest.mark.asyncio
async def test_non_audio_frames_are_pushed_unchanged() -> None:
    processor = CapturingVoiceModulationProcessor(VoiceModulationSettings(enabled=True))
    frame = LLMTextFrame(text="hello")

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [frame]
    assert processor.pushed[0] is frame


@pytest.mark.asyncio
async def test_real_dsp_chunks_share_processor_state() -> None:
    settings = VoiceModulationSettings(enabled=True, ring_mod_hz=40.0)
    processor = CapturingVoiceModulationProcessor(settings)
    frame = TTSAudioRawFrame(audio=(b"\x01\x00" * 240), sample_rate=24000, num_channels=1)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    state = processor._dsp_state
    first_phase = state.ring_phase
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert isinstance(state, VoiceModulationState)
    assert processor._dsp_state is state
    assert first_phase > 0.0
    assert state.ring_phase > first_phase


@pytest.mark.asyncio
@pytest.mark.parametrize("reset_frame", [TTSStoppedFrame(), CancelFrame(), EndFrame()])
async def test_reset_frames_push_unchanged_and_reset_dsp_state(
    monkeypatch,
    reset_frame: Frame,
) -> None:
    states: list[object] = []

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        states.append(state)
        return audio

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    processor = CapturingVoiceModulationProcessor(VoiceModulationSettings(enabled=True))
    audio_frame = TTSAudioRawFrame(audio=b"\x01\x00", sample_rate=24000, num_channels=1)

    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(reset_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed[1] is reset_frame
    assert len(states) == 2
    assert states[0] is not states[1]
