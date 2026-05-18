from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_modulation.dsp import VoiceModulationState
from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings

PCM_20MS_24K_MONO = b"\x01\x00" * 480
PCM_10MS_24K_MONO = b"\x01\x00" * 240


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
async def test_enabled_clean_settings_push_original_tts_audio_object_unchanged(
    monkeypatch,
) -> None:
    def fail_process_pcm16(*args: Any, **kwargs: Any) -> bytes:
        raise AssertionError("clean settings must not run DSP")

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fail_process_pcm16)
    processor = CapturingVoiceModulationProcessor(
        VoiceModulationSettings(enabled=True, preset_name="clean")
    )
    frame = TTSAudioRawFrame(audio=b"\x01", sample_rate=24000, num_channels=1)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed == [frame]
    assert processor.pushed[0] is frame


@pytest.mark.asyncio
async def test_enabled_settings_process_complete_blocks_via_thread_and_preserve_metadata(
    monkeypatch,
) -> None:
    calls: list[dict[str, Any]] = []
    thread_calls: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []

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
        return b"\x03\x00" * 480

    async def fake_to_thread(func: Any, /, *args: Any, **kwargs: Any) -> Any:
        thread_calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    settings = VoiceModulationSettings(enabled=True, gain_db=3.0)
    processor = CapturingVoiceModulationProcessor(settings)
    frame = TTSAudioRawFrame(
        audio=PCM_20MS_24K_MONO,
        sample_rate=24000,
        num_channels=1,
        context_id="tts-context",
    )
    frame.metadata["trace"] = "kept"

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    pushed = processor.pushed[0]
    assert isinstance(pushed, TTSAudioRawFrame)
    assert pushed is not frame
    assert pushed.audio == b"\x03\x00" * 480
    assert len(pushed.audio) == len(frame.audio)
    assert pushed.sample_rate == 24000
    assert pushed.num_channels == 1
    assert pushed.context_id == "tts-context"
    assert pushed.metadata == {"trace": "kept"}
    assert len(thread_calls) == 1
    assert calls == [
        {
            "audio": PCM_20MS_24K_MONO,
            "sample_rate": 24000,
            "num_channels": 1,
            "settings": settings,
            "state": calls[0]["state"],
        }
    ]


@pytest.mark.asyncio
async def test_streaming_audio_is_buffered_into_20ms_blocks(monkeypatch) -> None:
    calls: list[bytes] = []

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        calls.append(audio)
        return audio

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    processor = CapturingVoiceModulationProcessor(VoiceModulationSettings(enabled=True, gain_db=3.0))
    first = TTSAudioRawFrame(audio=PCM_10MS_24K_MONO, sample_rate=24000, num_channels=1)
    second = TTSAudioRawFrame(audio=PCM_10MS_24K_MONO, sample_rate=24000, num_channels=1)

    await processor.process_frame(first, FrameDirection.DOWNSTREAM)
    await processor.process_frame(second, FrameDirection.DOWNSTREAM)

    audio_frames = [frame for frame in processor.pushed if isinstance(frame, TTSAudioRawFrame)]
    assert len(audio_frames) == 1
    assert audio_frames[0].audio == PCM_20MS_24K_MONO
    assert calls == [PCM_20MS_24K_MONO]


@pytest.mark.asyncio
async def test_active_modulation_holds_tts_started_until_startup_prebuffer_ready(
    monkeypatch,
) -> None:
    calls: list[bytes] = []

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        calls.append(audio)
        return bytes([len(calls), 0]) * (len(audio) // 2)

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    processor = CapturingVoiceModulationProcessor(
        VoiceModulationSettings(enabled=True, gain_db=3.0)
    )
    start_frame = TTSStartedFrame()

    await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(
        TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    assert processor.pushed == []

    await processor.process_frame(
        TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    audio_frames = [frame for frame in processor.pushed if isinstance(frame, TTSAudioRawFrame)]
    assert processor.pushed[0] is start_frame
    assert [frame.audio[:2] for frame in audio_frames] == [b"\x01\x00", b"\x02\x00", b"\x03\x00"]

    await processor.process_frame(
        TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    audio_frames = [frame for frame in processor.pushed if isinstance(frame, TTSAudioRawFrame)]
    assert [frame.audio[:2] for frame in audio_frames] == [
        b"\x01\x00",
        b"\x02\x00",
        b"\x03\x00",
        b"\x04\x00",
    ]


@pytest.mark.asyncio
async def test_tts_stopped_force_releases_short_startup_prebuffer(monkeypatch) -> None:
    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        return b"\x07\x00" * (len(audio) // 2)

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    processor = CapturingVoiceModulationProcessor(
        VoiceModulationSettings(enabled=True, gain_db=3.0)
    )
    start_frame = TTSStartedFrame()
    stop_frame = TTSStoppedFrame()

    await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )

    assert processor.pushed == []

    await processor.process_frame(stop_frame, FrameDirection.DOWNSTREAM)

    audio_frames = [frame for frame in processor.pushed if isinstance(frame, TTSAudioRawFrame)]
    assert processor.pushed[0] is start_frame
    assert len(audio_frames) == 1
    assert audio_frames[0].audio == b"\x07\x00" * 480
    assert processor.pushed[-1] is stop_frame


@pytest.mark.asyncio
async def test_cancel_and_end_drop_held_startup_audio(monkeypatch) -> None:
    calls: list[bytes] = []

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        calls.append(audio)
        return audio

    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)

    for reset_frame in (CancelFrame(), EndFrame()):
        processor = CapturingVoiceModulationProcessor(
            VoiceModulationSettings(enabled=True, gain_db=3.0)
        )
        start_frame = TTSStartedFrame()

        await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
        await processor.process_frame(
            TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1),
            FrameDirection.DOWNSTREAM,
        )
        await processor.process_frame(reset_frame, FrameDirection.DOWNSTREAM)

        assert processor.pushed == [reset_frame]


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
    frame = TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1)

    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)
    state = processor._dsp_state
    first_phase = state.ring_phase
    await processor.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert isinstance(state, VoiceModulationState)
    assert processor._dsp_state is state
    assert first_phase > 0.0
    assert state.ring_phase != first_phase


@pytest.mark.asyncio
async def test_echo_tail_is_emitted_before_tts_stopped() -> None:
    settings = VoiceModulationSettings(
        enabled=True,
        preset_name="tail",
        echo_delay_ms=20.0,
        echo_mix=0.5,
    )
    processor = CapturingVoiceModulationProcessor(settings)
    audio_frame = TTSAudioRawFrame(
        audio=(b"\xe8\x03" * 480),
        sample_rate=24000,
        num_channels=1,
        context_id="tts-context",
    )
    stop_frame = TTSStoppedFrame()

    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(stop_frame, FrameDirection.DOWNSTREAM)

    assert processor.pushed[-1] is stop_frame
    tail_frames = [
        frame
        for frame in processor.pushed[1:-1]
        if isinstance(frame, TTSAudioRawFrame)
    ]
    assert isinstance(processor.pushed[0], TTSAudioRawFrame)
    assert tail_frames
    assert any(frame.audio != (b"\x00\x00" * 480) for frame in tail_frames)


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
    processor = CapturingVoiceModulationProcessor(
        VoiceModulationSettings(enabled=True, gain_db=3.0)
    )
    audio_frame = TTSAudioRawFrame(audio=PCM_20MS_24K_MONO, sample_rate=24000, num_channels=1)

    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(reset_frame, FrameDirection.DOWNSTREAM)
    await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)

    assert reset_frame in processor.pushed
    assert len(states) == 2
    assert states[0] is not states[1]
