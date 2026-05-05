from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.wake_command import (
    MaveVoiceCommandProcessors,
    WakeDetectedFrame,
    build_mave_voice_command_processors,
    strip_mave_wake_phrase,
)


class CapturingProcessor:
    def __init__(self) -> None:
        self.pushed: list[tuple[Frame, FrameDirection]] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append((frame, direction))


def _audio(value: int, samples: int = 1600, channels: int = 1) -> InputAudioRawFrame:
    pcm = np.full(samples * channels, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=channels)


def _capture(
    monkeypatch: pytest.MonkeyPatch, processors: MaveVoiceCommandProcessors
) -> tuple[CapturingProcessor, CapturingProcessor]:
    audio_capture = CapturingProcessor()
    transcript_capture = CapturingProcessor()
    monkeypatch.setattr(processors.audio_gate, "push_frame", audio_capture.push_frame)
    monkeypatch.setattr(
        processors.transcript_adapter, "push_frame", transcript_capture.push_frame
    )
    return audio_capture, transcript_capture


def test_strip_mave_wake_phrase_handles_common_transcription_variants() -> None:
    assert strip_mave_wake_phrase("Mave, move up") == "move up"
    assert strip_mave_wake_phrase("hey Maeve stop") == "stop"
    assert strip_mave_wake_phrase("move up") == "move up"
    assert strip_mave_wake_phrase("Mave") == ""


@pytest.mark.asyncio
async def test_audio_adapter_blocks_until_wake_replays_prebuffer_and_emits_wake_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = Mock()
    detector.detected.side_effect = [(False, None, 0.0), (True, "mave", 0.91)]
    processors = build_mave_voice_command_processors(detector=detector, pre_buffer_s=1.5)
    audio_capture, _ = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.audio_gate.process_frame(_audio(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [
        frame for frame, _ in audio_capture.pushed if isinstance(frame, InputAudioRawFrame)
    ]
    wake_events = [
        frame for frame, _ in audio_capture.pushed if isinstance(frame, WakeDetectedFrame)
    ]
    assert [np.frombuffer(frame.audio, dtype=np.int16)[0] for frame in pushed_audio] == [1, 2]
    assert wake_events[0].wake_phrase == "mave"
    assert wake_events[0].model_name == "mave"
    assert wake_events[0].score == 0.91
    assert audio_capture.pushed[0][0] is wake_events[0]
    assert processors.audio_gate.is_awake is True


@pytest.mark.asyncio
async def test_audio_adapter_converts_multichannel_audio_to_mono_for_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = Mock()
    detector.detected.return_value = (False, None, 0.0)
    processors = build_mave_voice_command_processors(detector=detector)
    _capture(monkeypatch, processors)

    audio = np.array([100, 300, 300, 700], dtype=np.int16).tobytes()
    frame = InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=2)
    await processors.audio_gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    pcm16 = detector.detected.call_args.args[0]
    np.testing.assert_array_equal(pcm16, np.array([200, 500], dtype=np.int16))


@pytest.mark.asyncio
async def test_transcript_adapter_cleans_finalized_command_and_rearms_audio_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(
        detector=detector, pre_buffer_s=1.5, rearm_delay_s=0.0
    )
    _, transcript_capture = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.transcript_adapter.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    transcripts = [
        frame for frame, _ in transcript_capture.pushed if isinstance(frame, TranscriptionFrame)
    ]
    assert transcripts[0].text == "move up"
    assert transcripts[0].user_id == "u"
    assert transcripts[0].timestamp == "t"
    assert processors.audio_gate.is_awake is False


@pytest.mark.asyncio
async def test_empty_cleaned_transcript_is_not_emitted_but_rearms_when_single_command_is_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(
        detector=detector, pre_buffer_s=1.5, rearm_delay_s=0.0
    )
    _, transcript_capture = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.transcript_adapter.process_frame(
        TranscriptionFrame(text="Mave", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    assert not [
        frame for frame, _ in transcript_capture.pushed if isinstance(frame, TranscriptionFrame)
    ]
    assert processors.audio_gate.is_awake is False


@pytest.mark.asyncio
async def test_single_command_false_does_not_rearm_on_finalized_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(
        detector=detector, pre_buffer_s=1.5, single_command=False
    )
    _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.transcript_adapter.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    assert processors.audio_gate.is_awake is True


@pytest.mark.asyncio
async def test_rearm_delay_drops_audio_until_delay_elapses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0

    def time_fn() -> float:
        return now

    detector = Mock()
    detector.detected.side_effect = [
        (True, "mave", 0.91),
        (True, "mave", 0.92),
    ]
    processors = build_mave_voice_command_processors(
        detector=detector, rearm_delay_s=0.75, time_fn=time_fn
    )
    audio_capture, _ = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    processors.audio_gate.reset()
    await processors.audio_gate.process_frame(_audio(2), FrameDirection.DOWNSTREAM)
    now = 10.75
    await processors.audio_gate.process_frame(_audio(3), FrameDirection.DOWNSTREAM)

    pushed_audio = [
        frame for frame, _ in audio_capture.pushed if isinstance(frame, InputAudioRawFrame)
    ]
    assert [np.frombuffer(frame.audio, dtype=np.int16)[0] for frame in pushed_audio] == [1, 3]
    assert detector.detected.call_count == 2


@pytest.mark.asyncio
async def test_awake_timeout_rearms_and_blocks_more_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 10.0

    def time_fn() -> float:
        return now

    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(
        detector=detector, max_awake_s=1.0, rearm_delay_s=0.0, time_fn=time_fn
    )
    audio_capture, _ = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    now = 11.1
    await processors.audio_gate.process_frame(_audio(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [
        frame for frame, _ in audio_capture.pushed if isinstance(frame, InputAudioRawFrame)
    ]
    assert [np.frombuffer(frame.audio, dtype=np.int16)[0] for frame in pushed_audio] == [1, 2]
    assert processors.audio_gate.is_awake is True


def test_processor_bundle_is_frozen() -> None:
    processors = build_mave_voice_command_processors(detector=Mock())

    with pytest.raises(FrozenInstanceError):
        setattr(processors, "audio_gate", processors.audio_gate)
