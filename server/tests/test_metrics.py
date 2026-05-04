import json
from pathlib import Path

import pytest
from pipecat.frames.frames import (
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from metrics import VoiceMetricsObserver, VoiceMetricsRecorder


def _read_jsonl_record(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pushed(frame):
    return FramePushed(
        source=FrameProcessor(),
        destination=FrameProcessor(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def test_writes_jsonl_turn_record(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        path=path,
        include_text=True,
    )

    turn = recorder.start_turn("turn-1")
    turn.transcript = "move up"
    turn.response = "Moving up."
    turn.mark("wake_detected")
    turn.mark("speech_captured")
    recorder.finish_turn("turn-1")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["profile"] == "hybrid_low_latency"
    assert data["turn_id"] == "turn-1"
    assert data["transcript"] == "move up"
    assert data["response"] == "Moving up."


def test_omits_text_when_disabled(tmp_path: Path):
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="local_current",
        category="local_debug",
        path=path,
        include_text=False,
    )

    turn = recorder.start_turn("turn-2")
    turn.transcript = "secret"
    turn.response = "secret response"
    recorder.finish_turn("turn-2")

    data = _read_jsonl_record(path)
    assert "transcript" not in data
    assert "response" not in data


def test_records_deterministic_stage_timings(monkeypatch, tmp_path: Path):
    perf_counter_values = iter(
        [
            100.000,
            100.010,
            100.060,
            100.210,
            100.240,
            100.320,
            100.400,
            100.500,
        ]
    )
    monkeypatch.setattr("metrics.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("metrics.time.time", lambda: 1_700_000_000.25)
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        path=path,
        include_text=True,
    )

    turn = recorder.start_turn("turn-3")
    turn.wake_phrase = "Mave"
    turn.mark("wake_detected")
    turn.mark("speech_captured")
    turn.mark("stt_done")
    turn.mark("agent_done")
    turn.mark("tts_first_audio")
    turn.mark("tts_done")
    recorder.finish_turn("turn-3")

    data = _read_jsonl_record(path)
    assert data["timestamp_unix"] == 1_700_000_000.25
    assert data["wake_phrase"] == "Mave"
    assert data["wake_latency_ms"] == 10.0
    assert data["speech_captured_ms"] == 50.0
    assert data["stt_latency_ms"] == 150.0
    assert data["agent_latency_ms"] == 30.0
    assert data["tts_first_audio_ms"] == 80.0
    assert data["tts_done_ms"] == 80.0
    assert data["total_to_first_audio_ms"] == 320.0
    assert data["total_turn_ms"] == 500.0
    assert "stt_done_ms" not in data
    assert "agent_done_ms" not in data


def test_speech_captured_timing_uses_turn_start_without_wake(monkeypatch, tmp_path: Path):
    perf_counter_values = iter([50.000, 50.125, 50.300])
    monkeypatch.setattr("metrics.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("metrics.time.time", lambda: 1_700_000_001.0)
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="no_wake_debug",
        category="local_debug",
        path=path,
        include_text=False,
    )

    turn = recorder.start_turn("turn-4")
    turn.mark("speech_captured")
    recorder.finish_turn("turn-4")

    data = _read_jsonl_record(path)
    assert data["wake_phrase"] == ""
    assert data["wake_latency_ms"] is None
    assert data["speech_captured_ms"] == 125.0
    assert data["stt_latency_ms"] is None
    assert data["total_to_first_audio_ms"] is None
    assert data["total_turn_ms"] == 300.0


@pytest.mark.asyncio
async def test_observer_emits_jsonl_from_turn_frames(monkeypatch, tmp_path: Path):
    perf_counter_values = iter([100.0, 100.1, 100.2, 100.3, 100.4, 100.5, 100.6])
    monkeypatch.setattr("metrics.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr("metrics.time.time", lambda: 1_700_000_002.0)
    path = tmp_path / "metrics.jsonl"
    recorder = VoiceMetricsRecorder(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        path=path,
        include_text=True,
    )
    observer = VoiceMetricsObserver(recorder)

    await observer.on_push_frame(_pushed(UserStartedSpeakingFrame()))
    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(
        _pushed(TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True))
    )
    await observer.on_push_frame(_pushed(LLMTextFrame(text="Moving up.")))
    await observer.on_push_frame(_pushed(LLMFullResponseEndFrame()))
    await observer.on_push_frame(
        _pushed(TTSAudioRawFrame(audio=b"\0\0", sample_rate=16000, num_channels=1))
    )
    await observer.on_push_frame(_pushed(TTSStoppedFrame()))

    data = _read_jsonl_record(path)
    assert data["turn_id"] == "turn-1"
    assert data["transcript"] == "move up"
    assert data["response"] == "Moving up."
    assert data["speech_captured_ms"] == 100.0
    assert data["stt_latency_ms"] == 100.0
    assert data["agent_latency_ms"] == 100.0
    assert data["tts_first_audio_ms"] == 100.0
    assert data["tts_done_ms"] == 100.0
    assert data["total_turn_ms"] == 600.0
