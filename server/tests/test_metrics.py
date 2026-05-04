import json
from pathlib import Path

from metrics import VoiceMetricsRecorder


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

    data = json.loads(path.read_text(encoding="utf-8"))
    assert "transcript" not in data
    assert "response" not in data
