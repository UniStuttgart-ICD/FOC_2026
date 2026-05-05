from voice_runtime.voice_metrics import VoiceTurnTimeline


def test_timeline_computes_stage_durations_without_pipecat_frames():
    timeline = VoiceTurnTimeline(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        turn_id="turn-1",
        started_at=10.0,
        now_fn=lambda: 10.0,
        wall_time_fn=lambda: 100.0,
    )
    timeline.wake_detected("mave", at=10.1)
    timeline.speech_captured(at=10.5)
    timeline.stt_done("move up", at=10.8)
    timeline.agent_done(at=11.8)
    timeline.tts_audio_started(at=12.0)
    timeline.tts_done(at=12.5)
    timeline.append_agent_text("Motion ")
    timeline.append_agent_text("completed.")

    record = timeline.to_record(finished_at=12.5, include_text=True)

    assert record["timestamp_unix"] == 100.0
    assert record["profile"] == "hybrid_low_latency"
    assert record["category"] == "benchmark_streaming"
    assert record["turn_id"] == "turn-1"
    assert record["wake_phrase"] == "mave"
    assert record["wake_latency_ms"] == 100.0
    assert record["speech_captured_ms"] == 400.0
    assert record["stt_latency_ms"] == 300.0
    assert record["agent_latency_ms"] == 1000.0
    assert record["tts_first_audio_ms"] == 200.0
    assert record["tts_done_ms"] == 500.0
    assert record["total_to_first_audio_ms"] == 2000.0
    assert record["total_turn_ms"] == 2500.0
    assert record["transcript"] == "move up"
    assert record["response"] == "Motion completed."


def test_timeline_without_wake_starts_speech_duration_at_turn_start():
    timeline = VoiceTurnTimeline(
        profile="no_wake_debug",
        category="local_debug",
        turn_id="turn-1",
        started_at=20.0,
        now_fn=lambda: 20.0,
        wall_time_fn=lambda: 200.0,
    )
    timeline.speech_captured(at=20.4)

    record = timeline.to_record(finished_at=21.0, include_text=False)

    assert record["wake_latency_ms"] is None
    assert record["speech_captured_ms"] == 400.0
    assert "transcript" not in record
    assert "response" not in record


def test_timeline_deduplicates_first_tts_audio_mark():
    timeline = VoiceTurnTimeline(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        turn_id="turn-1",
        started_at=1.0,
        now_fn=lambda: 1.0,
        wall_time_fn=lambda: 10.0,
    )
    timeline.agent_done(at=2.0)
    timeline.tts_audio_started(at=2.2)
    timeline.tts_audio_started(at=2.4)

    record = timeline.to_record(finished_at=3.0, include_text=False)

    assert record["tts_first_audio_ms"] == 200.0
