from pathlib import Path

import numpy as np
import pytest

from config import load_runtime_config
from wake_tuning.detector import WakeDecisionTracker
from wake_tuning.settings import (
    WakeTuningError,
    WakeTuningSettings,
    load_profile_settings,
    save_profile_settings,
)


def _profile_body() -> str:
    return """
[profiles.hybrid_low_latency]
category = "local_debug"
[profiles.hybrid_low_latency.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.6
vad_threshold = 0.0
candidate_log_threshold = 0.5
required_hits = 1
min_wake_rms = 35.0
min_wake_peak = 100
rearm_delay_s = 6.0
pre_buffer_s = 0.5
[profiles.hybrid_low_latency.emergency_stop]
enabled = false
[profiles.hybrid_low_latency.stt]
provider = "whisper"
model = "base"
[profiles.hybrid_low_latency.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.hybrid_low_latency.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.hybrid_low_latency.metrics]
enabled = false
""".strip()


def test_saved_tuning_overrides_runtime_config_wake_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profiles_path = tmp_path / "runtime_profiles.toml"
    profiles_path.write_text(_profile_body(), encoding="utf-8")
    settings_path = tmp_path / "wake_tuning_settings.json"
    monkeypatch.setenv("WAKE_TUNING_SETTINGS_PATH", str(settings_path))
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "oa")

    save_profile_settings(
        settings_path,
        "hybrid_low_latency",
        WakeTuningSettings(
            threshold=0.42,
            vad_threshold=0.2,
            candidate_log_threshold=0.31,
            required_hits=2,
            min_wake_rms=12.0,
            min_wake_peak=40,
            rearm_delay_s=1.5,
            pre_buffer_s=0.3,
        ),
    )

    config = load_runtime_config(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert config.wake.threshold == 0.42
    assert config.wake.vad_threshold == 0.2
    assert config.wake.required_hits == 2
    assert config.wake.min_wake_rms == 12.0
    assert config.wake.min_wake_peak == 40
    assert config.wake.rearm_delay_s == 1.5
    assert config.wake.pre_buffer_s == 0.3


def test_settings_round_trip_per_profile(tmp_path: Path) -> None:
    settings_path = tmp_path / "wake_tuning_settings.json"
    settings = WakeTuningSettings(
        threshold=0.5,
        vad_threshold=0.0,
        candidate_log_threshold=0.4,
        required_hits=1,
        min_wake_rms=0.0,
        min_wake_peak=0,
        rearm_delay_s=0.75,
        pre_buffer_s=0.5,
    )

    save_profile_settings(settings_path, "local_current", settings)

    assert load_profile_settings(settings_path, "local_current") == settings
    assert load_profile_settings(settings_path, "missing") is None


def test_settings_reject_invalid_threshold() -> None:
    with pytest.raises(WakeTuningError, match="threshold"):
        WakeTuningSettings.from_mapping(
            {
                "threshold": 1.2,
                "vad_threshold": 0.0,
                "candidate_log_threshold": 0.4,
                "required_hits": 1,
                "min_wake_rms": 0.0,
                "min_wake_peak": 0,
                "rearm_delay_s": 0.75,
                "pre_buffer_s": 0.5,
            }
        )


def test_decision_tracker_requires_consecutive_hits_and_audio_guards() -> None:
    settings = WakeTuningSettings(
        threshold=0.5,
        vad_threshold=0.0,
        candidate_log_threshold=0.4,
        required_hits=2,
        min_wake_rms=10.0,
        min_wake_peak=20,
        rearm_delay_s=0.75,
        pre_buffer_s=0.5,
    )
    tracker = WakeDecisionTracker(settings)
    quiet = np.zeros(1280, dtype=np.int16)
    audible = np.full(1280, 30, dtype=np.int16)

    rejected = tracker.evaluate({"mave": 0.8}, quiet)
    first = tracker.evaluate({"mave": 0.8}, audible)
    second = tracker.evaluate({"mave": 0.8}, audible)

    assert rejected.detected is False
    assert rejected.decision == "audio_level"
    assert rejected.threshold_hit is True
    assert rejected.level_hit is False
    assert first.detected is False
    assert first.decision == "waiting_for_hits"
    assert first.hits == 1
    assert second.detected is True
    assert second.decision == "triggered"


def test_decision_tracker_reports_below_threshold() -> None:
    settings = WakeTuningSettings(
        threshold=0.5,
        vad_threshold=0.0,
        candidate_log_threshold=0.4,
        required_hits=1,
        min_wake_rms=0.0,
        min_wake_peak=0,
        rearm_delay_s=0.75,
        pre_buffer_s=0.5,
    )
    tracker = WakeDecisionTracker(settings)

    result = tracker.evaluate({"mave": 0.49}, np.full(1280, 30, dtype=np.int16))

    assert result.detected is False
    assert result.decision == "below_threshold"
    assert result.threshold_hit is False
    assert result.level_hit is True
