from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from voice_modulation.settings import (
    BUILT_IN_PRESETS,
    VoiceModulationError,
    apply_saved_voice_modulation,
    default_settings_path,
    load_profile_settings,
    save_profile_settings,
    settings_from_mapping,
)
from voice_runtime.profiles import (
    AgentProfile,
    EmergencyStopProfile,
    MetricsProfile,
    ProcessTraceProfile,
    RuntimeProfile,
    STTProfile,
    TTSProfile,
    WakeProfile,
)


def _settings_mapping() -> dict[str, object]:
    return {
        "enabled": True,
        "preset_name": "robot",
        "gain_db": 3.0,
        "wet_mix": 0.8,
        "low_cut_hz": 120.0,
        "high_cut_hz": 3600.0,
        "drive": 0.35,
        "bit_depth": 8,
        "ring_mod_hz": 45.0,
        "tremolo_hz": 5.0,
        "tremolo_depth": 0.4,
        "limiter": True,
    }


def _profile(tmp_path: Path) -> RuntimeProfile:
    return RuntimeProfile(
        name="hybrid_low_latency",
        category="local_debug",
        wake=WakeProfile(provider="none", model_path=None),
        emergency_stop=EmergencyStopProfile(enabled=False),
        stt=STTProfile(provider="whisper", model="base", device="cpu"),
        tts=TTSProfile(provider="kokoro", voice="af_heart"),
        agent=AgentProfile(provider="openai_api", model="gpt-5.4-mini"),
        mcp_robot_url="http://127.0.0.1:8765/mcp",
        metrics=MetricsProfile(enabled=False, path=tmp_path / "metrics.jsonl", include_text=False),
        process_trace=ProcessTraceProfile(enabled=False, path=tmp_path / "trace.jsonl"),
        server_dir=tmp_path,
    )


def test_default_settings_path_uses_local_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VOICE_MODULATION_SETTINGS_PATH", raising=False)

    assert default_settings_path(tmp_path) == tmp_path / "state" / "voice_modulation_settings.json"


def test_missing_settings_file_returns_disabled_clean_preset(tmp_path: Path) -> None:
    settings = load_profile_settings(
        "hybrid_low_latency",
        server_dir=tmp_path,
        settings_path=tmp_path / "missing.json",
    )

    assert settings == BUILT_IN_PRESETS["clean"]
    assert settings.enabled is False


def test_settings_round_trip_per_profile(tmp_path: Path) -> None:
    path = tmp_path / "voice_modulation_settings.json"
    settings = settings_from_mapping(_settings_mapping())

    saved_path = save_profile_settings("hybrid_low_latency", settings, settings_path=path)

    assert saved_path == path
    assert load_profile_settings("hybrid_low_latency", settings_path=path) == settings
    assert load_profile_settings("missing", settings_path=path) == BUILT_IN_PRESETS["clean"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["profiles"]["hybrid_low_latency"]["preset_name"] == "robot"


def test_settings_from_mapping_validates_ranges() -> None:
    settings = settings_from_mapping(_settings_mapping())

    assert settings.to_dict() == _settings_mapping()

    bad = _settings_mapping() | {"bit_depth": 3}
    with pytest.raises(VoiceModulationError, match="bit_depth"):
        settings_from_mapping(bad)


def test_settings_from_mapping_rejects_boolean_numbers() -> None:
    bad = _settings_mapping() | {"gain_db": True}

    with pytest.raises(VoiceModulationError, match="gain_db"):
        settings_from_mapping(bad)


def test_apply_saved_voice_modulation_sets_runtime_profile_field(tmp_path: Path) -> None:
    settings_path = tmp_path / "voice_modulation_settings.json"
    settings = replace(BUILT_IN_PRESETS["robot"], gain_db=6.0)
    save_profile_settings("hybrid_low_latency", settings, settings_path=settings_path)

    profile = apply_saved_voice_modulation(_profile(tmp_path), settings_path=settings_path)

    assert profile.voice_modulation == settings
