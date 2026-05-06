from pathlib import Path

import pytest

from voice_runtime.profiles import (
    AgentProfile,
    EmergencyStopProfile,
    MetricsProfile,
    ProfileError,
    RuntimeProfile,
    STTProfile,
    TTSProfile,
    WakeProfile,
    default_profiles_path,
    load_runtime_profile,
)


def _write_profiles(path: Path) -> None:
    path.write_text(
        """
[profiles.hybrid_low_latency]
category = "benchmark_streaming"
[profiles.hybrid_low_latency.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
vad_threshold = 0.3
candidate_log_threshold = 0.3
required_hits = 2
pre_buffer_s = 1.5
single_command = true
[profiles.hybrid_low_latency.emergency_stop]
enabled = false
[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
model = "flux-general-en"
[profiles.hybrid_low_latency.tts]
provider = "cartesia"
model = "sonic-3"
voice = "voice-id"
[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.hybrid_low_latency.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.no_wake_debug]
category = "local_debug"
[profiles.no_wake_debug.wake]
provider = "none"
[profiles.no_wake_debug.emergency_stop]
enabled = false
[profiles.no_wake_debug.stt]
provider = "whisper"
model = "base"
device = "cpu"
[profiles.no_wake_debug.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.no_wake_debug.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.no_wake_debug.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.no_wake_debug.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )


def _write_profile(path: Path, body: str) -> None:
    path.write_text(body.strip(), encoding="utf-8")


def test_bundled_streaming_profiles_keep_wake_prebuffer_short() -> None:
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_low_latency",
    )

    assert profile.wake.pre_buffer_s <= 0.5


def test_loads_profile_without_constructing_adapters(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert profile.name == "hybrid_low_latency"
    assert profile.category == "benchmark_streaming"
    assert profile.wake.provider == "openwakeword"
    assert profile.wake.model_path == tmp_path / "models" / "mave.onnx"
    assert profile.wake.threshold == 0.5
    assert profile.wake.vad_threshold == 0.3
    assert profile.wake.candidate_log_threshold == 0.3
    assert profile.wake.required_hits == 2
    assert profile.wake.pre_buffer_s == 1.5
    assert profile.wake.single_command is True
    assert profile.emergency_stop.enabled is False
    assert profile.stt.provider == "deepgram_flux"
    assert profile.tts.provider == "cartesia"
    assert profile.agent.provider == "openai_codex_oauth"
    assert profile.agent.reasoning_effort == "medium"
    assert profile.mcp_robot_url == "http://127.0.0.1:8765/mcp"
    assert profile.metrics.path == tmp_path / "logs" / "voice_metrics.jsonl"


def test_profile_exports_typed_dataclasses(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert isinstance(profile, RuntimeProfile)
    assert isinstance(profile.wake, WakeProfile)
    assert isinstance(profile.emergency_stop, EmergencyStopProfile)
    assert isinstance(profile.stt, STTProfile)
    assert isinstance(profile.tts, TTSProfile)
    assert isinstance(profile.agent, AgentProfile)
    assert isinstance(profile.metrics, MetricsProfile)


def test_profile_reports_required_env_names_without_reading_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    def fail_getenv(name: str, default: str | None = None) -> str | None:
        raise AssertionError(f"unexpected os.environ read for {name}")

    monkeypatch.setattr("os.getenv", fail_getenv)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert profile.required_env_names() == ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY")


def test_cartesia_profile_without_voice_requires_voice_id_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.cartesia_without_voice]
category = "benchmark_streaming"
[profiles.cartesia_without_voice.wake]
provider = "none"
[profiles.cartesia_without_voice.emergency_stop]
enabled = false
[profiles.cartesia_without_voice.stt]
provider = "deepgram_flux"
[profiles.cartesia_without_voice.tts]
provider = "cartesia"
model = "sonic-3"
[profiles.cartesia_without_voice.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.cartesia_without_voice.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.cartesia_without_voice.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="cartesia_without_voice",
    )

    assert profile.required_env_names() == (
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "CARTESIA_VOICE_ID",
    )


def test_required_env_names_are_deterministic_and_unique(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.cloud_mix]
category = "local_debug"
[profiles.cloud_mix.wake]
provider = "none"
[profiles.cloud_mix.emergency_stop]
enabled = false
[profiles.cloud_mix.stt]
provider = "openai_realtime"
[profiles.cloud_mix.tts]
provider = "deepgram"
[profiles.cloud_mix.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.cloud_mix.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.cloud_mix.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="cloud_mix",
    )

    assert profile.required_env_names() == ("DEEPGRAM_API_KEY", "OPENAI_API_KEY")


def test_local_profile_has_no_cloud_stt_tts_env_requirements(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="no_wake_debug",
    )

    assert profile.required_env_names() == ()


def test_agent_reasoning_effort_rejects_invalid_value(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    profiles_path.write_text(
        profiles_path.read_text(encoding="utf-8").replace(
            'reasoning_effort = "medium"', 'reasoning_effort = "maximum"'
        ),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="reasoning_effort must be one of"):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="hybrid_low_latency",
        )


def test_legacy_claude_agent_provider_is_rejected(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.legacy]
category = "local_debug"
[profiles.legacy.wake]
provider = "none"
[profiles.legacy.emergency_stop]
enabled = false
[profiles.legacy.stt]
provider = "whisper"
model = "base"
[profiles.legacy.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.legacy.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
[profiles.legacy.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.legacy.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="provider must be one of"):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="legacy",
        )


def test_benchmark_profile_rejects_local_stt(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "benchmark_streaming"
[profiles.bad.wake]
provider = "none"
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
[profiles.bad.tts]
provider = "cartesia"
voice = "voice-id"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="benchmark_streaming profiles require streaming STT"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_benchmark_profile_rejects_local_tts(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "benchmark_streaming"
[profiles.bad.wake]
provider = "none"
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "deepgram_flux"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="benchmark_streaming profiles require streaming TTS"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_enabled_wake_requires_model_path(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "openwakeword"
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="wake.model_path"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_enabled_emergency_stop_requires_provider_and_model_path(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "none"
[profiles.bad.emergency_stop]
enabled = true
provider = "none"
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="emergency_stop.provider"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")

    profiles_path.write_text(
        profiles_path.read_text(encoding="utf-8").replace('provider = "none"\n[profiles.bad.stt]', 'provider = "openwakeword"\n[profiles.bad.stt]'),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="emergency_stop.model_path"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_wake_threshold_rejects_boolean(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "none"
threshold = true
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="threshold must be a number"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_wake_vad_threshold_rejects_boolean(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "none"
vad_threshold = true
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="vad_threshold must be a number"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_wake_required_hits_rejects_values_below_one(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "none"
required_hits = 0
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="required_hits must be at least 1"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")


def test_default_profile_path_and_name_load_current_app_profile():
    profile = load_runtime_profile()

    assert profile.name == "hybrid_low_latency"
    assert profile.wake.model_path is not None
    assert profile.wake.model_path.name == "mave.onnx"
    assert profile.tts.voice == "47c38ca4-5f35-497b-b1a3-415245fb35e1"
