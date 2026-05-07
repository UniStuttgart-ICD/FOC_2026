from pathlib import Path

import pytest

from voice_runtime.profiles import (
    AgentProfile,
    EmergencyStopProfile,
    MetricsProfile,
    ProcessTraceProfile,
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
provider = "openai_api"
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
provider = "openai_api"
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


def test_bundled_default_profile_keeps_short_wake_word_activation_usable() -> None:
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_low_latency",
    )

    assert profile.wake.threshold == 0.85
    assert profile.wake.vad_threshold == 0.0
    assert profile.wake.required_hits == 1
    assert profile.wake.min_wake_rms <= 4.7
    assert profile.wake.min_wake_peak <= 17


def test_bundled_default_profile_uses_gemini_flash_lite_high_reasoning_agent():
    profile = load_runtime_profile()

    assert profile.agent.provider == "gemini_api"
    assert profile.agent.model == "gemini-3.1-flash-lite-preview"
    assert profile.agent.reasoning_effort == "high"
    assert profile.agent.api_key_env == "GOOGLE_API_KEY"


def test_bundled_gemini_profile_is_available():
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_gemini",
    )

    assert profile.agent.provider == "gemini_api"
    assert profile.agent.model.startswith("gemini-")
    assert profile.agent.api_key_env in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}


def test_bundled_anthropic_profile_is_available():
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_anthropic",
    )

    assert profile.agent.provider == "anthropic_api"
    assert profile.agent.model.startswith("claude-")
    assert profile.agent.reasoning_effort == "medium"
    assert profile.agent.api_key_env == "ANTHROPIC_API_KEY"


def test_wake_profile_parses_audio_guards_and_rearm_delay(tmp_path: Path) -> None:
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.guarded]
category = "local_debug"
[profiles.guarded.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.7
vad_threshold = 0.0
candidate_log_threshold = 0.5
required_hits = 1
min_wake_rms = 50.0
min_wake_peak = 150
rearm_delay_s = 6.0
[profiles.guarded.emergency_stop]
enabled = false
[profiles.guarded.stt]
provider = "whisper"
model = "base"
[profiles.guarded.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.guarded.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
[profiles.guarded.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.guarded.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="guarded",
    )

    assert profile.wake.min_wake_rms == 50.0
    assert profile.wake.min_wake_peak == 150
    assert profile.wake.rearm_delay_s == 6.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_wake_rms", "-1"),
        ("min_wake_peak", "-1"),
        ("rearm_delay_s", "-0.1"),
        ("min_wake_rms", "true"),
        ("min_wake_peak", "false"),
        ("rearm_delay_s", "true"),
    ],
)
def test_wake_profile_rejects_invalid_audio_guard_values(
    tmp_path: Path, field: str, value: str
) -> None:
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        f"""
[profiles.bad]
category = "local_debug"
[profiles.bad.wake]
provider = "none"
{field} = {value}
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
model = "base"
[profiles.bad.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match=field):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="bad",
        )


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
    assert profile.agent.provider == "openai_api"
    assert profile.agent.reasoning_effort == "medium"
    assert profile.mcp_robot_url == "http://127.0.0.1:8765/mcp"
    assert profile.metrics.path == tmp_path / "logs" / "voice_metrics.jsonl"
    assert profile.process_trace == ProcessTraceProfile(
        enabled=True,
        path=tmp_path / "logs" / "process_trace.jsonl",
        include_text=True,
        include_tool_payloads=True,
    )


def test_profile_parses_explicit_process_trace_section(tmp_path: Path) -> None:
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.local]
category = "local_debug"
[profiles.local.wake]
provider = "none"
[profiles.local.emergency_stop]
enabled = false
[profiles.local.stt]
provider = "whisper"
model = "base"
[profiles.local.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.local.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
[profiles.local.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.local.metrics]
enabled = false
[profiles.local.process_trace]
enabled = false
path = "traces/process.jsonl"
include_text = false
include_tool_payloads = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="local",
    )

    assert profile.process_trace == ProcessTraceProfile(
        enabled=False,
        path=tmp_path / "traces" / "process.jsonl",
        include_text=False,
        include_tool_payloads=False,
    )


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
    assert isinstance(profile.process_trace, ProcessTraceProfile)


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

    assert profile.required_env_names() == (
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "OPENAI_API_KEY",
    )


def test_openai_api_agent_profile_requires_openai_key_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.openai_api]
category = "local_debug"
[profiles.openai_api.wake]
provider = "none"
[profiles.openai_api.emergency_stop]
enabled = false
[profiles.openai_api.stt]
provider = "whisper"
[profiles.openai_api.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.openai_api.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "low"
[profiles.openai_api.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.openai_api.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="openai_api",
    )

    assert profile.agent.provider == "openai_api"
    assert profile.agent.api_key_env == "OPENAI_API_KEY"
    assert profile.agent.reasoning_effort == "low"
    assert profile.required_env_names() == ("OPENAI_API_KEY",)


def test_gemini_api_agent_profile_accepts_thinking_budget_and_key_override(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.gemini_api]
category = "local_debug"
[profiles.gemini_api.wake]
provider = "none"
[profiles.gemini_api.emergency_stop]
enabled = false
[profiles.gemini_api.stt]
provider = "whisper"
[profiles.gemini_api.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.gemini_api.agent]
provider = "gemini_api"
model = "gemini-2.5-flash"
reasoning_effort = "medium"
thinking_budget = 1024
api_key_env = "GEMINI_API_KEY"
[profiles.gemini_api.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.gemini_api.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="gemini_api",
    )

    assert profile.agent.provider == "gemini_api"
    assert profile.agent.api_key_env == "GEMINI_API_KEY"
    assert profile.agent.thinking_budget == 1024
    assert profile.required_env_names() == ("GEMINI_API_KEY",)


def test_gemini_25_pro_rejects_disabled_thinking(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad_gemini]
category = "local_debug"
[profiles.bad_gemini.wake]
provider = "none"
[profiles.bad_gemini.emergency_stop]
enabled = false
[profiles.bad_gemini.stt]
provider = "whisper"
[profiles.bad_gemini.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad_gemini.agent]
provider = "gemini_api"
model = "gemini-2.5-pro"
thinking_budget = 0
[profiles.bad_gemini.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad_gemini.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="gemini-2.5-pro cannot disable thinking"):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="bad_gemini",
        )


def test_anthropic_api_agent_profile_uses_default_key_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.anthropic_api]
category = "local_debug"
[profiles.anthropic_api.wake]
provider = "none"
[profiles.anthropic_api.emergency_stop]
enabled = false
[profiles.anthropic_api.stt]
provider = "whisper"
[profiles.anthropic_api.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.anthropic_api.agent]
provider = "anthropic_api"
model = "claude-sonnet-4-6"
reasoning_effort = "medium"
[profiles.anthropic_api.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.anthropic_api.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="anthropic_api",
    )

    assert profile.agent.provider == "anthropic_api"
    assert profile.agent.api_key_env == "ANTHROPIC_API_KEY"
    assert profile.required_env_names() == ("ANTHROPIC_API_KEY",)


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
provider = "openai_api"
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
        "OPENAI_API_KEY",
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
provider = "openai_api"
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


def test_local_profile_requires_only_agent_api_key(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="no_wake_debug",
    )

    assert profile.required_env_names() == ("OPENAI_API_KEY",)


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
provider = "openai_api"
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
provider = "openai_api"
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
provider = "openai_api"
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
provider = "openai_api"
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
provider = "openai_api"
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
provider = "openai_api"
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
provider = "openai_api"
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
