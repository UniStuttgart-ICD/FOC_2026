from pathlib import Path

import pytest

from config import (
    AgentConfig,
    ConfigError,
    RuntimeConfig,
    load_runtime_config,
)


def test_loads_default_hybrid_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.hybrid_low_latency]
category = "benchmark_streaming"

[profiles.hybrid_low_latency.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
pre_buffer_s = 1.5
threshold = 0.5

[profiles.hybrid_low_latency.emergency_stop]
enabled = false

[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
model = "flux-general-en"

[profiles.hybrid_low_latency.tts]
provider = "cartesia"
model = "sonic-3"
voice = "test-voice"

[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.hybrid_low_latency.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")

    config = load_runtime_config(profiles_path=profiles, server_dir=tmp_path)

    assert config.profile_name == "hybrid_low_latency"
    assert config.category == "benchmark_streaming"
    assert config.wake.provider == "openwakeword"
    assert config.wake.model_path == tmp_path / "models" / "mave.onnx"
    assert config.wake.pre_buffer_s == 1.5
    assert config.stt.provider == "deepgram_flux"
    assert config.tts.provider == "cartesia"
    assert config.agent == AgentConfig(provider="openai_codex_oauth", model="gpt-5.5")
    assert config.mcp_robot_url == "http://127.0.0.1:8765/mcp"
    assert config.metrics.include_text is True


def test_cli_profile_overrides_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.hybrid_low_latency]
category = "benchmark_streaming"
[profiles.hybrid_low_latency.wake]
provider = "none"
[profiles.hybrid_low_latency.emergency_stop]
enabled = false
[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
[profiles.hybrid_low_latency.tts]
provider = "cartesia"
[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"
[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.hybrid_low_latency.metrics]
enabled = false

[profiles.local_current]
category = "local_debug"
[profiles.local_current.wake]
provider = "none"
[profiles.local_current.emergency_stop]
enabled = false
[profiles.local_current.stt]
provider = "whisper"
model = "base"
[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.local_current.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("VOICE_PROFILE", "local_current")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    monkeypatch.setenv("CARTESIA_VOICE_ID", "voice")

    config = load_runtime_config(
        profiles_path=profiles,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert config.profile_name == "hybrid_low_latency"


def test_missing_api_key_fails_for_default_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.hybrid_low_latency]
category = "benchmark_streaming"
[profiles.hybrid_low_latency.wake]
provider = "none"
[profiles.hybrid_low_latency.emergency_stop]
enabled = false
[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
[profiles.hybrid_low_latency.tts]
provider = "cartesia"
[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"
[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.hybrid_low_latency.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)

    with pytest.raises(ConfigError, match="DEEPGRAM_API_KEY"):
        load_runtime_config(profiles_path=profiles, server_dir=tmp_path)


def test_emergency_stop_requires_model_when_enabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.local_current]
category = "local_debug"
[profiles.local_current.wake]
provider = "none"
[profiles.local_current.emergency_stop]
enabled = true
provider = "openwakeword"
[profiles.local_current.stt]
provider = "whisper"
model = "base"
[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.local_current.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="emergency_stop.model_path"):
        load_runtime_config(
            profiles_path=profiles,
            server_dir=tmp_path,
            profile_name="local_current",
        )


def test_wake_threshold_rejects_boolean(tmp_path: Path):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.local_current]
category = "local_debug"
[profiles.local_current.wake]
provider = "none"
threshold = true
[profiles.local_current.emergency_stop]
enabled = false
[profiles.local_current.stt]
provider = "whisper"
model = "base"
[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.local_current.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="threshold must be a number"):
        load_runtime_config(
            profiles_path=profiles,
            server_dir=tmp_path,
            profile_name="local_current",
        )


def test_emergency_stop_requires_provider_when_enabled(tmp_path: Path):
    profiles = tmp_path / "runtime_profiles.toml"
    profiles.write_text(
        """
[profiles.local_current]
category = "local_debug"
[profiles.local_current.wake]
provider = "none"
[profiles.local_current.emergency_stop]
enabled = true
provider = "none"
model_path = "models/stop.onnx"
[profiles.local_current.stt]
provider = "whisper"
model = "base"
[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.local_current.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="emergency_stop.provider"):
        load_runtime_config(
            profiles_path=profiles,
            server_dir=tmp_path,
            profile_name="local_current",
        )
