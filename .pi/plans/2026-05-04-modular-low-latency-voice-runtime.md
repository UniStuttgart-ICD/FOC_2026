# Modular Low-Latency Voice Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modular, profile-driven Pipecat voice runtime with Mave wake word, low-latency streaming providers, OpenAI Codex OAuth agent support, emergency stop bypass scaffolding, and JSONL latency metrics.

**Architecture:** Keep Pipecat as the runtime. Add typed config/profile loading, simple provider factory functions, a standalone OpenWakeWord gate before STT, agent processor selection, observer/event-based metrics, and a pipeline builder that keeps `bot.py` thin. The implementation is split into parallelizable issues after the shared config contracts are defined.

**Tech Stack:** Python 3.10+, Pipecat 0.0.106, OpenWakeWord, Deepgram Flux, Cartesia Sonic, OpenAI Realtime STT/TTS, OpenAI Agents SDK/Codex OAuth, Claude Agent SDK fallback, pytest, ruff, pyright, TOML via `tomllib`/`tomli`.

---

## Execution Model

This plan is intentionally split into issues for parallel execution by subagents.

- **Issue 1 is the required foundation** and must land first.
- **Issues 2–6 can run in parallel** after Issue 1.
- **Issue 7 integrates all prior work** and should run after Issues 2–6.
- **Issue 8 docs can start after Issue 1**, then receive a final pass after Issue 7.

Recommended execution:

```text
Issue 1
├── Issue 2 Provider factories
├── Issue 3 Agent/OpenAI Codex provider
├── Issue 4 Wake gate
├── Issue 5 Emergency stop scaffold
└── Issue 6 Metrics
Issue 7 Pipeline integration
Issue 8 Docs/benchmarking
```

Each issue should be executed in its own subagent branch/worktree or in strict sequence with review checkpoints.

## Current Verified Facts

- Current repo root for implementation: `pipecat-agent/`.
- Current server path: `pipecat-agent/server/`.
- Current pipeline in `server/bot.py`:
  ```text
  transport.input() → WhisperSTTService → user_aggregator → ClaudeAgentProcessor → KokoroTTSService → transport.output() → assistant_aggregator
  ```
- Pipecat version reports `0.0.106`.
- Input audio frame class exists: `pipecat.frames.frames.InputAudioRawFrame(audio: bytes, sample_rate: int, num_channels: int)`.
- TTS audio frame class exists: `TTSAudioRawFrame(audio: bytes, sample_rate: int, num_channels: int, context_id: Optional[str] = None)`.
- Transcription frame class exists: `TranscriptionFrame(text, user_id, timestamp, language=None, result=None, finalized=False)`.
- Pipecat observer base exists: `pipecat.observers.base_observer.BaseObserver`.
- Deepgram extras are not currently installed; importing Deepgram services reported: `pip install pipecat-ai[deepgram]`.
- Cartesia and OpenAI service modules are importable in the current env, but dependencies should still be declared explicitly.
- Trained wake model source: `C:/Users/Samuel/Documents/github/DF2025_CLEAN/models/mave.onnx`.

## Target File Structure

```text
pipecat-agent/
├── .pi/plans/2026-05-04-modular-low-latency-voice-runtime.md
├── docs/benchmarking.md
├── server/
│   ├── bot.py
│   ├── config.py
│   ├── runtime_profiles.toml
│   ├── providers.py
│   ├── agent_processor_factory.py
│   ├── pipeline_builder.py
│   ├── metrics.py
│   ├── codex_auth.py
│   ├── openai_codex_agent_processor.py
│   ├── claude_agent_processor.py
│   ├── prompts.py
│   ├── models/mave.onnx
│   ├── wake/
│   │   ├── __init__.py
│   │   ├── openwakeword_detector.py
│   │   ├── wake_gate.py
│   │   ├── emergency_stop.py
│   │   └── transcript_cleanup.py
│   └── tests/
│       ├── test_config.py
│       ├── test_providers.py
│       ├── test_agent_processor_factory.py
│       ├── test_codex_auth.py
│       ├── test_wake_gate.py
│       ├── test_transcript_cleanup.py
│       ├── test_emergency_stop.py
│       └── test_metrics.py
```

---

# Issue 1: Runtime Config Foundation

**Parallelization:** Required foundation. Complete before Issues 2–6.

**Files:**
- Modify: `server/pyproject.toml`
- Create: `server/config.py`
- Create: `server/runtime_profiles.toml`
- Create: `server/tests/test_config.py`
- Modify: `server/.env.example`

## Task 1.1: Add dependencies

- [ ] **Step 1: Modify `server/pyproject.toml` dependencies**

Replace the dependency list with:

```toml
dependencies = [
    "pipecat-ai[cartesia,deepgram,kokoro,openai,runner,silero,webrtc,whisper]",
    "claude-agent-sdk<0.1.49",
    "openai-agents>=0.14.0,<1",
    "openai>=2.29.0,<3",
    "httpx>=0.28.0,<1",
    "openwakeword>=0.6.0,<1",
    "silero-vad>=6.0.0,<7",
    "tomli>=2.0.0; python_version < '3.11'",
]
```

Replace the dev dependency group with:

```toml
[dependency-groups]
dev = [
    "pyright>=1.1.404,<2",
    "ruff>=0.12.11,<1",
    "pytest>=8.0.0,<9",
    "pytest-asyncio>=0.24.0,<1",
]
```

- [ ] **Step 2: Install dependencies**

Run:

```bash
cd pipecat-agent/server
uv sync
```

Expected: dependency resolution succeeds and `uv.lock` changes.

- [ ] **Step 3: Commit dependency update**

```bash
cd pipecat-agent
git add server/pyproject.toml server/uv.lock
git commit -m "chore: add low latency voice runtime dependencies"
```

## Task 1.2: Write failing config tests

- [ ] **Step 1: Create `server/tests/test_config.py`**

```python
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
provider = "claude"
model = "claude-haiku-4-5-20251001"
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
provider = "claude"
model = "claude-haiku-4-5-20251001"
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

## Task 1.3: Implement config loader

- [ ] **Step 1: Create `server/config.py`**

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_PROFILE = "hybrid_low_latency"

WakeProvider = Literal["none", "openwakeword"]
STTProvider = Literal["deepgram_flux", "openai_realtime", "whisper"]
TTSProvider = Literal["cartesia", "openai", "deepgram", "kokoro"]
AgentProvider = Literal["claude", "openai_codex_oauth"]
Category = Literal["benchmark_streaming", "local_debug"]


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


@dataclass(frozen=True)
class WakeConfig:
    provider: WakeProvider
    model_path: Path | None
    threshold: float = 0.5
    candidate_log_threshold: float = 0.3
    pre_buffer_s: float = 1.5
    single_command: bool = True


@dataclass(frozen=True)
class EmergencyStopConfig:
    enabled: bool
    provider: WakeProvider = "none"
    model_path: Path | None = None
    threshold: float = 0.5


@dataclass(frozen=True)
class STTConfig:
    provider: STTProvider
    model: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class TTSConfig:
    provider: TTSProvider
    model: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class AgentConfig:
    provider: AgentProvider
    model: str


@dataclass(frozen=True)
class MetricsConfig:
    enabled: bool
    path: Path
    include_text: bool


@dataclass(frozen=True)
class RuntimeConfig:
    profile_name: str
    category: Category
    wake: WakeConfig
    emergency_stop: EmergencyStopConfig
    stt: STTConfig
    tts: TTSConfig
    agent: AgentConfig
    mcp_robot_url: str
    metrics: MetricsConfig
    server_dir: Path


def default_profiles_path(server_dir: Path | None = None) -> Path:
    root = server_dir or Path(__file__).resolve().parent
    return root / "runtime_profiles.toml"


def load_runtime_config(
    *,
    profiles_path: str | Path | None = None,
    server_dir: str | Path | None = None,
    profile_name: str | None = None,
) -> RuntimeConfig:
    server_root = Path(server_dir) if server_dir is not None else Path(__file__).resolve().parent
    selected_profile = profile_name or os.getenv("VOICE_PROFILE") or DEFAULT_PROFILE
    path = Path(profiles_path) if profiles_path is not None else default_profiles_path(server_root)
    if not path.exists():
        raise ConfigError(f"Runtime profiles file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    profiles = _table(data, "profiles")
    raw_profile = profiles.get(selected_profile)
    if not isinstance(raw_profile, dict):
        raise ConfigError(f"Unknown VOICE_PROFILE '{selected_profile}' in {path}")

    category = _literal(raw_profile, "category", {"benchmark_streaming", "local_debug"})
    wake = _parse_wake(_table(raw_profile, "wake"), server_root)
    emergency_stop = _parse_emergency_stop(_table(raw_profile, "emergency_stop"), server_root)
    stt = _parse_stt(_table(raw_profile, "stt"))
    tts = _parse_tts(_table(raw_profile, "tts"))
    agent = _parse_agent(_table(raw_profile, "agent"))
    mcp = _table(raw_profile, "mcp")
    robot = _table(mcp, "robot")
    metrics = _parse_metrics(_table(raw_profile, "metrics"), server_root)

    config = RuntimeConfig(
        profile_name=selected_profile,
        category=category,  # type: ignore[arg-type]
        wake=wake,
        emergency_stop=emergency_stop,
        stt=stt,
        tts=tts,
        agent=agent,
        mcp_robot_url=_string(robot, "url"),
        metrics=metrics,
        server_dir=server_root,
    )
    _validate_runtime_config(config)
    return config


def _parse_wake(table: dict[str, Any], server_dir: Path) -> WakeConfig:
    provider = _literal(table, "provider", {"none", "openwakeword"})
    model_path = _optional_path(table, "model_path", server_dir)
    return WakeConfig(
        provider=provider,  # type: ignore[arg-type]
        model_path=model_path,
        threshold=_float(table, "threshold", 0.5),
        candidate_log_threshold=_float(table, "candidate_log_threshold", 0.3),
        pre_buffer_s=_float(table, "pre_buffer_s", 1.5),
        single_command=_bool(table, "single_command", True),
    )


def _parse_emergency_stop(table: dict[str, Any], server_dir: Path) -> EmergencyStopConfig:
    enabled = _bool(table, "enabled", False)
    provider = _literal(table, "provider", {"none", "openwakeword"}, default="none")
    model_path = _optional_path(table, "model_path", server_dir)
    return EmergencyStopConfig(
        enabled=enabled,
        provider=provider,  # type: ignore[arg-type]
        model_path=model_path,
        threshold=_float(table, "threshold", 0.5),
    )


def _parse_stt(table: dict[str, Any]) -> STTConfig:
    provider = _literal(table, "provider", {"deepgram_flux", "openai_realtime", "whisper"})
    return STTConfig(
        provider=provider,  # type: ignore[arg-type]
        model=_optional_string(table, "model"),
        device=_optional_string(table, "device"),
    )


def _parse_tts(table: dict[str, Any]) -> TTSConfig:
    provider = _literal(table, "provider", {"cartesia", "openai", "deepgram", "kokoro"})
    return TTSConfig(
        provider=provider,  # type: ignore[arg-type]
        model=_optional_string(table, "model"),
        voice=_optional_string(table, "voice"),
    )


def _parse_agent(table: dict[str, Any]) -> AgentConfig:
    provider = _literal(table, "provider", {"claude", "openai_codex_oauth"})
    default_model = "claude-haiku-4-5-20251001" if provider == "claude" else "gpt-5.5"
    return AgentConfig(provider=provider, model=_string(table, "model", default_model))  # type: ignore[arg-type]


def _parse_metrics(table: dict[str, Any], server_dir: Path) -> MetricsConfig:
    return MetricsConfig(
        enabled=_bool(table, "enabled", True),
        path=_path(table, "path", server_dir, "logs/voice_metrics.jsonl"),
        include_text=_bool(table, "include_text", True),
    )


def _validate_runtime_config(config: RuntimeConfig) -> None:
    if config.wake.provider == "openwakeword" and config.wake.model_path is None:
        raise ConfigError("wake.model_path is required when wake.provider = 'openwakeword'")
    if config.emergency_stop.enabled and config.emergency_stop.model_path is None:
        raise ConfigError("emergency_stop.model_path is required when emergency stop is enabled")
    if config.category == "benchmark_streaming":
        if config.stt.provider not in {"deepgram_flux", "openai_realtime"}:
            raise ConfigError("benchmark_streaming profiles require streaming STT")
        if config.tts.provider not in {"cartesia", "openai", "deepgram"}:
            raise ConfigError("benchmark_streaming profiles require streaming TTS")
    required_env = []
    if config.stt.provider == "deepgram_flux" or config.tts.provider == "deepgram":
        required_env.append("DEEPGRAM_API_KEY")
    if config.tts.provider == "cartesia":
        required_env.append("CARTESIA_API_KEY")
    if config.stt.provider == "openai_realtime" or config.tts.provider == "openai":
        required_env.append("OPENAI_API_KEY")
    missing = [name for name in required_env if not os.getenv(name)]
    if missing:
        raise ConfigError(
            f"Profile {config.profile_name} requires missing environment variable(s): "
            + ", ".join(missing)
        )


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table")
    return value


def _string(table: dict[str, Any], key: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def _literal(table: dict[str, Any], key: str, allowed: set[str], default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ConfigError(f"{key} must be one of {sorted(allowed)}")
    return value


def _bool(table: dict[str, Any], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be true or false")
    return value


def _float(table: dict[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number")
    return float(value)


def _path(table: dict[str, Any], key: str, server_dir: Path, default: str) -> Path:
    raw = _string(table, key, default)
    path = Path(raw)
    return path if path.is_absolute() else server_dir / path


def _optional_path(table: dict[str, Any], key: str, server_dir: Path) -> Path | None:
    raw = _optional_string(table, key)
    if raw is None:
        return None
    path = Path(raw)
    return path if path.is_absolute() else server_dir / path
```

- [ ] **Step 2: Run config tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_config.py -v
```

Expected: all tests pass.

## Task 1.4: Add runtime profiles file

- [ ] **Step 1: Create `server/runtime_profiles.toml`**

```toml
[profiles.hybrid_low_latency]
category = "benchmark_streaming"

[profiles.hybrid_low_latency.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true

[profiles.hybrid_low_latency.emergency_stop]
enabled = false
# Enable only after a local stop model exists.
# provider = "openwakeword"
# model_path = "models/stop.onnx"

[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
model = "flux-general-en"

[profiles.hybrid_low_latency.tts]
provider = "cartesia"
model = "sonic-3"
voice = "af_heart"

[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.hybrid_low_latency.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.openai_all]
category = "benchmark_streaming"

[profiles.openai_all.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true

[profiles.openai_all.emergency_stop]
enabled = false

[profiles.openai_all.stt]
provider = "openai_realtime"
model = "gpt-4o-mini-transcribe"

[profiles.openai_all.tts]
provider = "openai"
model = "gpt-4o-mini-tts"
voice = "coral"

[profiles.openai_all.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[profiles.openai_all.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.openai_all.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.deepgram_all]
category = "benchmark_streaming"

[profiles.deepgram_all.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true

[profiles.deepgram_all.emergency_stop]
enabled = false

[profiles.deepgram_all.stt]
provider = "deepgram_flux"
model = "flux-general-en"

[profiles.deepgram_all.tts]
provider = "deepgram"
model = "aura-2"
voice = "aura-2-andromeda-en"

[profiles.deepgram_all.agent]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[profiles.deepgram_all.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.deepgram_all.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.local_current]
category = "local_debug"

[profiles.local_current.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true

[profiles.local_current.emergency_stop]
enabled = false

[profiles.local_current.stt]
provider = "whisper"
model = "base"
device = "cuda"

[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"

[profiles.local_current.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"

[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.local_current.metrics]
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
device = "cuda"

[profiles.no_wake_debug.tts]
provider = "kokoro"
voice = "af_heart"

[profiles.no_wake_debug.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"

[profiles.no_wake_debug.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.no_wake_debug.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true
```

- [ ] **Step 2: Update `.env.example`**

Add near the top:

```dotenv
# Runtime profile selection
# Default is hybrid_low_latency. CLI --profile overrides VOICE_PROFILE.
# VOICE_PROFILE=hybrid_low_latency

# Low-latency profile keys
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=

# OpenAI realtime STT/TTS profile key
OPENAI_API_KEY=

# Claude fallback profile
CLAUDE_MODEL=claude-haiku-4-5-20251001

# Local Whisper fallback. OPENAI_MODEL is accepted as legacy fallback temporarily.
WHISPER_MODEL=base
```

- [ ] **Step 3: Commit Issue 1**

```bash
cd pipecat-agent
git add server/config.py server/runtime_profiles.toml server/tests/test_config.py server/.env.example
git commit -m "feat: add runtime profile config"
```

---

# Issue 2: Provider Factories

**Parallelization:** Can run after Issue 1.

**Files:**
- Create: `server/providers.py`
- Create: `server/tests/test_providers.py`

## Task 2.1: Write provider factory tests

- [ ] **Step 1: Create `server/tests/test_providers.py`**

```python
from unittest.mock import Mock, patch

from config import STTConfig, TTSConfig
from providers import create_stt_service, create_tts_service


def test_creates_whisper_stt():
    with patch("providers.WhisperSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="whisper", model="base", device="cuda"))

    service.Settings.assert_called_once_with(model="base")
    service.assert_called_once_with(device="cuda", settings="settings")


def test_creates_kokoro_tts():
    with patch("providers.KokoroTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="kokoro", voice="af_heart"))

    service.Settings.assert_called_once_with(voice="af_heart")
    service.assert_called_once_with(settings="settings")


def test_creates_deepgram_flux_stt(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("providers.DeepgramFluxSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="deepgram_flux", model="flux-general-en"))

    service.Settings.assert_called_once_with(model="flux-general-en")
    service.assert_called_once_with(api_key="dg", settings="settings")


def test_creates_openai_realtime_stt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime", model="gpt-4o-mini-transcribe"))

    service.Settings.assert_called_once_with(model="gpt-4o-mini-transcribe")
    service.assert_called_once_with(api_key="oa", settings="settings", noise_reduction="near_field")


def test_creates_cartesia_tts(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    with patch("providers.CartesiaTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="cartesia", model="sonic-3", voice="voice-id"))

    service.Settings.assert_called_once_with(model="sonic-3", voice="voice-id")
    service.assert_called_once_with(api_key="ct", settings="settings")


def test_creates_openai_tts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("providers.OpenAITTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="openai", model="gpt-4o-mini-tts", voice="coral"))

    service.Settings.assert_called_once_with(model="gpt-4o-mini-tts", voice="coral")
    service.assert_called_once_with(api_key="oa", settings="settings")


def test_creates_deepgram_tts(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("providers.DeepgramTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="deepgram", model="aura-2", voice="aura-2-andromeda-en"))

    service.Settings.assert_called_once_with(model="aura-2", voice="aura-2-andromeda-en")
    service.assert_called_once_with(api_key="dg", settings="settings")
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd pipecat-agent/server
uv run pytest tests/test_providers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'providers'`.

## Task 2.2: Implement providers

- [ ] **Step 1: Create `server/providers.py`**

```python
from __future__ import annotations

import os

from config import STTConfig, TTSConfig
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.whisper.stt import WhisperSTTService


def create_stt_service(config: STTConfig) -> FrameProcessor:
    if config.provider == "whisper":
        return WhisperSTTService(
            device=config.device or "cuda",
            settings=WhisperSTTService.Settings(
                model=config.model or os.getenv("WHISPER_MODEL") or os.getenv("OPENAI_MODEL") or "base",
            ),
        )
    if config.provider == "deepgram_flux":
        return DeepgramFluxSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramFluxSTTService.Settings(model=config.model or "flux-general-en"),
        )
    if config.provider == "openai_realtime":
        return OpenAIRealtimeSTTService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIRealtimeSTTService.Settings(
                model=config.model or "gpt-4o-mini-transcribe",
            ),
            noise_reduction="near_field",
        )
    raise ValueError(f"Unsupported STT provider: {config.provider}")


def create_tts_service(config: TTSConfig) -> FrameProcessor:
    if config.provider == "kokoro":
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=config.voice or os.getenv("KOKORO_VOICE_ID") or "af_heart"),
        )
    if config.provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaTTSService.Settings(
                model=config.model or "sonic-3",
                voice=config.voice or os.getenv("CARTESIA_VOICE_ID") or "af_heart",
            ),
        )
    if config.provider == "openai":
        return OpenAITTSService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAITTSService.Settings(
                model=config.model or "gpt-4o-mini-tts",
                voice=config.voice or "coral",
            ),
        )
    if config.provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(
                model=config.model or "aura-2",
                voice=config.voice or "aura-2-andromeda-en",
            ),
        )
    raise ValueError(f"Unsupported TTS provider: {config.provider}")
```

- [ ] **Step 2: Run provider tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_providers.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Commit Issue 2**

```bash
cd pipecat-agent
git add server/providers.py server/tests/test_providers.py
git commit -m "feat: add voice provider factories"
```

---

# Issue 3: Agent Provider Factory + OpenAI Codex OAuth

**Parallelization:** Can run after Issue 1. Independent from STT/TTS/wake.

**Files:**
- Create: `server/codex_auth.py`
- Create: `server/openai_codex_agent_processor.py`
- Create: `server/agent_processor_factory.py`
- Modify: `server/claude_agent_processor.py`
- Create: `server/tests/test_codex_auth.py`
- Create: `server/tests/test_agent_processor_factory.py`

## Task 3.1: Refactor Claude processor constructor

- [ ] **Step 1: Modify `server/claude_agent_processor.py` constructor**

Replace:

```python
    def __init__(self, mcp_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
```

with:

```python
    def __init__(self, mcp_server_url: str, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
```

- [ ] **Step 2: Run import check**

```bash
cd pipecat-agent/server
uv run python - <<'PY'
from claude_agent_processor import ClaudeAgentProcessor
p = ClaudeAgentProcessor("http://127.0.0.1:8765/mcp", model="test-model")
print(type(p).__name__)
PY
```

Expected: prints `ClaudeAgentProcessor`.

## Task 3.2: Add Codex auth tests and implementation

- [ ] **Step 1: Create `server/tests/test_codex_auth.py`**

Use the test content from `.pi/plans/2026-05-04-openai-codex-oauth-provider.md`, Task 2, with imports unchanged:

```python
import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from codex_auth import CodexAuthError, PiCodexCredentialStore


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_reads_existing_pi_openai_codex_profile(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expires = int(time.time() * 1000) + 60_000
    access = _jwt({"exp": int(time.time()) + 60, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth", "access": access, "refresh": "refresh-token", "expires": expires, "accountId": "acct-1"}}), encoding="utf-8")

    credentials = PiCodexCredentialStore(auth_file=auth_file).get_credentials()

    assert credentials.access == access
    assert credentials.refresh == "refresh-token"
    assert credentials.account_id == "acct-1"


def test_missing_auth_profile_explains_pi_login(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")

    with pytest.raises(CodexAuthError, match="Run `pi`, then `/login`, then select ChatGPT Plus/Pro"):
        PiCodexCredentialStore(auth_file=auth_file).get_credentials()


def test_refreshes_expired_token_and_persists_result(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expired_access = _jwt({"exp": int(time.time()) - 60})
    refreshed_access = _jwt({"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-2"}})
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth", "access": expired_access, "refresh": "refresh-token", "expires": 1, "accountId": "acct-1"}}), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://auth.openai.com/oauth/token"
        assert "grant_type=refresh_token" in request.content.decode()
        return httpx.Response(200, json={"access_token": refreshed_access, "refresh_token": "new-refresh-token", "expires_in": 3600})

    store = PiCodexCredentialStore(auth_file=auth_file, client=httpx.Client(transport=httpx.MockTransport(handler)))
    credentials = store.get_credentials()

    assert credentials.access == refreshed_access
    assert credentials.refresh == "new-refresh-token"
    assert credentials.account_id == "acct-2"
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["openai-codex"]["access"] == refreshed_access
```

- [ ] **Step 2: Run tests and verify failure**

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_auth.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'codex_auth'`.

- [ ] **Step 3: Create `server/codex_auth.py`**

Use the `codex_auth.py` implementation from `.pi/plans/2026-05-04-openai-codex-oauth-provider.md`, Task 2, unchanged except keep imports local and do not print secrets.

- [ ] **Step 4: Run auth tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_auth.py -v
```

Expected: all tests pass.

## Task 3.3: Add OpenAI processor and factory

- [ ] **Step 1: Create `server/openai_codex_agent_processor.py`**

Use the processor implementation from `.pi/plans/2026-05-04-openai-codex-oauth-provider.md`, Task 3, with these changes:

- Import `AgentConfig` only from `config.py` if needed; avoid creating a separate `agent_config.py`.
- Constructor signature:
  ```python
  def __init__(self, mcp_server_url: str, model: str, **kwargs):
  ```
- Internally use `PiCodexCredentialStore()` with default Pi auth path/profile.
- Keep MCP URL passed in from `RuntimeConfig`.

- [ ] **Step 2: Create `server/tests/test_agent_processor_factory.py`**

```python
from config import AgentConfig
from agent_processor_factory import create_agent_processor
from claude_agent_processor import ClaudeAgentProcessor
from openai_codex_agent_processor import OpenAICodexAgentProcessor


def test_creates_claude_processor():
    processor = create_agent_processor(
        AgentConfig(provider="claude", model="claude-haiku-4-5-20251001"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, ClaudeAgentProcessor)


def test_creates_openai_codex_processor():
    processor = create_agent_processor(
        AgentConfig(provider="openai_codex_oauth", model="gpt-5.5"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, OpenAICodexAgentProcessor)
```

- [ ] **Step 3: Create `server/agent_processor_factory.py`**

```python
from __future__ import annotations

from config import AgentConfig
from claude_agent_processor import ClaudeAgentProcessor
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from pipecat.processors.frame_processor import FrameProcessor


def create_agent_processor(config: AgentConfig, *, mcp_server_url: str) -> FrameProcessor:
    if config.provider == "claude":
        return ClaudeAgentProcessor(mcp_server_url=mcp_server_url, model=config.model)
    if config.provider == "openai_codex_oauth":
        return OpenAICodexAgentProcessor(mcp_server_url=mcp_server_url, model=config.model)
    raise ValueError(f"Unsupported agent provider: {config.provider}")
```

- [ ] **Step 4: Run agent tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_auth.py tests/test_agent_processor_factory.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit Issue 3**

```bash
cd pipecat-agent
git add server/codex_auth.py server/openai_codex_agent_processor.py server/agent_processor_factory.py server/claude_agent_processor.py server/tests/test_codex_auth.py server/tests/test_agent_processor_factory.py
git commit -m "feat: add configurable agent providers"
```

---

# Issue 4: OpenWakeWord Mave Wake Gate

**Parallelization:** Can run after Issue 1. Independent from STT/TTS/agent provider construction.

**Files:**
- Create: `server/wake/__init__.py`
- Create: `server/wake/openwakeword_detector.py`
- Create: `server/wake/transcript_cleanup.py`
- Create: `server/wake/wake_gate.py`
- Create: `server/tests/test_transcript_cleanup.py`
- Create: `server/tests/test_wake_gate.py`
- Create: `server/models/mave.onnx`

## Task 4.1: Copy trained model

- [ ] **Step 1: Copy `mave.onnx`**

Run:

```bash
cd pipecat-agent
mkdir -p server/models
cp "C:/Users/Samuel/Documents/github/DF2025_CLEAN/models/mave.onnx" server/models/mave.onnx
```

Expected: `server/models/mave.onnx` exists and is about `859394` bytes.

- [ ] **Step 2: Verify copied model size**

```bash
cd pipecat-agent
python - <<'PY'
from pathlib import Path
p = Path('server/models/mave.onnx')
print(p.exists(), p.stat().st_size)
PY
```

Expected: prints `True 859394` or a very close byte count.

## Task 4.2: Implement transcript cleanup

- [ ] **Step 1: Create `server/tests/test_transcript_cleanup.py`**

```python
from wake.transcript_cleanup import strip_wake_phrase


def test_strips_leading_mave():
    assert strip_wake_phrase("Mave, move up a bit") == "move up a bit"


def test_strips_hey_mave():
    assert strip_wake_phrase("hey mave stop") == "stop"


def test_leaves_non_wake_text_unchanged():
    assert strip_wake_phrase("move up a bit") == "move up a bit"
```

- [ ] **Step 2: Create `server/wake/__init__.py`**

```python
"""Wake-word support for the Pipecat voice robot agent."""
```

- [ ] **Step 3: Create `server/wake/transcript_cleanup.py`**

```python
from __future__ import annotations

import re

_WAKE_PATTERN = re.compile(r"^\s*(?:hey\s+)?mave[\s,;:!?.-]*", re.IGNORECASE)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading Mave wake phrase from a transcript."""
    cleaned = _WAKE_PATTERN.sub("", text, count=1).strip()
    return cleaned or text.strip()
```

- [ ] **Step 4: Run cleanup tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_transcript_cleanup.py -v
```

Expected: all tests pass.

## Task 4.3: Implement OpenWakeWord detector wrapper

- [ ] **Step 1: Create `server/wake/openwakeword_detector.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from openwakeword.model import Model


class OpenWakeWordDetector:
    """Small wrapper around OpenWakeWord for one or more ONNX wake models."""

    def __init__(self, model_path: Path, *, threshold: float = 0.5):
        if not model_path.exists():
            raise FileNotFoundError(f"Wake model not found: {model_path}")
        self._threshold = threshold
        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def predict(self, pcm16: np.ndarray) -> dict[str, float]:
        if pcm16.dtype != np.int16:
            raise TypeError("OpenWakeWordDetector expects int16 PCM")
        return self._model.predict(pcm16)

    def detected(self, pcm16: np.ndarray) -> tuple[bool, str | None, float]:
        scores = self.predict(pcm16)
        if not scores:
            return False, None, 0.0
        name, score = max(scores.items(), key=lambda item: item[1])
        return score >= self._threshold, name, float(score)
```

- [ ] **Step 2: Run import check**

```bash
cd pipecat-agent/server
uv run python - <<'PY'
from pathlib import Path
from wake.openwakeword_detector import OpenWakeWordDetector
print(OpenWakeWordDetector)
PY
```

Expected: prints class path. Do not instantiate in this step; model loading can be slow.

## Task 4.4: Implement wake gate tests and gate

- [ ] **Step 1: Create `server/tests/test_wake_gate.py`**

```python
from unittest.mock import Mock

import numpy as np
import pytest

from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection
from wake.wake_gate import MaveWakeWordGate


class CapturingGate(MaveWakeWordGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def _frame(value: int, samples: int = 1600):
    audio = np.full(samples, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)


@pytest.mark.asyncio
async def test_blocks_audio_until_wake_detected():
    detector = Mock()
    detector.detected.return_value = (False, None, 0.0)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)

    assert gate.pushed == []


@pytest.mark.asyncio
async def test_replays_prebuffer_on_wake():
    detector = Mock()
    detector.detected.side_effect = [
        (False, None, 0.0),
        (True, "mave", 0.9),
    ]
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)
    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [item[0] for item in gate.pushed if isinstance(item[0], InputAudioRawFrame)]
    assert len(pushed_audio) == 2
    assert np.frombuffer(pushed_audio[0].audio, dtype=np.int16)[0] == 1
    assert np.frombuffer(pushed_audio[1].audio, dtype=np.int16)[0] == 2


@pytest.mark.asyncio
async def test_strips_wake_phrase_from_transcription_and_resets_to_sleep():
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.9)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)
    await gate.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    transcription = [item[0] for item in gate.pushed if isinstance(item[0], TranscriptionFrame)][0]
    assert transcription.text == "move up"
    assert gate.is_awake is False
```

- [ ] **Step 2: Create `server/wake/wake_gate.py`**

```python
from __future__ import annotations

from collections import deque

import numpy as np
from loguru import logger
from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from wake.openwakeword_detector import OpenWakeWordDetector
from wake.transcript_cleanup import strip_wake_phrase


class MaveWakeWordGate(FrameProcessor):
    """Blocks user audio until Mave is detected, then allows one command through."""

    def __init__(self, detector: OpenWakeWordDetector, *, pre_buffer_s: float = 1.5, **kwargs):
        super().__init__(**kwargs)
        self._detector = detector
        self._pre_buffer_s = pre_buffer_s
        self._ring: deque[InputAudioRawFrame] = deque()
        self._ring_samples = 0
        self._awake = False

    @property
    def is_awake(self) -> bool:
        return self._awake

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._process_audio_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame) and self._awake:
            cleaned = strip_wake_phrase(frame.text)
            await self.push_frame(
                TranscriptionFrame(
                    text=cleaned,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                    language=frame.language,
                    result=frame.result,
                    finalized=frame.finalized,
                ),
                direction,
            )
            if frame.finalized:
                self.reset()
            return

        await self.push_frame(frame, direction)

    async def _process_audio_frame(self, frame: InputAudioRawFrame, direction: FrameDirection) -> None:
        if self._awake:
            await self.push_frame(frame, direction)
            return

        self._append_ring(frame)
        pcm16 = self._to_mono_int16(frame)
        detected, name, score = self._detector.detected(pcm16)
        if not detected:
            return

        logger.info(f"Wake word detected: {name}={score:.3f}")
        self._awake = True
        buffered = list(self._ring)
        self._ring.clear()
        self._ring_samples = 0
        for buffered_frame in buffered:
            await self.push_frame(buffered_frame, direction)

    def _append_ring(self, frame: InputAudioRawFrame) -> None:
        self._ring.append(frame)
        self._ring_samples += len(frame.audio) // 2 // max(frame.num_channels, 1)
        max_samples = int(frame.sample_rate * self._pre_buffer_s)
        while self._ring and self._ring_samples > max_samples:
            old = self._ring.popleft()
            self._ring_samples -= len(old.audio) // 2 // max(old.num_channels, 1)

    def reset(self) -> None:
        self._awake = False
        self._ring.clear()
        self._ring_samples = 0

    @staticmethod
    def _to_mono_int16(frame: InputAudioRawFrame) -> np.ndarray:
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        if frame.num_channels <= 1:
            return pcm
        return pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
```

- [ ] **Step 3: Run wake tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_wake_gate.py tests/test_transcript_cleanup.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit Issue 4**

```bash
cd pipecat-agent
git add server/wake server/tests/test_wake_gate.py server/tests/test_transcript_cleanup.py server/models/mave.onnx
git commit -m "feat: add mave openwakeword gate"
```

---

# Issue 5: Emergency Stop Bypass Scaffold

**Parallelization:** Can run after Issue 1. Can run alongside Issue 4.

**Files:**
- Create/Modify: `server/wake/emergency_stop.py`
- Create: `server/tests/test_emergency_stop.py`

## Task 5.1: Implement config-only emergency stop scaffold

- [ ] **Step 1: Create `server/tests/test_emergency_stop.py`**

```python
from pathlib import Path

import pytest

from config import EmergencyStopConfig
from wake.emergency_stop import EmergencyStopDetector, build_emergency_stop_detector


def test_disabled_emergency_stop_returns_none():
    detector = build_emergency_stop_detector(EmergencyStopConfig(enabled=False))

    assert detector is None


def test_enabled_without_model_fails():
    with pytest.raises(ValueError, match="Emergency stop model is required"):
        build_emergency_stop_detector(EmergencyStopConfig(enabled=True, provider="openwakeword"))


def test_detector_interface_reports_no_detection_by_default(tmp_path: Path):
    model = tmp_path / "stop.onnx"
    model.write_bytes(b"fake")
    detector = EmergencyStopDetector(model_path=model, threshold=0.5)

    assert detector.command_text == "stop"
```

- [ ] **Step 2: Create `server/wake/emergency_stop.py`**

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import EmergencyStopConfig


@dataclass(frozen=True)
class EmergencyStopDetector:
    """Configuration holder for future local emergency stop detection."""

    model_path: Path
    threshold: float
    command_text: str = "stop"


def build_emergency_stop_detector(config: EmergencyStopConfig) -> EmergencyStopDetector | None:
    if not config.enabled:
        return None
    if config.model_path is None:
        raise ValueError("Emergency stop model is required when emergency stop is enabled")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Emergency stop model not found: {config.model_path}")
    return EmergencyStopDetector(model_path=config.model_path, threshold=config.threshold)
```

- [ ] **Step 3: Run tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_emergency_stop.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit Issue 5**

```bash
cd pipecat-agent
git add server/wake/emergency_stop.py server/tests/test_emergency_stop.py
git commit -m "feat: scaffold emergency stop detector"
```

---

# Issue 6: Metrics Recorder

**Parallelization:** Can run after Issue 1.

**Files:**
- Create: `server/metrics.py`
- Create: `server/tests/test_metrics.py`

## Task 6.1: Add metrics tests

- [ ] **Step 1: Create `server/tests/test_metrics.py`**

```python
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
```

- [ ] **Step 2: Create `server/metrics.py`**

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class TurnMetrics:
    turn_id: str
    started_at: float = field(default_factory=time.perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    transcript: str = ""
    response: str = ""

    def mark(self, name: str) -> None:
        self.marks[name] = time.perf_counter()

    def elapsed_ms(self, mark: str) -> float | None:
        value = self.marks.get(mark)
        if value is None:
            return None
        return round((value - self.started_at) * 1000, 2)


class VoiceMetricsRecorder:
    def __init__(self, *, profile: str, category: str, path: Path, include_text: bool):
        self._profile = profile
        self._category = category
        self._path = path
        self._include_text = include_text
        self._turns: dict[str, TurnMetrics] = {}
        self._disabled = False

    def start_turn(self, turn_id: str) -> TurnMetrics:
        turn = TurnMetrics(turn_id=turn_id)
        self._turns[turn_id] = turn
        return turn

    def get_turn(self, turn_id: str) -> TurnMetrics | None:
        return self._turns.get(turn_id)

    def finish_turn(self, turn_id: str) -> None:
        turn = self._turns.pop(turn_id, None)
        if turn is None:
            return
        record: dict[str, Any] = {
            "timestamp_unix": time.time(),
            "profile": self._profile,
            "category": self._category,
            "turn_id": turn.turn_id,
            "wake_latency_ms": turn.elapsed_ms("wake_detected"),
            "speech_captured_ms": turn.elapsed_ms("speech_captured"),
            "stt_done_ms": turn.elapsed_ms("stt_done"),
            "agent_done_ms": turn.elapsed_ms("agent_done"),
            "tts_first_audio_ms": turn.elapsed_ms("tts_first_audio"),
            "tts_done_ms": turn.elapsed_ms("tts_done"),
            "total_turn_ms": round((time.perf_counter() - turn.started_at) * 1000, 2),
        }
        if self._include_text:
            record["transcript"] = turn.transcript
            record["response"] = turn.response
        self._write(record)
        logger.info(
            "Voice metrics profile={} turn={} total={}ms transcript={!r}",
            self._profile,
            turn.turn_id,
            record["total_turn_ms"],
            turn.transcript[:120],
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._disabled = True
            logger.warning(f"Disabling voice metrics after write failure: {exc}")
```

- [ ] **Step 3: Run metrics tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit Issue 6**

```bash
cd pipecat-agent
git add server/metrics.py server/tests/test_metrics.py
git commit -m "feat: add voice metrics recorder"
```

---

# Issue 7: Pipeline Builder and `bot.py` Slimming

**Parallelization:** Depends on Issues 1–6.

**Files:**
- Create: `server/pipeline_builder.py`
- Modify: `server/bot.py`
- Optional Test: `server/tests/test_pipeline_builder.py`

## Task 7.1: Create pipeline builder

- [ ] **Step 1: Create `server/pipeline_builder.py`**

```python
from __future__ import annotations

from dataclasses import dataclass

from config import RuntimeConfig
from agent_processor_factory import create_agent_processor
from metrics import VoiceMetricsRecorder
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.transports.base_transport import BaseTransport
from providers import create_stt_service, create_tts_service
from wake.openwakeword_detector import OpenWakeWordDetector
from wake.wake_gate import MaveWakeWordGate


@dataclass
class BuiltPipeline:
    pipeline: Pipeline
    task: PipelineTask
    agent_processor: object
    user_aggregator: object
    assistant_aggregator: object
    metrics: VoiceMetricsRecorder | None


def build_pipeline(config: RuntimeConfig, transport: BaseTransport) -> BuiltPipeline:
    stt = create_stt_service(config.stt)
    tts = create_tts_service(config.tts)
    agent_processor = create_agent_processor(config.agent, mcp_server_url=config.mcp_robot_url)

    wake_gate = None
    if config.wake.provider == "openwakeword":
        assert config.wake.model_path is not None
        detector = OpenWakeWordDetector(config.wake.model_path, threshold=config.wake.threshold)
        wake_gate = MaveWakeWordGate(detector=detector, pre_buffer_s=config.wake.pre_buffer_s)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    processors = [transport.input()]
    if wake_gate is not None:
        processors.append(wake_gate)
    processors.extend([
        stt,
        user_aggregator,
        agent_processor,
        tts,
        transport.output(),
        assistant_aggregator,
    ])

    pipeline = Pipeline(processors)
    metrics = None
    if config.metrics.enabled:
        metrics = VoiceMetricsRecorder(
            profile=config.profile_name,
            category=config.category,
            path=config.metrics.path,
            include_text=config.metrics.include_text,
        )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=[],
    )
    return BuiltPipeline(
        pipeline=pipeline,
        task=task,
        agent_processor=agent_processor,
        user_aggregator=user_aggregator,
        assistant_aggregator=assistant_aggregator,
        metrics=metrics,
    )
```

- [ ] **Step 2: Run import check**

```bash
cd pipecat-agent/server
uv run python - <<'PY'
from pipeline_builder import build_pipeline, BuiltPipeline
print(build_pipeline, BuiltPipeline)
PY
```

Expected: import succeeds.

## Task 7.2: Slim `bot.py`

- [ ] **Step 1: Modify `server/bot.py` imports**

Add:

```python
import argparse
from config import load_runtime_config
from pipeline_builder import build_pipeline
```

Remove direct imports of:

```python
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair, LLMUserAggregatorParams
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from claude_agent_processor import ClaudeAgentProcessor
```

Keep transcript message classes if the event handlers still use them.

- [ ] **Step 2: Replace `run_bot` body**

Use this structure:

```python
async def run_bot(transport: BaseTransport, profile_name: str | None = None):
    """Main bot logic."""
    config = load_runtime_config(profile_name=profile_name)
    logger.info(
        "Starting voice robot agent profile={} category={} stt={} tts={} agent={}",
        config.profile_name,
        config.category,
        config.stt.provider,
        config.tts.provider,
        config.agent.provider,
    )

    built = build_pipeline(config, transport)
    task = built.task
    agent_processor = built.agent_processor

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await agent_processor.connect()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await agent_processor.disconnect()
        await task.cancel()

    @built.user_aggregator.event_handler("on_user_turn_stopped")
    async def on_user_turn_stopped(aggregator, strategy, message: UserTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}user: {message.content}")

    @built.assistant_aggregator.event_handler("on_assistant_turn_stopped")
    async def on_assistant_turn_stopped(aggregator, message: AssistantTurnStoppedMessage):
        timestamp = f"[{message.timestamp}] " if message.timestamp else ""
        logger.info(f"Transcript: {timestamp}assistant: {message.content}")

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)
```

- [ ] **Step 3: Add profile CLI parsing before `main()`**

At the bottom replace:

```python
if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
```

with:

```python
if __name__ == "__main__":
    from pipecat.runner.run import main

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--profile", dest="profile_name", default=None)
    known, remaining = parser.parse_known_args()
    if known.profile_name:
        os.environ["VOICE_PROFILE"] = known.profile_name
    main()
```

This keeps Pipecat runner compatibility by using `VOICE_PROFILE` internally.

- [ ] **Step 4: Update `bot()` to pass profile from env**

In `bot`, call:

```python
await run_bot(transport, profile_name=os.getenv("VOICE_PROFILE"))
```

- [ ] **Step 5: Run no-wake debug import/startup check**

```bash
cd pipecat-agent/server
VOICE_PROFILE=no_wake_debug uv run python - <<'PY'
from config import load_runtime_config
cfg = load_runtime_config(profile_name='no_wake_debug')
print(cfg.profile_name, cfg.wake.provider, cfg.stt.provider, cfg.tts.provider)
PY
```

Expected: prints `no_wake_debug none whisper kokoro`.

- [ ] **Step 6: Run focused tests**

```bash
cd pipecat-agent/server
uv run pytest tests/test_config.py tests/test_providers.py tests/test_agent_processor_factory.py tests/test_wake_gate.py tests/test_metrics.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit Issue 7**

```bash
cd pipecat-agent
git add server/pipeline_builder.py server/bot.py
git commit -m "feat: build pipeline from runtime profile"
```

---

# Issue 8: Docs and Benchmark Guide

**Parallelization:** Can start after Issue 1; final pass after Issue 7.

**Files:**
- Modify: `README.md`
- Modify: `server/.env.example`
- Create: `docs/benchmarking.md`

## Task 8.1: Update README

- [ ] **Step 1: Replace README configuration section with**

```markdown
## Runtime profiles

Default profile:

```text
hybrid_low_latency = Mave wake word + Deepgram Flux STT + OpenAI Codex OAuth agent + Cartesia Sonic TTS
```

Run the default profile:

```bash
cd server
uv run bot.py
```

Run a specific profile:

```bash
uv run bot.py --profile local_current
uv run bot.py --profile openai_all
uv run bot.py --profile deepgram_all
uv run bot.py --profile no_wake_debug
```

`--profile` overrides `VOICE_PROFILE`.

### Required keys

For the default profile, set:

```dotenv
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
```

For `openai_all`, set:

```dotenv
OPENAI_API_KEY=
```

For OpenAI Codex OAuth agent auth, run Pi and login with ChatGPT Plus/Pro Codex:

```text
pi
/login
```

The agent reads Pi's `~/.pi/agent/auth.json` `openai-codex` OAuth profile.

### Wake word

The trained Mave wake-word model lives at:

```text
server/models/mave.onnx
```

Normal commands require `mave`, for example:

```text
Mave, move up a bit.
```

Local debug profiles are available for offline testing, but benchmark profiles use streaming STT/TTS providers.
```

- [ ] **Step 2: Commit README update later with docs batch**

Do not commit yet if `docs/benchmarking.md` is not written.

## Task 8.2: Add benchmarking guide

- [ ] **Step 1: Create `docs/benchmarking.md`**

```markdown
# Voice Benchmarking

## Profiles

Benchmark profiles:

- `hybrid_low_latency`: Deepgram Flux STT + Cartesia Sonic TTS
- `openai_all`: OpenAI Realtime STT + OpenAI streaming TTS
- `deepgram_all`: Deepgram Flux STT + Deepgram Aura TTS

Debug profiles:

- `local_current`: local Whisper + Kokoro with Mave wake
- `no_wake_debug`: local Whisper + Kokoro without wake

## Running

```bash
cd server
uv run bot.py --profile hybrid_low_latency
```

## Metrics

Metrics are appended to:

```text
server/logs/voice_metrics.jsonl
```

Each JSONL record includes profile, category, transcript, response, and turn timing fields.

## Test utterances

Use the same utterances across profiles:

1. `Mave, what is the robot status?`
2. `Mave, what is the current position?`
3. `Mave, move up a bit.`
4. `Mave, stop.`

## Interpreting results

- Compare benchmark profiles to each other.
- Treat local profiles as debug/baseline, not equivalent streaming latency competitors.
- Do not compare runs if a profile silently failed or used fallback providers. Benchmark profiles fail startup instead of falling back.
```

- [ ] **Step 2: Commit Issue 8**

```bash
cd pipecat-agent
git add README.md server/.env.example docs/benchmarking.md
git commit -m "docs: document voice profiles and benchmarking"
```

---

# Final Verification

Run after all issues are merged.

- [ ] **Step 1: Run all tests**

```bash
cd pipecat-agent/server
uv run pytest -v
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

```bash
cd pipecat-agent/server
uv run ruff check .
```

Expected: no ruff errors.

- [ ] **Step 3: Run pyright**

```bash
cd pipecat-agent/server
uv run pyright .
```

Expected: no project-code type errors. If third-party SDK stubs are missing, add narrow ignores only at import lines and rerun.

- [ ] **Step 4: Verify default profile fails clearly without keys**

Temporarily unset keys:

```bash
cd pipecat-agent/server
DEEPGRAM_API_KEY= CARTESIA_API_KEY= uv run python - <<'PY'
from config import ConfigError, load_runtime_config
try:
    load_runtime_config(profile_name='hybrid_low_latency')
except ConfigError as e:
    print(e)
PY
```

Expected output mentions missing `DEEPGRAM_API_KEY` and/or `CARTESIA_API_KEY`.

- [ ] **Step 5: Verify local debug profile loads**

```bash
cd pipecat-agent/server
VOICE_PROFILE=no_wake_debug uv run python - <<'PY'
from config import load_runtime_config
cfg = load_runtime_config()
print(cfg.profile_name, cfg.category, cfg.wake.provider)
PY
```

Expected:

```text
no_wake_debug local_debug none
```

- [ ] **Step 6: Verify git status clean except ignored local files**

```bash
cd pipecat-agent
git status --short --ignored
```

Expected: no untracked non-ignored files. Ignored `.env`, `.venv`, caches may appear with `!!`.

## Self-Review Checklist

- Spec coverage:
  - Runtime profiles: Issue 1.
  - Low-latency default and fail-fast keys: Issue 1.
  - Local profiles retained: Issue 1 and Issue 2.
  - Provider factories: Issue 2.
  - OpenAI Codex OAuth migration: Issue 3.
  - Mave wake gate and model copy: Issue 4.
  - 1.5s pre-buffer and same-utterance support: Issue 4.
  - Emergency stop bypass scaffold: Issue 5.
  - Console + JSONL metrics with text: Issue 6.
  - Pipeline integration and thin `bot.py`: Issue 7.
  - Docs and benchmark instructions: Issue 8.
- Placeholder scan: no TODO/TBD placeholders remain as implementation instructions.
- Type consistency: all config dataclasses are defined in Issue 1 and reused by later issues.
- Parallel execution: Issues 2–6 depend only on Issue 1 contracts and can be assigned to separate subagents.
