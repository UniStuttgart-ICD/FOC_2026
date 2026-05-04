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
