from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

DEFAULT_PROFILE = "hybrid_low_latency"

WakeProvider = Literal["none", "openwakeword"]
STTProvider = Literal["deepgram_flux", "openai_realtime", "whisper"]
TTSProvider = Literal["cartesia", "openai", "deepgram", "kokoro"]
AgentProvider = Literal["openai_codex_oauth"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Category = Literal["benchmark_streaming", "local_debug"]

_WAKE_PROVIDERS = {"none", "openwakeword"}
_STT_PROVIDERS = {"deepgram_flux", "openai_realtime", "whisper"}
_TTS_PROVIDERS = {"cartesia", "openai", "deepgram", "kokoro"}
_AGENT_PROVIDERS = {"openai_codex_oauth"}
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
_CATEGORIES = {"benchmark_streaming", "local_debug"}
_STREAMING_STT_PROVIDERS = {"deepgram_flux", "openai_realtime"}
_STREAMING_TTS_PROVIDERS = {"cartesia", "openai", "deepgram"}


class ProfileError(ValueError):
    """Raised when a runtime profile is invalid."""


@dataclass(frozen=True)
class WakeProfile:
    provider: WakeProvider
    model_path: Path | None
    threshold: float = 0.5
    candidate_log_threshold: float = 0.3
    pre_buffer_s: float = 1.5
    single_command: bool = True


@dataclass(frozen=True)
class EmergencyStopProfile:
    enabled: bool
    provider: WakeProvider = "none"
    model_path: Path | None = None
    threshold: float = 0.5


@dataclass(frozen=True)
class STTProfile:
    provider: STTProvider
    model: str | None = None
    device: str | None = None


@dataclass(frozen=True)
class TTSProfile:
    provider: TTSProvider
    model: str | None = None
    voice: str | None = None


@dataclass(frozen=True)
class AgentProfile:
    provider: AgentProvider
    model: str
    reasoning_effort: ReasoningEffort | None = None


@dataclass(frozen=True)
class MetricsProfile:
    enabled: bool
    path: Path
    include_text: bool


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    category: Category
    wake: WakeProfile
    emergency_stop: EmergencyStopProfile
    stt: STTProfile
    tts: TTSProfile
    agent: AgentProfile
    mcp_robot_url: str
    metrics: MetricsProfile
    server_dir: Path

    def required_env_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.stt.provider == "deepgram_flux" or self.tts.provider == "deepgram":
            names.append("DEEPGRAM_API_KEY")
        if self.tts.provider == "cartesia":
            names.append("CARTESIA_API_KEY")
        if self.tts.provider == "cartesia" and self.tts.voice is None:
            names.append("CARTESIA_VOICE_ID")
        if self.stt.provider == "openai_realtime" or self.tts.provider == "openai":
            names.append("OPENAI_API_KEY")
        return tuple(names)

    @property
    def profile_name(self) -> str:
        return self.name


def default_profiles_path(server_dir: Path | None = None) -> Path:
    root = server_dir or Path(__file__).resolve().parent.parent
    return root / "runtime_profiles.toml"


def load_runtime_profile(
    *,
    profiles_path: str | Path | None = None,
    server_dir: str | Path | None = None,
    profile_name: str | None = None,
) -> RuntimeProfile:
    server_root = Path(server_dir) if server_dir is not None else Path(__file__).resolve().parent.parent
    selected_profile = profile_name or DEFAULT_PROFILE
    path = Path(profiles_path) if profiles_path is not None else default_profiles_path(server_root)
    if not path.exists():
        raise ProfileError(f"Runtime profiles file not found: {path}")

    with path.open("rb") as f:
        data = tomllib.load(f)

    profiles = _table(data, "profiles")
    raw_profile = profiles.get(selected_profile)
    if not isinstance(raw_profile, dict):
        raise ProfileError(f"Unknown profile '{selected_profile}' in {path}")

    category = cast(Category, _literal(raw_profile, "category", _CATEGORIES))
    wake = _parse_wake(_table(raw_profile, "wake"), server_root)
    emergency_stop = _parse_emergency_stop(_table(raw_profile, "emergency_stop"), server_root)
    stt = _parse_stt(_table(raw_profile, "stt"))
    tts = _parse_tts(_table(raw_profile, "tts"))
    agent = _parse_agent(_table(raw_profile, "agent"))
    mcp = _table(raw_profile, "mcp")
    robot = _table(mcp, "robot")
    metrics = _parse_metrics(_table(raw_profile, "metrics"), server_root)

    profile = RuntimeProfile(
        name=selected_profile,
        category=category,
        wake=wake,
        emergency_stop=emergency_stop,
        stt=stt,
        tts=tts,
        agent=agent,
        mcp_robot_url=_string(robot, "url"),
        metrics=metrics,
        server_dir=server_root,
    )
    _validate_runtime_profile(profile)
    return profile


def _parse_wake(table: dict[str, Any], server_dir: Path) -> WakeProfile:
    provider = cast(WakeProvider, _literal(table, "provider", _WAKE_PROVIDERS))
    return WakeProfile(
        provider=provider,
        model_path=_optional_path(table, "model_path", server_dir),
        threshold=_float(table, "threshold", 0.5),
        candidate_log_threshold=_float(table, "candidate_log_threshold", 0.3),
        pre_buffer_s=_float(table, "pre_buffer_s", 1.5),
        single_command=_bool(table, "single_command", True),
    )


def _parse_emergency_stop(table: dict[str, Any], server_dir: Path) -> EmergencyStopProfile:
    provider = cast(WakeProvider, _literal(table, "provider", _WAKE_PROVIDERS, default="none"))
    return EmergencyStopProfile(
        enabled=_bool(table, "enabled", False),
        provider=provider,
        model_path=_optional_path(table, "model_path", server_dir),
        threshold=_float(table, "threshold", 0.5),
    )


def _parse_stt(table: dict[str, Any]) -> STTProfile:
    provider = cast(STTProvider, _literal(table, "provider", _STT_PROVIDERS))
    return STTProfile(
        provider=provider,
        model=_optional_string(table, "model"),
        device=_optional_string(table, "device"),
    )


def _parse_tts(table: dict[str, Any]) -> TTSProfile:
    provider = cast(TTSProvider, _literal(table, "provider", _TTS_PROVIDERS))
    return TTSProfile(
        provider=provider,
        model=_optional_string(table, "model"),
        voice=_optional_string(table, "voice"),
    )


def _parse_agent(table: dict[str, Any]) -> AgentProfile:
    provider = cast(AgentProvider, _literal(table, "provider", _AGENT_PROVIDERS))
    reasoning_effort = cast(
        ReasoningEffort | None,
        _optional_literal(table, "reasoning_effort", _REASONING_EFFORTS),
    )
    return AgentProfile(
        provider=provider,
        model=_string(table, "model", "gpt-5.4-mini"),
        reasoning_effort=reasoning_effort,
    )


def _parse_metrics(table: dict[str, Any], server_dir: Path) -> MetricsProfile:
    return MetricsProfile(
        enabled=_bool(table, "enabled", True),
        path=_path(table, "path", server_dir, "logs/voice_metrics.jsonl"),
        include_text=_bool(table, "include_text", True),
    )


def _validate_runtime_profile(profile: RuntimeProfile) -> None:
    if profile.wake.provider == "openwakeword" and profile.wake.model_path is None:
        raise ProfileError("wake.model_path is required when wake.provider = 'openwakeword'")
    if profile.emergency_stop.enabled and profile.emergency_stop.provider == "none":
        raise ProfileError("emergency_stop.provider must not be 'none' when emergency stop is enabled")
    if profile.emergency_stop.enabled and profile.emergency_stop.model_path is None:
        raise ProfileError("emergency_stop.model_path is required when emergency stop is enabled")
    if profile.category == "benchmark_streaming":
        if profile.stt.provider not in _STREAMING_STT_PROVIDERS:
            raise ProfileError("benchmark_streaming profiles require streaming STT")
        if profile.tts.provider not in _STREAMING_TTS_PROVIDERS:
            raise ProfileError("benchmark_streaming profiles require streaming TTS")


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ProfileError(f"[{key}] must be a TOML table")
    return value


def _string(table: dict[str, Any], key: str, default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(table: dict[str, Any], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"{key} must be a non-empty string")
    return value.strip()


def _literal(table: dict[str, Any], key: str, allowed: set[str], default: str | None = None) -> str:
    value = table.get(key, default)
    if not isinstance(value, str) or value not in allowed:
        raise ProfileError(f"{key} must be one of {sorted(allowed)}")
    return value


def _optional_literal(table: dict[str, Any], key: str, allowed: set[str]) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or value not in allowed:
        raise ProfileError(f"{key} must be one of {sorted(allowed)}")
    return value


def _bool(table: dict[str, Any], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if not isinstance(value, bool):
        raise ProfileError(f"{key} must be true or false")
    return value


def _float(table: dict[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProfileError(f"{key} must be a number")
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
