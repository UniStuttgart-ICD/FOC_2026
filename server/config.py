from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from voice_modulation.settings import VoiceModulationError, apply_saved_voice_modulation
from voice_runtime.profiles import (
    DEFAULT_PROFILE,
    AgentProfile,
    AgentProvider,
    Category,
    EmbodimentProfile,
    EmergencyStopProfile,
    MetricsProfile,
    ProcessTraceProfile,
    ProfileError,
    RobotExecutionProfile,
    RuntimeProfile,
    STTProfile,
    STTProvider,
    TTSProfile,
    TTSProvider,
    WakeProfile,
    WakeProvider,
    default_profiles_path,
    load_runtime_profile,
)
from wake_tuning.settings import WakeTuningError, apply_saved_wake_tuning

WakeConfig = WakeProfile
EmergencyStopConfig = EmergencyStopProfile
STTConfig = STTProfile
TTSConfig = TTSProfile
AgentConfig = AgentProfile
MetricsConfig = MetricsProfile
ProcessTraceConfig = ProcessTraceProfile
RobotExecutionConfig = RobotExecutionProfile
EmbodimentConfig = EmbodimentProfile
VoiceModulationConfig = object


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
    process_trace: ProcessTraceConfig
    robot_execution: RobotExecutionConfig
    server_dir: Path
    embodiment: EmbodimentConfig = field(default_factory=EmbodimentConfig)
    voice_modulation: object | None = None

    @classmethod
    def from_profile(cls, profile: RuntimeProfile) -> RuntimeConfig:
        return cls(
            profile_name=profile.profile_name,
            category=profile.category,
            wake=profile.wake,
            emergency_stop=profile.emergency_stop,
            stt=profile.stt,
            tts=profile.tts,
            agent=profile.agent,
            mcp_robot_url=profile.mcp_robot_url,
            metrics=profile.metrics,
            process_trace=profile.process_trace,
            robot_execution=profile.robot_execution,
            embodiment=profile.embodiment,
            server_dir=profile.server_dir,
            voice_modulation=profile.voice_modulation,
        )

    def required_env_names(self) -> tuple[str, ...]:
        return RuntimeProfile(
            name=self.profile_name,
            category=self.category,
            wake=self.wake,
            emergency_stop=self.emergency_stop,
            stt=self.stt,
            tts=self.tts,
            agent=self.agent,
            mcp_robot_url=self.mcp_robot_url,
            metrics=self.metrics,
            process_trace=self.process_trace,
            robot_execution=self.robot_execution,
            embodiment=self.embodiment,
            server_dir=self.server_dir,
            voice_modulation=self.voice_modulation,
        ).required_env_names()


class ConfigError(ValueError):
    """Raised when runtime configuration is invalid."""


def load_runtime_config(
    *,
    profiles_path: str | Path | None = None,
    server_dir: str | Path | None = None,
    profile_name: str | None = None,
) -> RuntimeConfig:
    selected_profile = profile_name or os.getenv("VOICE_PROFILE") or DEFAULT_PROFILE
    try:
        profile = load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=server_dir,
            profile_name=selected_profile,
        )
        profile = apply_saved_wake_tuning(profile)
        profile = apply_saved_voice_modulation(profile)
    except ProfileError as exc:
        message = str(exc).replace("Unknown profile", "Unknown VOICE_PROFILE", 1)
        raise ConfigError(message) from exc
    except WakeTuningError as exc:
        raise ConfigError(str(exc)) from exc
    except VoiceModulationError as exc:
        raise ConfigError(str(exc)) from exc

    missing = [name for name in profile.required_env_names() if not os.getenv(name)]
    if missing:
        raise ConfigError(
            f"Profile {profile.profile_name} requires missing environment variable(s): "
            + ", ".join(missing)
        )
    return RuntimeConfig.from_profile(profile)


__all__ = [
    "AgentConfig",
    "AgentProvider",
    "Category",
    "ConfigError",
    "DEFAULT_PROFILE",
    "EmergencyStopConfig",
    "EmbodimentConfig",
    "MetricsConfig",
    "ProcessTraceConfig",
    "RobotExecutionConfig",
    "RuntimeConfig",
    "STTConfig",
    "STTProvider",
    "TTSConfig",
    "TTSProvider",
    "VoiceModulationConfig",
    "WakeConfig",
    "WakeProvider",
    "default_profiles_path",
    "load_runtime_config",
]
