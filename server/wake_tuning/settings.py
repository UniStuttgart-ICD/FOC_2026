from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from voice_runtime.profiles import RuntimeProfile, WakeProfile

SETTINGS_ENV = "WAKE_TUNING_SETTINGS_PATH"


class WakeTuningError(ValueError):
    """Raised when wake tuning settings are invalid."""


@dataclass(frozen=True)
class WakeTuningSettings:
    threshold: float
    vad_threshold: float
    candidate_log_threshold: float
    required_hits: int
    min_wake_rms: float
    min_wake_peak: int
    rearm_delay_s: float
    pre_buffer_s: float

    @classmethod
    def from_wake_profile(cls, wake: WakeProfile) -> WakeTuningSettings:
        return cls(
            threshold=wake.threshold,
            vad_threshold=wake.vad_threshold,
            candidate_log_threshold=wake.candidate_log_threshold,
            required_hits=wake.required_hits,
            min_wake_rms=wake.min_wake_rms,
            min_wake_peak=wake.min_wake_peak,
            rearm_delay_s=wake.rearm_delay_s,
            pre_buffer_s=wake.pre_buffer_s,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> WakeTuningSettings:
        try:
            settings = cls(
                threshold=_float(data, "threshold"),
                vad_threshold=_float(data, "vad_threshold"),
                candidate_log_threshold=_float(data, "candidate_log_threshold"),
                required_hits=_int(data, "required_hits"),
                min_wake_rms=_float(data, "min_wake_rms"),
                min_wake_peak=_int(data, "min_wake_peak"),
                rearm_delay_s=_float(data, "rearm_delay_s"),
                pre_buffer_s=_float(data, "pre_buffer_s"),
            )
        except KeyError as exc:
            raise WakeTuningError(f"Missing wake tuning field: {exc.args[0]}") from exc
        settings.validate()
        return settings

    def validate(self) -> None:
        _range(self.threshold, "threshold", 0.01, 0.99)
        _range(self.vad_threshold, "vad_threshold", 0.0, 1.0)
        _range(self.candidate_log_threshold, "candidate_log_threshold", 0.0, 0.99)
        if self.required_hits < 1 or self.required_hits > 5:
            raise WakeTuningError("required_hits must be between 1 and 5")
        if self.min_wake_rms < 0:
            raise WakeTuningError("min_wake_rms must be non-negative")
        if self.min_wake_peak < 0:
            raise WakeTuningError("min_wake_peak must be non-negative")
        _range(self.rearm_delay_s, "rearm_delay_s", 0.0, 15.0)
        _range(self.pre_buffer_s, "pre_buffer_s", 0.0, 3.0)

    def apply_to(self, wake: WakeProfile) -> WakeProfile:
        return replace(wake, **asdict(self))


def default_settings_path(server_dir: Path | None = None) -> Path:
    configured = os.getenv(SETTINGS_ENV)
    if configured:
        return Path(configured)
    root = server_dir or Path(__file__).resolve().parents[1]
    return root / "state" / "wake_tuning_settings.json"


def load_all_settings(path: Path) -> dict[str, WakeTuningSettings]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WakeTuningError(f"Wake tuning settings file is not valid JSON: {path}") from exc
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict):
        raise WakeTuningError("Wake tuning settings must contain a profiles object")
    settings: dict[str, WakeTuningSettings] = {}
    for profile_name, profile_settings in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(profile_settings, dict):
            raise WakeTuningError("Wake tuning profiles must be named objects")
        settings[profile_name] = WakeTuningSettings.from_mapping(profile_settings)
    return settings


def load_profile_settings(path: Path, profile_name: str) -> WakeTuningSettings | None:
    return load_all_settings(path).get(profile_name)


def save_profile_settings(path: Path, profile_name: str, settings: WakeTuningSettings) -> None:
    settings.validate()
    all_settings = load_all_settings(path) if path.exists() else {}
    all_settings[profile_name] = settings
    payload = {
        "profiles": {
            name: asdict(profile_settings)
            for name, profile_settings in sorted(all_settings.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_saved_wake_tuning(profile: RuntimeProfile, *, settings_path: Path | None = None) -> RuntimeProfile:
    path = settings_path or default_settings_path(profile.server_dir)
    settings = load_profile_settings(path, profile.profile_name)
    if settings is None or profile.wake.provider != "openwakeword":
        return profile
    return replace(profile, wake=settings.apply_to(profile.wake))


def _float(data: dict[str, Any], key: str) -> float:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WakeTuningError(f"{key} must be a number")
    return float(value)


def _int(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise WakeTuningError(f"{key} must be an integer")
    return value


def _range(value: float, key: str, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise WakeTuningError(f"{key} must be between {minimum} and {maximum}")

