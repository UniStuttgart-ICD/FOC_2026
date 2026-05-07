from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from voice_runtime.profiles import RuntimeProfile

SETTINGS_ENV = "VOICE_MODULATION_SETTINGS_PATH"


class VoiceModulationError(ValueError):
    """Raised when voice modulation settings are invalid."""


@dataclass(frozen=True)
class VoiceModulationSettings:
    enabled: bool = False
    preset_name: str = "clean"
    gain_db: float = 0.0
    wet_mix: float = 1.0
    low_cut_hz: float = 0.0
    high_cut_hz: float = 0.0
    drive: float = 0.0
    bit_depth: int = 16
    pitch_shift_semitones: float = 0.0
    ring_mod_hz: float = 0.0
    tremolo_hz: float = 0.0
    tremolo_depth: float = 0.0
    chorus_rate_hz: float = 0.0
    chorus_depth_ms: float = 0.0
    chorus_mix: float = 0.0
    echo_delay_ms: float = 0.0
    echo_feedback: float = 0.0
    echo_mix: float = 0.0
    noise_mix: float = 0.0
    limiter: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if not self.preset_name.strip():
            raise VoiceModulationError("preset_name must be a non-empty string")
        _range(self.gain_db, "gain_db", -24.0, 24.0)
        _range(self.wet_mix, "wet_mix", 0.0, 1.0)
        _range(self.low_cut_hz, "low_cut_hz", 0.0, 4000.0)
        _range(self.high_cut_hz, "high_cut_hz", 0.0, 24000.0)
        _range(self.drive, "drive", 0.0, 1.0)
        if self.bit_depth < 4 or self.bit_depth > 16:
            raise VoiceModulationError("bit_depth must be between 4 and 16")
        _range(self.pitch_shift_semitones, "pitch_shift_semitones", -12.0, 12.0)
        _range(self.ring_mod_hz, "ring_mod_hz", 0.0, 2000.0)
        _range(self.tremolo_hz, "tremolo_hz", 0.0, 20.0)
        _range(self.tremolo_depth, "tremolo_depth", 0.0, 1.0)
        _range(self.chorus_rate_hz, "chorus_rate_hz", 0.0, 8.0)
        _range(self.chorus_depth_ms, "chorus_depth_ms", 0.0, 35.0)
        _range(self.chorus_mix, "chorus_mix", 0.0, 1.0)
        _range(self.echo_delay_ms, "echo_delay_ms", 0.0, 600.0)
        _range(self.echo_feedback, "echo_feedback", 0.0, 0.95)
        _range(self.echo_mix, "echo_mix", 0.0, 1.0)
        _range(self.noise_mix, "noise_mix", 0.0, 0.2)


BUILT_IN_PRESETS: dict[str, VoiceModulationSettings] = {
    "clean": VoiceModulationSettings(enabled=False, preset_name="clean"),
    "robot": VoiceModulationSettings(
        enabled=True,
        preset_name="robot",
        gain_db=2.0,
        wet_mix=0.9,
        low_cut_hz=120.0,
        high_cut_hz=5200.0,
        drive=0.25,
        bit_depth=9,
        pitch_shift_semitones=0.0,
        ring_mod_hz=38.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        limiter=True,
    ),
    "radio": VoiceModulationSettings(
        enabled=True,
        preset_name="radio",
        gain_db=4.0,
        wet_mix=1.0,
        low_cut_hz=320.0,
        high_cut_hz=3200.0,
        drive=0.18,
        bit_depth=12,
        pitch_shift_semitones=0.0,
        ring_mod_hz=0.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        noise_mix=0.02,
        limiter=True,
    ),
    "small_speaker": VoiceModulationSettings(
        enabled=True,
        preset_name="small_speaker",
        gain_db=1.5,
        wet_mix=0.85,
        low_cut_hz=220.0,
        high_cut_hz=4200.0,
        drive=0.1,
        bit_depth=13,
        pitch_shift_semitones=0.0,
        ring_mod_hz=0.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        limiter=True,
    ),
    "low_battery": VoiceModulationSettings(
        enabled=True,
        preset_name="low_battery",
        gain_db=-1.0,
        wet_mix=0.95,
        low_cut_hz=80.0,
        high_cut_hz=2600.0,
        drive=0.32,
        bit_depth=7,
        pitch_shift_semitones=-1.5,
        ring_mod_hz=22.0,
        tremolo_hz=6.0,
        tremolo_depth=0.35,
        echo_delay_ms=85.0,
        echo_feedback=0.18,
        echo_mix=0.22,
        limiter=True,
    ),
    "giant": VoiceModulationSettings(
        enabled=True,
        preset_name="giant",
        gain_db=1.0,
        wet_mix=0.92,
        low_cut_hz=60.0,
        high_cut_hz=6200.0,
        drive=0.12,
        bit_depth=16,
        pitch_shift_semitones=-5.0,
        chorus_rate_hz=0.35,
        chorus_depth_ms=8.0,
        chorus_mix=0.16,
        limiter=True,
    ),
    "wide_chorus": VoiceModulationSettings(
        enabled=True,
        preset_name="wide_chorus",
        gain_db=0.5,
        wet_mix=0.85,
        low_cut_hz=90.0,
        high_cut_hz=9000.0,
        drive=0.04,
        bit_depth=16,
        pitch_shift_semitones=0.5,
        chorus_rate_hz=0.7,
        chorus_depth_ms=18.0,
        chorus_mix=0.36,
        limiter=True,
    ),
    "echo_room": VoiceModulationSettings(
        enabled=True,
        preset_name="echo_room",
        gain_db=0.0,
        wet_mix=0.9,
        low_cut_hz=120.0,
        high_cut_hz=7600.0,
        drive=0.06,
        bit_depth=16,
        echo_delay_ms=140.0,
        echo_feedback=0.38,
        echo_mix=0.34,
        limiter=True,
    ),
    "ghost": VoiceModulationSettings(
        enabled=True,
        preset_name="ghost",
        gain_db=-1.0,
        wet_mix=0.88,
        low_cut_hz=180.0,
        high_cut_hz=8200.0,
        drive=0.08,
        bit_depth=14,
        pitch_shift_semitones=4.0,
        tremolo_hz=2.4,
        tremolo_depth=0.18,
        chorus_rate_hz=0.45,
        chorus_depth_ms=22.0,
        chorus_mix=0.28,
        echo_delay_ms=190.0,
        echo_feedback=0.26,
        echo_mix=0.22,
        noise_mix=0.025,
        limiter=True,
    ),
}


def settings_from_mapping(data: dict[str, Any]) -> VoiceModulationSettings:
    try:
        settings = VoiceModulationSettings(
            enabled=_bool(data, "enabled", False),
            preset_name=_string(data, "preset_name", "custom"),
            gain_db=_float(data, "gain_db", 0.0),
            wet_mix=_float(data, "wet_mix", 1.0),
            low_cut_hz=_float(data, "low_cut_hz", 0.0),
            high_cut_hz=_float(data, "high_cut_hz", 0.0),
            drive=_float(data, "drive", 0.0),
            bit_depth=_int(data, "bit_depth", 16),
            pitch_shift_semitones=_float(data, "pitch_shift_semitones", 0.0),
            ring_mod_hz=_float(data, "ring_mod_hz", 0.0),
            tremolo_hz=_float(data, "tremolo_hz", 0.0),
            tremolo_depth=_float(data, "tremolo_depth", 0.0),
            chorus_rate_hz=_float(data, "chorus_rate_hz", 0.0),
            chorus_depth_ms=_float(data, "chorus_depth_ms", 0.0),
            chorus_mix=_float(data, "chorus_mix", 0.0),
            echo_delay_ms=_float(data, "echo_delay_ms", 0.0),
            echo_feedback=_float(data, "echo_feedback", 0.0),
            echo_mix=_float(data, "echo_mix", 0.0),
            noise_mix=_float(data, "noise_mix", 0.0),
            limiter=_bool(data, "limiter", True),
        )
    except KeyError as exc:
        raise VoiceModulationError(f"Missing voice modulation field: {exc.args[0]}") from exc
    settings.validate()
    return settings


def default_settings_path(server_dir: Path | None = None) -> Path:
    configured = os.getenv(SETTINGS_ENV)
    if configured:
        return Path(configured)
    root = server_dir or Path(__file__).resolve().parents[1]
    return root / "state" / "voice_modulation_settings.json"


def load_all_settings(
    path: str | Path | None = None,
    *,
    server_dir: Path | None = None,
) -> dict[str, VoiceModulationSettings]:
    resolved = Path(path) if path is not None else default_settings_path(server_dir)
    if not resolved.exists():
        return {}
    try:
        raw = json.loads(resolved.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VoiceModulationError(
            f"Voice modulation settings file is not valid JSON: {resolved}"
        ) from exc
    profiles = raw.get("profiles") if isinstance(raw, dict) else None
    if not isinstance(profiles, dict):
        raise VoiceModulationError("Voice modulation settings must contain a profiles object")
    settings: dict[str, VoiceModulationSettings] = {}
    for profile_name, profile_settings in profiles.items():
        if not isinstance(profile_name, str) or not isinstance(profile_settings, dict):
            raise VoiceModulationError("Voice modulation profiles must be named objects")
        settings[profile_name] = settings_from_mapping(profile_settings)
    return settings


def load_profile_settings(
    profile_name: str,
    *,
    server_dir: Path | None = None,
    settings_path: str | Path | None = None,
) -> VoiceModulationSettings:
    return load_all_settings(settings_path, server_dir=server_dir).get(
        profile_name,
        BUILT_IN_PRESETS["clean"],
    )


def save_profile_settings(
    profile_name: str,
    settings: VoiceModulationSettings,
    *,
    server_dir: Path | None = None,
    settings_path: str | Path | None = None,
) -> Path:
    settings.validate()
    path = Path(settings_path) if settings_path is not None else default_settings_path(server_dir)
    all_settings = load_all_settings(path) if path.exists() else {}
    all_settings[profile_name] = settings
    payload = {
        "profiles": {
            name: profile_settings.to_dict()
            for name, profile_settings in sorted(all_settings.items())
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def apply_saved_voice_modulation(
    profile: RuntimeProfile,
    settings_path: str | Path | None = None,
) -> RuntimeProfile:
    settings = load_profile_settings(
        profile.profile_name,
        server_dir=profile.server_dir,
        settings_path=settings_path,
    )
    return replace(profile, voice_modulation=settings)


def _string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise VoiceModulationError(f"{key} must be a non-empty string")
    return value.strip()


def _float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VoiceModulationError(f"{key} must be a number")
    return float(value)


def _int(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VoiceModulationError(f"{key} must be an integer")
    return value


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise VoiceModulationError(f"{key} must be true or false")
    return value


def _range(value: float, key: str, minimum: float, maximum: float) -> None:
    if value < minimum or value > maximum:
        raise VoiceModulationError(f"{key} must be between {minimum} and {maximum}")
