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
    body_shift: float = 0.0
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
    breath_mix: float = 0.0
    limiter: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def has_audible_effect(self) -> bool:
        if not self.enabled or self.wet_mix == 0.0:
            return False
        return (
            self.gain_db != 0.0
            or self.low_cut_hz > 0.0
            or self.high_cut_hz > 0.0
            or self.drive > 0.0
            or self.bit_depth < 16
            or self.pitch_shift_semitones != 0.0
            or self.body_shift != 0.0
            or self.ring_mod_hz > 0.0
            or (self.tremolo_hz > 0.0 and self.tremolo_depth > 0.0)
            or (self.chorus_mix > 0.0 and self.chorus_depth_ms > 0.0)
            or (self.echo_mix > 0.0 and self.echo_delay_ms > 0.0)
            or self.noise_mix > 0.0
            or self.breath_mix > 0.0
        )

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
        _range(self.body_shift, "body_shift", -1.0, 1.0)
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
        _range(self.breath_mix, "breath_mix", 0.0, 0.3)


BUILT_IN_PRESETS: dict[str, VoiceModulationSettings] = {
    "clean": VoiceModulationSettings(enabled=False, preset_name="clean"),
    "protocol_droid": VoiceModulationSettings(
        enabled=True,
        preset_name="protocol_droid",
        wet_mix=1.0,
        low_cut_hz=180.0,
        high_cut_hz=5600.0,
        drive=0.22,
        pitch_shift_semitones=1.4,
        body_shift=0.62,
        ring_mod_hz=38.0,
        chorus_rate_hz=0.45,
        chorus_depth_ms=7.0,
        chorus_mix=0.12,
        limiter=True,
    ),
    "masked_breather": VoiceModulationSettings(
        enabled=True,
        preset_name="masked_breather",
        wet_mix=1.0,
        low_cut_hz=60.0,
        high_cut_hz=2600.0,
        drive=0.26,
        pitch_shift_semitones=-4.0,
        body_shift=-0.9,
        ring_mod_hz=16.0,
        echo_delay_ms=95.0,
        echo_feedback=0.12,
        echo_mix=0.12,
        noise_mix=0.012,
        breath_mix=0.065,
        limiter=True,
    ),
    "helmet_comms": VoiceModulationSettings(
        enabled=True,
        preset_name="helmet_comms",
        wet_mix=1.0,
        low_cut_hz=360.0,
        high_cut_hz=3300.0,
        drive=0.14,
        bit_depth=12,
        noise_mix=0.035,
        limiter=True,
    ),
    "damaged_droid": VoiceModulationSettings(
        enabled=True,
        preset_name="damaged_droid",
        wet_mix=0.95,
        low_cut_hz=80.0,
        high_cut_hz=2600.0,
        drive=0.18,
        bit_depth=7,
        pitch_shift_semitones=-1.5,
        body_shift=0.2,
        ring_mod_hz=22.0,
        tremolo_hz=6.0,
        tremolo_depth=0.35,
        echo_delay_ms=85.0,
        echo_feedback=0.18,
        echo_mix=0.22,
        limiter=True,
    ),
    "ai_core": VoiceModulationSettings(
        enabled=True,
        preset_name="ai_core",
        wet_mix=1.0,
        high_cut_hz=8200.0,
        body_shift=0.12,
        chorus_rate_hz=0.35,
        chorus_depth_ms=18.0,
        chorus_mix=0.26,
        echo_delay_ms=110.0,
        echo_feedback=0.18,
        echo_mix=0.16,
        limiter=True,
    ),
    "titan_mech": VoiceModulationSettings(
        enabled=True,
        preset_name="titan_mech",
        wet_mix=1.0,
        low_cut_hz=45.0,
        high_cut_hz=3000.0,
        drive=0.34,
        pitch_shift_semitones=-6.0,
        body_shift=-1.0,
        ring_mod_hz=12.0,
        chorus_rate_hz=0.28,
        chorus_depth_ms=12.0,
        chorus_mix=0.14,
        echo_delay_ms=140.0,
        echo_feedback=0.2,
        echo_mix=0.18,
        breath_mix=0.025,
        limiter=True,
    ),
    "hologram": VoiceModulationSettings(
        enabled=True,
        preset_name="hologram",
        wet_mix=1.0,
        high_cut_hz=7600.0,
        pitch_shift_semitones=0.4,
        body_shift=0.05,
        chorus_rate_hz=0.9,
        chorus_depth_ms=22.0,
        chorus_mix=0.32,
        echo_delay_ms=155.0,
        echo_feedback=0.34,
        echo_mix=0.24,
        noise_mix=0.006,
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
            body_shift=_float(data, "body_shift", 0.0),
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
            breath_mix=_float(data, "breath_mix", 0.0),
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
    default: VoiceModulationSettings | None = None,
) -> VoiceModulationSettings:
    return load_all_settings(settings_path, server_dir=server_dir).get(
        profile_name,
        default or BUILT_IN_PRESETS["clean"],
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
    default = profile_default_settings(profile)
    settings = load_profile_settings(
        profile.profile_name,
        server_dir=profile.server_dir,
        settings_path=settings_path,
        default=default,
    )
    return replace(profile, voice_modulation=settings)


def profile_default_settings(profile: RuntimeProfile) -> VoiceModulationSettings:
    raw = profile.voice_modulation
    if isinstance(raw, VoiceModulationSettings):
        return raw
    if isinstance(raw, dict):
        return settings_from_mapping(raw)
    return BUILT_IN_PRESETS["clean"]


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
