from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from voice_modulation.gemini_voices import is_gemini_live_voice
from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.profiles import (
    EmbodimentMotionProfile,
    EmbodimentProfile,
    EmbodimentTouchTriggerProfile,
)


@dataclass(frozen=True)
class ProfileWriteResult:
    profile_name: str
    source_path: Path
    voice: str | None = None
    settings: VoiceModulationSettings | None = None
    embodiment: EmbodimentProfile | None = None


def save_gemini_tts_voice(
    profiles_path: str | Path,
    profile_name: str,
    voice: str,
) -> ProfileWriteResult:
    if not is_gemini_live_voice(voice):
        raise ValueError(f"Unsupported Gemini Live voice: {voice}")
    path = Path(profiles_path)
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    profile = _profile_table(document, profile_name)
    tts = _table(profile, "tts")
    if str(tts.get("provider", "")).strip() != "gemini_live":
        raise ValueError("Gemini voice selection requires a gemini_live TTS profile")
    tts["voice"] = voice
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    return ProfileWriteResult(profile_name=profile_name, source_path=path, voice=voice)


def save_voice_modulation_default(
    profiles_path: str | Path,
    profile_name: str,
    settings: VoiceModulationSettings,
) -> ProfileWriteResult:
    settings.validate()
    path = Path(profiles_path)
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    profile = _profile_table(document, profile_name)
    table = tomlkit.table()
    for key, value in settings.to_dict().items():
        table[key] = value
    profile["voice_modulation"] = table
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    return ProfileWriteResult(profile_name=profile_name, source_path=path, settings=settings)


def save_embodiment_default(
    profiles_path: str | Path,
    profile_name: str,
    settings: EmbodimentProfile,
) -> ProfileWriteResult:
    path = Path(profiles_path)
    document = tomlkit.parse(path.read_text(encoding="utf-8"))
    profile = _profile_table(document, profile_name)
    table = tomlkit.table()
    table["enabled"] = settings.enabled
    table["rosbridge_host"] = settings.rosbridge_host
    table["rosbridge_port"] = settings.rosbridge_port
    table["animation_topic"] = settings.animation_topic
    table["animation_topic_type"] = settings.animation_topic_type
    table["start_blink_on_connect"] = settings.start_blink_on_connect
    table["stop_blink_on_disconnect"] = settings.stop_blink_on_disconnect
    table["wave_duration_s"] = settings.wave_duration_s
    table["move_duration_s"] = settings.move_duration_s

    touch = tomlkit.table()
    touch["enabled"] = settings.touch_trigger.enabled
    if settings.touch_trigger.topic is not None:
        touch["topic"] = settings.touch_trigger.topic
    touch["topic_type"] = settings.touch_trigger.topic_type
    if settings.touch_trigger.link_name is not None:
        touch["link_name"] = settings.touch_trigger.link_name
    touch["motion"] = settings.touch_trigger.motion
    touch["cooldown_s"] = settings.touch_trigger.cooldown_s
    table["touch_trigger"] = touch
    if settings.motions:
        motions = tomlkit.table()
        for name, motion in sorted(settings.motions.items()):
            motion_table = tomlkit.table()
            motion_table["start_signal"] = motion.start_signal
            motion_table["stop_signal"] = motion.stop_signal
            motions[name] = motion_table
        table["motions"] = motions

    profile["embodiment"] = table
    path.write_text(tomlkit.dumps(document), encoding="utf-8")
    return ProfileWriteResult(profile_name=profile_name, source_path=path, embodiment=settings)


def embodiment_from_mapping(data: dict[str, Any]) -> EmbodimentProfile:
    touch_data = data.get("touch_trigger")
    if touch_data is None:
        touch_data = {}
    if not isinstance(touch_data, dict):
        raise ValueError("touch_trigger must be an object")
    motions = _motions_from_mapping(data.get("motions", {}))
    motion = _motion_name(_string(touch_data, "motion", "move"))
    if motion not in {"blink", "move", "wave"} and motion not in motions:
        raise ValueError("touch_trigger.motion must be a built-in or registered motion")
    touch_topic = _optional_topic(touch_data, "topic")
    if _bool(touch_data, "enabled", False) and touch_topic is None:
        raise ValueError("touch_trigger.topic is required when touch trigger is enabled")
    return EmbodimentProfile(
        enabled=_bool(data, "enabled", False),
        rosbridge_host=_string(data, "rosbridge_host", "127.0.0.1"),
        rosbridge_port=_port(data, "rosbridge_port", 9090),
        animation_topic=_topic(data, "animation_topic", "/HOLO1_AnimSignal"),
        animation_topic_type=_string(data, "animation_topic_type", "std_msgs/String"),
        start_blink_on_connect=_bool(data, "start_blink_on_connect", True),
        stop_blink_on_disconnect=_bool(data, "stop_blink_on_disconnect", True),
        wave_duration_s=_non_negative_float(data, "wave_duration_s", 0.8),
        move_duration_s=_non_negative_float(data, "move_duration_s", 1.0),
        touch_trigger=EmbodimentTouchTriggerProfile(
            enabled=_bool(touch_data, "enabled", False),
            topic=touch_topic,
            topic_type=_string(touch_data, "topic_type", "std_msgs/String"),
            link_name=_optional_string(touch_data, "link_name"),
            motion=motion,
            cooldown_s=_non_negative_float(touch_data, "cooldown_s", 1.0),
        ),
        motions=motions,
    )


def _profile_table(document: Any, profile_name: str) -> Any:
    profiles = _table(document, "profiles")
    if profile_name not in profiles:
        raise ValueError(f"Unknown runtime profile: {profile_name}")
    profile = profiles[profile_name]
    if not hasattr(profile, "items"):
        raise ValueError(f"Runtime profile must be a table: {profile_name}")
    return profile


def _table(parent: Any, key: str) -> Any:
    if key not in parent or not hasattr(parent[key], "items"):
        raise ValueError(f"{key} must be a TOML table")
    return parent[key]


def _string(data: dict[str, Any], key: str, default: str) -> str:
    value = data.get(key, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value.strip()


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value.strip() or None


def _topic(data: dict[str, Any], key: str, default: str) -> str:
    value = _string(data, key, default)
    if not value.startswith("/"):
        raise ValueError(f"{key} must start with /")
    return value


def _optional_topic(data: dict[str, Any], key: str) -> str | None:
    value = _optional_string(data, key)
    if value is None:
        return None
    if not value.startswith("/"):
        raise ValueError(f"{key} must start with /")
    return value


def _bool(data: dict[str, Any], key: str, default: bool) -> bool:
    value = data.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be true or false")
    return value


def _non_negative_float(data: dict[str, Any], key: str, default: float) -> float:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    value = float(value)
    if value < 0:
        raise ValueError(f"{key} must be non-negative")
    return value


def _port(data: dict[str, Any], key: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    if value < 1 or value > 65535:
        raise ValueError(f"{key} must be between 1 and 65535")
    return value


def _motions_from_mapping(value: Any) -> dict[str, EmbodimentMotionProfile]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("motions must be an object")
    motions: dict[str, EmbodimentMotionProfile] = {}
    for raw_name, raw_motion in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("motion names must be non-empty strings")
        if not isinstance(raw_motion, dict):
            raise ValueError(f"motion {raw_name} must be an object")
        name = _motion_name(raw_name)
        motions[name] = EmbodimentMotionProfile(
            start_signal=_string(raw_motion, "start_signal", f"start_{name}"),
            stop_signal=_string(raw_motion, "stop_signal", f"stop_{name}"),
        )
    return motions


def _motion_name(value: str) -> str:
    name = value.strip().casefold()
    if not name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("motion names may contain letters, numbers, _ and -")
    return name
