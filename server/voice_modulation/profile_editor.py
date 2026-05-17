from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomlkit

from voice_modulation.gemini_voices import is_gemini_live_voice
from voice_modulation.settings import VoiceModulationSettings


@dataclass(frozen=True)
class ProfileWriteResult:
    profile_name: str
    source_path: Path
    voice: str | None = None
    settings: VoiceModulationSettings | None = None


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
