from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from voice_modulation.dsp import VoiceModulationDspError
from voice_modulation.preview import (
    AudioBytes,
    VoicePreviewError,
    decode_preview,
    encode_preview,
    render_effect_preview,
    synthesize_tts_reference,
)
from voice_modulation.settings import (
    BUILT_IN_PRESETS,
    VoiceModulationError,
    default_settings_path,
    load_all_settings,
    load_profile_settings,
    save_profile_settings,
    settings_from_mapping,
)
from voice_runtime.profiles import (
    ProfileError,
    TTSProfile,
    default_profiles_path,
    load_runtime_profile,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

PreviewSynthesizer = Callable[[TTSProfile, str], AudioBytes]


def create_app(
    server_dir: Path | None = None,
    preview_synthesizer: PreviewSynthesizer | None = None,
) -> FastAPI:
    app = FastAPI(title="Voice Modulation Lab")
    root = server_dir or Path(__file__).resolve().parents[1]
    synthesize = preview_synthesizer or synthesize_tts_reference

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _static_index_path().read_text(encoding="utf-8")

    @app.get("/api/presets")
    def presets() -> dict[str, object]:
        return {
            "presets": [
                {"name": name, "settings": settings.to_dict()}
                for name, settings in BUILT_IN_PRESETS.items()
            ]
        }

    @app.get("/api/profiles")
    def profiles() -> dict[str, object]:
        return {"profiles": [_profile_summary(root, name) for name in _profile_names(root)]}

    @app.get("/api/settings/{profile_name}")
    def get_settings(profile_name: str) -> dict[str, object]:
        all_settings = load_all_settings(server_dir=root)
        settings = all_settings.get(profile_name, BUILT_IN_PRESETS["clean"])
        return {
            "profile": profile_name,
            "saved": profile_name in all_settings,
            "settings": settings.to_dict(),
            "settings_path": str(default_settings_path(root)),
        }

    @app.post("/api/settings/{profile_name}")
    def post_settings(profile_name: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            settings = settings_from_mapping(payload)
            path = save_profile_settings(profile_name, settings, server_dir=root)
        except VoiceModulationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "profile": profile_name,
            "settings": settings.to_dict(),
            "settings_path": str(path),
        }

    @app.post("/api/preview/effect")
    def preview_effect(payload: dict[str, object]) -> dict[str, object]:
        try:
            audio = decode_preview(_dict(payload.get("audio"), "audio"))
            settings = settings_from_mapping(_dict(payload.get("settings"), "settings"))
            rendered = render_effect_preview(audio, settings)
        except (VoicePreviewError, VoiceModulationError, VoiceModulationDspError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"audio": asdict(encode_preview(rendered))}

    @app.post("/api/preview/tts")
    def preview_tts(payload: dict[str, object]) -> dict[str, object]:
        profile_name = _string(payload.get("profile_name"), "profile_name")
        text = _string(payload.get("text"), "text")
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
            clean_audio = synthesize(profile.tts, text)
            settings = load_profile_settings(profile_name, server_dir=root)
            modulated_audio = render_effect_preview(clean_audio, settings)
        except ProfileError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (VoicePreviewError, VoiceModulationError, VoiceModulationDspError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "profile": profile_name,
            "audio": asdict(encode_preview(clean_audio)),
            "modulated": asdict(encode_preview(modulated_audio)),
            "settings": settings.to_dict(),
        }

    return app


def _static_index_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "index.html"


def _profile_names(server_dir: Path) -> list[str]:
    path = default_profiles_path(server_dir)
    if not path.exists():
        return []
    with path.open("rb") as f:
        data = tomllib.load(f)
    profiles = data.get("profiles", {})
    if not isinstance(profiles, dict):
        return []
    return sorted(str(name) for name in profiles)


def _profile_summary(server_dir: Path, name: str) -> dict[str, object]:
    profile = load_runtime_profile(server_dir=server_dir, profile_name=name)
    return {
        "name": profile.name,
        "category": profile.category,
        "tts": {
            "provider": profile.tts.provider,
            "model": profile.tts.model,
            "voice": profile.tts.voice,
        },
        "missing_env": [
            env_name for env_name in profile.required_env_names() if not os.getenv(env_name)
        ],
    }


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise VoicePreviewError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail=f"{name} must be a non-empty string")
    return value.strip()


app = create_app()
