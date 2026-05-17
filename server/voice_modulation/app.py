from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from agent_control.prompts import SPEAKING_AGENT_PERSONA, SPEECH_DELIVERY_STYLE
from voice_modulation.dsp import VoiceModulationDspError
from voice_modulation.gemini_voices import gemini_live_voice_options
from voice_modulation.persona_editor import (
    PersonaValidationError,
    list_persona_templates,
    load_persona_parts,
    load_persona_template,
    save_persona_part,
)
from voice_modulation.preview import (
    AudioBytes,
    VoicePreviewError,
    decode_preview,
    encode_preview,
    render_effect_preview,
    synthesize_tts_reference,
    tts_for_preview,
)
from voice_modulation.profile_editor import save_gemini_tts_voice, save_voice_modulation_default
from voice_modulation.settings import (
    BUILT_IN_PRESETS,
    VoiceModulationError,
    default_settings_path,
    load_all_settings,
    load_profile_settings,
    profile_default_settings,
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
CartesiaVoiceFetcher = Callable[[], dict[str, object]]

CARTESIA_VOICE_LIBRARY_URL = "https://play.cartesia.ai/voices"
CARTESIA_VOICES_API_URL = "https://api.cartesia.ai/voices"
DEFAULT_CARTESIA_VERSION = "2026-03-01"


def create_app(
    server_dir: Path | None = None,
    preview_synthesizer: PreviewSynthesizer | None = None,
    cartesia_voice_fetcher: CartesiaVoiceFetcher | None = None,
) -> FastAPI:
    app = FastAPI(title="Voice Modulation Lab")
    root = server_dir or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=True)
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

    @app.get("/api/cartesia/voices")
    def cartesia_voices() -> dict[str, object]:
        return _cartesia_voice_response(fetcher=cartesia_voice_fetcher)

    @app.get("/api/gemini/voices")
    def gemini_voices() -> dict[str, object]:
        return {"voices": gemini_live_voice_options()}

    @app.get("/api/persona")
    def persona() -> dict[str, object]:
        return {
            "speaking_persona": SPEAKING_AGENT_PERSONA,
            "speech_delivery": SPEECH_DELIVERY_STYLE,
            "sources": {
                "speaking_persona": "reasoning_agent_persona.md",
                "speech_delivery": "speech_delivery_style.md",
            },
        }

    @app.get("/api/persona/parts")
    def persona_parts() -> dict[str, object]:
        return {
            "parts": [
                asdict(part) for part in load_persona_parts(_prompt_parts_dir(root))
            ]
        }

    @app.post("/api/persona/parts/{part_id}")
    def post_persona_part(part_id: str, payload: dict[str, object]) -> dict[str, object]:
        content = _string(payload.get("content"), "content")
        try:
            part = save_persona_part(_prompt_parts_dir(root), part_id, content)
        except PersonaValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "part": asdict(part),
            "restart_required": True,
            "git_source_changed": True,
        }

    @app.get("/api/persona/templates")
    def persona_templates() -> dict[str, object]:
        return {"templates": list_persona_templates(root)}

    @app.post("/api/persona/templates/{template_id}/load")
    def post_persona_template(template_id: str) -> dict[str, object]:
        try:
            parts = load_persona_template(root, template_id)
        except PersonaValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "parts": [asdict(part) for part in parts],
            "restart_required": True,
            "git_source_changed": True,
        }

    @app.get("/api/settings/{profile_name}")
    def get_settings(profile_name: str) -> dict[str, object]:
        all_settings = load_all_settings(server_dir=root)
        default_settings = BUILT_IN_PRESETS["clean"]
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
            default_settings = profile_default_settings(profile)
        except ProfileError:
            pass
        settings = all_settings.get(profile_name, default_settings)
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

    @app.post("/api/profiles/{profile_name}/tts/voice")
    def post_tts_voice(profile_name: str, payload: dict[str, object]) -> dict[str, object]:
        voice = _string(payload.get("voice"), "voice")
        try:
            result = save_gemini_tts_voice(_profiles_path(root), profile_name, voice)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "profile": result.profile_name,
            "voice": result.voice,
            "restart_required": True,
            "source_path": str(result.source_path),
        }

    @app.post("/api/profiles/{profile_name}/voice-modulation-default")
    def post_voice_modulation_default(
        profile_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        try:
            settings = settings_from_mapping(payload)
            result = save_voice_modulation_default(_profiles_path(root), profile_name, settings)
        except (VoiceModulationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "profile": result.profile_name,
            "settings": settings.to_dict(),
            "restart_required": True,
            "source_path": str(result.source_path),
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

    @app.post("/api/preview/source")
    def preview_source(payload: dict[str, object]) -> dict[str, object]:
        profile_name = _string(payload.get("profile_name"), "profile_name")
        text = _string(payload.get("text"), "text")
        voice_id = _optional_string(payload.get("voice_id"), "voice_id")
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
            clean_audio = synthesize(tts_for_preview(profile.tts, voice_id), text)
        except ProfileError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except VoicePreviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "profile": profile_name,
            "audio": asdict(encode_preview(clean_audio)),
        }

    @app.post("/api/preview/tts")
    def preview_tts(payload: dict[str, object]) -> dict[str, object]:
        profile_name = _string(payload.get("profile_name"), "profile_name")
        text = _string(payload.get("text"), "text")
        voice_id = _optional_string(payload.get("voice_id"), "voice_id")
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
            clean_audio = synthesize(tts_for_preview(profile.tts, voice_id), text)
            settings = load_profile_settings(
                profile_name,
                server_dir=root,
                default=profile_default_settings(profile),
            )
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


def _profiles_path(root: Path) -> Path:
    return default_profiles_path(root)


def _prompt_parts_dir(root: Path) -> Path:
    return root / "agent_control" / "prompt_parts"


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


def _cartesia_voice_response(fetcher: CartesiaVoiceFetcher | None = None) -> dict[str, object]:
    if fetcher is None and not os.getenv("CARTESIA_API_KEY"):
        return _cartesia_voice_payload(
            available=False,
            voices=[],
            has_more=False,
            reason="Missing CARTESIA_API_KEY",
        )
    try:
        library = fetcher() if fetcher is not None else fetch_cartesia_voices()
    except VoicePreviewError as exc:
        return _cartesia_voice_payload(
            available=False,
            voices=[],
            has_more=False,
            reason=str(exc),
        )
    return _cartesia_voice_payload(
        available=True,
        voices=_normalize_cartesia_voices(library.get("voices")),
        has_more=bool(library.get("has_more", False)),
        reason=None,
    )


def _cartesia_voice_payload(
    *,
    available: bool,
    voices: list[dict[str, object]],
    has_more: bool,
    reason: str | None,
) -> dict[str, object]:
    return {
        "available": available,
        "voices": voices,
        "has_more": has_more,
        "reason": reason,
        "voice_library_url": CARTESIA_VOICE_LIBRARY_URL,
    }


def fetch_cartesia_voices() -> dict[str, object]:
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        raise VoicePreviewError("Missing CARTESIA_API_KEY")

    query = urlencode({"limit": 100, "language": "en"})
    request = Request(
        f"{CARTESIA_VOICES_API_URL}?{query}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Cartesia-Version": os.getenv("CARTESIA_VERSION", DEFAULT_CARTESIA_VERSION),
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise VoicePreviewError(f"Cartesia voice list failed: HTTP {exc.code}") from exc
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise VoicePreviewError(f"Cartesia voice list failed: {exc}") from exc

    if not isinstance(data, dict):
        raise VoicePreviewError("Cartesia voice list response must be an object")
    voices = data.get("data", [])
    if not isinstance(voices, list):
        raise VoicePreviewError("Cartesia voice list response must contain a data list")
    return {"voices": voices, "has_more": bool(data.get("has_more", False))}


def _normalize_cartesia_voices(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    voices: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        voice_id = _clean_optional_string(item.get("id"))
        if voice_id is None:
            continue
        voices.append(
            {
                "id": voice_id,
                "name": _clean_optional_string(item.get("name")) or voice_id,
                "language": _clean_optional_string(item.get("language")),
                "description": _clean_optional_string(item.get("description")),
            }
        )
    return voices


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise VoicePreviewError(f"{name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail=f"{name} must be a non-empty string")
    return value.strip()


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail=f"{name} must be a string")
    return value.strip() or None


def _clean_optional_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


app = create_app()
