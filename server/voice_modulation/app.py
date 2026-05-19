from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from embodiment.animations import (
    EmbodimentAnimationController,
    create_embodiment_animation_controller,
)
from voice_modulation.dsp import VoiceModulationDspError
from voice_modulation.gemini_voices import gemini_live_voice_options
from voice_modulation.persona_editor import (
    PersonaValidationError,
    list_persona_templates,
    load_persona_parts,
    load_persona_template,
    save_persona_part,
    save_persona_template_part,
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
from voice_modulation.profile_editor import (
    embodiment_from_mapping,
    save_embodiment_default,
    save_gemini_tts_voice,
    save_voice_modulation_default,
)
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
    EmbodimentProfile,
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
EmbodimentControllerFactory = Callable[
    [EmbodimentProfile], EmbodimentAnimationController | None
]

CARTESIA_VOICE_LIBRARY_URL = "https://play.cartesia.ai/voices"
CARTESIA_VOICES_API_URL = "https://api.cartesia.ai/voices"
DEFAULT_CARTESIA_VERSION = "2026-03-01"
PIPECAT_CLIENT_URL = "http://localhost:7860/client/"
OPERATOR_DASHBOARD_URL = "http://127.0.0.1:8787"
RUN_AGENT_STATUS_TIMEOUT_S = 1.0
HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def create_app(
    server_dir: Path | None = None,
    preview_synthesizer: PreviewSynthesizer | None = None,
    cartesia_voice_fetcher: CartesiaVoiceFetcher | None = None,
    embodiment_controller_factory: EmbodimentControllerFactory | None = None,
) -> FastAPI:
    app = FastAPI(title="Agent Persona Lab")
    root = server_dir or Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env", override=True)
    synthesize = preview_synthesizer or synthesize_tts_reference
    create_embodiment_controller = (
        embodiment_controller_factory or create_embodiment_animation_controller
    )

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

    @app.get("/api/run-agent/status")
    def run_agent_status() -> dict[str, object]:
        return {
            "pipecat_client_url": PIPECAT_CLIENT_URL,
            "dashboard_url": OPERATOR_DASHBOARD_URL,
            "pipecat_client_ready": _url_is_reachable(
                PIPECAT_CLIENT_URL,
                timeout_s=RUN_AGENT_STATUS_TIMEOUT_S,
            ),
        }

    @app.get("/api/cartesia/voices")
    def cartesia_voices() -> dict[str, object]:
        return _cartesia_voice_response(fetcher=cartesia_voice_fetcher)

    @app.get("/api/gemini/voices")
    def gemini_voices() -> dict[str, object]:
        return {"voices": gemini_live_voice_options()}

    @app.get("/api/persona")
    def persona() -> dict[str, object]:
        return {
            "speaking_persona": _prompt_part(root, "reasoning_agent_persona.md"),
            "speech_delivery": _prompt_part(root, "speech_delivery_style.md"),
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
        content = _content_string(payload.get("content"), "content")
        try:
            part = save_persona_part(_prompt_parts_dir(root), part_id, content)
        except PersonaValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "part": asdict(part),
            "restart_required": True,
            "git_source_changed": True,
            "template_source_changed": False,
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
            "template_id": template_id,
            "parts": [asdict(part) for part in parts],
            "restart_required": True,
            "git_source_changed": True,
        }

    @app.post("/api/persona/templates/{template_id}/parts/{part_id}")
    def post_persona_template_part(
        template_id: str,
        part_id: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        content = _content_string(payload.get("content"), "content")
        try:
            part = save_persona_template_part(root, template_id, part_id, content)
            save_persona_part(_prompt_parts_dir(root), part_id, content)
        except PersonaValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "template_id": template_id,
            "part": asdict(part),
            "restart_required": True,
            "git_source_changed": True,
            "template_source_changed": True,
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

    @app.get("/api/embodiment/{profile_name}")
    def get_embodiment(profile_name: str) -> dict[str, object]:
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
        except ProfileError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {
            "profile": profile_name,
            "settings": _embodiment_payload(profile.embodiment),
        }

    @app.post("/api/profiles/{profile_name}/embodiment")
    def post_embodiment(profile_name: str, payload: dict[str, object]) -> dict[str, object]:
        try:
            settings = embodiment_from_mapping(payload)
            result = save_embodiment_default(_profiles_path(root), profile_name, settings)
        except (ValueError, ProfileError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "ok": True,
            "profile": result.profile_name,
            "settings": _embodiment_payload(settings),
            "restart_required": True,
            "source_path": str(result.source_path),
        }

    @app.post("/api/profiles/{profile_name}/embodiment/test")
    async def post_embodiment_test(
        profile_name: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        try:
            profile = load_runtime_profile(server_dir=root, profile_name=profile_name)
        except ProfileError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        settings = profile.embodiment
        raw_settings = payload.get("settings")
        if raw_settings is not None:
            try:
                settings = embodiment_from_mapping(_dict(raw_settings, "settings"))
            except (ValueError, VoicePreviewError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        controller = create_embodiment_controller(settings)
        if controller is None:
            raise HTTPException(status_code=400, detail="Embodiment animations are disabled")
        action = _optional_string(payload.get("action"), "action") or "start"
        motion = _optional_string(payload.get("motion"), "motion") or "move"
        side = _optional_string(payload.get("side"), "side")
        result: dict[str, object] | None = None
        try:
            if action == "start":
                result = await controller.start_animation(motion, side=side)
            elif action == "stop":
                result = await controller.stop_animation(motion, side=side)
            else:
                raise ValueError("action must be start or stop")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if result is not None and result.get("ok"):
                await asyncio.sleep(0.2)
            await controller.stop()
        if result is None:
            raise HTTPException(status_code=400, detail="Embodiment animation test did not run")
        if not result.get("ok"):
            raise HTTPException(status_code=503, detail=str(result.get("error") or "Test failed"))
        return {"ok": True, "result": result}

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
            clean_audio = synthesize(
                tts_for_preview(
                    profile.tts,
                    voice_id,
                    speech_delivery_style=_speech_delivery_style(root, profile.tts.provider),
                ),
                text,
            )
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
            clean_audio = synthesize(
                tts_for_preview(
                    profile.tts,
                    voice_id,
                    speech_delivery_style=_speech_delivery_style(root, profile.tts.provider),
                ),
                text,
            )
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


def _speech_delivery_style(root: Path, tts_provider: str) -> str:
    if tts_provider != "gemini_live":
        return ""
    content = _prompt_part(root, "speech_delivery_style.md")
    if not content:
        raise VoicePreviewError("Speech delivery prompt must not be empty")
    return content


def _prompt_part(root: Path, filename: str) -> str:
    path = _prompt_parts_dir(root) / filename
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise VoicePreviewError(f"Prompt part is not readable: {path}") from exc
    return HTML_COMMENT_PATTERN.sub("", content).strip()


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
        "embodiment": _embodiment_payload(profile.embodiment),
    }


def _url_is_reachable(url: str, *, timeout_s: float) -> bool:
    request = Request(url, headers={"Accept": "text/html"})
    try:
        with urlopen(request, timeout=timeout_s):
            return True
    except (HTTPError, OSError, URLError):
        return False


def _embodiment_payload(settings: EmbodimentProfile) -> dict[str, object]:
    return asdict(settings)


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


def _content_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HTTPException(status_code=400, detail=f"{name} must be a non-empty string")
    return value


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
