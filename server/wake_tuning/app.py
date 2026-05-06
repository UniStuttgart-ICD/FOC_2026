from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from voice_runtime.profiles import (
    DEFAULT_PROFILE,
    ProfileError,
    default_profiles_path,
    load_runtime_profile,
)
from wake.openwakeword_detector import OpenWakeWordDetector
from wake_tuning.detector import WakeDecisionTracker
from wake_tuning.settings import (
    WakeTuningError,
    WakeTuningSettings,
    default_settings_path,
    load_profile_settings,
    save_profile_settings,
)

SERVER_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Mave Wake Word Tuning")


class SaveSettingsRequest(BaseModel):
    profile: str
    settings: dict[str, Any]


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/settings")
def get_settings(profile: str = Query(DEFAULT_PROFILE)) -> dict[str, Any]:
    runtime_profile = _load_profile(profile)
    saved = load_profile_settings(default_settings_path(SERVER_DIR), runtime_profile.profile_name)
    settings = saved or WakeTuningSettings.from_wake_profile(runtime_profile.wake)
    return {
        "profile": runtime_profile.profile_name,
        "provider": runtime_profile.wake.provider,
        "model_path": str(runtime_profile.wake.model_path) if runtime_profile.wake.model_path else None,
        "settings": asdict(settings),
        "saved": saved is not None,
        "settings_path": str(default_settings_path(SERVER_DIR)),
    }


@app.post("/api/settings")
def save_settings(request: SaveSettingsRequest) -> dict[str, Any]:
    runtime_profile = _load_profile(request.profile)
    if runtime_profile.wake.provider != "openwakeword":
        raise HTTPException(status_code=400, detail="Selected profile does not use openWakeWord")
    try:
        settings = WakeTuningSettings.from_mapping(request.settings)
        path = default_settings_path(SERVER_DIR)
        save_profile_settings(path, runtime_profile.profile_name, settings)
    except WakeTuningError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "profile": runtime_profile.profile_name,
        "settings": asdict(settings),
        "settings_path": str(path),
    }


@app.websocket("/ws/detect")
async def detect(
    websocket: WebSocket,
    profile: str = DEFAULT_PROFILE,
    settings: str | None = Query(default=None),
):
    await websocket.accept()
    runtime_profile = _load_profile(profile)
    if runtime_profile.wake.provider != "openwakeword" or runtime_profile.wake.model_path is None:
        await websocket.send_json({"type": "error", "message": "Selected profile has no wake model"})
        await websocket.close()
        return

    try:
        wake_settings = _resolve_detection_settings(runtime_profile, settings)
    except WakeTuningError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    try:
        detector = OpenWakeWordDetector(
            runtime_profile.wake.model_path,
            threshold=wake_settings.threshold,
            vad_threshold=wake_settings.vad_threshold,
        )
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        await websocket.close()
        return

    tracker = WakeDecisionTracker(wake_settings)
    await websocket.send_json(
        {
            "type": "ready",
            "settings": asdict(wake_settings),
            "vad_enabled": detector.vad_enabled,
            "vad_threshold": wake_settings.vad_threshold,
        }
    )
    try:
        while True:
            message = await websocket.receive_bytes()
            pcm16 = np.frombuffer(message, dtype=np.int16)
            scores = detector.predict(pcm16)
            vad_score = detector.last_vad_score()
            result = tracker.evaluate(scores, pcm16)
            await websocket.send_json(
                {
                    "type": "detection",
                    "detected": result.detected,
                    "model_name": result.model_name,
                    "score": result.score,
                    "rms": result.rms,
                    "peak": result.peak,
                    "hits": result.hits,
                    "required_hits": result.required_hits,
                    "decision": result.decision,
                    "threshold_hit": result.threshold_hit,
                    "level_hit": result.level_hit,
                    "vad_enabled": detector.vad_enabled,
                    "vad_threshold": wake_settings.vad_threshold,
                    "vad_score": vad_score,
                }
            )
    except WebSocketDisconnect:
        return


def _resolve_detection_settings(runtime_profile, raw_settings: str | None) -> WakeTuningSettings:
    if raw_settings:
        try:
            parsed = json.loads(raw_settings)
        except json.JSONDecodeError as exc:
            raise WakeTuningError("Detector settings must be valid JSON") from exc
        if not isinstance(parsed, dict):
            raise WakeTuningError("Detector settings must be a JSON object")
        return WakeTuningSettings.from_mapping(parsed)
    saved = load_profile_settings(default_settings_path(SERVER_DIR), runtime_profile.profile_name)
    return saved or WakeTuningSettings.from_wake_profile(runtime_profile.wake)


def _load_profile(profile_name: str):
    try:
        return load_runtime_profile(
            profiles_path=default_profiles_path(SERVER_DIR),
            server_dir=SERVER_DIR,
            profile_name=profile_name,
        )
    except (ProfileError, WakeTuningError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=9010, type=int)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("wake_tuning.app:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
