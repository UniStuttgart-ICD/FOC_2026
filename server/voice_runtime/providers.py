from __future__ import annotations

import os
from typing import Any

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.whisper.stt import WhisperSTTService

from voice_runtime.gemini_live_speech import (
    DEFAULT_GEMINI_LIVE_MODEL,
    DEFAULT_GEMINI_LIVE_VOICE,
    GeminiLiveSpeechRendererService,
)
from voice_runtime.profiles import STTProfile, TTSProfile

DEFAULT_CARTESIA_VOICE_ID = "47c38ca4-5f35-497b-b1a3-415245fb35e1"


def create_stt_service(config: STTProfile) -> FrameProcessor:
    if config.provider == "whisper":
        return WhisperSTTService(
            device=config.device or "cuda",
            settings=WhisperSTTService.Settings(
                model=config.model
                or os.getenv("WHISPER_MODEL")
                or os.getenv("OPENAI_MODEL")
                or "base",
            ),
        )
    if config.provider == "deepgram_flux":
        return DeepgramFluxSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramFluxSTTService.Settings(model=config.model or "flux-general-en"),
        )
    if config.provider == "openai_realtime":
        return OpenAIRealtimeSTTService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIRealtimeSTTService.Settings(
                model=config.model or "gpt-realtime-whisper",
                noise_reduction="near_field",
            ),
        )
    raise ValueError(f"Unsupported STT provider: {config.provider}")


def create_tts_service(config: TTSProfile) -> FrameProcessor:
    if config.provider == "kokoro":
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=config.voice or os.getenv("KOKORO_VOICE_ID") or "af_heart"
            ),
        )
    if config.provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaTTSService.Settings(
                model=config.model or "sonic-3",
                voice=config.voice or os.getenv("CARTESIA_VOICE_ID") or DEFAULT_CARTESIA_VOICE_ID,
            ),
        )
    if config.provider == "openai":
        settings: dict[str, Any] = {
            "model": config.model or "gpt-4o-mini-tts",
            "voice": config.voice or "coral",
        }
        if config.instructions is not None:
            settings["instructions"] = config.instructions
        if config.speed is not None:
            settings["speed"] = config.speed
        return OpenAITTSService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAITTSService.Settings(**settings),
        )
    if config.provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(
                model=config.model or "aura-2",
                voice=config.voice or "aura-2-andromeda-en",
            ),
        )
    if config.provider == "gemini_live":
        return GeminiLiveSpeechRendererService(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=config.model or DEFAULT_GEMINI_LIVE_MODEL,
            voice=config.voice or DEFAULT_GEMINI_LIVE_VOICE,
            instructions=config.instructions,
        )
    raise ValueError(f"Unsupported TTS provider: {config.provider}")
