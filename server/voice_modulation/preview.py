from __future__ import annotations

import asyncio
import base64
import binascii
import io
import os
import wave
from dataclasses import dataclass
from typing import Any

from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.profiles import TTSProfile

PREVIEW_SAMPLE_RATE = 24000


class VoicePreviewError(RuntimeError):
    """Raised when a preview cannot be rendered."""


@dataclass(frozen=True)
class AudioBytes:
    pcm16: bytes
    sample_rate: int
    channels: int = 1


@dataclass(frozen=True)
class PreviewAudio:
    pcm16_base64: str
    wav_base64: str
    sample_rate: int
    channels: int


def pcm16_to_wav_bytes(audio: AudioBytes) -> bytes:
    if audio.sample_rate <= 0:
        raise VoicePreviewError("sample_rate must be positive")
    if audio.channels <= 0:
        raise VoicePreviewError("channels must be positive")
    if len(audio.pcm16) % 2:
        raise VoicePreviewError("pcm16 audio must contain complete 16-bit samples")

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(audio.channels)
        wav.setsampwidth(2)
        wav.setframerate(audio.sample_rate)
        wav.writeframes(audio.pcm16)
    return buffer.getvalue()


def wav_bytes_to_pcm16(data: bytes) -> AudioBytes:
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            pcm16 = wav.readframes(wav.getnframes())
    except wave.Error as exc:
        raise VoicePreviewError(f"Invalid WAV preview: {exc}") from exc
    if sample_width != 2:
        raise VoicePreviewError("WAV preview must be PCM16")
    return AudioBytes(pcm16=pcm16, sample_rate=sample_rate, channels=channels)


def encode_preview(audio: AudioBytes) -> PreviewAudio:
    return PreviewAudio(
        pcm16_base64=base64.b64encode(audio.pcm16).decode("ascii"),
        wav_base64=base64.b64encode(pcm16_to_wav_bytes(audio)).decode("ascii"),
        sample_rate=audio.sample_rate,
        channels=audio.channels,
    )


def decode_preview(preview: PreviewAudio | dict[str, Any]) -> AudioBytes:
    data = preview if isinstance(preview, dict) else preview.__dict__
    try:
        if "pcm16_base64" in data:
            pcm16 = base64.b64decode(str(data["pcm16_base64"]), validate=True)
            return AudioBytes(
                pcm16=pcm16,
                sample_rate=int(data["sample_rate"]),
                channels=int(data.get("channels", 1)),
            )
        if "wav_base64" in data:
            return wav_bytes_to_pcm16(base64.b64decode(str(data["wav_base64"]), validate=True))
    except (binascii.Error, KeyError, TypeError, ValueError) as exc:
        raise VoicePreviewError(f"Invalid preview audio: {exc}") from exc
    raise VoicePreviewError("Preview audio requires pcm16_base64 or wav_base64")


def render_effect_preview(audio: AudioBytes, settings: VoiceModulationSettings) -> AudioBytes:
    try:
        from voice_modulation.dsp import process_pcm16
    except ModuleNotFoundError as exc:
        raise VoicePreviewError("voice_modulation.dsp.process_pcm16 is not available") from exc

    output = process_pcm16(
        audio.pcm16,
        sample_rate=audio.sample_rate,
        num_channels=audio.channels,
        settings=settings,
    )
    return AudioBytes(pcm16=output, sample_rate=audio.sample_rate, channels=audio.channels)


def synthesize_tts_reference(tts: TTSProfile, text: str) -> AudioBytes:
    if not text.strip():
        raise VoicePreviewError("Preview text must not be empty")
    _require_provider_env(tts)
    return asyncio.run(_synthesize_tts_reference(tts, text.strip()))


async def _synthesize_tts_reference(tts: TTSProfile, text: str) -> AudioBytes:
    from pipecat.frames.frames import EndFrame, ErrorFrame, TTSAudioRawFrame

    service, session_to_close = await _create_tts_service(tts)
    _prime_preview_sample_rate(service)
    chunks: list[bytes] = []
    sample_rate: int | None = None
    channels: int | None = None

    try:
        async for frame in service.run_tts(text, "voice-modulation-preview"):
            if frame is None:
                continue
            if isinstance(frame, TTSAudioRawFrame):
                chunks.append(frame.audio)
                sample_rate = frame.sample_rate
                channels = frame.num_channels
            elif isinstance(frame, ErrorFrame):
                raise VoicePreviewError(frame.error)
    finally:
        try:
            stop = getattr(service, "stop", None)
            if stop is not None:
                maybe_awaitable = stop(EndFrame())
                if hasattr(maybe_awaitable, "__await__"):
                    await maybe_awaitable
        finally:
            if session_to_close is not None and not session_to_close.closed:
                await session_to_close.close()

    if not chunks or sample_rate is None or channels is None:
        raise VoicePreviewError("TTS service did not produce audio")
    return AudioBytes(pcm16=b"".join(chunks), sample_rate=sample_rate, channels=channels)


def _prime_preview_sample_rate(service: Any) -> None:
    sample_rate = getattr(service, "sample_rate", None)
    if isinstance(sample_rate, int) and sample_rate <= 0 and hasattr(service, "_sample_rate"):
        setattr(service, "_sample_rate", PREVIEW_SAMPLE_RATE)


async def _create_tts_service(tts: TTSProfile) -> tuple[Any, Any | None]:
    if tts.provider == "openai":
        from pipecat.services.openai.tts import OpenAITTSService

        return (
            OpenAITTSService(
                api_key=os.environ["OPENAI_API_KEY"],
                model=tts.model or "gpt-4o-mini-tts",
                voice=tts.voice or "coral",
                sample_rate=PREVIEW_SAMPLE_RATE,
            ),
            None,
        )
    if tts.provider == "cartesia":
        from pipecat.services.cartesia.tts import CartesiaHttpTTSService

        return (
            CartesiaHttpTTSService(
                api_key=os.environ["CARTESIA_API_KEY"],
                model=tts.model or "sonic-3",
                voice_id=tts.voice or os.getenv("CARTESIA_VOICE_ID"),
                sample_rate=PREVIEW_SAMPLE_RATE,
            ),
            None,
        )
    if tts.provider == "deepgram":
        import aiohttp
        from pipecat.services.deepgram.tts import DeepgramHttpTTSService

        session = aiohttp.ClientSession()
        service = DeepgramHttpTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            aiohttp_session=session,
            sample_rate=PREVIEW_SAMPLE_RATE,
            settings=DeepgramHttpTTSService.Settings(
                voice=tts.voice or tts.model or "aura-2-andromeda-en"
            ),
        )
        return service, session
    if tts.provider == "kokoro":
        from pipecat.services.kokoro.tts import KokoroTTSService

        return (
            KokoroTTSService(voice_id=tts.voice or os.getenv("KOKORO_VOICE_ID") or "af_heart"),
            None,
        )
    raise VoicePreviewError(f"Unsupported TTS provider: {tts.provider}")


def _require_provider_env(tts: TTSProfile) -> None:
    missing: list[str] = []
    if tts.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        missing.append("OPENAI_API_KEY")
    if tts.provider == "cartesia":
        if not os.getenv("CARTESIA_API_KEY"):
            missing.append("CARTESIA_API_KEY")
        if tts.voice is None and not os.getenv("CARTESIA_VOICE_ID"):
            missing.append("CARTESIA_VOICE_ID")
    if tts.provider == "deepgram" and not os.getenv("DEEPGRAM_API_KEY"):
        missing.append("DEEPGRAM_API_KEY")
    if missing:
        raise VoicePreviewError(f"Missing environment variables: {', '.join(missing)}")
