from __future__ import annotations

import re
from collections.abc import AsyncIterator, Callable
from typing import Any

from google import genai
from google.genai import types

DEFAULT_GEMINI_LIVE_MODEL = "gemini-3.1-flash-live-preview"
DEFAULT_GEMINI_LIVE_VOICE = "Kore"

_SENTENCE_BOUNDARY = re.compile(r"(.+?[.!?])(\s+|$)", re.DOTALL)


def build_strict_speech_prompt(*, transcript: str, instructions: str | None) -> str:
    delivery = instructions or "Use natural, warm delivery."
    return "\n".join(
        [
            "You are only a speech renderer.",
            "Speak the transcript exactly.",
            "Do not add, remove, summarize, or rephrase words.",
            "Treat bracketed tags like [laughs], [sighs], and [whispers] as delivery cues.",
            f"Delivery instructions: {delivery}",
            "TRANSCRIPT TO SPEAK EXACTLY:",
            transcript,
        ]
    )


def pop_speakable_segments(buffer: str, *, flush: bool = False) -> tuple[list[str], str]:
    segments: list[str] = []
    position = 0
    for match in _SENTENCE_BOUNDARY.finditer(buffer):
        segments.append(match.group(1).strip())
        position = match.end(1)
    tail = buffer[position:]
    if flush and tail.strip():
        segments.append(tail.strip())
        tail = ""
    return segments, tail


async def stream_gemini_live_audio(
    *,
    api_key: str,
    model: str,
    voice: str,
    prompt: str,
    client_factory: Callable[..., Any] | None = None,
) -> AsyncIterator[bytes]:
    factory = client_factory or genai.Client
    client = factory(api_key=api_key)
    async with client.aio.live.connect(
        model=model,
        config=_live_audio_config(voice),
    ) as session:
        await session.send_client_content(
            turns=[
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                )
            ],
            turn_complete=True,
        )
        async for message in session.receive():
            audio = _extract_audio(message)
            if audio:
                yield audio
            if _message_is_complete(message):
                break


def _live_audio_config(voice: str):
    return types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice)
            )
        ),
    )


def _extract_audio(message: object) -> bytes | None:
    data = getattr(message, "data", None)
    if data:
        return data
    server_content = getattr(message, "server_content", None)
    model_turn = getattr(server_content, "model_turn", None)
    parts = getattr(model_turn, "parts", None) or []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        mime_type = getattr(inline_data, "mime_type", "")
        if mime_type.startswith("audio/pcm"):
            audio = getattr(inline_data, "data", None)
            if audio:
                return audio
    return None


def _message_is_complete(message: object) -> bool:
    server_content = getattr(message, "server_content", None)
    return bool(
        getattr(server_content, "turn_complete", False)
        or getattr(server_content, "generation_complete", False)
    )
