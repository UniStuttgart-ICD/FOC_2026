from __future__ import annotations

import asyncio
import os
from time import perf_counter
import wave
from pathlib import Path

from voice_runtime.gemini_live_speech import GeminiLiveSpeechRendererService


def _write_wav(path: Path, pcm: bytes, *, rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm)


async def main() -> None:
    api_key = os.environ["GOOGLE_API_KEY"]
    renderer = GeminiLiveSpeechRendererService(
        api_key=api_key,
        model=os.getenv("GEMINI_LIVE_TTS_MODEL", "gemini-3.1-flash-live-preview"),
        voice=os.getenv("GEMINI_LIVE_TTS_VOICE", "Kore"),
        instructions=(
            "Speak the transcript exactly. Use warm delivery and honor bracketed audio tags."
        ),
        connect_on_start=False,
    )
    chunks: list[bytes] = []
    first_audio_at: float | None = None

    async def capture(frame, direction=None):
        nonlocal first_audio_at
        audio = getattr(frame, "audio", None)
        if audio:
            if first_audio_at is None:
                first_audio_at = perf_counter()
            chunks.append(audio)

    renderer.push_frame = capture
    started_at = perf_counter()
    await renderer._stream_prompt_audio(
        "TRANSCRIPT TO SPEAK EXACTLY:\n[laughs softly] Okay, that is surprisingly nice."
    )
    finished_at = perf_counter()
    output = Path("evidence/gemini_live_speech_renderer_smoke.wav")
    pcm = b"".join(chunks)
    _write_wav(output, pcm)
    print(output)
    print(f"bytes={len(pcm)}")
    print(f"chunks={len(chunks)}")
    if first_audio_at is not None:
        print(f"first_audio_ms={round((first_audio_at - started_at) * 1000)}")
    print(f"total_ms={round((finished_at - started_at) * 1000)}")
    print(f"audio_duration_ms={round(len(pcm) / 2 / 24000 * 1000)}")


if __name__ == "__main__":
    asyncio.run(main())
