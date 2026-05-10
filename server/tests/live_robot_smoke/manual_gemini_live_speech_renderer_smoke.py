from __future__ import annotations

import asyncio
import os
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

    async def capture(frame, direction=None):
        audio = getattr(frame, "audio", None)
        if audio:
            chunks.append(audio)

    renderer.push_frame = capture
    await renderer._stream_prompt_audio(
        "TRANSCRIPT TO SPEAK EXACTLY:\n[laughs softly] Okay, that is surprisingly nice."
    )
    output = Path("evidence/gemini_live_speech_renderer_smoke.wav")
    _write_wav(output, b"".join(chunks))
    print(output)
    print(sum(len(chunk) for chunk in chunks))


if __name__ == "__main__":
    asyncio.run(main())
