from __future__ import annotations

from typing import Any

from voice_modulation.preview import AudioBytes, synthesize_tts_reference
from voice_runtime.profiles import TTSProfile


class _FakeServerContent:
    turn_complete = True
    generation_complete = True


class _FakeMessage:
    def __init__(self, data: bytes, *, complete: bool = False) -> None:
        self.data = data
        self.server_content = _FakeServerContent() if complete else None


class _FakeSession:
    def __init__(self) -> None:
        self.turns: list[Any] = []

    async def __aenter__(self) -> _FakeSession:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None

    async def send_client_content(self, *, turns: list[Any], turn_complete: bool) -> None:
        self.turns.extend(turns)
        self.turn_complete = turn_complete

    async def receive(self) -> Any:
        yield _FakeMessage(b"audio-1")
        yield _FakeMessage(b"audio-2", complete=True)


class _FakeLive:
    def __init__(self, session: _FakeSession) -> None:
        self.session = session
        self.connect_calls: list[dict[str, Any]] = []

    def connect(self, **kwargs: Any) -> _FakeSession:
        self.connect_calls.append(kwargs)
        return self.session


class _FakeAio:
    def __init__(self, live: _FakeLive) -> None:
        self.live = live


class _FakeClient:
    def __init__(self, live: _FakeLive) -> None:
        self.aio = _FakeAio(live)


def test_synthesize_tts_reference_supports_gemini_live(monkeypatch: Any) -> None:
    session = _FakeSession()
    live = _FakeLive(session)

    monkeypatch.setenv("GOOGLE_API_KEY", "fake-key")
    monkeypatch.setattr(
        "voice_modulation.preview.genai.Client",
        lambda **_kwargs: _FakeClient(live),
    )

    audio = synthesize_tts_reference(
        TTSProfile(
            provider="gemini_live",
            model="gemini-3.1-flash-live-preview",
            voice="Kore",
            instructions="Use calm, precise delivery.",
        ),
        "Status report.",
    )

    assert audio == AudioBytes(pcm16=b"audio-1audio-2", sample_rate=24000, channels=1)
    assert live.connect_calls[0]["model"] == "gemini-3.1-flash-live-preview"
    assert session.turn_complete is True
    prompt = session.turns[0].parts[0].text
    assert "Use calm, precise delivery." in prompt
    assert "Status report." in prompt
