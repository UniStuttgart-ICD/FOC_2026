import asyncio
from typing import Any

import pytest
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.gemini_live_speech import (
    GeminiLiveSpeechRendererService,
    build_strict_speech_prompt,
    pop_speakable_segments,
)
from voice_runtime.response_coordination import BotResponseCoordinator, BotSpeechOutputCoordinator


def test_build_strict_speech_prompt_keeps_transcript_in_fenced_section():
    prompt = build_strict_speech_prompt(
        transcript="[laughs] I can do that.",
        instructions="Use warm, delighted delivery.",
    )

    assert "Speak the transcript exactly" in prompt
    assert "Do not add, remove, summarize, or rephrase words." in prompt
    assert "Use warm, delighted delivery." in prompt
    assert "TRANSCRIPT TO SPEAK EXACTLY" in prompt
    assert "[laughs] I can do that." in prompt


def test_pop_speakable_segments_keeps_incomplete_tail():
    buffer = "Sure. Move up slowly and then"

    segments, tail = pop_speakable_segments(buffer)

    assert segments == ["Sure."]
    assert tail == " Move up slowly and then"


def test_pop_speakable_segments_flushes_tail_when_requested():
    segments, tail = pop_speakable_segments("Move up slowly", flush=True)

    assert segments == ["Move up slowly"]
    assert tail == ""


def test_renderer_defaults_are_conservative():
    service = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
    )

    assert service.model == "gemini-3.1-flash-live-preview"
    assert service.voice == "Kore"


class CapturingGeminiRenderer(GeminiLiveSpeechRendererService):
    def __init__(self) -> None:
        super().__init__(
            api_key="fake",
            model="gemini-3.1-flash-live-preview",
            voice="Kore",
            instructions="Speak the transcript exactly.",
            client_factory=lambda **_: object(),
            connect_on_start=False,
        )
        self.sent_prompts: list[str] = []
        self.pushed: list[Frame] = []

    async def _stream_prompt_audio(self, prompt: str) -> None:
        self.sent_prompts.append(prompt)
        await self.push_frame(TTSAudioRawFrame(audio=b"pcm", sample_rate=24000, num_channels=1))

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


class CapturingOutputCoordinator(BotSpeechOutputCoordinator):
    def __init__(self, coordinator: BotResponseCoordinator) -> None:
        super().__init__(coordinator=coordinator, enable_direct_mode=True)
        self.pushed: list[Frame] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append(frame)


@pytest.mark.asyncio
async def test_renderer_streams_complete_sentence_before_final_flush():
    renderer = CapturingGeminiRenderer()

    await renderer.process_frame(LLMTextFrame("Hello there. This is"), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(LLMTextFrame(" still forming"), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert len(renderer.sent_prompts) == 2
    assert "Hello there." in renderer.sent_prompts[0]
    assert "This is still forming" in renderer.sent_prompts[1]
    text_frames = [frame for frame in renderer.pushed if isinstance(frame, LLMTextFrame)]
    assert [frame.text for frame in text_frames] == ["Hello there. This is", " still forming"]
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in renderer.pushed)


@pytest.mark.asyncio
async def test_stream_prompt_audio_emits_tts_stop_when_stream_raises(monkeypatch):
    async def broken_stream(**_: Any):
        yield b"audio-1"
        raise RuntimeError("stream failed")

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        broken_stream,
    )
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
    )
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    renderer.push_frame = capture

    with pytest.raises(RuntimeError, match="stream failed"):
        await renderer._stream_prompt_audio("Speak this")

    assert any(isinstance(frame, TTSAudioRawFrame) for frame in pushed)
    assert isinstance(pushed[-1], TTSStoppedFrame)


@pytest.mark.asyncio
async def test_stream_prompt_audio_emits_tts_stop_before_reraising_cancellation(monkeypatch):
    async def cancelled_stream(**_: Any):
        yield b"audio-1"
        raise asyncio.CancelledError

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        cancelled_stream,
    )
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
    )
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    renderer.push_frame = capture

    with pytest.raises(asyncio.CancelledError):
        await renderer._stream_prompt_audio("Speak this")

    assert any(isinstance(frame, TTSAudioRawFrame) for frame in pushed)
    assert isinstance(pushed[-1], TTSStoppedFrame)


@pytest.mark.asyncio
async def test_gemini_stream_failure_does_not_leave_response_coordinator_locked(monkeypatch):
    async def broken_stream(**_: Any):
        yield b"audio-1"
        raise RuntimeError("stream failed")

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        broken_stream,
    )
    coordinator = BotResponseCoordinator()
    output = CapturingOutputCoordinator(coordinator)
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
    )

    async def route_to_output(
        frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        await output.process_frame(frame, direction)

    renderer.push_frame = route_to_output

    await coordinator.begin_response()
    await renderer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    with pytest.raises(RuntimeError, match="stream failed"):
        await renderer.process_frame(LLMTextFrame("Execution complete."), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert coordinator.is_response_active is False
    await asyncio.wait_for(coordinator.begin_response(), timeout=0.1)
    coordinator.finish_response()


class FakeInlineData:
    mime_type = "audio/pcm;rate=24000"
    data = b"audio-2"


class FakePart:
    inline_data = FakeInlineData()


class FakeModelTurn:
    parts = [FakePart()]


class FakeServerContent:
    model_turn = FakeModelTurn()
    turn_complete = True
    generation_complete = True


class FakeMessage:
    def __init__(self, data: bytes | None = None) -> None:
        self.data = data
        self.server_content = FakeServerContent() if data is None else None


class FakeSession:
    def __init__(self) -> None:
        self.turns: list[Any] = []

    async def send_client_content(self, *, turns, turn_complete: bool):
        assert not isinstance(turns, str)
        self.turns.append(turns)
        assert turn_complete is True

    async def receive(self):
        yield FakeMessage(data=b"audio-1")
        yield FakeMessage(data=None)


class FakeLive:
    def __init__(self, session: FakeSession) -> None:
        self._session = session

    def connect(self, *, model, config):
        return self

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


class FakeAio:
    def __init__(self, session: FakeSession) -> None:
        self.live = FakeLive(session)


class FakeClient:
    def __init__(self, session: FakeSession) -> None:
        self.aio = FakeAio(session)


@pytest.mark.asyncio
async def test_stream_prompt_audio_pushes_live_audio_frames():
    session = FakeSession()
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: FakeClient(session),
        connect_on_start=False,
    )
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    renderer.push_frame = capture

    await renderer._stream_prompt_audio("Speak this")

    audio_frames = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
    assert [frame.audio for frame in audio_frames] == [b"audio-1", b"audio-2"]
    assert all(frame.sample_rate == 24000 for frame in audio_frames)
    assert session.turns
    assert session.turns[0][0].role == "user"
    assert session.turns[0][0].parts[0].text == "Speak this"
