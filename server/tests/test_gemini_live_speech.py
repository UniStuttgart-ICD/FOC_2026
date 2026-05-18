import asyncio
import json
from typing import Any

import pytest
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.gemini_live_speech import (
    GeminiLiveSpeechRendererService,
    build_strict_speech_prompt,
    pop_speakable_segments,
)
from voice_runtime.response_coordination import BotResponseCoordinator, BotSpeechOutputCoordinator

PCM_20MS_24K_MONO = b"\x01\x00" * 480


class MemoryVoiceStreamTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def event(self, event: str, **attributes: Any) -> None:
        self.records.append({"event": event, **attributes})


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

    async def _stream_segment_audio(
        self,
        prompt: str,
        direction: FrameDirection,
    ) -> None:
        self.sent_prompts.append(prompt)
        await self.push_frame(
            TTSAudioRawFrame(audio=b"pcm", sample_rate=24000, num_channels=1),
            direction,
        )

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


class CapturingVoiceModulationProcessor(VoiceModulationProcessor):
    def __init__(
        self,
        settings: VoiceModulationSettings,
        *,
        voice_stream_tracer: MemoryVoiceStreamTracer | None = None,
    ) -> None:
        super().__init__(settings=settings, voice_stream_tracer=voice_stream_tracer)
        self.pushed: list[Frame] = []

    async def push_frame(
        self,
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
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
async def test_renderer_uses_one_tts_lifecycle_for_multiple_segments(monkeypatch):
    prompts: list[str] = []

    async def fake_stream(**kwargs: Any):
        prompts.append(kwargs["prompt"])
        if len(prompts) == 1:
            yield b"segment-1-audio"
            return
        yield b"segment-2-audio"

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        fake_stream,
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

    await renderer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(
        LLMTextFrame("Hello there. This is complete."),
        FrameDirection.DOWNSTREAM,
    )
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert len(prompts) == 2
    assert "Hello there." in prompts[0]
    assert "This is complete." in prompts[1]
    assert sum(isinstance(frame, TTSStartedFrame) for frame in pushed) == 1
    assert sum(isinstance(frame, TTSStoppedFrame) for frame in pushed) == 1
    audio_frames = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
    assert [frame.audio for frame in audio_frames] == [b"segment-1-audio", b"segment-2-audio"]
    start_index = next(i for i, frame in enumerate(pushed) if isinstance(frame, TTSStartedFrame))
    stop_index = next(i for i, frame in enumerate(pushed) if isinstance(frame, TTSStoppedFrame))
    audio_indexes = [i for i, frame in enumerate(pushed) if isinstance(frame, TTSAudioRawFrame)]
    assert start_index < min(audio_indexes)
    assert max(audio_indexes) < stop_index


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
    assert sum(isinstance(frame, TTSStoppedFrame) for frame in pushed) == 1
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
    assert sum(isinstance(frame, TTSStoppedFrame) for frame in pushed) == 1
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


@pytest.mark.asyncio
async def test_stream_prompt_audio_traces_chunks_and_attaches_metadata_without_payloads(
    monkeypatch,
) -> None:
    async def fake_stream(**_: Any):
        yield b"\x01\x00" * 480
        yield b"\x02\x00" * 240

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        fake_stream,
    )
    tracer = MemoryVoiceStreamTracer()
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
        voice_stream_tracer=tracer,
    )
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    renderer.push_frame = capture

    await renderer._stream_prompt_audio("Speak this sensitive transcript")

    start_frame = next(frame for frame in pushed if isinstance(frame, TTSStartedFrame))
    audio_frames = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
    stop_frame = next(frame for frame in pushed if isinstance(frame, TTSStoppedFrame))
    utterance_id = start_frame.metadata["voice_stream_utterance_id"]

    assert utterance_id.startswith("gemini-live-")
    assert [frame.metadata["voice_stream_utterance_id"] for frame in audio_frames] == [
        utterance_id,
        utterance_id,
    ]
    assert [frame.metadata["voice_stream_chunk_seq"] for frame in audio_frames] == [1, 2]
    assert [frame.metadata["voice_stream_source"] for frame in audio_frames] == [
        "gemini_live",
        "gemini_live",
    ]
    assert stop_frame.metadata["voice_stream_utterance_id"] == utterance_id
    assert [record["event"] for record in tracer.records].count("gemini.audio_chunk") == 2
    assert [record["event"] for record in tracer.records].count("gemini.tts_start") == 1
    assert [record["event"] for record in tracer.records].count("gemini.tts_stop") == 1
    first_chunk = next(record for record in tracer.records if record["event"] == "gemini.audio_chunk")
    assert first_chunk["audio_bytes"] == 960
    assert first_chunk["duration_ms"] == 20.0
    trace_json = json.dumps(tracer.records)
    assert "Speak this sensitive transcript" not in trace_json
    assert "0100" not in trace_json


@pytest.mark.asyncio
async def test_renderer_trace_uses_one_utterance_for_multiple_segments(monkeypatch) -> None:
    async def fake_stream(**kwargs: Any):
        if "Hello there." in kwargs["prompt"]:
            yield b"\x01\x00" * 480
            return
        yield b"\x02\x00" * 480

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        fake_stream,
    )
    tracer = MemoryVoiceStreamTracer()
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
        voice_stream_tracer=tracer,
    )
    pushed: list[Frame] = []

    async def capture(frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        pushed.append(frame)

    renderer.push_frame = capture

    await renderer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(
        LLMTextFrame("Hello there. This is complete."),
        FrameDirection.DOWNSTREAM,
    )
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    events = [record["event"] for record in tracer.records]
    assert events.count("gemini.segment_start") == 2
    assert events.count("gemini.tts_start") == 1
    assert events.count("gemini.tts_stop") == 1
    segment_records = [record for record in tracer.records if record["event"] == "gemini.segment_start"]
    assert [record["segment_seq"] for record in segment_records] == [1, 2]
    audio_frames = [frame for frame in pushed if isinstance(frame, TTSAudioRawFrame)]
    utterance_ids = {frame.metadata["voice_stream_utterance_id"] for frame in audio_frames}
    assert len(utterance_ids) == 1
    assert next(iter(utterance_ids)).startswith("gemini-live-")
    assert [frame.metadata["voice_stream_chunk_seq"] for frame in audio_frames] == [1, 2]
    chunk_records = [record for record in tracer.records if record["event"] == "gemini.audio_chunk"]
    assert [record["segment_seq"] for record in chunk_records] == [1, 2]
    assert [record["chunk_seq"] for record in chunk_records] == [1, 2]


@pytest.mark.asyncio
async def test_renderer_two_segments_only_restarts_modulation_once(
    monkeypatch,
) -> None:
    async def fake_stream(**kwargs: Any):
        if "Hello there." in kwargs["prompt"]:
            yield PCM_20MS_24K_MONO
            yield PCM_20MS_24K_MONO
            yield PCM_20MS_24K_MONO
            return
        yield PCM_20MS_24K_MONO

    def fake_process_pcm16(
        audio: bytes,
        *,
        sample_rate: int,
        num_channels: int,
        settings: VoiceModulationSettings,
        state: object,
    ) -> bytes:
        return audio

    monkeypatch.setattr(
        "voice_runtime.gemini_live_speech.stream_gemini_live_audio",
        fake_stream,
    )
    monkeypatch.setattr("voice_modulation.processor.dsp.process_pcm16", fake_process_pcm16)
    tracer = MemoryVoiceStreamTracer()
    renderer = GeminiLiveSpeechRendererService(
        api_key="fake",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
        client_factory=lambda **_: object(),
        connect_on_start=False,
    )
    modulation = CapturingVoiceModulationProcessor(
        VoiceModulationSettings(
            enabled=True,
            gain_db=3.0,
            echo_delay_ms=20.0,
            echo_mix=0.5,
        ),
        voice_stream_tracer=tracer,
    )

    async def route_to_modulation(
        frame: Frame,
        direction: FrameDirection = FrameDirection.DOWNSTREAM,
    ) -> None:
        await modulation.process_frame(frame, direction)

    renderer.push_frame = route_to_modulation

    await renderer.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(
        LLMTextFrame("Hello there. This is complete."),
        FrameDirection.DOWNSTREAM,
    )
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    events = [record["event"] for record in tracer.records]
    assert events.count("modulation.tts_start") == 1
    assert events.count("modulation.prebuffer_release") == 1
    assert events.count("modulation.tts_stop") == 1
    assert events.count("modulation.reset") == 1
    assert any(
        record["event"] == "modulation.audio_push"
        and record["release_mode"] == "tail"
        for record in tracer.records
    )
