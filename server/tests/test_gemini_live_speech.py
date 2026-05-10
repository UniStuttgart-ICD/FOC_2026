import pytest

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.gemini_live_speech import (
    GeminiLiveSpeechRendererService,
    build_strict_speech_prompt,
    pop_speakable_segments,
)


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


@pytest.mark.asyncio
async def test_renderer_streams_complete_sentence_before_final_flush():
    renderer = CapturingGeminiRenderer()

    await renderer.process_frame(LLMTextFrame("Hello there. This is"), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(LLMTextFrame(" still forming"), FrameDirection.DOWNSTREAM)
    await renderer.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert len(renderer.sent_prompts) == 2
    assert "Hello there." in renderer.sent_prompts[0]
    assert "This is still forming" in renderer.sent_prompts[1]
    assert any(isinstance(frame, TTSAudioRawFrame) for frame in renderer.pushed)
