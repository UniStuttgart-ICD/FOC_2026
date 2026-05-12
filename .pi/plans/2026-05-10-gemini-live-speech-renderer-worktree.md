# Gemini Live Speech Renderer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prototype Gemini Live as a streaming, TTS-like speech renderer while keeping the existing robot agent and stable runtime profiles untouched.

**Architecture:** Add a new opt-in `gemini_live` TTS provider that consumes the existing agent's `LLMTextFrame` stream, sends short strict-recitation turns to Gemini Live, and pushes streamed PCM audio frames downstream. This runs only in a new worktree and a new runtime profile, so the current Cartesia/OpenAI/Kokoro/Deepgram TTS paths remain unchanged.

**Tech Stack:** Python 3.12, Pipecat frame processors, Google GenAI Live API, `pipecat-ai[google]`, pytest, `uv`, Git worktrees.

---

## Safety Rules

- Do all implementation in `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/.worktrees/gemini-live-speech-renderer`.
- Do not edit the default profile name or the existing `hybrid_openai_stt`, `hybrid_openai_tts`, `hybrid_low_latency`, or `hybrid_gemini` profile behavior.
- Add a new profile named `hybrid_gemini_live_tts`.
- Keep the feature behind `tts.provider = "gemini_live"`.
- Do not run live API smoke tests unless `GOOGLE_API_KEY` is set and the user explicitly wants a live call.
- If the renderer fails to preserve exact text, keep it experimental and do not promote it to default.

## File Map

- Modify: `server/pyproject.toml`
  Add Google/Pipecat dependency support for Gemini Live.
- Modify: `server/voice_runtime/profiles.py`
  Add `gemini_live` as an opt-in streaming TTS provider and require `GOOGLE_API_KEY`.
- Create: `server/voice_runtime/gemini_live_speech.py`
  New strict-recitation prompt builder, text chunker, and Gemini Live renderer service.
- Modify: `server/voice_runtime/providers.py`
  Factory wiring for `gemini_live`.
- Modify: `server/runtime_profiles.toml`
  Add `hybrid_gemini_live_tts` only.
- Modify: `server/tests/test_providers.py`
  Unit-test provider factory wiring.
- Modify: `server/tests/test_voice_runtime_profiles.py`
  Unit-test profile parsing, env names, and streaming validation.
- Create: `server/tests/test_gemini_live_speech.py`
  Unit-test prompt safety, chunking, and mocked streamed audio behavior.
- Create: `server/tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py`
  Optional manual smoke that writes generated PCM/WAV evidence.

---

### Task 0: Create the Isolated Worktree

**Files:**
- No code changes.

- [ ] **Step 1: Confirm current repo state**

Run:

```powershell
git -C C:/Users/Samuel/Documents/github/pipecat/pipecat-agent status --short
git -C C:/Users/Samuel/Documents/github/pipecat/pipecat-agent worktree list
```

Expected: main checkout may be dirty. Existing worktrees are listed. Continue because the new work happens in a separate worktree from `HEAD`.

- [ ] **Step 2: Verify `.worktrees` is ignored**

Run:

```powershell
git -C C:/Users/Samuel/Documents/github/pipecat/pipecat-agent check-ignore -q .worktrees
if ($LASTEXITCODE -ne 0) { throw ".worktrees is not ignored; stop before creating a nested tracked worktree." }
```

Expected: no output, exit code `0`.

- [ ] **Step 3: Create the worktree branch**

Run:

```powershell
git -C C:/Users/Samuel/Documents/github/pipecat/pipecat-agent worktree add .worktrees/gemini-live-speech-renderer -b feature/gemini-live-speech-renderer HEAD
```

Expected: a new clean worktree at:

```text
C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/.worktrees/gemini-live-speech-renderer
```

- [ ] **Step 4: Enter the worktree and verify baseline**

Run:

```powershell
cd C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/.worktrees/gemini-live-speech-renderer/server
uv sync
uv run pytest tests/test_providers.py tests/test_voice_runtime_profiles.py
```

Expected: tests pass or any failures are clearly pre-existing in the clean worktree. If baseline fails, stop and report before implementing.

---

### Task 1: Add Gemini Live Dependency Support

**Files:**
- Modify: `server/pyproject.toml`

- [ ] **Step 1: Write the dependency test by reproducing the import**

Run:

```powershell
cd C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/.worktrees/gemini-live-speech-renderer/server
uv run python -c "from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService; print(GeminiLiveLLMService.__name__)"
```

Expected before the change: failure similar to `Missing module: No module named 'google.api_core'`.

- [ ] **Step 2: Update dependency extras**

In `server/pyproject.toml`, change the Pipecat dependency from:

```toml
"pipecat-ai[cartesia,deepgram,kokoro,openai,runner,silero,webrtc,whisper]",
```

to:

```toml
"pipecat-ai[cartesia,deepgram,google,kokoro,openai,runner,silero,webrtc,whisper]",
```

- [ ] **Step 3: Sync and verify the import**

Run:

```powershell
uv sync
uv run python -c "from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService; print(GeminiLiveLLMService.__name__)"
```

Expected:

```text
GeminiLiveLLMService
```

- [ ] **Step 4: Commit**

Run:

```powershell
git add pyproject.toml uv.lock
git commit -m "chore: enable google pipecat extra for gemini live"
```

---

### Task 2: Extend Runtime Profiles for an Experimental Provider

**Files:**
- Modify: `server/voice_runtime/profiles.py`
- Modify: `server/tests/test_voice_runtime_profiles.py`

- [ ] **Step 1: Add failing profile tests**

Add this test near the existing bundled profile tests in `server/tests/test_voice_runtime_profiles.py`:

```python
def test_hybrid_gemini_live_tts_profile_is_opt_in_streaming_renderer():
    profile = load_runtime_profile(profile_name="hybrid_gemini_live_tts")

    assert profile.category == "benchmark_streaming"
    assert profile.stt.provider == "openai_realtime"
    assert profile.tts.provider == "gemini_live"
    assert profile.tts.model == "gemini-3.1-flash-live-preview"
    assert profile.tts.voice == "Kore"
    assert profile.tts.instructions is not None
    assert "Speak the transcript exactly" in profile.tts.instructions
    assert profile.agent.provider == "gemini_api"
    assert profile.required_env_names() == ("OPENAI_API_KEY", "GOOGLE_API_KEY")
```

Add this parser test near the temporary profile parser tests:

```python
def test_gemini_live_tts_is_allowed_for_benchmark_streaming_profiles(tmp_path: Path) -> None:
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.gemini_live_renderer]
category = "benchmark_streaming"
[profiles.gemini_live_renderer.wake]
provider = "none"
[profiles.gemini_live_renderer.emergency_stop]
enabled = false
[profiles.gemini_live_renderer.stt]
provider = "openai_realtime"
model = "gpt-realtime-whisper"
[profiles.gemini_live_renderer.tts]
provider = "gemini_live"
model = "gemini-3.1-flash-live-preview"
voice = "Kore"
instructions = "Speak the transcript exactly. Do not add or remove words."
[profiles.gemini_live_renderer.agent]
provider = "gemini_api"
model = "gemini-3.1-flash-lite-preview"
[profiles.gemini_live_renderer.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.gemini_live_renderer.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="gemini_live_renderer",
    )

    assert profile.tts.provider == "gemini_live"
    assert profile.required_env_names() == ("OPENAI_API_KEY", "GOOGLE_API_KEY")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/test_voice_runtime_profiles.py -q
```

Expected: failure because `gemini_live` is not an allowed TTS provider and bundled profile does not exist.

- [ ] **Step 3: Update profile types**

In `server/voice_runtime/profiles.py`, change:

```python
TTSProvider = Literal["cartesia", "openai", "deepgram", "kokoro"]
```

to:

```python
TTSProvider = Literal["cartesia", "openai", "deepgram", "kokoro", "gemini_live"]
```

Change:

```python
_TTS_PROVIDERS = {"cartesia", "openai", "deepgram", "kokoro"}
_STREAMING_TTS_PROVIDERS = {"cartesia", "openai", "deepgram"}
```

to:

```python
_TTS_PROVIDERS = {"cartesia", "openai", "deepgram", "kokoro", "gemini_live"}
_STREAMING_TTS_PROVIDERS = {"cartesia", "openai", "deepgram", "gemini_live"}
```

In `RuntimeProfile.required_env_names`, add:

```python
        if self.tts.provider == "gemini_live":
            names.append("GOOGLE_API_KEY")
```

- [ ] **Step 4: Run parser tests again**

Run:

```powershell
uv run pytest tests/test_voice_runtime_profiles.py::test_gemini_live_tts_is_allowed_for_benchmark_streaming_profiles -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add voice_runtime/profiles.py tests/test_voice_runtime_profiles.py
git commit -m "feat: allow experimental gemini live tts provider"
```

---

### Task 3: Add Strict Prompt Builder and Chunker

**Files:**
- Create: `server/voice_runtime/gemini_live_speech.py`
- Create: `server/tests/test_gemini_live_speech.py`

- [ ] **Step 1: Write failing pure-unit tests**

Create `server/tests/test_gemini_live_speech.py` with:

```python
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py -q
```

Expected: import failure because `voice_runtime.gemini_live_speech` does not exist.

- [ ] **Step 3: Implement pure helpers and constructor**

Create `server/voice_runtime/gemini_live_speech.py` with the helper and constructor surface:

```python
from __future__ import annotations

import re
from collections.abc import Callable

from pipecat.processors.frame_processor import FrameProcessor

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


class GeminiLiveSpeechRendererService(FrameProcessor):
    def __init__(
        self,
        *,
        api_key: str,
        model: str = DEFAULT_GEMINI_LIVE_MODEL,
        voice: str = DEFAULT_GEMINI_LIVE_VOICE,
        instructions: str | None = None,
        client_factory: Callable[..., object] | None = None,
        connect_on_start: bool = True,
    ) -> None:
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.instructions = instructions
        self._client_factory = client_factory
        self._connect_on_start = connect_on_start
```

- [ ] **Step 4: Run tests and verify pass**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add voice_runtime/gemini_live_speech.py tests/test_gemini_live_speech.py
git commit -m "feat: add gemini live speech prompt helpers"
```

---

### Task 4: Implement Mocked Streaming Renderer Behavior

**Files:**
- Modify: `server/voice_runtime/gemini_live_speech.py`
- Modify: `server/tests/test_gemini_live_speech.py`

- [ ] **Step 1: Add a mocked streaming test**

Append this to `server/tests/test_gemini_live_speech.py`:

```python
import pytest

from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TTSAudioRawFrame,
)
from pipecat.processors.frame_processor import FrameDirection


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
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py::test_renderer_streams_complete_sentence_before_final_flush -q
```

Expected: failure because `process_frame` does not handle frames yet.

- [ ] **Step 3: Implement minimal frame handling**

In `server/voice_runtime/gemini_live_speech.py`, add imports:

```python
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
```

Add methods to `GeminiLiveSpeechRendererService`:

```python
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMTextFrame):
            self._text_buffer = getattr(self, "_text_buffer", "") + frame.text
            segments, self._text_buffer = pop_speakable_segments(self._text_buffer)
            for segment in segments:
                await self._speak_segment(segment)
            return

        if isinstance(frame, LLMFullResponseEndFrame):
            segments, self._text_buffer = pop_speakable_segments(
                getattr(self, "_text_buffer", ""),
                flush=True,
            )
            for segment in segments:
                await self._speak_segment(segment)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._text_buffer = ""
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _speak_segment(self, transcript: str) -> None:
        prompt = build_strict_speech_prompt(
            transcript=transcript,
            instructions=self.instructions,
        )
        await self._stream_prompt_audio(prompt)

    async def _stream_prompt_audio(self, prompt: str) -> None:
        raise NotImplementedError("Gemini Live streaming is implemented in the next task")
```

- [ ] **Step 4: Run tests**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add voice_runtime/gemini_live_speech.py tests/test_gemini_live_speech.py
git commit -m "feat: chunk llm text for gemini live speech rendering"
```

---

### Task 5: Implement Gemini Live Session Streaming

**Files:**
- Modify: `server/voice_runtime/gemini_live_speech.py`
- Modify: `server/tests/test_gemini_live_speech.py`

- [ ] **Step 1: Add mocked Live session tests**

Add a fake session that yields audio from both documented response shapes:

```python
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
        self.turns: list[object] = []

    async def send_client_content(self, *, turns, turn_complete: bool):
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
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py::test_stream_prompt_audio_pushes_live_audio_frames -q
```

Expected: failure because `_stream_prompt_audio` raises `NotImplementedError`.

- [ ] **Step 3: Implement Live streaming**

Update `server/voice_runtime/gemini_live_speech.py`:

```python
from google import genai
from google.genai import types
from pipecat.frames.frames import TTSAudioRawFrame, TTSStartedFrame, TTSStoppedFrame
```

Implement:

```python
    def _client(self):
        factory = self._client_factory or genai.Client
        return factory(api_key=self.api_key)

    def _live_config(self):
        return types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                )
            ),
        )

    async def _stream_prompt_audio(self, prompt: str) -> None:
        client = self._client()
        audio_started = False
        async with client.aio.live.connect(
            model=self.model,
            config=self._live_config(),
        ) as session:
            await session.send_client_content(turns=prompt, turn_complete=True)
            async for message in session.receive():
                audio = _extract_audio(message)
                if audio:
                    if not audio_started:
                        audio_started = True
                        await self.push_frame(TTSStartedFrame())
                    await self.push_frame(
                        TTSAudioRawFrame(audio=audio, sample_rate=24000, num_channels=1)
                    )
                if _message_is_complete(message):
                    break
        if audio_started:
            await self.push_frame(TTSStoppedFrame())


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
```

- [ ] **Step 4: Run renderer tests**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add voice_runtime/gemini_live_speech.py tests/test_gemini_live_speech.py
git commit -m "feat: stream gemini live audio from strict speech prompts"
```

---

### Task 6: Wire the Provider Factory

**Files:**
- Modify: `server/voice_runtime/providers.py`
- Modify: `server/tests/test_providers.py`

- [ ] **Step 1: Add failing provider factory test**

Add to `server/tests/test_providers.py`:

```python
def test_creates_gemini_live_tts(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg")
    with patch("voice_runtime.providers.GeminiLiveSpeechRendererService") as service:
        create_tts_service(
            TTSConfig(
                provider="gemini_live",
                model="gemini-3.1-flash-live-preview",
                voice="Kore",
                instructions="Speak the transcript exactly.",
            )
        )

    service.assert_called_once_with(
        api_key="gg",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
    )
```

- [ ] **Step 2: Run and verify failure**

Run:

```powershell
uv run pytest tests/test_providers.py::test_creates_gemini_live_tts -q
```

Expected: failure because provider factory has no `gemini_live` branch.

- [ ] **Step 3: Implement provider branch**

In `server/voice_runtime/providers.py`, add:

```python
from voice_runtime.gemini_live_speech import (
    DEFAULT_GEMINI_LIVE_MODEL,
    DEFAULT_GEMINI_LIVE_VOICE,
    GeminiLiveSpeechRendererService,
)
```

Add before the final `raise`:

```python
    if config.provider == "gemini_live":
        return GeminiLiveSpeechRendererService(
            api_key=os.environ["GOOGLE_API_KEY"],
            model=config.model or DEFAULT_GEMINI_LIVE_MODEL,
            voice=config.voice or DEFAULT_GEMINI_LIVE_VOICE,
            instructions=config.instructions,
        )
```

- [ ] **Step 4: Run provider tests**

Run:

```powershell
uv run pytest tests/test_providers.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add voice_runtime/providers.py tests/test_providers.py
git commit -m "feat: wire gemini live speech renderer provider"
```

---

### Task 7: Add the Isolated Runtime Profile

**Files:**
- Modify: `server/runtime_profiles.toml`
- Modify: `server/tests/test_voice_runtime_profiles.py`

- [ ] **Step 1: Add profile**

Add this profile to `server/runtime_profiles.toml` without changing any existing profile:

```toml
[profiles.hybrid_gemini_live_tts]
category = "benchmark_streaming"

[profiles.hybrid_gemini_live_tts.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.85
vad_threshold = 0.0
candidate_log_threshold = 0.3
required_hits = 1
min_wake_rms = 4.0
min_wake_peak = 12
pre_buffer_s = 0.35
single_command = true

[profiles.hybrid_gemini_live_tts.emergency_stop]
enabled = false

[profiles.hybrid_gemini_live_tts.stt]
provider = "openai_realtime"
model = "gpt-realtime-whisper"

[profiles.hybrid_gemini_live_tts.tts]
provider = "gemini_live"
model = "gemini-3.1-flash-live-preview"
voice = "Kore"
instructions = "Speak the transcript exactly. Do not add, remove, summarize, or rephrase words. Use warm, emotionally present delivery. Treat bracketed audio tags such as [laughs], [sighs], and [whispers] as performance cues."

[profiles.hybrid_gemini_live_tts.agent]
provider = "gemini_api"
model = "gemini-3.1-flash-lite-preview"
reasoning_effort = "high"
api_key_env = "GOOGLE_API_KEY"

[profiles.hybrid_gemini_live_tts.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.hybrid_gemini_live_tts.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.hybrid_gemini_live_tts.process_trace]
enabled = true
path = "logs/process_trace.jsonl"
include_text = true
include_tool_payloads = true
```

- [ ] **Step 2: Run profile test**

Run:

```powershell
uv run pytest tests/test_voice_runtime_profiles.py::test_hybrid_gemini_live_tts_profile_is_opt_in_streaming_renderer -q
```

Expected: pass.

- [ ] **Step 3: Run full focused tests**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py tests/test_providers.py tests/test_voice_runtime_profiles.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

Run:

```powershell
git add runtime_profiles.toml tests/test_voice_runtime_profiles.py
git commit -m "feat: add opt-in gemini live tts profile"
```

---

### Task 8: Add Manual Live Smoke Test

**Files:**
- Create: `server/tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py`

- [ ] **Step 1: Create smoke script**

Create `server/tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py`:

```python
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
```

- [ ] **Step 2: Run only when live credentials are available**

Run:

```powershell
if (-not $env:GOOGLE_API_KEY) { throw "GOOGLE_API_KEY is required for live smoke." }
uv run python tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py
```

Expected: script prints `evidence/gemini_live_speech_renderer_smoke.wav` and a byte count greater than `0`.

- [ ] **Step 3: Commit**

Run:

```powershell
git add tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py
git commit -m "test: add manual gemini live speech smoke"
```

---

### Task 9: Final Verification and Decision Gate

**Files:**
- No new source files unless tests reveal a focused fix.

- [ ] **Step 1: Run static focused test suite**

Run:

```powershell
uv run pytest tests/test_gemini_live_speech.py tests/test_providers.py tests/test_voice_runtime_profiles.py tests/test_config.py -q
```

Expected: all pass.

- [ ] **Step 2: Run import verification**

Run:

```powershell
uv run python -c "from voice_runtime.providers import create_tts_service; from voice_runtime.profiles import load_runtime_profile; p = load_runtime_profile(profile_name='hybrid_gemini_live_tts'); print(p.tts.provider, p.tts.model)"
```

Expected:

```text
gemini_live gemini-3.1-flash-live-preview
```

- [ ] **Step 3: Optional live timing check**

Run only with consent and `GOOGLE_API_KEY`:

```powershell
Measure-Command { uv run python tests/live_robot_smoke/manual_gemini_live_speech_renderer_smoke.py }
```

Expected: evidence WAV exists and audio starts from streamed chunks. Listen to the file and verify:

- It speaks only the requested words.
- It does not read the prompt instructions aloud.
- It honors `[laughs softly]` or similar tags better than OpenAI TTS.
- It returns audio fast enough to justify deeper integration.

- [ ] **Step 4: Record decision**

If the live smoke preserves exact transcript and latency is acceptable, keep the branch and propose a second plan for production hardening. If it adds/removes words or reads instructions aloud, leave the branch as experimental and prefer OpenAI streaming TTS or Cartesia for production.

Run:

```powershell
git status --short
git log --oneline --max-count=10
```

Expected: only intentional committed changes, no accidental edits to unrelated files.

---

## Self-Review

- Spec coverage: the plan isolates implementation in a worktree, adds a Gemini Live speech-rendering provider, keeps current defaults safe, and includes live testing only as an opt-in step.
- Placeholder scan: no `TBD`, no deferred test step, no vague "handle errors" step.
- Type consistency: provider name is consistently `gemini_live`; model is consistently `gemini-3.1-flash-live-preview`; default voice is consistently `Kore`.
- Risk note: Gemini Live is a generative conversational model, not strict TTS. The explicit decision gate prevents promoting it unless exact-recitation behavior is proven.
