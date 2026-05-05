# Orthogonal Reusable Voice Runtime Modules Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the Voice Runtime into orthogonal, reusable Modules with clear Interfaces and minimal intermingling between runtime profiles, wake commands, agent turns, robot safety, pipeline assembly, and metrics.

**Architecture:** Add a new `server/voice_runtime/` package that exposes reusable Modules. Parallel issues create deep Modules behind compatibility seams without changing `bot.py` or `pipeline_builder.py`; a final integration issue switches the app to the new Interfaces and adds a reusable Voice Runtime assembly Module. Each Module owns its invariants and has tests at its Interface, not tests that reach through implementation details.

**Tech Stack:** Python 3.10+, Pipecat, pytest, pytest-asyncio, ruff, pyright, TOML via `tomllib`/`tomli`.

---

## Orthogonality rules for every issue

- Use the vocabulary in `CONTEXT.md`: Voice Runtime, Runtime Profile, Voice Command, Agent Turn, Robot Safety, Robot Tool Adapter, Safety Coverage, Voice Metrics, Voice Runtime Assembly.
- Use the architecture vocabulary: Module, Interface, Implementation, Seam, Adapter, Depth, Leverage, Locality.
- Each parallel issue owns only its listed files. Do not edit `server/bot.py`, `server/pipeline_builder.py`, existing top-level modules, or unrelated tests until Issue 7.
- New reusable Modules live under `server/voice_runtime/`.
- A reusable Module must not import `bot.py`, `pipeline_builder.py`, `.env` loading, or project docs.
- Pure policy Modules must not import Pipecat. Pipecat-specific code belongs in Adapter Modules.
- The Interface is the test surface. Prefer tests that call the new public Interface over tests that patch implementation internals.
- Each issue must state what Implementation complexity sits behind its Interface and pass the deletion test in its final summary.
- Keep compatibility with the current app until Issue 7. Existing tests should continue to pass after each issue.
- Do not overclaim Robot Safety coverage: a backend is locally safety-enforced only if its Robot Tool Adapter calls `voice_runtime.robot_safety` before every robot tool call. Direct Claude MCP remains prompt-only unless a safe MCP proxy Adapter is added later.
- Voice Command spans two Pipecat positions: an audio Adapter before STT and a transcript Adapter after STT. Keep them in one Module and construct them together so the caller does not coordinate reset callbacks manually.
- Voice Metrics should consume semantic Voice Runtime events where available; do not couple metrics to processor class names.

## Execution graph

Run Issue 1 first. Then run Issues 2-6 in parallel in separate worktrees/branches. Run Issue 7 after the parallel issues are merged. Run Issue 8 last.

```text
Issue 1: Foundation package and import contract
├── Issue 2: Runtime Profile Module
├── Issue 3: Voice Command Module
├── Issue 4: Agent Turn Module
├── Issue 5: Robot Safety Module
└── Issue 6: Voice Metrics Module
Issue 7: Integrate Modules and add the Voice Runtime assembly Interface
Issue 8: Documentation and extractability review
```

---

## Target file structure

```text
pipecat-agent/
├── CONTEXT.md
├── .pi/plans/2026-05-04-orthogonal-reusable-voice-runtime.md
└── server/
    ├── voice_runtime/
    │   ├── __init__.py
    │   ├── contracts.py
    │   ├── profiles.py
    │   ├── wake_command.py
    │   ├── agent_turn.py
    │   ├── robot_safety.py
    │   ├── voice_metrics.py
    │   └── assembly.py
    ├── tests/
    │   ├── test_voice_runtime_contracts.py
    │   ├── test_voice_runtime_profiles.py
    │   ├── test_voice_runtime_wake_command.py
    │   ├── test_voice_runtime_agent_turn.py
    │   ├── test_voice_runtime_robot_safety.py
    │   ├── test_voice_runtime_voice_metrics.py
    │   ├── test_voice_runtime_assembly.py
    │   └── test_orthogonal_imports.py
    ├── config.py                       # compatibility wrapper after Issue 7
    ├── providers.py                    # compatibility wrapper after Issue 7
    ├── wake/wake_gate.py               # compatibility wrapper after Issue 7
    ├── wake/transcript_cleanup.py      # compatibility wrapper after Issue 7
    ├── robot_mcp_bridge.py             # uses robot_safety after Issue 7
    ├── metrics.py                      # uses voice_metrics after Issue 7
    ├── claude_agent_processor.py       # uses agent_turn after Issue 7
    ├── openai_codex_agent_processor.py # uses agent_turn + robot_safety after Issue 7
    ├── pipeline_builder.py             # delegates ordering to voice_runtime.assembly after Issue 7
    └── bot.py                          # calls explicit Agent Turn lifecycle after Issue 7
```

---

# Issue 1: Foundation package and import contract

**Parallelization:** Required foundation. Land this before Issues 2-6.

**Files:**
- Create: `server/voice_runtime/__init__.py`
- Create: `server/voice_runtime/contracts.py`
- Create: `server/tests/test_voice_runtime_contracts.py`

**Purpose:** Create a small reusable package and shared contract vocabulary without pulling in app-specific dependencies.

- [ ] **Step 1: Create the failing contract tests**

Create `server/tests/test_voice_runtime_contracts.py`:

```python
import pytest

from voice_runtime.contracts import AsyncLifecycle, VoiceRuntimeError


class FakeLifecycle:
    def __init__(self):
        self.events = []

    async def connect(self) -> None:
        self.events.append("connect")

    async def disconnect(self) -> None:
        self.events.append("disconnect")



@pytest.mark.asyncio
async def test_async_lifecycle_protocol_is_structural():
    lifecycle: AsyncLifecycle = FakeLifecycle()

    await lifecycle.connect()
    await lifecycle.disconnect()

    assert lifecycle.events == ["connect", "disconnect"]


def test_voice_runtime_error_is_runtime_specific():
    error = VoiceRuntimeError("bad profile")

    assert str(error) == "bad profile"
```

- [ ] **Step 2: Run the contract tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_contracts.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'voice_runtime'`.

- [ ] **Step 3: Create the foundation package**

Create `server/voice_runtime/__init__.py`:

```python
"""Reusable Voice Runtime Modules for Pipecat robot agents."""
```

Create `server/voice_runtime/contracts.py`:

```python
from __future__ import annotations

from typing import Protocol


class VoiceRuntimeError(RuntimeError):
    """Raised when a reusable Voice Runtime Module cannot satisfy its Interface."""


class AsyncLifecycle(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

```

- [ ] **Step 4: Run the contract tests and verify pass**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_contracts.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit Issue 1**

```bash
cd pipecat-agent
git add server/voice_runtime server/tests/test_voice_runtime_contracts.py
git commit -m "feat: add voice runtime foundation contracts"
```

---

# Issue 2: Runtime Profile Module

**Parallelization:** Can run after Issue 1. Owns only `profiles.py` and its tests.

**Files:**
- Create: `server/voice_runtime/profiles.py`
- Create: `server/tests/test_voice_runtime_profiles.py`

**Purpose:** Make the Runtime Profile Module own profile parsing, defaults, required environment variables, and provider policy without constructing Pipecat objects.

**Interface target:** `load_runtime_profile(...) -> RuntimeProfile`, `RuntimeProfile.required_env_names() -> tuple[str, ...]`, and typed config dataclasses. This Module should not import Pipecat.

- [ ] **Step 1: Write failing profile tests**

Create `server/tests/test_voice_runtime_profiles.py`:

```python
from pathlib import Path

import pytest

from voice_runtime.profiles import ProfileError, load_runtime_profile


def _write_profiles(path: Path) -> None:
    path.write_text(
        """
[profiles.hybrid_low_latency]
category = "benchmark_streaming"
[profiles.hybrid_low_latency.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true
[profiles.hybrid_low_latency.emergency_stop]
enabled = false
[profiles.hybrid_low_latency.stt]
provider = "deepgram_flux"
model = "flux-general-en"
[profiles.hybrid_low_latency.tts]
provider = "cartesia"
model = "sonic-3"
voice = "voice-id"
[profiles.hybrid_low_latency.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.hybrid_low_latency.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.hybrid_low_latency.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.no_wake_debug]
category = "local_debug"
[profiles.no_wake_debug.wake]
provider = "none"
[profiles.no_wake_debug.emergency_stop]
enabled = false
[profiles.no_wake_debug.stt]
provider = "whisper"
model = "base"
device = "cpu"
[profiles.no_wake_debug.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.no_wake_debug.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
[profiles.no_wake_debug.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.no_wake_debug.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )


def test_loads_profile_without_constructing_adapters(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert profile.name == "hybrid_low_latency"
    assert profile.category == "benchmark_streaming"
    assert profile.wake.provider == "openwakeword"
    assert profile.wake.model_path == tmp_path / "models" / "mave.onnx"
    assert profile.stt.provider == "deepgram_flux"
    assert profile.tts.provider == "cartesia"
    assert profile.agent.provider == "openai_codex_oauth"
    assert profile.mcp_robot_url == "http://127.0.0.1:8765/mcp"


def test_profile_reports_required_env_names_without_reading_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="hybrid_low_latency",
    )

    assert profile.required_env_names() == ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY")


def test_cartesia_profile_without_voice_requires_voice_id_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    profiles_path.write_text(
        """
[profiles.cartesia_without_voice]
category = "benchmark_streaming"
[profiles.cartesia_without_voice.wake]
provider = "none"
[profiles.cartesia_without_voice.emergency_stop]
enabled = false
[profiles.cartesia_without_voice.stt]
provider = "deepgram_flux"
[profiles.cartesia_without_voice.tts]
provider = "cartesia"
model = "sonic-3"
[profiles.cartesia_without_voice.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.cartesia_without_voice.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.cartesia_without_voice.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="cartesia_without_voice",
    )

    assert profile.required_env_names() == ("DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "CARTESIA_VOICE_ID")


def test_local_profile_has_no_cloud_stt_tts_env_requirements(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="no_wake_debug",
    )

    assert profile.required_env_names() == ()


def test_benchmark_profile_rejects_local_stt(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    profiles_path.write_text(
        """
[profiles.bad]
category = "benchmark_streaming"
[profiles.bad.wake]
provider = "none"
[profiles.bad.emergency_stop]
enabled = false
[profiles.bad.stt]
provider = "whisper"
[profiles.bad.tts]
provider = "cartesia"
voice = "voice-id"
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
[profiles.bad.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad.metrics]
enabled = false
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="benchmark_streaming profiles require streaming STT"):
        load_runtime_profile(profiles_path=profiles_path, server_dir=tmp_path, profile_name="bad")
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_profiles.py -v
```

Expected: FAIL with `ModuleNotFoundError` or missing `profiles.py` symbols.

- [ ] **Step 3: Implement `voice_runtime.profiles`**

Implement dataclasses equivalent to the existing `config.py` dataclasses, but with these changes:

```python
# public names required by tests and Issue 7
ProfileError
WakeProfile
EmergencyStopProfile
STTProfile
TTSProfile
AgentProfile
MetricsProfile
RuntimeProfile
load_runtime_profile
```

Required behavior:

- Parse TOML from `profiles_path`.
- Resolve relative paths against `server_dir`.
- Do not read `os.environ` in this Module.
- Put env policy behind `RuntimeProfile.required_env_names()`.
- Include `CARTESIA_VOICE_ID` when `tts.provider == "cartesia"` and the Runtime Profile omits `tts.voice`.
- Validate `benchmark_streaming` only uses streaming STT/TTS.
- Validate enabled wake and emergency stop profiles have model paths.
- Preserve current strict scalar validation, including rejecting bool values where numbers are expected.
- Use deterministic env order: Deepgram, Cartesia, Cartesia voice ID, OpenAI.
- Raise `ProfileError` for invalid profiles.

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_profiles.py -v
uv run ruff check voice_runtime/profiles.py tests/test_voice_runtime_profiles.py
uv run pyright voice_runtime/profiles.py tests/test_voice_runtime_profiles.py
```

Expected: PASS, ruff pass, pyright 0 errors.

- [ ] **Step 5: Commit Issue 2**

```bash
cd pipecat-agent
git add server/voice_runtime/profiles.py server/tests/test_voice_runtime_profiles.py
git commit -m "feat: add reusable runtime profile module"
```

---

# Issue 3: Voice Command Module

**Parallelization:** Can run after Issue 1. Owns only `wake_command.py` and its tests.

**Files:**
- Create: `server/voice_runtime/wake_command.py`
- Create: `server/tests/test_voice_runtime_wake_command.py`

**Purpose:** Concentrate Voice Command behavior in one Module while respecting Pipecat topology. The audio Adapter runs before STT and blocks/replays audio; the transcript Adapter runs after STT and strips the wake phrase, emits the finalized command, and rearms. Callers construct both through one factory so reset ordering is local to the Module.

**Interface target:** `build_mave_voice_command_processors(...) -> MaveVoiceCommandProcessors`, `WakeDetectedFrame`, and `strip_mave_wake_phrase(...)`. The Module owns the single-command invariant; callers should not coordinate a separate gate and cleaner callback.

- [ ] **Step 1: Write failing wake-command tests**

Create `server/tests/test_voice_runtime_wake_command.py`:

```python
from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.wake_command import (
    WakeDetectedFrame,
    build_mave_voice_command_processors,
    strip_mave_wake_phrase,
)


class CapturingAudioGate:
    def __init__(self, audio_gate):
        self.audio_gate = audio_gate
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


class CapturingTranscriptAdapter:
    def __init__(self, transcript_adapter):
        self.transcript_adapter = transcript_adapter
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def _audio(value: int, samples: int = 1600) -> InputAudioRawFrame:
    pcm = np.full(samples, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=pcm, sample_rate=16000, num_channels=1)


def _capture(monkeypatch, processors):
    audio_capture = CapturingAudioGate(processors.audio_gate)
    transcript_capture = CapturingTranscriptAdapter(processors.transcript_adapter)
    monkeypatch.setattr(processors.audio_gate, "push_frame", audio_capture.push_frame)
    monkeypatch.setattr(processors.transcript_adapter, "push_frame", transcript_capture.push_frame)
    return audio_capture, transcript_capture


def test_strip_mave_wake_phrase_handles_common_transcription_variants():
    assert strip_mave_wake_phrase("Mave, move up") == "move up"
    assert strip_mave_wake_phrase("hey Maeve stop") == "stop"
    assert strip_mave_wake_phrase("move up") == "move up"


@pytest.mark.asyncio
async def test_audio_adapter_blocks_until_wake_replays_prebuffer_and_emits_wake_event(monkeypatch):
    detector = Mock()
    detector.detected.side_effect = [(False, None, 0.0), (True, "mave", 0.91)]
    processors = build_mave_voice_command_processors(detector=detector, pre_buffer_s=1.5)
    audio_capture, _ = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.audio_gate.process_frame(_audio(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [frame for frame, _ in audio_capture.pushed if isinstance(frame, InputAudioRawFrame)]
    wake_events = [frame for frame, _ in audio_capture.pushed if isinstance(frame, WakeDetectedFrame)]
    assert [np.frombuffer(frame.audio, dtype=np.int16)[0] for frame in pushed_audio] == [1, 2]
    assert wake_events[0].wake_phrase == "mave"
    assert wake_events[0].score == 0.91
    assert processors.audio_gate.is_awake is True


@pytest.mark.asyncio
async def test_transcript_adapter_cleans_finalized_command_and_rearms_audio_gate(monkeypatch):
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(detector=detector, pre_buffer_s=1.5, rearm_delay_s=0.0)
    _, transcript_capture = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.transcript_adapter.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    transcripts = [frame for frame, _ in transcript_capture.pushed if isinstance(frame, TranscriptionFrame)]
    assert transcripts[0].text == "move up"
    assert processors.audio_gate.is_awake is False


@pytest.mark.asyncio
async def test_empty_cleaned_transcript_is_not_emitted_but_rearms_when_single_command_is_enabled(monkeypatch):
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.91)
    processors = build_mave_voice_command_processors(detector=detector, pre_buffer_s=1.5, rearm_delay_s=0.0)
    _, transcript_capture = _capture(monkeypatch, processors)

    await processors.audio_gate.process_frame(_audio(1), FrameDirection.DOWNSTREAM)
    await processors.transcript_adapter.process_frame(
        TranscriptionFrame(text="Mave", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    assert not [frame for frame, _ in transcript_capture.pushed if isinstance(frame, TranscriptionFrame)]
    assert processors.audio_gate.is_awake is False
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_wake_command.py -v
```

Expected: FAIL with missing `voice_runtime.wake_command` symbols.

- [ ] **Step 3: Implement the Module**

Create `server/voice_runtime/wake_command.py` with:

- `strip_mave_wake_phrase(text: str) -> str`
- `class WakeDetectedFrame(Frame)` with `wake_phrase: str`, `model_name: str | None`, and `score: float`
- `class MaveVoiceCommandAudioGate(FrameProcessor)` for pre-STT audio gating
- `class MaveVoiceCommandTranscriptAdapter(FrameProcessor)` for post-STT transcript cleanup and rearm
- `@dataclass(frozen=True) class MaveVoiceCommandProcessors` with `audio_gate` and `transcript_adapter`
- `build_mave_voice_command_processors(...) -> MaveVoiceCommandProcessors` that wires the transcript Adapter to reset the audio gate internally
- Constructor policy parameters: `pre_buffer_s=1.5`, `rearm_delay_s=0.75`, `max_awake_s=8.0`, `single_command=True`, `candidate_log_threshold=0.3`, `time_fn=time.monotonic`
- Behavior copied from current `wake_gate.py` and `transcript_cleanup.py`, but reset callback wiring is private to this Module
- Preserve `InputAudioRawFrame` mono conversion, timeout behavior, and rearm delay behavior
- Emit `WakeDetectedFrame` before replayed audio so Voice Metrics can consume a semantic event instead of checking processor class names
- If `single_command` is true, a finalized transcript rearms the audio gate even when the cleaned transcript is empty

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_wake_command.py tests/test_wake_gate.py tests/test_transcript_cleanup.py -v
uv run ruff check voice_runtime/wake_command.py tests/test_voice_runtime_wake_command.py
uv run pyright voice_runtime/wake_command.py tests/test_voice_runtime_wake_command.py
```

Expected: PASS, ruff pass, pyright 0 errors.

- [ ] **Step 5: Commit Issue 3**

```bash
cd pipecat-agent
git add server/voice_runtime/wake_command.py server/tests/test_voice_runtime_wake_command.py
git commit -m "feat: add reusable voice command wake module"
```

---

# Issue 4: Agent Turn Module

**Parallelization:** Can run after Issue 1. Owns only `agent_turn.py` and its tests.

**Files:**
- Create: `server/voice_runtime/agent_turn.py`
- Create: `server/tests/test_voice_runtime_agent_turn.py`

**Purpose:** Make Agent Turn behavior reusable across Claude and Codex Adapters: extract latest user text, run one backend turn, wrap output in Pipecat LLM start/text/end frames, and handle lifecycle explicitly.

**Interface target:** `AgentBackend` Protocol plus `AgentTurnProcessor`. Backends own external SDK details; the Agent Turn Module owns Pipecat frame semantics.

- [ ] **Step 1: Write failing Agent Turn tests**

Create `server/tests/test_voice_runtime_agent_turn.py`:

```python
import pytest
from pipecat.frames.frames import (
    CancelFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.agent_turn import AgentBackend, AgentTurnInput, AgentTurnProcessor, latest_user_text


class EchoBackend:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.turns = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    async def run_turn(self, turn: AgentTurnInput):
        self.turns.append(turn.user_text)
        yield f"echo: {turn.user_text}"


class CapturingProcessor(AgentTurnProcessor):
    def __init__(self, backend: AgentBackend):
        super().__init__(backend=backend)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


def _context_frame(messages) -> LLMContextFrame:
    return LLMContextFrame(context=LLMContext(messages=messages))


def test_latest_user_text_reads_string_and_text_parts():
    assert latest_user_text(_context_frame([{"role": "user", "content": "move up"}])) == "move up"
    assert latest_user_text(
        _context_frame([{"role": "user", "content": [{"type": "text", "text": "status"}]}])
    ) == "status"


@pytest.mark.asyncio
async def test_agent_turn_wraps_backend_output_in_llm_frames():
    backend = EchoBackend()
    processor = CapturingProcessor(backend)

    await processor.process_frame(_context_frame([{"role": "user", "content": "move up"}]), FrameDirection.DOWNSTREAM)

    assert [type(frame) for frame in processor.pushed] == [
        LLMFullResponseStartFrame,
        LLMTextFrame,
        LLMFullResponseEndFrame,
    ]
    assert processor.pushed[1].text == "echo: move up"
    assert backend.turns == ["move up"]


@pytest.mark.asyncio
async def test_agent_turn_disconnects_backend_on_cancel():
    backend = EchoBackend()
    processor = CapturingProcessor(backend)

    await processor.connect()
    await processor.process_frame(CancelFrame(), FrameDirection.DOWNSTREAM)

    assert backend.connected is True
    assert backend.disconnected is True
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_agent_turn.py -v
```

Expected: FAIL with missing `voice_runtime.agent_turn` symbols.

- [ ] **Step 3: Implement the Module**

Create `server/voice_runtime/agent_turn.py` with:

- `@dataclass(frozen=True) class AgentTurnInput` with `user_text: str` and `messages: list[Mapping[str, Any]]`
- `class AgentBackend(Protocol)` defining `connect`, `disconnect`, and `def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]` (declare `def`, not `async def`, so pyright treats async generators correctly)
- `latest_user_text(frame: LLMContextFrame) -> str | None`
- `agent_turn_input(frame: LLMContextFrame) -> AgentTurnInput | None`
- `class AgentTurnProcessor(FrameProcessor)` implementing explicit `connect`/`disconnect` and `process_frame`
- On a user turn: build one `AgentTurnInput`, push `LLMFullResponseStartFrame`, each backend text chunk as `LLMTextFrame`, then `LLMFullResponseEndFrame`
- If backend yields no text: emit `"I completed the action but have nothing to report."`
- If backend raises: log and emit `"I encountered an error. Please try again."`
- On `CancelFrame` or `EndFrame`: disconnect backend, then forward the frame

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_agent_turn.py tests/test_agent_processor_factory.py tests/test_openai_codex_agent_processor.py -v
uv run ruff check voice_runtime/agent_turn.py tests/test_voice_runtime_agent_turn.py
uv run pyright voice_runtime/agent_turn.py tests/test_voice_runtime_agent_turn.py
```

Expected: new tests pass. Existing processor tests should still pass because Issue 4 has not changed existing processors.

- [ ] **Step 5: Commit Issue 4**

```bash
cd pipecat-agent
git add server/voice_runtime/agent_turn.py server/tests/test_voice_runtime_agent_turn.py
git commit -m "feat: add reusable agent turn module"
```

---

# Issue 5: Robot Safety Module

**Parallelization:** Can run after Issue 1. Owns only `robot_safety.py` and its tests.

**Files:**
- Create: `server/voice_runtime/robot_safety.py`
- Create: `server/tests/test_voice_runtime_robot_safety.py`

**Purpose:** Make Robot Safety a reusable pure Module that Robot Tool Adapters can use before calling robot tools. It owns allowed tool names, canonical robot name, canonical-to-legacy tool name policy, argument validation, and execution-result interpretation. In this plan, the Codex Robot Tool Adapter is locally safety-enforced; direct Claude MCP remains prompt-only and must be documented as not crossing this safety seam.

**Interface target:** pure functions and dataclasses. This Module must not import MCP, Pipecat, OpenAI, Claude SDKs, or app-specific modules.

- [ ] **Step 1: Write failing Robot Safety tests**

Create `server/tests/test_voice_runtime_robot_safety.py`:

```python
import json

import pytest

from voice_runtime.robot_safety import (
    RobotSafetyError,
    canonical_mcp_tool_name,
    executable_plan_name,
    execution_result_text,
    validate_robot_tool_call,
)

VALID_POSE = {
    "position": {"x": 0.57, "y": 0.39, "z": 0.62},
    "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
}


def test_accepts_safe_free_motion_arguments():
    validate_robot_tool_call(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "position": VALID_POSE, "timeout_s": 25.0},
    )


def test_rejects_unknown_tool():
    with pytest.raises(RobotSafetyError, match="Tool is not allowed"):
        validate_robot_tool_call("move_to_position", {"robot_name": "UR10"})


def test_rejects_non_ur10_robot_name():
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call("moveit_open_gripper", {"robot_name": "UR5"})

    assert str(exc.value) == "Only Vizor robot UR10 is allowed"
    assert exc.value.correction == 'Retry with robot_name="UR10".'


def test_rejects_workspace_escape():
    unsafe_pose = {
        "position": {"x": 99.0, "y": 0.0, "z": 0.0},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call("moveit_plan_free_motion", {"robot_name": "UR10", "position": unsafe_pose})

    assert str(exc.value) == "Target is outside simulation workspace"
    assert "within +/-1.5 m" in exc.value.correction


def test_maps_canonical_agent_tool_to_legacy_mcp_tool_name():
    assert canonical_mcp_tool_name("moveit_plan_free_motion") == "plan_free_motion"
    assert canonical_mcp_tool_name("moveit_open_gripper") == "open_gripper"


def test_extracts_executable_plan_name_from_structured_tool_output():
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "plan-1"},
            }
        }
    )

    assert executable_plan_name(output) == "plan-1"


def test_execution_result_text_requires_passed_verification():
    success = json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
    failure = json.dumps({"structured_content": {"ok": True, "verification": {"result": "fail"}}})

    assert execution_result_text(success) == "Motion completed."
    assert execution_result_text(failure) == "I planned the motion, but execution could not be verified."
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_robot_safety.py -v
```

Expected: FAIL with missing `voice_runtime.robot_safety` symbols.

- [ ] **Step 3: Implement the pure safety Module**

Create `server/voice_runtime/robot_safety.py` with:

- `VIZOR_ROBOT_NAME = "UR10"`
- `WORKSPACE_ABS_LIMIT_M = 1.5`
- `DEFAULT_TIMEOUT_MAX_S = 60.0`
- `ALLOWED_ROBOT_TOOLS`
- `AGENT_TO_LEGACY_MCP_TOOL_NAMES`
- `class RobotSafetyError(ValueError)` with `.correction`
- `canonical_mcp_tool_name(agent_tool_name: str) -> str`
- `validate_robot_tool_call(name: str, arguments: dict[str, Any]) -> None`
- `executable_plan_name(output: str) -> str | None`
- `execution_result_text(output: str) -> str`

Move validation behavior and canonical/legacy tool-name policy from `server/robot_mcp_bridge.py` into this Module without changing current bridge behavior yet. Keep MCP type conversion, network calls, and function-tool shaping in the Robot Tool Adapter (`robot_mcp_bridge.py`).

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_robot_safety.py tests/test_robot_mcp_bridge.py -v
uv run ruff check voice_runtime/robot_safety.py tests/test_voice_runtime_robot_safety.py
uv run pyright voice_runtime/robot_safety.py tests/test_voice_runtime_robot_safety.py
```

Expected: new tests pass. Existing bridge tests still pass because Issue 5 has not changed the bridge.

- [ ] **Step 5: Commit Issue 5**

```bash
cd pipecat-agent
git add server/voice_runtime/robot_safety.py server/tests/test_voice_runtime_robot_safety.py
git commit -m "feat: add reusable robot safety module"
```

---

# Issue 6: Voice Metrics Module

**Parallelization:** Can run after Issue 1. Owns only `voice_metrics.py` and its tests.

**Files:**
- Create: `server/voice_runtime/voice_metrics.py`
- Create: `server/tests/test_voice_runtime_voice_metrics.py`

**Purpose:** Separate Voice Metrics stage semantics from Pipecat frame observation and JSONL persistence. The deep Module should be a semantic turn timeline; Pipecat observation becomes an Adapter in Issue 7.

**Interface target:** a pure `VoiceTurnTimeline` with typed lifecycle methods that records semantic Voice Runtime events and produces the current JSONL record shape.

- [ ] **Step 1: Write failing metrics tests**

Create `server/tests/test_voice_runtime_voice_metrics.py`:

```python
from voice_runtime.voice_metrics import VoiceTurnTimeline


def test_timeline_computes_stage_durations_without_pipecat_frames():
    timeline = VoiceTurnTimeline(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        turn_id="turn-1",
        started_at=10.0,
        now_fn=lambda: 10.0,
        wall_time_fn=lambda: 100.0,
    )
    timeline.wake_detected("mave", at=10.1)
    timeline.speech_captured(at=10.5)
    timeline.stt_done("move up", at=10.8)
    timeline.agent_done(at=11.8)
    timeline.tts_audio_started(at=12.0)
    timeline.tts_done(at=12.5)
    timeline.append_agent_text("Motion ")
    timeline.append_agent_text("completed.")

    record = timeline.to_record(finished_at=12.5, include_text=True)

    assert record["timestamp_unix"] == 100.0
    assert record["profile"] == "hybrid_low_latency"
    assert record["category"] == "benchmark_streaming"
    assert record["turn_id"] == "turn-1"
    assert record["wake_phrase"] == "mave"
    assert record["wake_latency_ms"] == 100.0
    assert record["speech_captured_ms"] == 400.0
    assert record["stt_latency_ms"] == 300.0
    assert record["agent_latency_ms"] == 1000.0
    assert record["tts_first_audio_ms"] == 200.0
    assert record["tts_done_ms"] == 500.0
    assert record["total_to_first_audio_ms"] == 2000.0
    assert record["total_turn_ms"] == 2500.0
    assert record["transcript"] == "move up"
    assert record["response"] == "Motion completed."


def test_timeline_without_wake_starts_speech_duration_at_turn_start():
    timeline = VoiceTurnTimeline(
        profile="no_wake_debug",
        category="local_debug",
        turn_id="turn-1",
        started_at=20.0,
        now_fn=lambda: 20.0,
        wall_time_fn=lambda: 200.0,
    )
    timeline.speech_captured(at=20.4)

    record = timeline.to_record(finished_at=21.0, include_text=False)

    assert record["wake_latency_ms"] is None
    assert record["speech_captured_ms"] == 400.0
    assert "transcript" not in record
    assert "response" not in record


def test_timeline_deduplicates_first_tts_audio_mark():
    timeline = VoiceTurnTimeline(
        profile="hybrid_low_latency",
        category="benchmark_streaming",
        turn_id="turn-1",
        started_at=1.0,
        now_fn=lambda: 1.0,
        wall_time_fn=lambda: 10.0,
    )
    timeline.agent_done(at=2.0)
    timeline.tts_audio_started(at=2.2)
    timeline.tts_audio_started(at=2.4)

    record = timeline.to_record(finished_at=3.0, include_text=False)

    assert record["tts_first_audio_ms"] == 200.0
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_voice_metrics.py -v
```

Expected: FAIL with missing `voice_runtime.voice_metrics` symbols.

- [ ] **Step 3: Implement the pure timeline Module**

Create `server/voice_runtime/voice_metrics.py` with:

- `class VoiceTurnTimeline`
- constructor fields: `profile`, `category`, `turn_id`, `started_at`, `now_fn`, `wall_time_fn`
- `wake_detected(wake_phrase: str, at: float | None = None) -> None`
- `speech_captured(at: float | None = None) -> None`
- `stt_done(transcript: str, at: float | None = None) -> None`
- `agent_done(at: float | None = None) -> None`
- `append_agent_text(text: str) -> None`
- `tts_audio_started(at: float | None = None) -> None` that preserves the first audio mark
- `tts_done(at: float | None = None) -> None`
- `to_record(finished_at: float | None = None, include_text: bool = True) -> dict[str, Any]`
- Internal mark names matching current JSONL output: `wake_detected`, `speech_captured`, `stt_done`, `agent_done`, `tts_first_audio`, `tts_done`
- Duration semantics and record fields matching `server/metrics.py`

- [ ] **Step 4: Run targeted tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_voice_metrics.py tests/test_metrics.py -v
uv run ruff check voice_runtime/voice_metrics.py tests/test_voice_runtime_voice_metrics.py
uv run pyright voice_runtime/voice_metrics.py tests/test_voice_runtime_voice_metrics.py
```

Expected: new tests pass. Existing metrics tests still pass because Issue 6 has not changed `server/metrics.py`.

- [ ] **Step 5: Commit Issue 6**

```bash
cd pipecat-agent
git add server/voice_runtime/voice_metrics.py server/tests/test_voice_runtime_voice_metrics.py
git commit -m "feat: add reusable voice metrics timeline"
```

---

# Issue 7: Integrate Modules and add the Voice Runtime assembly Interface

**Parallelization:** Serial integration after Issues 2-6 are merged.

**Files:**
- Create: `server/voice_runtime/assembly.py`
- Create: `server/tests/test_voice_runtime_assembly.py`
- Create: `server/tests/test_orthogonal_imports.py`
- Modify: `server/config.py`
- Modify: `server/providers.py`
- Modify: `server/pipeline_builder.py`
- Modify: `server/bot.py`
- Modify: `server/wake/wake_gate.py`
- Modify: `server/wake/transcript_cleanup.py`
- Modify: `server/agent_processor_factory.py`
- Modify: `server/claude_agent_processor.py`
- Modify: `server/openai_codex_agent_processor.py`
- Modify: `server/robot_mcp_bridge.py`
- Modify: `server/metrics.py`
- Modify tests as needed where old implementation-level assertions duplicate new Interface tests

**Purpose:** Switch the app to the reusable Modules while keeping public behavior unchanged except where this plan explicitly calls out an existing scaffold/limitation. Add a reusable Voice Runtime assembly Module so pipeline ordering is not trapped in `pipeline_builder.py`.

- [ ] **Step 1: Run the full baseline before integration**

Run:

```bash
cd pipecat-agent
git status --short
cd server
uv run pytest -v
uv run ruff check .
uv run pyright .
```

Expected: current suite passes before edits. `git status --short` must be clean except for already-merged plan/docs files; if not, stop and resolve the branch/worktree state before editing integration files.

- [ ] **Step 2: Make `config.py` and provider construction use the Runtime Profile Module**

Required behavior:

- `load_runtime_config(...)` still returns the existing `RuntimeConfig` dataclass or a compatible alias.
- Missing env validation stays in `config.py`, using `RuntimeProfile.required_env_names()`.
- TOML parsing and category/provider validation move to `voice_runtime.profiles`.
- Cartesia voice env behavior is preserved: missing profile `tts.voice` requires `CARTESIA_VOICE_ID`.
- `providers.py` remains an Adapter construction Module; it should not duplicate provider/category policy already owned by Runtime Profile.
- Existing `tests/test_config.py`, `tests/test_providers.py`, and new `tests/test_voice_runtime_profiles.py` pass.

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_config.py tests/test_providers.py tests/test_voice_runtime_profiles.py -v
```

Expected: PASS.

- [ ] **Step 3: Add the reusable Voice Runtime assembly Module**

Create `server/tests/test_voice_runtime_assembly.py`:

```python
from voice_runtime.assembly import VoiceRuntimeParts, ordered_voice_runtime_processors


def test_orders_voice_runtime_processors_with_wake_adapters():
    parts = VoiceRuntimeParts(
        transport_input="transport.input",
        voice_command_audio="wake.audio",
        stt="stt",
        voice_command_transcript="wake.transcript",
        user_aggregator="user_aggregator",
        agent_turn="agent_turn",
        tts="tts",
        transport_output="transport.output",
        assistant_aggregator="assistant_aggregator",
    )

    assert ordered_voice_runtime_processors(parts) == [
        "transport.input",
        "wake.audio",
        "stt",
        "wake.transcript",
        "user_aggregator",
        "agent_turn",
        "tts",
        "transport.output",
        "assistant_aggregator",
    ]


def test_orders_voice_runtime_processors_without_wake_adapters():
    parts = VoiceRuntimeParts(
        transport_input="transport.input",
        voice_command_audio=None,
        stt="stt",
        voice_command_transcript=None,
        user_aggregator="user_aggregator",
        agent_turn="agent_turn",
        tts="tts",
        transport_output="transport.output",
        assistant_aggregator="assistant_aggregator",
    )

    assert ordered_voice_runtime_processors(parts) == [
        "transport.input",
        "stt",
        "user_aggregator",
        "agent_turn",
        "tts",
        "transport.output",
        "assistant_aggregator",
    ]
```

Create `server/voice_runtime/assembly.py` with:

- `@dataclass(frozen=True) class VoiceRuntimeParts`
- `ordered_voice_runtime_processors(parts: VoiceRuntimeParts) -> list[object]`
- No imports from Pipecat, `bot.py`, `pipeline_builder.py`, providers, or app modules
- Ordering policy: transport input, optional Voice Command audio Adapter, STT, optional Voice Command transcript Adapter, user aggregator, Agent Turn, TTS, transport output, assistant aggregator

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_assembly.py -v
uv run ruff check voice_runtime/assembly.py tests/test_voice_runtime_assembly.py
uv run pyright voice_runtime/assembly.py tests/test_voice_runtime_assembly.py
```

Expected: PASS.

- [ ] **Step 4: Replace app-level wake wiring with Voice Command processors**

Required behavior:

- `pipeline_builder.py` calls `build_mave_voice_command_processors(...)` when wake is enabled.
- It inserts `processors.audio_gate` before STT and `processors.transcript_adapter` after STT through `ordered_voice_runtime_processors(...)`.
- `pipeline_builder.py` passes `pre_buffer_s`, `single_command`, and `candidate_log_threshold` from the Runtime Profile.
- Existing `server/wake/wake_gate.py` and `server/wake/transcript_cleanup.py` remain as compatibility imports or thin wrappers so existing imports do not break.
- `VoiceMetricsObserver` can observe `WakeDetectedFrame`; do not add class-name checks for `MaveVoiceCommandGate` or `MaveVoiceCommandAudioGate`.
- Tests should assert the two Adapter topology: one Voice Command audio Adapter before STT and one transcript Adapter after STT.

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_wake_command.py tests/test_wake_gate.py tests/test_transcript_cleanup.py tests/test_pipeline_builder.py tests/test_voice_runtime_assembly.py -v
```

Expected: PASS after updating tests that intentionally assert old callback wiring.

- [ ] **Step 5: Move shared Agent Turn frame behavior to `AgentTurnProcessor`**

Required behavior:

- Claude and Codex external SDK logic becomes backend behavior behind `AgentBackend`.
- `agent_processor_factory.py` still returns a `FrameProcessor`, specifically an `AgentTurnProcessor` wrapping the chosen backend Adapter.
- `bot.py` lists `AgentTurnProcessor` lifecycle explicitly; remove `_call_optional_agent_method()` reflection if practical. If reflection remains, add a test or comment that every factory-returned processor implements `connect` and `disconnect`.
- Latest-user extraction and message-to-turn conversion have one implementation in `voice_runtime.agent_turn`.
- Existing user-facing fallback text remains unchanged.
- Codex-specific credential lookup, robot tool loop, and pose context remain in the Codex backend Adapter; the shared Agent Turn Module should not import OpenAI, Claude, MCP, or robot safety Modules.

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_agent_turn.py tests/test_agent_processor_factory.py tests/test_openai_codex_agent_processor.py -v
```

Expected: PASS.

- [ ] **Step 6: Make Robot Tool Adapters use `voice_runtime.robot_safety` honestly**

Required behavior:

- `RobotMCPBridge.call_tool()` delegates validation to `validate_robot_tool_call()` and canonical name mapping to `canonical_mcp_tool_name()`.
- `RobotMCPBridge.call_tool()` catches `RobotSafetyError` and serializes `str(exc)` plus `exc.correction` so validation failure JSON remains byte-for-byte compatible with existing tests.
- `OpenAICodexAgentProcessor` uses `executable_plan_name()` and `execution_result_text()` from `voice_runtime.robot_safety`.
- Do not claim direct Claude MCP is locally safety-enforced. Preserve Claude debug profiles as prompt-only Robot Safety coverage unless this issue also implements a safe MCP proxy Adapter. Issue 8 must document the limitation.

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_robot_safety.py tests/test_robot_mcp_bridge.py tests/test_openai_codex_agent_processor.py -v
```

Expected: PASS.

- [ ] **Step 7: Make `metrics.py` use `VoiceTurnTimeline` and semantic wake events**

Required behavior:

- `VoiceMetricsRecorder` owns persistence and active-turn storage.
- `VoiceTurnTimeline` owns semantic stage transitions, de-duplication of first TTS audio, duration semantics, and JSONL record shape.
- `VoiceMetricsObserver` remains the Pipecat-frame Adapter.
- `VoiceMetricsObserver` marks wake detection from `WakeDetectedFrame`, not from source class names.
- JSONL record fields and duration semantics stay unchanged.

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_voice_runtime_voice_metrics.py tests/test_metrics.py tests/test_pipeline_builder.py tests/test_voice_runtime_wake_command.py -v
```

Expected: PASS.

- [ ] **Step 8: Add AST-based orthogonal import guard test**

Create `server/tests/test_orthogonal_imports.py`:

```python
import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
VOICE_RUNTIME_DIR = SERVER_DIR / "voice_runtime"

APP_MODULE_ROOTS = {
    "agent_processor_factory",
    "bot",
    "claude_agent_processor",
    "codex_auth",
    "codex_backend_client",
    "config",
    "metrics",
    "openai_codex_agent_processor",
    "pipeline_builder",
    "prompts",
    "providers",
    "robot_mcp_bridge",
    "wake",
}

PURE_MODULES = {
    "contracts.py",
    "profiles.py",
    "robot_safety.py",
    "voice_metrics.py",
    "assembly.py",
}
PURE_MODULE_FORBIDDEN_ROOTS = {
    "agents",
    "claude_agent_sdk",
    "dotenv",
    "mcp",
    "openai",
    "pipecat",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_voice_runtime_modules_do_not_import_app_modules():
    for path in VOICE_RUNTIME_DIR.glob("*.py"):
        imported = _import_roots(path)
        forbidden = imported & APP_MODULE_ROOTS
        assert not forbidden, f"{path.name} imports app-specific module(s): {sorted(forbidden)}"


def test_pure_voice_runtime_modules_do_not_import_runtime_adapters():
    for name in PURE_MODULES:
        path = VOICE_RUNTIME_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PURE_MODULE_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports adapter-specific module(s): {sorted(forbidden)}"
```

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_orthogonal_imports.py -v
```

Expected: PASS. Provider string literals such as `openai_codex_oauth` must not fail this guard; only real imports should fail.

- [ ] **Step 9: Run full integration validation**

Run:

```bash
cd pipecat-agent/server
uv run pytest -v
uv run ruff check .
uv run pyright .
```

Expected: all tests pass, ruff pass, pyright 0 errors.

- [ ] **Step 10: Commit Issue 7**

```bash
cd pipecat-agent
git add server
git commit -m "refactor: integrate orthogonal voice runtime modules"
```

---

# Issue 8: Documentation and extractability review

**Parallelization:** Serial after Issue 7.

**Files:**
- Modify: `README.md`
- Modify: `docs/benchmarking.md`
- Modify: `CONTEXT.md`
- Create: `docs/architecture.md`

**Purpose:** Document the new Module contracts and verify the reusable package can be understood without reading `bot.py` first.

- [ ] **Step 1: Create architecture docs**

Create `docs/architecture.md` with these sections:

```markdown
# Voice Runtime Architecture

## Orthogonality goal

The Voice Runtime is split into reusable Modules. Each Module has a small Interface and hides app-specific implementation details behind seams and Adapters.

## Modules

### Runtime Profile

Owns Runtime Profile parsing and provider policy. It does not construct Pipecat processors.

### Voice Command

Owns wake phrase detection, pre-buffer replay, wake phrase stripping, finalized command emission, semantic wake events, and rearming. It exposes two Pipecat Adapters because audio gating happens before STT and transcript cleanup happens after STT.

### Agent Turn

Owns Pipecat LLM frame semantics for one Agent Turn. Claude and Codex are Adapters behind this seam.

### Robot Safety

Owns allowed robot tools, UR10-only validation, workspace limits, canonical tool-name policy, plan-before-execute helpers, and execution result interpretation. This safety is locally enforced only for Robot Tool Adapters that call the Robot Safety Module before tool execution. Current Codex uses that seam; direct Claude MCP is prompt-only until a safe MCP proxy Adapter exists.

### Voice Metrics

Owns per-turn semantic stage transitions and timing semantics. Pipecat frame observation and JSONL persistence are Adapters around the timeline.

### Voice Runtime Assembly

Owns reusable Pipecat processor ordering: transport input, optional Voice Command audio Adapter, STT, optional Voice Command transcript Adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.

## App integration

`bot.py` remains the app entrypoint. `pipeline_builder.py` constructs concrete Adapters and delegates ordering to the Voice Runtime Assembly Module.

## Reuse checklist

To reuse these Modules in a similar project:

1. Provide Runtime Profiles for the target providers.
2. Provide Robot Tool Adapters for the target MCP or tool layer.
3. Choose an Agent Turn backend Adapter.
4. Build a Pipecat pipeline with Voice Runtime Assembly using the Voice Command, STT, Agent Turn, TTS, and Voice Metrics Adapters.
5. Document Robot Safety coverage per Agent Turn backend. Do not imply direct MCP backends are locally safety-enforced.
6. Treat emergency stop as scaffold-only unless a runtime bypass Adapter is implemented.
```

- [ ] **Step 2: Update README and benchmarking docs**

Required changes:

- Link `docs/architecture.md` from `README.md`.
- In `docs/benchmarking.md`, explain that Voice Metrics duration semantics live in the Voice Metrics Module.
- In `docs/benchmarking.md`, clarify that the `Mave, stop.` utterance is a normal Voice Command test, not an emergency-stop bypass.
- In `README.md`, keep runtime profile commands unchanged.
- In `README.md` and `docs/architecture.md`, state Robot Safety coverage per Agent Turn backend: Codex through `RobotMCPBridge` is locally enforced; direct Claude MCP is prompt-only unless a safe MCP proxy Adapter is added.
- In `docs/architecture.md`, state emergency stop is currently a Runtime Profile scaffold unless Issue 7 adds runtime bypass behavior.

- [ ] **Step 3: Update `CONTEXT.md` if names changed during implementation**

Update terms only if Issue 7 used different names. Keep definitions short and domain-focused.

- [ ] **Step 4: Run docs/import validation**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_orthogonal_imports.py -v
uv run pytest -v
uv run ruff check .
uv run pyright .
```

Expected: all checks pass.

- [ ] **Step 5: Commit Issue 8**

```bash
cd pipecat-agent
git add README.md docs/benchmarking.md docs/architecture.md CONTEXT.md
git commit -m "docs: document orthogonal voice runtime modules"
```

---

## Final verification checklist

Run after Issue 8:

```bash
cd pipecat-agent/server
uv run pytest -v
uv run ruff check .
uv run pyright .
uv run python - <<'PY'
from voice_runtime.profiles import load_runtime_profile
profile = load_runtime_profile(profile_name="no_wake_debug")
print(profile.name, profile.category, profile.required_env_names())
PY
```

Expected:

- `pytest` passes.
- `ruff` passes.
- `pyright` reports 0 errors.
- The profile check prints `no_wake_debug local_debug ()`.

## Parallel-agent launch guide

After Issue 1 lands, dispatch these agents concurrently:

1. Runtime Profile agent: Issue 2 only.
2. Voice Command agent: Issue 3 only.
3. Agent Turn agent: Issue 4 only.
4. Robot Safety agent: Issue 5 only.
5. Voice Metrics agent: Issue 6 only.

Each agent should work in an isolated worktree and return:

- Files changed.
- Tests added.
- Validation commands run and outputs.
- Any Interface decisions that differ from this plan.
- Any reason an expected seam was not deepened.

Do not run Issue 7 until the parallel work is merged and all targeted tests pass.
