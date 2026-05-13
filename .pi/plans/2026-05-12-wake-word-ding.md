# Wake Word Ding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Play a short non-blocking ding when the wake word opens the audio gate.

**Architecture:** Add a Voice Runtime processor immediately before transport output, after TTS and optional voice modulation. The processor forwards every frame unchanged, and when it sees `WakeDetectedFrame`, it also pushes a short generated PCM `OutputAudioRawFrame` directly downstream so only the output side plays it. No profile setting, client patch, or wake-gate state change is added.

**Tech Stack:** Python, Pipecat frame processors, pytest, numpy.

---

### Task 1: Wake Ding Processor

**Files:**
- Create: `server/voice_runtime/wake_tone.py`
- Modify: `server/tests/test_voice_runtime_wake_tone.py`

- [x] **Step 1: Write the failing test**

Add `server/tests/test_voice_runtime_wake_tone.py` with tests that:
- `WakeToneProcessor` forwards `WakeDetectedFrame`.
- It emits one `OutputAudioRawFrame` after the wake event.
- It forwards ordinary frames without tone output.
- The generated tone is short, mono, PCM16, and uses the requested sample rate.

- [x] **Step 2: Run the wake tone test to verify it fails**

Run: `uv run pytest tests/test_voice_runtime_wake_tone.py -q`

Expected: import failure because `voice_runtime.wake_tone` does not exist yet.

- [x] **Step 3: Implement the processor**

Create `server/voice_runtime/wake_tone.py` with:
- `WakeToneProcessor(FrameProcessor)`.
- A small `_build_ding_pcm16()` helper using `math.sin`.
- `process_frame()` that calls `super()`, forwards the original frame, then pushes an `OutputAudioRawFrame` only for `WakeDetectedFrame`.

- [x] **Step 4: Run the wake tone test to verify it passes**

Run: `uv run pytest tests/test_voice_runtime_wake_tone.py -q`

Expected: all tests pass.

### Task 2: Pipeline Wiring

**Files:**
- Modify: `server/voice_runtime/assembly.py`
- Modify: `server/pipeline_builder.py`
- Modify: `server/tests/test_voice_runtime_assembly.py`
- Modify: `server/tests/test_pipeline_builder.py`

- [x] **Step 1: Write failing wiring tests**

Update assembly and pipeline-builder tests to expect:
- `wake_tone` appears after TTS/optional voice modulation and before `transport.output` when wake is enabled.
- `wake_tone` is absent when wake is disabled.
- `build_pipeline()` creates `WakeToneProcessor` only for enabled wake profiles.

- [x] **Step 2: Run the wiring tests to verify they fail**

Run: `uv run pytest tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py::test_wake_enabled_uses_two_voice_command_adapters_around_stt -q`

Expected: failures because `VoiceRuntimeParts` has no `wake_tone` field and the pipeline does not create the processor.

- [x] **Step 3: Wire the processor**

Update:
- `VoiceRuntimeParts` with `wake_tone: object | None`.
- `ordered_voice_runtime_processors()` to append `wake_tone` immediately before `transport_output`.
- `pipeline_builder.build_pipeline()` to create `WakeToneProcessor()` when wake is enabled and pass it into `VoiceRuntimeParts`.

- [x] **Step 4: Run targeted tests**

Run: `uv run pytest tests/test_voice_runtime_wake_tone.py tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py::test_wake_enabled_uses_two_voice_command_adapters_around_stt tests/test_pipeline_builder.py::test_metrics_observer_is_not_wired_when_metrics_disabled -q`

Expected: all targeted tests pass.

### Task 3: Final Verification

**Files:**
- No new edits unless verification exposes a scoped failure.

- [x] **Step 1: Run lint on touched files**

Run: `uv run ruff check voice_runtime/wake_tone.py tests/test_voice_runtime_wake_tone.py voice_runtime/assembly.py pipeline_builder.py tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py`

Expected: pass.

- [x] **Step 2: Run relevant Voice Runtime tests**

Run: `uv run pytest tests/test_voice_runtime_wake_tone.py tests/test_voice_runtime_wake_command.py tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py -q`

Expected: all selected tests pass.
