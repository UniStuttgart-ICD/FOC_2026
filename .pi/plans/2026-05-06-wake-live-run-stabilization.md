# Wake Live Run Stabilization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the wake gate from opening on weak false positives, strip live-run wake-word STT variants, and keep the gate suppressed while an agent turn is still in flight.

**Architecture:** Keep wake behavior in `server/voice_runtime/`. Extend `WakeProfile` as the config contract, pass values through `pipeline_builder.py`, and coordinate active-turn suppression with small callbacks on `AgentTurnProcessor`.

**Tech Stack:** Python 3, pytest, Pipecat frame processors, loguru/runtime profile TOML.

---

## Evidence From The Run

- True-ish first wake: `score=0.996 threshold=0.700 rms=67.2 peak=174`.
- Weak false wakes: `rms=25.5 peak=75`, `rms=22.5 peak=64`.
- STT wake residue reached Codex as commands: `may move robot up.`, `Names,`, `Mail up the robot wave.`
- The gate rearmed while Codex/TTS was still active, causing detections during the same interaction.

## Rules For This Plan

- Follow TDD. Write or update failing tests before each implementation step.
- Keep defaults backwards compatible in constructors; enable stricter behavior from `runtime_profiles.toml`.
- Do not touch the existing untracked plan files unless the user asks.
- Commit after each green step with a focused message.

## Setup

- [ ] Start from `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent` on `master`.

```powershell
git status --short --branch
```

Expected: branch is `master`. Untracked `.pi/plans/*.md` files may exist.

- [ ] Create an implementation branch.

```powershell
git switch -c wake-live-run-stabilization
```

Expected: new branch checked out.

---

## Step 1: Expose Wake Audio Guards In Profiles

### Red

- [ ] Add failing tests in `server/tests/test_voice_runtime_profiles.py`.

Add coverage that a wake profile can parse:

```python
def test_wake_profile_parses_audio_guards_and_rearm_delay() -> None:
    profile = load_profile_from_text(
        """
        [wake]
        enabled = true
        detector = "openwakeword"
        threshold = 0.7
        vad_threshold = 0.0
        candidate_log_threshold = 0.5
        required_hits = 1
        min_wake_rms = 50.0
        min_wake_peak = 150
        rearm_delay_s = 6.0
        """
    )

    assert profile.wake.min_wake_rms == 50.0
    assert profile.wake.min_wake_peak == 150
    assert profile.wake.rearm_delay_s == 6.0
```

Add invalid-value coverage for negative numbers and bools:

```python
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("min_wake_rms", "-1"),
        ("min_wake_peak", "-1"),
        ("rearm_delay_s", "-0.1"),
        ("min_wake_rms", "true"),
        ("min_wake_peak", "false"),
        ("rearm_delay_s", "true"),
    ],
)
def test_wake_profile_rejects_invalid_audio_guard_values(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=field):
        load_profile_from_text(
            f"""
            [wake]
            enabled = true
            detector = "openwakeword"
            {field} = {value}
            """
        )
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_profiles.py -q
```

Expected: fails because `WakeProfile` lacks these fields.

### Green

- [ ] Update `server/voice_runtime/profiles.py`.

Add fields:

```python
min_wake_rms: float = 0.0
min_wake_peak: int = 0
rearm_delay_s: float = 0.75
```

Parse with the same helper style as `threshold`, `vad_threshold`, and `required_hits`. Reject bools and negative values.

- [ ] Update `server/runtime_profiles.toml`.

For each bundled openwakeword wake profile, add:

```toml
min_wake_rms = 50.0
min_wake_peak = 150
rearm_delay_s = 6.0
```

These values reject the observed weak false wakes while keeping the first observed activation above the guard.

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_profiles.py -q
```

Expected: profile tests pass.

- [ ] Commit.

```powershell
git add server/voice_runtime/profiles.py server/runtime_profiles.toml server/tests/test_voice_runtime_profiles.py
git commit -m "test: expose wake audio guard profile settings"
```

---

## Step 2: Reject Weak-Audio Wake Detections

### Red

- [ ] Add failing tests in `server/tests/test_voice_runtime_wake_command.py`.

Add a low-audio detected frame test:

```python
async def test_audio_gate_rejects_detected_wake_below_audio_guard() -> None:
    detector = ScriptedDetector([(True, "mave", 0.996)])
    gate = MaveVoiceCommandAudioGate(
        detector,
        min_wake_rms=50.0,
        min_wake_peak=150,
        wake_threshold=0.7,
        time_fn=FakeClock().time,
    )

    frame = audio_frame([25, -25, 20, -20])
    await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert gate.is_awake is False
    assert [type(frame).__name__ for frame in gate.pushed_frames] == []
```

Add a sufficient-audio control:

```python
async def test_audio_gate_accepts_detected_wake_above_audio_guard() -> None:
    detector = ScriptedDetector([(True, "mave", 0.996)])
    gate = MaveVoiceCommandAudioGate(
        detector,
        min_wake_rms=50.0,
        min_wake_peak=150,
        wake_threshold=0.7,
        time_fn=FakeClock().time,
    )

    frame = audio_frame([200, -200, 180, -180])
    await gate.process_frame(frame, FrameDirection.DOWNSTREAM)

    assert gate.is_awake is True
    assert any(isinstance(frame, WakeDetectedFrame) for frame in gate.pushed_frames)
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_wake_command.py -q
```

Expected: constructor rejects unknown args or weak detection opens the gate.

### Green

- [ ] Update `server/voice_runtime/wake_command.py`.

Add constructor args:

```python
min_wake_rms: float = 0.0
min_wake_peak: int = 0
```

Reject invalid values:

```python
if min_wake_rms < 0:
    raise ValueError("min_wake_rms must be non-negative")
if min_wake_peak < 0:
    raise ValueError("min_wake_peak must be non-negative")
```

Store:

```python
self._min_wake_rms = min_wake_rms
self._min_wake_peak = min_wake_peak
```

After `_audio_levels(pcm16)`, before incrementing `_consecutive_hits`, reject weak audio:

```python
if detected and (rms < self._min_wake_rms or peak < self._min_wake_peak):
    self._consecutive_hits = 0
    logger.debug(
        self._diagnostic_message(
            "Wake candidate rejected",
            model_name=model_name,
            score=score,
            rms=rms,
            peak=peak,
            gate_open=False,
        )
        + " reason=audio_level"
    )
    return
```

Pass the values through `build_mave_voice_command_processors()`.

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_wake_command.py -q
```

Expected: wake command tests pass.

- [ ] Commit.

```powershell
git add server/voice_runtime/wake_command.py server/tests/test_voice_runtime_wake_command.py
git commit -m "fix: reject weak-audio wake detections"
```

---

## Step 3: Wire Wake Guards Through Pipeline Builder

### Red

- [ ] Add failing tests in `server/tests/test_pipeline_builder.py`.

Extend the existing wake-processor construction test to assert:

```python
assert processors.audio_gate._min_wake_rms == 50.0
assert processors.audio_gate._min_wake_peak == 150
assert processors.audio_gate._rearm_delay_s == 6.0
```

Use a `WakeProfile` containing:

```python
min_wake_rms=50.0,
min_wake_peak=150,
rearm_delay_s=6.0,
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_pipeline_builder.py -q
```

Expected: fails because `pipeline_builder.py` does not pass the new fields.

### Green

- [ ] Update `server/pipeline_builder.py`.

When calling `build_mave_voice_command_processors`, pass:

```python
rearm_delay_s=profile.wake.rearm_delay_s,
min_wake_rms=profile.wake.min_wake_rms,
min_wake_peak=profile.wake.min_wake_peak,
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_pipeline_builder.py -q
```

Expected: pipeline builder tests pass.

- [ ] Commit.

```powershell
git add server/pipeline_builder.py server/tests/test_pipeline_builder.py
git commit -m "fix: wire wake guard settings into pipeline"
```

---

## Step 4: Strip Live-Run Wake Variants From STT

### Red

- [ ] Add failing tests in `server/tests/test_voice_runtime_wake_command.py`.

```python
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("May move robot up.", "move robot up."),
        ("Mail up the robot wave.", "up the robot wave."),
        ("Names,", ""),
        ("Name, move robot left.", "move robot left."),
        ("Nave stop.", "stop."),
    ],
)
def test_strip_mave_wake_phrase_handles_live_run_variants(raw: str, expected: str) -> None:
    assert strip_mave_wake_phrase(raw) == expected
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_wake_command.py -q
```

Expected: variants are not stripped.

### Green

- [ ] Update `_WAKE_PATTERN` in `server/voice_runtime/wake_command.py`.

Use this pattern:

```python
_WAKE_PATTERN = re.compile(
    r"^\s*(?:hey\s+)?(?:mae?ve|may|mail|nave|names?)(?:\b|[\s,;:!?.-])[\s,;:!?.-]*",
    re.IGNORECASE,
)
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_wake_command.py -q
```

Expected: wake command tests pass.

- [ ] Commit.

```powershell
git add server/voice_runtime/wake_command.py server/tests/test_voice_runtime_wake_command.py
git commit -m "fix: strip wake transcript variants"
```

---

## Step 5: Drop Probable Wake-Junk Agent Turns

### Red

- [ ] Add failing tests in `server/tests/test_voice_runtime_agent_turn.py`.

```python
@pytest.mark.parametrize(
    "text",
    [
        "Names,",
        "Name.",
        "Mail.",
        "Nave.",
        "up the robot wave.",
    ],
)
def test_is_actionable_user_text_rejects_live_run_wake_junk(text: str) -> None:
    assert is_actionable_user_text(text) is False
```

Keep the positive control:

```python
def test_is_actionable_user_text_keeps_command_after_wake_variant_cleanup() -> None:
    assert is_actionable_user_text("move robot up.") is True
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_agent_turn.py -q
```

Expected: wake junk still passes.

### Green

- [ ] Update `server/voice_runtime/agent_turn.py`.

Expand wake-only words:

```python
_WAKE_ONLY_TEXT = {"mave", "maeve", "may", "mail", "nave", "name", "names"}
```

Add a small exact-fragment block for the observed corrupted command:

```python
_PROBABLE_WAKE_JUNK_TEXT = {"up the robot wave"}
```

Normalize text in `is_actionable_user_text`:

```python
normalized = " ".join(words)
if normalized in _PROBABLE_WAKE_JUNK_TEXT:
    return False
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_voice_runtime_agent_turn.py -q
```

Expected: agent turn tests pass.

- [ ] Commit.

```powershell
git add server/voice_runtime/agent_turn.py server/tests/test_voice_runtime_agent_turn.py
git commit -m "fix: drop probable wake junk turns"
```

---

## Step 6: Suppress Wake During Active Agent Turns

### Red

- [ ] Add failing tests in `server/tests/test_voice_runtime_agent_turn.py`.

Use an existing fake backend pattern and assert lifecycle callbacks fire around a real actionable turn:

```python
async def test_agent_turn_processor_calls_lifecycle_callbacks() -> None:
    events: list[str] = []
    processor = AgentTurnProcessor(
        backend=FakeBackend(["done"]),
        on_turn_started=lambda: events.append("started"),
        on_turn_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(context_frame("move robot up."), FrameDirection.DOWNSTREAM)

    assert events == ["started", "finished"]
```

Add failure coverage:

```python
async def test_agent_turn_processor_finishes_lifecycle_after_backend_error() -> None:
    events: list[str] = []
    processor = AgentTurnProcessor(
        backend=FailingBackend(),
        on_turn_started=lambda: events.append("started"),
        on_turn_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(context_frame("move robot up."), FrameDirection.DOWNSTREAM)

    assert events == ["started", "finished"]
```

- [ ] Add failing tests in `server/tests/test_pipeline_builder.py`.

Assert the factory is called with callbacks when wake mode is enabled. If the current test monkeypatches `create_agent_processor`, capture kwargs and assert:

```python
assert callable(captured_kwargs["on_turn_started"])
assert callable(captured_kwargs["on_turn_finished"])
```

- [ ] Run red tests.

```powershell
cd server
uv run pytest tests/test_voice_runtime_agent_turn.py tests/test_pipeline_builder.py -q
```

Expected: callbacks are not accepted or not wired.

### Green

- [ ] Update `server/voice_runtime/agent_turn.py`.

Accept optional callbacks:

```python
def __init__(
    self,
    *,
    backend: AgentBackend,
    on_turn_started: Callable[[], None] | None = None,
    on_turn_finished: Callable[[], None] | None = None,
    **kwargs: Any,
) -> None:
```

Call them only for actionable turns:

```python
if self._on_turn_started is not None:
    self._on_turn_started()
try:
    await self.push_frame(LLMFullResponseStartFrame())
    await self._run_turn(turn)
    await self.push_frame(LLMFullResponseEndFrame())
finally:
    if self._on_turn_finished is not None:
        self._on_turn_finished()
```

- [ ] Update `server/agent_processor_factory.py`.

Change the signature:

```python
def create_agent_processor(
    config: AgentConfig,
    *,
    mcp_server_url: str,
    on_turn_started: Callable[[], None] | None = None,
    on_turn_finished: Callable[[], None] | None = None,
) -> FrameProcessor:
```

Pass callbacks into `AgentTurnProcessor`.

- [ ] Update `server/voice_runtime/wake_command.py`.

Add methods on `MaveVoiceCommandAudioGate`:

```python
def suppress(self) -> None:
    self._reset(now=self._time_fn())
    self._suppressed = True

def unsuppress(self) -> None:
    self._suppressed = False
    self._reset(now=self._time_fn())
```

Initialize `self._suppressed = False`. In `_process_audio_frame`, before appending to the ring:

```python
if self._suppressed:
    return
```

- [ ] Update `server/pipeline_builder.py`.

When wake command processors exist, create the agent processor with:

```python
on_turn_started=voice_command_audio.audio_gate.suppress,
on_turn_finished=voice_command_audio.audio_gate.unsuppress,
```

Keep the no-wake path passing no callbacks.

- [ ] Run green tests.

```powershell
cd server
uv run pytest tests/test_voice_runtime_agent_turn.py tests/test_pipeline_builder.py tests/test_voice_runtime_wake_command.py -q
```

Expected: targeted tests pass.

- [ ] Commit.

```powershell
git add server/voice_runtime/agent_turn.py server/agent_processor_factory.py server/voice_runtime/wake_command.py server/pipeline_builder.py server/tests/test_voice_runtime_agent_turn.py server/tests/test_pipeline_builder.py server/tests/test_voice_runtime_wake_command.py
git commit -m "fix: suppress wake gate during agent turns"
```

---

## Step 7: Log Effective Wake Config At Startup

### Red

- [ ] Add a failing test in `server/tests/test_pipeline_builder.py` that monkeypatches `pipeline_builder.logger.info` and asserts the startup log contains:

```text
Wake config
threshold=0.7
vad_threshold=0.0
min_wake_rms=50.0
min_wake_peak=150
required_hits=1
rearm_delay_s=6.0
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_pipeline_builder.py -q
```

Expected: no startup config log exists.

### Green

- [ ] Update `server/pipeline_builder.py`.

Log after the wake detector and processors are configured:

```python
logger.info(
    "Wake config detector={} threshold={} vad_threshold={} candidate_log_threshold={} "
    "required_hits={} min_wake_rms={} min_wake_peak={} rearm_delay_s={}",
    profile.wake.detector,
    profile.wake.threshold,
    profile.wake.vad_threshold,
    profile.wake.candidate_log_threshold,
    profile.wake.required_hits,
    profile.wake.min_wake_rms,
    profile.wake.min_wake_peak,
    profile.wake.rearm_delay_s,
)
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_pipeline_builder.py -q
```

Expected: pipeline builder tests pass.

- [ ] Commit.

```powershell
git add server/pipeline_builder.py server/tests/test_pipeline_builder.py
git commit -m "chore: log effective wake config"
```

---

## Step 8: Full Verification

- [ ] Run targeted tests.

```powershell
cd server
uv run pytest tests/test_voice_runtime_profiles.py tests/test_voice_runtime_wake_command.py tests/test_voice_runtime_agent_turn.py tests/test_pipeline_builder.py -q
```

Expected: all targeted tests pass.

- [ ] Run full tests.

```powershell
cd server
uv run pytest -q
```

Expected: all tests pass.

- [ ] Run static checks.

```powershell
cd server
uv run ruff check .
uv run pyright .
```

Expected: ruff passes and pyright reports `0 errors`.

- [ ] Commit any verification-only fixes.

```powershell
git status --short
```

Expected: clean except pre-existing untracked `.pi/plans/*.md` files.

---

## Step 9: Live Verification Script

- [ ] Restart the server from `server/`.
- [ ] Open `/client/`.
- [ ] Stay quiet for 30 seconds.

Expected:

- No `Wake word detected` lines.
- Weak candidates, if any, log as `Wake candidate rejected ... reason=audio_level`.

- [ ] Say `mave` at normal speaking volume.

Expected:

- One `Wake word detected` line.
- Log includes configured `threshold`, `rms`, `peak`, `hits=1/1`, and `gate_open=true`.

- [ ] Say `mave move robot up`.

Expected:

- Transcript reaching Codex is `move robot up.` or equivalent without leading wake residue.
- No Codex turn for `Names,`, `Name.`, `Mail.`, or `up the robot wave.`

- [ ] Speak while the assistant is generating or talking.

Expected:

- No new wake detection until the agent turn finishes and `rearm_delay_s` has elapsed.

---

## Merge Back Locally

- [ ] Switch to `master`.

```powershell
git switch master
```

- [ ] Merge the branch.

```powershell
git merge --no-ff wake-live-run-stabilization
```

- [ ] Re-run the final verification commands from Step 8 on `master`.

Expected: all tests and checks still pass.
