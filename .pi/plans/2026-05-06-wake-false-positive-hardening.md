# Wake False Positive Hardening Plan

**Goal:** Reduce false `mave` wake activations and prevent junk wake-only transcripts from reaching the Codex robot agent, using TDD.

**Observed symptoms from live run:**
- Wake model triggered while user was not speaking (`mave=0.678`, repeated `mave=0.983`).
- STT produced junk commands like `May` and `Nave moved the robot.`.
- The agent treated junk transcripts as real turns.
- One Codex request appeared to start without a visible matching end log in the pasted run; crash stack trace still needed.

---

## Task 1: Add wake-only / too-short transcript filtering

**Files:**
- Modify: `server/voice_runtime/agent_turn.py` or `server/voice_runtime/wake_command.py`
- Add/update tests in `server/tests/`

### TDD
1. Add failing tests that assert these do **not** create agent turns:
   - `"Mave"`
   - `"Maeve"`
   - `"May"`
   - empty / whitespace text
   - one-token wake-like false positives
2. Run targeted tests and verify RED.
3. Implement minimal filtering:
   - after wake phrase cleanup, drop empty text
   - add a conservative `is_actionable_user_text()` helper
   - reject wake-only / likely wake misrecognitions before `AgentTurnProcessor` calls backend
4. Verify GREEN.

**Success:** false wake transcripts do not call Codex.

---

## Task 2: Make openWakeWord VAD threshold configurable

**Files:**
- Modify: `server/wake/openwakeword_detector.py`
- Modify: `server/voice_runtime/profiles.py`
- Modify: `server/runtime_profiles.toml`
- Modify tests for profile parsing and detector construction

### TDD
1. Add failing test that profile field `wake.vad_threshold` is parsed.
2. Add failing test that `OpenWakeWordDetector(..., vad_threshold=0.3)` passes `vad_threshold=0.3` into `openwakeword.Model`.
3. Verify RED.
4. Implement:
   - `WakeProfile.vad_threshold: float = 0.0`
   - parse/validate float, reject bool
   - pass to `OpenWakeWordDetector`
   - pass to `Model(..., vad_threshold=...)`
5. Verify GREEN.

**Initial config suggestion:**
```toml
threshold = 0.85
vad_threshold = 0.3
candidate_log_threshold = 0.5
```

**Success:** wake activation requires both wake score and speech activity when configured.

---

## Task 3: Require consecutive wake detections

**Files:**
- Modify: `server/voice_runtime/wake_command.py`
- Update wake command tests

### TDD
1. Add failing test: one high-scoring frame does **not** open the gate when `required_hits = 2`.
2. Add failing test: two consecutive high-scoring frames open the gate.
3. Add failing test: low/intervening frame resets hit count.
4. Verify RED.
5. Implement minimal state:
   - `required_hits`
   - `_consecutive_hits`
   - reset on non-detection / rearm / timeout
6. Verify GREEN.

**Success:** transient spikes no longer trigger wake.

---

## Task 4: Improve wake diagnostics

**Files:**
- Modify: `server/voice_runtime/wake_command.py`
- Possibly modify metrics tests

### TDD
1. Add tests for logging/metrics fields if practical, otherwise add unit coverage for computed diagnostics.
2. Log on candidate and detection:
   - wake score
   - threshold
   - required hit count progress
   - RMS/peak audio level
   - whether gate opened
3. Keep logs concise.

**Success:** next live run can distinguish noise spike, speech false positive, and threshold tuning issues.

---

## Task 5: Add agent request cancellation/end diagnostics

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/openai_codex_agent_processor.py` if needed
- Add tests if cancellation behavior can be simulated

### TDD
1. Add test/fake model that raises `asyncio.CancelledError` or a normal exception.
2. Verify logs/error path are observable and no silent hanging state remains.
3. Implement `try/finally` around model call logging:
   - start
   - end on success
   - cancelled
   - failed
4. Verify GREEN.

**Success:** every model request start has a visible success/failure/cancel log.

---

## Task 6: Validation sequence

Run from `server/`:

```bash
uv run pytest tests/test_voice_runtime_wake_command.py tests/test_voice_runtime_profiles.py tests/test_pipeline_builder.py tests/test_wake_gate.py tests/test_transcript_cleanup.py -q
uv run pytest -q
uv run ruff check .
uv run pyright .
```

Then live test:
1. Quiet room, no speech for 60 seconds: expect no wake detections.
2. Background speech without `mave`: expect no wake detections.
3. Say `mave, what can you do?`: expect one wake and one agent turn.
4. Say `mave` only: expect no Codex turn or a local prompt asking for a command.
5. Say `mave, have the robot wave to me`: expect timing logs for observation, model request, tool call, follow-up model request.

---

## Open questions

- Need actual crash stack trace to plan the crash fix precisely.
- Need empirical threshold tuning for `models/mave.onnx`; proposed `0.85` is a starting point, not guaranteed.
- Decide whether wake-only `"mave"` should be ignored silently or answered locally with “What would you like me to do?”
