# OpenAI Realtime Whisper STT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add OpenAI Realtime Whisper as a selectable low-latency STT path and prepare it to replace Deepgram Flux after live evidence.

**Architecture:** Keep the existing cascade pipeline: transport audio -> wake gate -> STT -> user aggregator -> agent -> TTS -> transport output. Use Pipecat's existing `OpenAIRealtimeSTTService` instead of adding a custom WebSocket client. Add a mixed profile so OpenAI STT can be compared against the current Deepgram Flux default without changing the default profile in the first branch.

**Tech Stack:** Python 3.12, Pipecat `OpenAIRealtimeSTTService`, OpenAI Realtime transcription, TOML runtime profiles, pytest.

---

## Source Context

- Official docs: `https://developers.openai.com/api/docs/guides/realtime`
- Official docs: `https://developers.openai.com/api/docs/guides/realtime-transcription`
- Current default STT profile: `server/runtime_profiles.toml`
- Current STT factory: `server/voice_runtime/providers.py`
- Current pipeline assembly: `server/pipeline_builder.py`

## Scope

This branch integrates OpenAI Realtime Whisper STT and adds a profile for comparison. It does not switch `DEFAULT_PROFILE` away from `hybrid_low_latency`. Make that replacement only after live latency and command-quality evidence is captured.

## File Map

- Modify: `server/voice_runtime/providers.py`
  - Change the OpenAI Realtime STT default model to `gpt-realtime-whisper`.
  - Move `noise_reduction="near_field"` into `OpenAIRealtimeSTTService.Settings(...)` to avoid the deprecated constructor parameter.
- Modify: `server/runtime_profiles.toml`
  - Update `openai_all.stt.model` to `gpt-realtime-whisper`.
  - Add `hybrid_openai_stt`: current default wake, Gemini agent, Cartesia TTS, OpenAI Realtime Whisper STT.
- Modify: `server/tests/test_providers.py`
  - Update expectations for the OpenAI STT factory default and settings object.
- Modify: `server/tests/test_voice_runtime_profiles.py`
  - Add profile tests for `hybrid_openai_stt`.
  - Update `openai_all` expectations if existing tests assert the old STT model.
- Modify: `README.md`
  - Document the new profile and required keys.

## Acceptance Criteria

- `create_stt_service(STTProfile(provider="openai_realtime"))` creates `OpenAIRealtimeSTTService` with model `gpt-realtime-whisper`.
- No deprecation warning is emitted for the OpenAI STT `noise_reduction` constructor argument.
- `openai_all` uses `gpt-realtime-whisper` for STT.
- `hybrid_openai_stt` loads as a `benchmark_streaming` profile and requires `CARTESIA_API_KEY`, `OPENAI_API_KEY`, and `GOOGLE_API_KEY`.
- Deterministic tests pass.
- Live smoke evidence exists before any later branch changes the default profile.

---

### Task 1: Create The Implementation Branch

**Files:**
- No file changes.

- [ ] **Step 1: Start from the project repo**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git status --short
```

Expected: review output and note unrelated user changes. Do not revert unrelated changes.

- [ ] **Step 2: Create the branch**

Run:

```powershell
git switch -c openai-realtime-whisper-stt
```

Expected: `Switched to a new branch 'openai-realtime-whisper-stt'`.

- [ ] **Step 3: Capture baseline tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_providers.py tests/test_voice_runtime_profiles.py -q
```

Expected: all selected tests pass before editing.

---

### Task 2: Update OpenAI STT Factory Defaults

**Files:**
- Modify: `server/tests/test_providers.py`
- Modify: `server/voice_runtime/providers.py`

- [ ] **Step 1: Write the failing provider test**

In `server/tests/test_providers.py`, replace `test_creates_openai_realtime_stt` with:

```python
def test_creates_openai_realtime_stt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime"))

    service.Settings.assert_called_once_with(
        model="gpt-realtime-whisper",
        noise_reduction="near_field",
    )
    service.assert_called_once_with(api_key="oa", settings="settings")
```

- [ ] **Step 2: Add an explicit configured-model test**

In `server/tests/test_providers.py`, add this test after `test_creates_openai_realtime_stt`:

```python
def test_creates_openai_realtime_stt_with_configured_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime", model="custom-transcribe"))

    service.Settings.assert_called_once_with(
        model="custom-transcribe",
        noise_reduction="near_field",
    )
    service.assert_called_once_with(api_key="oa", settings="settings")
```

- [ ] **Step 3: Run the provider tests and verify failure**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_providers.py::test_creates_openai_realtime_stt tests/test_providers.py::test_creates_openai_realtime_stt_with_configured_model -q
```

Expected: first test fails because the factory still uses `gpt-4o-mini-transcribe` and passes `noise_reduction` to the service constructor.

- [ ] **Step 4: Implement the factory change**

In `server/voice_runtime/providers.py`, change the `openai_realtime` block to:

```python
    if config.provider == "openai_realtime":
        return OpenAIRealtimeSTTService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIRealtimeSTTService.Settings(
                model=config.model or "gpt-realtime-whisper",
                noise_reduction="near_field",
            ),
        )
```

- [ ] **Step 5: Run provider tests and verify pass**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_providers.py -q
```

Expected: provider tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add server/voice_runtime/providers.py server/tests/test_providers.py
git commit -m "feat: default OpenAI realtime STT to whisper"
```

---

### Task 3: Add The Hybrid OpenAI STT Runtime Profile

**Files:**
- Modify: `server/runtime_profiles.toml`
- Modify: `server/tests/test_voice_runtime_profiles.py`

- [ ] **Step 1: Write the failing profile test**

In `server/tests/test_voice_runtime_profiles.py`, add this test near the existing runtime profile tests:

```python
def test_hybrid_openai_stt_profile_uses_realtime_whisper():
    profile = load_runtime_profile(profile_name="hybrid_openai_stt")

    assert profile.category == "benchmark_streaming"
    assert profile.wake.provider == "openwakeword"
    assert profile.stt.provider == "openai_realtime"
    assert profile.stt.model == "gpt-realtime-whisper"
    assert profile.tts.provider == "cartesia"
    assert profile.agent.provider == "gemini_api"
    assert profile.required_env_names() == (
        "CARTESIA_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
    )
```

- [ ] **Step 2: Update the openai_all profile test**

If `server/tests/test_voice_runtime_profiles.py` has an assertion for `openai_all.stt.model`, make it:

```python
assert profile.stt.model == "gpt-realtime-whisper"
```

- [ ] **Step 3: Run the profile tests and verify failure**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_voice_runtime_profiles.py -q
```

Expected: the new test fails with `Unknown profile 'hybrid_openai_stt'`.

- [ ] **Step 4: Add the new runtime profile**

In `server/runtime_profiles.toml`, add this block after `profiles.hybrid_low_latency`:

```toml
[profiles.hybrid_openai_stt]
category = "benchmark_streaming"

[profiles.hybrid_openai_stt.wake]
provider = "openwakeword"
model_path = "models/mave.onnx"
threshold = 0.85
vad_threshold = 0.0
candidate_log_threshold = 0.45
required_hits = 1
min_wake_rms = 0.0
min_wake_peak = 12
rearm_delay_s = 6.0
pre_buffer_s = 0.2
single_command = true

[profiles.hybrid_openai_stt.emergency_stop]
enabled = false

[profiles.hybrid_openai_stt.stt]
provider = "openai_realtime"
model = "gpt-realtime-whisper"

[profiles.hybrid_openai_stt.tts]
provider = "cartesia"
model = "sonic-3"
voice = "47c38ca4-5f35-497b-b1a3-415245fb35e1"

[profiles.hybrid_openai_stt.agent]
provider = "gemini_api"
model = "gemini-3.1-flash-lite-preview"
reasoning_effort = "high"
api_key_env = "GOOGLE_API_KEY"

[profiles.hybrid_openai_stt.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.hybrid_openai_stt.metrics]
enabled = true
path = "logs/voice_metrics.jsonl"
include_text = true

[profiles.hybrid_openai_stt.process_trace]
enabled = true
path = "logs/process_trace.jsonl"
include_text = true
include_tool_payloads = true
```

- [ ] **Step 5: Update openai_all STT model**

In `server/runtime_profiles.toml`, change:

```toml
[profiles.openai_all.stt]
provider = "openai_realtime"
model = "gpt-4o-mini-transcribe"
```

to:

```toml
[profiles.openai_all.stt]
provider = "openai_realtime"
model = "gpt-realtime-whisper"
```

- [ ] **Step 6: Run profile tests and verify pass**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_voice_runtime_profiles.py -q
```

Expected: profile tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add server/runtime_profiles.toml server/tests/test_voice_runtime_profiles.py
git commit -m "feat: add hybrid OpenAI realtime STT profile"
```

---

### Task 4: Document The New Profile

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the runtime profile list**

In `README.md`, under `Run a specific profile`, make the command block include:

```bash
uv run bot.py --profile local_current
uv run bot.py --profile hybrid_openai_stt
uv run bot.py --profile openai_all
uv run bot.py --profile deepgram_all
uv run bot.py --profile no_wake_debug
```

- [ ] **Step 2: Add required keys for the hybrid OpenAI STT profile**

In `README.md`, under `Required keys`, add this block after the default profile keys:

````markdown
For `hybrid_openai_stt`, set:

```dotenv
OPENAI_API_KEY=
CARTESIA_API_KEY=
GOOGLE_API_KEY=
```
````

- [ ] **Step 3: Add a one-line profile description**

In `README.md`, under the default profile description, add:

```markdown
`hybrid_openai_stt` keeps the default wake word, Gemini agent, and Cartesia TTS stack, but replaces Deepgram Flux STT with OpenAI Realtime Whisper STT.
```

- [ ] **Step 4: Verify Markdown formatting**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git diff -- README.md
```

Expected: fenced code blocks render correctly and the new profile is listed once.

- [ ] **Step 5: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: document OpenAI realtime STT profile"
```

---

### Task 5: Run Deterministic Verification

**Files:**
- No source changes unless tests fail because of the branch changes.

- [ ] **Step 1: Run focused tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_providers.py tests/test_voice_runtime_profiles.py tests/test_config.py tests/test_pipeline_builder.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run lint import check**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m ruff check .
```

Expected: no new lint errors. If unrelated existing lint errors appear, record them in the final branch notes and do not refactor unrelated files.

- [ ] **Step 3: Run profile load smoke**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -c "from voice_runtime.profiles import load_runtime_profile; p=load_runtime_profile(profile_name='hybrid_openai_stt'); print(p.stt.provider, p.stt.model, p.required_env_names())"
```

Expected output includes:

```text
openai_realtime gpt-realtime-whisper ('CARTESIA_API_KEY', 'OPENAI_API_KEY', 'GOOGLE_API_KEY')
```

- [ ] **Step 4: Commit fixes if verification required changes**

If Task 5 required edits, run:

```powershell
git add server README.md
git commit -m "test: verify OpenAI realtime STT profile"
```

If Task 5 required no edits, do not create an empty commit.

---

### Task 6: Live STT Smoke Test

**Files:**
- No source changes.
- Generated evidence: `server/logs/voice_metrics/...`
- Generated evidence: `server/logs/process_trace/...`

- [ ] **Step 1: Confirm credentials are available**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -c "import os; missing=[k for k in ('OPENAI_API_KEY','CARTESIA_API_KEY','GOOGLE_API_KEY') if not os.getenv(k)]; print('missing=' + ','.join(missing) if missing else 'all required keys present')"
```

Expected: `all required keys present`.

- [ ] **Step 2: Start the bot with the new profile**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe bot.py --profile hybrid_openai_stt
```

Expected: bot starts with `stt=openai_realtime`, `tts=cartesia`, and `agent=gemini_api`.

- [ ] **Step 3: Speak three test commands**

Use these commands in one live session:

```text
Mave, move up a bit.
Mave, move left five centimeters.
Mave, stop.
```

Expected: final transcripts are complete enough for the agent to identify the intended robot command. Do not judge from interim deltas alone.

- [ ] **Step 4: Inspect traces and metrics**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
Get-ChildItem .\logs\voice_metrics -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 3 FullName,LastWriteTime
Get-ChildItem .\logs\process_trace -Recurse | Sort-Object LastWriteTime -Descending | Select-Object -First 3 FullName,LastWriteTime
```

Expected: new files exist for the live session.

- [ ] **Step 5: Record live result in branch notes**

Add this exact summary to the branch final message or PR body:

```markdown
Live STT smoke:
- Profile: `hybrid_openai_stt`
- Commands tested: `move up`, `move left five centimeters`, `stop`
- Transcript quality: pass/fail per command
- Observed STT latency: copied from latest voice metrics
- Evidence files: latest `server/logs/voice_metrics/...` and `server/logs/process_trace/...`
```

Do not commit generated log files unless the project owner explicitly requests evidence artifacts in git.

---

### Task 7: Replacement Decision Gate

**Files:**
- No source changes in this branch unless the owner explicitly asks to make OpenAI STT the default now.

- [ ] **Step 1: Compare against the current default profile**

Run the same three commands with:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe bot.py --profile hybrid_low_latency
```

Expected: comparable voice metrics and process traces exist for the Deepgram Flux baseline.

- [ ] **Step 2: Apply the replacement rule**

Use this rule:

```text
Replace the default STT only if OpenAI Realtime Whisper has comparable or better command transcript quality and no unacceptable latency regression in the live metrics.
```

- [ ] **Step 3: Leave default unchanged if evidence is mixed**

If evidence is mixed, keep `DEFAULT_PROFILE = "hybrid_low_latency"` in `server/voice_runtime/profiles.py` and keep `hybrid_openai_stt` as an opt-in benchmark profile.

- [ ] **Step 4: If explicitly approved, switch default in a follow-up branch**

If the owner approves replacing the default, create a new branch and change:

```python
DEFAULT_PROFILE = "hybrid_low_latency"
```

to:

```python
DEFAULT_PROFILE = "hybrid_openai_stt"
```

Then update README's default profile description and run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
.\.venv\Scripts\python.exe -m pytest tests/test_voice_runtime_profiles.py tests/test_config.py -q
```

Expected: tests pass with the new default.

---

## Self-Review Notes

- Spec coverage: plan covers provider integration, profile integration, tests, docs, deterministic verification, live smoke, and replacement gate.
- Placeholder scan: no placeholder markers or unspecified implementation steps remain.
- Type consistency: profile name is consistently `hybrid_openai_stt`; STT provider is `openai_realtime`; model is `gpt-realtime-whisper`; required env names match `RuntimeProfile.required_env_names()` ordering.
