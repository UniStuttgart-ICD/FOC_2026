# Wake Tuning State Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop treating wake tuning UI output as tracked source while preserving the current tuned default wake behaviour.

**Architecture:** Runtime Profile values remain committed app configuration in `server/runtime_profiles.toml`. Wake tuning UI saves become local runtime state under an ignored `server/state/` directory. `WAKE_TUNING_SETTINGS_PATH` stays the explicit override seam for tests and developer experiments.

**Tech Stack:** Python, pytest, FastAPI TestClient, TOML runtime profiles, git ignore rules.

---

## File Structure

- Modify: `server/tests/test_wake_tuning.py` - behaviour tests for promoted Runtime Profile defaults and the new local state path.
- Modify: `server/tests/test_wake_tuning_app.py` - app-level test that the settings endpoint reports the new default path.
- Modify: `server/wake_tuning/settings.py` - change the default wake tuning state path.
- Modify: `server/runtime_profiles.toml` - promote the current tracked `wake_tuning_settings.json` values into `profiles.hybrid_low_latency.wake`.
- Modify: `.gitignore` - ignore local wake tuning state and the legacy root settings path.
- Modify: `README.md` - document the new local state path and promotion workflow.
- Delete: `server/wake_tuning_settings.json` - remove tracked runtime state.

## Current Values To Preserve

Promote these values from the currently tracked `server/wake_tuning_settings.json` into `[profiles.hybrid_low_latency.wake]`:

```toml
threshold = 0.85
vad_threshold = 0.0
candidate_log_threshold = 0.45
required_hits = 1
min_wake_rms = 0.0
min_wake_peak = 12
rearm_delay_s = 6.0
pre_buffer_s = 0.2
```

Leave unrelated current changes in `server/runtime_profiles.toml` alone, especially the agent provider/model lines already modified in the working tree.

---

### Task 1: Lock Current Tuned Defaults Into Runtime Profile

**Files:**
- Modify: `server/tests/test_wake_tuning.py`
- Modify: `server/runtime_profiles.toml`

- [ ] **Step 1: Write the failing test**

Append this test to `server/tests/test_wake_tuning.py`:

```python
def test_default_hybrid_profile_contains_promoted_wake_tuning_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WAKE_TUNING_SETTINGS_PATH", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")

    config = load_runtime_config(profile_name="hybrid_low_latency")

    assert config.wake.threshold == 0.85
    assert config.wake.vad_threshold == 0.0
    assert config.wake.candidate_log_threshold == 0.45
    assert config.wake.required_hits == 1
    assert config.wake.min_wake_rms == 0.0
    assert config.wake.min_wake_peak == 12
    assert config.wake.rearm_delay_s == 6.0
    assert config.wake.pre_buffer_s == 0.2
```

- [ ] **Step 2: Run test to verify it fails**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py::test_default_hybrid_profile_contains_promoted_wake_tuning_values -q
```

Expected: FAIL because the committed profile still has `threshold = 0.55`, `min_wake_rms = 4.0`, and `pre_buffer_s = 0.5`.

- [ ] **Step 3: Write minimal implementation**

In `server/runtime_profiles.toml`, update only `[profiles.hybrid_low_latency.wake]` to match the promoted values:

```toml
[profiles.hybrid_low_latency.wake]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py::test_default_hybrid_profile_contains_promoted_wake_tuning_values -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add server/tests/test_wake_tuning.py server/runtime_profiles.toml
git commit -m "test: lock promoted wake tuning defaults"
```

---

### Task 2: Move Default Wake Tuning State To Ignored Runtime State

**Files:**
- Modify: `server/tests/test_wake_tuning.py`
- Modify: `server/wake_tuning/settings.py`

- [ ] **Step 1: Write the failing test**

Append this test to `server/tests/test_wake_tuning.py`:

```python
from wake_tuning.settings import default_settings_path


def test_default_wake_tuning_settings_path_uses_local_state_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WAKE_TUNING_SETTINGS_PATH", raising=False)

    assert default_settings_path(tmp_path) == tmp_path / "state" / "wake_tuning_settings.json"
```

If `default_settings_path` is already imported in the existing import block, only add it there once.

- [ ] **Step 2: Run test to verify it fails**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py::test_default_wake_tuning_settings_path_uses_local_state_dir -q
```

Expected: FAIL because the function currently returns `tmp_path / "wake_tuning_settings.json"`.

- [ ] **Step 3: Write minimal implementation**

Change `default_settings_path` in `server/wake_tuning/settings.py` to:

```python
def default_settings_path(server_dir: Path | None = None) -> Path:
    configured = os.getenv(SETTINGS_ENV)
    if configured:
        return Path(configured)
    root = server_dir or Path(__file__).resolve().parents[1]
    return root / "state" / "wake_tuning_settings.json"
```

- [ ] **Step 4: Run test to verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py::test_default_wake_tuning_settings_path_uses_local_state_dir -q
```

Expected: PASS.

- [ ] **Step 5: Run existing override coverage**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py::test_saved_tuning_overrides_runtime_config_wake_values tests/test_wake_tuning.py::test_settings_round_trip_per_profile -q
```

Expected: PASS. This proves `WAKE_TUNING_SETTINGS_PATH` still works and explicit paths still round-trip.

- [ ] **Step 6: Commit**

```bash
git add server/tests/test_wake_tuning.py server/wake_tuning/settings.py
git commit -m "fix: store wake tuning state under local state dir"
```

---

### Task 3: Show New State Path Through Wake Tuning App

**Files:**
- Modify: `server/tests/test_wake_tuning_app.py`
- Modify: `server/wake_tuning/app.py` only if the test exposes a stale path

- [ ] **Step 1: Write the failing or confirming test**

Append this test to `server/tests/test_wake_tuning_app.py`:

```python
def test_settings_api_reports_default_local_state_path(monkeypatch):
    monkeypatch.delenv("WAKE_TUNING_SETTINGS_PATH", raising=False)
    client = TestClient(wake_tuning_app.app)

    response = client.get("/api/settings?profile=hybrid_low_latency")

    assert response.status_code == 200
    assert response.json()["settings_path"].endswith("state/wake_tuning_settings.json")
```

- [ ] **Step 2: Run test**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning_app.py::test_settings_api_reports_default_local_state_path -q
```

Expected after Task 2: PASS. If it fails with the old root path, update `server/wake_tuning/app.py` to call `default_settings_path(SERVER_DIR)` everywhere it reports, loads, or saves settings. The current implementation already does this, so no production change should be needed.

- [ ] **Step 3: Commit**

If only the test changed:

```bash
git add server/tests/test_wake_tuning_app.py
git commit -m "test: cover wake tuning app state path"
```

If `server/wake_tuning/app.py` also changed:

```bash
git add server/tests/test_wake_tuning_app.py server/wake_tuning/app.py
git commit -m "fix: report wake tuning local state path"
```

---

### Task 4: Remove Tracked Runtime State And Ignore Future State

**Files:**
- Modify: `.gitignore`
- Delete: `server/wake_tuning_settings.json`

- [ ] **Step 1: Write ignore rules**

Add this block near the runtime artifact section in `.gitignore`:

```gitignore
# Local runtime state
server/state/
server/wake_tuning_settings.json
```

- [ ] **Step 2: Remove tracked runtime state**

Run from repo root:

```bash
git rm server/wake_tuning_settings.json
```

Expected: file is staged for deletion.

- [ ] **Step 3: Verify ignore behaviour**

Run from repo root:

```bash
git check-ignore server/state/wake_tuning_settings.json server/wake_tuning_settings.json
```

Expected output:

```text
server/state/wake_tuning_settings.json
server/wake_tuning_settings.json
```

- [ ] **Step 4: Run wake tuning tests**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py tests/test_wake_tuning_app.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add .gitignore
git commit -m "chore: ignore local wake tuning state"
```

---

### Task 5: Document Local State And Promotion Workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update README text**

Replace the wake tuning paragraph in `README.md` with:

```markdown
Open `http://127.0.0.1:9010`, start the mic, tune the sliders, then use **Save / implement**. Saved values go to ignored local state at `server/state/wake_tuning_settings.json` and override the selected profile's wake settings when the Pipecat bot starts.

To make tuned values the shared default, copy the saved profile values into `server/runtime_profiles.toml` and commit the profile change. Do not commit `server/state/wake_tuning_settings.json`.
```

- [ ] **Step 2: Verify docs mention the new path only**

Run from repo root:

```bash
rg -n "wake_tuning_settings\\.json|server/state" README.md server docs .gitignore
```

Expected: no docs claim that `server/wake_tuning_settings.json` is the save target. `.gitignore` may still mention the legacy root path as an ignored compatibility guard.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document wake tuning local state"
```

---

### Task 6: Final Verification

**Files:**
- No code changes.

- [ ] **Step 1: Run targeted tests**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py tests/test_wake_tuning_app.py tests/test_config.py -q
```

Expected: PASS.

- [ ] **Step 2: Run structural checks**

Run from `server/`:

```bash
uv run ruff check wake_tuning/settings.py tests/test_wake_tuning.py tests/test_wake_tuning_app.py
```

Expected: PASS.

- [ ] **Step 3: Confirm no tracked wake tuning state remains**

Run from repo root:

```bash
git ls-files server/wake_tuning_settings.json
```

Expected: no output.

- [ ] **Step 4: Confirm local state stays ignored**

Run from repo root:

```bash
git check-ignore server/state/wake_tuning_settings.json
```

Expected output:

```text
server/state/wake_tuning_settings.json
```

- [ ] **Step 5: Inspect final diff**

Run from repo root:

```bash
git diff --stat HEAD
git diff -- server/runtime_profiles.toml server/wake_tuning/settings.py server/tests/test_wake_tuning.py server/tests/test_wake_tuning_app.py .gitignore README.md
```

Expected: diff only covers wake tuning state cleanup, promoted wake values, tests, ignore rules, and docs.

---

## Self-Review

- Spec coverage: the plan removes tracked UI output, preserves current tuned Runtime Profile behaviour, keeps `WAKE_TUNING_SETTINGS_PATH`, ignores future local state, and updates docs.
- Red-flag scan: no unfinished-marker steps remain.
- Type consistency: all snippets use existing `WakeTuningSettings`, `default_settings_path`, `load_runtime_config`, `TestClient`, and pytest names.
- TDD check: each behaviour change starts with one failing or confirming test before production changes. The non-code ignore cleanup uses command verification because git ignore behaviour is repository metadata, not Python runtime behaviour.
