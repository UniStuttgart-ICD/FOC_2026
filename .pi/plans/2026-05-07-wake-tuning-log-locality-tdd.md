# Wake Tuning Log Locality Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep ad hoc wake tuning run logs out of `server/` by standardizing them under `server/logs/wake_tuning/`, while documenting that wake tuning settings remain a local override read by runtime config.

**Architecture:** The Wake Tuning Module should own tiny path helpers for local run logs. Runtime Profile TOML remains the checked-in app config; `wake_tuning_settings.json` remains the local saved tuning overlay that `config.load_runtime_config()` already reads through `apply_saved_wake_tuning()`.

**Tech Stack:** Python 3.10+, pytest, FastAPI wake tuning app, PowerShell manual run commands, existing `.gitignore` `server/logs/` rule.

---

## File Structure

- Create: `server/wake_tuning/log_paths.py`
  - Owns Wake Tuning log directory and file naming helpers.
  - No process launching and no logging side effects.
- Create: `server/tests/test_wake_tuning_log_paths.py`
  - Tests the helper through its public interface.
- Modify: `README.md`
  - Update the wake tuning run instructions to create `server/logs/wake_tuning/` and redirect stdout/stderr there.
  - Clarify that saved tuning settings are a local override read at bot startup, not an automatic edit to `runtime_profiles.toml`.
- Leave unchanged: `server/config.py`
  - It already reads saved tuning through `apply_saved_wake_tuning()`.
- Leave unchanged: `server/wake_tuning/settings.py`
  - The current settings path behavior is correct for this task.
- Local cleanup only: move existing ignored `server/wake_tuning_*.log` files into `server/logs/wake_tuning/`.

## Current Behavior To Preserve

Wake tuning settings currently follow this flow:

```text
Wake tuning UI POST /api/settings
-> wake_tuning.settings.save_profile_settings()
-> server/wake_tuning_settings.json, or WAKE_TUNING_SETTINGS_PATH
-> config.load_runtime_config()
-> wake_tuning.settings.apply_saved_wake_tuning()
-> selected Runtime Profile wake values are overridden at bot startup
```

Do not make the UI edit `server/runtime_profiles.toml` in this plan. A local settings overlay is safer for tuning because developers can try values without changing checked-in profile defaults.

## Task 1: Add Wake Tuning Log Path Helper

**Files:**
- Create: `server/tests/test_wake_tuning_log_paths.py`
- Create: `server/wake_tuning/log_paths.py`

- [ ] **Step 1: Write the failing test**

Create `server/tests/test_wake_tuning_log_paths.py`:

```python
from pathlib import Path

from wake_tuning.log_paths import default_log_dir, log_paths


def test_default_log_dir_lives_under_server_logs_wake_tuning(tmp_path: Path) -> None:
    assert default_log_dir(tmp_path) == tmp_path / "logs" / "wake_tuning"


def test_log_paths_use_safe_label_and_out_err_suffixes(tmp_path: Path) -> None:
    paths = log_paths("manual run", server_dir=tmp_path)

    assert paths.stdout == tmp_path / "logs" / "wake_tuning" / "wake_tuning_manual_run.out.log"
    assert paths.stderr == tmp_path / "logs" / "wake_tuning" / "wake_tuning_manual_run.err.log"


def test_log_paths_default_to_server_label(tmp_path: Path) -> None:
    paths = log_paths(server_dir=tmp_path)

    assert paths.stdout.name == "wake_tuning_server.out.log"
    assert paths.stderr.name == "wake_tuning_server.err.log"
```

- [ ] **Step 2: Run the test to verify RED**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning_log_paths.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'wake_tuning.log_paths'`.

- [ ] **Step 3: Write the minimal implementation**

Create `server/wake_tuning/log_paths.py`:

```python
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WakeTuningLogPaths:
    stdout: Path
    stderr: Path


def default_log_dir(server_dir: Path | None = None) -> Path:
    root = server_dir or Path(__file__).resolve().parents[1]
    return root / "logs" / "wake_tuning"


def log_paths(label: str = "server", *, server_dir: Path | None = None) -> WakeTuningLogPaths:
    safe_label = _safe_label(label)
    log_dir = default_log_dir(server_dir)
    basename = f"wake_tuning_{safe_label}"
    return WakeTuningLogPaths(
        stdout=log_dir / f"{basename}.out.log",
        stderr=log_dir / f"{basename}.err.log",
    )


def _safe_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return normalized or "server"
```

- [ ] **Step 4: Run the test to verify GREEN**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning_log_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Run focused wake tuning tests**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py tests/test_wake_tuning_app.py tests/test_wake_tuning_log_paths.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add server/wake_tuning/log_paths.py server/tests/test_wake_tuning_log_paths.py
git commit -m "test: add wake tuning log path helper"
```

## Task 2: Document Log Locality And Settings Semantics

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Write the failing docs test**

Add this test to `server/tests/test_wake_tuning_log_paths.py`:

```python
def test_readme_routes_wake_tuning_logs_under_server_logs() -> None:
    readme = Path(__file__).resolve().parents[2] / "README.md"
    text = readme.read_text(encoding="utf-8")

    assert "logs/wake_tuning" in text
    assert "wake_tuning_server.out.log" in text
    assert "wake_tuning_server.err.log" in text
    assert "local override" in text
    assert "does not edit `server/runtime_profiles.toml`" in text
```

- [ ] **Step 2: Run the test to verify RED**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning_log_paths.py::test_readme_routes_wake_tuning_logs_under_server_logs -q
```

Expected: FAIL because README does not mention the new log directory or local override wording yet.

- [ ] **Step 3: Update README wake tuning section**

Replace the current wake tuning run block in `README.md` with:

````markdown
Run the independent wake tuning page from `server/`:

```powershell
$logDir = "logs/wake_tuning"
New-Item -ItemType Directory -Force $logDir | Out-Null
uv run python -m wake_tuning.app 1> "$logDir/wake_tuning_server.out.log" 2> "$logDir/wake_tuning_server.err.log"
```

Open `http://127.0.0.1:9010`, start the mic, tune the sliders, then use **Save / implement**. Saved values go to `server/wake_tuning_settings.json` by default, or `WAKE_TUNING_SETTINGS_PATH` when set. This file is a local override read by runtime config at bot startup; it does not edit `server/runtime_profiles.toml`.
````

- [ ] **Step 4: Run the docs test to verify GREEN**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning_log_paths.py::test_readme_routes_wake_tuning_logs_under_server_logs -q
```

Expected: PASS.

- [ ] **Step 5: Run all wake tuning tests**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py tests/test_wake_tuning_app.py tests/test_wake_tuning_log_paths.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add README.md server/tests/test_wake_tuning_log_paths.py
git commit -m "docs: route wake tuning logs under server logs"
```

## Task 3: Move Existing Ignored Wake Tuning Logs

**Files:**
- Move local ignored files only:
  - From: `server/wake_tuning_*.log`
  - To: `server/logs/wake_tuning/`

- [ ] **Step 1: Confirm current root wake tuning logs**

Run from repo root:

```powershell
Get-ChildItem server -Filter 'wake_tuning_*.log' | Select-Object Name
```

Expected: shows any current root wake tuning logs, or no output if they were already moved.

- [ ] **Step 2: Move logs under server logs**

Run from repo root:

```powershell
New-Item -ItemType Directory -Force 'server/logs/wake_tuning' | Out-Null
Get-ChildItem server -Filter 'wake_tuning_*.log' | Move-Item -Destination 'server/logs/wake_tuning'
```

Expected: no errors.

- [ ] **Step 3: Verify root is clean of wake tuning logs**

Run from repo root:

```powershell
Get-ChildItem server -Filter 'wake_tuning_*.log'
```

Expected: no output.

- [ ] **Step 4: Verify moved logs are ignored under server logs**

Run from repo root:

```bash
git status --ignored --short -- server/logs/wake_tuning
```

Expected: moved logs appear as ignored entries under `server/logs/wake_tuning/`.

No commit is required for moved ignored log files.

## Task 4: Final Verification

**Files:**
- Verify changed source/docs only.

- [ ] **Step 1: Run focused tests**

Run from `server/`:

```bash
uv run pytest tests/test_wake_tuning.py tests/test_wake_tuning_app.py tests/test_wake_tuning_log_paths.py -q
```

Expected: PASS.

- [ ] **Step 2: Run lint on changed Python files**

Run from `server/`:

```bash
uv run ruff check wake_tuning/log_paths.py tests/test_wake_tuning_log_paths.py
```

Expected: PASS.

- [ ] **Step 3: Run pyright on changed Python files**

Run from `server/`:

```bash
uv run pyright wake_tuning/log_paths.py tests/test_wake_tuning_log_paths.py
```

Expected: 0 errors.

- [ ] **Step 4: Inspect final status**

Run from repo root:

```bash
git status --short
```

Expected: only intended tracked changes remain, and root `server/wake_tuning_*.log` files are gone.

- [ ] **Step 5: Commit final fixes if needed**

If Task 4 produced any small tracked corrections, commit them:

```bash
git add README.md server/wake_tuning/log_paths.py server/tests/test_wake_tuning_log_paths.py
git commit -m "chore: keep wake tuning logs under logs"
```

## Self-Review

- Spec coverage: The plan routes ad hoc wake tuning logs under `server/logs/wake_tuning/`, preserves current wake tuning config overlay behavior, and documents the difference between local override and checked-in Runtime Profile config.
- Placeholder scan: No TBD, TODO, or unspecified implementation steps.
- Type consistency: `WakeTuningLogPaths`, `default_log_dir()`, and `log_paths()` names match across tests and implementation.
