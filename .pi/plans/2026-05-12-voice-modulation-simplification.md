# Voice Modulation Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Voice Mod Lab usable without audio-DSP expertise by exposing five high-impact voice controls and five focused presets.

**Architecture:** Keep `VoiceModulationSettings` and DSP fields compatible. Reduce built-in presets in `settings.py`; map five browser macro sliders onto the existing fields in `static/index.html`.

**Tech Stack:** Python dataclasses/FastAPI, static HTML/CSS/JavaScript, pytest, playwright-cli.

---

### Task 1: Focus The User-Facing Presets

**Files:**
- Modify: `server/tests/test_voice_modulation_app.py`
- Modify: `server/voice_modulation/settings.py`

- [ ] **Step 1: Write the failing test**

Add a test that asserts `/api/presets` exposes only:

```python
["clean", "robot", "radio", "small_speaker", "low_battery"]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run pytest server/tests/test_voice_modulation_app.py::test_presets_route_exposes_focused_preset_set -q
```

Expected: fail because extra presets are still exposed.

- [ ] **Step 3: Write minimal implementation**

Delete the extra built-in entries from `BUILT_IN_PRESETS`: `giant`, `wide_chorus`, `echo_room`, and `ghost`.

- [ ] **Step 4: Run test to verify it passes**

Run the same focused pytest command. Expected: pass.

### Task 2: Replace Raw DSP Sliders With Macro Controls

**Files:**
- Modify: `server/tests/test_voice_modulation_app.py`
- Modify: `server/voice_modulation/static/index.html`

- [ ] **Step 1: Write the failing test**

Extend the index-page test to require these labels:

```text
Voice size
Robot edge
Radio filter
Glitch
Space
```

and reject raw expert labels:

```text
Ring modulation
Tremolo depth
Echo feedback
Bit depth
Limiter
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
uv run pytest server/tests/test_voice_modulation_app.py::test_index_page_serves_voice_mod_lab_workbench -q
```

Expected: fail because the current page still shows raw controls.

- [ ] **Step 3: Write minimal implementation**

Replace `controlGroups` with five macro controls. Keep `state.settings` as the saved schema and have each macro update only the existing DSP fields it owns:

```text
Voice size -> pitch_shift_semitones
Robot edge -> ring_mod_hz, drive
Radio filter -> low_cut_hz, high_cut_hz, noise_mix
Glitch -> tremolo_hz, tremolo_depth, bit_depth
Space -> chorus_rate_hz, chorus_depth_ms, chorus_mix, echo_delay_ms, echo_feedback, echo_mix
```

Remove the visible limiter row and keep its stored value unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run the same focused pytest command. Expected: pass.

### Task 3: Verify Runtime Compatibility And UI

**Files:**
- Test only.

- [ ] **Step 1: Run voice modulation tests**

```powershell
uv run pytest server/tests/test_voice_modulation_settings.py server/tests/test_voice_modulation_dsp.py server/tests/test_voice_modulation_app.py -q
```

- [ ] **Step 2: Run lint/type checks scoped to touched code**

```powershell
uv run ruff check server/voice_modulation server/tests/test_voice_modulation_app.py
uv run pyright server/voice_modulation server/tests/test_voice_modulation_app.py
```

- [ ] **Step 3: Verify browser-visible UI**

Start Voice Mod Lab on a local port, open it with `playwright-cli`, capture a screenshot, and confirm the five macro controls render without raw expert labels.
