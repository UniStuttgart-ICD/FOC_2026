# Move Voice Providers Into Voice Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move STT/TTS provider construction from the legacy top-level `providers.py` file into the `voice_runtime` Module without changing runtime behavior.

**Architecture:** `pipeline_builder.py` stays the app composition root and remains the only production caller of Voice Providers. `voice_runtime.providers` becomes the Voice Providers Adapter Module and must not import app-root `config`; it should depend on `voice_runtime.profiles` profile types. The legacy top-level `server/providers.py` is deleted after callers and tests move.

**Tech Stack:** Python 3.10+, pytest, ruff, pyright, Pipecat STT/TTS adapter classes.

---

## File Structure

- Create: `server/voice_runtime/providers.py` - Voice Providers Adapter Module containing `create_stt_service()` and `create_tts_service()`.
- Delete: `server/providers.py` - legacy top-level placement.
- Modify: `server/pipeline_builder.py` - import Voice Providers from `voice_runtime.providers`.
- Modify: `server/tests/test_providers.py` - test the new Module path and patch targets.
- Modify: `server/tests/test_orthogonal_imports.py` - remove `providers` from app-root Modules and add a deleted-legacy assertion for `server/providers.py`.
- Optional docs check: `ARCHITECTURE.md` already states the target; update only if the final wording becomes inaccurate.

## Task 1: Add Red Tests For The New Module Home

**Files:**
- Modify: `server/tests/test_providers.py`
- Modify: `server/tests/test_orthogonal_imports.py`

- [ ] **Step 1: Write the failing provider import test**

Replace the imports and patch targets in `server/tests/test_providers.py` so the tests expect `voice_runtime.providers`:

```python
from unittest.mock import Mock, patch

from config import STTConfig, TTSConfig
from voice_runtime.providers import create_stt_service, create_tts_service


def test_creates_whisper_stt():
    with patch("voice_runtime.providers.WhisperSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="whisper", model="base", device="cuda"))

    service.Settings.assert_called_once_with(model="base")
    service.assert_called_once_with(device="cuda", settings="settings")


def test_creates_kokoro_tts():
    with patch("voice_runtime.providers.KokoroTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="kokoro", voice="af_heart"))

    service.Settings.assert_called_once_with(voice="af_heart")
    service.assert_called_once_with(settings="settings")


def test_creates_deepgram_flux_stt(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("voice_runtime.providers.DeepgramFluxSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="deepgram_flux", model="flux-general-en"))

    service.Settings.assert_called_once_with(model="flux-general-en")
    service.assert_called_once_with(api_key="dg", settings="settings")


def test_creates_openai_realtime_stt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime", model="gpt-4o-mini-transcribe"))

    service.Settings.assert_called_once_with(model="gpt-4o-mini-transcribe")
    service.assert_called_once_with(api_key="oa", settings="settings", noise_reduction="near_field")


def test_creates_cartesia_tts(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    with patch("voice_runtime.providers.CartesiaTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="cartesia", model="sonic-3", voice="voice-id"))

    service.Settings.assert_called_once_with(model="sonic-3", voice="voice-id")
    service.assert_called_once_with(api_key="ct", settings="settings")


def test_creates_cartesia_tts_with_default_voice_id(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    with patch("voice_runtime.providers.CartesiaTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="cartesia", model="sonic-3"))

    service.Settings.assert_called_once_with(
        model="sonic-3", voice="47c38ca4-5f35-497b-b1a3-415245fb35e1"
    )
    service.assert_called_once_with(api_key="ct", settings="settings")


def test_creates_openai_tts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAITTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="openai", model="gpt-4o-mini-tts", voice="coral"))

    service.Settings.assert_called_once_with(model="gpt-4o-mini-tts", voice="coral")
    service.assert_called_once_with(api_key="oa", settings="settings")


def test_creates_deepgram_tts(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("voice_runtime.providers.DeepgramTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(
            TTSConfig(provider="deepgram", model="aura-2", voice="aura-2-andromeda-en")
        )

    service.Settings.assert_called_once_with(model="aura-2", voice="aura-2-andromeda-en")
    service.assert_called_once_with(api_key="dg", settings="settings")
```

- [ ] **Step 2: Write the failing legacy-placement test**

In `server/tests/test_orthogonal_imports.py`, remove `"providers"` from `APP_MODULE_ROOTS` and add this set plus test near the other deleted legacy Module tests:

```python
DELETED_LEGACY_VOICE_PROVIDER_MODULES = {
    SERVER_DIR / "providers.py",
}


def test_legacy_voice_provider_module_is_not_left_at_app_root():
    for path in DELETED_LEGACY_VOICE_PROVIDER_MODULES:
        assert not path.exists(), f"Legacy Voice Providers module still exists: {path}"
```

- [ ] **Step 3: Verify red**

Run from `server/`:

```bash
uv run pytest tests/test_providers.py tests/test_orthogonal_imports.py -q
```

Expected: FAIL because `voice_runtime.providers` does not exist and/or `server/providers.py` still exists. If the failure is a typo or collection error unrelated to the missing Module, fix the test and rerun until it fails for the expected reason.

- [ ] **Step 4: Commit the red tests**

```bash
git add tests/test_providers.py tests/test_orthogonal_imports.py
git commit -m "test: expect voice providers under voice runtime"
```

## Task 2: Move The Voice Providers Implementation

**Files:**
- Create: `server/voice_runtime/providers.py`
- Delete: `server/providers.py`

- [ ] **Step 1: Add the minimal implementation at the new Module path**

Create `server/voice_runtime/providers.py` with the existing provider construction logic, but import profile types from `voice_runtime.profiles`:

```python
from __future__ import annotations

import os

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.flux.stt import DeepgramFluxSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.stt import OpenAIRealtimeSTTService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.whisper.stt import WhisperSTTService

from voice_runtime.profiles import STTProfile, TTSProfile

DEFAULT_CARTESIA_VOICE_ID = "47c38ca4-5f35-497b-b1a3-415245fb35e1"


def create_stt_service(config: STTProfile) -> FrameProcessor:
    if config.provider == "whisper":
        return WhisperSTTService(
            device=config.device or "cuda",
            settings=WhisperSTTService.Settings(
                model=config.model or os.getenv("WHISPER_MODEL") or os.getenv("OPENAI_MODEL") or "base",
            ),
        )
    if config.provider == "deepgram_flux":
        return DeepgramFluxSTTService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramFluxSTTService.Settings(model=config.model or "flux-general-en"),
        )
    if config.provider == "openai_realtime":
        return OpenAIRealtimeSTTService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAIRealtimeSTTService.Settings(
                model=config.model or "gpt-4o-mini-transcribe",
            ),
            noise_reduction="near_field",
        )
    raise ValueError(f"Unsupported STT provider: {config.provider}")


def create_tts_service(config: TTSProfile) -> FrameProcessor:
    if config.provider == "kokoro":
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(
                voice=config.voice or os.getenv("KOKORO_VOICE_ID") or "af_heart"
            ),
        )
    if config.provider == "cartesia":
        return CartesiaTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaTTSService.Settings(
                model=config.model or "sonic-3",
                voice=config.voice or os.getenv("CARTESIA_VOICE_ID") or DEFAULT_CARTESIA_VOICE_ID,
            ),
        )
    if config.provider == "openai":
        return OpenAITTSService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAITTSService.Settings(
                model=config.model or "gpt-4o-mini-tts",
                voice=config.voice or "coral",
            ),
        )
    if config.provider == "deepgram":
        return DeepgramTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramTTSService.Settings(
                model=config.model or "aura-2",
                voice=config.voice or "aura-2-andromeda-en",
            ),
        )
    raise ValueError(f"Unsupported TTS provider: {config.provider}")
```

- [ ] **Step 2: Delete the legacy file**

Delete `server/providers.py`.

- [ ] **Step 3: Verify green for provider Module tests**

Run from `server/`:

```bash
uv run pytest tests/test_providers.py tests/test_orthogonal_imports.py -q
```

Expected: PASS for provider tests and the legacy-placement test. If import-guard failures mention `voice_runtime.providers` importing `config`, fix the new Module to import only from `voice_runtime.profiles`.

- [ ] **Step 4: Commit the moved Module**

```bash
git add voice_runtime/providers.py providers.py tests/test_orthogonal_imports.py
git commit -m "refactor: move voice providers into voice runtime"
```

## Task 3: Rewire The App Composition Root

**Files:**
- Modify: `server/pipeline_builder.py`
- Modify: `server/tests/test_pipeline_builder.py`

- [ ] **Step 1: Write or confirm the failing composition-root expectation**

The existing `server/tests/test_pipeline_builder.py` patches `pipeline_builder.create_stt_service` and `pipeline_builder.create_tts_service`, which remains correct because `pipeline_builder.py` should import the factory functions into its own composition-root namespace. Add this import-only regression test if it is not already covered:

```python
def test_pipeline_builder_uses_voice_runtime_provider_factories():
    import pipeline_builder
    import voice_runtime.providers

    assert pipeline_builder.create_stt_service is voice_runtime.providers.create_stt_service
    assert pipeline_builder.create_tts_service is voice_runtime.providers.create_tts_service
```

- [ ] **Step 2: Verify red**

Run from `server/`:

```bash
uv run pytest tests/test_pipeline_builder.py::test_pipeline_builder_uses_voice_runtime_provider_factories -q
```

Expected: FAIL because `pipeline_builder.py` still imports from the deleted legacy `providers` Module.

- [ ] **Step 3: Update the production import**

In `server/pipeline_builder.py`, replace:

```python
from providers import create_stt_service, create_tts_service
```

with:

```python
from voice_runtime.providers import create_stt_service, create_tts_service
```

- [ ] **Step 4: Verify green for pipeline wiring**

Run from `server/`:

```bash
uv run pytest tests/test_pipeline_builder.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit the composition-root rewire**

```bash
git add pipeline_builder.py tests/test_pipeline_builder.py
git commit -m "refactor: wire pipeline builder to voice runtime providers"
```

## Task 4: Clean Stale References And Run Focused Static Checks

**Files:**
- Modify only if stale references are found: `server/tests/*`, `AGENTS.md`, `ARCHITECTURE.md`, `CONTEXT.md`, `docs/*`

- [ ] **Step 1: Search for stale root-provider imports**

Run from repo root:

```bash
rg -n "from providers|import providers|patch\\(\"providers\\.|providers\\.py" server AGENTS.md ARCHITECTURE.md CONTEXT.md docs
```

Expected: no production or test imports from `providers`. It is acceptable for `ARCHITECTURE.md` to mention `providers.py` as historical legacy placement only if that sentence still reads accurately after deletion; otherwise update it to say the legacy placement has been removed.

- [ ] **Step 2: Run focused tests**

Run from `server/`:

```bash
uv run pytest tests/test_providers.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py -q
```

Expected: PASS.

- [ ] **Step 3: Run focused lint and type checks**

Run from `server/`:

```bash
uv run ruff check voice_runtime/providers.py pipeline_builder.py tests/test_providers.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py
uv run pyright voice_runtime/providers.py pipeline_builder.py tests/test_providers.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py
```

Expected: ruff passes and pyright reports 0 errors.

- [ ] **Step 4: Commit cleanup**

```bash
git add voice_runtime/providers.py pipeline_builder.py tests/test_providers.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py AGENTS.md ARCHITECTURE.md CONTEXT.md docs
git commit -m "chore: remove stale voice provider references"
```

Only include docs in the commit if they actually changed.

## Task 5: Final Verification

**Files:**
- No planned file changes.

- [ ] **Step 1: Run the relevant architecture and runtime suites**

Run from `server/`:

```bash
uv run pytest tests/test_providers.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py tests/test_voice_runtime_profiles.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full static checks**

Run from `server/`:

```bash
uv run ruff check .
uv run pyright .
```

Expected: ruff passes and pyright reports 0 errors.

- [ ] **Step 3: Run the full test suite if time allows**

Run from `server/`:

```bash
uv run pytest -q
```

Expected: PASS. If unrelated existing failures appear, record exact failing tests and confirm the focused provider/pipeline/import suites still pass.

- [ ] **Step 4: Final stale-reference check**

Run from repo root:

```bash
rg -n "from providers|import providers|patch\\(\"providers\\.|server/providers.py" server AGENTS.md ARCHITECTURE.md CONTEXT.md docs
```

Expected: no stale imports or patch targets. If architecture docs intentionally mention historical `providers.py`, verify the wording says it was legacy and has been moved.

## Self-Review

- Spec coverage: The plan moves Voice Providers into `voice_runtime`, keeps `pipeline_builder.py` as the only production caller, deletes the legacy app-root Module, and verifies import direction.
- Placeholder scan: Clean; no deferred implementation language remains.
- Type consistency: The new Module takes `STTProfile` and `TTSProfile`; existing `STTConfig` and `TTSConfig` are aliases from `config.py`, so callers do not need runtime changes beyond the import path.
