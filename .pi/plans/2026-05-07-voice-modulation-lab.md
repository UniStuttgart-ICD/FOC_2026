# Voice Modulation Lab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local Voice Mod Lab web interface that generates short TTS reference recordings, previews post-TTS voice effects, saves a per-profile Voice Modulation Preset, and applies the saved preset in the live Pipecat audio pipeline after TTS and before transport output.

**Architecture:** Create one deep module at `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/`. Keep DSP, local preset state, preview synthesis, FastAPI routes, and the static UI inside that module. Expose one runtime processor interface to Voice Runtime, then insert that processor between `tts` and `transport.output()` through the existing assembly seam.

**Tech Stack:** Python 3.10-3.12, FastAPI, Pipecat frame processors, NumPy for deterministic PCM16 DSP, standard-library `wave`/`io`/`json`, vanilla HTML/CSS/JS. No new audio library dependency in v1; avoid GPL/LGPL dependencies.

---

## Scope

- Create `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/`.
- Add a local ignored preset file at `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/state/voice_modulation_settings.json`.
- Add a local web app served by `uvicorn voice_modulation.app:app`.
- Add deterministic tests for settings, DSP, frame processing, app routes, config loading, assembly order, and pipeline wiring.
- Preserve existing provider voice selection. The module modifies audio after TTS; it does not replace provider TTS.
- Do not edit or revert `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/ARCHITECTURE.md`; it is already dirty from unrelated work.

## File Map

Create:

- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/__init__.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/settings.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/dsp.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/processor.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/preview.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/app.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_modulation/static/index.html`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_voice_modulation_settings.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_voice_modulation_dsp.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_voice_modulation_processor.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_voice_modulation_app.py`

Modify:

- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_runtime/profiles.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/config.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/voice_runtime/assembly.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/pipeline_builder.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_config.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_voice_runtime_assembly.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server/tests/test_pipeline_builder.py`
- `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/README.md`

---

## Execution Tasks

### 1. Preflight

- [ ] Inspect status and keep unrelated changes untouched.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git status --short
```

Expected output may include `M ARCHITECTURE.md`. Do not revert it.

- [ ] Run baseline tests for the files this plan will touch.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_config.py tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py -q
```

Expected output: all selected tests pass before the feature work starts.

### 2. Settings Schema And Local Presets

- [ ] Add red tests in `server/tests/test_voice_modulation_settings.py`.

Cover these cases:

- `default_settings_path(tmp_path)` returns `tmp_path / "state" / "voice_modulation_settings.json"`.
- Missing settings file returns the disabled clean preset.
- `save_profile_settings()` writes a profile-keyed JSON object.
- `load_profile_settings()` reads only the requested profile.
- `settings_from_mapping()` accepts the schema below and rejects out-of-range values with `VoiceModulationError`.
- `apply_saved_voice_modulation(profile, settings_path=path)` returns a `RuntimeProfile` with `voice_modulation` set.

Use this schema in the tests:

```python
{
    "enabled": True,
    "preset_name": "robot",
    "gain_db": 3.0,
    "wet_mix": 0.8,
    "low_cut_hz": 120.0,
    "high_cut_hz": 3600.0,
    "drive": 0.35,
    "bit_depth": 8,
    "ring_mod_hz": 45.0,
    "tremolo_hz": 5.0,
    "tremolo_depth": 0.4,
    "limiter": True,
}
```

- [ ] Implement `server/voice_modulation/settings.py`.

Required public interface:

```python
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

SETTINGS_ENV = "VOICE_MODULATION_SETTINGS_PATH"


class VoiceModulationError(ValueError):
    """Raised when voice modulation settings are invalid."""


@dataclass(frozen=True)
class VoiceModulationSettings:
    enabled: bool = False
    preset_name: str = "clean"
    gain_db: float = 0.0
    wet_mix: float = 1.0
    low_cut_hz: float = 0.0
    high_cut_hz: float = 0.0
    drive: float = 0.0
    bit_depth: int = 16
    ring_mod_hz: float = 0.0
    tremolo_hz: float = 0.0
    tremolo_depth: float = 0.0
    limiter: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

Built-in presets:

```python
BUILT_IN_PRESETS: dict[str, VoiceModulationSettings] = {
    "clean": VoiceModulationSettings(enabled=False, preset_name="clean"),
    "robot": VoiceModulationSettings(
        enabled=True,
        preset_name="robot",
        gain_db=2.0,
        wet_mix=0.9,
        low_cut_hz=120.0,
        high_cut_hz=5200.0,
        drive=0.25,
        bit_depth=9,
        ring_mod_hz=38.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        limiter=True,
    ),
    "radio": VoiceModulationSettings(
        enabled=True,
        preset_name="radio",
        gain_db=4.0,
        wet_mix=1.0,
        low_cut_hz=320.0,
        high_cut_hz=3200.0,
        drive=0.18,
        bit_depth=12,
        ring_mod_hz=0.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        limiter=True,
    ),
    "small_speaker": VoiceModulationSettings(
        enabled=True,
        preset_name="small_speaker",
        gain_db=1.5,
        wet_mix=0.85,
        low_cut_hz=220.0,
        high_cut_hz=4200.0,
        drive=0.1,
        bit_depth=13,
        ring_mod_hz=0.0,
        tremolo_hz=0.0,
        tremolo_depth=0.0,
        limiter=True,
    ),
    "low_battery": VoiceModulationSettings(
        enabled=True,
        preset_name="low_battery",
        gain_db=-1.0,
        wet_mix=0.95,
        low_cut_hz=80.0,
        high_cut_hz=2600.0,
        drive=0.32,
        bit_depth=7,
        ring_mod_hz=22.0,
        tremolo_hz=6.0,
        tremolo_depth=0.35,
        limiter=True,
    ),
}
```

Validation ranges:

- `gain_db`: `-24.0` to `24.0`
- `wet_mix`: `0.0` to `1.0`
- `low_cut_hz`: `0.0` to `4000.0`
- `high_cut_hz`: `0.0` to `24000.0`; `0.0` means disabled
- `drive`: `0.0` to `1.0`
- `bit_depth`: integer `4` to `16`
- `ring_mod_hz`: `0.0` to `2000.0`
- `tremolo_hz`: `0.0` to `20.0`
- `tremolo_depth`: `0.0` to `1.0`

Required function signatures:

- `settings_from_mapping(data: dict[str, Any]) -> VoiceModulationSettings`
- `default_settings_path(server_dir: Path | None = None) -> Path`
- `load_all_settings(path: str | Path | None = None) -> dict[str, VoiceModulationSettings]`
- `load_profile_settings(profile_name: str, *, server_dir: Path | None = None, settings_path: str | Path | None = None) -> VoiceModulationSettings`
- `save_profile_settings(profile_name: str, settings: VoiceModulationSettings, *, server_dir: Path | None = None, settings_path: str | Path | None = None) -> Path`
- `apply_saved_voice_modulation(profile, settings_path: str | Path | None = None)`

The `apply_saved_voice_modulation` body should be:

```python
def apply_saved_voice_modulation(profile, settings_path: str | Path | None = None):
    settings = load_profile_settings(
        profile.profile_name,
        server_dir=profile.server_dir,
        settings_path=settings_path,
    )
    return replace(profile, voice_modulation=settings)
```

The JSON shape on disk is:

```json
{
  "profiles": {
    "hybrid_low_latency": {
      "enabled": true,
      "preset_name": "robot",
      "gain_db": 2.0,
      "wet_mix": 0.9,
      "low_cut_hz": 120.0,
      "high_cut_hz": 5200.0,
      "drive": 0.25,
      "bit_depth": 9,
      "ring_mod_hz": 38.0,
      "tremolo_hz": 0.0,
      "tremolo_depth": 0.0,
      "limiter": true
    }
  }
}
```

- [ ] Modify `server/voice_runtime/profiles.py`.

Add a defaulted field to `RuntimeProfile`:

```python
voice_modulation: Any | None = None
```

Place it after `process_trace` and before `server_dir` so the runtime-local options stay grouped.

- [ ] Modify `server/config.py`.

Add import:

```python
from voice_modulation.settings import VoiceModulationError, apply_saved_voice_modulation
```

Add alias:

```python
VoiceModulationConfig = object
```

Add field to `RuntimeConfig`:

```python
voice_modulation: object | None
```

Pass the field through `RuntimeConfig.from_profile()` and `RuntimeConfig.required_env_names()`.

Apply settings after wake tuning:

```python
profile = apply_saved_wake_tuning(profile)
profile = apply_saved_voice_modulation(profile)
```

Catch `VoiceModulationError` beside `WakeTuningError` and raise `ConfigError(str(exc))`.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_settings.py tests/test_config.py tests/test_voice_runtime_profiles.py -q
```

Expected output: all selected tests pass.

### 3. PCM16 DSP Core

- [ ] Add red tests in `server/tests/test_voice_modulation_dsp.py`.

Use deterministic sine fixtures:

```python
import numpy as np

def sine_pcm16(sample_rate: int = 24000, hz: float = 440.0, seconds: float = 0.1) -> bytes:
    t = np.arange(int(sample_rate * seconds), dtype=np.float32) / sample_rate
    samples = np.sin(2.0 * np.pi * hz * t) * 0.25
    return np.asarray(samples * 32767.0, dtype=np.int16).tobytes()
```

Cover:

- Disabled settings return exact original bytes.
- `wet_mix=0.0` returns exact original bytes.
- `gain_db=6.0` increases RMS.
- `bit_depth=4` reduces the number of unique sample values.
- `ring_mod_hz=40.0` changes bytes and preserves length.
- `tremolo_hz=5.0, tremolo_depth=0.5` changes bytes and preserves length.
- `low_cut_hz=300.0, high_cut_hz=3000.0` changes bytes and preserves length.
- Invalid channel or byte alignment raises `VoiceModulationDspError`.

- [ ] Implement `server/voice_modulation/dsp.py`.

Required public interface:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from voice_modulation.settings import VoiceModulationSettings


class VoiceModulationDspError(ValueError):
    """Raised when PCM audio cannot be processed."""


@dataclass
class VoiceModulationState:
    ring_phase: float = 0.0
    tremolo_phase: float = 0.0
    low_cut_last: NDArray[np.float32] | None = None
    high_cut_last: NDArray[np.float32] | None = None
    high_cut_prev_input: NDArray[np.float32] | None = None

    def reset(self) -> None:
        self.ring_phase = 0.0
        self.tremolo_phase = 0.0
        self.low_cut_last = None
        self.high_cut_last = None
        self.high_cut_prev_input = None
```

Add public functions with these signatures:

- `process_pcm16(audio: bytes, *, sample_rate: int, num_channels: int, settings: VoiceModulationSettings, state: VoiceModulationState | None = None) -> bytes`
- `pcm16_rms(audio: bytes) -> float`

Processing order:

1. Validate `num_channels >= 1`, `sample_rate > 0`, and `len(audio) % (2 * num_channels) == 0`.
2. Return `audio` unchanged when `settings.enabled is False` or `settings.wet_mix == 0.0`.
3. Convert interleaved PCM16 to `float32` frames in `[-1.0, 1.0]`.
4. Keep a dry copy.
5. Apply gain.
6. Apply one-pole high-pass when `low_cut_hz > 0.0`.
7. Apply one-pole low-pass when `high_cut_hz > 0.0`.
8. Apply drive with `np.tanh(samples * (1.0 + drive * 8.0)) / np.tanh(1.0 + drive * 8.0)`.
9. Apply bit crush when `bit_depth < 16`.
10. Apply ring modulation when `ring_mod_hz > 0.0`.
11. Apply tremolo when `tremolo_hz > 0.0 and tremolo_depth > 0.0`.
12. Mix dry/wet with `settings.wet_mix`.
13. Apply limiter with `np.clip(samples, -0.98, 0.98)` when enabled; otherwise clip to `[-1.0, 1.0]`.
14. Return interleaved PCM16 bytes.

Keep stateful oscillator phase across chunks so the runtime processor has no clicks between adjacent frames.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_dsp.py -q
```

Expected output: all DSP tests pass.

### 4. Post-TTS Frame Processor

- [ ] Add red tests in `server/tests/test_voice_modulation_processor.py`.

Use a capture subclass:

```python
class CapturingVoiceModulationProcessor(VoiceModulationProcessor):
    def __init__(self, settings):
        super().__init__(settings)
        self.pushed = []

    async def push_frame(self, frame, direction):
        self.pushed.append((frame, direction))
```

Cover:

- Non-audio frames are pushed unchanged.
- `TTSAudioRawFrame` is replaced with a new `TTSAudioRawFrame` whose metadata is unchanged and audio length is unchanged.
- Disabled settings push the original `TTSAudioRawFrame` object unchanged.
- `TTSStoppedFrame` resets DSP state and is pushed unchanged.

- [ ] Implement `server/voice_modulation/processor.py`.

Required behavior:

```python
from __future__ import annotations

from dataclasses import replace
from typing import Any

from pipecat.frames.frames import CancelFrame, EndFrame, Frame, TTSAudioRawFrame, TTSStoppedFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from voice_modulation.dsp import VoiceModulationState, process_pcm16
from voice_modulation.settings import VoiceModulationSettings


class VoiceModulationProcessor(FrameProcessor):
    def __init__(self, settings: VoiceModulationSettings, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._settings = settings
        self._state = VoiceModulationState()

    @property
    def settings(self) -> VoiceModulationSettings:
        return self._settings

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, TTSAudioRawFrame):
            if not self._settings.enabled:
                await self.push_frame(frame, direction)
                return
            audio = process_pcm16(
                frame.audio,
                sample_rate=frame.sample_rate,
                num_channels=frame.num_channels,
                settings=self._settings,
                state=self._state,
            )
            await self.push_frame(replace(frame, audio=audio), direction)
            return

        if isinstance(frame, (TTSStoppedFrame, CancelFrame, EndFrame)):
            self._state.reset()

        await self.push_frame(frame, direction)
```

Only transform `TTSAudioRawFrame` in v1. That keeps the processor specific to provider speech and avoids changing arbitrary output audio frames.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_processor.py -q
```

Expected output: all processor tests pass.

### 5. Runtime Assembly And Pipeline Wiring

- [ ] Update `server/tests/test_voice_runtime_assembly.py`.

Add `voice_modulation="voice_modulation"` to the first test and assert the order:

```python
[
    "transport.input",
    "wake.audio",
    "stt",
    "wake.transcript",
    "user_aggregator",
    "agent_turn",
    "tts",
    "voice_modulation",
    "transport.output",
    "assistant_aggregator",
]
```

Add `voice_modulation=None` to the no-wake test and keep the old order.

- [ ] Modify `server/voice_runtime/assembly.py`.

Add field:

```python
voice_modulation: object | None
```

Insert it after `parts.tts` and before `parts.transport_output`:

```python
processors.extend([
    parts.user_aggregator,
    parts.agent_turn,
    parts.tts,
])
if parts.voice_modulation is not None:
    processors.append(parts.voice_modulation)
processors.extend([
    parts.transport_output,
    parts.assistant_aggregator,
])
```

- [ ] Update `server/tests/test_pipeline_builder.py`.

Add a fake enabled settings object to `_config()`:

```python
from voice_modulation.settings import VoiceModulationSettings
```

Extend `_config()` with `voice_modulation: object | None = None` and pass it into `RuntimeConfig`.

Add a test:

```python
def test_voice_modulation_is_inserted_between_tts_and_transport_output(monkeypatch, tmp_path: Path):
    tts = FrameProcessor()
    transport_output = FrameProcessor()
    mod = FrameProcessor()
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr("pipeline_builder.create_tts_service", lambda config: tts)
    monkeypatch.setattr("pipeline_builder._create_voice_modulation_processor", lambda settings: mod)

    class FakeTransportWithStableOutput(FakeTransport):
        def output(self):
            return transport_output

    built = build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            voice_modulation=VoiceModulationSettings(enabled=True, preset_name="robot"),
        ),
        cast(BaseTransport, FakeTransportWithStableOutput()),
    )
    processors = cast(FakePipeline, built.pipeline).processors
    assert processors[processors.index(tts) + 1] is mod
    assert processors[processors.index(mod) + 1] is transport_output
```

Add another test that disabled settings do not insert the processor.

- [ ] Modify `server/pipeline_builder.py`.

Import:

```python
from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings
```

Add helper:

```python
def _create_voice_modulation_processor(
    settings: object | None,
) -> VoiceModulationProcessor | None:
    if not isinstance(settings, VoiceModulationSettings):
        return None
    if not settings.enabled:
        return None
    return VoiceModulationProcessor(settings)
```

In `build_pipeline()`, create:

```python
voice_modulation = _create_voice_modulation_processor(config.voice_modulation)
```

Pass it to `VoiceRuntimeParts` with the new `voice_modulation=voice_modulation` argument included.

Add `voice_modulation: VoiceModulationProcessor | None` to `BuiltPipeline` only if tests need direct access. Prefer leaving `BuiltPipeline` unchanged unless asserting through the processor order is not enough.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py -q
```

Expected output: all selected tests pass.

### 6. Preview Synthesis And WAV Helpers

- [ ] Add preview tests in `server/tests/test_voice_modulation_app.py` or a separate section in `server/tests/test_voice_modulation_dsp.py`.

Cover:

- `pcm16_to_wav_bytes()` returns bytes beginning with `RIFF`.
- `wav_bytes_to_pcm16()` round-trips mono PCM16 metadata.
- `collect_tts_audio()` concatenates only `TTSAudioRawFrame` bytes and raises `VoicePreviewError` on `ErrorFrame`.
- Preview service selection uses HTTP TTS services for Cartesia and Deepgram and normal services for OpenAI and Kokoro.

- [ ] Implement `server/voice_modulation/preview.py`.

Required public interface:

```python
from __future__ import annotations

import base64
import io
import os
import uuid
import wave
from dataclasses import dataclass
from typing import AsyncIterable, Protocol

import aiohttp
from pipecat.frames.frames import EndFrame, ErrorFrame, StartFrame, TTSAudioRawFrame
from pipecat.services.cartesia.tts import CartesiaHttpTTSService
from pipecat.services.deepgram.tts import DeepgramHttpTTSService
from pipecat.services.kokoro.tts import KokoroTTSService
from pipecat.services.openai.tts import OpenAITTSService

from voice_modulation.dsp import process_pcm16
from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.providers import DEFAULT_CARTESIA_VOICE_ID
from voice_runtime.profiles import TTSProfile
```

Data structures:

```python
class VoicePreviewError(ValueError):
    """Raised when a preview recording cannot be generated."""


@dataclass(frozen=True)
class AudioBytes:
    audio: bytes
    sample_rate: int
    num_channels: int


@dataclass(frozen=True)
class PreviewAudio:
    wav_base64: str
    sample_rate: int
    num_channels: int
```

Function signatures:

- `pcm16_to_wav_bytes(audio: bytes, *, sample_rate: int, num_channels: int) -> bytes`
- `wav_bytes_to_pcm16(wav_bytes: bytes) -> AudioBytes`
- `encode_preview(audio: bytes, *, sample_rate: int, num_channels: int) -> PreviewAudio`
- `decode_preview(wav_base64: str) -> AudioBytes`
- `synthesize_tts_reference(tts: TTSProfile, text: str) -> AudioBytes`
- `render_effect_preview(source: AudioBytes, settings: VoiceModulationSettings) -> AudioBytes`

Provider service selection:

```python
async def _build_preview_service(tts: TTSProfile, session: aiohttp.ClientSession | None):
    if tts.provider == "openai":
        return OpenAITTSService(
            api_key=os.environ["OPENAI_API_KEY"],
            settings=OpenAITTSService.Settings(
                model=tts.model or "gpt-4o-mini-tts",
                voice=tts.voice or "coral",
            ),
        )
    if tts.provider == "cartesia":
        return CartesiaHttpTTSService(
            api_key=os.environ["CARTESIA_API_KEY"],
            settings=CartesiaHttpTTSService.Settings(
                model=tts.model or "sonic-3",
                voice=tts.voice or os.getenv("CARTESIA_VOICE_ID") or DEFAULT_CARTESIA_VOICE_ID,
            ),
            sample_rate=24000,
            aiohttp_session=session,
        )
    if tts.provider == "deepgram":
        if session is None:
            raise VoicePreviewError("Deepgram preview requires an aiohttp session")
        return DeepgramHttpTTSService(
            api_key=os.environ["DEEPGRAM_API_KEY"],
            settings=DeepgramHttpTTSService.Settings(
                model=tts.model or "aura-2",
                voice=tts.voice or "aura-2-andromeda-en",
            ),
            sample_rate=24000,
            aiohttp_session=session,
        )
    if tts.provider == "kokoro":
        return KokoroTTSService(
            settings=KokoroTTSService.Settings(voice=tts.voice or "af_heart"),
        )
    raise VoicePreviewError(f"Unsupported TTS provider: {tts.provider}")
```

Start services with:

```python
await service.start(StartFrame(audio_out_sample_rate=24000))
```

Then collect `TTSAudioRawFrame` chunks from:

```python
async for frame in service.run_tts(text, str(uuid.uuid4())):
    frames.append(frame)
```

Stop services with:

```python
await service.stop(EndFrame(reason="voice_modulation_preview"))
```

When env vars are missing, raise `VoicePreviewError` with the missing variable name.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_app.py tests/test_voice_modulation_dsp.py -q
```

Expected output: all selected tests pass.

### 7. FastAPI App

- [ ] Add route tests in `server/tests/test_voice_modulation_app.py`.

Use `fastapi.testclient.TestClient` and `create_app(server_dir=tmp_path, preview_synthesizer=fake)`.

Cover:

- `GET /` returns HTML.
- `GET /api/presets` returns built-in preset names.
- `GET /api/profiles` returns runtime profile names and TTS metadata from a temporary `runtime_profiles.toml`.
- `GET /api/settings/{profile_name}` returns default settings when no local state exists.
- `POST /api/settings/{profile_name}` saves validated settings.
- `POST /api/preview/effect` accepts a base64 WAV and settings, returns a base64 WAV.
- `POST /api/preview/tts` uses the injected fake synthesizer and returns clean plus modulated WAVs.
- Invalid settings return HTTP 400.
- Missing profile returns HTTP 404.

- [ ] Implement `server/voice_modulation/app.py`.

Required public interface:

```python
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Awaitable, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from voice_modulation.preview import (
    AudioBytes,
    VoicePreviewError,
    decode_preview,
    encode_preview,
    render_effect_preview,
    synthesize_tts_reference,
)
from voice_modulation.settings import (
    BUILT_IN_PRESETS,
    VoiceModulationError,
    default_settings_path,
    load_profile_settings,
    save_profile_settings,
    settings_from_mapping,
)
from voice_runtime.profiles import default_profiles_path, load_runtime_profile
```

Models:

```python
class SettingsPayload(BaseModel):
    settings: dict[str, Any]


class EffectPreviewPayload(BaseModel):
    wav_base64: str
    settings: dict[str, Any]


class TtsPreviewPayload(BaseModel):
    profile_name: str
    text: str
    settings: dict[str, Any]
```

Routes:

- `GET /`
- `GET /api/presets`
- `GET /api/profiles`
- `GET /api/settings/{profile_name}`
- `POST /api/settings/{profile_name}`
- `POST /api/preview/effect`
- `POST /api/preview/tts`

`GET /api/profiles` should parse `runtime_profiles.toml`, load each profile through `load_runtime_profile()`, and return:

```json
{
  "profiles": [
    {
      "name": "hybrid_low_latency",
      "category": "benchmark_streaming",
      "tts": {
        "provider": "cartesia",
        "model": "sonic-3",
        "voice": "47c38ca4-5f35-497b-b1a3-415245fb35e1"
      },
      "missing_env": []
    }
  ]
}
```

Use `profile.required_env_names()` and `os.getenv()` to populate `missing_env`. The route must still return profiles when env vars are missing so the UI can explain why a Generate action is unavailable.

- [ ] Run focused verification.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_app.py -q
```

Expected output: all app route tests pass.

### 8. Static HTML Interface

- [ ] Create `server/voice_modulation/static/index.html`.

Build a complete vanilla interface with:

- A top toolbar with profile select, provider/model/voice summary, and save status.
- A reference phrase textarea with three default phrase buttons:
  - `Status`
  - `Motion`
  - `Longer`
- A Generate button that calls `/api/preview/tts`.
- Two audio players labeled `Clean` and `Modulated`.
- Preset buttons for `clean`, `robot`, `radio`, `small_speaker`, and `low_battery`.
- Sliders for gain, wet mix, low cut, high cut, drive, bit depth, ring modulation, tremolo rate, and tremolo depth.
- A limiter checkbox.
- A Save button that calls `/api/settings/{profile_name}`.
- A compact status region for provider credentials, preview errors, and save success.

Use stable dimensions:

- Sliders use a two-column grid on desktop and one column on mobile.
- Audio players have fixed-width containers with responsive max-width.
- Buttons do not resize when status text changes.
- Text labels stay inside their controls at widths down to 360px.

Use this endpoint contract in JS:

```javascript
async function apiJson(path, options = {}) {
  const request = Object.assign({
    headers: { "Content-Type": "application/json" },
  }, options);
  const response = await fetch(path, request);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.detail || `Request failed: ${response.status}`);
  }
  return data;
}
```

Use `URL.createObjectURL()` for WAV playback. Keep the last clean WAV base64 in memory so slider changes can call `/api/preview/effect` without another TTS call.

Visual direction:

- Workbench, not landing page.
- Palette: graphite text, off-white surface, teal controls, coral active accents, amber warnings.
- No gradient-orb decoration.
- Cards only for the repeated audio panels and preset tiles.
- No tutorial copy; use concise labels and state text.

- [ ] Verify static app route still passes tests.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_app.py -q
```

Expected output: all app tests pass.

### 9. README Usage

- [ ] Add a concise section to `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/README.md`.

Content:

````markdown
## Voice Mod Lab

Run the local voice modulation workbench:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897
```

Open `http://127.0.0.1:8897`.

The lab saves per-profile presets to `server/state/voice_modulation_settings.json`. The live voice runtime loads the saved preset and applies it after TTS, before transport output.
````

- [ ] Run docs-adjacent check.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git diff --check
```

Expected output: no whitespace errors.

### 10. Full Automated Verification

- [ ] Run all focused Python tests.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_modulation_settings.py tests/test_voice_modulation_dsp.py tests/test_voice_modulation_processor.py tests/test_voice_modulation_app.py tests/test_config.py tests/test_voice_runtime_profiles.py tests/test_voice_runtime_assembly.py tests/test_pipeline_builder.py -q
```

Expected output: all selected tests pass.

- [ ] Run lints and type checks.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run ruff check .
uv run pyright
```

Expected output:

- Ruff: `All checks passed!`
- Pyright: `0 errors`

### 11. Browser Verification

- [ ] Start the app.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897
```

Expected output includes:

```text
Uvicorn running on http://127.0.0.1:8897
```

- [ ] Verify with `playwright-cli`.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat
playwright-cli --help
playwright-cli open http://127.0.0.1:8897
playwright-cli snapshot
playwright-cli click "text=Robot"
playwright-cli fill "textarea" "Voice modulation online. Safety checks nominal."
playwright-cli click "text=Save"
playwright-cli snapshot
playwright-cli screenshot
```

Expected evidence:

- Snapshot includes `Voice Mod Lab`, profile selector, preset controls, sliders, `Clean`, `Modulated`, and `Save`.
- After save, status text reports the preset was saved.
- Screenshot path is printed by `playwright-cli screenshot`; include that path in the completion note.

If an API key is not present, do not fail browser verification on TTS generation. Confirm the UI shows the missing env var and that preset saving still works.

### 12. Manual Live Smoke

- [ ] Save an enabled preset through the UI.
- [ ] Start the normal bot with the same runtime profile.
- [ ] Confirm the pipeline includes `VoiceModulationProcessor` between TTS and transport output by adding a temporary debug log only during local smoke, or by asserting through a focused test. Remove any temporary log before finalizing.
- [ ] Speak to the bot and confirm assistant speech is audibly modulated.

If live credentials or hardware are unavailable, record that limitation and rely on the automated processor and pipeline tests.

### 13. Final Review

- [ ] Inspect the final diff.

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git diff -- server/voice_modulation server/voice_runtime server/config.py server/pipeline_builder.py server/tests README.md
git status --short
```

Check:

- `ARCHITECTURE.md` remains untouched by this work.
- No generated WAVs or local state files are tracked.
- `server/state/` remains ignored by `.gitignore`.
- Voice Modulation code owns DSP and preview behavior; Voice Runtime only sees an optional processor.
- Runtime order is `TTS -> VoiceModulationProcessor -> transport.output()`.

---

## Parallelization

After Task 2 lands, these slices can run in parallel with disjoint write scopes:

- DSP worker: `server/voice_modulation/dsp.py`, `server/tests/test_voice_modulation_dsp.py`.
- Processor and runtime worker: `server/voice_modulation/processor.py`, `server/voice_runtime/assembly.py`, `server/pipeline_builder.py`, related tests.
- App/UI worker: `server/voice_modulation/preview.py`, `server/voice_modulation/app.py`, `server/voice_modulation/static/index.html`, app tests.

The App/UI worker depends on the settings schema from Task 2. The runtime worker depends on the processor public class from Task 4, but can add assembly tests first.

## Completion Criteria

- Saved presets are local, profile-keyed, validated, and ignored by git.
- The UI can load profiles, preview built-in presets, adjust sliders, save settings, and show missing provider credentials.
- Automated tests prove DSP determinism, frame metadata preservation, config loading, and processor order.
- Browser verification proves the page renders and basic UI actions work.
- No new GPL/LGPL dependency is introduced.
