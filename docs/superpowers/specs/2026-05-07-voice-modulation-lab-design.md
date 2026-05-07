# Voice Modulation Lab Design

## Summary

Add a `voice_modulation` module that lets Pi tune a provider-agnostic post-TTS voice effect through a local web app. The app generates short TTS reference recordings, previews clean versus modulated audio, saves a **Voice Modulation Preset**, and the bot applies that preset with a Pipecat `FrameProcessor` after TTS and before transport output.

## Goals

- Tune robot/radio/synthetic voice effects through a usable HTML interface.
- Generate reference recordings from the configured TTS providers.
- Save local preset state without editing `server/runtime_profiles.toml`.
- Apply the saved preset in the live Voice Runtime path.
- Keep default tests deterministic and free of live provider credentials.

## Non-Goals

- No full DAW, waveform editor, or multitrack soundboard in v1.
- No pitch/formant library in v1 unless a permissive integration is trivial.
- No committed generated TTS audio.
- No Agent Control or Robot Control changes.

## Architecture

`voice_modulation` is a deep Voice Runtime module. Its interface is small:

- `VoiceModulationSettings`: validated preset values.
- `load_profile_settings()` / `save_profile_settings()`: local state persistence.
- `apply_saved_voice_modulation()`: applies saved local state to a runtime profile.
- `VoiceModulationProcessor`: transforms output audio frames.

The implementation hides JSON shape, DSP math, preview generation, and UI endpoints. This gives callers leverage: the bot only asks whether a profile has a preset and inserts one processor.

Runtime order:

```text
transport.input()
-> wake/STT/user aggregation
-> Agent Turn
-> TTS
-> VoiceModulationProcessor
-> transport.output()
-> assistant aggregation
```

The processor transforms `TTSAudioRawFrame` / `OutputAudioRawFrame` PCM and passes every other frame through unchanged. It resets effect state on interruption and TTS context changes.

## Module Layout

```text
server/voice_modulation/
  __init__.py
  app.py
  dsp.py
  preview.py
  processor.py
  settings.py
  static/index.html
```

`settings.py` mirrors the `wake_tuning` local-state pattern and writes to `server/state/voice_modulation_settings.json` by default. An environment variable can override the path for tests and local experiments.

`dsp.py` uses Python and NumPy only for v1. That keeps licensing simple and avoids adding GPL/LGPL audio libraries before the runtime value is proven. Future pitch/formant work can use a permissive library such as Signalsmith Stretch behind the same processor interface.

## Preset Fields

V1 presets use bounded numeric fields:

- `enabled`
- `preset_name`
- `gain_db`
- `wet_mix`
- `low_cut_hz`
- `high_cut_hz`
- `drive`
- `bit_depth`
- `ring_mod_hz`
- `tremolo_hz`
- `tremolo_depth`
- `limiter`

Built-in presets:

- `clean`
- `robot`
- `radio`
- `small_speaker`
- `low_battery`

The saved preset is profile-scoped. Missing settings mean `clean`.

## Voice Mod Lab

The HTML app is a real tool, not a landing page. It should feel like a compact industrial synth panel for repeated tuning:

- Profile/provider selector.
- Voice/model summary from the Runtime Profile.
- Prompt lines for sample generation.
- Generate buttons for OpenAI, Cartesia, Deepgram, and Kokoro when configured.
- Clean and modulated audio players.
- Sliders with immediate browser-side preview.
- Save / implement button that writes the active preset.
- Clear unavailable-provider messages when credentials are missing.

Preview generation calls provider adapters from the server and returns short WAV audio. Generated clips are not committed. The app may keep in-memory clips during the session; persistent caching is out of scope for v1.

## Provider Handling

Preview generation supports the providers already in the repo:

- `openai`
- `cartesia`
- `deepgram`
- `kokoro`

Missing API keys or unsupported profile providers are reported as UI errors. They do not block editing existing presets.

Provider-native controls remain outside v1 except for the existing model/voice fields. Exact modulation comes from post-TTS DSP so presets behave similarly across providers.

## Runtime Integration

`RuntimeProfile` gains optional voice-modulation settings after local overrides are applied. `pipeline_builder.py` creates a `VoiceModulationProcessor` only when a preset is enabled, then passes it into `VoiceRuntimeParts`.

`VoiceRuntimeParts` gains an optional `voice_modulation` slot placed after `tts`.

This keeps processor ordering inside **Voice Runtime Assembly** and keeps `pipeline_builder.py` as the composition root.

## Error Handling

- Invalid preset JSON raises a clear `VoiceModulationError`.
- Out-of-range slider values are rejected before save.
- Non-audio frames pass through unchanged.
- Unsupported channel counts pass through unchanged in v1.
- DSP exceptions fail closed by pushing original audio and logging the error once per session.
- Missing provider credentials return a 400 preview error with the missing env var name.

## Testing

Default tests stay deterministic:

- settings validation, load, save, and env override path.
- DSP transforms on synthetic PCM without live providers.
- processor passes non-audio frames through.
- processor transforms `TTSAudioRawFrame` and preserves sample rate/channel metadata.
- assembly order inserts `voice_modulation` after `tts`.
- app page and settings APIs load.
- preview endpoint uses fake provider adapters.

Manual browser verification should use `playwright-cli` after implementation because the feature has a browser-visible UI.

## Architecture Notes

This design deepens the Voice Runtime module instead of scattering effect logic through provider construction, transport output, or observers. Deleting `voice_modulation` would force preset parsing, DSP, preview generation, and post-TTS frame mutation back into several callers, so the module earns its seam.

The interface is the test surface. Tests should prove preset behavior through `VoiceModulationSettings`, `VoiceModulationProcessor`, and app endpoints, not by reaching into DSP internals unless a math function has a direct invariant.
