# Modular Low-Latency Voice Runtime Design

## Goal

Add a modular, profile-driven voice runtime to `pipecat-agent` for low-latency robot control.

Default runtime:

```text
Mave wake word → Deepgram Flux STT → OpenAI Codex OAuth agent → Cartesia Sonic TTS
```

The system must remain modular, DRY, and Pipecat-native. It should support local debug profiles, benchmarking metrics, the trained `mave.onnx` wake model from `DF2025_CLEAN`, and a separate OpenAI Codex OAuth agent-provider workstream.

## Key Decisions

- Default profile: `hybrid_low_latency`.
- Missing required API keys fail startup with clear setup instructions.
- Local profiles remain available for debugging and offline baseline testing.
- Wake mode: single-command mode. Every normal command requires `mave`.
- Same-utterance wake is primary: `Mave, move up a bit`.
- Two-step wake remains fallback: `Mave` → speak command.
- Wake pre-buffer: `1.5s` rolling audio buffer.
- Copy trained model to `server/models/mave.onnx`.
- Wake can be disabled only through explicit debug profiles.
- Metrics go to console and JSONL.
- JSONL includes transcript and response text for debugging.
- Emergency `stop` bypasses the wake word for safety.
- Emergency stop requires a separate local stop detector/model; fail fast if enabled without one.
- OpenAI Codex OAuth migration should be a separate but coordinated issue using the same runtime config.

## Architecture

Keep Pipecat as the pipeline framework. Add only thin, repo-specific configuration and construction layers.

```text
runtime profile
→ validate config/env/model files
→ create wake/STT/agent/TTS/metrics components
→ assemble Pipecat pipeline
→ run SmallWebRTC bot
```

Target pipeline:

```text
SmallWebRTC input
→ MaveWakeWordGate
→ STT service
→ user aggregator
→ agent processor
→ TTS service
→ SmallWebRTC output
→ assistant aggregator
```

## Module Boundaries

```text
server/
  bot.py                    # entrypoint only: args/env → transport → runner
  config.py                 # CLI/env/TOML parsing, typed config, validation
  runtime_profiles.toml     # named stack profiles
  providers.py              # create_stt_service(), create_tts_service()
  agent_processor_factory.py# create Claude/OpenAI agent processor
  pipeline_builder.py       # assemble Pipecat Pipeline/PipelineTask
  metrics.py                # console + JSONL turn metrics

  wake/
    openwakeword_detector.py # OpenWakeWord wrapper
    wake_gate.py             # Pipecat FrameProcessor before STT
    emergency_stop.py        # emergency stop detector interface
    transcript_cleanup.py    # strip leading wake phrase

  claude_agent_processor.py  # existing Claude SDK + robot MCP processor
  openai_codex_agent_processor.py # OpenAI Codex OAuth agent processor
  codex_auth.py              # Pi auth.json OAuth reader/refresher
  prompts.py                 # robot behavior prompt
  models/mave.onnx           # trained Mave wake model
```

Orthogonality rules:

- Profiles choose component config; they do not construct components.
- Provider factories construct STT/TTS only; they know nothing about wake, metrics, or robot tools.
- Wake gate handles raw-audio activation only.
- Emergency stop handles only stop bypass.
- Agent processors handle robot reasoning/tool use only.
- Metrics observe/record; they do not alter pipeline behavior.
- `bot.py` stays thin and contains no provider-specific branches.

Avoid:

- A generic plugin framework.
- Abstract provider class trees too early.
- One giant `AppContext` object.
- Wake logic inside STT or agent processors.
- Metrics as a frame-transforming processor unless needed.
- Silent fallback from low-latency to local profiles.
- Raw audio logging.

## Runtime Profiles

Initial profiles:

```text
hybrid_low_latency   benchmark_streaming  mave + Deepgram Flux + OpenAI Codex + Cartesia
openai_all           benchmark_streaming  mave + OpenAI Realtime STT + OpenAI Codex + OpenAI TTS
deepgram_all         benchmark_streaming  mave + Deepgram Flux + OpenAI Codex + Deepgram Aura
local_current        local_debug          mave + Whisper + OpenAI/Claude selectable agent + Kokoro
no_wake_debug        local_debug          no wake + Whisper + OpenAI/Claude selectable agent + Kokoro
```

Profile selection:

```text
CLI --profile wins
else VOICE_PROFILE env var
else hybrid_low_latency
```

Benchmark profiles must use streaming-capable STT/TTS providers. Local profiles are allowed but labelled as debug/baseline.

## Wake Gate Design

Normal path:

```text
asleep
→ OpenWakeWord detects mave
→ replay 1.5s pre-buffer + current/post-wake audio
→ stream one command to STT
→ strip leading wake phrase from transcript
→ process command
→ return asleep
```

The existing `DF2025_CLEAN` code is the source reference:

- `langgraph_system/audio/vad.py`
- `langgraph_system/audio/vad_processor.py`
- `models/mave.onnx`
- `tests/test_vad_processor.py`
- `tests/test_vad_keep_warm.py`

Important change from `DF2025_CLEAN`: do not clear the pre-buffer on wake for Pipecat. Include enough buffered audio to support `Mave, command` same-utterance usage.

Wake configuration defaults:

```text
provider = openwakeword
model_path = models/mave.onnx
threshold = 0.5
candidate_log_threshold = 0.3
pre_buffer_s = 1.5
single_command = true
```

## Emergency Stop Design

Emergency path:

```text
asleep or awake
→ local emergency detector hears stop/emergency stop/halt
→ bypass normal wake requirement
→ inject/send “stop” as the command
→ agent/robot path invokes stop tool
→ return asleep after response/cooldown
```

`mave.onnx` cannot detect `stop`. Initial emergency-stop implementation must expose the interface and validate config. If emergency stop is enabled without a local stop model/provider, startup fails clearly.

## Agent Provider / OpenAI Codex OAuth

Fold `.pi/plans/2026-05-04-openai-codex-oauth-provider.md` into this project as a separate issue, but use the same runtime config system.

Do not create a second unrelated `agent_config.py` path. Instead:

```text
config.py RuntimeConfig
→ AgentConfig(provider="openai_codex_oauth", model="gpt-5.5", ...)
→ agent_processor_factory.py
→ OpenAICodexAgentProcessor
```

The OpenAI workstream owns:

- `codex_auth.py`
- `openai_codex_agent_processor.py`
- `agent_processor_factory.py`
- OpenAI/Codex auth tests
- factory tests

It does not own wake, STT/TTS providers, metrics, or pipeline assembly.

Claude remains as a selectable fallback provider until OpenAI is live-validated.

## Metrics Design

Metrics are turn-level, not audio-frame-level.

Output:

- console summary per turn
- append JSONL at `server/logs/voice_metrics.jsonl`

Record includes:

- timestamp
- profile/category
- turn id
- wake phrase
- transcript
- response
- wake latency
- speech duration
- STT latency
- agent latency
- TTS time-to-first-byte when available
- TTS total latency
- total-to-first-audio
- total turn time

Metrics write failure should log a warning and disable metrics, not crash an active robot session.

## Error Handling

- Unsupported provider: fail startup clearly.
- Missing API key for default profile: fail startup clearly.
- Missing `models/mave.onnx` when wake is enabled: fail startup clearly.
- Emergency stop enabled without stop model/provider: fail startup clearly.
- Metrics write failure: warn and disable metrics.
- MCP unavailable: agent returns a user-facing robot-control-server error.
- No silent provider fallback in benchmark profiles.

## Parallel Implementation Issues

1. **Runtime config foundation**
   - `config.py`, `runtime_profiles.toml`, `tests/test_config.py`
   - Defines `RuntimeConfig`, `WakeConfig`, `EmergencyStopConfig`, `STTConfig`, `TTSConfig`, `AgentConfig`, `MCPConfig`, `MetricsConfig`.

2. **Provider factories**
   - `providers.py`, `tests/test_providers.py`
   - STT: `deepgram_flux`, `openai_realtime`, `whisper`
   - TTS: `cartesia`, `openai`, `deepgram`, `kokoro`

3. **Agent provider factory + OpenAI Codex OAuth**
   - Adapt existing OpenAI Codex OAuth plan into unified config.
   - `codex_auth.py`, `openai_codex_agent_processor.py`, `agent_processor_factory.py`, tests.

4. **OpenWakeWord Mave wake gate**
   - `wake/openwakeword_detector.py`, `wake/wake_gate.py`, `wake/transcript_cleanup.py`, `server/models/mave.onnx`, tests.

5. **Emergency stop bypass**
   - `wake/emergency_stop.py`, tests.
   - Interface and validation first; real stop model/provider can follow.

6. **Metrics**
   - `metrics.py`, tests.
   - Console + JSONL turn metrics.

7. **Pipeline builder + bot.py slimming**
   - `pipeline_builder.py`, `bot.py`.
   - Wires outputs from issues 1–6.

8. **Docs and benchmark guide**
   - `README.md`, `server/.env.example`, `docs/benchmarking.md`.

Issues 2–6 can run in parallel after Issue 1 defines shared contracts. Issue 7 depends on the others.

## Testing Strategy

- Unit tests for config loading/validation.
- Unit tests for provider factory selection and missing env handling.
- Unit tests for wake-gate state machine and transcript cleanup.
- Unit tests for emergency-stop config validation.
- Unit tests for metrics JSONL schema.
- Factory tests for Claude/OpenAI processor selection.
- Smoke tests with wake disabled.
- Integration test with `mave.onnx` enabled.
- Benchmark runs across `hybrid_low_latency`, `openai_all`, `deepgram_all`, and local debug profiles.

## Implementation Decisions to Finalize in the Plan

These do not change the architecture, but the implementation plan must resolve them with code inspection or live validation before assigning work:

- Choose the first emergency-stop detector provider/model. Until then, emergency stop config validation must fail fast when enabled without a model.
- Live-validate OpenAI Codex through OpenAI Agents SDK. If rejected by the Codex backend, keep the auth/config boundary and replace only the processor call loop with direct Codex Responses usage.
- Confirm exact Pipecat input-audio frame class names in the installed version before implementing `wake_gate.py`.
- Confirm exact Pipecat observer/metrics events for STT/TTS TTFB before implementing `metrics.py`.
