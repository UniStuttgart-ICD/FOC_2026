# Voice Benchmarking

## Profiles

Benchmark profiles:

- `hybrid_low_latency`: Deepgram Flux STT + Cartesia Sonic TTS
- `openai_all`: OpenAI Realtime STT + OpenAI streaming TTS
- `deepgram_all`: Deepgram Flux STT + Deepgram Aura TTS

Debug profiles:

- `local_current`: local Whisper + Kokoro with Mave wake
- `no_wake_debug`: local Whisper + Kokoro without wake

## Running

```bash
cd server
uv run bot.py --profile hybrid_low_latency
```

## Metrics

Metrics are appended to:

```text
server/logs/voice_metrics.jsonl
```

Each JSONL record includes profile, category, transcript, response, and turn timing fields.

## Test utterances

Use the same utterances across profiles:

1. `Mave, what is the robot status?`
2. `Mave, what is the current position?`
3. `Mave, move up a bit.`
4. `Mave, stop.`

## Interpreting results

- Compare benchmark profiles to each other.
- Treat local profiles as debug/baseline, not equivalent streaming latency competitors.
- Do not compare runs if a profile silently failed or used fallback providers. Benchmark profiles fail startup instead of falling back.
