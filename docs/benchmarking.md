# Voice Benchmarking

## Profiles

Benchmark profiles:

- `hybrid_low_latency`: Deepgram Flux STT + Cartesia Sonic TTS
- `openai_all`: OpenAI Realtime STT + OpenAI streaming TTS
- `deepgram_all`: Deepgram Flux STT + Deepgram Aura TTS

Debug profiles:

- `local_current`: local Whisper + Kokoro STT/TTS with Mave wake; Claude cloud agent
- `no_wake_debug`: local Whisper + Kokoro STT/TTS without wake; Claude cloud agent

These are local STT/TTS debug profiles, not fully offline profiles. They need Claude auth/connectivity and the profile MCP URL.

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

Each JSONL record includes profile, category, wake phrase, transcript, response, and turn timing fields.

Duration semantics live in the Voice Metrics Module. Timing fields are milliseconds with deterministic stage semantics:

- `wake_latency_ms`: turn start to wake detection.
- `speech_captured_ms`: wake detection, or turn start without wake, to speech captured.
- `stt_latency_ms`: speech captured to STT done.
- `agent_latency_ms`: STT done to agent done.
- `tts_first_audio_ms`: agent done to first TTS audio.
- `tts_done_ms`: first TTS audio to TTS done.
- `total_to_first_audio_ms`: turn start to first TTS audio.
- `total_turn_ms`: turn start to finish.

Missing marks are recorded as `null`.

## Test utterances

Use the same utterances across profiles:

1. `Mave, what is the robot status?`
2. `Mave, what is the current position?`
3. `Mave, move up a bit.`
4. `Mave, stop.`

`Mave, stop.` is a normal Voice Command test utterance, not an emergency-stop bypass.

## Interpreting results

- Compare benchmark profiles to each other.
- Treat local STT/TTS profiles as debug/baseline, not fully offline or equivalent streaming latency competitors.
- Do not compare runs if a profile silently failed or used fallback providers. Benchmark profiles fail startup instead of falling back.
