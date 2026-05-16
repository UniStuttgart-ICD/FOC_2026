# Voice Benchmarking

## Profile

The repository currently bundles one app profile:

- `hybrid_gemini_live_tts`: Mave wake word, OpenAI Realtime Whisper STT, Gemini Live TTS, and `gemini_api` Agent Control.

This profile needs:

```dotenv
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

It also needs the configured MoveIt MCP URL reachable. The default is `http://127.0.0.1:8765/mcp`.

## Running

From `server/`:

```bash
uv run bot.py
```

Equivalent explicit run:

```bash
uv run bot.py --profile hybrid_gemini_live_tts
```

## Metrics

The runtime profile configures these base paths:

```text
server/logs/voice_metrics.jsonl
server/logs/process_trace.jsonl
```

`pipeline_builder.py` expands each base path into a session-scoped JSONL file under:

```text
server/logs/voice_metrics/
server/logs/process_trace/
```

Voice Metrics records include profile, category, wake phrase, transcript, response, and turn timing fields.

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

## Test Utterances

Use the same utterances between runs:

1. `Mave, what is the robot status?`
2. `Mave, what is the current position?`
3. `Mave, move up a bit.`
4. `Mave, stop.`

`Mave, stop.` is a normal Voice Command test utterance, not an emergency-stop bypass.

## Interpreting Results

- Compare repeated runs of the same profile unless a new profile is intentionally added.
- Do not compare runs if startup silently failed or the configured providers were changed.
- Treat Process Trace as the detailed debugging artifact and Voice Metrics as the compact timing summary.
