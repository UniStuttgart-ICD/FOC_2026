# pipecat-agent

A Pipecat AI voice agent built with a cascade pipeline (STT → LLM → TTS).

See [Architecture](ARCHITECTURE.md) for the target Module boundaries.

## Runtime profiles

Default profile:

```text
hybrid_openai_stt = Mave wake word + OpenAI Realtime Whisper STT + Gemini API LangChain agent + Cartesia Sonic TTS
```

Run the default profile:

```bash
cd server
uv run bot.py
```

Run a specific profile:

```bash
uv run bot.py --profile local_current
uv run bot.py --profile hybrid_openai_stt
uv run bot.py --profile openai_all
uv run bot.py --profile deepgram_all
uv run bot.py --profile no_wake_debug
```

`--profile` overrides `VOICE_PROFILE`.

`hybrid_openai_stt` uses Mave wake word, OpenAI Realtime Whisper STT, Gemini API agent, and Cartesia Sonic TTS.

### Wake tuning

Run the independent wake tuning page from `server/`:

```powershell
$logDir = "logs/wake_tuning"
New-Item -ItemType Directory -Force $logDir | Out-Null
uv run python -m wake_tuning.app 1> "$logDir/wake_tuning_server.out.log" 2> "$logDir/wake_tuning_server.err.log"
```

Open `http://127.0.0.1:9010`, start the mic, tune the sliders, then use **Save / implement**. Saved values go to ignored local state at `server/state/wake_tuning_settings.json`. This file is a local override read when the Pipecat bot starts; it does not edit `server/runtime_profiles.toml`.

To make tuned values the shared default, copy the saved profile values into `server/runtime_profiles.toml` and commit the profile change. Do not commit `server/state/wake_tuning_settings.json`.

### Voice Mod Lab

Run the independent voice modulation workbench:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897
```

Open `http://127.0.0.1:8897`, choose a runtime profile, generate a clean TTS preview, tune the modulation controls, then save local profile overrides to `server/state/voice_modulation_settings.json`. Do not commit the local state file.

### Required keys

For the default profile, set:

```dotenv
CARTESIA_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

For `openai_all`, set:

```dotenv
OPENAI_API_KEY=
```

For `hybrid_openai_stt`, set:

```dotenv
CARTESIA_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
```

`local_current` and `no_wake_debug` use local STT/TTS with the OpenAI API LangChain agent backend. Keep `OPENAI_API_KEY` and the configured robot MCP URL reachable.

### Wake word

The trained Mave wake-word model lives at:

```text
server/models/mave.onnx
```

Normal commands require `mave`, for example:

```text
Mave, move up a bit.
```

Local debug profiles are local STT/TTS debug profiles, not fully offline profiles. Benchmark profiles use streaming STT/TTS providers.

### Robot movement safety

Robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack. The voice agent routes movement through MoveIt workflows. Local validation may reject unsupported or malformed `moveit_*` calls for clearer errors, but it is not the source of movement safety.

## Setup

### Server

1. **Navigate to server directory**:

   ```bash
   cd server
   ```

2. **Install dependencies**:

   ```bash
   uv sync
   ```

3. **Configure environment variables**:

   ```bash
   cp .env.example .env
   # Edit .env and add your API keys
   ```

4. **Run the bot**:

   - SmallWebRTC: `uv run bot.py`

## Project Structure

```
pipecat-agent/
├── server/              # Python bot server
│   ├── bot.py           # Main bot implementation
│   ├── pyproject.toml   # Python dependencies
│   ├── .env.example     # Environment variables template
│   ├── .env             # Your API keys (git-ignored)
│   └── ...
├── .gitignore           # Git ignore patterns
└── README.md            # This file
```
## Testing

See [Testing](docs/testing.md) for deterministic tests and manual live LLM robot smoke tests.

## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)
