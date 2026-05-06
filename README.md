# pipecat-agent

A Pipecat AI voice agent built with a cascade pipeline (STT → LLM → TTS).

See [Voice Runtime Architecture](docs/architecture.md) for the reusable Module boundaries.

## Runtime profiles

Default profile:

```text
hybrid_low_latency = Mave wake word + Deepgram Flux STT + OpenAI API LangChain agent + Cartesia Sonic TTS
```

Run the default profile:

```bash
cd server
uv run bot.py
```

Run a specific profile:

```bash
uv run bot.py --profile local_current
uv run bot.py --profile openai_all
uv run bot.py --profile deepgram_all
uv run bot.py --profile no_wake_debug
```

`--profile` overrides `VOICE_PROFILE`.

### Wake tuning

Run the independent wake tuning page from `server/`:

```bash
uv run python -m wake_tuning.app
```

Open `http://127.0.0.1:9010`, start the mic, tune the sliders, then use **Save / implement**. Saved values go to `server/wake_tuning_settings.json` and override the selected profile's wake settings when the Pipecat bot starts.

### Required keys

For the default profile, set:

```dotenv
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
OPENAI_API_KEY=
```

For `openai_all`, set:

```dotenv
OPENAI_API_KEY=
```

`local_current` and `no_wake_debug` use local STT/TTS with the same OpenAI API LangChain agent backend as the benchmark profiles. Keep `OPENAI_API_KEY` and the configured robot MCP URL reachable.

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
