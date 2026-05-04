# pipecat-agent

A Pipecat AI voice agent built with a cascade pipeline (STT → LLM → TTS).

## Runtime profiles

Default profile:

```text
hybrid_low_latency = Mave wake word + Deepgram Flux STT + OpenAI Codex OAuth agent + Cartesia Sonic TTS
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

### Required keys

For the default profile, set:

```dotenv
DEEPGRAM_API_KEY=
CARTESIA_API_KEY=
```

For `openai_all`, set:

```dotenv
OPENAI_API_KEY=
```

For OpenAI Codex OAuth agent auth, run Pi and login with ChatGPT Plus/Pro Codex:

```text
pi
/login
```

The agent reads Pi's `~/.pi/agent/auth.json` `openai-codex` OAuth profile.

`local_current` and `no_wake_debug` use local STT/TTS, but still use the Claude cloud agent. Authenticate with `claude auth login` and keep Claude plus the profile MCP URL reachable.

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
## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)