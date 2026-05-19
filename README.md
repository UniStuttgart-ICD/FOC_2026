# Pipecat Agent

Voice-controlled UR10 robot agent built on Pipecat, LangGraph/LangChain, MCP, and a local Vizor/MoveIt simulation stack.

## Highlights

- One launcher starts the operator dashboard and creates the Python environment when needed.
- The dashboard starts and monitors RViz/noVNC, Vizor MCP, MoveIt MCP, ModelTracker hologram sync, verified execution, and the Pipecat browser client.
- The default profile uses the `mave` wake word, OpenAI Realtime Whisper STT, Gemini Live TTS, and a Gemini API LangChain agent.
- Robot actions are planned and executed through MoveIt workflows, with process trace and voice metrics written under `server/logs/`.
- Deterministic tests run locally without live model keys, browser audio, or the robot simulation stack.

## Overview

This repo is a realtime voice pipeline for robot operation. A browser client streams microphone audio into Pipecat, the agent turns spoken commands into canonical `moveit_*` tool calls, and the local workshop stack shows the UR10 in RViz while MCP services expose planning, execution, and user-sensing context.

Start with the dashboard path below. Use the manual commands only when you need to debug individual services.

Key references:

- [Architecture](ARCHITECTURE.md)
- [Domain context](CONTEXT.md)
- [Operator dashboard](docs/operator-dashboard.md)
- [Vizor MoveIt MCP](docs/VIZOR_MOVEIT_MCP.md)
- [Testing](docs/testing.md)

## Dependencies

Host requirements:

- Git for Windows
- Windows with PowerShell or Command Prompt for `Start-MAVE-Workshop.cmd`
- Docker Desktop running Linux containers, with Compose v2 available as `docker compose`
- `uv` on `PATH`; verify with `uv --version` after installing
- Python `>=3.10,<3.13`; `uv sync` creates `server/.venv` and selects a compatible Python
- A Chromium-based browser or another browser with microphone support

Fresh installs need outbound HTTPS access to:

- `github.tik.uni-stuttgart.de` for cloning this repo
- `astral.sh` for installing `uv`
- Python package indexes used by `uv sync`
- Docker registries and Linux package mirrors used by Docker builds
- `github.com` for the Robotiq gripper checkout during the RViz image build
- OpenAI and Google Gemini APIs at runtime

Default profile API keys:

- `OPENAI_API_KEY` for OpenAI Realtime Whisper STT
- `GOOGLE_API_KEY` for Gemini Live TTS and the Gemini API agent

Repo-managed dependencies:

- Python dependencies are declared in `server/pyproject.toml` and locked in `server/uv.lock`.
- Direct runtime packages include `pipecat-ai`, `openai`, `mcp`, `langgraph`, `langchain-*`, `fastapi`, `uvicorn`, `httpx`, `openwakeword`, `silero-vad`, `roslibpy`, `ur-rtde`, `psutil`, and `tomlkit`.
- Development packages include `pytest`, `pytest-asyncio`, `ruff`, and `pyright`.
- Docker dependencies are declared in `docker/compose/workshop.yml` and the Dockerfiles under `docker/`.
- Docker builds pull `cxy201/noetic-vizor` and `python:3.12-slim`, install apt packages, install MCP Python packages from PyPI, and clone `https://github.com/KevinGalassi/Robotiq-2f-85.git` at a pinned commit.
- The wake-word model is bundled at `server/models/mave.onnx`.

Useful local ports:

| Port | Service |
|---:|---|
| 11311 | ROS master inside Compose |
| 5901 | raw VNC for RViz desktop |
| 6080 | noVNC/RViz |
| 8787 | Operator dashboard |
| 7860 | Pipecat browser client |
| 8765 | MoveIt MCP |
| 8788 | ModelTracker hologram sync |
| 8001 | Vizor MCP |
| 8770 | Verified execution server |
| 8898 | Robot job blackboard |
| 9090 | rosbridge from `vizor-demo` |
| 9010 | Wake tuning lab |
| 8897 | Agent Persona Lab |
| 10000-10003 | Vizor bridge ports from `vizor-demo` |

MCP port matrix:

| MCP server | Dashboard/Compose port | Pipecat URL | Bare local default | Required local override |
|---|---:|---|---:|---|
| MoveIt MCP | 8765 | `http://127.0.0.1:8765/mcp` | 8000 | `--http-port 8765` |
| Vizor MCP | 8001 | `http://127.0.0.1:8001/mcp` via `MCP_VIZOR_URL` | 8001 | none |

Do not run `python -m moveit_mcp` without `--http-port 8765` for the Pipecat profile in this repo. The module's bare local HTTP default is `8000`, while the bundled Docker/dashboard/Pipecat path expects `8765`.

## Installation

On a fresh Windows machine, install the host tools first:

- Git for Windows: https://git-scm.com/download/win
- Docker Desktop for Windows: https://docs.docker.com/desktop/setup/install/windows-install/
- uv installation docs: https://docs.astral.sh/uv/getting-started/installation/

PowerShell install commands:

```powershell
winget install --id Git.Git -e
winget install --id Docker.DockerDesktop -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

If `winget` is unavailable, use the links above. Finish Docker Desktop setup, restart if prompted, and start Docker Desktop before continuing.

Open a new terminal if `uv --version` is not found immediately after installation.

Verify the tools:

```powershell
git --version
docker version
docker compose version
uv --version
```

Clone the repository:

```powershell
git clone https://github.tik.uni-stuttgart.de/ac147490/Robot_buddy pipecat-agent
cd pipecat-agent
```

Create your local environment file:

```powershell
Copy-Item server\.env.example server\.env
notepad server\.env
```

Set at least:

```dotenv
OPENAI_API_KEY="..."
GOOGLE_API_KEY="..."
```

Install the Python environment:

```powershell
cd server
uv sync
cd ..
```

Plain `uv sync` installs the runtime dependencies and the default `dev` group, including `pytest`, `pytest-asyncio`, `ruff`, and `pyright`. No separate `pip install` step is needed.

The workshop launcher also runs `uv sync` if `server/.venv\Scripts\python.exe` does not exist. After pulling dependency changes, run `cd server; uv sync` yourself to refresh an existing environment.

## Run E2E

Start Docker Desktop before running the launcher, keep it in Linux containers mode, then run the workshop launcher from the repo root:

```cmd
Start-MAVE-Workshop.cmd
```

Keep the launcher window open. It prints a tokenized URL like:

```text
http://127.0.0.1:8787/?token=...
```

Open that URL if the browser does not open automatically.

The launcher starts only the dashboard. In the dashboard, **Start system** starts the main services in this order: Vizor + RViz Compose stack, ModelTracker hologram sync, verified execution server, then Pipecat voice agent. Wake tuning and Agent Persona Lab are optional and are not included in Start system.

In the dashboard:

1. Click **Start system**.
2. Wait until Vizor + RViz, ModelTracker sync, verified execution, and Pipecat report ready.
3. Open RViz to confirm the UR10 is visible.
4. Open Pipecat at `http://localhost:7860/client/`.
5. Allow microphone access.
6. Say a wake-word command, for example:

```text
Mave, move up a bit.
```

The first Docker build can take several minutes. RViz is available through noVNC at:

```text
http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote
```

## Manual Service Commands

Use these only when debugging outside the dashboard.

Do not start the dashboard-managed stack and the manual services at the same time. The Compose stack owns ports `6080`, `5901`, `9090`, `10000-10003`, `11311`, `8001`, and `8765`; host services own `8770`, `7860`, `8788`, and `8898`.

Create or refresh the Python environment:

```powershell
cd server
uv sync
```

Run the Pipecat voice agent:

```powershell
uv run bot.py --profile hybrid_gemini_live_tts
```

Run the operator dashboard without the `.cmd` launcher:

```powershell
cd server
uv run python ..\scripts\run_operator_dashboard.py
```

Run Vizor + RViz directly:

```powershell
docker compose -f docker/compose/workshop.yml up --build
```

This Compose stack also starts `vizor-mcp` on port `8001` and `moveit-mcp` on port `8765`.

Run MoveIt MCP directly:

```powershell
cd server
uv run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8765
```

Do not run the direct MCP command while the Compose `moveit-mcp` service is already bound to port `8765`.

Run ModelTracker hologram sync directly:

```powershell
cd server
uv run python -m robot_control.shared_geometry.modeltracker_sync_server
```

Do not bind ModelTracker hologram sync to port `8765`; the Vizor/RViz Compose stack uses that port for MoveIt MCP.

Run Vizor MCP directly:

```powershell
cd server
uv run python ..\scripts\run_vizor_mcp_server.py --host 127.0.0.1 --port 8001 --rosbridge-host localhost --rosbridge-port 9090 --enable-holo1-tracking-on-startup
```

Do not run the direct Vizor MCP command while the Compose `vizor-mcp` service is already bound to port `8001`.

## Local Configuration

Machine-specific dashboard settings belong in:

```text
configs/operator_dashboard.local.toml
```

Start by copying the example:

```powershell
Copy-Item configs\operator_dashboard.example.toml configs\operator_dashboard.local.toml
```

The launcher uses `configs/operator_dashboard.example.toml` when `configs/operator_dashboard.local.toml` does not exist. Create the local file only for machine-specific overrides.

Use the local file for machine-specific values such as a physical robot IP. Do not commit local overrides.

Default runtime profile:

```text
hybrid_gemini_live_tts = Mave wake word + OpenAI Realtime Whisper STT + Gemini Live TTS + Gemini API LangChain agent
```

`server/runtime_profiles.toml` intentionally carries one bundled profile. `--profile` overrides `VOICE_PROFILE`.

## Tuning Tools

Wake tuning:

```powershell
cd server
$logDir = "logs/wake_tuning"
New-Item -ItemType Directory -Force $logDir | Out-Null
uv run python -m wake_tuning.app 1> "$logDir/wake_tuning_server.out.log" 2> "$logDir/wake_tuning_server.err.log"
```

Open `http://127.0.0.1:9010`.

Saved wake tuning values are a local override under `server/state/wake_tuning_settings.json`; saving from the lab does not edit `server/runtime_profiles.toml`.

Agent Persona Lab:

```powershell
cd server
uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897
```

Open `http://127.0.0.1:8897`.

Both tools save ignored local state under `server/state/`. Promote shared defaults by editing `server/runtime_profiles.toml`.

## Verification

Run deterministic tests from `server/`:

```powershell
cd server
uv run pytest
```

Optional checks:

```powershell
uv run ruff check .
uv run pyright .
```

Tests marked `live`, `llm`, `native_llm`, `robot_sim`, or `integration` require explicit credentials or external services and are not part of the default local check.

## Project Layout

```text
pipecat-agent/
├── Start-MAVE-Workshop.cmd        # Windows launcher for the dashboard
├── configs/                       # Dashboard configuration
├── docker/                        # Compose stack and Docker images
├── docs/                          # Operator, MCP, testing, ADR, and design docs
├── scripts/                       # Repo-root service launchers
├── server/                        # Python runtime and tests
│   ├── bot.py                     # Pipecat runner entrypoint
│   ├── pipeline_builder.py        # App composition root
│   ├── runtime_profiles.toml      # Bundled runtime profile
│   ├── pyproject.toml             # Python dependencies
│   └── tests/                     # Deterministic pytest suite
└── README.md
```

## Feedback and Contributing

Keep setup changes focused on making the clone-to-run path shorter and easier to verify. When changing runtime boundaries or terminology, update [Architecture](ARCHITECTURE.md) and [Domain context](CONTEXT.md) with the same terms.
