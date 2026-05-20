# FOC 2026 Workshop Runtime

Voice-controlled UR10 workshop runtime with a Python operator dashboard, Pipecat voice agent, and a local Vizor/RViz/MoveIt stack supplied as Docker images.

## What Students Run

Start from the repo root on Windows:

```cmd
Start-MAVE-Workshop.cmd
```

To preinstall `ur-rtde` for physical UR robot verified execution:

```cmd
Start-MAVE-Workshop.cmd --with-ur-rtde
```

Start from the repo root on macOS:

```bash
chmod +x ./Start-MAVE-Workshop.command
./Start-MAVE-Workshop.command
```

To preinstall `ur-rtde` for physical UR robot verified execution:

```bash
./Start-MAVE-Workshop.command --with-ur-rtde
```

The launchers use the base `uv sync` install by default and do not install `ur-rtde` unless `--with-ur-rtde` is passed.
The PyPI package is named `ur-rtde`; its Python modules are `rtde_receive`, `rtde_control`, and `rtde_io`, not `ur_rtde`.

The launcher creates `server/.venv` when needed, starts the operator dashboard, and opens a local URL like:

```text
http://127.0.0.1:8787/?token=...
```

In the dashboard, click **Start system**. It starts:

- Vizor + RViz/noVNC, rosbridge, Vizor MCP, and MoveIt MCP through `workshop.compose.yml`
- ModelTracker hologram sync
- verified execution server
- Pipecat browser client

## Requirements

- Windows with PowerShell or Command Prompt, or macOS with Terminal
- Git
- Docker Desktop in Linux containers mode
- Docker Compose v2 as `docker compose`
- `uv` on `PATH`
- Python `>=3.10,<3.13`; `uv` manages the environment
- Browser with microphone support
- `ur-rtde` is only needed for verified execution against a physical UR robot
- Pullable Docker images used by `workshop.compose.yml`:
  - `samulienko/noetic-vizor-rviz:latest`
  - `ghcr.io/samulko/noetic-vizor-local:latest`
  - `ghcr.io/samulko/01-docker-multi-actor-mcp:latest`

Install tools on Windows:

```powershell
winget install --id Git.Git -e
winget install --id Docker.DockerDesktop -e
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
git --version
docker version
docker compose version
uv --version
```

## Setup

Clone:

```powershell
git clone https://github.com/Samulko/-FOC_2026.git
cd -FOC_2026
```

Create `server\.env`:

```powershell
Copy-Item server\.env.example server\.env
notepad server\.env
```

Set at least:

```dotenv
OPENAI_API_KEY="..."
GOOGLE_API_KEY="..."
```

Install Python dependencies:

```powershell
cd server
uv sync
cd ..
```

That base install supports the operator dashboard, persona/modulation lab, Pipecat, MCP servers, and simulation services. For physical UR robot verified execution, install the optional robot extra:

```powershell
cd server
uv sync --extra robot
cd ..
```

After pulling dependency changes, rerun `cd server; uv sync`.

## Run

```cmd
Start-MAVE-Workshop.cmd
```

Then:

1. Click **Start system**.
2. Wait for Vizor + RViz, ModelTracker sync, verified execution, and Pipecat to report ready.
3. Open RViz from the dashboard.
4. Open Pipecat at `http://localhost:7860/client/`.
5. Allow microphone access.
6. Say a wake-word command, for example:

```text
Mave, move up a bit.
```

RViz/noVNC is also available at:

```text
http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote
```

## iOS Browser Client

This repo does not install or run natively on iOS. The workshop runtime runs on Windows; an iPhone or iPad can join only as a browser client for Pipecat and RViz/noVNC.

Requirements:

- iPhone or iPad on the same trusted LAN as the Windows host
- iOS browser with microphone permission enabled
- Windows Firewall allowing inbound LAN access to ports `7860` and `6080`

Expose Pipecat on the LAN with an ignored local dashboard config:

```powershell
Copy-Item server\operator_dashboard\default_config.toml operator_dashboard.local.toml
notepad operator_dashboard.local.toml
```

In `[services.pipecat]`, change:

```toml
command = ["uv", "run", "bot.py", "--host", "0.0.0.0"]
```

Keep `[dashboard] host = "127.0.0.1"`. The dashboard is intentionally local-only, and non-localhost dashboard hosts are rejected by config validation.

Find the Windows host IPv4 address:

```powershell
ipconfig
```

After starting the runtime from Windows, open these URLs on iOS with that IPv4 address:

```text
http://<windows-lan-ip>:7860/client/
```

```text
http://<windows-lan-ip>:6080/vnc_auto.html?host=<windows-lan-ip>&port=6080&path=websockify&autoconnect=true&resize=remote
```

Do not use `localhost` or `127.0.0.1` on iOS; those point to the iOS device. If iOS blocks microphone or WebRTC use over plain LAN HTTP, the current repo does not provide an HTTPS mobile mode.

## Manual Commands

Use these only when debugging outside the dashboard.

Run the dashboard without installing `ur-rtde`:

```powershell
cd server
uv run python -m operator_dashboard
```

The dashboard itself does not need `ur-rtde`. Starting **Verified Execution Server** uses the `robot` extra.

Run the Agent Persona app only:

```powershell
cd server
uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897
```

Open `http://127.0.0.1:8897`. This does not start the dashboard, Pipecat, Docker, or `ur-rtde`.

Run the image-based Vizor/RViz/MCP stack:

```powershell
docker compose -f workshop.compose.yml up
```

Run Pipecat:

```powershell
cd server
uv run bot.py --profile hybrid_gemini_live_tts
```

Run ModelTracker hologram sync:

```powershell
cd server
uv run python -m robot_control.shared_geometry.modeltracker_sync_server
```

Run verified execution:

```powershell
cd server
uv run --extra robot python -m verified_execution_server
```

Run MoveIt MCP directly:

```powershell
cd server
uv run python -m moveit_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8765
```

Run Vizor MCP directly:

```powershell
cd server
uv run python -m vizor_mcp --rosbridge-host localhost --rosbridge-port 9090 --transport streamable-http --http-host 127.0.0.1 --http-port 8001 --enable-holo1-tracking-on-startup
```

Do not run direct MCP commands while the Compose services already use ports `8001` or `8765`.

## Local Configuration

The bundled dashboard defaults live in `server/operator_dashboard/default_config.toml`.

For machine-specific overrides, create an ignored root file:

```powershell
Copy-Item server\operator_dashboard\default_config.toml operator_dashboard.local.toml
```

Use that file for local values such as a physical robot IP. Do not commit local overrides.

## Tuning Tools

Wake tuning writes logs under `server/logs/wake_tuning`:

```powershell
cd server
$logDir = "logs/wake_tuning"
New-Item -ItemType Directory -Force $logDir | Out-Null
uv run python -m wake_tuning.app 1> "$logDir/wake_tuning_server.out.log" 2> "$logDir/wake_tuning_server.err.log"
```

Open `http://127.0.0.1:9010`.

Saved wake tuning values are a local override under `server/state/wake_tuning_settings.json`; saving from the lab does not edit `server/runtime_profiles.toml`.

## Ports

| Port | Service |
|---:|---|
| 11311 | ROS master |
| 5901 | raw VNC |
| 6080 | noVNC/RViz |
| 7860 | Pipecat browser client |
| 8001 | Vizor MCP |
| 8765 | MoveIt MCP |
| 8770 | verified execution |
| 8787 | operator dashboard |
| 8788 | ModelTracker hologram sync |
| 8898 | robot job blackboard |
| 9090 | rosbridge |
| 10000-10003 | Vizor bridge ports |

If Docker reports an old `/ros-core`, `/vizor-demo`, `/vizor-mcp`, or `/moveit-mcp` container conflict, remove the old workshop containers:

```powershell
docker rm -f ros-core vizor-demo vizor-mcp moveit-mcp
```

## Testing

```powershell
cd server
uv run pytest
uv run ruff check .
uv run pyright .
cd ..
docker compose -f workshop.compose.yml config --quiet
```

Live provider, browser audio, and robot simulation checks are not part of the default deterministic test suite.

## Project Layout

```text
-FOC_2026/
├── Start-MAVE-Workshop.cmd
├── workshop.compose.yml
├── examples/
├── server/
│   ├── bot.py
│   ├── operator_dashboard/
│   ├── runtime_profiles.toml
│   ├── pyproject.toml
│   └── tests/
└── README.md
```
