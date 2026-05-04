# Voice-Controlled UR Robot Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the scaffolded Pipecat bot with a simulation-only voice-controlled UR robot agent using Claude Agent SDK + MAVE MCP.

**Architecture:** Pipecat handles voice I/O only: Whisper STT turns speech into text, a custom `ClaudeAgentProcessor` sends the latest user turn to Claude Agent SDK via `query()`, Claude calls robot MCP tools over HTTP, and Kokoro speaks the final response. This v1 is intentionally simplified for URSim only: no HoloLens, no world model, no user-relative spatial reasoning.

**Tech Stack:** pipecat-ai (Whisper, Kokoro, SmallWebRTC), claude-agent-sdk, MAVE FastMCP HTTP server, URSim Docker

---

### Task 1: Update dependencies and environment docs

**Files:**
- Modify: `server/pyproject.toml`
- Modify: `server/.env.example`

**Step 1: Update `server/pyproject.toml`**

Replace the scaffold dependency set so the bot no longer depends on Google Gemini and instead installs Claude Agent SDK.

```toml
[project]
name = "pipecat-agent"
version = "0.1.0"
description = "Voice-controlled UR robot agent"
requires-python = ">=3.10,<3.13"
dependencies = [
    "pipecat-ai[kokoro,runner,silero,webrtc,whisper]",
    "claude-agent-sdk",
]

[dependency-groups]
dev = [
    "pyright>=1.1.404,<2",
    "ruff>=0.12.11,<1",
]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["I"]
```

Notes:
- Drop the `google` extra entirely.
- Keep `<3.13` because the robot stack is still tied to Python versions below 3.13.

**Step 2: Update `server/.env.example`**

Document the verified auth and MCP assumptions. Do **not** claim OAuth is the default. For this plan, assume `ANTHROPIC_API_KEY` is required unless a different auth flow is manually validated later.

```env
# pipecat-agent - Environment Variables
# Copy this file to .env and fill in your local configuration

# Claude Agent SDK / Anthropic
ANTHROPIC_API_KEY=

# Whisper STT model (local)
OPENAI_MODEL=base

# Kokoro TTS voice
KOKORO_VOICE_ID=af_heart

# MAVE MCP server URL.
# Current simulation docs/examples in Multi-Actor-Interface-Library use port 8000.
# If that project is reconfigured locally, update this value to match.
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
```

**Step 3: Install dependencies**

Run: `cd server && uv sync`

Expected:
- `claude-agent-sdk` installs successfully
- No Google-specific dependency is required anymore

**Step 4: Commit**

```bash
git add server/pyproject.toml server/.env.example
git commit -m "chore: switch pipecat agent to Claude SDK config"
```

---

### Task 2: Create a simulation-only robot system prompt

**Files:**
- Create: `server/prompts.py`

**Step 1: Create `server/prompts.py`**

Start from MAVE's robot prompt, but simplify it deliberately for simulation-only usage. Keep only rules that are true in this v1:
- simulation connection target is `127.0.0.1`
- no HoloLens/world-model references
- no user-relative left/right disambiguation logic
- explicit one-tool-at-a-time rule
- explicit parameter-shape examples for movement tools

Important: because this is simulation-only, prefer robot-frame commands and ask for clarification on ambiguous references like "that" or "to me".

```python
"""System prompt for the simulation-only voice robot agent."""

SYSTEM_PROMPT = """You are a voice-controlled robot agent for a Universal Robot (UR) arm running in simulation.

Users speak commands to you via voice. Respond conversationally but briefly (1-2 sentences).

## SCOPE
- This version is simulation-only.
- There is no HoloLens, gaze target, world model, or user-position data.
- If the user says "that", "this", "bring it here", or another ambiguous reference, ask a clarifying question instead of guessing.

## AVAILABLE MCP TOOLS
- connect_robot
- disconnect_robot
- get_robot_status
- get_joints
- get_tcp_pose
- move_to_position
- move_to_pose
- move_linear
- move_joints
- stop
- pause
- resume
- control_gripper
- control_gripper_position
- get_gripper_status
- robot_control

## TOOL PARAMETER FORMATS
- move_to_position: positions=[[x, y, z]]
- move_to_pose: poses=[[x, y, z, rx, ry, rz]]
- move_linear: poses=[[x, y, z, rx, ry, rz]]
- move_joints: positions=[[j1, j2, j3, j4, j5, j6]]
- Always wrap single targets in an outer list.
- WRONG: positions=[0.3, -0.2, 0.4]
- CORRECT: positions=[[0.3, -0.2, 0.4]]

## MOVEMENT RULES
- For simple positioning and pick/place, prefer move_to_position.
- Use move_to_pose only when orientation matters.
- Use move_linear only when a straight TCP path matters.
- Before movement, call get_tcp_pose.
- Call tools one at a time and wait for each result.
- If the same move fails twice, stop and report the failure.

## COORDINATE SYSTEM
- +X: forward from the base
- +Y: left from the base
- +Z: up
- "up" means +Z, "down" means -Z

## MAGNITUDE
- "a bit" / "slightly" = 0.05m
- no modifier = 0.10m
- "a lot" / "far" = 0.30m

## CONNECTION
- Simulation robot IP: 127.0.0.1
- If a tool reports no robot connection, call connect_robot(robot_ip="127.0.0.1") and retry once.

## RESPONSE STYLE
- Keep responses to 1-2 short sentences.
- Report positions in mm to the user.
- No emojis.
"""
```

**Step 2: Commit**

```bash
git add server/prompts.py
git commit -m "feat: add simulation-only robot prompt"
```

---

### Task 3: Create `ClaudeAgentProcessor` using `query()`

**Files:**
- Create: `server/claude_agent_processor.py`

This is the core integration piece. Use `query()` for each turn and keep a small manual conversation history. Do **not** describe or implement this as `ClaudeSDKClient`; that is a different session model.

**Step 1: Create the processor skeleton**

```python
"""Pipecat processor that runs Claude Agent SDK against the robot MCP server."""

from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, SystemMessage, query
from loguru import logger
from pipecat.frames.frames import (
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMMessagesFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from prompts import SYSTEM_PROMPT
```

**Step 2: Configure MCP correctly**

Anthropic's current MCP docs use `type: "http"` for URL-based non-SSE HTTP endpoints. MAVE's server is a FastMCP streamable HTTP endpoint mounted at `/mcp`, so configure it as `http`, not `sse`.

```python
options = ClaudeAgentOptions(
    mcp_servers={
        "robot": {
            "type": "http",
            "url": self._mcp_server_url,
        }
    },
    allowed_tools=["mcp__robot__*"],
)
```

**Step 3: Add init-message health checking**

Do not rely on a nonexistent `get_mcp_status()` tool. Check the SDK's initial `SystemMessage` and fail early if MCP is unavailable.

```python
async for message in query(prompt=prompt, options=options):
    if isinstance(message, SystemMessage) and message.subtype == "init":
        failed_servers = [
            server
            for server in message.data.get("mcp_servers", [])
            if server.get("status") != "connected"
        ]
        if failed_servers:
            logger.error(f"MCP connection failed: {failed_servers}")
            return "I can't reach the robot control server right now."
```

**Step 4: Collect the assistant response and handle SDK errors explicitly**

Use the final `ResultMessage` subtype plus assistant text blocks, instead of only stringifying random message objects.

```python
response_text = ""

async for message in query(prompt=prompt, options=options):
    if isinstance(message, AssistantMessage):
        for block in message.content:
            if getattr(block, "type", None) == "text":
                response_text += block.text

    if isinstance(message, ResultMessage):
        if message.subtype == "error_during_execution":
            logger.error("Claude Agent SDK execution failed")
            return "I hit an execution error while talking to the robot."
        if message.subtype == "success" and not response_text:
            response_text = str(message.result)
```

**Step 5: Build the full implementation**

Important requirements:
- Extract the latest user text from `LLMMessagesFrame`
- Keep only the last ~10 turns in manual history
- Prepend `SYSTEM_PROMPT` to the query prompt
- Push `LLMFullResponseStartFrame`, then `LLMTextFrame`, then `LLMFullResponseEndFrame`
- Return a short fallback string if the SDK returns no user-facing text

Implementation sketch:

```python
class ClaudeAgentProcessor(FrameProcessor):
    def __init__(self, mcp_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._conversation_history: list[dict[str, str]] = []

    async def _process_with_agent(self, user_text: str) -> str:
        self._conversation_history.append({"role": "user", "content": user_text})

        history = self._conversation_history[-10:]
        prompt_lines = [SYSTEM_PROMPT, "", "## Conversation"]
        for msg in history:
            prompt_lines.append(f"{msg['role']}: {msg['content']}")
        prompt_lines.append("")
        prompt_lines.append("Respond to the latest user message.")
        prompt = "\n".join(prompt_lines)

        # Run query(...) here using the verified options and health checks above.
        ...
```

**Step 6: Commit**

```bash
git add server/claude_agent_processor.py
git commit -m "feat: add query-based Claude robot processor"
```

---

### Task 4: Replace the scaffolded LLM service in `bot.py`

**Files:**
- Modify: `server/bot.py`

**Step 1: Remove the scaffolded Google LLM**

Delete:
- `GoogleLLMService` import
- `LLMRunFrame` import
- the `on_client_ready` handler that asks the bot to introduce itself

This bot should wait for speech input instead of greeting first.

**Step 2: Keep the existing voice pipeline and swap in `ClaudeAgentProcessor`**

Use the current Pipecat scaffold shape and only replace the LLM stage.

```python
stt = WhisperSTTService(
    settings=WhisperSTTService.Settings(
        model=os.getenv("OPENAI_MODEL", "base"),
    ),
)

tts = KokoroTTSService(
    settings=KokoroTTSService.Settings(
        voice=os.getenv("KOKORO_VOICE_ID"),
    ),
)

claude_agent = ClaudeAgentProcessor(
    mcp_server_url=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8000/mcp"),
)
```

**Step 3: Keep aggregator/event wiring intact**

Preserve transcript logging and the existing SmallWebRTC transport setup.

```python
context = LLMContext()
user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
    context,
    user_params=LLMUserAggregatorParams(
        vad_analyzer=SileroVADAnalyzer(),
    ),
)

pipeline = Pipeline(
    [
        transport.input(),
        stt,
        user_aggregator,
        claude_agent,
        tts,
        transport.output(),
        assistant_aggregator,
    ]
)
```

**Step 4: Verify the final `bot.py` behavior**

Expected result:
- Browser connects
- No auto-introduction occurs
- User speaks first
- Transcript logging still works
- Assistant output still reaches Kokoro and WebRTC audio out

**Step 5: Commit**

```bash
git add server/bot.py
git commit -m "feat: swap scaffolded Google LLM for Claude MCP processor"
```

---

### Task 5: Run a simulation-first integration test

**Files:**
- Modify: `server/.env` locally only
- Verify external dependency: `C:/Users/Samuel/Documents/github/Multi-Actor-Interface-Library`

**Step 1: Start URSim**

Run:

```bash
docker compose -f C:/Users/Samuel/Documents/github/Multi-Actor-Interface-Library/docker/ursim/docker-compose.yml up
```

Expected:
- URSim starts
- VNC is reachable at `http://localhost:6080`

**Step 2: Start the MAVE MCP server**

Run:

```bash
cd C:/Users/Samuel/Documents/github/Multi-Actor-Interface-Library
uv run python -m mcp_server.server
```

Expected:
- HTTP server starts
- MCP server is reachable at `http://127.0.0.1:8000/mcp` in the default simulation setup

**Step 3: Create local bot config**

```bash
cd C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server
cp .env.example .env
```

Set:

```env
ANTHROPIC_API_KEY=...
OPENAI_MODEL=base
KOKORO_VOICE_ID=af_heart
MCP_SERVER_URL=http://127.0.0.1:8000/mcp
```

**Step 4: Start the voice bot**

Run:

```bash
uv run bot.py
```

Expected:
- Bot starts cleanly
- SmallWebRTC client URL is printed
- No immediate assistant speech

**Step 5: Test startup health before motion**

Open the SmallWebRTC client and say:

1. `"What is the robot status?"`
2. Expected: Claude can reach MCP, calls `get_robot_status()`, and gives a short spoken reply

If this fails with a robot/MCP connectivity message, stop here and fix server config before movement tests.

**Step 6: Test simple simulation commands**

Voice tests:

1. `"Connect to the robot"`
Expected: If the robot is already connected, the agent says so or confirms connection cleanly.

2. `"What is the robot's current position?"`
Expected: `get_tcp_pose()` is called and the position is read back in mm.

3. `"Move the robot up a bit"`
Expected: agent calls `get_tcp_pose()`, computes `+0.05m` on Z, calls `move_to_position([[x, y, z]])`, and confirms the new Z.

4. `"Stop"`
Expected: `stop()` is called and the agent confirms the stop action.

**Step 7: Verify in URSim**

Use `http://localhost:6080` to confirm the simulated robot actually moved for the movement test.

**Step 8: Commit**

```bash
git add server/bot.py server/claude_agent_processor.py server/prompts.py server/pyproject.toml server/.env.example
git commit -m "feat: add simulation-only voice robot agent"
```

---

## Iteration Notes

After the initial simulation version works, possible follow-ups:
- Switch from `query()` to `ClaudeSDKClient` if multi-turn session quality becomes a problem
- Add sentence-chunked responses for faster TTS start
- Reintroduce MAVE sensor/world-model grounding in a separate non-simulation plan
- Add stronger MCP retry/reconnect behavior after transient server failures
