# Agent Speed Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce per-turn latency by switching from stateless `query()` to persistent `ClaudeSDKClient`, tuning agent parameters, adding MCP resilience, and streaming partial text to TTS.

**Architecture:** Replace the subprocess-per-turn pattern with a single long-lived `ClaudeSDKClient` that persists for the WebRTC session. System prompt set via `ClaudeAgentOptions(system_prompt=...)`. The SDK maintains conversation state natively, eliminating manual history management and per-turn MCP re-handshakes. Fast-path for deterministic commands deferred until after benchmarking.

**Tech Stack:** claude-agent-sdk 0.1.48 (`ClaudeSDKClient`, `StreamEvent`), pipecat-ai (pipeline/frames)

---

## Phase 1: Foundation

### Task 1: Pin model and add env config

**Files:**
- Modify: `server/claude_agent_processor.py:47-55`
- Modify: `server/.env.example`

**Step 1: Add CLAUDE_MODEL to server/.env.example**

Add after the OAuth comment block:

```
# Claude model (see claude-agent-sdk docs for valid model IDs)
CLAUDE_MODEL=claude-sonnet-4-5
```

**Step 2: Read model from env in ClaudeAgentProcessor.__init__**

In `server/claude_agent_processor.py`, add `os` import and store the model:

```python
import os
```

In `__init__`, add:

```python
self._model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
```

**Step 3: Set model on ClaudeAgentOptions**

In `_process_with_agent`, update the options:

```python
options = ClaudeAgentOptions(
    model=self._model,
    mcp_servers={
        "robot": {
            "type": "http",
            "url": self._mcp_server_url,
        }
    },
    allowed_tools=["mcp__robot__*"],
)
```

**Step 4: Log model from first AssistantMessage**

Add `self._model_logged = False` to `__init__`. Inside the message loop:

```python
if isinstance(message, AssistantMessage):
    if not self._model_logged and message.model:
        logger.info(f"Claude model: {message.model}")
        self._model_logged = True
```

**Step 5: Manual test**

Run: `uv run bot.py`
Expected: Log line `Claude model: claude-sonnet-4-5` on first user utterance.

**Step 6: Commit**

```bash
git add server/claude_agent_processor.py server/.env.example
git commit -m "feat: pin Claude model explicitly via CLAUDE_MODEL env var"
```

---

### Task 2: Switch from query() to ClaudeSDKClient

**Files:**
- Rewrite: `server/claude_agent_processor.py` (full file)
- Modify: `server/bot.py:92-99` (lifecycle hooks)

**Step 1: Rewrite server/claude_agent_processor.py**

Replace the entire file. Key differences from old code:
- `ClaudeSDKClient` instead of `query()`
- `system_prompt=SYSTEM_PROMPT` in `ClaudeAgentOptions`, NOT in `connect(prompt=...)`
- No manual `self._history` — SDK maintains conversation state
- Double-connect guard in `connect()`
- Cleanup on `CancelFrame`/`EndFrame`

```python
"""Pipecat processor that runs Claude Agent SDK against the robot MCP server."""

import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
)
from loguru import logger
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from prompts import SYSTEM_PROMPT


class ClaudeAgentProcessor(FrameProcessor):
    """Routes user turns through a persistent ClaudeSDKClient with robot MCP tools.

    Maintains a single Claude session for the lifetime of the WebRTC connection.
    The SDK handles conversation history natively.
    """

    def __init__(self, mcp_server_url: str, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")
        self._client: ClaudeSDKClient | None = None
        self._model_logged = False

    async def connect(self):
        """Initialize the persistent Claude SDK client."""
        if self._client:
            return
        options = ClaudeAgentOptions(
            model=self._model,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={
                "robot": {
                    "type": "http",
                    "url": self._mcp_server_url,
                }
            },
            allowed_tools=["mcp__robot__*"],
        )
        self._client = ClaudeSDKClient(options=options)
        await self._client.connect()
        logger.info("ClaudeSDKClient connected")

    async def disconnect(self):
        """Shut down the Claude SDK client."""
        if self._client:
            await self._client.disconnect()
            self._client = None
            logger.info("ClaudeSDKClient disconnected")

    async def _process_with_agent(self, user_text: str) -> str:
        if not self._client:
            return "Agent not connected."

        await self._client.query(user_text)

        response_text = ""
        async for message in self._client.receive_response():
            if isinstance(message, AssistantMessage):
                if not self._model_logged and message.model:
                    logger.info(f"Claude model: {message.model}")
                    self._model_logged = True
                for block in message.content:
                    if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                        response_text += block.text

            if isinstance(message, ResultMessage):
                if message.is_error:
                    return "I hit an error while talking to the robot."
                if not response_text and message.result:
                    response_text = str(message.result)

        return response_text or "I completed the action but have nothing to report."

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages if frame.context else []
            user_text = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        user_text = content.strip()
                        break
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                user_text = part["text"].strip()
                                break
                        if user_text:
                            break

            if user_text:
                logger.info(f"User said: {user_text}")
                await self.push_frame(LLMFullResponseStartFrame())
                response = await self._process_with_agent(user_text)
                await self.push_frame(LLMTextFrame(text=response))
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
```

**Step 2: Wire lifecycle in server/bot.py**

Update the event handlers:

```python
@transport.event_handler("on_client_connected")
async def on_client_connected(transport, client):
    logger.info("Client connected")
    await claude_agent.connect()

@transport.event_handler("on_client_disconnected")
async def on_client_disconnected(transport, client):
    logger.info("Client disconnected")
    await claude_agent.disconnect()
    await task.cancel()
```

**Step 3: Manual test**

Run: `uv run bot.py`
Expected:
1. Log: `ClaudeSDKClient connected` on WebRTC client connect
2. First utterance works without re-establishing MCP
3. Second utterance reuses the same session (no subprocess spawn)
4. Log: `ClaudeSDKClient disconnected` on client disconnect

**Step 4: Commit**

```bash
git add server/claude_agent_processor.py server/bot.py
git commit -m "feat: switch from stateless query() to persistent ClaudeSDKClient"
```

---

## Phase 2: Tuning

### Task 3: Narrow get_tcp_pose to relative moves only

**Files:**
- Modify: `server/prompts.py:39-45`

**Step 1: Update the MOVEMENT RULES section**

Replace lines 39-45:

Old:
```
## MOVEMENT RULES
- For simple positioning and pick/place, prefer move_to_position.
- Use move_to_pose only when orientation matters.
- Use move_linear only when a straight TCP path matters.
- Before movement, call get_tcp_pose.
- Call tools one at a time and wait for each result.
- If the same move fails twice, stop and report the failure.
```

New:
```
## MOVEMENT RULES
- For simple positioning and pick/place, prefer move_to_position.
- Use move_to_pose only when orientation matters.
- Use move_linear only when a straight TCP path matters.
- Before relative movement (e.g. "up a bit", "left", "forward"), call get_tcp_pose to get the current position, then offset.
- For absolute coordinates, move directly without reading pose first.
- Call tools one at a time and wait for each result.
- If the same move fails twice, stop and report the failure.
```

**Step 2: Manual test**

- "Move to position 0.3 -0.2 0.4" — should NOT call `get_tcp_pose` first
- "Move up a bit" — should call `get_tcp_pose` first, then offset +Z by 0.05

**Step 3: Commit**

```bash
git add server/prompts.py
git commit -m "feat: only require get_tcp_pose before relative moves"
```

---

### Task 4: Add max_turns and low effort

**Files:**
- Modify: `server/claude_agent_processor.py` (options in `connect()`)

**Step 1: Add max_turns and effort to ClaudeAgentOptions**

In the `connect()` method, update the options:

```python
options = ClaudeAgentOptions(
    model=self._model,
    system_prompt=SYSTEM_PROMPT,
    max_turns=3,
    effort="low",
    mcp_servers={
        "robot": {
            "type": "http",
            "url": self._mcp_server_url,
        }
    },
    allowed_tools=["mcp__robot__*"],
)
```

`max_turns=3`: allows tool call + result + response. `effort="low"`: reduces thinking overhead.

**Step 2: Manual test**

Run: `uv run bot.py`
- "Move up a bit" — should complete in ≤3 turns (get_tcp_pose, move, respond)
- "What's the robot status?" — should complete in ≤2 turns
- If a complex command fails due to turn limit, raise `max_turns` later

**Step 3: Commit**

```bash
git add server/claude_agent_processor.py
git commit -m "feat: add max_turns=3 and effort=low for faster inference"
```

---

### Task 5: Benchmark persistent client

**No code changes.** Run before/after timing comparisons.

**Step 1: Measure baseline**

With the persistent client now active, time several commands end-to-end (from user utterance to TTS start):
- "What's the robot status?"
- "Move up a bit"
- "Open the gripper"

Log timestamps at key points (`User said:` → first `LLMTextFrame` push).

**Step 2: Record results**

Note latency per command type. This informs whether Phase 4 fast-path is still needed and what the actual per-turn savings are from the `ClaudeSDKClient` switch.

**Step 3: Decide on fast-path priority**

If simple commands ("stop", "open gripper") still take >1s through the LLM path, fast-path is worth building. If they're already fast enough, defer it.

---

## Phase 3: Resilience

### Task 6: Add MCP reconnect error handling

**Files:**
- Modify: `server/claude_agent_processor.py`

**Step 1: Add reconnect helper**

```python
async def _ensure_mcp_connected(self) -> bool:
    """Check MCP status and reconnect if needed. Returns True if connected."""
    if not self._client:
        return False
    try:
        status = await self._client.get_mcp_status()
        for server in status.get("mcpServers", []):
            if server.get("name") == "robot" and server.get("status") != "connected":
                logger.warning("Robot MCP disconnected, reconnecting...")
                await self._client.reconnect_mcp_server("robot")
                return True
        return True
    except Exception as e:
        logger.error(f"MCP status check failed: {e}")
        return False
```

**Step 2: Guard _process_with_agent entry**

At the start of `_process_with_agent`, before `self._client.query()`:

```python
if not await self._ensure_mcp_connected():
    return "I can't reach the robot control server right now."
```

**Step 3: Manual test**

1. Start bot, send a command (works)
2. Kill the MCP server process
3. Send another command — should see "reconnecting" log
4. Restart MCP server, send command — should recover

**Step 4: Commit**

```bash
git add server/claude_agent_processor.py
git commit -m "feat: add MCP reconnect handling for persistent client"
```

---

### Task 7: Pre-warm MCP at connect time

**Files:**
- Modify: `server/claude_agent_processor.py` (`connect()`)

**Step 1: Add pre-warm after client.connect()**

At the end of `connect()`:

```python
# Pre-warm: verify robot MCP is reachable
try:
    status = await self._client.get_mcp_status()
    for server in status.get("mcpServers", []):
        if server.get("name") == "robot":
            if server.get("status") == "connected":
                logger.info("Robot MCP server connected and ready")
            else:
                logger.warning(f"Robot MCP server not ready: {server.get('status')}")
except Exception as e:
    logger.warning(f"Could not verify MCP status at startup: {e}")
```

**Step 2: Manual test**

- MCP server running: log `Robot MCP server connected and ready`
- MCP server down: log warning, bot still starts

**Step 3: Commit**

```bash
git add server/claude_agent_processor.py
git commit -m "feat: pre-warm MCP connection check at startup"
```

---

## Phase 4: Streaming & Fast Paths

### Task 8: Add partial streaming to TTS

**Files:**
- Modify: `server/claude_agent_processor.py`

**Step 1: Enable include_partial_messages in options**

```python
options = ClaudeAgentOptions(
    model=self._model,
    system_prompt=SYSTEM_PROMPT,
    include_partial_messages=True,
    max_turns=3,
    effort="low",
    mcp_servers={
        "robot": {
            "type": "http",
            "url": self._mcp_server_url,
        }
    },
    allowed_tools=["mcp__robot__*"],
)
```

**Step 2: Import StreamEvent and handle it**

```python
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    StreamEvent,
)
```

**Step 3: Rewrite _process_with_agent to stream frames directly**

Change return type from `str` to `None` — push frames as text arrives:

```python
async def _process_with_agent(self, user_text: str):
    """Send user text to Claude and stream response frames directly."""
    if not self._client:
        await self.push_frame(LLMTextFrame(text="Agent not connected."))
        return

    if not await self._ensure_mcp_connected():
        await self.push_frame(LLMTextFrame(text="I can't reach the robot control server right now."))
        return

    await self._client.query(user_text)

    has_text = False
    try:
        async for message in self._client.receive_response():
            if isinstance(message, StreamEvent):
                # Parse incremental text from raw API stream events
                event = message.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        has_text = True
                        await self.push_frame(LLMTextFrame(text=delta["text"]))

            elif isinstance(message, AssistantMessage):
                if not self._model_logged and message.model:
                    logger.info(f"Claude model: {message.model}")
                    self._model_logged = True
                # AssistantMessage text may duplicate what StreamEvent already pushed.
                # Only use it as fallback if no streaming happened.
                if not has_text:
                    for block in message.content:
                        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                            has_text = True
                            await self.push_frame(LLMTextFrame(text=block.text))

            elif isinstance(message, ResultMessage):
                if message.is_error:
                    logger.error("Claude Agent SDK execution error")
                    await self.push_frame(LLMTextFrame(text="I hit an error while talking to the robot."))
                    return
                if not has_text and message.result:
                    await self.push_frame(LLMTextFrame(text=str(message.result)))

    except Exception as e:
        logger.error(f"Claude Agent SDK error: {e}")
        await self.push_frame(LLMTextFrame(text="I encountered an error. Please try again."))
        return

    if not has_text:
        await self.push_frame(LLMTextFrame(text="I completed the action but have nothing to report."))
```

**Step 4: Update process_frame caller**

```python
if user_text:
    logger.info(f"User said: {user_text}")
    await self.push_frame(LLMFullResponseStartFrame())
    await self._process_with_agent(user_text)
    await self.push_frame(LLMFullResponseEndFrame())
```

**Step 5: Manual test**

Run: `uv run bot.py`
Expected: TTS begins speaking before the full response is assembled. For multi-sentence responses, first words should play while Claude is still processing.

**Step 6: Commit**

```bash
git add server/claude_agent_processor.py
git commit -m "feat: stream partial text to TTS via StreamEvent"
```

---

### Task 9: Fast-path for deterministic commands (conditional)

**Prerequisite:** Task 5 benchmarks show simple commands still too slow through the LLM.

**Decision point:** Choose one approach:

**Option A: Defer** — If benchmarks show acceptable latency for "stop"/"pause"/"gripper" through LLM with persistent client + low effort, skip this task.

**Option B: Dedicated non-MCP endpoint** — Add REST endpoints to the robot server (`/fast/stop`, `/fast/gripper/open`) that bypass MCP protocol overhead entirely. Simple `httpx.post()` is correct here because these are NOT MCP calls.

**Option C: Proper MCP client** — Use a real MCP client library (e.g. `mcp` Python package) to maintain a separate MCP session for fast-path tool calls. Protocol-correct but more complex.

Do NOT use raw `httpx.post()` with JSON-RPC `tools/call` against the MCP streamable HTTP endpoint — MCP requires proper session handling.

**If implementing Option B:**

Create `server/fast_path.py`:

```python
"""Fast-path dispatcher for deterministic robot commands that bypass the LLM."""

import re

import httpx

# Pattern -> (endpoint path, verbal confirmation)
FAST_PATH_COMMANDS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"^stop$", re.IGNORECASE), "/fast/stop", "Stopping."),
    (re.compile(r"^pause$", re.IGNORECASE), "/fast/pause", "Paused."),
    (re.compile(r"^resume$", re.IGNORECASE), "/fast/resume", "Resuming."),
    (re.compile(r"^open\s*(the\s*)?gripper$", re.IGNORECASE), "/fast/gripper/open", "Done."),
    (re.compile(r"^close\s*(the\s*)?gripper$", re.IGNORECASE), "/fast/gripper/close", "Done."),
]


def match_fast_path(text: str) -> tuple[str, str] | None:
    """Returns (endpoint, confirmation) or None."""
    cleaned = text.strip().rstrip(".!?")
    for pattern, endpoint, confirmation in FAST_PATH_COMMANDS:
        if pattern.match(cleaned):
            return endpoint, confirmation
    return None


async def call_fast_endpoint(base_url: str, endpoint: str) -> bool:
    """Call a dedicated fast-path REST endpoint. Returns True on success."""
    url = base_url.rsplit("/mcp", 1)[0] + endpoint
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.post(url)
        return resp.is_success
```

This requires corresponding endpoints on the robot server side (cross-repo coordination).

---

### Task 10: Add composite MCP tools (server-side coordination)

**Files:**
- This task modifies the **MCP server** (outside pipecat-agent). Document what's needed.

**Step 1: Required MCP server changes**

The robot MCP server should add:

- `move_relative(axis: str, distance_m: float)` — internally reads pose, computes offset, moves
- `ensure_connected(robot_ip: str)` — connects only if not already connected

**Step 2: Update server/prompts.py when available**

Once the MCP server exposes these tools, add to `AVAILABLE MCP TOOLS`:

```
- move_relative (preferred for relative movement like "up a bit", "left")
- ensure_connected
```

And update `MOVEMENT RULES`:

```
- For relative movement (e.g. "up a bit", "left"), prefer move_relative.
- move_relative handles reading current pose internally — do not call get_tcp_pose before it.
```

**Step 3: Commit** (only after MCP server deploys the tools)

```bash
git add server/prompts.py
git commit -m "feat: add composite MCP tools to prompt (move_relative, ensure_connected)"
```

---

## Verification Checklist

After all tasks complete, verify end-to-end:

1. `uv run bot.py` starts without errors
2. Log shows `ClaudeSDKClient connected` and `Robot MCP server connected and ready`
3. Log shows `Claude model: claude-sonnet-4-5` on first utterance
4. "Move up a bit" → calls `get_tcp_pose` then moves (LLM path)
5. "Move to position 0.3 0 0.4" → moves directly without `get_tcp_pose`
6. Second utterance is noticeably faster than with old `query()` approach
7. Disconnecting WebRTC client shows `ClaudeSDKClient disconnected`
8. Killing MCP server mid-session → reconnect log → recovery on next command
