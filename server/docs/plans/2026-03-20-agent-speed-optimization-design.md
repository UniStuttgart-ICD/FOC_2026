# Agent Speed Optimization Design

## Problem

Every user utterance spawns a new Claude CLI subprocess, re-establishes the MCP connection, re-discovers tools, and rebuilds the full prompt from scratch. This adds 1-3s of overhead per turn on top of model inference.

Secondary issues: unnecessary tool round-trips, no partial streaming to TTS, no fast-path for deterministic commands.

## Scope

Files affected:
- `claude_agent_processor.py` — primary rewrite target
- `prompts.py` — prompt rule changes
- `bot.py` — client lifecycle wiring

Out of scope: STT/VAD tuning, Haiku model routing, interrupt/barge-in handling.

## Design

### 1. Pin model explicitly, log it

Set `model` on `ClaudeAgentOptions`. Log `message.model` from first `AssistantMessage` to confirm.

```python
options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    ...
)
```

Add to `.env.example`:
```
CLAUDE_MODEL=claude-sonnet-4-5
```

### 2. Switch from `query()` to `ClaudeSDKClient`

Replace the stateless `query()` call with a persistent `ClaudeSDKClient` that lives for the duration of the WebRTC session.

Lifecycle:
- `connect()` in processor init or on first frame
- `client.query(user_text)` per turn (sends into existing session)
- `disconnect()` on pipeline cancellation / client disconnect

Key change: conversation history is maintained by the SDK natively. Remove the manual `self._history` list and prompt reconstruction.

System prompt delivered once via the initial `connect(prompt=SYSTEM_PROMPT)` call.

Caveat: `ClaudeSDKClient` cannot cross async contexts (SDK v0.1.48 limitation). The client must be created and used within the same pipecat pipeline runner context.

### 3. Reduce `get_tcp_pose` to relative moves only

Change prompt rule from:
```
- Before movement, call get_tcp_pose.
```
To:
```
- Before relative movement (e.g. "up a bit", "left"), call get_tcp_pose to get the current position.
- For absolute coordinates, move directly without reading pose first.
```

### 4. Add `max_turns` and low effort settings

Limit agent loop iterations to prevent wandering:

```python
options = ClaudeAgentOptions(
    max_turns=3,
    ...
)
```

Start at 3 (allows: tool call + result + response). Tune down if benchmarks show it's safe.

For thinking/effort — set via SDK if exposed, otherwise via `extra_args`.

### 5. Stream partial text to TTS

Instead of accumulating all text into `response_text` and pushing one `LLMTextFrame` at the end, push each `AssistantMessage` text block immediately as it arrives.

```python
async for msg in client.receive_response():
    if isinstance(msg, AssistantMessage):
        for block in msg.content:
            if block.type == "text":
                await self.push_frame(LLMTextFrame(text=block.text))
```

TTS begins synthesizing while the agent may still be executing tool calls.

### 6. Add MCP reconnect error handling

With a persistent client, a transient MCP disconnect no longer self-heals (unlike `query()` which re-establishes each call).

On MCP-related errors:
1. Call `client.reconnect_mcp_server("robot")`
2. Retry the failed turn once
3. If still failing, report to user

Also check `get_mcp_status()` after reconnect to confirm.

### 7. Deterministic fast-path for simple commands

Before sending to Claude, match user text against a small set of deterministic intents:

| Intent pattern | MCP tool | Args |
|---|---|---|
| "stop" | `stop` | — |
| "pause" | `pause` | — |
| "resume" | `resume` | — |
| "open gripper" | `control_gripper` | `{action: "open"}` |
| "close gripper" | `control_gripper` | `{action: "close"}` |

Implementation: simple keyword/regex match in `process_frame` before calling `_process_with_agent`. Dispatch directly to MCP via HTTP POST, bypassing Claude entirely.

This is the biggest latency win for the most common commands — zero model inference.

### 8. Composite MCP tools

Add higher-level tools to the robot MCP server to reduce multi-step tool chains:

- `move_relative(axis, distance_m)` — reads current pose + applies offset internally
- `ensure_connected(robot_ip)` — connects only if not already connected

This eliminates the `get_tcp_pose` → math → `move_to_position` chain for relative moves, reducing it to a single tool call.

Requires changes on the MCP server side (out of scope for this processor rewrite, but should be coordinated).

### 9. Pre-warm with `get_mcp_status()` at connect time

After `client.connect()`, immediately call `await client.get_mcp_status()` and log/fail-fast if the robot server isn't connected. This surfaces connectivity problems before the first user utterance instead of discovering them lazily.

## Implementation Order

Phases grouped by dependency:

**Phase 1 — Foundation (items 1-2)**
Pin model + switch to ClaudeSDKClient. These are coupled since the client rewrite touches the same code.

**Phase 2 — Tuning (items 3-4-5)**
Prompt changes, max_turns, and streaming. Independent of each other, can be done in parallel.

**Phase 3 — Resilience (items 6, 9)**
MCP reconnect handling and pre-warm. Depend on Phase 1 (persistent client).

**Phase 4 — Fast paths (items 7-8)**
Deterministic dispatcher and composite MCP tools. Independent of Phases 1-3 but benefit from benchmarking the earlier phases first.
