# Voice-Controlled UR Robot Agent — Review Brief

> Historical note: this review brief describes an early Claude SDK direction and stale robot tool names. Current architecture and domain language live in `ARCHITECTURE.md`, `CONTEXT.md`, and `server/runtime_profiles.toml`.

**Date:** 2026-03-19
**Author:** Samuel (via Claude Code)
**Status:** Plan complete, awaiting review before implementation
**Plan document:** `docs/plans/2026-03-19-voice-robot-agent-plan.md`
**Design document:** `docs/plans/2026-03-19-voice-robot-agent-design.md`

---

## What are we building?

A voice-controlled robot agent. Users speak natural language commands into a microphone, an AI agent reasons about the command and controls a UR robot arm, then responds with synthesized speech.

**Example interaction:**
> User (voice): "Move the robot up a bit"
> Agent: calls `get_tcp_pose()` → gets current position → calls `move_to_position()` with Z+50mm
> Agent (voice): "Moved up 50mm to Z=250mm."

## Why?

MAVE (Multi-Actor-Interface-Library) currently has a **text-based** Gradio chat interface for robot control. This project replaces that with a **hands-free voice interface** — critical for an embodied robot scenario where the operator's hands may be occupied.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Pipecat Voice Pipeline                 │
│                                                           │
│  Microphone → WebRTC → Whisper STT → User Aggregator     │
│                                         ↓                 │
│                                ClaudeAgentProcessor       │
│                                ┌─────────────────┐       │
│                                │ Claude Agent SDK │       │
│                                │      ↕           │       │
│                                │ UR Robot MCP ────┼──→ localhost:8000/mcp
│                                └─────────────────┘       │
│                                         ↓                 │    ↕
│            Assistant Aggregator ← WebRTC ← Kokoro TTS    │  URSim Docker
└─────────────────────────────────────────────────────────┘  (or real robot)
```

### Components

| Component | Technology | Runs where | Role |
|-----------|-----------|------------|------|
| Voice I/O | SmallWebRTC (Pipecat) | Browser ↔ bot server | Audio capture and playback |
| STT | Whisper | Local (GPU/CPU) | Speech → text |
| TTS | Kokoro | Local (GPU/CPU) | Text → speech |
| Reasoning | Claude Agent SDK | API call (Anthropic) | Interprets commands, plans actions, calls tools |
| Robot control | MAVE MCP Server | localhost:8000 | 20+ tools: move, gripper, status, stop |
| Simulator | URSim Docker | localhost:6080 (VNC) | UR robot simulation |

### What's new vs. what exists

| | Before (MAVE) | After (this project) |
|---|---|---|
| **Interface** | Gradio text chat | Voice (WebRTC) |
| **LLM** | Google Gemini (via LangGraph) | Claude (via Agent SDK) |
| **MCP connection** | langchain-mcp-adapters | Claude Agent SDK native MCP |
| **Framework** | LangGraph ReAct agent | Claude Agent SDK agent loop |
| **STT/TTS** | None (text only) | Whisper + Kokoro (both local) |
| **Sensors** | HoloLens gaze + custom MongoDB sensors | Not included in v1 |

## Key design decisions

### 1. Claude Agent SDK instead of LangGraph

The Claude Agent SDK has **native MCP support** — it connects directly to the robot MCP server without adapters. It also handles the full agent loop (reasoning → tool calls → observation → response) internally, reducing custom code.

Trade-off: Anthropic API dependency instead of Google. Uses OAuth for authentication.

### 2. Custom Pipecat FrameProcessor (not LLM service subclass)

Pipecat's built-in LLM services (GoogleLLMService, AnthropicLLMService) expect a standard chat API. The Agent SDK is different — it's an autonomous agent loop that may make multiple tool calls before producing a response. A custom `FrameProcessor` gives us full control over the frame flow.

Trade-off: We lose Pipecat's built-in streaming (token-by-token TTS). The agent completes its full reasoning+tool-calling loop before TTS starts. This adds latency but ensures the robot actions complete before the voice response.

### 3. Stateless query() with manual history

Using `query()` per turn (with conversation history passed in the prompt) rather than `ClaudeSDKClient` (persistent session). Simpler lifecycle management within Pipecat's frame processing. Can upgrade to `ClaudeSDKClient` later for better context handling.

### 4. No sensors in v1

Dropped HoloLens gaze tracking, user position, and custom MongoDB sensors to keep v1 simple. The system prompt still includes directional commands and coordinate system rules. Sensors can be added in a follow-up.

## Implementation plan summary

| Task | What | Files | Effort |
|------|------|-------|--------|
| 1 | Update dependencies (drop Google, add claude-agent-sdk) | `pyproject.toml`, `.env.example` | Small |
| 2 | Create robot system prompt | `server/prompts.py` (new) | Small |
| 3 | Create ClaudeAgentProcessor | `server/claude_agent_processor.py` (new) | Medium |
| 4 | Update bot.py pipeline | `server/bot.py` | Small |
| 5 | Integration test with URSim | Manual testing | Medium |

Total new code: ~150 lines across 2 new files + minor edits to 3 existing files.

## Risks and open questions

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Latency**: Agent SDK tool-calling loop may take 5-15s before voice response | Medium | Acceptable for v1. Future: stream partial responses, add "thinking" audio cue |
| **MCP transport type**: Design says `sse`, MAVE server uses `streamable_http`. May need adjustment | Low | Test both `sse` and `http` types during integration |
| **OAuth setup**: Claude Agent SDK OAuth may need additional configuration | Low | Fall back to API key if OAuth isn't straightforward |
| **Whisper model size**: Local Whisper accuracy depends on model size vs available VRAM | Low | Default to `base` model, configurable via env var |
| **No streaming TTS**: Full response must complete before speech starts | Medium | Future improvement: sentence-level chunking |

## How to test

**Prerequisites:**
1. Docker running with URSim: `docker compose -f Multi-Actor-Interface-Library/docker/ursim/docker-compose.yml up`
2. Robot powered on via VNC at `http://localhost:6080` (password: `easybot`)
3. MAVE MCP server: `cd Multi-Actor-Interface-Library && uv run python -m mcp_server.server`

**Run:**
```bash
cd pipecat-agent/server
cp .env.example .env  # Configure API keys
uv sync
uv run bot.py
```

**Test commands (voice):**
1. "Connect to the robot" → should call connect_robot
2. "What's the robot's position?" → should call get_tcp_pose, read back
3. "Move up a bit" → should move robot +50mm Z in URSim
4. "Stop" → should emergency stop

## File structure after implementation

```
pipecat-agent/
├── docs/plans/
│   ├── 2026-03-19-voice-robot-agent-design.md
│   ├── 2026-03-19-voice-robot-agent-plan.md
│   └── 2026-03-19-voice-robot-agent-review-brief.md  ← this file
├── server/
│   ├── bot.py                      # Pipeline: WebRTC → STT → Agent → TTS
│   ├── claude_agent_processor.py   # ClaudeAgentProcessor (new)
│   ├── prompts.py                  # Robot control system prompt (new)
│   ├── pyproject.toml              # Updated dependencies
│   ├── .env.example                # Updated env vars
│   └── .env                        # Local config (not committed)
└── README.md
```
