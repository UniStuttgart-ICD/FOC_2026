# Voice-Controlled UR Robot Agent

## Purpose

Replace MAVE's text-based Gradio agent with a voice-controlled agent. Users speak commands, Claude reasons and controls the UR robot via MCP, and responds with voice.

## Architecture

```
Pipecat Pipeline:
  Mic → SmallWebRTC → Whisper STT → User Aggregator
                                       ↓
                              ClaudeAgentProcessor
                              (ClaudeSDKClient + UR Robot MCP via HTTP)
                                       ↓
          Assistant Aggregator ← Transport ← Kokoro TTS
```

External:
- MAVE's MCP server (`localhost:8000/mcp`) — 20+ robot control tools
- URSim Docker — UR robot simulator

## Components

### ClaudeAgentProcessor (custom Pipecat FrameProcessor)

Wraps `ClaudeSDKClient` with:
- UR Robot MCP server via HTTP transport (`http://localhost:8000/mcp`)
- System prompt adapted from MAVE's robot agent
- OAuth authentication for Anthropic API
- Allowed tools: `mcp__robot__*`

**Input**: `LLMContextFrame` with transcribed user text
**Output**: `LLMFullResponseStartFrame` → `LLMTextFrame`(s) → `LLMFullResponseEndFrame`

Robot tool calls (movement, gripper, status) happen inside the Agent SDK loop — transparent to Pipecat.

### Pipeline

```python
Pipeline([
    transport.input(),
    stt,                        # Whisper STT (local)
    user_aggregator,
    claude_agent_processor,     # Claude Agent SDK + MCP
    tts,                        # Kokoro TTS (local)
    transport.output(),
    assistant_aggregator,
])
```

### File Structure

```
server/
├── bot.py                      # Pipeline setup, entry point
├── claude_agent_processor.py   # Custom Pipecat processor
├── prompts.py                  # Robot agent system prompt
├── .env.example                # ANTHROPIC_API_KEY, MCP_SERVER_URL
└── pyproject.toml              # Dependencies
```

## Dependencies

```toml
dependencies = [
    "pipecat-ai[webrtc,whisper,kokoro,silero,runner]",
    "claude-agent-sdk",
]
```

## Error Handling

- **MCP server down**: Agent tells user it can't reach the robot. Check via `get_mcp_status()`.
- **Robot tool failure**: Agent SDK handles errors in its loop (retry/explain).
- **Agent timeout**: Push error text frame after reasonable timeout.

## Testing

1. Start URSim Docker + MCP server
2. Run bot with `uv run bot.py`
3. Open SmallWebRTC client in browser
4. Speak commands, verify robot moves in URSim VNC viewer (`localhost:6080`)
