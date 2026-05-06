# Testing

## Default tests

Run deterministic tests from `server/`:

```bash
uv run pytest
```

Default tests must not require live provider API keys, MoveIt MCP, STT/TTS providers, wake-word models beyond existing unit-test fixtures, browser audio, or robot simulation infrastructure.

## Manual live LLM robot smoke tests

Manual live smoke tests send text through the Agent Turn seam:

```text
AgentTurnInput -> LangChainAgentProcessor -> API-backed chat model -> RobotMCPBridge -> MoveIt simulation
```

They do not exercise wake, STT, TTS, browser audio, or the full Pipecat voice pipeline.

### Prerequisites

- `OPENAI_API_KEY` is set.
- The MoveIt MCP server is reachable.
- The UR10 simulation is running in safe simulation mode.

### Run

From `server/`:

```bash
RUN_LIVE_LLM_ROBOT_SMOKE=1 uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -v
```

Optional overrides:

```bash
LIVE_LLM_ROBOT_MCP_URL=http://127.0.0.1:8765/mcp
LIVE_LLM_ROBOT_MODEL=gpt-5.4-mini
LIVE_LLM_ROBOT_EVIDENCE_DIR=evidence/live_smoke
```

### Scenarios

The v1 smoke suite covers:

1. `what is the current position?` — observes pose and does not move.
2. `move up a bit` — observes pose, executes verified bounded +Z movement.
3. `move down a bit` — observes pose, executes verified bounded -Z movement.
4. `move there` — asks for clarification and does not move.

Each case writes minimal JSON evidence under `server/evidence/live_smoke/` by default.

## Exploratory gesture evals

Prompts such as `wave to me` and `draw a star` are exploratory evals. They are useful for behavior review, but they are not part of the pass/fail testing pipeline until their assertions become deterministic and actionable.
