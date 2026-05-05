# Codex-Only Agent Cleanup Design

## Goal

Remove Claude agent support and make OpenAI Codex OAuth the only supported agent backend before the LangGraph migration.

## Scope

This cleanup covers the current agent framework only. It does not introduce LangGraph yet and does not change the Pipecat media pipeline.

In scope:
- Delete the Claude backend adapter.
- Remove the `claude` agent provider from profile parsing and validation.
- Remove the `claude-agent-sdk` dependency.
- Convert `local_current` and `no_wake_debug` profiles to `openai_codex_oauth`.
- Remove Claude setup instructions from docs.
- Update tests to assert Codex-only behavior.

Out of scope:
- LangGraph implementation.
- Durable LangGraph checkpointing.
- Changes to STT, TTS, wake word, transport, metrics, or robot safety behavior.
- Changes to the Codex model/tool loop beyond what is required to remove Claude branching.

## Architecture

Pipecat remains the runtime and pipeline owner. The pipeline continues to route finalized user turns through `AgentTurnProcessor`, which delegates to an `AgentBackend`.

After this cleanup, the only production backend adapter is `OpenAICodexAgentProcessor`. `agent_processor_factory.py` should either construct that backend directly for `openai_codex_oauth` or reject unsupported providers. The `AgentBackend` seam stays because it is the future LangGraph insertion point.

## Components

### Agent provider configuration

`server/voice_runtime/profiles.py` should define only one agent provider: `openai_codex_oauth`. Runtime profile validation should fail for any unknown provider, including old `claude` values.

### Agent factory

`server/agent_processor_factory.py` should no longer import or branch to `ClaudeAgentProcessor`. It should return an `AgentTurnProcessor` wrapping `OpenAICodexAgentProcessor` for the Codex provider.

### Runtime profiles

`server/runtime_profiles.toml` should keep all existing profiles, but `local_current` and `no_wake_debug` should use:

```toml
[profiles.<name>.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

This preserves local STT/TTS debug workflows while making the agent path Codex-only.

### Dependencies

`server/pyproject.toml` should remove `claude-agent-sdk`. Existing Codex-related dependencies remain.

### Documentation

`README.md` should remove Claude authentication instructions and state that all profiles use Pi OAuth Codex credentials from Pi's `~/.pi/agent/auth.json` `openai-codex` profile.

## Data flow

No runtime data flow changes are intended:

```text
transport.input()
  -> wake processors
  -> STT
  -> user aggregator
  -> AgentTurnProcessor
  -> OpenAICodexAgentProcessor
  -> TTS
  -> transport.output()
  -> assistant aggregator
```

The Codex backend still receives `AgentTurnInput`, maps Pipecat context messages into Codex input items, executes robot tools through `RobotMCPBridge`, and yields assistant text back to `AgentTurnProcessor`.

## Error handling

Existing Codex error behavior should be preserved:
- OAuth errors yield the credential/auth message.
- Robot/MCP connection failures yield the current robot control server message.
- Codex backend failures yield the existing generic retry message.
- Robot safety validation remains in `RobotMCPBridge` and `voice_runtime.robot_safety`.

Unknown `agent.provider` values should fail during profile parsing with a clear validation error.

## Testing

Tests should cover:
- Runtime profile parsing accepts `openai_codex_oauth`.
- Runtime profile parsing rejects `claude`.
- `local_current` and `no_wake_debug` load with Codex agent settings.
- `create_agent_processor()` creates the Codex-backed `AgentTurnProcessor`.
- No tests import or depend on `ClaudeAgentProcessor`.
- Existing Codex backend, robot bridge, pipeline builder, and config tests still pass.

Verification commands:

```bash
cd server
uv run pytest
uv run ruff check .
uv run pyright .
```

## Migration notes

This cleanup intentionally keeps `AgentBackend` and `AgentTurnProcessor`. They are not legacy; they are the seam for the later LangGraph Codex backend.

After this cleanup, the LangGraph migration can focus on replacing Codex orchestration internals without carrying Claude provider branches or unsafe MCP access paths.
