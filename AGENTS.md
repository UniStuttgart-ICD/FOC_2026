# AGENTS.md

Pipecat voice robot agent: a Python cascade voice pipeline for controlling a UR robot through a safe Codex OAuth agent backend.

## Project map

- `server/bot.py` - Pipecat runner entrypoint and transport lifecycle hooks.
- `server/pipeline_builder.py` - Pipeline assembly for transport, wake, STT, agent turn, TTS, metrics.
- `server/voice_runtime/` - Reusable runtime Modules: agent turn seam, profiles, assembly, robot safety, wake command, metrics types.
- `server/openai_codex_agent_processor.py` - Current Codex OAuth agent backend implementation.
- `server/codex_backend_client.py` and `server/codex_auth.py` - Codex backend API client and Pi OAuth credential loading/refresh.
- `server/robot_mcp_bridge.py` - Safe robot MCP tool Adapter used by Codex.
- `server/runtime_profiles.toml` - Runtime profile definitions.
- `server/tests/` - Pytest coverage for config, pipeline assembly, agent backend, robot safety, and Codex behavior.
- `docs/superpowers/specs/` and `docs/superpowers/plans/` - Approved specs and implementation plans.

<important if="you need to run commands to install, test, lint, typecheck, or run the bot">

Run server commands from `server/`.

| Command | What it does |
|---|---|
| `uv sync` | Install/update server dependencies from `pyproject.toml` and `uv.lock` |
| `uv lock` | Refresh `uv.lock` after dependency changes |
| `uv run pytest` | Run all tests |
| `uv run ruff check .` | Check lint/import ordering |
| `uv run pyright .` | Run static type checks |
| `uv run bot.py` | Run the default voice bot profile |
| `uv run bot.py --profile <name>` | Run a specific runtime profile |
</important>

<important if="you are changing the voice runtime architecture">
- Pipecat owns transport, audio frames, wake, STT, TTS, interruption behavior, and pipeline backpressure.
- Agent changes should stay behind `AgentBackend` / `AgentTurnProcessor` unless the approved spec says otherwise.
</important>

<important if="you are changing agent backend selection, auth, or runtime profiles">
- The target architecture is Codex-only. Do not add new Claude support.
- Current Codex auth reads Pi's `~/.pi/agent/auth.json` `openai-codex` OAuth profile.
- Keep `local_current` and `no_wake_debug` as local STT/TTS profiles, but their agent backend should be Codex.
</important>

<important if="you are changing robot tool execution or MCP integration">
- Robot tool calls must go through `RobotMCPBridge` and `voice_runtime.robot_safety` unless a reviewed safety seam replaces them.
- Preserve local safety validation for canonical `moveit_*` tools.
</important>

<important if="you are implementing the future LangGraph migration">
- Use LangGraph as the agent/dialogue orchestration layer only.
- Keep Pipecat pipeline wiring as the primary runtime.
- Add LangGraph behind the existing `AgentBackend` seam before considering deeper pipeline changes.
- Start with `InMemorySaver` for tests/prototype unless the plan explicitly requires durable checkpointing.
</important>
