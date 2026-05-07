# AGENTS.md

Pipecat voice robot agent: a Python cascade voice pipeline for controlling a UR robot through a MoveIt-routed API-key LangGraph agent backend.

## Project map

- `ARCHITECTURE.md` - Target architecture map and package seams.
- `CONTEXT.md` - Voice Runtime and robot-control domain language.
- `server/bot.py` - Runner startup, transport creation, profile selection, and client lifecycle hooks.
- `server/pipeline_builder.py` - App composition root for concrete adapters and pipeline task assembly.
- `server/voice_runtime/` - Reusable Pipecat/audio runtime Modules: profiles, voice providers, wake command, Agent Turn seam, assembly, and metrics.
- `server/robot_control/` - Robot Control Modules: Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context.
- `server/agent_control/` - Agent Control Module: native LangChain API Backend, LangGraph Agent Orchestration, Robot Agent Prompt, and Agent Turn factory.
- `server/runtime_profiles.toml` - App runtime profile definitions.
- `server/tests/` - Pytest coverage for config, pipeline assembly, Agent Backend, Agent Orchestration, Robot Call Validation, and Agent Control behavior.
- `.pi/plans/`, `docs/superpowers/specs/`, and `docs/superpowers/plans/` - Approved specs and implementation plans.

## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Enterprise at `github.tik.uni-stuttgart.de/ac147490/Robot_buddy`. See `docs/agents/issue-tracker.md`.

### Triage labels

Use canonical triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context repo: read `CONTEXT.md`, `ARCHITECTURE.md`, and relevant ADRs in `docs/adr/`. See `docs/agents/domain.md`.

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

<important if="you are changing architecture, module placement, or package seams">
- Follow `ARCHITECTURE.md` as the target map.
- Target packages are `voice_runtime`, `agent_control`, and `robot_control`.
- Robot-side policy, context, validation, and adapter changes belong under `server/robot_control/`.
- Keep app wiring in the composition root; Agent Control implementation belongs under `server/agent_control/`.
- Update `CONTEXT.md` when domain terms or ownership decisions change.
</important>

<important if="you are changing imports between Voice Runtime, Agent Control, or Robot Control">
- `pipeline_builder.py` is the composition root and may import all three packages.
- `voice_runtime` must not import `agent_control` or `robot_control`.
- `agent_control` may import `robot_control` and only these Voice Runtime seams: `voice_runtime.agent_turn`, `voice_runtime.profiles`, `voice_runtime.agent_providers`, and `voice_runtime.timing`.
- `robot_control` must not import `voice_runtime` or `agent_control`.
- Add/update structural import tests when target packages exist.
</important>

<important if="you are changing Pipecat pipeline wiring, wake, STT, TTS, interruption behavior, or metrics">
- Voice Runtime owns transport, audio frames, wake, STT, TTS, interruption behavior, pipeline backpressure, processor ordering, and voice metrics.
- Keep robot-control and LangChain/LangGraph logic out of `voice_runtime`.
- `pipeline_builder.py` constructs concrete adapters; `voice_runtime.assembly` owns processor ordering.
</important>

<important if="you are changing bot.py">
- Keep `bot.py` as the runner/lifecycle shell only.
- Do not construct STT/TTS internals, Agent Backend internals, robot tools, task policy, or graph nodes in `bot.py`.
</important>

<important if="you are changing agent backend selection, API-key auth, or runtime profiles">
- The default live profile `hybrid_low_latency` must use `gemini_api` with `GOOGLE_API_KEY`, not Codex OAuth.
- Agent profiles must use native LangChain API providers: `openai_api`, `gemini_api`, or `anthropic_api`.
- Do not reintroduce Codex OAuth profile support unless a new architecture decision asks for it.
- Runtime profile parsing belongs to `voice_runtime`; concrete profile files remain app configuration.
</important>

<important if="you are changing LangGraph or Agent Orchestration">
- LangGraph is Agent Orchestration behind the `AgentBackend` / `AgentTurnProcessor` seam.
- LangGraph must not own Pipecat transport, audio frames, wake handling, STT/TTS, interruption behavior, or pipeline ordering.
- Start with `InMemorySaver` for tests/prototype unless the plan explicitly requires durable checkpointing.
</important>

<important if="you are changing robot tool execution, MCP integration, or MoveIt workflows">
- Robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack.
- Route robot movement through MoveIt workflows; do not describe local validation as the source of movement safety.
- Prefer agent-friendly workflow tools over raw MoveIt API wrappers: current-pose observation, free/cartesian planning, plan-and-execute workflows, execute, gripper, attach.
- Tool failures should return concise structured corrections with `ok`, `error`, `correction`, `retryable`, and `suggested_next_tool` when applicable.
</important>

<important if="you are changing Robot Call Validation">
- Robot Call Validation checks tool names, `robot_name`, argument shape, target bounds, timeouts, and executable plan names.
- Robot Call Validation is not Task Policy and is not the source of movement safety.
- Implementation lives in `server/robot_control/call_validation.py`.
</important>

<important if="you are implementing or changing Task Policy">
- Target home is `server/robot_control/task_policy.py`.
- Task Policy v1 checks only obvious pre-tool preconditions: fresh pose before motion, no blind execute, and basic gripper/attach ordering.
- A blocked Task Policy Decision should return structured feedback with correction text and a suggested next tool.
- Task Policy does not prove semantic task safety, object perception, holding state, arbitrary pick/place workflows, or emergency stop.
</important>

<important if="you are changing Robot Context">
- Robot Context is advisory state only.
- Require fresh `moveit_get_current_pose` before movement, relative commands, retries, or safety-sensitive actions.
- Implementation lives in `server/robot_control/context.py`.
</important>

<important if="you are changing the robot agent prompt or tool descriptions">
- Prompt behavior should be outcome-oriented: observe when state matters, plan before execution, execute only returned valid plans, verify results, and respond briefly.
- Keep prompt/tool descriptions aligned with canonical `moveit_*` tools exposed by Robot Call Validation and the Robot Tool Adapter.
- Do not mention stale tools such as `move_to_position`, `move_linear`, `get_tcp_pose`, or `connect_robot` unless they are reintroduced through the robot tool adapter/MCP contract.
</important>
