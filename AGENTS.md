# AGENTS.md

Pipecat voice robot agent: a Python realtime voice pipeline that turns spoken browser commands into MoveIt-planned UR10 robot actions and spoken responses through API-key-backed LangChain/LangGraph Agent Control.

Stack: Python, Pipecat, MCP, LangGraph, native LangChain providers, uv, pytest, Ruff, and Pyright.

## Project map

- `ARCHITECTURE.md` - Target architecture, package seams, invariants, and cross-cutting boundaries.
- `CONTEXT.md` - Voice Runtime, Agent Control, Robot Control, Process Trace, and eval domain language.
- `docs/adr/` - Architecture decisions; `0001-no-deterministic-text-to-motion-fallback.md` is active.
- `docs/agents/` - Issue tracker, triage labels, and domain-doc routing for agent workflows.
- `server/bot.py` - Runner startup, transport creation, profile selection, and client lifecycle hooks.
- `server/pipeline_builder.py` - App composition root for concrete adapters and pipeline task assembly.
- `server/voice_runtime/` - Pipecat/audio runtime: profiles, providers, wake command, Agent Turn seam, assembly, timing, and metrics.
- `server/agent_control/` - Native LangChain API backend, LangGraph orchestration, robot prompt, and Agent Turn factory.
- `server/robot_control/` - Task Policy, Robot Call Validation, Robot Context, MCP/tool adapters, Robot Job Blackboard, Robot Job Worker, and verified execution client.
- `server/process_trace/` - Process Trace core, JSONL writer, trace context, and Pipecat observer adapter.
- `server/model_eval/` - API-backed model-candidate evals, scenario packs, validators, scoring, simulated MoveIt adapter, and evidence writing.
- `server/user_sensing/` - Advisory Vizor user-sensing context and MCP bridge.
- `server/voice_modulation/` - Voice Mod Lab, settings, DSP, preview, and post-TTS processor.
- `server/wake_tuning/` - Wake tuning app, detector helpers, log paths, and local saved settings.
- `server/runtime_profiles.toml` - Single bundled app profile: `hybrid_gemini_live_tts`.
- `server/tests/` - Deterministic pytest coverage plus gated manual live/eval suites.
- `.pi/plans/`, `docs/superpowers/specs/`, and `docs/superpowers/plans/` - Plans/specs and historical design provenance; verify against current code and docs before treating as authoritative.

Current behavior and terminology should line up with `ARCHITECTURE.md`, `CONTEXT.md`, and `server/runtime_profiles.toml`.

<important if="you are about to run multiple independent tasks, tool calls, searches, checks, or implementation steps">
- Run independent work in parallel when there is no ordering dependency or file conflict risk.
- Keep work sequential only for real dependencies, shared state, or conflict-prone edits.
</important>

<important if="you are writing an implementation plan">
- Save implementation plans into `.pi/plans/`.
</important>

<important if="you need to run commands to install, test, lint, typecheck, run local tools, or run the bot">

Run server commands from `server/`.

| Command | What it does |
|---|---|
| `uv sync` | Install/update server dependencies from `pyproject.toml` and `uv.lock` |
| `uv lock` | Refresh `uv.lock` after dependency changes |
| `uv run pytest` | Run deterministic tests |
| `uv run ruff check .` | Check lint/import ordering |
| `uv run pyright .` | Run static type checks |
| `uv run bot.py` | Run the default voice bot profile |
| `uv run bot.py --profile <name>` | Run a specific runtime profile |
| `uv run python -m wake_tuning.app` | Run the wake tuning app on `127.0.0.1:9010` |
| `uv run uvicorn voice_modulation.app:app --host 127.0.0.1 --port 8897` | Run the Voice Mod Lab |
| `uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands` | Run simulated model evals |

Tests marked `live`, `llm`, `native_llm`, or `robot_sim` require explicit credentials or external stacks and are not normal CI.
</important>

<important if="you need API, SDK, or library documentation">
- Prefer `chub` before relying on memory or general web search.
- On first `chub` use in a session, run `chub --help` and follow its current guidance.
- Typical flow: `chub search "<library or API> <topic>" --json`, choose the best ID, then `chub get <id> --lang <py|js|ts>` when language matters.
- Do not include secrets, private code, or private architecture details in `chub feedback` or annotations.
</important>

<important if="you need to self-verify work that can be checked in a browser or web UI">
- Run `playwright-cli --help` first to inspect available commands and options.
- Use `playwright-cli` to inspect, interact, and capture screenshots or video when browser-visible claims matter.
- Do not claim browser/UI work is complete without command output, screenshots, video, tests, or another concrete artifact.
</important>

<important if="a coding task has unclear requirements, multiple plausible interpretations, or hidden tradeoffs">
- State assumptions explicitly instead of silently choosing an interpretation.
- Ask before implementing when the safer interpretation cannot be discovered from local context.
- Present meaningful tradeoffs when choices affect scope, behavior, risk, or maintainability.
</important>

<important if="you are designing, implementing, or modifying code">
- Apply YAGNI: implement only the strict requirements of the current task.
- Apply KISS: prefer simple, explicit, immediately understandable code over complex abstractions.
- Apply DRY only when abstraction clearly improves maintainability.
</important>

<important if="you are editing existing files">
- Make surgical changes; every changed line should trace to the user's request.
- Match the existing style and avoid drive-by refactors, reformatting, renames, or unrelated cleanup.
- Clean up only unused imports, variables, functions, or files made unused by your change.
- Mention unrelated dead code or issues; do not delete or fix them unless asked.
</important>

<important if="you are fixing a bug, adding validation, refactoring, or completing a multi-step code change">
- Convert the request into verifiable success criteria before changing code.
- Reproduce bugs with a test or concrete check before fixing when practical.
- For refactors, verify behavior before and after the change.
- Loop until the relevant checks pass or clearly report the blocker.
</important>

<important if="you are making claims, recommendations, or saying something works">
- Be honest about uncertainty and limitations.
- If you do not know, say "I don't know."
- Provide evidence, reasoning, command output, tests, documentation links, or code references for important claims.
</important>

<important if="you are creating, triaging, or preparing issues or PRDs">
- Issues and PRDs are tracked in GitHub Enterprise at `github.tik.uni-stuttgart.de/ac147490/Robot_buddy`; see `docs/agents/issue-tracker.md`.
- Use canonical triage labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`; see `docs/agents/triage-labels.md`.
</important>

<important if="you are changing architecture, module placement, package seams, or domain terminology">
- Follow `ARCHITECTURE.md` as the target map.
- Use `CONTEXT.md` for project language; update it when domain terms or ownership decisions change.
- Target runtime/control packages are `voice_runtime`, `agent_control`, `robot_control`, `process_trace`, `model_eval`, and `user_sensing`.
- Keep app wiring in `pipeline_builder.py`.
</important>

<important if="you are changing imports between Voice Runtime, Agent Control, Robot Control, Process Trace, or app wiring">
- `pipeline_builder.py` is the composition root and may import Voice Runtime, Agent Control, Robot Control, Process Trace, and User Sensing packages.
- `voice_runtime` must not import `agent_control` or `robot_control`.
- `agent_control` may import `robot_control` and only these Voice Runtime seams: `voice_runtime.agent_turn`, `voice_runtime.profiles`, `voice_runtime.agent_providers`, and `voice_runtime.timing`.
- `robot_control` must not import `voice_runtime` or `agent_control`.
- Pure `process_trace` core modules must not import Pipecat, LangGraph, LangChain, MCP, Voice Runtime, Agent Control, or Robot Control.
- Add/update `server/tests/test_orthogonal_imports.py` and `server/tests/test_robot_control_imports.py` when package boundaries change.
</important>

<important if="you are changing Pipecat pipeline wiring, wake, STT, TTS, interruption behavior, voice modulation, or voice metrics">
- Voice Runtime owns transport, audio frames, wake command, STT, TTS, interruption behavior, pipeline backpressure, processor ordering, post-TTS voice modulation, and voice metrics.
- Keep Robot Control, LangChain/LangGraph orchestration, and MoveIt logic out of `voice_runtime`.
- `voice_runtime.assembly` owns processor ordering; `pipeline_builder.py` constructs concrete adapters.
- Wake tuning and Voice Mod Lab save ignored local state under `server/state/`; promote shared defaults by editing `server/runtime_profiles.toml`.
</important>

<important if="you are changing bot.py">
- Keep `bot.py` as the runner/lifecycle shell only.
- Do not construct STT/TTS internals, Agent Backend internals, robot tools, task policy, graph nodes, metrics internals, or process tracing internals in `bot.py`.
</important>

<important if="you are changing agent backend selection, API-key auth, runtime profiles, or model controls">
- The only bundled runtime profile is `hybrid_gemini_live_tts`.
- Keep `server/runtime_profiles.toml` single-profile unless a new architecture decision asks for a matrix again.
- The main profile uses OpenAI Realtime Whisper STT, Gemini Live TTS, and `gemini_api` with `GOOGLE_API_KEY`.
- Current main agent model settings are `gemini-3.1-flash-lite-preview`, `reasoning_effort = "high"`, and `temperature = 0.7`.
- Agent model controls live in `voice_runtime.profiles.AgentProfile`: `provider`, `model`, `reasoning_effort`, `temperature`, `api_key_env`, and `thinking_budget`.
- Map `AgentProfile` controls to LangChain provider kwargs only in `agent_control.model_factory`.
- Agent profiles must use native LangChain API providers: `openai_api`, `gemini_api`, or `anthropic_api`.
- Runtime profile parsing belongs to `voice_runtime`; concrete profile files remain app configuration.
- Do not reintroduce Codex OAuth profile support unless a new architecture decision asks for it.
</important>

<important if="you are changing LangGraph or Agent Orchestration">
- LangGraph is Agent Orchestration behind the `AgentBackend` / `AgentTurnProcessor` seam.
- LangGraph must not own Pipecat transport, audio frames, wake handling, STT/TTS, interruption behavior, or pipeline ordering.
- Start with `InMemorySaver` for tests/prototype unless the plan explicitly requires durable checkpointing.
- Do not synthesize MoveIt action calls from loose user-text substrings after a model tool-call failure; follow `docs/adr/0001-no-deterministic-text-to-motion-fallback.md`.
</important>

<important if="you are changing robot tool execution, MCP integration, MoveIt workflows, or verified execution">
- Robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack.
- Route movement through canonical MoveIt tools: current-pose observation, planning-scene object grounding, planning-only free/cartesian/pick/place tools, explicit execute, gripper, attach, and verification.
- Do not expose combined `moveit_plan_and_execute_*` tools by default.
- Use `moveit_execute_plan` only with a recent returned `raw.plan_name` and explicit execution intent.
- Tool failures should return concise structured corrections with `ok`, `error`, `correction`, `retryable`, and `suggested_next_tool` when applicable.
- Verified real robot execution is host-side actuation, not an MCP server; avoid RTDE Control for production motion or gripper control.
</important>

<important if="you are changing object-relative motion, pick/place behavior, or planning-scene grounding">
- Use `moveit_list_scene_objects` before object-relative or pick/place tasks, then `moveit_get_object_context` for one returned object name.
- Pick/place planning returns candidate plans; execution and object attachment/release proof are separate steps.
- After executing a pick/place plan, verify attachment or release evidence before claiming the object moved, was picked, or was placed.
</important>

<important if="you are changing Robot Call Validation">
- Robot Call Validation checks tool names, `robot_name`, argument shape, target bounds, timeouts, and executable plan names.
- Robot Call Validation is not Task Policy and is not the source of movement safety.
- Implementation lives in `server/robot_control/call_validation.py`.
</important>

<important if="you are implementing or changing Task Policy">
- Implementation lives in `server/robot_control/task_policy.py`.
- Task Policy v1 checks only obvious pre-tool preconditions: fresh pose before motion, no blind execute, explicit execution intent, and basic gripper/attach ordering.
- A blocked Task Policy Decision should return structured feedback with correction text and a suggested next tool.
- Task Policy does not prove semantic task safety, object perception, holding state, arbitrary pick/place workflows, or emergency stop.
</important>

<important if="you are changing Robot Context">
- Robot Context is advisory state only.
- Require fresh `moveit_get_current_pose` before movement, relative commands, retries, or safety-sensitive actions.
- Implementation lives in `server/robot_control/context.py`.
</important>

<important if="you are changing Robot Job Blackboard, Robot Job Worker, or long-running robot execution">
- Robot jobs live in `server/robot_control/job_board.py` and `server/robot_control/job_worker.py`.
- Agent Control queues exact robot jobs; Robot Job Worker validates and executes the submitted tool call without inventing new calls, repairing arguments, or making LLM decisions.
- Terminal job events should be typed and observable by callers.
</important>

<important if="you are changing the robot agent prompt or tool descriptions">
- Prompt behavior should be outcome-oriented: observe when state matters, plan before execution, execute only returned valid plans, verify results, and respond briefly.
- Keep prompt/tool descriptions aligned with canonical `moveit_*` tools exposed by Robot Call Validation and the Robot Tool Adapter.
- Do not mention stale tools such as `move_to_position`, `move_linear`, `get_tcp_pose`, or `connect_robot` unless they are reintroduced through the robot tool adapter/MCP contract.
- Avoid stale safety terms such as "Robot Safety", "Motion Safety Layer", or "Safety Coverage"; use `Robot Call Validation`, `Task Policy`, and `MoveIt Safety Boundary`.
</important>

<important if="you are changing Vizor user sensing or user-sensing context">
- Vizor user sensing is advisory grounding for references like "this", "that", "there", and "near me".
- It is not a movement-safety boundary and is not proof of user intent when missing, stale, or low confidence.
- The long-running Vizor MCP owns attention history; Pipecat stores the returned summary in `server/user_sensing`.
</important>

<important if="you are changing Process Trace, trace records, or metrics persistence">
- Process Trace observes behavior; it does not own runtime behavior, model calls, policy decisions, validation, MCP execution, or robot context mutation.
- Pure `process_trace` core stays dependency-light; Pipecat-specific tracing belongs in `process_trace.pipecat_observer`.
- Runtime profiles configure base JSONL paths; `pipeline_builder.py` expands them into session-scoped files under `server/logs/`.
</important>

<important if="you are changing model evals, live robot smoke tests, eval evidence, or manual robot testing">
- Default tests must stay deterministic and require no live provider keys, MoveIt MCP, browser audio, or robot simulation stack.
- Live LLM robot smoke tests and live MCP model evals are manual opt-in only.
- `model_eval` evaluates Agent Control and Robot Control through existing seams; it must not own prompts, Task Policy, Robot Call Validation, the MoveIt Safety Boundary, Pipecat transport, wake, STT, TTS, or pipeline ordering.
- Evidence should be minimal local JSON/JSONL suitable for review and replay.
</important>
