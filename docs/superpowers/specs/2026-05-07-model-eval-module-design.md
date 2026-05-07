# Model Eval Module Design

## Goal

Add a reusable `model_eval` module for comparing API-backed LangGraph model candidates on this robot-agent system.

The module should answer:

- Does the model make correct robot tool decisions?
- Is it fast enough for a realtime voice robot?
- Does it use the current prompt and MCP tool contract correctly?
- Does it show bounded embodied improvisation for clear gesture requests?

V1 is offline/simulated-first. It uses a deterministic simulated MoveIt tool adapter by default and keeps live MCP as an optional proof adapter.

## Decisions

- Module name: `model_eval`.
- Primary interface: CLI runner plus thin pytest wrapper.
- Primary adapter: simulated MoveIt Robot Tool Adapter.
- Optional adapter: live MCP wrapper around `RobotMCPBridge`.
- V1 scenario pack: `core_robot_commands`.
- V1 scoring: correctness-gated, then latency-weighted.
- Provider cost is optional metadata only, not part of ranking.
- Evidence format is local JSON/JSONL under `server/evidence/model_eval/`.
- The module is built for LangGraph-backed Agent Turn systems, not the full wake/STT/TTS voice pipeline.

## Non-Goals

- Do not run wake, STT, TTS, browser audio, or Pipecat transport.
- Do not move the live robot by default.
- Do not replace unit tests or the existing manual live smoke test.
- Do not make cost a v1 ranking input.
- Do not introduce LangSmith, Langfuse, or external observability as a required dependency.
- Do not generalize beyond LangGraph before this robot eval is stable.

## Architecture

`model_eval` is a Testing module. It evaluates Agent Control and Robot Control through existing seams.

```text
Model Matrix
  -> Eval Runner
  -> Eval Scenario Pack
  -> LangChainAgentProcessor
  -> LangGraphRobotAgent
  -> Eval Tool Adapter
  -> Validator
  -> Live Eval Evidence
```

The module should not own the Robot Agent Prompt, Agent Orchestration, Task Policy Layer, Robot Call Validation, Robot Context, or MoveIt Safety Boundary. It should call the same Agent Turn seam that production and manual smoke tests use.

The deep module interface is small:

```python
run_eval_suite(config) -> EvalSuiteResult
```

Everything else is behind that interface: model construction, scenario iteration, simulated robot state, timing, retry handling, evidence writing, and summary scoring.

## Target Package

```text
server/model_eval/
  __init__.py
  __main__.py          # CLI entrypoint
  config.py            # model matrix and run config parsing
  runner.py            # run_eval_suite
  candidates.py        # ModelCandidate
  scenarios.py         # Scenario, ScenarioPack, core_robot_commands
  adapters.py          # EvalToolAdapter protocol and adapter factory
  simulated_moveit.py  # deterministic simulated MoveIt adapter
  validators.py        # validator registry/wrappers around smoke validators
  scoring.py           # correctness-gated ranking
  evidence.py          # attempts.jsonl, summary.json, optional summary.md
```

The existing `server/test_support/live_robot_smoke.py` can donate behavior, but the reusable eval interface should live in `model_eval`.

## Interfaces

### ModelCandidate

One model configuration:

```text
provider
model
reasoning_effort
api_key_env
label
```

It maps directly to `AgentProfile` so `agent_model_factory.build_agent_chat_model` remains the construction seam.

### EvalScenario

One prompt plus validator:

```text
name
prompt
validator_name
tags
expected_behavior
```

Validators return the existing `ValidationResult` shape.

### EvalToolAdapter

The runner depends on the existing robot adapter interface:

```python
async connect() -> None
async disconnect() -> None
function_tools() -> list[dict[str, Any]]
async call_tool(name: str, arguments: dict[str, Any]) -> str
```

This makes `RobotToolAdapterLike` a real seam. The simulated adapter and live MCP adapter are peer adapters behind the same interface.

## Scenario Pack

V1 includes `core_robot_commands`:

| Scenario | Prompt | Validator |
| --- | --- | --- |
| `current-position` | `what is the current position?` | `validate_position_query` |
| `move-up-bit` | `move up a bit` | `validate_bit_movement(direction="up")` |
| `move-down-bit` | `move down a bit` | `validate_bit_movement(direction="down")` |
| `visible-wave` | `Maive, can you wave to me?` | `validate_wave_motion` |
| `ambiguous-move-there` | `move there` | `validate_ambiguous_clarification` |

The pack intentionally contains both embodied initiative and restraint:

- `visible-wave` should reward bounded improvisation.
- `ambiguous-move-there` should require clarification instead of guessing.

Future packs can add drawing, richer gesture improvisation, repair cases, or provider-specific stress tests.

## Adapters

### SimulatedMoveItAdapter

Default adapter. It should:

- expose canonical MoveIt tool schemas matching `RobotMCPBridge.function_tools()`;
- start from a deterministic UR10 TCP pose;
- mutate internal pose after verified movement tools;
- return structured tool outputs compatible with `RobotContextStore` and existing validators;
- enforce `Robot Call Validation` by calling `validate_robot_tool_call`;
- keep simulation behavior deterministic and cheap.

This adapter is for model fit comparison, not physics fidelity.

### LiveMCPMoveItAdapter

Optional proof adapter. It wraps `RobotMCPBridge` and requires:

- MCP URL;
- ROS 1/MoveIt simulation running;
- explicit CLI flag or pytest env gate.

Live MCP runs are evidence/proof runs, not the default benchmark path.

## Timing and Metrics

V1 records:

- total scenario duration;
- model candidate label;
- scenario name;
- attempt index;
- pass/fail;
- validator reason and details;
- assistant reply;
- recorded tool calls and outputs;
- tool call count;
- model/tool loop count when available from Agent Control state;
- failure exception if the run errors.

Later, `Process Trace` can provide richer spans such as model-call duration and time to first tool call. V1 should not block on that integration.

## Model Fit Score

Correctness is a gate.

Rules:

1. If any required correctness scenario fails, candidate fit is `fail`.
2. Passing candidates are ranked by median total scenario latency.
3. Tie-breakers:
   - fewer tool turns;
   - fewer validation repairs;
   - cleaner final response;
   - consistent bounded improvisation.
4. Cost is recorded only if available later and is not a v1 ranking input.

The summary should make the recommendation explicit, but preserve raw evidence so a developer can override based on qualitative behavior.

## Improvisation Fit

`visible-wave` should record lightweight qualitative notes:

- `initiative`: took action for a clear embodied gesture request;
- `boundedness`: stayed near fresh pose and preserved orientation;
- `expressiveness`: made a visible gesture, not a symbolic micro-motion;
- `overreach`: invented user position, objects, gaze targets, or scene facts.

This is not a replacement for validators. It is supporting evidence for choosing a model whose behavior feels right for the robot.

## CLI

Primary use:

```bash
cd server
uv run python -m model_eval run --matrix evals/model_matrix.toml --pack core_robot_commands
```

Live proof:

```bash
cd server
uv run python -m model_eval run \
  --matrix evals/model_matrix.toml \
  --pack core_robot_commands \
  --adapter live-mcp \
  --mcp-url http://127.0.0.1:8765/mcp
```

Example matrix:

```toml
[[candidates]]
label = "gpt-5.4-mini-medium"
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "gemini-3.1-flash-lite-medium"
provider = "gemini_api"
model = "gemini-3.1-flash-lite-preview"
reasoning_effort = "medium"
api_key_env = "GOOGLE_API_KEY"
```

## Pytest Wrapper

The pytest wrapper should be thin:

```bash
cd server
RUN_MODEL_EVAL=1 uv run pytest tests/live_robot_smoke/manual_model_eval.py -v
```

It should call the same runner and use env overrides for matrix, adapter, scenario pack, sample count, and evidence directory. This keeps ad hoc model comparison and manual regression from drifting.

## Evidence

Default output:

```text
server/evidence/model_eval/<timestamp>/
  attempts.jsonl
  summary.json
  summary.md
```

`attempts.jsonl` contains one record per candidate/scenario/sample attempt.

`summary.json` contains aggregate metrics and the recommended candidate.

`summary.md` is optional but useful for quick review in the terminal or PR notes.

Evidence must not include API keys, raw environment dumps, auth headers, or secrets.

## Reuse

For another LangGraph system:

1. Keep `ModelCandidate`, runner, evidence, timing, and scoring.
2. Replace `ScenarioPack`.
3. Replace `EvalToolAdapter`.
4. Keep the Agent Turn/LangGraph invocation assumption.

The module should not know about UR10 except inside the default `core_robot_commands` pack and `SimulatedMoveItAdapter`.

## Testing Plan

Unit tests:

- parse model matrix TOML;
- build candidates into `AgentProfile`;
- simulate each MoveIt tool output shape;
- validate scoring for pass/fail and latency ranking;
- write evidence without secrets;
- prove CLI argument parsing selects simulated or live adapter;
- prove pytest wrapper calls the runner only when gated.

Manual tests:

- run simulated model matrix with two samples per candidate;
- run one live MCP proof with a known passing model.

Default test suite must not call provider APIs or live MCP.

## Architectural Fit

This design deepens existing testing support.

`live_robot_smoke.py` is useful but currently shallow for model comparison: callers need to assemble model candidates, tool adapters, timing, validators, and evidence themselves. `model_eval` concentrates that complexity behind `run_eval_suite`.

Benefits:

- **Leverage**: adding a new model or scenario pack should not require another bespoke script.
- **Locality**: scoring, evidence, timing, and adapter selection live in one module.
- **Reusable seam**: two Eval Tool Adapters make the robot adapter interface real.
- **Test surface**: the runner and scoring interfaces become the stable test surface.

## Follow-Up Work

- Add a `gesture_improvisation` pack with line, circle, greeting, and excitement prompts.
- Integrate Process Trace for model-call duration and time to first tool call.
- Add optional HTML or notebook report generation if JSON/Markdown is not enough.
- Add provider-specific model discovery helpers after the core runner is stable.
