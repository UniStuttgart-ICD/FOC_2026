# Pipecat Voice Robot Agent Architecture

This file is the target architecture map for agents and maintainers. It may contain only the Harness Engineering architecture-doc outline: Bird's Eye View, Code Map, Architecture Invariants, and Cross-Cutting Concerns. API boundaries belong inside the Code Map. Do not put runtime logs, debugging hypotheses, active plans, source-file inventories, or volatile implementation notes here.

## Bird's Eye View

The system turns a spoken user command into a robot action and a spoken response.

Two planes keep the architecture understandable:

1. **Voice Runtime plane**: owns realtime audio transport, wake command handling, STT, user/assistant aggregation, TTS, interruption behavior, pipeline ordering, and voice metrics.
2. **Agent/Robot Control plane**: owns intent handling, Codex-backed Agent Orchestration, deterministic robot task policy, robot call validation, MoveIt tool execution, and robot context.

The high-level flow is:

```text
Browser audio
  -> Voice Runtime
  -> Agent Turn
  -> Agent Orchestration
  -> Task Policy Layer
  -> Robot Call Validation
  -> MoveIt MCP
  -> UR10 simulation
  -> Agent response
  -> Voice Runtime speech output
```

Movement safety is delegated to MoveIt planning/execution and the robot simulation stack. The voice agent must route movement through MoveIt workflows. Local validation may exist for ergonomics and clearer errors, but it is not the source of movement safety.

## Code Map

This section names the target Modules and seams. Use symbol search for the mentioned names; do not rely on this file as a source-file inventory.

### `voice_runtime`

`voice_runtime` is the reusable Voice Runtime Module. It owns Pipecat-facing runtime semantics and must stay independent of Codex, MCP, and robot task policy.

It contains these target submodules:

- **Runtime Profile**: parses runtime profiles and provider policy without constructing processors.
- **Voice Providers**: constructs STT/TTS adapters for the Voice Runtime; `providers.py` is the legacy top-level placement.
- **Voice Command**: owns wake detection, pre-buffer replay, wake phrase stripping, and rearming.
- **Agent Turn**: exposes the AgentBackend seam and wraps one backend turn in Pipecat LLM frames.
- **Voice Runtime Assembly**: owns processor ordering.
- **Voice Metrics**: owns semantic turn timing. Pipecat frame observation is a Voice Runtime adapter; JSONL persistence is app configuration.

**API Boundary:** `AgentBackend` is the seam from Voice Runtime into Agent/Robot Control. Voice Runtime knows that an Agent Backend can connect, disconnect, and run one Agent Turn; it does not know how LangChain, LangGraph, MCP, or MoveIt work.

### `agent_control`

`agent_control` is the target Module for API-key-backed LangChain Agent Orchestration.

It contains these target submodules:

- **LangChain API Backend**: the target Agent Backend; it uses native LangChain chat models with provider API keys.
- **Agent Orchestration**: the LangGraph ReAct-style loop that calls the LangChain model, executes returned robot tools, observes robot state, and repeats until done or blocked.
- **Robot Agent Prompt**: the concise behavior prompt aligned with Agent Orchestration, robot tool feedback, and the robot tool contract.

**API Boundary:** `agent_control` satisfies `voice_runtime.AgentBackend` and depends on `robot_control` for robot execution. LangGraph is an implementation of Agent Orchestration behind the Agent Turn seam; it must not own Pipecat transport, audio frames, wake handling, STT/TTS, interruption behavior, or pipeline ordering.

### `robot_control`

`robot_control` is the Module for robot-side control concerns.

It contains these target submodules:

- **Task Policy Layer**: deterministic pre-tool checks for obvious robot-step preconditions.
- **Robot Call Validation**: structural and local tool-call validation for allowed MoveIt tools, UR10 arguments, target bounds, timeouts, and executable plan names.
- **Robot Tool Adapter**: exposes MoveIt MCP tools to Agent Orchestration and executes tool calls.
- **Robot Context**: stores advisory recent observations, planning results, gripper state, and execution results.

The package shape is:

```text
robot_control/
  task_policy.py
  call_validation.py
  mcp_bridge.py
  context.py
```

Robot Call Validation, Robot Context, Task Policy, and the Robot Tool Adapter live under `robot_control`.

After `robot_control` extraction, extract `agent_control`, then keep any remaining app wiring in the composition root.

**API Boundary:** `robot_control` exposes robot tools and structured tool feedback to `agent_control`. It owns robot-specific vocabulary and must not depend on Pipecat pipeline modules.

### `process_trace`

`process_trace` is the reusable Process Trace Module for voice robot observability. It records semantic spans and events across Voice Runtime, Agent Orchestration, and Robot Control, then writes local append-only JSONL.

It owns trace IDs, session IDs, turn IDs, span parentage, timestamps, JSON-safe attributes, redaction, writer failure handling, and no-op tracing. It does not own runtime behavior, pipeline ordering, model calls, policy decisions, validation, MCP execution, or robot context mutation.

Voice Runtime emits wake, speech capture, STT, Agent Turn, and TTS spans. Agent Control emits backend turn, LangGraph node, graph turn, and model-call spans. Robot Control emits task policy, robot call validation, Robot Context update, and MCP spans.

**API Boundary:** pure `process_trace` core modules must not import Pipecat, LangGraph, LangChain, MCP, Voice Runtime, Agent Control, or Robot Control. Pipecat-specific tracing lives in a thin adapter.

### Task Policy Layer

Task Policy v1 blocks only obvious under-observed or incorrectly ordered tool calls before robot tools run:

```text
LangChain tool call
  -> Task Policy Layer
  -> Robot Call Validation
  -> MoveIt MCP
```

V1 policies:

- Fresh pose before motion/planning/execution.
- No blind `moveit_execute_plan`; the plan name must come from a recent successful planning result.
- Basic gripper/attach ordering before `moveit_attach_object`.

A blocked Task Policy Decision is returned to Agent Orchestration as structured tool feedback with correction text and a suggested next tool. It is not a movement-safety claim.

### MoveIt MCP Boundary

MoveIt MCP is the execution seam into the ROS 1 robot simulation stack. The voice agent routes movement through MoveIt planning/execution workflows. MoveIt and the robot simulation stack are the movement-safety boundary.

The host-side ROS 1 MoveIt MCP lives in `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\moveit_mcp`. It exposes FastMCP tools and talks to ROS 1 through rosbridge. The main entrypoint is `moveit_mcp.server`, the agent-facing tool wrappers live in `moveit_mcp.tools`, and the ROS 1 topic/service adapter lives in `moveit_mcp.vizor_client`.

The Vizor ROS 1 container owns the downstream MoveIt node and robot control code. In the running `vizor-demo` container, the MoveIt server is `/UR10/move_group`, the app-facing control node is `/vizor_robot_control`, and the robot logic is under `/root/catkin_ws/src/vizor_lib/src/vizor_lib/`. Treat container paths as runtime locators; persistent fixes belong in the Docker image source. The local RViz/Vizor image build context is `C:\Users\Samuel\Documents\github\Multi-Actor-Interface-Library\docker\vizor-rviz`; its `patch-vizor-robot.py` applies the ROS 1 `compute_cartesian_path(..., avoid_collisions=True)` compatibility patch inherited from `cxy201/noetic-vizor`.

Agent-facing robot tools should stay semantic and narrow: observation tools, planning tools, verified execution tools, gripper tools, and future failure-explanation tools. Do not expose broad ROS control or raw topic mutation tools to Agent Orchestration by default.

### App composition root

`pipeline_builder.py` is the app composition root. It constructs concrete adapters from runtime profiles across Voice Runtime, Agent Control, and Robot Control, then delegates processor ordering to Voice Runtime Assembly.

`agent_processor_factory.py` is a short-term compatibility seam while profiles still carry `agent.provider`. It constructs the native LangChain backend for supported API providers.

`bot.py` is the runner and lifecycle shell. It owns runner startup, transport creation, profile selection, and client lifecycle hooks only.

## Architecture Invariants

### Voice Runtime owns realtime audio

Pipecat owns transport, audio frames, wake command handling, STT, TTS, interruption behavior, pipeline backpressure, and processor ordering. Agent/Robot Control Modules must not reorder the Voice Runtime pipeline.

### Agent Orchestration stays behind Agent Turn

Agent Orchestration happens behind the `AgentBackend` / `AgentTurnProcessor` seam. The Voice Runtime sees one Agent Turn; it does not see LangChain tool loops, LangGraph state, MCP calls, or robot policy decisions.

### Native LangChain backend target

The target Agent Backend is the native LangChain backend using provider APIs. Do not reintroduce Codex OAuth unless a new architecture decision changes this target.

### Robot tools follow the policy-validation-execution order

Robot tool calls are invoked in this order:

```text
Task Policy Layer
  -> Robot Call Validation
  -> MoveIt MCP
```

Task Policy may block obvious under-observed or incorrectly ordered steps. Robot Call Validation may reject unsupported or malformed tool calls. MoveIt planning/execution and the robot simulation stack are the source of movement safety.

### Robot Call Validation is not Task Policy

Robot Call Validation does not understand user intent, validate arbitrary multi-step tasks, prove object/world state, enforce semantic task safety, handle emergency stop, or decide whether a sequence of moves is logically correct. Those concerns belong to Task Policy or future higher-level robot reasoning Modules.

### Import directions are constrained

`pipeline_builder.py` is the composition root and may import Voice Runtime, Agent Control, Robot Control, and Process Trace packages. `voice_runtime` must not import `agent_control` or `robot_control`. `agent_control` may import `voice_runtime.agent_turn` types and `robot_control`. `robot_control` must not import `voice_runtime` or `agent_control`. Pure `process_trace` core modules must not import runtime/control Modules.

### Robot Control does not belong in Voice Runtime

Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context belong to `robot_control`, not `voice_runtime`.

### Runtime profile files are app configuration

Runtime profile parsing belongs to `voice_runtime`; concrete runtime profile files remain app configuration because they choose adapters across Voice Runtime, Agent Control, and Robot Control.

### STT/TTS provider construction belongs to Voice Runtime

STT/TTS provider construction is Voice Runtime adapter work. `providers.py` is a legacy top-level placement; its target home is under `voice_runtime`. `pipeline_builder.py` should remain the only caller.

### App shell stays thin

`bot.py` must not construct STT/TTS/agent internals, robot tools, or graph nodes directly. `pipeline_builder.py` constructs adapters and delegates ordering to `voice_runtime.assembly`.

### Adapters hide providers

Default providers are adapter choices, not architecture. The architecture names roles and seams; provider-specific classes stay behind adapter Modules.

### Repository docs are the system of record

Agent-facing knowledge must live in repository-local, versioned files. `AGENTS.md` is a map; `CONTEXT.md` defines domain language; this file defines target architecture.

## Cross-Cutting Concerns

### Observability

Voice Metrics are compact semantic turn timing records.

Process Trace is the always-on local trace of a robot voice run. It records wake, speech capture, STT, Agent Turn, LangGraph node, model call, task policy, robot call validation, MCP tool call, Robot Context update, and TTS spans/events in `logs/process_trace.jsonl` by default. The trace is observational; it must not change runtime behavior.

Pipecat frame observation belongs with Voice Runtime adapters. JSONL persistence is app configuration. Robot tool feedback should be structured enough for Codex and humans to understand blocked steps and next actions.

### Reference inspirations

Use these as inspiration for agent-first robotics patterns, not as dependencies or sources of truth:

- [NASA JPL ROSA](https://github.com/nasa-jpl/rosa): ROS agent pattern for introspection-first operation and diagnosis.
- [RobotMCP ROS MCP Server](https://github.com/robotmcp/ros-mcp-server): MCP boundary pattern for ROS topic/service/action observation and control.
- [RAI](https://robotecai.github.io/rai/faq/ROS_2_Overview/): connector pattern for agent tools, robot status, and readiness-gated interaction.
- [ROS-LLM](https://arxiv.org/abs/2406.19741): structured behavior execution and reflection pattern for ROS actions/services.
- [APYROBO](https://github.com/apyrobo/apyrobo): semantic capability, safety policy, observability, and replay ideas.
- [Pipecat function calling](https://docs.pipecat.ai/pipecat/learn/function-calling): voice runtime pattern for tool calls inside a conversational pipeline.
- [OpenAI Realtime MCP](https://developers.openai.com/api/docs/guides/realtime-mcp): MCP lifecycle, tool narrowing, and approval patterns for realtime agents.

Common lesson: the agent owns sequencing and tool choice, while the robot layer owns typed capability boundaries, readiness checks, planning, execution verification, and hard safety constraints.

### Testing

Test Modules through their Interfaces. Voice Runtime tests should not need Codex, MCP, or robot simulation. Robot Control tests should exercise Task Policy, Robot Call Validation, Robot Context, and Robot Tool Adapter behavior without Pipecat. Agent Control tests should exercise Codex/LangGraph behavior through fake Codex and fake robot adapters.

Import direction invariants should be enforced by structural tests once the target packages exist.

### Documentation hygiene

Keep this file stable and short. Put implementation plans under `.pi/plans/` or `docs/superpowers/plans/`. Put debugging notes and incident hypotheses in separate docs. Update `CONTEXT.md` when domain terms change.
