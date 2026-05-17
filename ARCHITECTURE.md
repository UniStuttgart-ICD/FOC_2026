# Pipecat Voice Robot Agent Architecture

This file is the target architecture map for agents and maintainers. It may contain only the Harness Engineering architecture-doc outline: Bird's Eye View, Code Map, Architecture Invariants, and Cross-Cutting Concerns. API boundaries belong inside the Code Map. Do not put runtime logs, debugging hypotheses, active plans, source-file inventories, or volatile implementation notes here.

## Bird's Eye View

The system turns a spoken user command into a robot action and a spoken response.

Two planes keep the architecture understandable:

1. **Voice Runtime plane**: owns realtime audio transport, wake command handling, STT, user/assistant aggregation, TTS, interruption behavior, pipeline ordering, and voice metrics.
2. **Agent/Robot Control plane**: owns intent handling, API-key-backed LangChain Agent Orchestration, deterministic robot task policy, robot call validation, MoveIt tool execution, robot context, and user-sensing context.

The high-level flow is:

```text
Browser audio
  -> Voice Runtime
  -> Agent Turn
  -> User Sensing Context refresh
  -> Agent Orchestration
  -> Task Policy Layer
  -> Robot Call Validation
  -> MoveIt MCP and Robot Control execution bridge
  -> UR10 simulation or Verified Real Robot Execution
  -> Agent response
  -> Voice Runtime speech output
```

Movement safety is delegated to MoveIt planning/execution and the robot simulation stack. Physical robot actuation happens only after an executable MoveIt plan or explicit operator command. The voice agent must route movement through MoveIt workflows. Local validation may exist for ergonomics and clearer errors, but it is not the source of movement safety.

## Code Map

This section names the target Modules and seams. Use symbol search for the mentioned names; do not rely on this file as a source-file inventory.

### `voice_runtime`

`voice_runtime` is the reusable Voice Runtime Module. It owns Pipecat-facing runtime semantics and must stay independent of Codex, MCP, and robot task policy.

It contains these target submodules:

- **Runtime Profile**: parses the main runtime profile and provider policy without constructing processors.
- **Voice Providers**: constructs STT/TTS adapters for the Voice Runtime in `voice_runtime.providers`.
- **Voice Command**: owns wake detection, pre-buffer replay, wake phrase stripping, and rearming.
- **Agent Turn**: exposes the AgentBackend seam and wraps one backend turn in Pipecat LLM frames.
- **Voice Runtime Assembly**: owns processor ordering.
- **Voice Metrics**: owns semantic turn timing. Pipecat frame observation is a Voice Runtime adapter; JSONL persistence is app configuration.

**API Boundary:** `AgentBackend` is the seam from Voice Runtime into Agent/Robot Control. Voice Runtime knows that an Agent Backend can connect, disconnect, and run one Agent Turn; it does not know how LangChain, LangGraph, MCP, or MoveIt work.

### `agent_control`

`agent_control` is the Module for API-key-backed LangChain Agent Orchestration.

It contains:

- **LangChain API Backend**: builds native LangChain chat models and satisfies the Agent Turn backend seam.
- **Agent Orchestration**: the LangGraph loop that calls the model, executes robot tools through Robot Control, observes Robot Context, and repeats until done, retried, or blocked.
- **Robot Agent Prompt**: the prompt renderer and prompt parts aligned with Robot Call Validation and Robot Tool Adapter feedback.
- **Agent Turn Factory**: builds the native LangChain backend and wraps it in the Voice Runtime Agent Turn processor for the app composition root.

**API Boundary:** `agent_control` satisfies `voice_runtime.AgentBackend` and depends on `robot_control` for robot execution. It must not own Pipecat transport, audio frames, wake handling, STT/TTS, interruption behavior, or pipeline ordering.

### `robot_control`

`robot_control` is the Module for robot-side control concerns.

It contains these target submodules:

- **Task Policy Layer**: deterministic pre-tool checks for obvious robot-step preconditions.
- **Robot Call Validation**: structural and local tool-call validation for allowed MoveIt tools, UR10 arguments, target bounds, timeouts, and executable plan names.
- **Robot Tool Adapter**: exposes MoveIt MCP tools and Robot Control bridge tools to Agent Orchestration, adapts LangChain tool-call messages, executes tool calls, and normalizes MCP timeouts/exceptions.
- **Robot Context**: stores advisory recent observations, planning results, gripper state, execution results, and held objects proven by attach or verification evidence.
- **Robot Job Blackboard**: stores queued/running/completed/failed robot jobs and terminal events for long-running action execution.
- **Robot Job Worker**: deterministic executor for queued MoveIt jobs; it calls the exact tool and arguments submitted by Agent Control.

The package shape is:

```text
robot_control/
  task_policy.py
  call_validation.py
  mcp_bridge.py
  context.py
  job_board.py
  job_worker.py
```

Robot Call Validation, Robot Context, Task Policy, MCP bridging, LangChain tool-message adaptation, Robot Job Blackboard, Robot Job Worker, and the Robot Tool Adapter live under `robot_control`.

**API Boundary:** `robot_control` exposes robot tools and structured tool feedback to `agent_control`. It owns robot-specific vocabulary and must not depend on Pipecat pipeline modules.

### `process_trace`

`process_trace` is the reusable Process Trace Module for voice robot observability. It records semantic spans and events across Voice Runtime, Agent Orchestration, and Robot Control, then writes local append-only JSONL.

It owns trace IDs, session IDs, turn IDs, span parentage, timestamps, JSON-safe attributes, redaction, writer failure handling, and no-op tracing. It does not own runtime behavior, pipeline ordering, model calls, policy decisions, validation, MCP execution, or robot context mutation.

Voice Runtime emits wake, speech capture, STT, Agent Turn, and TTS spans. Agent Control emits backend turn, LangGraph node, graph turn, and model-call spans. Robot Control emits task policy, robot call validation, Robot Context update, and MCP spans.

**API Boundary:** pure `process_trace` core modules must not import Pipecat, LangGraph, LangChain, MCP, Voice Runtime, Agent Control, or Robot Control. Pipecat-specific tracing lives in a thin adapter.

### Archived Model Benchmarking

Historical model-candidate benchmarking lives under `archive/model-benchmarking/`. It is not part of the workshop runtime.

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
- Pick/place planning is motion planning: it still needs fresh robot state and returns a plan for later execution.

A blocked Task Policy Decision is returned to Agent Orchestration as structured tool feedback with correction text and a suggested next tool. It is not a movement-safety claim.

### MoveIt MCP Boundary

MoveIt MCP is the execution seam into the ROS 1 robot simulation stack. The voice agent routes movement through MoveIt planning/execution workflows. MoveIt and the robot simulation stack are the movement-safety boundary.

The live MoveIt MCP runs as the `moveit-mcp` service in the ROS/Vizor Docker Compose stack, beside `ros-core`, `vizor-demo`, and `vizor-mcp`. It is built from the shared `local/multi-actor-mcp:latest` image, runs `python -m moveit_mcp`, and talks to ROS 1 through rosbridge at `vizor-demo:9090`. Host clients use `http://127.0.0.1:8765/mcp`; stack-internal clients use `http://moveit-mcp:8765/mcp`.

The source package lives in `server/moveit_mcp`. It exposes FastMCP tools. The main entrypoint is `moveit_mcp.server`, the agent-facing tool wrappers live in `moveit_mcp.tools`, and the ROS 1 topic/service adapter lives in `moveit_mcp.vizor_client`.

The Vizor ROS 1 container owns the downstream MoveIt node and robot control code. In the running `vizor-demo` container, the MoveIt server is `/UR10/move_group`, the app-facing control node is `/vizor_robot_control`, and the robot logic is under `/root/catkin_ws/src/vizor_lib/src/vizor_lib/`. Treat container paths as runtime locators; persistent fixes belong in the Docker image source. The local RViz/Vizor image build context is `docker/vizor-rviz`; its `patch-vizor-robot.py` applies the ROS 1 `compute_cartesian_path(..., avoid_collisions=True)` compatibility patch inherited from `cxy201/noetic-vizor`.

RViz is a Planning Scene consumer, not an agent boundary. The RViz config in the Vizor image must subscribe to the namespaced MoveIt scene stream, typically `/UR10/move_group/monitored_planning_scene`, through the MoveIt MotionPlanning display. For MTC visualization, the same RViz config must also load `moveit_task_constructor/Motion Planning Tasks` against the conventional `robot_description` parameter and subscribe to `/solution`; the Vizor desktop startup aliases `/UR10/robot_description*` params to the global MoveIt names before launching RViz. The MTC display only animates when an MTC backend publishes `moveit_task_constructor_msgs/Solution` on `/solution`; otherwise it is expected to remain present but idle. The geometry path is ROSBridge topic input, `/vizor_robot_control`, MoveIt planning scene, then RViz visualization.

Agent-facing robot tools should stay semantic and narrow. Tool tiers are observation, planning, execution, diagnostic, and admin. Task-level pick/place planning tools belong to MoveIt MCP; Pipecat Robot Control consumes them through Robot Call Validation, Robot Context, and Robot Agent Prompt seams. Robot Control may also advertise synthetic bridge tools that decompose a returned task solution into backing MoveIt MCP and Verified Real Robot Execution calls.

The current agent-facing robot contract includes:

- Observation: `moveit_get_current_pose`, `moveit_get_robot_state`, `moveit_list_scene_objects`, and `moveit_get_object_context`.
- Planning: `moveit_plan_free_motion`, `moveit_plan_cartesian_motion`, `moveit_plan_pick`, `moveit_plan_place`, `moveit_plan_pick_task`, `moveit_plan_place_task`, and `moveit_plan_compound_task`.
- Execution: `moveit_execute_plan` with a recent returned `raw.plan_name`, and `moveit_execute_task` with a recent approved `raw.task_solution_id` and supported `raw.execution_contract` for one task-level call backed by stage-by-stage execution proof. Legacy `moveit_execute_task_plan` and `moveit_execute_task_solution` remain internal compatibility paths.
- Diagnostic: `moveit_explain_motion_failure`, `moveit_verify_attached_object`, and `moveit_verify_released_object`.
- Admin/state mutation: `moveit_open_gripper`, `moveit_close_gripper`, `moveit_attach_object`, and `moveit_release_object`. In verified task-plan execution, physical gripper open/close is routed through Verified Real Robot Execution; MCP tools only synchronize MoveIt planning-scene attach/release state after verified gripper evidence.

Agent Orchestration exposes `moveit_execute_task` as the single model-visible task execution tool. The MCP adapter may still know legacy task execution tools for internal compatibility.

The verified dynamic pick path stays semantic: task planning, explicit approval bound to the returned task solution, then `moveit_execute_task`.

Supported compound manipulation stays semantic too: the model may request `moveit_plan_compound_task` with hard `requirements` and optional non-executable `preferences`; `stage_intents` such as `approach_object`, `move_to_pose`, `release_object`, and `verify_released` are optional planner hints. It does not author executable stages, raw MTC graphs, code, or waypoints. `moveit_plan_compound_task` requires `backend="mtc"` and either returns a solved `TaskSolution` with `raw.execution_contract` or fails with no task solution id. Unsupported contact manipulation such as `slide` and `push` fails at planning.

Do not expose broad ROS control, raw topic mutation tools, or combined `moveit_plan_and_execute_*` tools to Agent Orchestration by default. Planning and execution are separate agent-visible verbs.

The MTC Backend is not the default MoveIt MCP backend. The default pick task backend remains emulated. Emulated mode is the configured default, not a fallback from failed MTC. A configured MTC backend must return solved stage evidence and a task solution before it is executable; failures return structured `ok=false` evidence with `failed_stage` and `blocker`, no silent emulated fallback, and no task solution.

#### Dynamic Pick and MTC Backend Contract

Dynamic pick remains a semantic task path:

```text
moveit_plan_pick_task
  -> explicit approval for the returned task_solution_id
  -> moveit_execute_task_plan
```

The agent does not author waypoint lists for dynamic pick. `moveit_plan_pick_task` owns object grounding, grasp candidate evidence, backend choice, stage evidence, and the returned `task_solution_id`. `moveit_execute_task_plan` owns verified execution of that exact task solution.

The configured backend determines planning behavior:

- `backend="emulated"` is the default. It returns MTC-shaped stage evidence and multiple candidate attempts for dynamic or vertical objects so a single low side/back attempt is not treated as the whole search.
- `backend="mtc"` is explicit opt-in. It must solve through the ROS-side MTC boundary and return `backend="mtc"`, stage summaries, candidate evidence, selected cost, selected grasp, and a task solution.
- Failed MTC does not fall back to emulated planning. It returns `ok=false`, `failed_stage`, `blocker`, retry guidance, and no task solution id.

The ROS-side MTC boundary uses current MoveIt Task Constructor stage terminology: `CurrentState`, `Connect`, `GenerateGraspPose`, `ComputeIK`, `MoveRelative`, and `ModifyPlanningScene`. The current service endpoints are `/vizor_mtc/plan_pick_task` for MTC pick proof and `/vizor_mtc/plan_compound_task` for requirements/preferences compound planning with optional stage-intent hints. Until the UR10+Robotiq typed service and semantic config are complete, these endpoints must fail closed instead of reporting fake MTC success.

#### Compound Task Execution Contract

`moveit_execute_task_plan` is now the verified executor for supported task-solution kinds, not a pick-only executor. It still requires the exact recent `task_solution_id`, explicit approval bound to the same scene snapshot, and a supported `raw.execution_contract`.

The contract is typed and proof-backed:

- Each ordered step names a supported handler: `motion`, `close_gripper`, `open_gripper`, `attach_object`, `release_object`, `verify_attached_object`, or `verify_released_object`.
- Each step must include source-stage and proof metadata such as `source_stage` and `required_proof`.
- Motion steps carry a backend plan handle or enough solved stage evidence for the bridge to produce a concrete MoveIt plan.
- Attach/release steps carry the object name and scene snapshot context; release also carries the released object pose.
- Motion steps are converted into MoveIt plans and executed through Verified Real Robot Execution.
- Gripper open/close is executed through the verified gripper path.
- Attach and release only synchronize the MoveIt planning scene after verified gripper evidence.
- Success is reported only after attachment or release verification proof. Robot Context does not mark a held object on a model-written claim.

Supported v1 task kinds are `pick`, `place`, `hold`, `move_and_release`, `approach_hold_adjust_release`, and `pick_place`. Unknown task kinds, unknown handlers, raw waypoint-only recipes, missing proof fields, stale approval, and unsupported intents fail with structured corrections. This is not an automatic fallback path.

### Verified Real Robot Execution Boundary

Verified Real Robot Execution is the host-side actuation boundary from MoveIt plans to the physical UR10 and Robotiq gripper. It is intentionally not an MCP server. The agent still plans through MoveIt MCP; execution requires an explicit returned plan name or an explicit operator command.

For verified task execution, `moveit_execute_task` is the agent-facing Robot Control bridge. It preserves one task-level execution call while requiring the exact recent `task_solution_id` and matching approval payload, interpreting a supported typed `execution_contract`, converting motion stages into concrete MoveIt plans, retrying failed task stages with attempt-scoped plan names inside the same task path, advancing AR/RViz and physical execution through the same ordered stage evidence when physical execution is connected, executing each returned `plan_name` through Verified Real Robot Execution, routing physical gripper open/close through Verified Real Robot Execution, synchronizing MoveIt attachment/release state with explicit MCP tools, and verifying attachment or release before reporting success. If physical execution is not connected or does not respond, the same task execution continues through the digital/AR/RViz path and reports physical status as unavailable. If physical execution responds but fails after digital/AR/RViz success, the user-facing response must report partial success: execution completed in AR/RViz, but physical execution failed. Stage retry is not fallback to another backend or legacy tool. Final user-facing execution text is bounded to short status or correction text.

The verified execution server caches MoveIt planned trajectories from ROSBridge and exposes narrow HTTP commands for execute, home, and gripper control. The operator dashboard may start this server and call those commands. Agent Control may call the execute tool only when the user explicitly requests execution and a planned action is waiting.

`runtime_profiles.toml` controls the execution mode with `robot_execution.simulation_only`. When it is `true`, Pipecat does not create a Verified Real Robot Execution client and execution stays in MoveIt MCP/RViz. Real robot execution requires `simulation_only = false` and a `verified_execution_url` or `VERIFIED_EXECUTION_URL`.

Physical UR motion and home commands use the UR script socket, normally port `30002`, sending one generated URScript program with `movej` or `servoj` commands. This path must not instantiate `RTDEControlInterface` for production execution, because RTDE Control can contend with robot controller input registers and other adapters.

Robot readiness and alignment verification may use UR RTDE Receive for read-only observation, including actual joint positions and TCP pose from the connected UR controller. These physical observations are used only when the robot responds and must not block the digital/AR/RViz path when the physical robot is unavailable. Gripper commands use the direct Robotiq URCap socket, normally port `63352`. Gripper control must not be routed through RTDE Control.

### Vizor User Sensing MCP Boundary

Vizor user sensing is a separate advisory context boundary from MoveIt robot execution. The live Vizor MCP runs as the `vizor-mcp` service in the same ROS/Vizor Docker Compose stack as `moveit-mcp`, built from the shared `local/multi-actor-mcp:latest` image. It runs `python -m vizor_mcp`, talks to rosbridge at `vizor-demo:9090`, exposes `http://127.0.0.1:8001/mcp` to host clients, and exposes `http://vizor-mcp:8001/mcp` to stack-internal clients.

The source package lives in `server/vizor_mcp`. It subscribes continuously to Vizor ROSBridge topics such as `/HOLO1_GazePoint`, `/HOLO1_Transform`, and `/Robot/target_manual`, keeps bounded in-memory history for gaze attention, and exposes read-only FastMCP tools such as `vizor_get_sensor_context`.

The attention buffer belongs in the long-running Vizor MCP process, not in Pipecat. Pipecat only calls the MCP tool before model turns and stores the returned summary in `server/user_sensing`. That summary may include current gaze, user pose, manual target, and ranked recent attention. It is advisory grounding for references like "this", "that", "there", and "near me"; it is not a movement-safety boundary and should not be treated as proof of user intent when missing, stale, or low confidence.

The operator dashboard treats Vizor MCP and MoveIt MCP as part of the ROS/Vizor Docker stack. It may start and stop the Compose stack and wait for rosbridge, noVNC, Vizor MCP, and MoveIt MCP readiness, but it must not launch separate MCP server processes. Pipecat receives the Vizor MCP URL as app configuration, typically `MCP_VIZOR_URL=http://127.0.0.1:8001/mcp`. Vizor MCP should tolerate ROSBridge or HoloLens being unavailable at startup: it stays up, reports disconnected or stale context, and queues the idempotent `HOLO1_position_on` tracking command until ROSBridge is ready.

### App composition root

`pipeline_builder.py` is the app composition root. It constructs concrete adapters from runtime profiles across Voice Runtime, Agent Control, and Robot Control, then delegates processor ordering to Voice Runtime Assembly.

`agent_control.factory` (`server/agent_control/factory.py`) is a short-term compatibility seam while profiles still carry `agent.provider`. It constructs the native LangChain backend for supported API providers.

Agent model controls flow from `voice_runtime.profiles.AgentProfile` into `agent_control.model_factory`. `AgentProfile` owns the provider-neutral app settings: provider, model, reasoning effort, temperature, API-key environment variable, and Gemini thinking budget. `agent_control.model_factory` is the provider-specific mapping layer for LangChain kwargs.

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

Planning tools return a candidate plan or Task Solution and execution gate fields. `moveit_execute_plan` executes ordinary plans. `moveit_execute_task` is the single task-level execution call for supported proof-backed task solutions, and it must expose stage-by-stage execution proof rather than hiding execution inside an opaque monolithic timeout. It should not route through `moveit_execute_task_solution` first unless that internal path provides equivalent stage proof and progress behavior. Legacy task execution tools remain internal compatibility paths. Pick/place/compound proof is separate from planning: after executing a task workflow, the agent must verify attachment or release evidence before claiming the object moved, was picked, held, placed, or released.

Robot Tool Adapter normalizes MCP transport timeouts and exceptions into structured robot feedback before Agent Control sees them.

### Physical actuation avoids RTDE Control

Verified real robot motion uses URScript over the robot script socket. Verified gripper control uses the Robotiq socket. RTDE Control is not part of the production actuation path.

### ROS/Vizor stack owns live MCP processes

`vizor-mcp` and `moveit-mcp` run inside the ROS/Vizor Docker Compose stack as siblings of `vizor-demo`. The operator dashboard may manage the Compose stack and readiness checks, but must not start duplicate dashboard-managed MCP servers.

### Long-running robot execution is blackboarded

Agent Control may queue long-running MoveIt action tools as Robot Jobs after Task Policy accepts the step. The Robot Job Worker owns deterministic execution and writes terminal events. The LLM may decide what tool call to submit, but the worker must not improvise, repair, or reinterpret the tool arguments.

Robot Control owns the Robot Job Blackboard state and exposes a concise rendered job summary to Agent Control. Raw plan names, job arguments, structured tool results, and trace attributes remain available for execution and debugging; user-facing speech should use short status phrases such as `Plan ready.`, `Execution queued.`, and `Execution complete.` rather than reading raw plan identifiers aloud.

Diagnostic and proof tools, such as `moveit_explain_motion_failure` and `moveit_verify_attached_object`, are immediate feedback tools rather than queued action execution.

### Robot Call Validation is not Task Policy

Robot Call Validation does not understand user intent, validate arbitrary multi-step tasks, prove object/world state, enforce semantic task safety, handle emergency stop, or decide whether a sequence of moves is logically correct. Those concerns belong to Task Policy or future higher-level robot reasoning Modules.

### Import directions are constrained

`pipeline_builder.py` is the composition root and may import Voice Runtime, Agent Control, Robot Control, and Process Trace packages. `voice_runtime` must not import `agent_control` or `robot_control`. `agent_control` may import `robot_control` and only these Voice Runtime seams: `voice_runtime.agent_turn`, `voice_runtime.profiles`, `voice_runtime.agent_providers`, and `voice_runtime.timing`. `robot_control` must not import `voice_runtime` or `agent_control`. Pure `process_trace` core modules must not import runtime/control Modules.

### Robot Control does not belong in Voice Runtime

Task Policy, Robot Call Validation, Robot Tool Adapter, Robot Context, Robot Job Blackboard, and Robot Job Worker belong to `robot_control`, not `voice_runtime`.

### Runtime profile files are app configuration

Runtime profile parsing belongs to `voice_runtime`; concrete runtime profile files remain app configuration because they choose adapters across Voice Runtime, Agent Control, and Robot Control.

`server/runtime_profiles.toml` intentionally carries one bundled app profile: `hybrid_gemini_live_tts`. Do not rebuild the old provider matrix without a new architecture decision.

Robot execution mode is app configuration, not prompt behavior: `robot_execution.simulation_only = true` is the default RViz/noVNC test mode; `false` enables the Verified Real Robot Execution bridge when a verified execution URL is configured.

### STT/TTS provider construction belongs to Voice Runtime

STT/TTS provider construction lives in `voice_runtime.providers`. `pipeline_builder.py` should remain the only caller.

### App shell stays thin

`bot.py` must not construct STT/TTS/agent internals, robot tools, or graph nodes directly. `pipeline_builder.py` constructs adapters and delegates ordering to `voice_runtime.assembly`.

### Adapters hide providers

Default providers are adapter choices, not architecture. The architecture names roles and seams; provider-specific classes stay behind adapter Modules.

### Repository docs are the system of record

Shared knowledge must live in repository-local, versioned files. `CONTEXT.md` defines domain language; this file defines target architecture. Agent-only instructions are local handover material and are ignored by the workshop repo.

## Cross-Cutting Concerns

### Observability

Voice Metrics are compact semantic turn timing records.

Process Trace is the always-on local trace of a robot voice run. It records wake, speech capture, STT, Agent Turn, LangGraph node, model call, task policy, robot call validation, MCP tool call, Robot Context update, and TTS spans/events. The trace is observational; it must not change runtime behavior.

Runtime profiles configure base JSONL paths for Voice Metrics and Process Trace. `pipeline_builder.py` expands those base paths into session-scoped files such as `logs/process_trace/process_trace-<timestamp>-<session>.jsonl` and `logs/voice_metrics/voice_metrics-<timestamp>-<session>.jsonl`.

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

Test Modules through their Interfaces. Voice Runtime tests should not need LangChain, MCP, or robot simulation. Robot Control tests should exercise Task Policy, Robot Call Validation, Robot Context, Robot Job Blackboard, Robot Job Worker, and Robot Tool Adapter behavior without Pipecat. Agent Control tests should exercise LangChain/LangGraph behavior through fake models and fake robot adapters.

Import direction invariants should be enforced by structural tests once the target packages exist.

### Documentation hygiene

Keep this file stable and short. Keep implementation plans and debugging notes out of the workshop-facing docs. Retain durable decisions in `docs/adr/`, and archive historical provenance under `archive/`. Update `CONTEXT.md` when domain terms change.
