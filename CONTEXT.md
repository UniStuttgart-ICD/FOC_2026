# Voice Robot Agent Context

This glossary defines the project language for the Pipecat voice robot agent target architecture. It describes concepts and ownership, not transient implementation details.

## Language

### Voice Runtime

**Voice Runtime**:
The reusable runtime that turns realtime browser audio into an **Agent Turn** and assistant speech.

**Runtime Profile**:
App configuration that selects voice, agent, robot, and metrics adapters without constructing them.

**Voice Command**:
A wake-gated spoken command, normally beginning with "Mave".

**Agent Turn**:
One backend response to the latest user command, exposed to Voice Runtime as assistant text chunks.

**Voice Metrics**:
Semantic timing for wake, speech capture, STT, agent response, TTS first audio, and TTS completion.

**Voice Modulation**:
Provider-agnostic post-TTS audio shaping applied inside **Voice Runtime** before speech reaches transport output.

**Voice Mod Lab**:
The local web tuning app for generating TTS reference recordings, previewing **Voice Modulation**, and saving **Voice Modulation Presets**.

**Voice Modulation Preset**:
A saved set of validated audio-effect settings used by **Voice Modulation** for one **Runtime Profile**.

**Voice Runtime Assembly**:
The processor-ordering interface for transport input, optional Voice Command audio gate, STT, optional Voice Command transcript adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.

### Agent Control

**Agent Control Module**:
The target module for API-key-backed LangChain intent handling and Agent Orchestration.

**Workshop Agent Location Contrast**:
The workshop research contrast between a **Robot-Inhabiting Agent** and a **Separate Floating AR Avatar**, where embodiment, voice source, visual position, visual appearance, persona, and authority cues are intentionally tunable by participants.

**Robot-Inhabiting Agent**:
A workshop agent state where participants design the agent as perceived through, or as part of, the robot body.

**Separate Floating AR Avatar**:
A workshop agent state where participants design the agent as a distinct AR participant beside the robot rather than as the robot.

**Agent Location Fit**:
How well a participant-tuned agent location and embodiment configuration fits a specific construction uncertainty, including the tradeoffs it creates for authority, clarity, trust, responsibility, and collaboration.

**Post-Run Evaluation Questions**:
The stable workshop questions participants answer after each construction run to compare Agent Location Fit across tuned agent states.

**Starter Agent Persona Card Deck**:
A workshop card deck of initial agent concepts that participants adapt into paired Robot-Inhabiting Agent and Separate Floating AR Avatar states.

**MAVE Starter Persona**:
A robot-oriented starter persona where the agent feels like a self-directed machine-body collaborator, with expressive movement cues and a little independent judgment, while still respecting robot-control limits and human material judgment.

**Robot Nonverbal Cues**:
Small bounded robot movements that communicate turn-taking, attention, excitement, hesitation, refusal, or confirmation without changing the construction state.

**LangChain API Backend**:
The native LangChain chat-model backend accessed with provider API keys.

**Agent Orchestration**:
Dialogue and tool-loop control behind the Agent Turn seam; LangGraph may own this, but Pipecat remains the Voice Runtime owner.

**Robot Agent Prompt**:
The concise behavior prompt aligned with Agent Orchestration, robot tool feedback, and the robot tool contract.

### Robot Control

**Robot Control Module**:
The target module for robot-side control concerns: Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context.

**Robot Observation**:
A fresh robot-state read before movement, retries, relative commands, or safety-sensitive actions; last-known context is advisory only.

**Robot Context**:
Advisory recent robot observations, planning results, gripper state, and execution results.

**Shared Geometry Model**:
A geometry-operation model, expressed as abstract primitives, transforms, geometric features, and constraints, exchanged between the agent and the ROS/Vizor environment for spatial manipulation and synchronization.

**Task Policy Layer**:
A deterministic pre-tool layer for obvious robot-step preconditions before Robot Call Validation and MoveIt; v1 covers fresh pose before motion, no blind execute, and basic gripper/attach ordering.

**Task Policy Decision**:
The structured allow/block result from the Task Policy Layer, with correction text and a suggested next tool when a step is blocked.

**Robot Call Validation**:
Lightweight local validation for allowed MoveIt tool names, UR10 robot name, argument shape, target bounds, timeouts, canonical-to-legacy tool names, executable plan names, and clearer error text; it is not a task policy layer and is not the source of movement safety.

**Robot Tool Adapter**:
The Agent/Robot Control seam that exposes and executes robot tools while routing movement through MoveIt workflows.

**Robot Job Blackboard**:
The shared typed job/event surface for long-running robot action execution. Agent Control writes queued robot jobs; Robot Control workers write started, completed, and failed events.

**Robot Job Worker**:
A deterministic Robot Control worker that validates and executes the exact queued MoveIt tool call. It does not invent new tool calls, repair arguments, or make LLM decisions.

**Executable Plan**:
A successful MoveIt planning result with `ok=true`, `feedback.can_execute=true`, and a valid returned `raw.plan_name` that can be executed through a MoveIt execution workflow.

**Task Solution**:
A successful task-level MoveIt MCP planning result with `ok=true`, `feedback.can_execute=true`, and a valid returned `raw.task_solution_id` for ordered pick/place stages. It is planning evidence, not physical execution evidence.

**Verified Real Robot Execution**:
The host-side actuation boundary that executes cached MoveIt plans on the physical UR10 and Robotiq path after explicit execution intent.

**Simulation-Only Robot Execution**:
The runtime profile mode where robot execution stays inside MoveIt MCP/RViz/noVNC and Pipecat does not create a Verified Real Robot Execution client.

**Verified Task Plan Execution Bridge**:
The `moveit_execute_task_plan` Robot Control bridge that consumes a recent pick **Task Solution**, plans concrete motion stages, executes returned plan names through **Verified Real Robot Execution**, interleaves gripper and attachment tools, and verifies attachment before success.

**Task-Level Pick**:
A MoveIt MCP pick workflow that plans observe, approach, gripper, attach, lift, and attachment-verification stages as one **Task Solution**.

**Task-Level Place**:
A MoveIt MCP place workflow that plans object placement stages as one **Task Solution** and still requires execution plus release or placed-object evidence before a success claim.

**MTC Backend**:
An optional MoveIt Task Constructor implementation backend for task-level tools. The current default backend remains emulated; MTC proof startup is opt-in with `VIZOR_ENABLE_MTC_PROOF=1`.

**Partial Pick Diagnostic**:
A failed legacy pick planning result where only a preposition or earlier segment solved. It is diagnostic evidence, not an executable pick.

**Execution Approval Payload**:
Structured approval evidence bound to the exact plan or **Task Solution**, source tool, object, expected movement, scene snapshot, approval turn, and approval time.

**Scene Snapshot Evidence**:
Compact evidence that binds a planning result to the grounded scene object, planning frame, pose age, and `scene_snapshot_id`.

**MoveIt Safety Boundary**:
The accepted movement-safety boundary; robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack.

### Observability

**Process Trace**:
The reusable cross-cutting trace of a voice robot run, recorded as correlated local spans and events across Voice Runtime, Agent Control, Robot Control, and MCP tool execution.

**Trace Session**:
One bot runtime session, used to correlate all turns and background lifecycle events in a single run.

**Trace Turn**:
One spoken user command inside a Trace Session, normally rooted at wake or first user speech and ending after the assistant response/TTS path completes.

**Trace Span**:
A timed Process Trace record with a name, parent span, status, timestamps, duration, and structured attributes.

**Trace Event**:
An instant Process Trace record for a semantic event such as wake detection, policy block, validation result, Robot Context update, or lifecycle event.

### Testing

**Live LLM Robot Eval**:
An opt-in manual run that uses the real API-key LangChain backend and the MoveIt simulation stack to evaluate robot-agent behavior from natural-language commands.

**Live LLM Robot Smoke Test**:
A deterministic-leaning Live LLM Robot Eval that sends text through the Agent Turn seam and uses pass/fail assertions based on observed tool calls and MoveIt simulation results.

**Exploratory Gesture Eval**:
A non-blocking Live LLM Robot Eval for high-level gestures such as "wave to me" or "draw a star"; it records behavior for review but is not part of the pass/fail testing pipeline.

**Manual Live Eval Gate**:
A repository policy where Live LLM Robot Evals are never part of normal CI and run only when a developer explicitly opts in with live credentials and a prepared MoveIt simulation.

**Live Eval Evidence**:
The minimal JSON artifact saved by a Live LLM Robot Eval, containing prompts, assistant replies, recorded tool calls, tool outputs, validator results, and pass/fail reasons.

**Replay Artifact**:
A compact local artifact recording tool order, typed tool outputs, policy decisions, validation results, approvals, execution results, verification results, and terminal job events for review and replay.

**Recording Robot Tool Adapter**:
A test-only wrapper around the real Robot Tool Adapter that records each robot tool call and output for Live Eval Evidence without changing runtime behavior.

**Smoke Movement Bound**:
A Live LLM Robot Smoke Test rule where "a bit" means about 0.05 m along the intended axis, with tolerant final-pose checks for simulation/controller drift.

**Model Eval Module**:
The reusable module for comparing LangGraph-backed model candidates against robot-agent scenario packs, timing, validator results, tool behavior, and Live Eval Evidence.

**Model Candidate**:
One model configuration under evaluation, including provider, model id, reasoning effort, and API key environment variable.

**Eval Scenario Pack**:
A named set of prompts, validators, and scoring metadata used by the Model Eval Module.

**Eval Tool Adapter**:
The Robot Tool Adapter used during a Model Eval Module run; v1 defaults to a deterministic simulated MoveIt adapter and can optionally use live MCP.

**Model Fit Score**:
A correctness-gated ranking for Model Candidates. Robot correctness must pass first; passing candidates are then ranked mainly by realtime latency and tool-loop efficiency.

**Improvisation Fit**:
The qualitative part of a Model Fit Score that checks whether a model takes bounded embodied initiative for clear gesture requests without inventing scene facts or unsafe targets.

## Relationships

- **Voice Runtime Assembly** contains exactly one **Agent Turn** processor in the voice pipeline.
- **Voice Modulation** belongs to **Voice Runtime** and runs after TTS, before transport output.
- **Voice Mod Lab** may call TTS providers for reference recordings, but it saves **Voice Modulation Presets**, not shared Runtime Profile files.
- **Agent Orchestration** happens behind **Agent Turn** and does not reorder **Voice Runtime Assembly**.
- **Agent Control Module** satisfies the **Agent Turn** backend seam and may use **Robot Control Module**.
- **Voice Runtime** must not own **Task Policy Layer**, **Robot Call Validation**, **Robot Tool Adapter**, or **Robot Context**.
- **Shared Geometry Model** may inform agent spatial reasoning, but the **MoveIt Safety Boundary** remains authoritative for planning and execution safety.
- **Task Policy Layer** runs before **Robot Call Validation**.
- **Robot Call Validation** may reject malformed tool calls, but it does not validate task-level intent and is not the source of movement safety.
- **Robot Tool Adapter** routes movement through the **MoveIt Safety Boundary**.
- **Robot Job Blackboard** decouples slow robot action execution from the spoken **Agent Turn**.
- **Robot Job Worker** owns deterministic execution of queued robot jobs and writes terminal events back to the **Robot Job Blackboard**.
- An **Executable Plan** may be auto-executed only through a MoveIt execution workflow.
- **Simulation-Only Robot Execution** is selected by `robot_execution.simulation_only = true` and is the default mode for RViz/noVNC testing.
- A **Task Solution** may be executed through `moveit_execute_task_solution` only for sim/emulated task-solution execution; verified real-robot pick execution uses the **Verified Task Plan Execution Bridge** after a matching **Execution Approval Payload**.
- The **Verified Task Plan Execution Bridge** currently supports **Task-Level Pick** only.
- A **Partial Pick Diagnostic** must not be stored as an **Executable Plan** or **Task Solution**.
- A blocked **Task Policy Decision** is returned to **Agent Orchestration** as structured tool feedback, not as a movement-safety claim.
- A **Live LLM Robot Smoke Test** belongs to the manual pass/fail testing pipeline and does not exercise wake, STT, TTS, or browser audio.
- An **Exploratory Gesture Eval** stays outside the pass/fail testing pipeline until its assertions become deterministic and actionable.
- A **Manual Live Eval Gate** keeps Live LLM Robot Evals out of normal CI.
- **Live Eval Evidence** is saved as minimal JSON, not as a human HTML report.
- A **Replay Artifact** preserves the task-solution workflow evidence needed to review observe, plan, approve, execute, verify, and summarize loops.
- A **Recording Robot Tool Adapter** observes live smoke tests without adding production logging hooks.
- The **Model Eval Module** runs through the **Agent Turn** seam and evaluates **Agent Orchestration**; it does not own the Robot Agent Prompt, Task Policy Layer, Robot Call Validation, or MoveIt Safety Boundary.
- An **Eval Tool Adapter** satisfies the same robot adapter interface as the production Robot Tool Adapter so model evaluation can switch between simulated and live MCP runs.
- A **Model Fit Score** treats correctness as a gate and latency as a primary ranking factor for realtime robot use; provider cost is optional metadata, not a v1 ranking input.
- **Improvisation Fit** rewards bounded expressive action for clear gesture requests such as waving, while ambiguous spatial references still require clarification.
- **Process Trace** observes runtime behavior but does not own Voice Runtime, Agent Control, Robot Control, policy, validation, MCP execution, or robot safety behavior.
- A **Trace Turn** may contain Voice Runtime, Agent Control, Robot Control, and MCP **Trace Spans** under one correlated tree.
- **Voice Metrics** are summary timing records; **Process Trace** is the detailed span/event record for debugging, bottleneck analysis, and future visualization.

## Example dialogue

> **Dev:** "Should Robot Call Validation decide whether a whole pick-and-place task is safe?"
> **Domain expert:** "No. **Robot Call Validation** rejects malformed or unsupported tool calls. **Task Policy Layer** handles obvious step preconditions. **MoveIt Safety Boundary** owns movement safety."

## Flagged ambiguities

- "Safety Coverage" previously implied local movement-safety enforcement; resolved: movement safety means the **MoveIt Safety Boundary**, while **Robot Call Validation** is ergonomic validation only.
- "Motion Safety Layer" is ambiguous; resolved: use **Robot Call Validation** for local tool-call validation and **MoveIt Safety Boundary** for movement safety.
- Robot-side policy, context, validation, and adapter ownership is resolved to the **Robot Control Module**.
- "Live test" was used for both pass/fail smoke testing and open-ended gesture exploration; resolved: use **Live LLM Robot Smoke Test** for manual pass/fail coverage and **Exploratory Gesture Eval** for wave/star-style behavior review.
- "Metrics" can mean summary turn timing or detailed process tracing; resolved: use **Voice Metrics** for summary timing and **Process Trace** for correlated spans/events.
- **Shared Geometry Model** detail is partially resolved: abstract primitives, transforms, geometric features, and constraints are primary; exact render/planning geometry should be derived or referenced unless a later design decision changes this.

## Current limitation

Emergency stop is currently a Runtime Profile scaffold and detector configuration holder. It does not implement a runtime audio bypass or preemptive stop path.
