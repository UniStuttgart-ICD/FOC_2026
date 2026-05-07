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

**Task Policy Layer**:
A deterministic pre-tool layer for obvious robot-step preconditions before Robot Call Validation and MoveIt; v1 covers fresh pose before motion, no blind execute, and basic gripper/attach ordering.

**Task Policy Decision**:
The structured allow/block result from the Task Policy Layer, with correction text and a suggested next tool when a step is blocked.

**Robot Call Validation**:
Lightweight local validation for allowed MoveIt tool names, UR10 robot name, argument shape, target bounds, timeouts, canonical-to-legacy tool names, executable plan names, and clearer error text; it is not a task policy layer and is not the source of movement safety.

**Robot Tool Adapter**:
The Agent/Robot Control seam that exposes and executes robot tools while routing movement through MoveIt workflows.

**Executable Plan**:
A successful MoveIt planning result with `ok=true`, `feedback.can_execute=true`, and a valid returned `raw.plan_name` that can be executed through a MoveIt execution workflow.

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
- **Task Policy Layer** runs before **Robot Call Validation**.
- **Robot Call Validation** may reject malformed tool calls, but it does not validate task-level intent and is not the source of movement safety.
- **Robot Tool Adapter** routes movement through the **MoveIt Safety Boundary**.
- An **Executable Plan** may be auto-executed only through a MoveIt execution workflow.
- A blocked **Task Policy Decision** is returned to **Agent Orchestration** as structured tool feedback, not as a movement-safety claim.
- A **Live LLM Robot Smoke Test** belongs to the manual pass/fail testing pipeline and does not exercise wake, STT, TTS, or browser audio.
- An **Exploratory Gesture Eval** stays outside the pass/fail testing pipeline until its assertions become deterministic and actionable.
- A **Manual Live Eval Gate** keeps Live LLM Robot Evals out of normal CI.
- **Live Eval Evidence** is saved as minimal JSON, not as a human HTML report.
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

## Current limitation

Emergency stop is currently a Runtime Profile scaffold and detector configuration holder. It does not implement a runtime audio bypass or preemptive stop path.
