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

**Voice Runtime Assembly**:
The processor-ordering interface for transport input, optional Voice Command audio gate, STT, optional Voice Command transcript adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.

### Agent Control

**Agent Control Module**:
The target module for Codex-backed intent handling and Agent Orchestration.

**Codex OAuth Backend**:
The ChatGPT Codex backend accessed with Pi-managed OpenAI Codex OAuth tokens, not a standard OpenAI API-key chat model.

**Agent Orchestration**:
Dialogue and tool-loop control behind the Agent Turn seam; LangGraph may own this, but Pipecat remains the Voice Runtime owner.

**Robot Agent Prompt**:
The concise Codex behavior prompt aligned with Agent Orchestration, robot tool feedback, and the robot tool contract.

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

## Relationships

- **Voice Runtime Assembly** contains exactly one **Agent Turn** processor in the voice pipeline.
- **Agent Orchestration** happens behind **Agent Turn** and does not reorder **Voice Runtime Assembly**.
- **Agent Control Module** satisfies the **Agent Turn** backend seam and may use **Robot Control Module**.
- **Voice Runtime** must not own **Task Policy Layer**, **Robot Call Validation**, **Robot Tool Adapter**, or **Robot Context**.
- **Task Policy Layer** runs before **Robot Call Validation**.
- **Robot Call Validation** may reject malformed tool calls, but it does not validate task-level intent and is not the source of movement safety.
- **Robot Tool Adapter** routes movement through the **MoveIt Safety Boundary**.
- An **Executable Plan** may be auto-executed only through a MoveIt execution workflow.
- A blocked **Task Policy Decision** is returned to **Agent Orchestration** as structured tool feedback, not as a movement-safety claim.

## Example dialogue

> **Dev:** "Should Robot Call Validation decide whether a whole pick-and-place task is safe?"
> **Domain expert:** "No. **Robot Call Validation** rejects malformed or unsupported tool calls. **Task Policy Layer** handles obvious step preconditions. **MoveIt Safety Boundary** owns movement safety."

## Flagged ambiguities

- "Safety Coverage" previously implied local movement-safety enforcement; resolved: movement safety means the **MoveIt Safety Boundary**, while **Robot Call Validation** is ergonomic validation only.
- "Motion Safety Layer" is ambiguous; resolved: use **Robot Call Validation** for local tool-call validation and **MoveIt Safety Boundary** for movement safety.
- Robot-side policy, context, validation, and adapter ownership is resolved to the **Robot Control Module**.

## Current limitation

Emergency stop is currently a Runtime Profile scaffold and detector configuration holder. It does not implement a runtime audio bypass or preemptive stop path.
