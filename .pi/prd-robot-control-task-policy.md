## Problem Statement

The voice robot agent can run a Pipecat voice pipeline and route robot commands through Codex, LangGraph, MoveIt MCP, and the UR10 simulation, but the robot-side architecture is not yet modular enough for reliable agent-driven development. Voice Runtime, Agent Orchestration, Robot Call Validation, Robot Context, and the Robot Tool Adapter are partly mixed across legacy locations, making ownership unclear and making it easier for future agents to put robot-specific policy inside the realtime audio runtime.

From the user's perspective, this creates uncertainty about where robot behavior belongs, what currently counts as movement safety, and how to add deterministic pre-tool checks without overclaiming semantic task safety. The user wants a modular system where Voice Runtime stays focused on realtime audio, Agent Control owns Codex/LangGraph orchestration, Robot Control owns robot-side policy and tool execution, and MoveIt remains the movement-safety boundary.

## Solution

Introduce and enforce the target Robot Control architecture. Robot Control becomes the module for Task Policy, Robot Call Validation, Robot Context, and the Robot Tool Adapter. Add a minimal deterministic Task Policy Layer that blocks obvious under-observed or incorrectly ordered robot tool calls before they reach Robot Call Validation and MoveIt MCP.

The solution keeps Codex responsible for open-ended intent and task planning, while Task Policy handles only small deterministic preconditions: fresh pose before motion, no blind execute, and basic gripper/attach ordering. Local Robot Call Validation remains structural and ergonomic. Movement safety remains delegated to MoveIt planning/execution and the robot simulation stack.

## User Stories

1. As a voice robot operator, I want robot movement commands routed through MoveIt workflows, so that motion planning and execution stay in the robot simulation stack.
2. As a voice robot operator, I want the agent to request a fresh pose before movement, so that relative or repeated commands are based on recent robot state.
3. As a voice robot operator, I want the agent blocked from executing invented plan names, so that execution only follows a returned MoveIt plan.
4. As a voice robot operator, I want attach operations blocked until the gripper is known closed, so that the robot does not attach objects out of sequence.
5. As a voice robot operator, I want blocked robot steps to produce clear spoken recovery paths, so that I know what the agent needs next.
6. As a voice robot operator, I want tool failures to include suggested next tools, so that Codex can recover without me restating the full command.
7. As a developer, I want Voice Runtime to own only realtime audio concerns, so that robot policy changes do not risk breaking STT, TTS, wake, or transport behavior.
8. As a developer, I want Robot Control to own robot-side concerns, so that Task Policy, Robot Call Validation, Robot Context, and Robot Tool Adapter changes have locality.
9. As a developer, I want Agent Control to depend on Robot Control through clear seams, so that LangGraph orchestration can execute robot tools without knowing low-level validation internals.
10. As a developer, I want Robot Call Validation separated from Task Policy, so that structural tool validation is not confused with multi-step task semantics.
11. As a developer, I want MoveIt documented as the movement-safety boundary, so that local validation is not overclaimed as full robot safety.
12. As a developer, I want Task Policy decisions returned as structured tool feedback, so that Codex can reason over failures in its ReAct loop.
13. As a developer, I want Robot Context to expose recent pose, plan, and gripper state, so that Task Policy can make deterministic decisions without parsing prompts.
14. As a developer, I want Robot Control pure modules testable without Pipecat, Codex, MCP, or LangGraph, so that policy and validation tests are fast and reliable.
15. As a developer, I want structural import tests for module direction, so that future agents cannot accidentally reintroduce robot dependencies into Voice Runtime.
16. As a developer, I want legacy robot modules migrated out of Voice Runtime, so that the codebase reflects the target architecture and is easier for agents to navigate.
17. As a developer, I want the Robot Tool Adapter to expose only canonical MoveIt tools, so that prompts and tool descriptions stay aligned with actual execution capabilities.
18. As a developer, I want Agent Orchestration to run Task Policy before Robot Call Validation, so that obvious missing preconditions are caught before lower-level argument validation and MCP execution.
19. As a developer, I want auto-execution to record executable plan names before execution, so that no-blind-execute policy applies consistently to automatic and model-requested execution.
20. As a developer, I want docs and agent guidance to use the same domain language, so that future planning and code changes do not drift back to stale safety terminology.
21. As a future agent worker, I want Architecture and Context docs to be the source of truth, so that I can find the correct module for a change without reading every file.
22. As a future agent worker, I want implementation plans to use Robot Control target paths, so that I do not add new robot policy code under Voice Runtime.
23. As a maintainer, I want the composition root to remain thin, so that provider wiring and processor ordering do not leak into robot policy or Agent Orchestration.
24. As a maintainer, I want Codex-only backend guidance preserved, so that new non-Codex agent backends are not reintroduced without an explicit architecture decision.
25. As a maintainer, I want the minimal policy layer explicitly scoped, so that it does not become a vague semantic safety planner.

## Implementation Decisions

- Use three target modules: Voice Runtime, Agent Control, and Robot Control.
- Keep Voice Runtime focused on realtime audio, wake command handling, STT, TTS, aggregation, interruption behavior, pipeline ordering, and voice metrics.
- Treat AgentBackend as the seam between Voice Runtime and Agent Control.
- Keep LangGraph as Agent Orchestration behind the Agent Turn seam.
- Keep Codex OAuth as the only target Agent Backend unless a new architecture decision changes that target.
- Introduce Robot Control as the target home for Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context.
- Implement Task Policy as a small deterministic module, not as a semantic task planner.
- Define TaskPolicyDecision as structured allow/block feedback with correction text and an optional suggested next tool.
- Run robot tool calls through Task Policy, then Robot Call Validation, then MoveIt MCP.
- Delegate movement safety to MoveIt planning/execution and the robot simulation stack.
- Keep local Robot Call Validation as ergonomic structural validation for clearer errors.
- Preserve the existing Codex/LangGraph ReAct loop: observe, call Codex, execute tools, observe again, call Codex again.
- Record recent robot observations in Robot Context for Task Policy use.
- Record recent executable plan names in Robot Context for no-blind-execute policy.
- Record recent gripper state in Robot Context for attach ordering policy.
- Do not make plans single-use unless a later decision requires that behavior.
- Keep the browser/Pipecat audio pipeline out of scope for Robot Control changes.
- Move robot-side modules out of legacy Voice Runtime and top-level placements after Task Policy lands.
- Enforce import direction with structural tests once target modules exist.
- Update agent instructions, context language, architecture documentation, and current architecture docs to use the same terms.

## Testing Decisions

- Test modules through their interfaces, not by reaching into implementation details.
- Test Task Policy as a pure Robot Control module using fake context objects.
- Test Robot Context through public recent-state and rendering methods.
- Test Robot Call Validation through its public validation and structured error interfaces.
- Test the Robot Tool Adapter with fake MCP servers and serialized MCP results.
- Test Agent Orchestration with fake Codex responses and fake robot adapters.
- Test policy failures as tool outputs returned to Codex, not as graph exceptions.
- Test auto-execution through the same policy-checked path as ordinary execution.
- Test import-direction invariants structurally so Voice Runtime cannot import Robot Control or Agent Control.
- Keep Voice Runtime tests independent from Codex, MCP, and robot simulation.
- Keep Robot Control tests independent from Pipecat, Codex, and LangGraph except adapter-specific tests.
- Prior art exists in the current tests for LangGraph orchestration, Robot MCP bridge behavior, Robot Context rendering, Robot Call Validation, and orthogonal import guards.

## Out of Scope

- Fixing the observed follow-up Codex HTTP 400 issue.
- Changing STT, TTS, wake word behavior, browser transport, or Pipecat pipeline ordering.
- Adding emergency stop runtime behavior.
- Building object perception or scene understanding.
- Proving that the robot is holding an object.
- Implementing full pick-and-place, wave, pour, stack, or other semantic task policies.
- Adding non-Codex LLM backends.
- Changing MoveIt MCP server behavior.
- Making Robot Control an installable package outside the server codebase.
- Extracting the full Agent Control package unless done after Robot Control extraction.

## Further Notes

This PRD follows the target architecture captured in the repository architecture and context docs. The first implementation sequence should be: minimal Task Policy Layer, then Robot Control extraction, then later Agent Control extraction. The docs and plans created during this session should be treated as the source of truth for implementation sequencing.
