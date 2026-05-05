# Voice Runtime Architecture

## Orthogonality goal

The Voice Runtime is split into reusable Modules. Each Module has a small Interface and hides app-specific implementation details behind seams and Adapters.

## Modules

### Runtime Profile

Owns Runtime Profile parsing and provider policy. It does not construct Pipecat processors. Emergency stop is currently a Runtime Profile scaffold; no runtime bypass Adapter is implemented.

### Voice Command

Owns wake phrase detection, pre-buffer replay, wake phrase stripping, finalized command emission, semantic wake events, and rearming. It exposes two Pipecat Adapters because audio gating happens before STT and transcript cleanup happens after STT.

### Agent Turn

Owns Pipecat LLM frame semantics for one Agent Turn. Codex OAuth is the only supported backend Adapter behind this seam.

### Robot Safety

Owns allowed robot tools, UR10-only validation, workspace limits, canonical tool-name policy, plan-before-execute helpers, and execution result interpretation.

Safety coverage is local to the Codex robot bridge:

- Codex through `RobotMCPBridge` is locally enforced because the bridge validates each robot tool call before MCP execution.
- All supported robot tool calls go through `RobotMCPBridge` and `voice_runtime.robot_safety`.

### Voice Metrics

Owns per-turn semantic stage transitions and timing semantics. Pipecat frame observation and JSONL persistence are Adapters around the timeline.

### Voice Runtime Assembly

Owns reusable Pipecat processor ordering: transport input, optional Voice Command audio Adapter, STT, optional Voice Command transcript Adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.

## App integration

`bot.py` remains the app entrypoint. `pipeline_builder.py` constructs concrete Adapters and delegates ordering to the Voice Runtime Assembly Module.

## Reuse checklist

To reuse these Modules in a similar project:

1. Provide Runtime Profiles for the target providers.
2. Provide Robot Tool Adapters for the target MCP or tool layer.
3. Choose an Agent Turn backend Adapter.
4. Build a Pipecat pipeline with Voice Runtime Assembly using the Voice Command, STT, Agent Turn, TTS, and Voice Metrics Adapters.
5. Document Robot Safety coverage per Agent Turn backend. Do not imply direct MCP backends are locally safety-enforced.
6. Treat emergency stop as scaffold-only unless a runtime bypass Adapter is implemented.
