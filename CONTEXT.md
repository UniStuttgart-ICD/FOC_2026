# Voice Runtime Context

This glossary describes the implemented orthogonal Voice Runtime modules in `server/voice_runtime/` after Issue 7.

## Domain glossary

- **Voice Runtime**: The reusable module set that runs a Pipecat robot voice pipeline: profile policy, wake command handling, agent turn framing, robot safety, metrics, and processor assembly.
- **Runtime Profile**: A TOML-selected configuration parsed by `voice_runtime.profiles`; it owns provider/category validation, defaults, required environment names, wake and emergency-stop profile fields, MCP URL, and metrics policy without constructing processors.
- **Voice Command**: The Mave command module in `voice_runtime.wake_command`; its audio Adapter gates audio before STT, emits `WakeDetectedFrame`, replays buffered audio, and its transcript Adapter strips the wake phrase after STT and rearms the gate.
- **Agent Turn**: One backend response to the latest user text; `AgentTurnProcessor` wraps Codex OAuth backend output in Pipecat LLM frames and exposes explicit connect/disconnect lifecycle.
- **Robot Safety**: A pure policy module that validates allowed MoveIt tool names, UR10 robot name, workspace bounds, timeouts, canonical-to-legacy tool names, plan-before-execute helpers, and execution-result text.
- **Robot Tool Adapter**: The app/backend seam that exposes or executes robot tools and must call Robot Safety before robot execution to be locally enforced; `RobotMCPBridge` is the current Codex Robot Tool Adapter.
- **Safety Coverage**: Codex robot tools through `RobotMCPBridge` are locally enforced by `voice_runtime.robot_safety` before MCP execution.
- **Voice Metrics**: The semantic turn timeline in `voice_runtime.voice_metrics` plus app Adapters for Pipecat frame observation and JSONL persistence; wake metrics use `WakeDetectedFrame` rather than processor class names.
- **Voice Runtime Assembly**: The pure processor-ordering interface in `voice_runtime.assembly`; it orders transport input, optional Voice Command audio Adapter, STT, optional Voice Command transcript Adapter, user aggregation, Agent Turn, TTS, transport output, and assistant aggregation.
- **Orthogonality Goal**: Keep each Module small, reusable, and locally owned: pure policy Modules avoid Pipecat and app imports, Adapters isolate provider/backend details, and assembly owns ordering instead of scattering topology across the app.

## Current limitation

Emergency stop is currently a Runtime Profile scaffold and detector configuration holder. It does not implement a runtime audio bypass or preemptive stop path.
