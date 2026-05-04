# Code Context

## Files Retrieved
1. `CONTEXT.md` (1-33): authoritative definitions for term-level assumptions (`Voice Runtime`, `Runtime Profile`, `Robot Safety`, `Voice Metrics`, `Voice Runtime Assembly`).
2. `.pi/plans/2026-05-04-orthogonal-reusable-voice-runtime.md` (13-31, 84-120, 1053-1180, 1231-1267, 1231-1267, 1470-1491): execution graph, per-issue contracts, Issue 7 baseline/validation and final checks.
3. `.pi/plans/reviews/orthogonal-plan-execution-review.md` (4-11, 11-12): prior high-risk findings (import guard + cartesia/env policy + safety/error typing + lifecycle order).
4. `.pi/plans/reviews/orthogonal-plan-reuse-review.md` (9-16): prior high-risk findings (import guard + Claude bypass + guard scope + metrics/wake seam).
5. `server/config.py` (90-220): current runtime profile parsing + env validation.
6. `server/bot.py` (53-60, 79-88): lifecycle reflection helper remains in bot startup path.
7. `server/pipeline_builder.py` (36-60, 73-82): wake/cleaner/STT/agent/TTS assembly and metrics observer wiring.
8. `server/wake/wake_gate.py` (14-50, 82-102) and `server/wake/transcript_cleanup.py` (12-44): current two-processor wake design and callback-based reset.
9. `server/metrics.py` (128-162): wake detection currently relies on source class-name check.
10. `server/agent_processor_factory.py` (10-15): direct processor return from concrete adapters.
11. `server/claude_agent_processor.py` (45-58, 157-166): direct MCP wildcard tool access (`mcp__robot__*`) without local safety seam.
12. `server/openai_codex_agent_processor.py` (21-24, 150-202, 275-296): full tool-loop + helper logic still embedded in processor.
13. `server/robot_mcp_bridge.py` (148-170, 174-205, 334-357): current validation, mapping, and failure serialization are in bridge module.
14. `server/tests/test_config.py` (159-188, 191-260): existing env-policy/bool validation behavior to preserve.
15. `server/tests/test_wake_gate.py` and `server/tests/test_transcript_cleanup.py` (96-190, 55-123): current wake behavior and callback assumptions.
16. `server/tests/test_robot_mcp_bridge.py` (47-130, 146-214, 296-319): current safety and canonical tool-call contract.
17. `server/tests/test_openai_codex_agent_processor.py` (110-158, 161-225): codex loop/context behavior that should remain compatible.

## Key Code
- Plan expects pre-implementation baseline with Issue 7: clean tree + passing `pytest`, `ruff`, `pyright` (plan lines 1089-1090, 1088-1089).
- Current `config.py` preserves strict env policy: `required_envs` includes `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY`, and `CARTESIA_VOICE_ID` when cartesia voice is omitted (config lines 205-214).
- `pipeline_builder.py` currently constructs wake gate + STT + transcript cleaner in-band and conditionally adds metrics observer, so Issue 7 must reorder through a new assembly module without regressing order.
- Wake behavior is split: `MaveWakeWordGate` in `wake/wake_gate.py` and `WakePhraseTranscriptCleaner` in `wake/transcript_cleanup.py`, with cleaner calling a provided callback (`on_finalized_transcription`) on finalized frames.
- `metrics.py` still uses class-name coupling (`InputAudioRawFrame` source class `"MaveWakeWordGate"`) for wake marks; no semantic `WakeDetectedFrame` is currently observed.
- Robot safety/error surface today lives in `robot_mcp_bridge.py` (`RobotMCPError`, canonical mapping, validation, serialization) and is exercised by bridge tests.
- `bot.py` currently uses `_call_optional_agent_method()` reflection for processor lifecycle; not yet explicit.

## Architecture
Current entrypoint chain is `bot.py -> config.load_runtime_config -> pipeline_builder.build_pipeline`; builder wires concrete STT/TTS providers, wake processors, context aggregators, adapter processors, TTS, and metrics observer. Planned architecture replaces those pieces with `server/voice_runtime/*` adapters/seams while preserving compatibility shims at wake/agent/bridge/metrics boundaries.

## Start Here
Open `server/config.py` (90-220) and `.pi/plans/2026-05-04-orthogonal-reusable-voice-runtime.md` (84-120) first: this is where Issue 1 and runtime-policy extraction assumptions are anchored.

## Blockers / Concerns (before implementation)
- **Dirty git state:** `git status --short` is not clean (modified tracked files and untracked items in `README.md`, `server/{bot.py,config.py,providers.py,openai_codex_agent_processor.py,robot_mcp_bridge.py,...}`, plus untracked `.pi/plans`, `server/tests/...`, `server/.env.example`, `docs/VIZOR_MOVEIT_MCP.md`). This directly violates Issue 7 Step 1 baseline requirement.
- **Not ready for baseline checks:** current tree has `ruff check .` failures (import sorting in `openai_codex_agent_processor.py` and tests) and `pyright .` failures in `robot_mcp_bridge.py` and `server/tests/test_openai_codex_agent_processor.py`. So the plan’s “pass before edits” gate is not currently achievable.
- **No `server/voice_runtime/