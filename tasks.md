# Modular Low-Latency Voice Runtime Tasks

Plan: `.pi/plans/2026-05-04-modular-low-latency-voice-runtime.md`
Branch: `feature/modular-voice-runtime`
Worktree: `.worktrees/modular-voice-runtime`

## Setup
- [x] Created isolated worktree at `.worktrees/modular-voice-runtime`.
- [x] Added `.worktrees/` to `.gitignore` on `master` before worktree creation (`b8e8485`).
- [x] Ran baseline `uv sync` in `server/`.
- [!] Baseline `uv run pytest -v` could not run because `pytest` was not installed before Issue 1 dev dependencies.

## Issue tracking
- [x] Issue 1: Runtime config foundation
- [ ] Issue 2: Provider factories
- [ ] Issue 3: Agent provider factory + OpenAI Codex OAuth
- [ ] Issue 4: OpenWakeWord Mave wake gate
- [ ] Issue 5: Emergency stop bypass scaffold
- [ ] Issue 6: Metrics recorder
- [ ] Issue 7: Pipeline builder + bot.py slimming
- [ ] Issue 8: Docs and benchmarking
- [ ] Final verification

## Activity log
- 2026-05-04: Started execution using subagents.
- 2026-05-04: Issue 1 complete. Commits: `ec202d0` dependencies, `b100a27` config foundation, `84bd526` validation hardening from code review. Spec review: PASS. Quality review: PASS after fixes. Validation: `uv run pytest tests/test_config.py -v` (6 passed), `uv run ruff check config.py tests/test_config.py` (pass), `uv run pyright config.py tests/test_config.py` (pass).
