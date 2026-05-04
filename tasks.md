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
- [x] Issue 2: Provider factories
- [x] Issue 3: Agent provider factory + OpenAI Codex OAuth
- [x] Issue 4: OpenWakeWord Mave wake gate
- [x] Issue 5: Emergency stop bypass scaffold
- [x] Issue 6: Metrics recorder
- [x] Issue 7: Pipeline builder + bot.py slimming
- [x] Issue 8: Docs and benchmarking
- [x] Final verification

## Activity log
- 2026-05-04: Started execution using subagents.
- 2026-05-04: Issue 1 complete. Commits: `ec202d0` dependencies, `b100a27` config foundation, `84bd526` validation hardening from code review. Spec review: PASS. Quality review: PASS after fixes. Validation: `uv run pytest tests/test_config.py -v` (6 passed), `uv run ruff check config.py tests/test_config.py` (pass), `uv run pyright config.py tests/test_config.py` (pass).
- 2026-05-04: Ran Issues 2, 3, 4, 5, 6, and preliminary Issue 8 in parallel subagent worktrees, then applied and committed their patches on the integration worktree.
- 2026-05-04: Issue 2 complete. Commit: `36fc9cd`. Spec review: PASS. Quality review: PASS. Validation: provider tests passed.
- 2026-05-04: Issue 3 complete. Commits: `54626fd`, `e0b753a` review hardening. Spec review: PASS. Quality review: PASS after fixes. Validation: Codex/factory tests, ruff, and targeted pyright passed.
- 2026-05-04: Issue 4 complete. Commits: `2ece6a0`, `6eed009` review hardening, `b8917a9` artifact cleanup. Spec review: PASS. Quality review: PASS for Issue 4 files after fixes; full-project pyright pending Issue 7 bot changes. Validation: wake/transcript tests, ruff, targeted pyright, and detector instantiation passed.
- 2026-05-04: Issue 5 complete. Commit: `4bf7347`. Spec review: PASS. Quality review: PASS. Validation: emergency stop tests passed.
- 2026-05-04: Issue 6 complete. Commits: `5f53fc5`, `53fd688` review hardening. Spec review: PASS. Quality review: PASS after fixes. Validation: metrics tests, ruff, and targeted pyright passed.
- 2026-05-04: Preliminary Issue 8 docs commit `6535db9` landed. Spec review: PASS. Quality review deferred final approval until after Issue 7 because README documents `--profile` behavior that Issue 7 implements.
- 2026-05-04: Issue 7 complete. Commits: `809b271`, `1f473e9` review hardening. Spec review: PASS. Quality review: PASS after wake reset + metrics observer fixes. Validation: full server pytest (41 passed), ruff, and pyright passed in review.
- 2026-05-04: Issue 8 complete after final docs fixes in `1f473e9`. Final docs quality review: PASS. `.env.example` no longer advertises ignored `MCP_SERVER_URL`; local debug docs now clarify local STT/TTS but Claude cloud agent.
- 2026-05-04: Final maintainability review found metrics logs needed ignoring; fixed in `2454678`. Re-review: FINAL_PASS.
- 2026-05-04: Final verification complete. `uv run pytest -v` (41 passed, 1 third-party deprecation warning), `uv run ruff check .` (pass), `uv run pyright .` (0 errors), default profile missing-key check prints `DEEPGRAM_API_KEY, CARTESIA_API_KEY`, no-wake debug loads as `no_wake_debug local_debug none`, and `git status --short --ignored` shows only ignored caches/venv.
