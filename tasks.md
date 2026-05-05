# Robot Control Task Policy + Extraction Tasks

Parent issue: https://github.tik.uni-stuttgart.de/ac147490/Robot_buddy/issues/1
Plans:
- `.pi/plans/2026-05-05-minimal-task-policy-layer.md`
- `.pi/plans/2026-05-05-robot-control-extraction.md`
Branch: `feature/robot-control-task-policy`
Worktree: `.worktrees/robot-control-task-policy`

## Execution strategy

Use subagents in this session, but keep implementation mostly sequential where files overlap. Run review/verification after each completed implementation slice before dispatching dependent work.

## Issue tracking

- [x] #2 Add fresh-pose Task Policy feedback path — commit `9480de9`
- [x] #3 Block blind MoveIt plan execution, including auto-execute — commit `31835c6`
- [x] #4 Block attach until gripper is recently known closed — commit `84e4d75`
- [x] #5 Expose canonical MoveIt tools through the Robot Tool Adapter — commit `f9739b3`
- [x] #6 Move Robot Call Validation into `robot_control` — commit `2375ccb`
- [x] #7 Move Robot Context into `robot_control` — commit `1b06ba1`
- [x] #8 Move Robot Tool Adapter into `robot_control` — commit `a400245`
- [x] #9 Enforce target module import directions structurally — commit `96e1dfb`
- [x] #11 Align docs and agent guidance with Robot Control language — commit `7abddc4`
- [x] #12 Run final Robot Control extraction verification and scope review — final verification recorded below

Deferred / separate from these plans:
- [ ] #10 Extract Agent Control behind the Agent Turn seam

## Activity log

- 2026-05-05: Created isolated worktree `.worktrees/robot-control-task-policy` on branch `feature/robot-control-task-policy`.
- 2026-05-05: Baseline targeted tests passed in the worktree: `uv run pytest tests/test_robot_context.py tests/test_langgraph_robot_agent.py tests/test_robot_mcp_bridge.py tests/test_voice_runtime_robot_safety.py -q` (`43 passed`).
- 2026-05-05: Subagent runner for #2 crashed before writing a result, but changes were recovered from the worktree.
- 2026-05-05: #2 complete. Commit `9480de9` added pure `robot_control.task_policy`, initial Robot Control import guard, recent-pose Robot Context API, and LangGraph policy feedback path. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py -q` (`25 passed`), `uv run ruff check robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py` (pass), `uv run pyright robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py` (0 errors).
- 2026-05-05: #3 complete. Commit `31835c6` added executable-plan memory to legacy Robot Context, no-blind-execute Task Policy checks, and LangGraph auto-execute plan recording before policy-checked execution. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py -v` (`32 passed`), `uv run ruff check robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_robot_context.py tests/test_langgraph_robot_agent.py` (pass), `uv run pyright robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_robot_context.py tests/test_langgraph_robot_agent.py` (0 errors).
- 2026-05-05: #4 complete via parallel subagent worktree and integrated manually. Commit `84e4d75` added recent gripper state memory to legacy Robot Context, attach-object Task Policy checks, and LangGraph attach-ordering coverage. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py -q` (`40 passed`), targeted ruff (pass), targeted pyright (0 errors).
- 2026-05-05: #5 complete via parallel subagent worktree and integrated manually. Commit `f9739b3` maps legacy high-level MoveIt workflow MCP tools to canonical `moveit_*` adapter names while preserving canonical tool preference. Validation: `uv run pytest tests/test_robot_mcp_bridge.py tests/test_voice_runtime_robot_safety.py tests/test_prompts.py -q` (`31 passed`), targeted ruff (pass), targeted pyright (0 errors).
- 2026-05-05: #6 complete. Commit `2375ccb` moved Robot Call Validation from `voice_runtime.robot_safety` to `robot_control.call_validation`, renamed `RobotCallValidationError` / `structured_robot_call_error`, updated legacy top-level adapter and LangGraph imports, and expanded the Robot Control pure import guard for `call_validation.py`. Validation: `uv run pytest tests/test_robot_call_validation.py tests/test_robot_mcp_bridge.py tests/test_langgraph_robot_agent.py tests/test_robot_control_imports.py -v` (`46 passed`), targeted ruff (pass), targeted pyright (0 errors).
- 2026-05-05: #7 complete. Commit `1b06ba1` moved Robot Context from `voice_runtime.robot_context` to `robot_control.context`, updated Agent Orchestration/Agent Backend/test imports, and expanded the Robot Control pure import guard for `context.py`. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_robot_control_imports.py -v` (`48 passed`), targeted ruff (pass after import sorting), targeted pyright (0 errors).
- 2026-05-05: #8 complete. Commit `a400245` moved Robot MCP Bridge from top-level `robot_mcp_bridge.py` to `robot_control.mcp_bridge`, updated Agent Orchestration/Agent Backend/test imports, and kept validation serialization on `RobotCallValidationError` / `structured_robot_call_error`. Validation: `uv run pytest tests/test_robot_mcp_bridge.py tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_robot_control_imports.py -v` (`40 passed`), targeted ruff (pass after import sorting), targeted pyright (0 errors).
- 2026-05-05: #9 complete. Commit `96e1dfb` strengthened Robot Control import guards, updated Voice Runtime orthogonal import roots after extraction, added legacy robot module deletion coverage, and verified server stale-reference grep had no matches. Validation: `uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py tests/test_robot_call_validation.py tests/test_robot_context.py tests/test_robot_mcp_bridge.py tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py -v` (`64 passed`), targeted ruff (pass), targeted pyright (0 errors), server stale-reference grep (no matches).
- 2026-05-05: #11 complete. Commit `7abddc4` aligned `AGENTS.md`, `CONTEXT.md`, `ARCHITECTURE.md`, and `docs/architecture.md` with Robot Control as the implementation home for Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context. Validation: docs stale-guidance grep (no matches), `uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py -q` (`5 passed`).
- 2026-05-05: #12 final verification complete. Targeted Robot Control tests passed (`50 passed`), affected Agent/Voice tests passed (`33 passed`), full `uv run pytest -q` passed (`175 passed`, one third-party `audioop` deprecation warning), `uv run ruff check .` passed, `uv run pyright .` reported 0 errors, stale-reference grep found no matches, final scope diff showed no changes to `server/pipeline_builder.py`, `server/bot.py`, `server/voice_runtime/assembly.py`, `server/voice_runtime/wake_command.py`, or `server/voice_runtime/agent_turn.py`.

## Next wave

1. #10 Agent Control extraction is deferred/separate from these plans.
2. Prepare review/merge for `feature/robot-control-task-policy` when requested.
