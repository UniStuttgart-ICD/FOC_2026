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
- [ ] #4 Block attach until gripper is recently known closed
- [ ] #5 Expose canonical MoveIt tools through the Robot Tool Adapter
- [ ] #6 Move Robot Call Validation into `robot_control`
- [ ] #7 Move Robot Context into `robot_control`
- [ ] #8 Move Robot Tool Adapter into `robot_control`
- [ ] #9 Enforce target module import directions structurally
- [ ] #11 Align docs and agent guidance with Robot Control language
- [ ] #12 Run final Robot Control extraction verification and scope review

Deferred / separate from these plans:
- [ ] #10 Extract Agent Control behind the Agent Turn seam

## Activity log

- 2026-05-05: Created isolated worktree `.worktrees/robot-control-task-policy` on branch `feature/robot-control-task-policy`.
- 2026-05-05: Baseline targeted tests passed in the worktree: `uv run pytest tests/test_robot_context.py tests/test_langgraph_robot_agent.py tests/test_robot_mcp_bridge.py tests/test_voice_runtime_robot_safety.py -q` (`43 passed`).
- 2026-05-05: Subagent runner for #2 crashed before writing a result, but changes were recovered from the worktree.
- 2026-05-05: #2 complete. Commit `9480de9` added pure `robot_control.task_policy`, initial Robot Control import guard, recent-pose Robot Context API, and LangGraph policy feedback path. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py -q` (`25 passed`), `uv run ruff check robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py` (pass), `uv run pyright robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py` (0 errors).
- 2026-05-05: #3 complete. Commit `31835c6` added executable-plan memory to legacy Robot Context, no-blind-execute Task Policy checks, and LangGraph auto-execute plan recording before policy-checked execution. Validation: `uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py -v` (`32 passed`), `uv run ruff check robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_robot_context.py tests/test_langgraph_robot_agent.py` (pass), `uv run pyright robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_robot_context.py tests/test_langgraph_robot_agent.py` (0 errors).

## Next wave

1. #3: add executable plan memory and no-blind-execute policy.
2. #4: add gripper state memory and attach-ordering policy.
3. #5/#6 can be considered after #3/#4; avoid parallel edits to the same files unless using separate worktrees and manual patch integration.
