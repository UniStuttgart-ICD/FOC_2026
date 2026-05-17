# Restore `ee0c4ba`-Style Task Planning

## Goal

Restore the `ee0c4ba` responsibility split while keeping the current public API:
`moveit_plan_manipulation_task` returns a task-level contract, and `moveit_execute_task`
proves and runs the motion/gripper/scene stages.

## Implementation

- Change `moveit_plan_manipulation_task` for `requirements.goal="hold"` to build one
  selected pick workflow from object context and return a task solution immediately.
- Keep `execution_contract`, `waypoints`, `workflow_steps`, `approval`, selected grasp
  face, object context, and scene snapshot in the planning result.
- Do not run staged MoveIt preview candidate planning in the default hold planning path.
- Keep candidate helpers available for explicit future retry workflows.
- Leave `moveit_execute_task` stage execution and `TASK_PLAN_STAGE_MAX_ATTEMPTS = 2`
  unchanged.

## Tests

- Add a regression where a horizontal `dynamic_2` beam returns a hold task contract with
  no queued preview planning feedback and no RViz preview publishes.
- Update existing hold planning tests to expect contract-first output instead of candidate
  preview success/failure.
- Keep E2E expectations that execution performs approach, pre-grasp, gripper close, attach,
  lift, and verify, with no `moveit_execute_task_solution` call.

## Verification

Run from `server/`:

- `uv run pytest tests/test_moveit_mcp_planning_tools.py -q`
- `uv run pytest tests/test_langgraph_robot_agent.py -q`
- `uv run pytest tests/test_langgraph_robot_agent_e2e.py -q`
- `uv run pytest tests/test_robot_call_validation.py tests/test_robot_mcp_bridge.py tests/test_prompts.py -q`
- `uv run ruff check .`
- `uv run pyright .`
