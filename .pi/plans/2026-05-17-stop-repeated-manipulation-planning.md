# Stop Repeated Manipulation Planning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop repeated manipulation planning after success, avoid same-turn replanning after planner timeouts, align Task Policy with the canonical planner, and keep the model-facing robot tool surface strict.

**Architecture:** Agent Control owns LangGraph turn termination and user-facing status text. Robot Control owns tool schemas, validation, Task Policy, and MCP bridge tool exposure. Long-running planner blackboard jobs are documented as a follow-up, not implemented in this patch.

**Tech Stack:** Python, LangGraph/LangChain-style tool binding, MCP bridge adapters, pytest, Ruff.

---

### Task 1: Tighten Model-Visible Robot Tool Surface

**Files:**
- Modify: `server/robot_control/mcp_bridge.py`
- Modify: `server/robot_control/call_validation.py`
- Test: `server/tests/test_robot_mcp_bridge.py`
- Test: `server/tests/test_robot_call_validation.py`

- [ ] Ensure `RobotMCPBridge.function_tools()` advertises `moveit_plan_manipulation_task` and synthetic `moveit_execute_task`.
- [ ] Hide `moveit_execute_task_plan`, `moveit_execute_task_solution`, and `moveit_execute_plan` from model-facing function tools.
- [ ] Make the manipulation planner schema require `robot_name` and `requirements`, forbid top-level `backend`, and disallow extra keys inside `requirements`.
- [ ] Return a structured Agent Control required error if the bridge is called directly with `moveit_execute_task`.

### Task 2: Align Task Policy

**Files:**
- Modify: `server/robot_control/task_policy.py`
- Test: `server/tests/test_robot_task_policy.py`

- [ ] Allow `moveit_plan_manipulation_task` without prior fresh pose evidence.
- [ ] Keep release and move-and-release held-object evidence checks.
- [ ] Keep no-blind-execute and low-level motion pose gates.

### Task 3: Stop LangGraph Replanning Loops

**Files:**
- Modify: `server/agent_control/langgraph_robot_agent.py`
- Test: `server/tests/test_langgraph_robot_agent.py`
- Test: `server/tests/test_langgraph_robot_agent_e2e.py`

- [ ] Parse successful `moveit_plan_manipulation_task` output with `parse_task_solution_result`.
- [ ] Return `Plan ready.` and route to `END` when a valid `task_solution_id` is present.
- [ ] Turn planner timeout or incomplete Task Solution output into terminal feedback for the current turn.
- [ ] Stop after one repeated schema repair for validation failures such as unsupported `backend`.

### Task 4: Normalize Unified Task Execution Text

**Files:**
- Modify: `server/agent_control/langgraph_robot_agent.py`
- Test: `server/tests/test_langgraph_robot_agent.py`

- [ ] Keep `moveit_execute_task` as the only model-visible task executor.
- [ ] Run sim/RViz first and real robot execution only when connected.
- [ ] Return `Execution complete in RViz; real robot not connected.` when sim/RViz succeeds and real robot execution is unavailable.

### Task 5: Document Blackboard Planner Follow-Up

**Files:**
- Modify: `docs/adr/0002-single-compound-task-planner-surface.md`

- [ ] Add a short follow-up note that long-running manipulation planning should move to Robot Job Blackboard planner jobs.
- [ ] State that the planner worker must return either a complete Task Solution or a terminal failure without LLM-driven intermediate stages.

### Verification

- [ ] Run from `server/`: `uv run pytest tests/test_robot_task_policy.py tests/test_robot_mcp_bridge.py tests/test_langgraph_robot_agent_e2e.py`
- [ ] Run from `server/`: `uv run pytest tests/test_langgraph_robot_agent.py -k "moveit_execute_task or manipulation or task_solution or backend"`
- [ ] Run from `server/`: `uv run ruff check agent_control/langgraph_robot_agent.py robot_control/call_validation.py robot_control/mcp_bridge.py robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_mcp_bridge.py tests/test_langgraph_robot_agent.py tests/test_langgraph_robot_agent_e2e.py`
