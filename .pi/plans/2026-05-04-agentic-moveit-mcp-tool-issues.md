# Agentic MoveIt MCP Tool Issues Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or equivalent isolated-worktree execution. Each issue is self-contained and intended for one agent. Do not edit files outside the issue's allowed scope without reporting why.

**Goal:** Improve MoveIt MCP tool names, schemas, descriptions, responses, and evaluations using Anthropic's agent-tool design guidance.

**Architecture:** Keep low-level MoveIt planning/execution safety boundaries, but make the agent-facing contract namespaced as `moveit_*`, explicit, concise, and recovery-oriented. Split work into independent issues so multiple agents can implement in parallel with minimal file conflicts.

**Tech Stack:** Python, FastMCP, Pydantic/FastMCP tool schemas, pytest, GitHub issues.

---

## Parallel execution map

### Wave 1: safe to run in parallel

1. **Issue A: Namespaced MoveIt MCP tools with agent-grade descriptions and schemas**
   - Primary repo: `Samulko/Multi-Actor-Interface-Library`
   - Allowed files: `moveit_mcp/server.py`, `tests/test_moveit_mcp_server.py`, docs snippets if needed.
   - Avoid: `moveit_mcp/tools.py`, `moveit_mcp/models.py`.

2. **Issue B: Recovery-oriented, token-efficient MoveIt MCP result envelopes**
   - Primary repo: `Samulko/Multi-Actor-Interface-Library`
   - Allowed files: `moveit_mcp/models.py`, `moveit_mcp/tools.py`, `tests/test_moveit_mcp_models.py`, planning/execution/gripper tool tests.
   - Avoid: `moveit_mcp/server.py`.

3. **Issue C: Pipecat bridge exposes namespaced MoveIt tools and structured validation failures**
   - Local repo: `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent` currently has no remote configured.
   - Allowed files: `server/robot_mcp_bridge.py`, `server/prompts.py`, `server/tests/test_robot_mcp_bridge.py`, `server/tests/test_prompts.py`, `docs/VIZOR_MOVEIT_MCP.md`.
   - Avoid: Multi-Actor repo files.

### Wave 2: after Wave 1 contracts are merged

4. **Issue D: Add high-level safe workflow tools for common MoveIt sequencing**
   - Depends on A and B.
   - Primary repo: `Samulko/Multi-Actor-Interface-Library`.
   - Goal: add compound tools that reduce repeated plan/execute loops without weakening safety.
   - Required tools: `moveit_plan_and_execute_free_motion` and `moveit_plan_and_execute_cartesian_motion`.

5. **Issue E: Add MCP tool-use evaluation harness and docs**
   - Can start in parallel as a draft, but final assertions should target the merged Wave 1 contract.
   - Primary repo: `Samulko/Multi-Actor-Interface-Library`.

---

## Shared naming decision

Use `moveit_*` names, not `vizor_*` names:

- `moveit_get_current_pose`
- `moveit_plan_free_motion`
- `moveit_plan_cartesian_motion`
- `moveit_execute_plan`
- `moveit_plan_and_execute_free_motion`
- `moveit_plan_and_execute_cartesian_motion`
- `moveit_open_gripper`
- `moveit_close_gripper`
- `moveit_attach_object`

Legacy names may remain as compatibility aliases only if tests document deprecation and the agent-facing prompt/bridge advertises only `moveit_*` names.

---

## Required verification commands

Multi-Actor repo:

```bash
cd C:/Users/Samuel/Documents/github/Multi-Actor-Interface-Library
uv run pytest tests/test_moveit_mcp_models.py tests/test_moveit_mcp_planning_tools.py tests/test_moveit_mcp_execution_tools.py tests/test_moveit_mcp_gripper_tools.py tests/test_moveit_mcp_server.py -v
```

Pipecat repo:

```bash
cd C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server
uv run pytest tests/test_robot_mcp_bridge.py tests/test_prompts.py -v
```
