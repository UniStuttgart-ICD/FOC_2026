# Issue C: Pipecat bridge exposes namespaced MoveIt tools and structured validation failures

> Local issue because `C:/Users/Samuel/Documents/github/pipecat/pipecat-agent` currently has no git remote configured.

## Parallel-agent contract

This issue is safe to run in parallel with Multi-Actor MCP server issues.

**Allowed files**
- `server/robot_mcp_bridge.py`
- `server/prompts.py`
- `server/tests/test_robot_mcp_bridge.py`
- `server/tests/test_prompts.py`
- `docs/VIZOR_MOVEIT_MCP.md`

**Do not edit**
- `C:/Users/Samuel/Documents/github/Multi-Actor-Interface-Library/*`

## Goal

Expose robot tools to the voice/Codex agent as canonical `moveit_*` function tools and return structured validation failures instead of raising opaque bridge errors.

## Requirements

Use these canonical names in prompts and function tools:

- `moveit_get_current_pose`
- `moveit_plan_free_motion`
- `moveit_plan_cartesian_motion`
- `moveit_execute_plan`
- `moveit_open_gripper`
- `moveit_close_gripper`
- `moveit_attach_object`

The upstream MCP may temporarily expose legacy names. The bridge should map canonical agent-facing names to upstream names until the MCP server is migrated.

## Suggested implementation

Add a mapping in `server/robot_mcp_bridge.py`:

```python
_AGENT_TO_MCP_TOOL_NAMES = {
    "moveit_get_current_pose": "get_current_pose",
    "moveit_plan_free_motion": "plan_free_motion",
    "moveit_plan_cartesian_motion": "plan_cartesian_motion",
    "moveit_execute_plan": "execute_plan",
    "moveit_open_gripper": "open_gripper",
    "moveit_close_gripper": "close_gripper",
    "moveit_attach_object": "attach_object",
}
```

When upstream tools already use `moveit_*`, mapping can point canonical name to itself.

For validation failures, prefer a serialized tool-like result:

```json
{
  "ok": false,
  "is_error": true,
  "error": "Only Vizor robot UR10 is allowed",
  "correction": "Retry with robot_name=\"UR10\"."
}
```

Do not weaken safety validation.

## Tests

Update `server/tests/test_robot_mcp_bridge.py`:

- Bridge advertises only `moveit_*` names to the agent.
- Calling `moveit_open_gripper` calls upstream `open_gripper` when only legacy upstream tools exist.
- Calling canonical names works when upstream server already exposes canonical names.
- Unknown tool is rejected before MCP call.
- Validation failures serialize actionable correction fields.
- Prompt tests assert `server/prompts.py` uses `moveit_*` names and does not advertise legacy names.

## Verification

Run:

```bash
cd C:/Users/Samuel/Documents/github/pipecat/pipecat-agent/server
uv run pytest tests/test_robot_mcp_bridge.py tests/test_prompts.py -v
```

## Acceptance criteria

- Agent-facing tool list uses `moveit_*` names.
- Legacy upstream MCP names can still be called through the canonical bridge mapping.
- Prompt safety contract references canonical names.
- Bridge validation failures are actionable and structured.
