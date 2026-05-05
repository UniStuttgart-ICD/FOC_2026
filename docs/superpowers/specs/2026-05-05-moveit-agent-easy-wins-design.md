# MoveIt Agent Easy Wins Design

## Goal

Improve the voice robot agent with small, parallelizable changes that make MoveIt tool use more reliable before the LangGraph migration.

Success means the agent uses the real MoveIt tools, gets fresh robot state when needed, plans before executing, handles failures with actionable corrections, and preserves a compact robot context for each turn.

## Principles

- Prefer a few agent-friendly workflow tools over many raw MoveIt wrappers.
- Tool names, prompt text, schemas, and safety validation must agree.
- Tool responses should be concise, structured, and useful for the next agent decision.
- Last-known context is advisory. Fresh status is required before movement, relative commands, retries, and safety-sensitive actions.
- Keep this LangGraph-compatible by treating robot context and memory as explicit state.

## Issue graph

### Issue 1 — Align prompt with real MoveIt tools

Update `server/prompts.py` to remove stale tools and describe only the currently exposed MoveIt tools:

- `moveit_get_robot_status`
- `moveit_plan_free_motion`
- `moveit_plan_linear_motion`
- `moveit_execute_plan`
- `moveit_open_gripper`
- `moveit_close_gripper`

Add concise behavior rules:

1. Observe when current state matters.
2. Plan before execution.
3. Execute only a returned valid plan.
4. Verify tool results.
5. Reply briefly.

### Issue 2 — Define agent-friendly MoveIt tool contracts

Specify schemas, descriptions, and response shapes for:

- richer `moveit_get_robot_status`
- `moveit_plan_relative_motion`
- `moveit_list_named_poses`
- `moveit_plan_named_pose`

The contract should include examples and clarify when each tool should be used.

### Issue 3 — Improve `moveit_get_robot_status`

Make status the agent's "look at the robot" tool. Return concise structured context:

- robot name and connection/planning state
- TCP pose
- joint positions
- gripper state
- planning frame and end-effector frame
- last plan and last execution summary, if available
- safety state, including workspace bounds and emergency stop status

### Issue 4 — Hybrid context injection

Before each LLM turn, inject compact last-known robot context, including status age. The injected block must clearly say it is advisory only and that fresh status is required before movement, retries, and safety-sensitive actions.

This should later map cleanly to LangGraph state.

### Issue 5 — Structured tool errors and retry guidance

Normalize robot tool failures to an actionable shape:

```json
{
  "ok": false,
  "error": "Target is outside workspace",
  "correction": "Retry with x/y/z within +/-1.5 m",
  "retryable": true,
  "suggested_next_tool": "moveit_get_robot_status"
}
```

Prompt behavior: apply a retryable correction once. If the same action fails twice, stop and explain briefly.

### Issue 6 — Relative and named motion tools

Add workflow-level tools:

- `moveit_plan_relative_motion` for commands like "move up a bit" or "go left"
- `moveit_list_named_poses` for discoverability
- `moveit_plan_named_pose` for commands like "go home" or "reset"

These tools should reduce LLM coordinate math and make voice commands more reliable.

### Issue 7 — Agent behavior evals

Create realistic behavior evals around tool choice and response quality.

Example prompts:

- "Move up a bit."
- "Go home."
- "Open the gripper."
- "Move left, no, a little more."
- "Try that again."
- "Move outside the workspace."
- "Where is the robot now?"

Track:

- correct tool selection
- fresh status used when required
- no stale tool names
- plan-before-execute behavior
- retry behavior
- concise final response

## Parallel execution plan

### Batch 0 — immediate

- Issue 1: prompt/tool alignment
- Issue 7: eval scenario design

### Batch 1 — after tool contracts are accepted

- Issue 3: richer status tool
- Issue 4: context injection
- Issue 5: structured errors
- Issue 6: relative/named pose tools

### Batch 2 — integration

- Update prompt from real tool outputs.
- Run behavior evals.
- Refine tool descriptions based on failures.
- Update docs.

## Data flow

```text
voice transcript
  -> compact last-known robot context
  -> LLM agent
  -> safe MoveIt tool call
  -> structured tool result
  -> context/memory update
  -> brief voice response
```

For movement and retry flows:

```text
fresh status -> plan -> execute returned plan -> verify -> respond
```

## Error handling

- Validation failures return structured corrections, not opaque exceptions.
- Retryable failures may be corrected once.
- Repeated failures stop the action and explain the blocker.
- Safety failures are not retried unless the correction makes the request clearly safe.

## Testing and validation

Each issue should include targeted tests or behavior evals. Integration validation should check:

- prompt contains only real tool names
- stale tools are absent
- safety validation still rejects invalid robot names, unknown tools, unsafe coordinates, and unsupported arguments
- movement commands plan before execution
- relative commands require fresh status
- structured errors are understandable by the agent

## Out of scope

- Planning-scene/object tools; these come later.
- Full LangGraph migration; this design should remain compatible with it.
- Direct Claude MCP safety proxy unless explicitly added in a separate issue.
