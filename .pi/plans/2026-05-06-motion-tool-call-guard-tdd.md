# Motion Tool-Call Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent clear robot motion requests from ending as text-only promises when no MoveIt action tool was called.

**Architecture:** Add a small policy guard inside `LangGraphRobotAgent`: detect clear motion intent, track whether a non-observation robot action tool ran, and give the model one corrective retry if it replies with no tool calls. Keep the existing MCP bridge, prompt, and wake pipeline intact.

**Tech Stack:** Python 3.12, pytest, pytest-asyncio, LangGraph, LangChain messages, live Codex OAuth smoke tests, local MoveIt MCP at `http://127.0.0.1:8765/mcp`.

---

## Diagnosis Summary

The 2026-05-06 12:56-12:57 run shows wake, WebRTC, MCP, ROS, and OAuth text generation working. The failure is behavioral:

- User: `Have the robot move up and down`
- Agent log: `Codex LangChain request start ... tools=9`
- Agent log: `Codex LangChain request end ... tool_calls=[] text_len=66`
- Assistant: `I’ll get my current pose first, then make a short up-down gesture.`
- Missing: any `Robot tool start name=moveit_plan...` after the response.

The current graph pre-observes pose before calling the model, but if the model returns text with no tool calls, `_route_after_model()` immediately routes to `final_response`. That permits “I’ll do it” responses for movement commands.

## File Map

- Modify: `server/langgraph_robot_agent.py`
  - Add motion-intent detection.
  - Add action-tool execution tracking.
  - Add one corrective retry node when a movement request receives text without a movement tool call.
- Modify: `server/tests/test_langgraph_robot_agent.py`
  - Add deterministic unit tests with a scripted model and fake tool bridge.
- Modify: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
  - Add a manual live case for `Have the robot move up and down`.
- Optionally modify: `server/test_support/live_robot_smoke.py`
  - Add a validator for an up-down Cartesian gesture if no reusable validator already fits.

## Task 1: Add A Failing Unit Test For Text-Only Motion Promise

**Files:**
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Inspect existing test helpers**

Run:

```powershell
rg -n "LangGraphRobotAgent|Fake|Scripted|tool_calls|moveit_get_current_pose" server/tests/test_langgraph_robot_agent.py server/tests -S
```

Expected: find existing fake model/tool bridge patterns. Reuse them where possible.

- [ ] **Step 2: Write the failing test**

Add this test or adapt it to the existing helper style:

```python
import json

import pytest
from langchain_core.messages import AIMessage

from langgraph_robot_agent import LangGraphRobotAgent
from robot_control.context import RobotContextStore
from voice_runtime.agent_turn import AgentTurnInput


class ScriptedRobotModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def bind_tools(self, tools):
        self.calls.append({"tools": tools})
        return self

    async def ainvoke(self, messages):
        self.calls[-1]["messages"] = messages
        if not self.responses:
            raise AssertionError("No scripted model responses left")
        return self.responses.pop(0)


class FakeRobotToolBridge:
    def __init__(self):
        self.calls = []

    def function_tools(self):
        return [
            {
                "type": "function",
                "name": "moveit_get_current_pose",
                "description": "Get current pose",
                "parameters": {"type": "object", "properties": {}},
            },
            {
                "type": "function",
                "name": "moveit_plan_and_execute_cartesian_motion",
                "description": "Plan, execute, and verify Cartesian motion",
                "parameters": {"type": "object", "properties": {}},
            },
        ]

    async def call_tool(self, name, arguments):
        self.calls.append({"name": name, "arguments": arguments})
        if name == "moveit_get_current_pose":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "tcp_pose": {
                            "position": {"x": 0.5, "y": 0.2, "z": 0.6},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    }
                }
            )
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "tool": name,
                    "feedback": {"phase": "executed", "status": "final joint state matched"},
                    "verification": {"result": "pass"},
                }
            }
        )


@pytest.mark.asyncio
async def test_motion_request_retries_when_model_only_promises_action():
    first = AIMessage(content="I’ll get a fresh pose, then do a simple up-down gesture.")
    second = AIMessage(
        content="",
        tool_calls=[
            {
                "id": "call-1",
                "name": "moveit_plan_and_execute_cartesian_motion",
                "args": {
                    "robot_name": "UR10",
                    "waypoints": [
                        {"position": {"x": 0.5, "y": 0.2, "z": 0.68}},
                        {"position": {"x": 0.5, "y": 0.2, "z": 0.52}},
                        {"position": {"x": 0.5, "y": 0.2, "z": 0.6}},
                    ],
                },
            }
        ],
    )
    final = AIMessage(content="Moved up and down.")
    model = ScriptedRobotModel([first, second, final])
    bridge = FakeRobotToolBridge()
    agent = LangGraphRobotAgent(
        model=model,
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
        thread_id="test-motion-repair",
    )

    reply = await agent.run_turn(
        AgentTurnInput(
            user_text="Have the robot move up and down",
            messages=[{"role": "user", "content": "Have the robot move up and down"}],
        )
    )

    assert reply == "Moved up and down."
    assert [call["name"] for call in bridge.calls] == [
        "moveit_get_current_pose",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_get_current_pose",
    ]
    assert len(model.calls) == 3
    corrective_messages = model.calls[1]["messages"]
    assert any("did not call a MoveIt action tool" in str(message.content) for message in corrective_messages)
```

- [ ] **Step 3: Run the failing test**

Run:

```powershell
uv run pytest tests/test_langgraph_robot_agent.py::test_motion_request_retries_when_model_only_promises_action -q
```

Expected: FAIL because `LangGraphRobotAgent` currently routes no-tool text directly to `final_response`.

- [ ] **Step 4: Commit the failing test if working on a branch**

```powershell
git add server/tests/test_langgraph_robot_agent.py
git commit -m "test: reproduce text-only robot motion promise"
```

If the team prefers not to commit red tests, skip this commit and keep the failure output in the implementation notes.

## Task 2: Implement The Minimal Motion Guard

**Files:**
- Modify: `server/langgraph_robot_agent.py`

- [ ] **Step 1: Add state fields and constants**

Near the existing constants:

```python
ACTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}
MAX_MISSING_ACTION_REPAIRS = 1
```

Extend `RobotAgentState`:

```python
class RobotAgentState(TypedDict):
    user_text: str
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list[dict[str, Any]]
    tool_turns: int
    observed_this_turn: bool
    needs_action_tool: bool
    action_tool_ran: bool
    missing_action_repairs: int
    final_text: str
    error_text: str | None
```

Initialize in `run_turn()`:

```python
"needs_action_tool": _looks_like_robot_action_request(turn.user_text),
"action_tool_ran": False,
"missing_action_repairs": 0,
```

- [ ] **Step 2: Add the repair node**

In `_compile_graph()`:

```python
builder.add_node("repair_missing_action", self._repair_missing_action)
builder.add_edge("repair_missing_action", "call_model")
```

Update the conditional return type:

```python
def _route_after_model(
    self, state: RobotAgentState
) -> Literal["execute_robot_tool", "repair_missing_action", "final_response"]:
```

Use this route logic:

```python
last = _last_ai_message(state["messages"])
if last is None:
    return "final_response"
if last.tool_calls:
    if state["tool_turns"] >= MAX_CODEX_TOOL_TURNS:
        return "final_response"
    return "execute_robot_tool"
if _should_repair_missing_action(state, last):
    return "repair_missing_action"
return "final_response"
```

Add the node:

```python
def _repair_missing_action(self, state: RobotAgentState) -> dict[str, Any]:
    return {
        "messages": [
            HumanMessage(
                content=(
                    "The previous response described a future robot action but did not call "
                    "a MoveIt action tool. For this movement request, call exactly one available "
                    "MoveIt action tool now, or explain a concrete blocker if no safe tool call "
                    "is possible. Do not say you will do it later."
                )
            )
        ],
        "missing_action_repairs": state["missing_action_repairs"] + 1,
    }
```

- [ ] **Step 3: Track action tool execution**

Inside `_execute_robot_tool()`, when a non-observation action tool is executed:

```python
action_tool_ran = state["action_tool_ran"]
...
if name in OBSERVE_TOOL_NAMES:
    output, observed_this_turn = await self._execute_observation_tool(name, dict(args))
else:
    output = await self._execute_tool(name, dict(args))
    action_tool_ran = action_tool_ran or name in ACTION_TOOL_NAMES
    observed_this_turn = False
...
return {
    "messages": tool_messages,
    "tool_turns": state["tool_turns"] + 1,
    "observed_this_turn": observed_this_turn,
    "action_tool_ran": action_tool_ran,
}
```

- [ ] **Step 4: Add helper predicates**

Near existing helper functions:

```python
ROBOT_ACTION_TERMS = (
    "move",
    "go",
    "raise",
    "lower",
    "lift",
    "drop",
    "wave",
    "draw",
    "point",
    "gesture",
    "open",
    "close",
    "grab",
    "release",
)

FUTURE_PROMISE_TERMS = (
    "i'll",
    "i will",
    "i’m going to",
    "i am going to",
    "let me",
    "first, then",
    "then make",
    "then do",
)


def _looks_like_robot_action_request(text: str) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in ROBOT_ACTION_TERMS)


def _should_repair_missing_action(state: RobotAgentState, last: AIMessage) -> bool:
    if not state["needs_action_tool"]:
        return False
    if state["action_tool_ran"]:
        return False
    if state["missing_action_repairs"] >= MAX_MISSING_ACTION_REPAIRS:
        return False
    if state["tool_turns"] >= MAX_CODEX_TOOL_TURNS:
        return False
    text = str(last.content or "").casefold()
    return not text or any(term in text for term in FUTURE_PROMISE_TERMS)
```

- [ ] **Step 5: Run the unit test**

Run:

```powershell
uv run pytest tests/test_langgraph_robot_agent.py::test_motion_request_retries_when_model_only_promises_action -q
```

Expected: PASS.

- [ ] **Step 6: Run focused graph tests**

Run:

```powershell
uv run pytest tests/test_langgraph_robot_agent.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```powershell
git add server/langgraph_robot_agent.py server/tests/test_langgraph_robot_agent.py
git commit -m "fix: retry motion requests that only promise action"
```

## Task 3: Add A Live Regression Case For Up-Down Motion

**Files:**
- Modify: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- Optionally modify: `server/test_support/live_robot_smoke.py`

- [ ] **Step 1: Check existing validators**

Run:

```powershell
rg -n "validate_.*motion|plan_and_execute_cartesian|verified_execution|waypoints" server/test_support/live_robot_smoke.py
```

Expected: identify whether `validate_wave_motion` can be reused or whether a new validator is needed.

- [ ] **Step 2: Add a dedicated validator if needed**

If no validator fits, add:

```python
def validate_up_down_motion(run: LiveSmokeRun) -> ValidationResult:
    if not _has_call(run.tool_calls, "moveit_get_current_pose"):
        return ValidationResult(False, "up-down motion did not observe current pose")
    if not _has_verified_execution(run.tool_calls):
        return ValidationResult(False, "up-down motion did not execute and verify")
    cartesian_calls = [
        call for call in run.tool_calls if call.name == "moveit_plan_and_execute_cartesian_motion"
    ]
    if not cartesian_calls:
        return ValidationResult(False, "up-down motion did not use Cartesian plan-and-execute")
    return ValidationResult(True, "up-down motion executed through verified Cartesian tool")
```

- [ ] **Step 3: Add the manual live case**

In `test_manual_live_llm_robot_smoke_suite`, add:

```python
("up-down-motion", "Have the robot move up and down", validate_up_down_motion),
```

Keep the existing cases. This live suite is skipped unless `RUN_LIVE_LLM_ROBOT_SMOKE=1`.

- [ ] **Step 4: Run live smoke against the already-running MCP server**

Preconditions:

- MoveIt MCP server is running at `http://127.0.0.1:8765/mcp`.
- Codex OAuth is logged in via `~/.pi/agent/auth.json`.
- The robot simulator is in safe simulation mode.

Run:

```powershell
$env:RUN_LIVE_LLM_ROBOT_SMOKE='1'
$env:LIVE_LLM_ROBOT_MCP_URL='http://127.0.0.1:8765/mcp'
uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -q -s
Remove-Item Env:\RUN_LIVE_LLM_ROBOT_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:\LIVE_LLM_ROBOT_MCP_URL -ErrorAction SilentlyContinue
```

Expected: PASS. Evidence files are written under the existing evidence directory used by the smoke harness.

- [ ] **Step 5: Commit**

```powershell
git add server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py server/test_support/live_robot_smoke.py
git commit -m "test: cover live up-down motion smoke"
```

## Task 4: Final Verification And Local Merge

**Files:**
- No code changes expected.

- [ ] **Step 1: Run the full non-live suite**

```powershell
uv run pytest -q
```

Expected: all tests pass; live tests remain skipped unless their env vars are set.

- [ ] **Step 2: Run static checks**

```powershell
uv run ruff check .
uv run pyright .
```

Expected:

- Ruff: `All checks passed!`
- Pyright: `0 errors, 0 warnings, 0 informations`

- [ ] **Step 3: Run the focused live proof**

If MCP and simulator are still running:

```powershell
$env:RUN_LIVE_LLM_ROBOT_SMOKE='1'
$env:LIVE_LLM_ROBOT_MCP_URL='http://127.0.0.1:8765/mcp'
uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -q -s
Remove-Item Env:\RUN_LIVE_LLM_ROBOT_SMOKE -ErrorAction SilentlyContinue
Remove-Item Env:\LIVE_LLM_ROBOT_MCP_URL -ErrorAction SilentlyContinue
```

Expected: the new `up-down-motion` case records at least:

- `moveit_get_current_pose`
- `moveit_plan_and_execute_cartesian_motion`
- final response confirming the gesture or a concrete failure from the tool

- [ ] **Step 4: Inspect git state**

```powershell
git status --short --branch
git log --oneline -5
```

Expected: only intentional commits are present. Existing untracked `.pi/plans/*.md` files may remain untracked unless explicitly added.

- [ ] **Step 5: Merge locally if working on a feature branch**

```powershell
git checkout master
git merge --no-ff <feature-branch> -m "merge: motion tool-call guard"
uv run pytest -q
```

Expected: merge succeeds and tests pass on `master`.

## Self-Review Checklist

- [ ] The unit regression fails before the guard and passes after it.
- [ ] The guard is narrow: it only applies to likely robot action requests.
- [ ] The guard only retries once to avoid loops.
- [ ] The model still can answer normal chat like `How are you?` without tool calls.
- [ ] The live smoke proves real OAuth + real MCP can produce an action tool call for `Have the robot move up and down`.
- [ ] No wake-word behavior changes are included in this plan.
- [ ] No ROS/MCP server implementation changes are included unless live proof reveals a separate MCP failure.

