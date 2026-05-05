# LangGraph Agent Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Codex robot-dialogue orchestration into LangGraph while keeping Pipecat as the voice/runtime owner and preserving the existing `AgentBackend` seam.

**Architecture:** Add `server/langgraph_robot_agent.py` as the Codex/LangGraph orchestration layer. Keep `OpenAICodexAgentProcessor` as the public `AgentBackend` implementation, but make `run_turn()` invoke a compiled `StateGraph` with `InMemorySaver`. All robot calls remain routed through `RobotMCPBridge` and `voice_runtime.robot_safety`.

**Tech Stack:** Python 3.10+, Pipecat, LangGraph, OpenAI Codex OAuth backend client, MCP bridge, pytest, ruff, pyright.

---

## Current acceptance source

Use current code/tests as the source of truth. The implemented canonical robot tools are:

- `moveit_get_current_pose`
- `moveit_plan_free_motion`
- `moveit_plan_cartesian_motion`
- `moveit_plan_and_execute_free_motion`
- `moveit_plan_and_execute_cartesian_motion`
- `moveit_execute_plan`
- `moveit_open_gripper`
- `moveit_close_gripper`
- `moveit_attach_object`

Do not reintroduce older `moveit_get_robot_status`, relative-motion, or named-pose tool contracts unless a later plan explicitly does so.

## Grilled decisions

- Replace `OpenAICodexAgentProcessor.run_turn()` internals directly behind the existing `AgentBackend` seam; do not add a profile or runtime fallback path.
- Preserve current-pose observation before every Codex request when `moveit_get_current_pose` is available, including non-robot questions.
- Preserve auto-execution for executable outputs from `moveit_plan_free_motion` and `moveit_plan_cartesian_motion`.
- Use one LangGraph `thread_id` per `OpenAICodexAgentProcessor` / graph-runner instance for this prototype; do not change Pipecat turn inputs to carry session IDs.
- Keep `RobotContextStore` as the deep robot-context module. LangGraph nodes read/update it instead of expanding pose/gripper parsing into graph state.
- Put the graph in `server/langgraph_robot_agent.py`, not `server/voice_runtime/`, because LangGraph/Codex/MCP are app/backend adapter details.
- Use a custom `execute_robot_tool` node that calls `RobotMCPBridge.call_tool()`; do not use LangGraph `ToolNode` for robot execution.
- Keep the existing `CodexBackendClient` and Pi-managed OpenAI Codex OAuth token source. Do not switch to LangChain/OpenAI API-key chat models.
- Treat structured robot safety failures as normal tool outputs sent back to Codex, not graph exceptions.
- Keep checkpointed graph state secret-free: no OAuth tokens, credential stores, clients, or bridge objects.

## Files and responsibilities

- Modify: `server/pyproject.toml` — add LangGraph dependency.
- Modify: `server/uv.lock` — refresh lockfile after dependency addition.
- Create: `server/langgraph_robot_agent.py` — LangGraph state, nodes, routing, and runner.
- Modify: `server/openai_codex_agent_processor.py` — keep lifecycle/error handling; delegate turn orchestration to `LangGraphRobotAgent`.
- Create: `server/tests/test_langgraph_robot_agent.py` — graph-specific TDD coverage for state, routing, repair, safety errors, and memory.
- Modify: `server/tests/test_openai_codex_agent_processor.py` — assert backend is graph-backed without changing public behavior.
- Existing tests must keep passing: `server/tests/test_moveit_agent_behavior_contracts.py`, `server/tests/test_robot_mcp_bridge.py`, `server/tests/test_voice_runtime_robot_safety.py`, `server/tests/test_voice_runtime_agent_turn.py`, `server/tests/test_pipeline_builder.py`.

---

## Task 1: Add LangGraph dependency with an import-level failing test

**Files:**
- Modify: `server/pyproject.toml`
- Modify: `server/uv.lock`
- Create: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Write the failing dependency/import test**

Create `server/tests/test_langgraph_robot_agent.py` with only this test first:

```python
def test_langgraph_dependency_is_available() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    assert InMemorySaver is not None
    assert StateGraph is not None
    assert START != END
```

- [ ] **Step 2: Run the test and verify it fails**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_langgraph_dependency_is_available -v
```

Expected: failure with `ModuleNotFoundError: No module named 'langgraph'`.

- [ ] **Step 3: Add LangGraph and refresh the lockfile**

In `server/pyproject.toml`, add this dependency to `[project].dependencies`:

```toml
"langgraph>=1.0,<2",
```

Then run from `server/`:

```bash
uv lock
```

Expected: `server/uv.lock` updates and includes LangGraph packages.

- [ ] **Step 4: Run the dependency test and verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_langgraph_dependency_is_available -v
```

Expected: test passes.

---

## Task 2: Introduce the graph runner and preserve simple Codex turns

**Files:**
- Create/modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`
- Modify: `server/openai_codex_agent_processor.py`

- [ ] **Step 1: Add failing graph-backed text-response test**

Append test helpers and this test to `server/tests/test_langgraph_robot_agent.py`:

```python
import json

import pytest

from codex_auth import CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from langgraph_robot_agent import LangGraphRobotAgent
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.robot_context import RobotContextStore


class FakeStore:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class FakeBackend:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def create_response(self, credentials, *, model, instructions, input_items, tools):
        self.requests.append(
            {
                "credentials": credentials,
                "model": model,
                "instructions": instructions,
                "input_items": list(input_items),
                "tools": list(tools),
            }
        )
        return self.results.pop(0)


class FakeBridge:
    def __init__(self):
        self.calls = []

    def function_tools(self):
        return [
            {"type": "function", "name": "moveit_get_current_pose", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_free_motion", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_cartesian_motion", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_execute_plan", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_and_execute_free_motion", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_and_execute_cartesian_motion", "parameters": {"type": "object"}, "strict": None},
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "moveit_get_current_pose":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "raw": {
                            "pose": {
                                "position": {"x": 0.1, "y": 0.2, "z": 0.3},
                                "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
                            }
                        },
                    }
                }
            )
        if name == "moveit_plan_free_motion":
            return json.dumps(
                {"structured_content": {"ok": True, "feedback": {"can_execute": True}, "raw": {"plan_name": "plan-1"}}}
            )
        if name == "moveit_execute_plan":
            return json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
        return json.dumps({"structured_content": {"ok": True}})


def tool_call(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return CodexToolCall(
        call_id=call_id,
        item_id=item_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def output_item(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return {"type": "function_call", "id": item_id, "call_id": call_id, "name": name, "arguments": json.dumps(arguments)}


@pytest.mark.asyncio
async def test_graph_observes_current_pose_before_simple_codex_response() -> None:
    backend = FakeBackend([CodexResponseResult(text="oauth-ok")])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
    )

    text = await graph.run_turn(AgentTurnInput(user_text="hello", messages=[{"role": "user", "content": "hello"}]))

    assert text == "oauth-ok"
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert backend.requests[0]["model"] == "gpt-5.4-mini"
    assert backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert "Last-known robot context" in backend.requests[0]["instructions"]
    assert "robot: UR10" in backend.requests[0]["instructions"]
```

- [ ] **Step 2: Run the new test and verify it fails**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_observes_current_pose_before_simple_codex_response -v
```

Expected: import failure for `langgraph_robot_agent`.

- [ ] **Step 3: Implement the minimal graph runner**

Create `server/langgraph_robot_agent.py` with:

- `RobotAgentState(TypedDict)` containing `user_text`, `messages`, `input_items`, `tools`, `codex_result`, `pending_tool_calls`, `tool_turns`, `final_text`, and `error_text`.
- `LangGraphRobotAgent.__init__()` storing model, credential store, backend client, tool bridge, `RobotContextStore`, a unique `thread_id`, and compiling `StateGraph` with `InMemorySaver()`.
- Nodes:
  - `observe_current_pose`: call first available observation tool through `tool_bridge.call_tool()` and update `RobotContextStore`.
  - `call_codex`: build input items, tools, instructions, call `backend_client.create_response()`, and return pending tool calls/final text.
  - `repair_tool_arguments`: call the same relative repair semantics currently in `OpenAICodexAgentProcessor`.
  - `execute_robot_tool`: execute pending tools through the bridge, append `function_call_output`, auto-execute executable plans, and increment `tool_turns`.
  - `final_response`: choose final text or no-text fallback.
- Conditional routing:
  - `START -> observe_current_pose -> call_codex`
  - `call_codex -> final_response` when no tool calls or max tool turns reached
  - `call_codex -> repair_tool_arguments` when tools are pending
  - `repair_tool_arguments -> execute_robot_tool -> call_codex`
  - `final_response -> END`

Reuse helper functions from `openai_codex_agent_processor.py` at first, then remove duplication in Task 3.

- [ ] **Step 4: Run the graph text-response test and verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_observes_current_pose_before_simple_codex_response -v
```

Expected: test passes.

---

## Task 3: Port tool-loop behavior into graph nodes and delegate the public backend

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/openai_codex_agent_processor.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`
- Modify: `server/tests/test_openai_codex_agent_processor.py`

- [ ] **Step 1: Add failing graph tests for tool loop and max loop stop**

Append tests that assert:

```python
@pytest.mark.asyncio
async def test_graph_sends_tool_output_back_to_codex() -> None:
    pose = tool_call("moveit_get_current_pose")
    backend = FakeBackend([
        CodexResponseResult(tool_calls=[pose], output_items=[output_item("moveit_get_current_pose")]),
        CodexResponseResult(text="Robot pose is ready."),
    ])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
    )

    text = await graph.run_turn(AgentTurnInput(user_text="where is the pose?", messages=[{"role": "user", "content": "where is the pose?"}]))

    assert text == "Robot pose is ready."
    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert backend.requests[1]["input_items"][-1]["type"] == "function_call_output"
    assert backend.requests[1]["input_items"][-1]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_graph_stops_after_max_tool_turns() -> None:
    calls = [CodexResponseResult(tool_calls=[tool_call("moveit_get_current_pose", call_id=f"call-{i}")], output_items=[output_item("moveit_get_current_pose", call_id=f"call-{i}")]) for i in range(4)]
    backend = FakeBackend(calls)
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
    )

    text = await graph.run_turn(AgentTurnInput(user_text="pose", messages=[{"role": "user", "content": "pose"}]))

    assert text == "I completed the action but have nothing to report."
    assert len(backend.requests) == 4
```

- [ ] **Step 2: Run these graph tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_sends_tool_output_back_to_codex tests/test_langgraph_robot_agent.py::test_graph_stops_after_max_tool_turns -v
```

Expected: failures until the loop routing is implemented.

- [ ] **Step 3: Implement loop routing and public backend delegation**

In `server/openai_codex_agent_processor.py`:

- Import `LangGraphRobotAgent`.
- Keep `connect()`, `disconnect()`, credential/connection error handling, and public constructor unchanged.
- Replace the imperative Codex/tool-loop body in `run_turn()` with construction/invocation of `LangGraphRobotAgent` using the existing `_credential_store`, `_backend_client`, `_tool_bridge`, `_robot_context`, and `_model`.
- Yield the single string returned by `graph.run_turn(turn)`.
- Keep user-facing error strings unchanged.
- Remove or re-export helper functions only after tests prove equivalent behavior.

- [ ] **Step 4: Run graph and existing processor tests**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py -v
```

Expected: all tests pass.

---

## Task 4: Preserve relative repair, Cartesian repair, and plan auto-execution

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Add failing graph tests for movement repairs and auto-execution**

Append tests equivalent to these acceptance behaviors:

```python
@pytest.mark.asyncio
async def test_graph_repairs_missing_relative_target_pose_and_preserves_orientation() -> None:
    tool = tool_call("moveit_plan_and_execute_free_motion", arguments={"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10})
    backend = FakeBackend([
        CodexResponseResult(tool_calls=[tool], output_items=[output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)]),
        CodexResponseResult(text="Moved up 50 mm."),
    ])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(model="gpt-5.4-mini", credential_store=FakeStore(), backend_client=backend, tool_bridge=bridge, robot_context=RobotContextStore())

    await graph.run_turn(AgentTurnInput(user_text="move up a bit", messages=[{"role": "user", "content": "move up a bit"}]))

    assert bridge.calls[1][1]["target_pose"] == {
        "position": {"x": 0.1, "y": 0.2, "z": 0.35},
        "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
    }


@pytest.mark.asyncio
async def test_graph_repairs_back_up_as_negative_x() -> None:
    tool = tool_call("moveit_plan_and_execute_free_motion", arguments={"robot_name": "UR10", "plan_name": "back_up_50mm", "timeout_s": 10})
    backend = FakeBackend([
        CodexResponseResult(tool_calls=[tool], output_items=[output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)]),
        CodexResponseResult(text="Moved back 50 mm."),
    ])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(model="gpt-5.4-mini", credential_store=FakeStore(), backend_client=backend, tool_bridge=bridge, robot_context=RobotContextStore())

    await graph.run_turn(AgentTurnInput(user_text="back up a bit", messages=[{"role": "user", "content": "back up a bit"}]))

    assert bridge.calls[1][1]["target_pose"]["position"] == {"x": 0.05, "y": 0.2, "z": 0.3}


@pytest.mark.asyncio
async def test_graph_repairs_cartesian_waypoints_from_current_pose() -> None:
    tool = tool_call("moveit_plan_and_execute_cartesian_motion", arguments={"robot_name": "UR10", "plan_name": "move_up_cartesian_50mm", "timeout_s": 10})
    backend = FakeBackend([
        CodexResponseResult(tool_calls=[tool], output_items=[output_item("moveit_plan_and_execute_cartesian_motion", arguments=tool.arguments)]),
        CodexResponseResult(text="Moved up 50 mm."),
    ])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(model="gpt-5.4-mini", credential_store=FakeStore(), backend_client=backend, tool_bridge=bridge, robot_context=RobotContextStore())

    await graph.run_turn(AgentTurnInput(user_text="move up a bit", messages=[{"role": "user", "content": "move up a bit"}]))

    assert bridge.calls[1][1]["waypoints"] == [{
        "position": {"x": 0.1, "y": 0.2, "z": 0.35},
        "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
    }]


@pytest.mark.asyncio
async def test_graph_auto_executes_executable_plan() -> None:
    plan_args = {"robot_name": "UR10", "target_pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.35}, "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}}}
    plan = tool_call("moveit_plan_free_motion", arguments=plan_args)
    backend = FakeBackend([
        CodexResponseResult(tool_calls=[plan], output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)]),
        CodexResponseResult(text="Moved up 50 mm."),
    ])
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(model="gpt-5.4-mini", credential_store=FakeStore(), backend_client=backend, tool_bridge=bridge, robot_context=RobotContextStore())

    await graph.run_turn(AgentTurnInput(user_text="move up a bit", messages=[{"role": "user", "content": "move up a bit"}]))

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "plan-1"}),
    ]
```

- [ ] **Step 2: Run the new graph repair tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py -k "repairs or auto_executes" -v
```

Expected: failures until repair/auto-execute logic is in the graph.

- [ ] **Step 3: Implement repair and auto-execution in graph nodes**

In `server/langgraph_robot_agent.py`:

- Keep constants: `VIZOR_ROBOT_NAME = "UR10"`, `MAX_CODEX_TOOL_TURNS = 3`, `PLAN_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}`, `FREE_MOTION_TOOL_NAMES = {"moveit_plan_free_motion", "moveit_plan_and_execute_free_motion"}`, `CARTESIAN_MOTION_TOOL_NAMES = {"moveit_plan_cartesian_motion", "moveit_plan_and_execute_cartesian_motion"}`.
- Implement `_relative_delta()` exactly as current behavior: bit/slightly=0.05m, default=0.10m, lot/far=0.30m; back/backward=-X, forward=+X, left=+Y, right=-Y, up/raise=+Z, down/lower=-Z.
- Implement `_relative_target_pose()` from `RobotContextStore.latest_tcp_pose()` and preserve orientation when present.
- Implement `_execute_tool_call()` to call only `tool_bridge.call_tool()`, update robot context, auto-execute plan tools when `executable_plan_name(output)` returns a name, and serialize planned/execution output with `execution_result_text()`.

- [ ] **Step 4: Run graph and behavior-contract tests**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py tests/test_moveit_agent_behavior_contracts.py -v
```

Expected: all tests pass.

---

## Task 5: Verify structured safety failures and multi-turn context persistence

**Files:**
- Modify: `server/tests/test_langgraph_robot_agent.py`
- Modify: `server/langgraph_robot_agent.py`

- [ ] **Step 1: Add failing structured safety failure and multi-turn tests**

Append:

```python
@pytest.mark.asyncio
async def test_graph_preserves_structured_robot_tool_failure_as_tool_output() -> None:
    bad_tool = tool_call("moveit_execute_plan", arguments={"robot_name": "UR10", "plan_name": ""})

    class FailureBridge(FakeBridge):
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            return json.dumps({"ok": False, "error": "Expected a non-empty plan_name", "correction": "Plan first.", "retryable": True, "suggested_next_tool": "moveit_get_current_pose"})

    backend = FakeBackend([
        CodexResponseResult(tool_calls=[bad_tool], output_items=[output_item("moveit_execute_plan", arguments=bad_tool.arguments)]),
        CodexResponseResult(text="I need a valid plan before executing."),
    ])
    graph = LangGraphRobotAgent(model="gpt-5.4-mini", credential_store=FakeStore(), backend_client=backend, tool_bridge=FailureBridge(), robot_context=RobotContextStore())

    text = await graph.run_turn(AgentTurnInput(user_text="execute it", messages=[{"role": "user", "content": "execute it"}]))

    assert text == "I need a valid plan before executing."
    output = json.loads(backend.requests[1]["input_items"][-1]["output"])
    assert output["ok"] is False
    assert output["retryable"] is True
    assert output["suggested_next_tool"] == "moveit_get_current_pose"


@pytest.mark.asyncio
async def test_graph_persists_context_between_turns_with_same_instance() -> None:
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=FakeBackend([CodexResponseResult(text="first"), CodexResponseResult(text="second")]),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    await graph.run_turn(AgentTurnInput(user_text="first", messages=[{"role": "user", "content": "first"}]))
    await graph.run_turn(AgentTurnInput(user_text="second", messages=[{"role": "user", "content": "second"}]))

    assert "robot: UR10" in graph.latest_state()["instructions"]
    assert graph.latest_state()["tool_turns"] == 0
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_preserves_structured_robot_tool_failure_as_tool_output tests/test_langgraph_robot_agent.py::test_graph_persists_context_between_turns_with_same_instance -v
```

Expected: failures until `latest_state()` and output preservation are implemented.

- [ ] **Step 3: Implement state inspection and preserve JSON error outputs**

In `server/langgraph_robot_agent.py`:

- Store the last final state in `self._latest_state` after graph invocation.
- Add `latest_state(self) -> dict[str, Any]` returning a shallow copy for tests/debugging.
- Do not parse or alter structured JSON failure outputs from `RobotMCPBridge.call_tool()`; append them to Codex input exactly as returned.
- Keep checkpoint state secret-free: no credentials, no backend client, no tool bridge.

- [ ] **Step 4: Run focused graph tests**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py -v
```

Expected: all graph tests pass.

---

## Task 6: Full integration verification and cleanup

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run targeted integration tests**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_moveit_agent_behavior_contracts.py tests/test_voice_runtime_agent_turn.py tests/test_robot_mcp_bridge.py tests/test_voice_runtime_robot_safety.py -v
```

Expected: all targeted tests pass.

- [ ] **Step 2: Run full verification**

Run from `server/`:

```bash
uv run pytest -q
uv run ruff check .
uv run pyright .
```

Expected: all commands exit successfully.

- [ ] **Step 3: Review final diff for scope boundaries**

Run from repo root:

```bash
git diff --stat
git diff -- server/pipeline_builder.py server/voice_runtime/assembly.py server/voice_runtime/agent_turn.py server/robot_mcp_bridge.py server/voice_runtime/robot_safety.py
```

Expected:

- `pipeline_builder.py` and `assembly.py` unchanged.
- `agent_turn.py` unchanged unless only type-only comments were needed.
- `RobotMCPBridge` and `robot_safety` unchanged except tests proving safety boundary are still green.

- [ ] **Step 4: Commit**

```bash
git add server/pyproject.toml server/uv.lock server/langgraph_robot_agent.py server/openai_codex_agent_processor.py server/tests/test_langgraph_robot_agent.py server/tests/test_openai_codex_agent_processor.py .pi/plans/2026-05-05-langgraph-agent-migration.md
git commit -m "feat: orchestrate Codex robot turns with LangGraph"
```

---

## Completion checklist

- [ ] LangGraph dependency is added and locked.
- [ ] `server/langgraph_robot_agent.py` exists and compiles a graph with `InMemorySaver`.
- [ ] `OpenAICodexAgentProcessor` still implements `AgentBackend` and is still Codex-only.
- [ ] Pipecat pipeline wiring is unchanged.
- [ ] Robot calls still go through `RobotMCPBridge.call_tool()`.
- [ ] Safety validation remains in `voice_runtime.robot_safety`.
- [ ] Current-pose observation happens before Codex requests when available.
- [ ] Relative motion repairs preserve orientation.
- [ ] `back up` maps to negative X.
- [ ] Cartesian waypoint repair is covered.
- [ ] Structured safety failures stay structured.
- [ ] Multi-turn context persists within the graph/backend instance using `InMemorySaver` prototype state.
- [ ] `cd server && uv run pytest -q` passes.
- [ ] `cd server && uv run ruff check .` passes.
- [ ] `cd server && uv run pyright .` passes.
