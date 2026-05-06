# Full LangChain Codex OAuth Agent Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom Codex response loop with a real LangChain/LangGraph agent loop using `ChatCodexOAuth` as the chat model, Pi OAuth credentials as the auth source, and `ToolMessage` results from MoveIt MCP tools.

**Architecture:** Codex remains the agent brain. Python observes current robot pose, injects it into the model context, binds robot tools to the Codex OAuth LangChain model, executes the tool calls requested by Codex through the existing validated MCP bridge, returns `ToolMessage` results to Codex, and lets Codex produce the final answer. Keep domain policy, validation, MCP mapping, and metrics outside the model.

**Tech Stack:** Python 3.12, LangGraph, LangChain Core, `langchain-codex-oauth`, Pi OAuth credentials, Pipecat, MoveIt MCP.

---

## Handoff prompt for a new context window

Use this prompt if starting fresh:

```text
Implement the plan in `pipecat-agent/.pi/plans/2026-05-05-full-langchain-codex-oauth-agent-loop.md`.
Important constraints:
- Use TDD: write each failing test, run it red, then implement minimal green.
- Keep Codex as the brain. Do not add local fast paths or hardcoded movement intent routing.
- Use `langchain-codex-oauth` / `ChatCodexOAuth` as a normal LangChain chat model.
- Keep `RobotMCPBridge` as the validated execution layer.
- Implement the full agent loop: model tool call -> Python MCP execution -> ToolMessage -> model final response.
- Add timing logs around model calls and tool execution.
```

---

## Parallel execution map

### Can run in parallel first

- **Task 1:** Dependency + Pi OAuth auth-store adapter.
- **Task 2:** Robot tool execution adapter tests/helpers.
- **Task 3:** Timing/logging helper.

### Run after Tasks 1-3

- **Task 4:** Refactor `LangGraphRobotAgent` to LangChain message/tool loop.
- **Task 5:** Wire processor lifecycle/errors to the LangChain model.

### Run after Tasks 4-5

- **Task 6:** Compatibility cleanup and old-client retirement decision.
- **Task 7:** Full validation and live diagnostic checklist.

Do not run Task 4 and Task 5 as simultaneous writers unless using isolated worktrees and merging carefully; both touch agent wiring.

---

## Task 1: Add dependency and Pi OAuth auth-store adapter

**Files:**
- Modify: `server/pyproject.toml`
- Modify: `server/uv.lock` via `uv sync`
- Create: `server/codex_langchain_auth.py`
- Create: `server/tests/test_codex_langchain_auth.py`

### Step 1: Add failing tests for Pi auth-store compatibility

Create `server/tests/test_codex_langchain_auth.py`:

```python
import json
from pathlib import Path

from codex_langchain_auth import PiLangChainCodexAuthStore


def test_pi_langchain_auth_store_loads_pi_oauth_profile(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "access-token",
                    "refresh": "refresh-token",
                    "expires": 9_999_999_999_999,
                    "accountId": "account-id",
                }
            }
        ),
        encoding="utf-8",
    )

    store = PiLangChainCodexAuthStore(auth_file=auth_file)

    creds = store.load()

    assert creds.access == "access-token"
    assert creds.refresh == "refresh-token"
    assert creds.expires == 9_999_999_999_999
    assert creds.account_id == "account-id"


def test_pi_langchain_auth_store_saves_back_to_pi_profile(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "old-access",
                    "refresh": "old-refresh",
                    "expires": 1,
                    "accountId": "old-account",
                }
            }
        ),
        encoding="utf-8",
    )
    store = PiLangChainCodexAuthStore(auth_file=auth_file)
    creds_type = type(store.load())

    store.save(
        creds_type(
            access="new-access",
            refresh="new-refresh",
            expires=2,
            account_id="new-account",
        )
    )

    data = json.loads(auth_file.read_text(encoding="utf-8"))
    assert data["openai-codex"] == {
        "type": "oauth",
        "access": "new-access",
        "refresh": "new-refresh",
        "expires": 2,
        "accountId": "new-account",
    }
```

### Step 2: Run tests and verify RED

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_langchain_auth.py -q
```

Expected before implementation: import failure for `codex_langchain_auth`.

### Step 3: Add dependencies

Modify `server/pyproject.toml` dependencies:

```toml
    "langchain-core>=1.2,<2",
    "langchain-codex-oauth>=1.0,<1.1",
```

Run:

```bash
cd pipecat-agent/server
uv sync
```

### Step 4: Implement Pi auth adapter

Create `server/codex_langchain_auth.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codex_auth import CODEX_PROFILE, CodexAuthError, _account_id_from_jwt, _default_auth_file
from codex_oauth.exceptions import NotAuthenticatedError
from codex_oauth.store import OAuthCredentials


class PiLangChainCodexAuthStore:
    """AuthStore-compatible adapter over Pi's ~/.pi/agent/auth.json Codex profile."""

    def __init__(self, *, auth_file: str | Path | None = None, profile: str = CODEX_PROFILE):
        self.auth_path = Path(auth_file) if auth_file is not None else _default_auth_file()
        self._profile = profile

    def load(self) -> OAuthCredentials:
        data = self._read_auth_file()
        profile = self._profile_data(data)
        access = _required_string(profile, "access")
        refresh = _required_string(profile, "refresh")
        expires = _required_int(profile, "expires")
        account_id = _optional_string(profile, "accountId") or _account_id_from_jwt(access)
        if not account_id:
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth account id is missing. Re-run Pi login.")
        return OAuthCredentials(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=account_id,
        )

    def save(self, creds: OAuthCredentials) -> None:
        data = self._read_auth_file() if self.auth_path.exists() else {}
        data[self._profile] = {
            "type": "oauth",
            "access": creds.access,
            "refresh": creds.refresh,
            "expires": creds.expires,
            "accountId": creds.account_id,
        }
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_path.write_text(f"{json.dumps(data, indent=2, sort_keys=True)}\n", encoding="utf-8")

    def _read_auth_file(self) -> dict[str, Any]:
        if not self.auth_path.exists():
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth credentials not found. Run `pi`, then `/login`.")
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CodexAuthError("Pi OpenAI Codex OAuth auth file is invalid JSON.") from exc
        if not isinstance(data, dict):
            raise CodexAuthError("Pi OpenAI Codex OAuth auth file must contain profiles.")
        return data

    def _profile_data(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = data.get(self._profile)
        if not isinstance(profile, dict) or profile.get("type") != "oauth":
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth profile not found. Run `pi`, then `/login`.")
        return profile


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is missing.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is invalid.")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is invalid.")
    if isinstance(value, (int, float)):
        return int(value)
    raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is missing.")
```

### Step 5: Verify Task 1

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_langchain_auth.py -q
```

Expected: pass.

---

## Task 2: Add LangChain robot tool execution adapter

**Files:**
- Create: `server/robot_control/langchain_tools.py`
- Create: `server/tests/test_robot_langchain_tools.py`

### Step 1: Add failing tests

Create `server/tests/test_robot_langchain_tools.py`:

```python
import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from robot_control.langchain_tools import execute_langchain_tool_calls


class FakeRobotExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return json.dumps({"structured_content": {"ok": True}})


@pytest.mark.asyncio
async def test_executes_langchain_tool_calls_and_returns_tool_messages() -> None:
    executor = FakeRobotExecutor()
    tool_calls = [
        {
            "id": "call-1",
            "name": "moveit_get_current_pose",
            "args": {"robot_name": "UR10"},
            "type": "tool_call",
        }
    ]

    messages = await execute_langchain_tool_calls(tool_calls, executor)

    assert executor.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert messages == [
        ToolMessage(
            content=json.dumps({"structured_content": {"ok": True}}),
            tool_call_id="call-1",
        )
    ]
```

### Step 2: Run tests and verify RED

```bash
cd pipecat-agent/server
uv run pytest tests/test_robot_langchain_tools.py -q
```

Expected before implementation: import failure for `robot_control.langchain_tools`.

### Step 3: Implement adapter

Create `server/robot_control/langchain_tools.py`:

```python
from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import ToolMessage


class RobotToolExecutor(Protocol):
    async def execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str: ...


async def execute_langchain_tool_calls(
    tool_calls: list[dict[str, Any]], executor: RobotToolExecutor
) -> list[ToolMessage]:
    messages: list[ToolMessage] = []
    for call in tool_calls:
        call_id = str(call.get("id") or "")
        name = str(call.get("name") or "")
        args = call.get("args")
        arguments = args if isinstance(args, dict) else {}
        output = await executor.execute_tool_call(name, arguments)
        messages.append(ToolMessage(content=output, tool_call_id=call_id))
    return messages
```

### Step 4: Verify Task 2

```bash
cd pipecat-agent/server
uv run pytest tests/test_robot_langchain_tools.py -q
```

Expected: pass.

---

## Task 3: Add reusable timing logs

**Files:**
- Create: `server/voice_runtime/timing.py`
- Create: `server/tests/test_timing.py`

### Step 1: Add failing tests

Create `server/tests/test_timing.py`:

```python
from voice_runtime.timing import elapsed_ms_since


def test_elapsed_ms_since_rounds_to_two_decimals() -> None:
    assert elapsed_ms_since(10.0, now=10.123456) == 123.46
```

### Step 2: Run test and verify RED

```bash
cd pipecat-agent/server
uv run pytest tests/test_timing.py -q
```

Expected before implementation: import failure for `voice_runtime.timing`.

### Step 3: Implement helper

Create `server/voice_runtime/timing.py`:

```python
from __future__ import annotations

import time


def monotonic_s() -> float:
    return time.monotonic()


def elapsed_ms_since(start_s: float, *, now: float | None = None) -> float:
    current_s = monotonic_s() if now is None else now
    return round((current_s - start_s) * 1000.0, 2)
```

### Step 4: Verify Task 3

```bash
cd pipecat-agent/server
uv run pytest tests/test_timing.py -q
```

Expected: pass.

---

## Task 4: Refactor `LangGraphRobotAgent` to full LangChain agent loop

**Files:**
- Replace most of: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

### Step 1: Rewrite graph tests around LangChain messages

In `server/tests/test_langgraph_robot_agent.py`, replace custom `CodexResponseResult` fake backend with a fake chat model.

Add imports:

```python
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
```

Add fake model:

```python
class FakeChatModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.requests: list[list[BaseMessage]] = []
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        clone = FakeBoundChatModel(self.responses, self.requests, list(tools))
        self.bound_tools = clone.bound_tools
        return clone


class FakeBoundChatModel:
    def __init__(
        self,
        responses: list[AIMessage],
        requests: list[list[BaseMessage]],
        bound_tools: list[dict[str, Any]],
    ):
        self.responses = responses
        self.requests = requests
        self.bound_tools = bound_tools

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.requests.append(list(messages))
        return self.responses.pop(0)
```

Add helper:

```python
def ai_tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )
```

### Step 2: Add failing full-loop test

Add this test:

```python
@pytest.mark.asyncio
async def test_graph_runs_full_langchain_tool_loop_and_returns_final_model_text() -> None:
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            AIMessage(content="The pose is ready."),
        ]
    )
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model=model,
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    text = await graph.run_turn(turn("where is the pose?"))

    assert text == "The pose is ready."
    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert isinstance(model.requests[1][-1], ToolMessage)
    assert model.requests[1][-1].tool_call_id == "call-1"
```

The two pose calls are expected here: one preflight observation injected into context, one because Codex explicitly requested pose.

### Step 3: Run test and verify RED

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_runs_full_langchain_tool_loop_and_returns_final_model_text -q
```

Expected before refactor: constructor/signature or backend mismatch failure.

### Step 4: Refactor state schema

In `server/langgraph_robot_agent.py`, replace `RobotAgentState` fields with LangChain messages:

```python
import operator
from typing import Annotated
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

class RobotAgentState(TypedDict):
    user_text: str
    messages: Annotated[list[BaseMessage], operator.add]
    tools: list[dict[str, Any]]
    tool_turns: int
    observed_this_turn: bool
    final_text: str
    error_text: str | None
```

### Step 5: Refactor constructor

Change `LangGraphRobotAgent.__init__` to accept a LangChain model directly:

```python
    def __init__(
        self,
        *,
        model: Any,
        tool_bridge: Any,
        robot_context: RobotContextStore,
        thread_id: str | None = None,
    ) -> None:
        self._model = model
        self._tool_bridge = tool_bridge
        self._robot_context = robot_context
        self._thread_id = thread_id or f"codex-robot-agent-{uuid.uuid4()}"
        self._latest_state: dict[str, Any] | None = None
        self._graph = self._compile_graph()
```

Remove `_credential_store`, `_backend_client`, `_reasoning_effort`, and `_turn_credentials` from this class.

### Step 6: Refactor `run_turn()` initial state

Use LangChain `HumanMessage`:

```python
    async def run_turn(self, turn: AgentTurnInput) -> str:
        state: RobotAgentState = {
            "user_text": turn.user_text,
            "messages": _messages_from_turn(turn),
            "tools": [],
            "tool_turns": 0,
            "observed_this_turn": False,
            "final_text": "",
            "error_text": None,
        }
        result = await self._graph.ainvoke(
            state,
            {"configurable": {"thread_id": self._thread_id}},
        )
        self._latest_state = result
        return str(result.get("final_text") or NO_TEXT_RESPONSE)
```

Add helper:

```python
def _messages_from_turn(turn: AgentTurnInput) -> list[BaseMessage]:
    return [HumanMessage(content=turn.user_text)]
```

### Step 7: Refactor model-call node

Replace `_call_codex` with `_call_model`:

```python
    async def _call_model(self, state: RobotAgentState) -> dict[str, Any]:
        tools = state["tools"] or self._tool_bridge.function_tools()
        model = self._model.bind_tools(tools)
        system = SystemMessage(content=self._instructions())
        started = monotonic_s()
        logger.info(
            "Codex LangChain request start tool_turns={} messages={} tools={}",
            state["tool_turns"],
            len(state["messages"]),
            len(tools),
        )
        message = await model.ainvoke([system, *state["messages"]])
        logger.info(
            "Codex LangChain request end elapsed_ms={} tool_calls={} text_len={}",
            elapsed_ms_since(started),
            [call.get("name") for call in getattr(message, "tool_calls", [])],
            len(str(message.content or "")),
        )
        return {"messages": [message], "tools": tools}
```

Import timing helpers:

```python
from voice_runtime.timing import elapsed_ms_since, monotonic_s
```

### Step 8: Refactor routing

Use AIMessage tool calls:

```python
    def _route_after_model(self, state: RobotAgentState) -> Literal["execute_robot_tool", "final_response"]:
        if state["error_text"]:
            return "final_response"
        last = _last_ai_message(state["messages"])
        if last is None or not last.tool_calls:
            return "final_response"
        if state["tool_turns"] >= MAX_CODEX_TOOL_TURNS:
            return "final_response"
        return "execute_robot_tool"
```

Add helper:

```python
def _last_ai_message(messages: list[BaseMessage]) -> AIMessage | None:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return message
    return None
```

### Step 9: Refactor tool execution node

Use LangChain tool calls and return `ToolMessage`s:

```python
    async def _execute_robot_tool(self, state: RobotAgentState) -> dict[str, Any]:
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"messages": [], "tool_turns": state["tool_turns"]}

        tool_messages: list[ToolMessage] = []
        observed_this_turn = state["observed_this_turn"]
        for tool_call in last.tool_calls:
            name = str(tool_call.get("name") or "")
            args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
            call_id = str(tool_call.get("id") or "")
            repaired_args = self._repaired_tool_arguments(name, dict(args), state["user_text"])
            started = monotonic_s()
            logger.info("Robot tool start name={} call_id={}", name, call_id)
            if name in OBSERVE_TOOL_NAMES:
                output, observed_this_turn = await self._execute_observation_tool(name, repaired_args)
            else:
                output = await self._execute_tool(name, repaired_args)
                observed_this_turn = False
            logger.info(
                "Robot tool end name={} call_id={} elapsed_ms={}",
                name,
                call_id,
                elapsed_ms_since(started),
            )
            tool_messages.append(ToolMessage(content=output, tool_call_id=call_id))

        return {
            "messages": tool_messages,
            "tool_turns": state["tool_turns"] + 1,
            "observed_this_turn": observed_this_turn,
        }
```

### Step 10: Refactor final response

```python
    def _final_response(self, state: RobotAgentState) -> dict[str, Any]:
        if state["error_text"]:
            return {"final_text": state["error_text"]}
        last = _last_ai_message(state["messages"])
        if last is None:
            return {"final_text": NO_TEXT_RESPONSE}
        text = str(last.content or "").strip()
        return {"final_text": text or NO_TEXT_RESPONSE}
```

### Step 11: Wire graph nodes

In `_compile_graph()`:

```python
        builder.add_node("observe_current_pose", self._observe_current_pose)
        builder.add_node("call_model", self._call_model)
        builder.add_node("execute_robot_tool", self._execute_robot_tool)
        builder.add_node("final_response", self._final_response)
        builder.add_edge(START, "observe_current_pose")
        builder.add_edge("observe_current_pose", "call_model")
        builder.add_conditional_edges("call_model", self._route_after_model)
        builder.add_edge("execute_robot_tool", "call_model")
        builder.add_edge("final_response", END)
```

Remove `repair_tool_arguments` node because repair happens immediately before execution.

### Step 12: Verify Task 4 tests

```bash
cd pipecat-agent/server
uv run pytest tests/test_langgraph_robot_agent.py -q
```

Expected: update old CodexResponseResult-based tests to `AIMessage` fakes until all pass.

---

## Task 5: Wire `OpenAICodexAgentProcessor` to `ChatCodexOAuth`

**Files:**
- Modify: `server/openai_codex_agent_processor.py`
- Modify: `server/tests/test_openai_codex_agent_processor.py`

### Step 1: Add test that processor constructs graph from injected LangChain model

In `server/tests/test_openai_codex_agent_processor.py`, add a fake model similar to Task 4 and test:

```python
@pytest.mark.asyncio
async def test_processor_uses_injected_langchain_model_for_turn() -> None:
    model = FakeChatModel([AIMessage(content="ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=bridge,
    )

    chunks = await _run_turn(processor, "hello")

    assert chunks.text == "ok"
    assert model.requests
```

### Step 2: Run test and verify RED

```bash
cd pipecat-agent/server
uv run pytest tests/test_openai_codex_agent_processor.py::test_processor_uses_injected_langchain_model_for_turn -q
```

Expected before implementation: constructor does not accept `chat_model`.

### Step 3: Update processor constructor

In `server/openai_codex_agent_processor.py`, change constructor args:

```python
        chat_model: Any | None = None,
```

Store:

```python
        self._chat_model = chat_model
        self._graph_chat_model: Any | None = None
```

Remove backend-client ownership where no longer needed for production. Keep `backend_client` only as temporary test compatibility if old tests still use it, or delete after tests are migrated.

### Step 4: Build `ChatCodexOAuth` with Pi auth store

Add imports:

```python
from langchain_codex_oauth import ChatCodexOAuth
from codex_langchain_auth import PiLangChainCodexAuthStore
```

Add method:

```python
    def _chat_model_for_turn(self) -> Any:
        if self._chat_model is not None:
            return self._chat_model
        self._chat_model = ChatCodexOAuth(
            model=self._model,
            auth_store=PiLangChainCodexAuthStore(),
            reasoning_effort=self._reasoning_effort,
            text_verbosity="low",
            system_prompt_mode="strict",
        )
        return self._chat_model
```

### Step 5: Remove direct credential fetch from normal run path

In `run_turn()`, keep a credential check only to fail early with Pi-specific guidance:

```python
        try:
            await self._ensure_connected()
            self._credential_store.get_credentials()
        except CodexAuthError as exc:
            ...
```

Then:

```python
        chat_model = self._chat_model_for_turn()
        graph = self._graph_agent_for(chat_model, tool_bridge)
        yield await graph.run_turn(turn)
```

### Step 6: Update graph factory

```python
    def _graph_agent_for(self, chat_model: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_chat_model is not chat_model
            or self._graph_tool_bridge is not tool_bridge
        ):
            self._graph_agent = LangGraphRobotAgent(
                model=chat_model,
                tool_bridge=tool_bridge,
                robot_context=self._robot_context,
                thread_id=self._thread_id,
            )
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent
```

### Step 7: Verify Task 5

```bash
cd pipecat-agent/server
uv run pytest tests/test_openai_codex_agent_processor.py -q
```

Expected: pass after migrating old backend-client-based tests.

---

## Task 6: Compatibility cleanup and old-client retirement decision

**Files:**
- Modify: `server/tests/test_orthogonal_imports.py`
- Modify: `server/tests/test_robot_control_imports.py`
- Possibly keep: `server/codex_backend_client.py`
- Possibly delete later: `server/codex_backend_client.py`, `server/tests/test_codex_backend_client.py`

### Step 1: Decide whether to keep `codex_backend_client.py`

Recommended first pass: **keep it unused** for one branch to reduce risk. Mark it legacy in a module docstring:

```python
"""Legacy custom Codex backend client.

The production robot agent now uses `langchain-codex-oauth` through `ChatCodexOAuth`.
Keep this module temporarily for parser regression tests and rollback safety.
"""
```

### Step 2: Update import tests only if needed

If import tests fail because dependencies changed, update expected module names. Do not delete old tests until live validation succeeds.

### Step 3: Run compatibility tests

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_backend_client.py tests/test_orthogonal_imports.py tests/test_robot_control_imports.py -q
```

Expected: pass.

---

## Task 7: Full validation and live diagnostic checklist

**Files:**
- No required code changes.
- Update docs only if the project has a live-run notes file.

### Step 1: Run targeted suite

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_langchain_auth.py tests/test_robot_langchain_tools.py tests/test_timing.py tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_robot_mcp_bridge.py tests/test_robot_call_validation.py -q
```

Expected: all pass.

### Step 2: Run full suite

```bash
cd pipecat-agent/server
uv run pytest -q
```

Expected: all pass.

### Step 3: Run static checks

```bash
cd pipecat-agent/server
uv run ruff check .
uv run pyright .
```

Expected: no errors.

### Step 4: Live voice diagnostic expectations

Run the same command: “have the robot wave to me”. Look for logs shaped like:

```text
Codex LangChain request start tool_turns=0 messages=1 tools=...
Codex LangChain request end elapsed_ms=... tool_calls=['moveit_plan_and_execute_cartesian_motion'] text_len=0
Robot tool start name=moveit_plan_and_execute_cartesian_motion call_id=...
Robot tool end name=moveit_plan_and_execute_cartesian_motion call_id=... elapsed_ms=...
Codex LangChain request start tool_turns=1 messages=3 tools=...
Codex LangChain request end elapsed_ms=... tool_calls=[] text_len=...
```

If it hangs, the timing logs identify which boundary is slow:

- first Codex request = planning/model issue
- robot tool = MCP/MoveIt issue
- second Codex request = final-answer/model issue

### Step 5: Success criteria

- Codex OAuth is used as a normal LangChain chat model.
- Current pose appears in the system message sent to Codex.
- Codex returns `AIMessage.tool_calls` through `.bind_tools(...)`.
- Python executes MCP tools and returns `ToolMessage`s.
- Codex receives tool result and writes final response.
- No local movement intent router is introduced.
- Timing logs clearly show where latency occurs.
