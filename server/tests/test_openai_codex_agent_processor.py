import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from codex_auth import CodexAuthError, CodexCredentials
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput


class Store:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class AuthErrorStore:
    def get_credentials(self):
        raise CodexAuthError("login required")


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


class FakeBridge:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.calls = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    def function_tools(self):
        return [
            {
                "type": "function",
                "name": "moveit_get_current_pose",
                "parameters": {"type": "object"},
                "strict": None,
            }
        ]

    async def call_tool(self, name, arguments) -> str:
        self.calls.append((name, arguments))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": "UR10",
                    "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
                }
            }
        )


class LegacyBackend:
    def __init__(self):
        self.closed = False

    async def close(self):
        self.closed = True


@dataclass(frozen=True)
class TurnResult:
    chunks: list[str]
    processor: OpenAICodexAgentProcessor


async def _run_turn(processor: OpenAICodexAgentProcessor, text: str, messages=None) -> TurnResult:
    turn = AgentTurnInput(user_text=text, messages=messages or [{"role": "user", "content": text}])
    chunks = [chunk async for chunk in processor.run_turn(turn)]
    return TurnResult(chunks=chunks, processor=processor)


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def ai_tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


@pytest.mark.asyncio
async def test_auth_error_does_not_observe_robot_before_returning_guidance():
    bridge = FakeBridge()
    model = FakeChatModel([ai_text("should-not-run")])
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=AuthErrorStore(),
        chat_model=model,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["login required"]
    assert bridge.calls == []
    assert model.requests == []


@pytest.mark.asyncio
async def test_reuses_langgraph_runner_between_turns():
    model = FakeChatModel([ai_text("one"), ai_text("two")])
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=FakeBridge(),
    )

    await _run_turn(processor, "one")
    first_graph = processor._graph_agent
    await _run_turn(processor, "two")

    assert first_graph is not None
    assert processor._graph_agent is first_graph


@pytest.mark.asyncio
async def test_processor_uses_injected_langchain_model_for_turn() -> None:
    model = FakeChatModel([ai_text("ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["ok"]
    assert model.requests
    assert bridge.connected is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]


@pytest.mark.asyncio
async def test_executes_one_tool_iteration_and_sends_tool_message_back():
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("Robot pose is ready."),
        ]
    )
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "where is the pose?")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert isinstance(model.requests[1][-1], ToolMessage)
    assert model.requests[1][-1].tool_call_id == "call-1"
    assert result.chunks == ["Robot pose is ready."]


@pytest.mark.asyncio
async def test_disconnect_closes_legacy_backend_and_bridge():
    backend = LegacyBackend()
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        chat_model=FakeChatModel([]),
        tool_bridge=bridge,
    )

    await processor.connect()
    await processor.disconnect()

    assert backend.closed is True
    assert bridge.disconnected is True


@pytest.mark.asyncio
async def test_injects_compact_robot_context_into_model_system_message():
    model = FakeChatModel([ai_text("ok")])
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=FakeBridge(),
    )

    await _run_turn(processor, "what can you do?")

    system = model.requests[0][0]
    assert isinstance(system, SystemMessage)
    instructions = str(system.content)
    assert "Last-known robot context" in instructions
    assert "advisory only" in instructions
    assert "moveit_get_current_pose" in instructions


@pytest.mark.asyncio
async def test_updates_robot_context_after_current_pose_tool_result():
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("Robot pose is ready."),
            ai_text("ok"),
        ]
    )
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        chat_model=model,
        tool_bridge=FakeBridge(),
    )

    await _run_turn(processor, "pose")
    await _run_turn(processor, "what can you do?")

    instructions = str(model.requests[-1][0].content)
    assert "robot: UR10" in instructions
    assert "x=0.100" in instructions


def test_chat_model_for_turn_builds_codex_oauth_model(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class CapturedChatModel:
        def __init__(self, **kwargs: Any):
            created.update(kwargs)

    monkeypatch.setattr("openai_codex_agent_processor.ChatCodexOAuth", CapturedChatModel)
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        reasoning_effort="medium",
        credential_store=Store(),
        tool_bridge=FakeBridge(),
    )

    chat_model = processor._chat_model_for_turn()

    assert isinstance(chat_model, CapturedChatModel)
    assert created["model"] == "gpt-5.4-mini"
    assert created["reasoning_effort"] == "medium"
    assert created["text_verbosity"] == "low"
    assert created["system_prompt_mode"] == "strict"
    assert created["auth_store"] is not None
