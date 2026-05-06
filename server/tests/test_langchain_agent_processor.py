import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from langchain_agent_processor import LangChainAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput


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


@dataclass(frozen=True)
class TurnResult:
    chunks: list[str]
    processor: LangChainAgentProcessor


async def _run_turn(processor: LangChainAgentProcessor, text: str) -> TurnResult:
    turn = AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])
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
async def test_generic_processor_runs_langgraph_without_codex_credentials():
    model = FakeChatModel([ai_text("ready")])
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gpt-5.4-mini",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["ready"]
    assert bridge.connected is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert model.requests


@pytest.mark.asyncio
async def test_generic_processor_executes_model_tool_call():
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("pose observed"),
        ]
    )
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gemini-2.5-flash",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "where are you?")

    assert result.chunks == ["pose observed"]
    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
