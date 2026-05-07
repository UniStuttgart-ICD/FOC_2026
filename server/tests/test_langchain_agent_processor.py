import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from agent_control.langchain_agent_processor import LangChainAgentProcessor
from process_trace import MemoryTraceWriter, ProcessTracer
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


def records_named(writer: MemoryTraceWriter, name: str) -> list[dict[str, Any]]:
    return [record for record in writer.records if record["name"] == name]


@pytest.mark.asyncio
async def test_generic_processor_runs_langgraph_without_oauth_credentials():
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


@pytest.mark.asyncio
async def test_processor_emits_backend_turn_and_passes_tracer_to_created_bridge_and_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_bridges: list[Any] = []
    created_graphs: list[Any] = []

    class CreatedBridge(FakeBridge):
        def __init__(self, url: str, *, tracer: ProcessTracer):
            super().__init__()
            self.url = url
            self.tracer = tracer
            created_bridges.append(self)

    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            thread_id: str,
            tracer: ProcessTracer,
        ):
            self.model = model
            self.tool_bridge = tool_bridge
            self.robot_context = robot_context
            self.thread_id = thread_id
            self.tracer = tracer
            created_graphs.append(self)

        async def run_turn(self, turn: AgentTurnInput) -> str:
            return f"fake graph: {turn.user_text}"

    monkeypatch.setattr("agent_control.langchain_agent_processor.RobotMCPBridge", CreatedBridge)
    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="gpt-5.4-mini",
        tracer=tracer,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["fake graph: hello"]
    assert created_bridges[0].tracer is tracer
    assert created_graphs[0].tracer is tracer
    backend_span = records_named(writer, "agent.backend_turn")[-1]
    assert backend_span["record_type"] == "span"
    assert backend_span["module"] == "agent_control"
    assert backend_span["status"] == "ok"
    assert backend_span["attributes"] == {
        "model_label": "gpt-5.4-mini",
        "message_count": 1,
    }


@pytest.mark.asyncio
async def test_backend_turn_span_is_recorded_before_yielded_chunk_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            thread_id: str,
            tracer: ProcessTracer,
        ):
            pass

        async def run_turn(self, turn: AgentTurnInput) -> str:
            return f"fake graph: {turn.user_text}"

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    writer = MemoryTraceWriter()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="gpt-5.4-mini",
        tool_bridge=FakeBridge(),
        tracer=ProcessTracer(writer),
    )
    turn = AgentTurnInput(user_text="hello", messages=[{"role": "user", "content": "hello"}])
    chunks = processor.run_turn(turn)

    first_chunk = await chunks.__anext__()

    assert first_chunk == "fake graph: hello"
    backend_spans = records_named(writer, "agent.backend_turn")
    assert len(backend_spans) == 1
    assert backend_spans[0]["status"] == "ok"
    assert "error_type" not in backend_spans[0]["attributes"]

    await chunks.aclose()

    assert records_named(writer, "agent.backend_turn") == backend_spans
