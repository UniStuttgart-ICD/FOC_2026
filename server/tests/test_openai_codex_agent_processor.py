import json
from dataclasses import dataclass

import pytest

from codex_auth import CodexAuthError, CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput


class FakeStore:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class AuthErrorStore:
    def get_credentials(self):
        raise CodexAuthError("login required")


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
            {"type": "function", "name": "moveit_get_current_pose", "parameters": {"type": "object"}, "strict": None}
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


class FakeBackend:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []
        self.closed = False

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


@pytest.mark.asyncio
async def test_auth_error_does_not_observe_robot_before_returning_guidance():
    backend = FakeBackend([CodexResponseResult(text="should-not-run")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=AuthErrorStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["login required"]
    assert bridge.calls == []
    assert backend.requests == []


@pytest.mark.asyncio
async def test_reuses_langgraph_runner_between_turns():
    backend = FakeBackend([CodexResponseResult(text="one"), CodexResponseResult(text="two")])
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=FakeBridge(),
    )

    await _run_turn(processor, "one")
    first_graph = processor._graph_agent
    await _run_turn(processor, "two")

    assert first_graph is not None
    assert processor._graph_agent is first_graph


@pytest.mark.asyncio
async def test_processes_text_response_through_codex_backend():
    backend = FakeBackend([CodexResponseResult(text="oauth-ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["oauth-ok"]
    assert backend.requests[0]["model"] == "gpt-5.4-mini"
    assert backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert bridge.connected is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]


@pytest.mark.asyncio
async def test_executes_one_tool_iteration_and_sends_tool_output_back():
    tool_call = CodexToolCall(
        call_id="call-1",
        item_id="item-1",
        name="moveit_get_current_pose",
        arguments={"robot_name": "UR10"},
        raw_arguments='{"robot_name":"UR10"}',
    )
    backend = FakeBackend(
        [
            CodexResponseResult(
                tool_calls=[tool_call],
                output_items=[
                    {
                        "type": "function_call",
                        "id": "item-1",
                        "call_id": "call-1",
                        "name": "moveit_get_current_pose",
                        "arguments": '{"robot_name":"UR10"}',
                    }
                ],
            ),
            CodexResponseResult(text="Robot pose is ready."),
        ]
    )
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "where is the pose?")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert backend.requests[1]["input_items"][-1]["type"] == "function_call_output"
    assert backend.requests[1]["input_items"][-1]["call_id"] == "call-1"
    assert result.chunks == ["Robot pose is ready."]


@pytest.mark.asyncio
async def test_sends_available_context_history_to_codex_backend():
    backend = FakeBackend([CodexResponseResult(text="ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )
    messages = [
        {"role": "user", "content": "move up"},
        {"role": "assistant", "content": "Moved up."},
        {"role": "user", "content": "again"},
    ]

    await _run_turn(processor, "again", messages=messages)

    assert backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "move up"}]},
        {"role": "assistant", "content": "Moved up."},
        {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
    ]
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]


@pytest.mark.asyncio
async def test_disconnect_closes_backend_and_bridge():
    backend = FakeBackend([])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await processor.connect()
    await processor.disconnect()

    assert backend.closed is True
    assert bridge.disconnected is True


@pytest.mark.asyncio
async def test_injects_compact_robot_context_into_codex_instructions():
    backend = FakeBackend([CodexResponseResult(text="ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await _run_turn(processor, "what can you do?")

    instructions = backend.requests[0]["instructions"]
    assert "Last-known robot context" in instructions
    assert "advisory only" in instructions
    assert "moveit_get_current_pose" in instructions


@pytest.mark.asyncio
async def test_updates_robot_context_after_current_pose_tool_result():
    pose_call = CodexToolCall(
        call_id="call-1",
        item_id="item-1",
        name="moveit_get_current_pose",
        arguments={"robot_name": "UR10"},
        raw_arguments='{"robot_name":"UR10"}',
    )
    backend = FakeBackend(
        [
            CodexResponseResult(
                tool_calls=[pose_call],
                output_items=[
                    {
                        "type": "function_call",
                        "id": "item-1",
                        "call_id": "call-1",
                        "name": "moveit_get_current_pose",
                        "arguments": '{"robot_name":"UR10"}',
                    }
                ],
            ),
            CodexResponseResult(text="Robot pose is ready."),
        ]
    )

    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=FakeBridge(),
    )

    await _run_turn(processor, "pose")

    followup_backend = FakeBackend([CodexResponseResult(text="ok")])
    processor._backend_client = followup_backend
    await _run_turn(processor, "what can you do?")

    instructions = followup_backend.requests[0]["instructions"]
    assert "robot: UR10" in instructions
    assert "x=0.100" in instructions
