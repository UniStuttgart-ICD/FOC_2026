from dataclasses import dataclass

import pytest
from pipecat.frames.frames import LLMContextFrame, LLMTextFrame
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection

from codex_auth import CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from openai_codex_agent_processor import OpenAICodexAgentProcessor


class CapturingProcessor(OpenAICodexAgentProcessor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append(frame)


class FakeStore:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


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
        return [{"type": "function", "name": "get_robot_status", "parameters": {"type": "object"}, "strict": None}]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return '{"success": true}'


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


def _context_frame(text: str) -> LLMContextFrame:
    context = LLMContext(messages=[{"role": "user", "content": text}])
    return LLMContextFrame(context=context)


@pytest.mark.asyncio
async def test_processes_text_response_through_codex_backend():
    backend = FakeBackend([CodexResponseResult(text="oauth-ok")])
    bridge = FakeBridge()
    processor = CapturingProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await processor.process_frame(_context_frame("hello"), FrameDirection.DOWNSTREAM)

    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    assert text_frames[0].text == "oauth-ok"
    assert backend.requests[0]["model"] == "gpt-5.4-mini"
    assert backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert bridge.connected is True


@pytest.mark.asyncio
async def test_executes_one_tool_iteration_and_sends_tool_output_back():
    tool_call = CodexToolCall(
        call_id="call-1",
        item_id="item-1",
        name="get_robot_status",
        arguments={"robot_ip": "127.0.0.1"},
        raw_arguments='{"robot_ip":"127.0.0.1"}',
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
                        "name": "get_robot_status",
                        "arguments": '{"robot_ip":"127.0.0.1"}',
                    }
                ],
            ),
            CodexResponseResult(text="Robot is ready."),
        ]
    )
    bridge = FakeBridge()
    processor = CapturingProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await processor.process_frame(_context_frame("status"), FrameDirection.DOWNSTREAM)

    assert bridge.calls == [("get_robot_status", {"robot_ip": "127.0.0.1"})]
    assert backend.requests[1]["input_items"][-1] == {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": '{"success": true}',
    }
    text_frames = [frame for frame in processor.pushed if isinstance(frame, LLMTextFrame)]
    assert text_frames[0].text == "Robot is ready."


@pytest.mark.asyncio
async def test_sends_available_context_history_to_codex_backend():
    backend = FakeBackend([CodexResponseResult(text="ok")])
    bridge = FakeBridge()
    processor = CapturingProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )
    context = LLMContext(
        messages=[
            {"role": "user", "content": "move up"},
            {"role": "assistant", "content": "Moved up."},
            {"role": "user", "content": "again"},
        ]
    )

    await processor.process_frame(LLMContextFrame(context=context), FrameDirection.DOWNSTREAM)

    assert backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "move up"}]},
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "Moved up.", "annotations": []}],
            "status": "completed",
            "id": "history-assistant-1",
        },
        {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
    ]


@pytest.mark.asyncio
async def test_disconnect_closes_backend_and_bridge():
    backend = FakeBackend([])
    bridge = FakeBridge()
    processor = CapturingProcessor(
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
