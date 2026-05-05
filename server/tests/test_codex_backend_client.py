import json

import httpx
import pytest

from codex_auth import CodexCredentials
from codex_backend_client import CodexBackendClient, CodexBackendError


def _sse(*events: dict) -> bytes:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events).encode()


@pytest.mark.asyncio
async def test_posts_to_codex_backend_with_oauth_headers_and_body():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = request.headers
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            content=_sse(
                {"type": "response.output_text.delta", "delta": "oauth"},
                {"type": "response.output_text.delta", "delta": "-ok"},
                {"type": "response.completed", "response": {"id": "resp-1", "status": "completed"}},
            ),
            headers={"content-type": "text/event-stream"},
        )

    client = CodexBackendClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await client.create_response(
        CodexCredentials(access="access-token", refresh="refresh-token", account_id="acct-1"),
        model="gpt-5.4-mini",
        instructions="system",
        input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
        tools=[{"type": "function", "name": "get_robot_status", "parameters": {"type": "object"}, "strict": None}],
    )

    assert captured["url"] == "https://chatgpt.com/backend-api/codex/responses"
    assert captured["headers"]["authorization"] == "Bearer access-token"
    assert captured["headers"]["chatgpt-account-id"] == "acct-1"
    assert captured["headers"]["originator"] == "pi"
    assert captured["headers"]["openai-beta"] == "responses=experimental"
    assert captured["body"]["model"] == "gpt-5.4-mini"
    assert captured["body"]["store"] is False
    assert captured["body"]["stream"] is True
    assert captured["body"]["instructions"] == "system"
    assert captured["body"]["tool_choice"] == "auto"
    assert captured["body"]["parallel_tool_calls"] is False
    assert result.text == "oauth-ok"
    assert result.response_id == "resp-1"

    await client.close()


@pytest.mark.asyncio
async def test_parses_function_call_arguments_and_output_item():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {
                    "type": "response.output_item.added",
                    "item": {"type": "function_call", "id": "item-1", "call_id": "call-1", "name": "get_robot_status", "arguments": ""},
                },
                {"type": "response.function_call_arguments.delta", "delta": '{"robot'},
                {"type": "response.function_call_arguments.delta", "delta": '_ip":"127.0.0.1"}'},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "item-1",
                        "call_id": "call-1",
                        "name": "get_robot_status",
                        "arguments": '{"robot_ip":"127.0.0.1"}',
                    },
                },
                {"type": "response.completed", "response": {"id": "resp-1", "status": "completed"}},
            ),
        )

    client = CodexBackendClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await client.create_response(
        CodexCredentials(access="access-token", refresh="refresh-token", account_id="acct-1"),
        model="gpt-5.4-mini",
        instructions="system",
        input_items=[],
        tools=[],
    )

    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call.call_id == "call-1"
    assert call.item_id == "item-1"
    assert call.name == "get_robot_status"
    assert call.arguments == {"robot_ip": "127.0.0.1"}
    assert result.output_items == [
        {
            "type": "function_call",
            "id": "item-1",
            "call_id": "call-1",
            "name": "get_robot_status",
            "arguments": '{"robot_ip":"127.0.0.1"}',
        }
    ]

    await client.close()


@pytest.mark.asyncio
async def test_preserves_reasoning_output_items_for_tool_continuation():
    reasoning_item = {
        "type": "reasoning",
        "id": "rs-1",
        "summary": [],
        "encrypted_content": "encrypted",
        "status": "completed",
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=_sse(
                {"type": "response.output_item.added", "item": reasoning_item},
                {"type": "response.output_item.done", "item": reasoning_item},
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "function_call",
                        "id": "item-1",
                        "call_id": "call-1",
                        "name": "get_robot_status",
                        "arguments": "{}",
                    },
                },
                {"type": "response.completed", "response": {"id": "resp-1", "status": "completed"}},
            ),
        )

    client = CodexBackendClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    result = await client.create_response(
        CodexCredentials(access="access-token", refresh="refresh-token", account_id="acct-1"),
        model="gpt-5.4-mini",
        instructions="system",
        input_items=[],
        tools=[],
    )

    assert result.output_items[0] == reasoning_item
    assert result.output_items[1]["type"] == "function_call"

    await client.close()


@pytest.mark.asyncio
async def test_backend_error_includes_safe_response_diagnostics_without_tokens():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            400,
            json={"detail": "Invalid input item", "code": "bad_request"},
        )

    client = CodexBackendClient(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))

    with pytest.raises(CodexBackendError) as exc_info:
        await client.create_response(
            CodexCredentials(access="secret-token", refresh="refresh-token", account_id="acct-1"),
            model="gpt-5.4-mini",
            instructions="system",
            input_items=[{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            tools=[{"type": "function", "name": "moveit_get_current_pose"}],
        )

    message = str(exc_info.value)
    assert "OpenAI Codex backend request failed: HTTP 400" in message
    assert "Invalid input item" in message
    assert "input_summary=[{'index': 0, 'type': None, 'role': 'user'}]" in message
    assert "tool_names=['moveit_get_current_pose']" in message
    assert "secret-token" not in message
    assert "refresh-token" not in message
    assert captured["body"]["model"] == "gpt-5.4-mini"

    await client.close()
