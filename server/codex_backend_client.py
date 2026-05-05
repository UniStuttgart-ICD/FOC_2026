from __future__ import annotations

import json
import platform
from dataclasses import dataclass, field
from typing import Any

import httpx

from codex_auth import CodexCredentials

CODEX_BACKEND_URL = "https://chatgpt.com/backend-api/codex/responses"


class CodexBackendError(RuntimeError):
    """Raised when the ChatGPT Codex backend request fails."""


@dataclass(frozen=True)
class CodexToolCall:
    call_id: str
    item_id: str | None
    name: str
    arguments: dict[str, Any]
    raw_arguments: str


@dataclass(frozen=True)
class CodexResponseResult:
    text: str = ""
    tool_calls: list[CodexToolCall] = field(default_factory=list)
    output_items: list[dict[str, Any]] = field(default_factory=list)
    response_id: str | None = None


class CodexBackendClient:
    """Small SSE client for the ChatGPT Codex backend."""

    def __init__(self, *, http_client: httpx.AsyncClient | None = None, url: str = CODEX_BACKEND_URL):
        self._client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self._owns_client = http_client is None
        self._url = url

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def create_response(
        self,
        credentials: CodexCredentials,
        *,
        model: str,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CodexResponseResult:
        if not credentials.account_id:
            raise CodexBackendError("OpenAI Codex OAuth account id is missing. Re-run Pi login.")

        body = _build_body(model=model, instructions=instructions, input_items=input_items, tools=tools)
        headers = _build_headers(credentials)

        try:
            async with self._client.stream("POST", self._url, headers=headers, json=body) as response:
                if response.status_code >= 400:
                    error_detail = await _response_error_detail(response)
                    raise CodexBackendError(
                        _backend_error_message(response.status_code, error_detail, body)
                    )
                return await _parse_sse_response(response)
        except httpx.HTTPError as exc:
            raise CodexBackendError("OpenAI Codex backend request failed") from exc


def _build_body(
    *, model: str, instructions: str, input_items: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "instructions": instructions,
        "input": input_items,
        "text": {"verbosity": "low"},
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }
    if tools:
        body["tools"] = tools
    return body


def _build_headers(credentials: CodexCredentials) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {credentials.access}",
        "chatgpt-account-id": credentials.account_id or "",
        "originator": "pi",
        "User-Agent": f"pipecat-agent ({platform.system()} {platform.release()}; {platform.machine()})",
        "OpenAI-Beta": "responses=experimental",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


async def _response_error_detail(response: httpx.Response) -> str:
    content = await response.aread()
    if not content:
        return ""
    text = content.decode(response.encoding or "utf-8", errors="replace").strip()
    if not text:
        return ""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return _truncate(text)
    return _truncate(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _backend_error_message(status_code: int, detail: str, body: dict[str, Any]) -> str:
    parts = [f"OpenAI Codex backend request failed: HTTP {status_code}"]
    if detail:
        parts.append(detail)
    parts.append(f"input_summary={_input_summary(body.get('input'))}")
    parts.append(f"tool_names={_tool_names(body.get('tools'))}")
    return "; ".join(parts)


def _input_summary(input_items: Any) -> list[dict[str, Any]]:
    if not isinstance(input_items, list):
        return []
    summary: list[dict[str, Any]] = []
    for index, item in enumerate(input_items):
        if isinstance(item, dict):
            summary.append(
                {
                    "index": index,
                    "type": item.get("type"),
                    "role": item.get("role"),
                }
            )
        else:
            summary.append({"index": index, "type": type(item).__name__, "role": None})
    return summary


def _tool_names(tools: Any) -> list[str]:
    if not isinstance(tools, list):
        return []
    names: list[str] = []
    for tool in tools:
        if isinstance(tool, dict) and isinstance(tool.get("name"), str):
            names.append(tool["name"])
    return names


def _truncate(text: str, max_length: int = 1000) -> str:
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."


async def _parse_sse_response(response: httpx.Response) -> CodexResponseResult:
    text_parts: list[str] = []
    tool_calls: dict[str, dict[str, Any]] = {}
    output_items: list[dict[str, Any]] = []
    response_ids: list[str] = []
    event_lines: list[str] = []

    async for line in response.aiter_lines():
        if not line:
            _process_sse_record("\n".join(event_lines), text_parts, tool_calls, output_items, response_ids)
            event_lines.clear()
            continue
        if line.startswith("data:"):
            event_lines.append(line.removeprefix("data:").strip())

    if event_lines:
        _process_sse_record("\n".join(event_lines), text_parts, tool_calls, output_items, response_ids)

    parsed_calls = [_tool_call_from_state(state) for state in tool_calls.values()]
    for state in tool_calls.values():
        item = _function_call_item_from_state(state)
        if item is not None and item not in output_items:
            output_items.append(item)

    response_id = response_ids[-1] if response_ids else None

    return CodexResponseResult(
        text="".join(text_parts).strip(),
        tool_calls=parsed_calls,
        output_items=output_items,
        response_id=response_id,
    )


def _process_sse_record(
    record: str,
    text_parts: list[str],
    tool_calls: dict[str, dict[str, Any]],
    output_items: list[dict[str, Any]],
    response_ids: list[str],
) -> None:
    if not record or record == "[DONE]":
        return
    try:
        event = json.loads(record)
    except json.JSONDecodeError as exc:
        raise CodexBackendError("OpenAI Codex backend returned invalid SSE JSON") from exc
    if not isinstance(event, dict):
        return

    event_type = event.get("type")
    if event_type == "error":
        raise CodexBackendError("OpenAI Codex backend returned an error event")
    if event_type == "response.failed":
        raise CodexBackendError("OpenAI Codex backend response failed")
    if event_type in {"response.output_text.delta", "response.refusal.delta"}:
        delta = event.get("delta")
        if isinstance(delta, str):
            text_parts.append(delta)
        return
    if event_type == "response.output_item.added":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "function_call":
            _state_for_tool_call(tool_calls, item)
        return
    if event_type == "response.function_call_arguments.delta":
        call_id = _event_call_id(event, tool_calls)
        delta = event.get("delta")
        if call_id and isinstance(delta, str):
            tool_calls.setdefault(call_id, {"call_id": call_id, "arguments": ""})["arguments"] = (
                str(tool_calls[call_id].get("arguments", "")) + delta
            )
        return
    if event_type == "response.function_call_arguments.done":
        call_id = _event_call_id(event, tool_calls)
        arguments = event.get("arguments")
        if call_id and isinstance(arguments, str):
            tool_calls.setdefault(call_id, {"call_id": call_id})["arguments"] = arguments
        return
    if event_type == "response.output_item.done":
        item = event.get("item")
        if not isinstance(item, dict):
            return
        if item.get("type") == "function_call":
            state = _state_for_tool_call(tool_calls, item)
            if isinstance(item.get("arguments"), str):
                state["arguments"] = item["arguments"]
            function_item = _function_call_item_from_state(state)
            if function_item is not None:
                output_items.append(function_item)
        elif item.get("type") in {"reasoning", "message"}:
            output_items.append(item)
        return
    if event_type in {"response.completed", "response.done", "response.incomplete"}:
        response = event.get("response")
        if isinstance(response, dict) and isinstance(response.get("id"), str):
            response_ids.append(response["id"])
        return


def _state_for_tool_call(tool_calls: dict[str, dict[str, Any]], item: dict[str, Any]) -> dict[str, Any]:
    call_id = item.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        item_id_value = item.get("id")
        call_id = item_id_value if isinstance(item_id_value, str) else f"call-{len(tool_calls) + 1}"
    state = tool_calls.setdefault(call_id, {"call_id": call_id, "arguments": ""})
    state["item_id"] = item.get("id") if isinstance(item.get("id"), str) else None
    state["name"] = item.get("name") if isinstance(item.get("name"), str) else ""
    if isinstance(item.get("arguments"), str):
        state["arguments"] = item["arguments"]
    return state


def _event_call_id(event: dict[str, Any], tool_calls: dict[str, dict[str, Any]]) -> str | None:
    call_id = event.get("call_id")
    if isinstance(call_id, str):
        return call_id
    item_id = event.get("item_id")
    if isinstance(item_id, str):
        for existing_call_id, state in tool_calls.items():
            if state.get("item_id") == item_id:
                return existing_call_id
    if len(tool_calls) == 1:
        return next(iter(tool_calls))
    return None


def _tool_call_from_state(state: dict[str, Any]) -> CodexToolCall:
    raw_arguments = str(state.get("arguments") or "{}")
    try:
        parsed = json.loads(raw_arguments)
    except json.JSONDecodeError:
        parsed = {}
    arguments = parsed if isinstance(parsed, dict) else {}
    return CodexToolCall(
        call_id=str(state.get("call_id") or ""),
        item_id=state.get("item_id") if isinstance(state.get("item_id"), str) else None,
        name=str(state.get("name") or ""),
        arguments=arguments,
        raw_arguments=raw_arguments,
    )


def _function_call_item_from_state(state: dict[str, Any]) -> dict[str, Any] | None:
    call_id = state.get("call_id")
    name = state.get("name")
    if not isinstance(call_id, str) or not call_id or not isinstance(name, str) or not name:
        return None
    item: dict[str, Any] = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": str(state.get("arguments") or "{}"),
    }
    item_id = state.get("item_id")
    if isinstance(item_id, str) and item_id:
        item["id"] = item_id
    return item
