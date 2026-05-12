from __future__ import annotations

import json
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult, TextContent

from process_trace import NoopProcessTracer, ProcessTracer

ProcessTracerLike = ProcessTracer | NoopProcessTracer


class UserSensingMCPError(RuntimeError):
    """Raised when Vizor user-sensing MCP setup or execution fails."""


class UserSensingMCPBridge:
    def __init__(
        self,
        mcp_server_url: str,
        *,
        tracer: ProcessTracerLike | None = None,
        client_session_timeout_seconds: float = 30,
    ) -> None:
        self._mcp_server_url = mcp_server_url
        self._tracer = tracer or NoopProcessTracer()
        self._timeout = timedelta(seconds=client_session_timeout_seconds)
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        if self._session is not None:
            return
        stack = AsyncExitStack()
        try:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamable_http_client(self._mcp_server_url)
            )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=self._timeout,
                )
            )
            await session.initialize()
        except Exception:
            await stack.aclose()
            raise
        self._exit_stack = stack
        self._session = session

    async def disconnect(self) -> None:
        if self._exit_stack is not None:
            await self._exit_stack.aclose()
        self._exit_stack = None
        self._session = None

    async def read_context(self, *, max_age_s: float) -> str:
        if self._session is None:
            raise UserSensingMCPError("Vizor MCP server is not connected")
        result = await self._session.call_tool(
            "vizor_get_sensor_context",
            {"max_age_s": max_age_s},
        )
        return _serialize_tool_result(result)


def _serialize_tool_result(result: CallToolResult) -> str:
    content: list[str] = []
    for item in result.content:
        if isinstance(item, TextContent):
            content.append(item.text)
        else:
            content.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    return json.dumps(
        {
            "content": content,
            "structured_content": result.structuredContent,
            "is_error": result.isError,
        },
        ensure_ascii=False,
    )
