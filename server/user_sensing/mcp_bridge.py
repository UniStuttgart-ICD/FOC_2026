from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
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
        self._connected = False
        # Test hook for exercising call behavior without opening a network session.
        self._session: ClientSession | None = None

    async def connect(self) -> None:
        if self._session is not None:
            self._connected = True
            return
        async with self._client_session():
            self._connected = True

    async def disconnect(self) -> None:
        self._connected = False
        self._session = None

    async def read_context(self, *, max_age_s: float) -> str:
        if self._session is not None:
            result = await self._session.call_tool(
                "vizor_get_sensor_context",
                {"max_age_s": max_age_s},
            )
            return _serialize_tool_result(result)
        async with self._client_session() as session:
            result = await session.call_tool(
                "vizor_get_sensor_context",
                {"max_age_s": max_age_s},
            )
        return _serialize_tool_result(result)

    @asynccontextmanager
    async def _client_session(self) -> AsyncIterator[ClientSession]:
        async with streamable_http_client(self._mcp_server_url) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(
                read_stream,
                write_stream,
                read_timeout_seconds=self._timeout,
            ) as session:
                await session.initialize()
                yield session


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
