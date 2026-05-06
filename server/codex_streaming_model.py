from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, AIMessageChunk, message_chunk_to_message


class StreamingAinvokeChatModel:
    """Use streaming chunks to implement ainvoke for Codex OAuth responses."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> StreamingAinvokeChatModel:
        return StreamingAinvokeChatModel(self._model.bind_tools(tools, **kwargs))

    async def ainvoke(self, messages: list[Any], *args: Any, **kwargs: Any) -> AIMessage:
        merged: AIMessageChunk | None = None
        async for chunk in self._model.astream(messages, *args, **kwargs):
            if not isinstance(chunk, AIMessageChunk):
                content = getattr(chunk, "content", "")
                chunk = AIMessageChunk(content=content)
            merged = chunk if merged is None else merged + chunk

        if merged is None:
            return AIMessage(content="")

        message = message_chunk_to_message(merged)
        if isinstance(message, AIMessage):
            return message
        return AIMessage(content=message.content or "")
