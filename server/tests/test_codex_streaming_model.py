from typing import Any

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from codex_streaming_model import StreamingAinvokeChatModel


class FakeStreamingModel:
    def __init__(self, chunks: list[AIMessageChunk], *, tools: list[dict[str, Any]] | None = None):
        self.chunks = chunks
        self.tools = tools
        self.seen_messages: list[Any] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        return FakeStreamingModel(self.chunks, tools=tools)

    async def astream(self, messages: list[Any]):
        self.seen_messages.append(messages)
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_streaming_ainvoke_model_aggregates_text_chunks() -> None:
    base = FakeStreamingModel([AIMessageChunk(content="O"), AIMessageChunk(content="K")])
    model = StreamingAinvokeChatModel(base)

    message = await model.ainvoke([HumanMessage(content="say ok")])

    assert isinstance(message, AIMessage)
    assert message.content == "OK"
    assert base.seen_messages == [[HumanMessage(content="say ok")]]


@pytest.mark.asyncio
async def test_streaming_ainvoke_model_delegates_bind_tools() -> None:
    base = FakeStreamingModel([AIMessageChunk(content="tool-ready")])
    bound = StreamingAinvokeChatModel(base).bind_tools([{"type": "function", "name": "ping"}])

    message = await bound.ainvoke([HumanMessage(content="use tool")])

    assert message.content == "tool-ready"
    assert bound._model.tools == [{"type": "function", "name": "ping"}]
