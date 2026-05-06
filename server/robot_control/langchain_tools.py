from __future__ import annotations

from typing import Any, Protocol

from langchain_core.messages import ToolMessage


class RobotToolExecutor(Protocol):
    async def execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str: ...


async def execute_langchain_tool_calls(
    tool_calls: list[dict[str, Any]], executor: RobotToolExecutor
) -> list[ToolMessage]:
    messages: list[ToolMessage] = []
    for call in tool_calls:
        call_id = str(call.get("id") or "")
        name = str(call.get("name") or "")
        args = call.get("args")
        arguments = args if isinstance(args, dict) else {}
        output = await executor.execute_tool_call(name, arguments)
        messages.append(ToolMessage(content=output, tool_call_id=call_id))
    return messages
