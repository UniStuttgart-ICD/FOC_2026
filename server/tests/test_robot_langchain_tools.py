import json
from typing import Any

import pytest
from langchain_core.messages import ToolMessage

from robot_control.langchain_tools import execute_langchain_tool_calls


class FakeRobotExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return json.dumps({"structured_content": {"ok": True}})


@pytest.mark.asyncio
async def test_executes_langchain_tool_calls_and_returns_tool_messages() -> None:
    executor = FakeRobotExecutor()
    tool_calls = [
        {
            "id": "call-1",
            "name": "moveit_get_current_pose",
            "args": {"robot_name": "UR10"},
            "type": "tool_call",
        }
    ]

    messages = await execute_langchain_tool_calls(tool_calls, executor)

    assert executor.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert messages == [
        ToolMessage(
            content=json.dumps({"structured_content": {"ok": True}}),
            tool_call_id="call-1",
        )
    ]
