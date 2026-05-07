from __future__ import annotations

from typing import Any, Protocol

from model_eval.config import EvalAdapterName
from model_eval.simulated_moveit import SimulatedMoveItAdapter
from robot_control.mcp_bridge import RobotMCPBridge

DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"


class EvalToolAdapter(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def function_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


def create_eval_tool_adapter(
    adapter: EvalAdapterName,
    mcp_url: str | None = None,
) -> EvalToolAdapter:
    if adapter == "simulated":
        return SimulatedMoveItAdapter()
    if adapter == "live-mcp":
        return RobotMCPBridge(mcp_url or DEFAULT_MCP_URL)
    raise ValueError(f"unknown model eval adapter: {adapter}")
