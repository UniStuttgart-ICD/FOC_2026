from __future__ import annotations

from typing import get_args

from model_eval.adapters import EvalAdapterName, create_eval_tool_adapter
from model_eval.simulated_moveit import SimulatedMoveItAdapter
from robot_control.mcp_bridge import RobotMCPBridge


def test_eval_adapter_name_supports_simulated_and_live_mcp() -> None:
    assert set(get_args(EvalAdapterName)) == {"simulated", "live-mcp"}


def test_create_eval_tool_adapter_returns_simulated_adapter() -> None:
    adapter = create_eval_tool_adapter("simulated")

    assert isinstance(adapter, SimulatedMoveItAdapter)


def test_create_eval_tool_adapter_returns_live_mcp_bridge_with_default_url() -> None:
    adapter = create_eval_tool_adapter("live-mcp")

    assert isinstance(adapter, RobotMCPBridge)
    assert adapter._mcp_server_url == "http://127.0.0.1:8765/mcp"


def test_create_eval_tool_adapter_accepts_live_mcp_url() -> None:
    adapter = create_eval_tool_adapter("live-mcp", mcp_url="http://example.test/mcp")

    assert isinstance(adapter, RobotMCPBridge)
    assert adapter._mcp_server_url == "http://example.test/mcp"
