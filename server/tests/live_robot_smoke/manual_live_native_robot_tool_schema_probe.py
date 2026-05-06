from __future__ import annotations

import os
from typing import cast

import pytest

from agent_model_factory import build_agent_chat_model
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.profiles import AgentProfile, ReasoningEffort

pytestmark = [pytest.mark.live, pytest.mark.llm, pytest.mark.native_llm, pytest.mark.robot_sim]
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


def _reasoning_effort_from_env() -> ReasoningEffort:
    raw = os.getenv("LIVE_REASONING_EFFORT", "low")
    if raw not in _REASONING_EFFORTS:
        raise ValueError(f"Unsupported LIVE_REASONING_EFFORT: {raw}")
    return cast(ReasoningEffort, raw)


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_NATIVE_ROBOT_SCHEMA_PROBE") != "1",
    reason="set RUN_LIVE_NATIVE_ROBOT_SCHEMA_PROBE=1",
)
@pytest.mark.asyncio
async def test_live_model_fills_real_moveit_tool_arguments_without_execution():
    bridge = RobotMCPBridge(os.getenv("LIVE_ROBOT_MCP_URL", "http://127.0.0.1:8765/mcp"))
    await bridge.connect()
    try:
        tools = bridge.function_tools()
        motion_tool = next(t for t in tools if t["name"] == "moveit_plan_and_execute_free_motion")
        model = build_agent_chat_model(
            AgentProfile(
                provider="openai_api",
                model=os.getenv("LIVE_OPENAI_MODEL", "gpt-5.5"),
                reasoning_effort=_reasoning_effort_from_env(),
                api_key_env="OPENAI_API_KEY",
            )
        )
        bound = model.bind_tools([motion_tool])
        msg = await bound.ainvoke(
            "For robot UR10, prepare a moveit_plan_and_execute_free_motion call that moves "
            "the end effector up by about 5 cm from the current pose. Only call the tool."
        )
    finally:
        await bridge.disconnect()

    assert msg.tool_calls
    call = msg.tool_calls[0]
    assert call["name"] == "moveit_plan_and_execute_free_motion"
    assert call["args"]["robot_name"] == "UR10"
    assert "target_pose" in call["args"]
