from __future__ import annotations

import os
from typing import Literal, cast

import pytest
from langchain_core.tools import tool

from agent_model_factory import build_agent_chat_model
from voice_runtime.profiles import AgentProfile, ReasoningEffort

pytestmark = [pytest.mark.live, pytest.mark.llm, pytest.mark.native_llm]
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


@tool
def report_robot_motion(
    robot_name: str,
    motion: Literal["up_down", "wave"],
    distance_m: float,
) -> str:
    """Report the exact robot motion the assistant intends to perform."""
    return f"{robot_name}:{motion}:{distance_m}"


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL") == "1"


def _reasoning_effort_from_env() -> ReasoningEffort:
    raw = os.getenv("LIVE_REASONING_EFFORT", "low")
    if raw not in _REASONING_EFFORTS:
        raise ValueError(f"Unsupported LIVE_REASONING_EFFORT: {raw}")
    return cast(ReasoningEffort, raw)


def _profile_from_env() -> AgentProfile:
    provider = os.getenv("LIVE_AGENT_PROVIDER", "openai_api")
    if provider == "gemini_api":
        return AgentProfile(
            provider="gemini_api",
            model=os.getenv("LIVE_GEMINI_MODEL", "gemini-2.5-flash"),
            reasoning_effort=_reasoning_effort_from_env(),
            thinking_budget=int(os.getenv("LIVE_GEMINI_THINKING_BUDGET", "1024")),
            api_key_env=os.getenv("LIVE_GEMINI_KEY_ENV", "GOOGLE_API_KEY"),
        )
    if provider == "anthropic_api":
        return AgentProfile(
            provider="anthropic_api",
            model=os.getenv("LIVE_ANTHROPIC_MODEL", "claude-sonnet-4-6-20250827"),
            reasoning_effort=_reasoning_effort_from_env(),
            api_key_env=os.getenv("LIVE_ANTHROPIC_KEY_ENV", "ANTHROPIC_API_KEY"),
        )
    return AgentProfile(
        provider="openai_api",
        model=os.getenv("LIVE_OPENAI_MODEL", "gpt-5.4-mini"),
        reasoning_effort=_reasoning_effort_from_env(),
        api_key_env=os.getenv("LIVE_OPENAI_KEY_ENV", "OPENAI_API_KEY"),
    )


@pytest.mark.skipif(not _live_enabled(), reason="set RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL=1")
@pytest.mark.parametrize("attempt", range(5))
def test_live_provider_emits_native_tool_call(attempt: int):
    model = build_agent_chat_model(_profile_from_env())
    bound = model.bind_tools([report_robot_motion])

    msg = bound.invoke(
        "Call report_robot_motion for robot UR10 doing an up_down motion over 0.05 meters. "
        "Do not answer in text; use the tool."
    )

    assert msg.tool_calls, f"attempt {attempt} produced no tool_calls: {msg!r}"
    call = msg.tool_calls[0]
    assert call["name"] == "report_robot_motion"
    assert call["args"]["robot_name"] == "UR10"
    assert call["args"]["motion"] == "up_down"
    assert 0.01 <= float(call["args"]["distance_m"]) <= 0.10
