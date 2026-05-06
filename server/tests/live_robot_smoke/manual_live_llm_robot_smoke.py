from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

RUN_ENV = "RUN_LIVE_LLM_ROBOT_SMOKE"
MCP_URL_ENV = "LIVE_LLM_ROBOT_MCP_URL"
MODEL_ENV = "LIVE_LLM_ROBOT_MODEL"
EVIDENCE_DIR_ENV = "LIVE_LLM_ROBOT_EVIDENCE_DIR"
DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_REASONING_EFFORT = "medium"

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.live,
    pytest.mark.llm,
    pytest.mark.robot_sim,
    pytest.mark.skipif(
        os.getenv(RUN_ENV) != "1",
        reason=f"manual live robot smoke tests require {RUN_ENV}=1, OPENAI_API_KEY, and MoveIt MCP",
    ),
]


@pytest_asyncio.fixture
async def live_agent() -> AsyncIterator[tuple[Any, Any]]:
    from agent_model_factory import build_agent_chat_model
    from langchain_agent_processor import LangChainAgentProcessor
    from robot_control.mcp_bridge import RobotMCPBridge
    from test_support.live_robot_smoke import RecordingRobotToolAdapter
    from voice_runtime.profiles import AgentProfile

    mcp_url = os.getenv(MCP_URL_ENV, DEFAULT_MCP_URL)
    model = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    reasoning_effort = os.getenv("LIVE_LLM_ROBOT_REASONING_EFFORT", DEFAULT_REASONING_EFFORT)
    recorder = RecordingRobotToolAdapter(RobotMCPBridge(mcp_url))
    processor = LangChainAgentProcessor(
        mcp_url,
        chat_model=build_agent_chat_model(
            AgentProfile(
                provider="openai_api",
                model=model,
                reasoning_effort=reasoning_effort,
                api_key_env="OPENAI_API_KEY",
            )
        ),
        model_label=model,
        tool_bridge=recorder,
    )
    try:
        yield processor, recorder
    finally:
        await processor.disconnect()


async def test_manual_live_llm_robot_smoke_suite(live_agent: tuple[Any, Any]) -> None:
    from test_support.live_robot_smoke import (
        DEFAULT_EVIDENCE_DIR,
        LiveSmokeRun,
        ValidationResult,
        run_agent_turn,
        validate_ambiguous_clarification,
        validate_bit_movement,
        validate_position_query,
        validate_up_down_motion,
        validate_wave_motion,
        write_evidence,
    )

    processor, recorder = live_agent
    evidence_dir = Path(os.getenv(EVIDENCE_DIR_ENV, str(DEFAULT_EVIDENCE_DIR)))
    cases: list[tuple[str, str, Callable[[LiveSmokeRun], ValidationResult]]] = [
        ("current-position", "what is the current position?", validate_position_query),
        ("move-up-bit", "move up a bit", lambda run: validate_bit_movement(run, direction="up")),
        ("move-down-bit", "move down a bit", lambda run: validate_bit_movement(run, direction="down")),
        ("up-down-motion", "Have the robot move up and down", validate_up_down_motion),
        ("visible-wave", "wave to me", validate_wave_motion),
        ("ambiguous-move-there", "move there", validate_ambiguous_clarification),
    ]

    failures: list[str] = []
    for case_name, prompt, validator in cases:
        try:
            run = await run_agent_turn(processor, recorder, prompt)
            validation = validator(run)
        except Exception as exc:
            run = LiveSmokeRun(prompt=prompt, reply="", tool_calls=recorder.calls)
            validation = ValidationResult(False, f"case raised {type(exc).__name__}: {exc}")
        evidence_path = write_evidence(
            evidence_dir=evidence_dir,
            case_name=case_name,
            run=run,
            validation=validation,
        )
        if not validation.passed:
            failures.append(f"{case_name}: {validation.reason}; evidence={evidence_path}")

    assert not failures, "\n".join(failures)
