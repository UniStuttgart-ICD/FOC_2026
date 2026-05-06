from __future__ import annotations

from collections.abc import Callable

from pipecat.processors.frame_processor import FrameProcessor

from config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnProcessor


def create_agent_processor(
    config: AgentConfig,
    *,
    mcp_server_url: str,
    on_turn_started: Callable[[], None] | None = None,
    on_turn_finished: Callable[[], None] | None = None,
) -> FrameProcessor:
    if config.provider != "openai_codex_oauth":
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=OpenAICodexAgentProcessor(
            mcp_server_url=mcp_server_url,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        ),
        on_turn_started=on_turn_started,
        on_turn_finished=on_turn_finished,
    )
