from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor

from config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnProcessor


def create_agent_processor(config: AgentConfig, *, mcp_server_url: str) -> FrameProcessor:
    if config.provider != "openai_codex_oauth":
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=OpenAICodexAgentProcessor(
            mcp_server_url=mcp_server_url,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
    )
