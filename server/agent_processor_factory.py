from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor

from claude_agent_processor import ClaudeAgentProcessor
from config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor


def create_agent_processor(config: AgentConfig, *, mcp_server_url: str) -> FrameProcessor:
    if config.provider == "claude":
        return ClaudeAgentProcessor(mcp_server_url=mcp_server_url, model=config.model)
    if config.provider == "openai_codex_oauth":
        return OpenAICodexAgentProcessor(mcp_server_url=mcp_server_url, model=config.model)
    raise ValueError(f"Unsupported agent provider: {config.provider}")
