from __future__ import annotations

from collections.abc import Callable

from pipecat.processors.frame_processor import FrameProcessor

from agent_model_factory import build_agent_chat_model
from config import AgentConfig
from langchain_agent_processor import LangChainAgentProcessor
from voice_runtime.agent_providers import NATIVE_LANGCHAIN_AGENT_PROVIDERS
from voice_runtime.agent_turn import AgentTurnProcessor


def create_agent_processor(
    config: AgentConfig,
    *,
    mcp_server_url: str,
    on_turn_started: Callable[[], None] | None = None,
    on_turn_finished: Callable[[], None] | None = None,
) -> FrameProcessor:
    if config.provider in NATIVE_LANGCHAIN_AGENT_PROVIDERS:
        backend = LangChainAgentProcessor(
            mcp_server_url,
            chat_model=build_agent_chat_model(config),
            model_label=config.model,
        )
    else:
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=backend,
        on_turn_started=on_turn_started,
        on_turn_finished=on_turn_finished,
    )
