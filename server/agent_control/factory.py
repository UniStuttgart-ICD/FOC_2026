from __future__ import annotations

from collections.abc import Callable

from pipecat.processors.frame_processor import FrameProcessor

from agent_control.langchain_agent_processor import LangChainAgentProcessor
from agent_control.model_factory import build_agent_chat_model
from process_trace import NoopProcessTracer, ProcessTracer
from voice_runtime.agent_providers import NATIVE_LANGCHAIN_AGENT_PROVIDERS
from voice_runtime.agent_turn import AgentTurnProcessor, ResponseCoordinator
from voice_runtime.profiles import AgentProfile

ProcessTracerLike = ProcessTracer | NoopProcessTracer


def create_agent_processor(
    config: AgentProfile,
    *,
    mcp_server_url: str,
    mcp_vizor_url: str | None = None,
    user_sensing_max_age_s: float = 2.0,
    verified_execution_url: str | None = None,
    tracer: ProcessTracerLike | None = None,
    on_turn_started: Callable[[], None] | None = None,
    on_turn_finished: Callable[[], None] | None = None,
    response_coordinator: ResponseCoordinator | None = None,
) -> FrameProcessor:
    if config.provider in NATIVE_LANGCHAIN_AGENT_PROVIDERS:
        backend = LangChainAgentProcessor(
            mcp_server_url,
            chat_model=build_agent_chat_model(config),
            model_label=config.model,
            mcp_vizor_url=mcp_vizor_url,
            user_sensing_max_age_s=user_sensing_max_age_s,
            verified_execution_url=verified_execution_url,
            tracer=tracer,
        )
    else:
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=backend,
        tracer=tracer,
        on_turn_started=on_turn_started,
        on_turn_finished=on_turn_finished,
        response_coordinator=response_coordinator,
    )
