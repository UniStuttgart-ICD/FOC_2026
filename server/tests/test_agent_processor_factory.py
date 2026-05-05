from pipecat.processors.frame_processor import FrameProcessor

from agent_processor_factory import create_agent_processor
from config import AgentConfig
from voice_runtime.agent_turn import AgentTurnProcessor


def test_creates_openai_codex_agent_turn_processor():
    processor = create_agent_processor(
        AgentConfig(provider="openai_codex_oauth", model="gpt-5.5"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor, FrameProcessor)


def test_passes_reasoning_effort_to_openai_codex_backend():
    processor = create_agent_processor(
        AgentConfig(provider="openai_codex_oauth", model="gpt-5.5", reasoning_effort="medium"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert processor._backend._reasoning_effort == "medium"
