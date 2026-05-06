from pipecat.processors.frame_processor import FrameProcessor

from agent_processor_factory import create_agent_processor
from config import AgentConfig
from langchain_agent_processor import LangChainAgentProcessor
from voice_runtime.agent_turn import AgentTurnProcessor


class FakeChatModel:
    pass


def test_creates_openai_api_agent_turn_processor(monkeypatch):
    monkeypatch.setattr(
        "agent_processor_factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
        raising=False,
    )

    processor = create_agent_processor(
        AgentConfig(
            provider="openai_api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor._backend, LangChainAgentProcessor)


def test_creates_gemini_api_agent_turn_processor(monkeypatch):
    monkeypatch.setattr(
        "agent_processor_factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
        raising=False,
    )

    processor = create_agent_processor(
        AgentConfig(
            provider="gemini_api",
            model="gemini-2.5-flash",
            api_key_env="GOOGLE_API_KEY",
            thinking_budget=1024,
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor._backend, LangChainAgentProcessor)


def test_creates_anthropic_api_agent_turn_processor(monkeypatch):
    monkeypatch.setattr(
        "agent_processor_factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
        raising=False,
    )

    processor = create_agent_processor(
        AgentConfig(
            provider="anthropic_api",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            reasoning_effort="medium",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor._backend, LangChainAgentProcessor)
