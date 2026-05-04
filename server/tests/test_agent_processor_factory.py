from agent_processor_factory import create_agent_processor
from claude_agent_processor import ClaudeAgentProcessor
from config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor


def test_creates_claude_processor():
    processor = create_agent_processor(
        AgentConfig(provider="claude", model="claude-haiku-4-5-20251001"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, ClaudeAgentProcessor)


def test_creates_openai_codex_processor():
    processor = create_agent_processor(
        AgentConfig(provider="openai_codex_oauth", model="gpt-5.5"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, OpenAICodexAgentProcessor)
