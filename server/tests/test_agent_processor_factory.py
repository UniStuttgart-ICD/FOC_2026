from pipecat.processors.frame_processor import FrameProcessor

from agent_control.factory import create_agent_processor
from agent_control.langchain_agent_processor import LangChainAgentProcessor
from config import AgentConfig
from process_trace import MemoryTraceWriter, ProcessTracer
from voice_runtime.agent_turn import AgentTurnProcessor


class FakeChatModel:
    pass


class FakeLangChainAgentProcessor(FrameProcessor):
    def __init__(self, mcp_server_url, *, chat_model, model_label, tracer):
        super().__init__()
        self.mcp_server_url = mcp_server_url
        self.chat_model = chat_model
        self.model_label = model_label
        self.tracer = tracer


class FakeAgentTurnProcessor(FrameProcessor):
    def __init__(self, *, backend, tracer, on_turn_started=None, on_turn_finished=None):
        super().__init__()
        self._backend = backend
        self.tracer = tracer
        self.on_turn_started = on_turn_started
        self.on_turn_finished = on_turn_finished


def _patch_factory_dependencies(monkeypatch):
    monkeypatch.setattr(
        "agent_control.factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
        raising=False,
    )
    monkeypatch.setattr(
        "agent_control.factory.LangChainAgentProcessor",
        FakeLangChainAgentProcessor,
    )
    monkeypatch.setattr(
        "agent_control.factory.AgentTurnProcessor",
        FakeAgentTurnProcessor,
    )


def test_creates_openai_api_agent_turn_processor(monkeypatch):
    _patch_factory_dependencies(monkeypatch)

    processor = create_agent_processor(
        AgentConfig(
            provider="openai_api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, FakeAgentTurnProcessor)
    assert isinstance(processor._backend, FakeLangChainAgentProcessor)


def test_creates_gemini_api_agent_turn_processor(monkeypatch):
    _patch_factory_dependencies(monkeypatch)

    processor = create_agent_processor(
        AgentConfig(
            provider="gemini_api",
            model="gemini-2.5-flash",
            api_key_env="GOOGLE_API_KEY",
            thinking_budget=1024,
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, FakeAgentTurnProcessor)
    assert isinstance(processor._backend, FakeLangChainAgentProcessor)


def test_creates_anthropic_api_agent_turn_processor(monkeypatch):
    _patch_factory_dependencies(monkeypatch)

    processor = create_agent_processor(
        AgentConfig(
            provider="anthropic_api",
            model="claude-sonnet-4-6",
            api_key_env="ANTHROPIC_API_KEY",
            reasoning_effort="medium",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, FakeAgentTurnProcessor)
    assert isinstance(processor._backend, FakeLangChainAgentProcessor)


def test_passes_tracer_to_backend_and_agent_turn_processor(monkeypatch):
    tracer = ProcessTracer(MemoryTraceWriter())
    _patch_factory_dependencies(monkeypatch)

    processor = create_agent_processor(
        AgentConfig(
            provider="openai_api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
        tracer=tracer,
    )

    assert isinstance(processor, FakeAgentTurnProcessor)
    assert processor.tracer is tracer
    assert isinstance(processor._backend, FakeLangChainAgentProcessor)
    assert processor._backend.tracer is tracer


def test_factory_real_processors_accept_tracer(monkeypatch):
    tracer = ProcessTracer(MemoryTraceWriter())
    monkeypatch.setattr(
        "agent_control.factory.build_agent_chat_model",
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
        tracer=tracer,
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert processor._tracer is tracer
    assert isinstance(processor._backend, LangChainAgentProcessor)
    assert processor._backend._tracer is tracer
