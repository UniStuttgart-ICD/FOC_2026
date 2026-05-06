import os

import pytest
from langchain_codex_oauth import ChatCodexOAuth
from langchain_core.messages import HumanMessage, SystemMessage

from codex_langchain_auth import PiLangChainCodexAuthStore
from codex_streaming_model import StreamingAinvokeChatModel
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput

pytestmark = [
    pytest.mark.live,
    pytest.mark.llm,
    pytest.mark.skipif(
        os.getenv("RUN_LIVE_CODEX_OAUTH") != "1",
        reason="set RUN_LIVE_CODEX_OAUTH=1 to call the real Codex OAuth backend",
    ),
]


@pytest.mark.asyncio
async def test_pi_oauth_credentials_can_call_real_codex_backend() -> None:
    auth_store = PiLangChainCodexAuthStore()
    credentials = auth_store.load()
    assert credentials.access
    assert credentials.refresh
    assert credentials.account_id

    model = StreamingAinvokeChatModel(
        ChatCodexOAuth(
            model=os.getenv("LIVE_CODEX_MODEL", "gpt-5.4-mini"),
            auth_store=auth_store,
            reasoning_effort=os.getenv("LIVE_CODEX_REASONING_EFFORT", "medium"),
            text_verbosity="low",
            system_prompt_mode="strict",
            timeout=60,
            max_retries=0,
        )
    )

    response = await model.ainvoke(
        [
            SystemMessage(content="You are a connectivity smoke test. Follow the user exactly."),
            HumanMessage(content="Reply with exactly: OAUTH_OK"),
        ]
    )

    text = str(response.content or "").strip()
    assert "OAUTH_OK" in text


class LiveNoopBridge:
    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    def function_tools(self) -> list[dict]:
        return []


@pytest.mark.asyncio
async def test_openai_codex_agent_processor_can_call_real_codex_oauth_model() -> None:
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model=os.getenv("LIVE_CODEX_MODEL", "gpt-5.4-mini"),
        reasoning_effort=os.getenv("LIVE_CODEX_REASONING_EFFORT", "medium"),
        tool_bridge=LiveNoopBridge(),
    )
    turn = AgentTurnInput(
        user_text="Connectivity test. Reply with exactly: AGENT_OK",
        messages=[{"role": "user", "content": "Connectivity test. Reply with exactly: AGENT_OK"}],
    )

    chunks = [chunk async for chunk in processor.run_turn(turn)]

    assert "AGENT_OK" in "".join(chunks)
