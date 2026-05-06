from typing import Any

import pytest

from agent_model_factory import build_agent_chat_model
from voice_runtime.profiles import AgentProfile


class CapturedChatModel:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


def test_builds_chat_openai_with_reasoning_effort_and_key():
    model = build_agent_chat_model(
        AgentProfile(
            provider="openai_api",
            model="gpt-5.5",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        env={"OPENAI_API_KEY": "sk-test"},
        chat_openai_cls=CapturedChatModel,
    )

    assert isinstance(model, CapturedChatModel)
    assert model.kwargs["model"] == "gpt-5.5"
    assert model.kwargs["api_key"] == "sk-test"
    assert model.kwargs["reasoning_effort"] == "low"


def test_builds_chat_google_with_thinking_budget_and_key():
    model = build_agent_chat_model(
        AgentProfile(
            provider="gemini_api",
            model="gemini-2.5-flash",
            reasoning_effort="medium",
            api_key_env="GEMINI_API_KEY",
            thinking_budget=1024,
        ),
        env={"GEMINI_API_KEY": "gem-test"},
        chat_google_cls=CapturedChatModel,
    )

    assert isinstance(model, CapturedChatModel)
    assert model.kwargs["model"] == "gemini-2.5-flash"
    assert model.kwargs["google_api_key"] == "gem-test"
    assert model.kwargs["thinking_budget"] == 1024


def test_builds_chat_anthropic_with_effort_and_key():
    model = build_agent_chat_model(
        AgentProfile(
            provider="anthropic_api",
            model="claude-sonnet-4-6-20250827",
            reasoning_effort="medium",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        env={"ANTHROPIC_API_KEY": "anth-test"},
        chat_anthropic_cls=CapturedChatModel,
    )

    assert isinstance(model, CapturedChatModel)
    assert model.kwargs["model_name"] == "claude-sonnet-4-6-20250827"
    assert model.kwargs["api_key"] == "anth-test"
    assert model.kwargs["effort"] == "medium"


def test_missing_provider_key_raises_clear_error():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        build_agent_chat_model(
            AgentProfile(
                provider="openai_api",
                model="gpt-5.4-mini",
                api_key_env="OPENAI_API_KEY",
            ),
            env={},
            chat_openai_cls=CapturedChatModel,
        )


def test_builds_gemini_3_with_thinking_level():
    model = build_agent_chat_model(
        AgentProfile(
            provider="gemini_api",
            model="gemini-3.1-pro-preview",
            reasoning_effort="low",
            api_key_env="GOOGLE_API_KEY",
        ),
        env={"GOOGLE_API_KEY": "gem-test"},
        chat_google_cls=CapturedChatModel,
    )

    assert model.kwargs["thinking_level"] == "low"
    assert "thinking_budget" not in model.kwargs
