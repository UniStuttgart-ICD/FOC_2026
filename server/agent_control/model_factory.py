from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from voice_runtime.profiles import AgentProfile

_GEMINI_25_BUDGET_BY_EFFORT = {
    "none": 0,
    "minimal": 0,
    "low": 512,
    "medium": 1024,
    "high": 4096,
    "xhigh": 8192,
}


def build_agent_chat_model(
    config: AgentProfile,
    *,
    env: Mapping[str, str] | None = None,
    chat_openai_cls: type[Any] | None = None,
    chat_google_cls: type[Any] | None = None,
    chat_anthropic_cls: type[Any] | None = None,
) -> Any:
    resolved_env = os.environ if env is None else env
    if config.provider == "openai_api":
        return _build_openai(config, resolved_env, chat_openai_cls)
    if config.provider == "gemini_api":
        return _build_gemini(config, resolved_env, chat_google_cls)
    if config.provider == "anthropic_api":
        return _build_anthropic(config, resolved_env, chat_anthropic_cls)
    raise ValueError(f"Unsupported native LangChain agent provider: {config.provider}")


def _required_key(config: AgentProfile, env: Mapping[str, str]) -> str:
    if config.api_key_env is None:
        raise ValueError(f"{config.provider} requires api_key_env")
    key = env.get(config.api_key_env)
    if not key:
        raise ValueError(f"{config.api_key_env} is required for {config.provider}")
    return key


def _build_openai(
    config: AgentProfile,
    env: Mapping[str, str],
    chat_openai_cls: type[Any] | None,
) -> Any:
    if chat_openai_cls is None:
        from langchain_openai import ChatOpenAI

        chat_openai_cls = ChatOpenAI
    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": _required_key(config, env),
        "use_responses_api": True,
    }
    if config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    return chat_openai_cls(**kwargs)


def _build_gemini(
    config: AgentProfile,
    env: Mapping[str, str],
    chat_google_cls: type[Any] | None,
) -> Any:
    if chat_google_cls is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat_google_cls = ChatGoogleGenerativeAI
    kwargs: dict[str, Any] = {
        "model": config.model,
        "google_api_key": _required_key(config, env),
    }
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    if config.model.startswith("gemini-3") and config.reasoning_effort is not None:
        if config.reasoning_effort in {"none", "xhigh"}:
            raise ValueError("Gemini 3 thinking_level supports minimal, low, medium, or high")
        kwargs["thinking_level"] = config.reasoning_effort
    elif config.model.startswith("gemini-2.5"):
        if config.thinking_budget is not None:
            kwargs["thinking_budget"] = config.thinking_budget
        elif config.reasoning_effort is not None:
            kwargs["thinking_budget"] = _GEMINI_25_BUDGET_BY_EFFORT[config.reasoning_effort]
    return chat_google_cls(**kwargs)


def _build_anthropic(
    config: AgentProfile,
    env: Mapping[str, str],
    chat_anthropic_cls: type[Any] | None,
) -> Any:
    if chat_anthropic_cls is None:
        from langchain_anthropic import ChatAnthropic

        chat_anthropic_cls = ChatAnthropic
    kwargs: dict[str, Any] = {
        "model_name": config.model,
        "api_key": _required_key(config, env),
    }
    if config.reasoning_effort in {"low", "medium", "high", "xhigh"}:
        kwargs["effort"] = config.reasoning_effort
    if config.temperature is not None:
        kwargs["temperature"] = config.temperature
    return chat_anthropic_cls(**kwargs)
