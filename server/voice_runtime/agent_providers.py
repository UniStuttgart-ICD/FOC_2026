from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AgentProvider = Literal["openai_api", "gemini_api", "anthropic_api"]


@dataclass(frozen=True)
class AgentProviderSpec:
    default_api_key_env: str
    native_langchain: bool = True


AGENT_PROVIDER_SPECS: dict[AgentProvider, AgentProviderSpec] = {
    "openai_api": AgentProviderSpec(default_api_key_env="OPENAI_API_KEY"),
    "gemini_api": AgentProviderSpec(default_api_key_env="GOOGLE_API_KEY"),
    "anthropic_api": AgentProviderSpec(default_api_key_env="ANTHROPIC_API_KEY"),
}

AGENT_PROVIDERS: frozenset[AgentProvider] = frozenset(AGENT_PROVIDER_SPECS)
NATIVE_LANGCHAIN_AGENT_PROVIDERS: frozenset[AgentProvider] = frozenset(
    provider for provider, spec in AGENT_PROVIDER_SPECS.items() if spec.native_langchain
)


def default_agent_key_env(provider: AgentProvider) -> str:
    return AGENT_PROVIDER_SPECS[provider].default_api_key_env
