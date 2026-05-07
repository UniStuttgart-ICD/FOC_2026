from __future__ import annotations

from dataclasses import dataclass

from voice_runtime.agent_providers import AgentProvider
from voice_runtime.profiles import AgentProfile, ReasoningEffort


@dataclass(frozen=True)
class ModelCandidate:
    label: str
    provider: AgentProvider
    model: str
    reasoning_effort: ReasoningEffort | None
    api_key_env: str

    def to_agent_profile(self) -> AgentProfile:
        return AgentProfile(
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            api_key_env=self.api_key_env,
        )
