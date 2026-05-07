from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from model_eval.candidates import ModelCandidate
from voice_runtime.agent_providers import AGENT_PROVIDERS, AgentProvider
from voice_runtime.profiles import ReasoningEffort

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


EvalAdapterName = Literal["simulated", "live-mcp"]
_REASONING_EFFORTS = {None, "none", "minimal", "low", "medium", "high", "xhigh"}


@dataclass(frozen=True)
class EvalRunConfig:
    matrix_path: Path
    pack_name: str
    adapter: EvalAdapterName = "simulated"
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    samples: int = 1
    evidence_root: Path = Path("evidence/model_eval")

    def __post_init__(self) -> None:
        if self.samples < 1:
            raise ValueError("samples must be at least 1")


def load_model_matrix(path: Path) -> tuple[ModelCandidate, ...]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_candidates = data.get("candidates", [])
    if not raw_candidates:
        raise ValueError(f"{path} must define at least one candidate")
    if not isinstance(raw_candidates, list):
        raise ValueError(f"{path} candidates must be a list")

    candidates = tuple(_parse_candidate(raw) for raw in raw_candidates)
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{path} contains duplicate candidate labels")
    return candidates


def _parse_candidate(raw: Any) -> ModelCandidate:
    if not isinstance(raw, dict):
        raise ValueError("candidate must be a TOML table")

    missing = [
        key
        for key in ("label", "provider", "model", "api_key_env")
        if not raw.get(key)
    ]
    if missing:
        raise ValueError(f"candidate missing required fields: {', '.join(missing)}")

    provider = str(raw["provider"])
    if provider not in AGENT_PROVIDERS:
        raise ValueError(f"unsupported candidate provider: {provider}")

    reasoning_effort = raw.get("reasoning_effort")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise ValueError("candidate reasoning_effort must be a string when set")
    if reasoning_effort not in _REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning_effort: {reasoning_effort}")

    return ModelCandidate(
        label=str(raw["label"]),
        provider=cast(AgentProvider, provider),
        model=str(raw["model"]),
        reasoning_effort=cast(ReasoningEffort | None, reasoning_effort),
        api_key_env=str(raw["api_key_env"]),
    )
