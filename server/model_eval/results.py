from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AttemptResult:
    candidate_label: str
    scenario_name: str
    attempt_index: int
    prompt: str
    elapsed_s: float
    passed: bool
    reason: str
    details: dict[str, Any]
    assistant_reply: str
    tool_calls: list[dict[str, Any]]
    tool_call_count: int
    model_turn_count: int
    exception: str | None


@dataclass(frozen=True)
class CandidateSummary:
    candidate_label: str
    pass_count: int
    total_count: int
    correctness_passed: bool
    median_latency_s: float | None
    average_tool_call_count: float
    failure_reasons: tuple[str, ...]
    recommended: bool = False
