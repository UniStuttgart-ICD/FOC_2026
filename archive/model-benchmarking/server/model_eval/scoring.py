from __future__ import annotations

from collections import defaultdict
from statistics import median

from model_eval.results import AttemptResult, CandidateSummary


def summarize_candidate(
    candidate_label: str,
    attempts: tuple[AttemptResult, ...],
    *,
    recommended: bool = False,
) -> CandidateSummary:
    pass_count = sum(1 for attempt in attempts if attempt.passed)
    total_count = len(attempts)
    passed_latencies = [attempt.elapsed_s for attempt in attempts if attempt.passed]
    tool_counts = [attempt.tool_call_count for attempt in attempts]
    failure_reasons = tuple(attempt.reason for attempt in attempts if not attempt.passed)

    return CandidateSummary(
        candidate_label=candidate_label,
        pass_count=pass_count,
        total_count=total_count,
        correctness_passed=total_count > 0 and pass_count == total_count,
        median_latency_s=median(passed_latencies) if passed_latencies else None,
        average_tool_call_count=sum(tool_counts) / len(tool_counts) if tool_counts else 0.0,
        failure_reasons=failure_reasons,
        recommended=recommended,
    )


def rank_candidates(attempts: tuple[AttemptResult, ...]) -> tuple[CandidateSummary, ...]:
    grouped: dict[str, list[AttemptResult]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.candidate_label].append(attempt)

    ranked = sorted(
        (
            summarize_candidate(label, tuple(candidate_attempts))
            for label, candidate_attempts in grouped.items()
        ),
        key=lambda summary: (
            not summary.correctness_passed,
            -(summary.pass_count / summary.total_count if summary.total_count else 0.0),
            -summary.pass_count,
            summary.median_latency_s if summary.median_latency_s is not None else float("inf"),
            summary.average_tool_call_count,
            summary.candidate_label,
        ),
    )
    first_passing_label = next(
        (
            summary.candidate_label
            for summary in ranked
            if summary.correctness_passed
        ),
        None,
    )
    if first_passing_label is None:
        return tuple(ranked)

    return tuple(
        CandidateSummary(
            candidate_label=summary.candidate_label,
            pass_count=summary.pass_count,
            total_count=summary.total_count,
            correctness_passed=summary.correctness_passed,
            median_latency_s=summary.median_latency_s,
            average_tool_call_count=summary.average_tool_call_count,
            failure_reasons=summary.failure_reasons,
            recommended=summary.candidate_label == first_passing_label,
        )
        for summary in ranked
    )
