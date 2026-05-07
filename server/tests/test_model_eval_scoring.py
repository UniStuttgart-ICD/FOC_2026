from model_eval.results import AttemptResult
from model_eval.scoring import rank_candidates, summarize_candidate


def _attempt(
    candidate: str,
    scenario: str,
    *,
    passed: bool,
    elapsed_s: float,
    tool_call_count: int = 1,
) -> AttemptResult:
    return AttemptResult(
        candidate_label=candidate,
        scenario_name=scenario,
        attempt_index=0,
        prompt="prompt",
        elapsed_s=elapsed_s,
        passed=passed,
        reason="ok" if passed else "failed",
        details={},
        assistant_reply="done",
        tool_calls=[],
        tool_call_count=tool_call_count,
        model_turn_count=1,
        exception=None,
    )


def test_summarize_candidate_requires_all_attempts_to_pass() -> None:
    summary = summarize_candidate(
        "fast-failing",
        (
            _attempt("fast-failing", "current-position", passed=True, elapsed_s=0.5),
            _attempt("fast-failing", "visible-wave", passed=False, elapsed_s=0.4),
        ),
    )

    assert summary.pass_count == 1
    assert summary.total_count == 2
    assert summary.correctness_passed is False
    assert summary.median_latency_s == 0.5
    assert summary.failure_reasons == ("failed",)


def test_rank_candidates_correctness_before_latency() -> None:
    attempts = (
        _attempt("fast-failing", "visible-wave", passed=False, elapsed_s=0.2),
        _attempt("slow-correct", "visible-wave", passed=True, elapsed_s=4.0),
        _attempt("fast-correct", "visible-wave", passed=True, elapsed_s=1.0),
    )

    ranked = rank_candidates(attempts)

    assert [summary.candidate_label for summary in ranked] == [
        "fast-correct",
        "slow-correct",
        "fast-failing",
    ]
    assert ranked[0].recommended is True
    assert ranked[1].recommended is False
    assert ranked[2].recommended is False


def test_rank_candidates_partial_correctness_before_partial_latency() -> None:
    attempts = (
        _attempt("mostly-correct", "scenario-1", passed=True, elapsed_s=3.0),
        _attempt("mostly-correct", "scenario-2", passed=True, elapsed_s=3.0),
        _attempt("mostly-correct", "scenario-3", passed=True, elapsed_s=3.0),
        _attempt("mostly-correct", "scenario-4", passed=False, elapsed_s=3.0),
        _attempt("fast-weak", "scenario-1", passed=True, elapsed_s=0.1),
        _attempt("fast-weak", "scenario-2", passed=False, elapsed_s=0.1),
        _attempt("fast-weak", "scenario-3", passed=False, elapsed_s=0.1),
        _attempt("fast-weak", "scenario-4", passed=False, elapsed_s=0.1),
    )

    ranked = rank_candidates(attempts)

    assert [summary.candidate_label for summary in ranked] == [
        "mostly-correct",
        "fast-weak",
    ]


def test_rank_candidates_uses_tool_calls_then_label_as_tiebreakers() -> None:
    attempts = (
        _attempt("b-candidate", "visible-wave", passed=True, elapsed_s=1.0, tool_call_count=3),
        _attempt("a-candidate", "visible-wave", passed=True, elapsed_s=1.0, tool_call_count=3),
        _attempt("lower-tool-count", "visible-wave", passed=True, elapsed_s=1.0, tool_call_count=2),
    )

    ranked = rank_candidates(attempts)

    assert [summary.candidate_label for summary in ranked] == [
        "lower-tool-count",
        "a-candidate",
        "b-candidate",
    ]
    assert ranked[0].recommended is True


def test_rank_candidates_marks_none_recommended_when_all_fail() -> None:
    ranked = rank_candidates(
        (
            _attempt("a-candidate", "visible-wave", passed=False, elapsed_s=1.0),
            _attempt("b-candidate", "visible-wave", passed=False, elapsed_s=0.5),
        )
    )

    assert [summary.candidate_label for summary in ranked] == [
        "a-candidate",
        "b-candidate",
    ]
    assert all(summary.recommended is False for summary in ranked)
