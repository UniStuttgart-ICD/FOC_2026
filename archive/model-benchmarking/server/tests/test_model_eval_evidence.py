import json
from datetime import UTC, datetime
from pathlib import Path

from model_eval.evidence import EvidenceWriter
from model_eval.results import AttemptResult, CandidateSummary


def test_evidence_writer_writes_readable_run_artifacts(tmp_path: Path) -> None:
    attempt = AttemptResult(
        candidate_label="gpt-test",
        scenario_name="current-position",
        attempt_index=0,
        prompt="what is the current position?",
        elapsed_s=0.25,
        passed=True,
        reason="ok",
        details={"pose": {"z": 0.4}, "checks": ["observed"]},
        assistant_reply="The robot is at x=0.1, y=0.2, z=0.4.",
        tool_calls=[
            {
                "name": "get_pose",
                "args": {"frame": "base_link"},
                "result": {"z": 0.4},
            }
        ],
        tool_call_count=1,
        model_turn_count=1,
        exception=None,
    )
    summary = CandidateSummary(
        candidate_label="gpt-test",
        pass_count=1,
        total_count=1,
        correctness_passed=True,
        median_latency_s=0.25,
        average_tool_call_count=1.0,
        failure_reasons=(),
        recommended=True,
    )

    evidence_dir = EvidenceWriter(tmp_path / "model_eval").write(
        attempts=(attempt,),
        summaries=(summary,),
        metadata={"pack": "core_robot_commands", "samples": 1},
    )

    assert evidence_dir.parent == tmp_path / "model_eval"
    assert evidence_dir.is_dir()
    assert (evidence_dir / "metadata.json").read_text(encoding="utf-8").startswith(
        "{\n  "
    )

    metadata = json.loads((evidence_dir / "metadata.json").read_text(encoding="utf-8"))
    created_at = datetime.fromisoformat(metadata["created_at"].replace("Z", "+00:00"))
    assert created_at.tzinfo == UTC
    assert metadata["pack"] == "core_robot_commands"
    assert metadata["samples"] == 1

    attempts = json.loads((evidence_dir / "attempts.json").read_text(encoding="utf-8"))
    assert attempts == [
        {
            "assistant_reply": "The robot is at x=0.1, y=0.2, z=0.4.",
            "attempt_index": 0,
            "candidate_label": "gpt-test",
            "details": {"checks": ["observed"], "pose": {"z": 0.4}},
            "elapsed_s": 0.25,
            "exception": None,
            "model_turn_count": 1,
            "passed": True,
            "prompt": "what is the current position?",
            "reason": "ok",
            "scenario_name": "current-position",
            "tool_call_count": 1,
            "tool_calls": [
                {
                    "args": {"frame": "base_link"},
                    "name": "get_pose",
                    "result": {"z": 0.4},
                }
            ],
        }
    ]

    summaries = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    assert summaries == [
        {
            "average_tool_call_count": 1.0,
            "candidate_label": "gpt-test",
            "correctness_passed": True,
            "failure_reasons": [],
            "median_latency_s": 0.25,
            "pass_count": 1,
            "recommended": True,
            "total_count": 1,
        }
    ]


def test_evidence_run_appends_attempts_before_final_summary(tmp_path: Path) -> None:
    attempt = AttemptResult(
        candidate_label="gpt-test",
        scenario_name="current-position",
        attempt_index=0,
        prompt="what is the current position?",
        elapsed_s=0.25,
        passed=True,
        reason="ok",
        details={},
        assistant_reply="Observed.",
        tool_calls=[],
        tool_call_count=0,
        model_turn_count=1,
        exception=None,
    )
    summary = CandidateSummary(
        candidate_label="gpt-test",
        pass_count=1,
        total_count=1,
        correctness_passed=True,
        median_latency_s=0.25,
        average_tool_call_count=0.0,
        failure_reasons=(),
        recommended=True,
    )

    evidence_run = EvidenceWriter(tmp_path / "model_eval").start(
        metadata={"pack": "core_robot_commands", "samples": 1},
    )
    evidence_run.append_attempt(attempt)

    attempts_jsonl = evidence_run.evidence_dir / "attempts.jsonl"
    assert attempts_jsonl.is_file()
    assert not (evidence_run.evidence_dir / "summary.json").exists()
    assert [json.loads(line) for line in attempts_jsonl.read_text(encoding="utf-8").splitlines()] == [
        {
            "assistant_reply": "Observed.",
            "attempt_index": 0,
            "candidate_label": "gpt-test",
            "details": {},
            "elapsed_s": 0.25,
            "exception": None,
            "model_turn_count": 1,
            "passed": True,
            "prompt": "what is the current position?",
            "reason": "ok",
            "scenario_name": "current-position",
            "tool_call_count": 0,
            "tool_calls": [],
        }
    ]

    evidence_run.finalize(attempts=(attempt,), summaries=(summary,))

    assert (evidence_run.evidence_dir / "attempts.json").is_file()
    assert (evidence_run.evidence_dir / "summary.json").is_file()
