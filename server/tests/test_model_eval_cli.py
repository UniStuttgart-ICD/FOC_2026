from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from model_eval.__main__ import async_main, build_parser
from model_eval.config import EvalRunConfig


def test_run_parser_defaults_to_simulated_adapter(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(["run", "--matrix", str(tmp_path / "matrix.toml")])

    config = EvalRunConfig(
        matrix_path=args.matrix,
        pack_name=args.pack,
        adapter=args.adapter,
        mcp_url=args.mcp_url,
        samples=args.samples,
        evidence_root=args.evidence_root,
    )

    assert config.adapter == "simulated"
    assert config.pack_name == "core_robot_commands"
    assert config.mcp_url == "http://127.0.0.1:8765/mcp"
    assert config.samples == 1
    assert config.evidence_root == Path("evidence/model_eval")


def test_run_parser_accepts_live_mcp_and_repeated_scenarios(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "run",
            "--matrix",
            str(tmp_path / "matrix.toml"),
            "--adapter",
            "live-mcp",
            "--mcp-url",
            "http://127.0.0.1:8765/mcp",
            "--samples",
            "3",
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--scenario",
            "current-position",
            "--scenario",
            "move-up-bit",
        ]
    )

    assert args.adapter == "live-mcp"
    assert args.mcp_url == "http://127.0.0.1:8765/mcp"
    assert args.samples == 3
    assert args.evidence_root == tmp_path / "evidence"
    assert args.scenarios == ["current-position", "move-up-bit"]


@pytest.mark.asyncio
async def test_async_main_prints_evidence_and_candidate_summaries(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeSummary:
        candidate_label: str
        pass_count: int
        total_count: int
        median_latency_s: float | None
        correctness_passed: bool
        recommended: bool

    @dataclass(frozen=True)
    class FakeResult:
        evidence_dir: Path
        summaries: tuple[FakeSummary, ...]

    calls: list[tuple[EvalRunConfig, tuple[str, ...] | None]] = []

    async def fake_run_eval_suite(
        config: EvalRunConfig,
        *,
        scenario_names: tuple[str, ...] | None = None,
    ) -> FakeResult:
        calls.append((config, scenario_names))
        return FakeResult(
            evidence_dir=tmp_path / "evidence" / "run",
            summaries=(
                FakeSummary("alpha", 2, 2, 0.125, True, True),
                FakeSummary("beta", 1, 2, None, False, False),
            ),
        )

    monkeypatch.setattr("model_eval.__main__.run_eval_suite", fake_run_eval_suite)

    exit_code = await async_main(
        [
            "run",
            "--matrix",
            str(tmp_path / "matrix.toml"),
            "--scenario",
            "current-position",
        ]
    )

    assert exit_code == 0
    assert calls == [
        (
            EvalRunConfig(
                matrix_path=tmp_path / "matrix.toml",
                pack_name="core_robot_commands",
                adapter="simulated",
                mcp_url="http://127.0.0.1:8765/mcp",
                samples=1,
                evidence_root=Path("evidence/model_eval"),
            ),
            ("current-position",),
        )
    ]
    assert capsys.readouterr().out.splitlines() == [
        f"Evidence: {tmp_path / 'evidence' / 'run'}",
        "alpha\t2/2\t0.12s\trecommended",
        "beta\t1/2\tn/a\t",
    ]


@pytest.mark.asyncio
async def test_async_main_returns_failure_when_no_candidate_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    @dataclass(frozen=True)
    class FakeSummary:
        candidate_label: str
        pass_count: int
        total_count: int
        median_latency_s: float | None
        correctness_passed: bool
        recommended: bool

    @dataclass(frozen=True)
    class FakeResult:
        evidence_dir: Path
        summaries: tuple[FakeSummary, ...]

    async def fake_run_eval_suite(
        config: EvalRunConfig,
        *,
        scenario_names: tuple[str, ...] | None = None,
    ) -> FakeResult:
        return FakeResult(
            evidence_dir=tmp_path / "evidence" / "run",
            summaries=(FakeSummary("alpha", 0, 1, None, False, False),),
        )

    monkeypatch.setattr("model_eval.__main__.run_eval_suite", fake_run_eval_suite)

    exit_code = await async_main(["run", "--matrix", str(tmp_path / "matrix.toml")])

    assert exit_code == 1


def test_run_parser_rejects_non_positive_samples(tmp_path: Path) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--matrix",
                str(tmp_path / "matrix.toml"),
                "--samples",
                "0",
            ]
        )
