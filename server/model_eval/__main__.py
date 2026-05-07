from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from model_eval.config import EvalRunConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m model_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--matrix", required=True, type=Path)
    run_parser.add_argument("--pack", default="core_robot_commands")
    run_parser.add_argument(
        "--adapter",
        choices=("simulated", "live-mcp"),
        default="simulated",
    )
    run_parser.add_argument("--mcp-url", default="http://127.0.0.1:8765/mcp")
    run_parser.add_argument("--samples", type=_positive_int, default=1)
    run_parser.add_argument("--attempt-timeout", type=_positive_float, default=120.0)
    run_parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("evidence/model_eval"),
    )
    run_parser.add_argument("--scenario", action="append", dest="scenarios")
    return parser


async def run_eval_suite(
    config: EvalRunConfig,
    *,
    scenario_names: tuple[str, ...] | None = None,
    on_attempt: Any | None = None,
) -> Any:
    from model_eval.runner import run_eval_suite as runner_run_eval_suite

    return await runner_run_eval_suite(
        config,
        scenario_names=scenario_names,
        on_attempt=on_attempt,
    )


async def async_main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        config = EvalRunConfig(
            matrix_path=args.matrix,
            pack_name=args.pack,
            adapter=args.adapter,
            mcp_url=args.mcp_url,
            samples=args.samples,
            evidence_root=args.evidence_root,
            attempt_timeout_s=args.attempt_timeout,
        )
        result = await run_eval_suite(
            config,
            scenario_names=tuple(args.scenarios) if args.scenarios else None,
            on_attempt=_print_attempt_progress,
        )
        print(f"Evidence: {result.evidence_dir}")
        for summary in result.summaries:
            latency = (
                f"{summary.median_latency_s:.2f}s"
                if summary.median_latency_s is not None
                else "n/a"
            )
            marker = "recommended" if summary.recommended else ""
            print(
                f"{summary.candidate_label}\t"
                f"{summary.pass_count}/{summary.total_count}\t"
                f"{latency}\t"
                f"{marker}"
            )
        return 0 if any(summary.correctness_passed for summary in result.summaries) else 1
    parser.error(f"unsupported command: {args.command}")
    return 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("samples must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("attempt timeout must be greater than 0")
    return parsed


def _print_attempt_progress(attempt: Any) -> None:
    status = "pass" if attempt.passed else "fail"
    print(
        "Attempt: "
        f"{attempt.candidate_label}\t"
        f"{attempt.scenario_name}\t"
        f"{attempt.attempt_index}\t"
        f"{status}\t"
        f"{attempt.elapsed_s:.2f}s\t"
        f"{attempt.reason}",
        flush=True,
    )


if __name__ == "__main__":
    main()
