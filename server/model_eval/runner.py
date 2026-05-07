from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_model_factory import build_agent_chat_model
from langchain_agent_processor import LangChainAgentProcessor
from model_eval.adapters import EvalToolAdapter, create_eval_tool_adapter
from model_eval.candidates import ModelCandidate
from model_eval.config import EvalAdapterName, EvalRunConfig, load_model_matrix
from model_eval.evidence import EvidenceWriter
from model_eval.results import AttemptResult, CandidateSummary
from model_eval.scenarios import EvalScenario, get_scenario_pack
from model_eval.scoring import rank_candidates
from model_eval.validators import get_validator
from test_support.live_robot_smoke import (
    CURRENT_POSE_TOOL_NAME,
    EXECUTION_TOOL_NAMES,
    LiveSmokeRun,
    RecordingRobotToolAdapter,
    run_agent_turn,
)

AdapterFactory = Callable[[EvalAdapterName, str | None], EvalToolAdapter]
ProcessorFactory = Callable[[ModelCandidate, RecordingRobotToolAdapter, str], Any]


@dataclass(frozen=True)
class EvalSuiteResult:
    attempts: tuple[AttemptResult, ...]
    summaries: tuple[CandidateSummary, ...]
    evidence_dir: Path


async def run_eval_suite(
    config: EvalRunConfig,
    *,
    scenario_names: tuple[str, ...] | None = None,
    processor_factory: ProcessorFactory | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> EvalSuiteResult:
    candidates = load_model_matrix(config.matrix_path)
    pack = get_scenario_pack(config.pack_name)
    scenarios = _select_scenarios(pack.scenarios, scenario_names)
    resolved_processor_factory = processor_factory or _build_processor
    resolved_adapter_factory = adapter_factory or _build_adapter

    attempts: list[AttemptResult] = []
    for candidate in candidates:
        for scenario in scenarios:
            for attempt_index in range(config.samples):
                attempts.append(
                    await _run_attempt(
                        candidate=candidate,
                        scenario=scenario,
                        attempt_index=attempt_index,
                        adapter_name=config.adapter,
                        mcp_url=config.mcp_url,
                        processor_factory=resolved_processor_factory,
                        adapter_factory=resolved_adapter_factory,
                    )
                )

    attempt_results = tuple(attempts)
    summaries = rank_candidates(attempt_results)
    evidence_dir = EvidenceWriter(config.evidence_root).write(
        attempts=attempt_results,
        summaries=summaries,
        metadata={
            "pack": config.pack_name,
            "adapter": config.adapter,
            "samples": config.samples,
        },
    )
    return EvalSuiteResult(
        attempts=attempt_results,
        summaries=summaries,
        evidence_dir=evidence_dir,
    )


async def _run_attempt(
    *,
    candidate: ModelCandidate,
    scenario: EvalScenario,
    attempt_index: int,
    adapter_name: EvalAdapterName,
    mcp_url: str | None,
    processor_factory: ProcessorFactory,
    adapter_factory: AdapterFactory,
) -> AttemptResult:
    adapter = adapter_factory(adapter_name, mcp_url)
    recorder = RecordingRobotToolAdapter(adapter)
    processor: Any | None = None
    started = time.perf_counter()

    try:
        processor = processor_factory(candidate, recorder, mcp_url or "")
        run = await run_agent_turn(processor, recorder, scenario.prompt)
        if _needs_final_pose_observation(run):
            await recorder.call_tool(CURRENT_POSE_TOOL_NAME, {"robot_name": "UR10"})
            run = LiveSmokeRun(
                prompt=run.prompt,
                reply=run.reply,
                tool_calls=recorder.calls,
            )
        validation = get_validator(scenario.validator_name)(run)
    except Exception as exc:
        return _exception_attempt(
            candidate=candidate,
            scenario=scenario,
            attempt_index=attempt_index,
            started=started,
            recorder=recorder,
            exc=exc,
        )
    finally:
        if processor is not None:
            disconnect = getattr(processor, "disconnect", None)
            if callable(disconnect):
                result = disconnect()
                if inspect.isawaitable(result):
                    await result

    return AttemptResult(
        candidate_label=candidate.label,
        scenario_name=scenario.name,
        attempt_index=attempt_index,
        prompt=scenario.prompt,
        elapsed_s=time.perf_counter() - started,
        passed=validation.passed,
        reason=validation.reason,
        details=validation.details,
        assistant_reply=run.reply,
        tool_calls=[call.as_json() for call in run.tool_calls],
        tool_call_count=len(run.tool_calls),
        model_turn_count=1,
        exception=None,
    )


def _exception_attempt(
    *,
    candidate: ModelCandidate,
    scenario: EvalScenario,
    attempt_index: int,
    started: float,
    recorder: RecordingRobotToolAdapter,
    exc: Exception,
) -> AttemptResult:
    return AttemptResult(
        candidate_label=candidate.label,
        scenario_name=scenario.name,
        attempt_index=attempt_index,
        prompt=scenario.prompt,
        elapsed_s=time.perf_counter() - started,
        passed=False,
        reason="exception",
        details={},
        assistant_reply="",
        tool_calls=[call.as_json() for call in recorder.calls],
        tool_call_count=len(recorder.calls),
        model_turn_count=0,
        exception=f"{type(exc).__name__}: {exc}",
    )


def _select_scenarios(
    scenarios: tuple[EvalScenario, ...],
    names: tuple[str, ...] | None,
) -> tuple[EvalScenario, ...]:
    if names is None:
        return scenarios

    by_name = {scenario.name: scenario for scenario in scenarios}
    missing = [name for name in names if name not in by_name]
    if missing:
        raise ValueError(f"unknown scenario names: {', '.join(missing)}")
    return tuple(by_name[name] for name in names)


def _needs_final_pose_observation(run: LiveSmokeRun) -> bool:
    if not any(call.name in EXECUTION_TOOL_NAMES for call in run.tool_calls):
        return False
    return not run.tool_calls or run.tool_calls[-1].name != CURRENT_POSE_TOOL_NAME


def _build_processor(
    candidate: ModelCandidate,
    recorder: RecordingRobotToolAdapter,
    mcp_url: str,
) -> LangChainAgentProcessor:
    return LangChainAgentProcessor(
        mcp_url,
        chat_model=build_agent_chat_model(candidate.to_agent_profile()),
        model_label=candidate.label,
        tool_bridge=recorder,
    )


def _build_adapter(adapter: EvalAdapterName, mcp_url: str | None) -> EvalToolAdapter:
    return create_eval_tool_adapter(adapter, mcp_url=mcp_url)
