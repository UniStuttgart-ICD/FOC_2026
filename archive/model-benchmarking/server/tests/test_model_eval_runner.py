from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from model_eval.config import EvalRunConfig
from model_eval.runner import run_eval_suite
from model_eval.simulated_moveit import SimulatedMoveItAdapter
from test_support.live_robot_smoke import RecordingRobotToolAdapter
from voice_runtime.agent_turn import AgentTurnInput


class PoseOnlyBackend:
    def __init__(self, recorder: RecordingRobotToolAdapter) -> None:
        self._recorder = recorder
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True

    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        await self._recorder.call_tool(
            "moveit_get_current_pose",
            {"robot_name": "UR10"},
        )
        yield "The UR10 is at its current position."


class FailingAfterToolBackend(PoseOnlyBackend):
    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        await self._recorder.call_tool(
            "moveit_get_current_pose",
            {"robot_name": "UR10"},
        )
        raise RuntimeError("model stopped")
        yield ""


class MoveUpWithoutFinalObservationBackend(PoseOnlyBackend):
    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        pose_text = await self._recorder.call_tool(
            "moveit_get_current_pose",
            {"robot_name": "UR10"},
        )
        import json

        pose = json.loads(pose_text)["structured_content"]["raw"]["pose"]
        plan_text = await self._recorder.call_tool(
            "moveit_plan_cartesian_motion",
            {
                "robot_name": "UR10",
                "waypoints": [
                    {
                        "position": {
                            "x": pose["position"]["x"],
                            "y": pose["position"]["y"],
                            "z": pose["position"]["z"] + 0.05,
                        },
                        "orientation": pose["orientation"],
                    }
                ],
            },
        )
        plan_name = json.loads(plan_text)["structured_content"]["raw"]["plan_name"]
        await self._recorder.call_tool(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": plan_name},
        )
        yield "Moved up a bit."


class HangingBackend(PoseOnlyBackend):
    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        await asyncio.sleep(10)
        yield "never"


def _write_matrix(path: Path) -> None:
    path.write_text(
        """
[[candidates]]
label = "static"
provider = "openai_api"
model = "static"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )


def _config(tmp_path: Path) -> EvalRunConfig:
    matrix_path = tmp_path / "matrix.toml"
    _write_matrix(matrix_path)
    return EvalRunConfig(
        matrix_path=matrix_path,
        pack_name="core_robot_commands",
        samples=1,
        evidence_root=tmp_path / "evidence",
        attempt_timeout_s=120.0,
    )


def _write_multi_candidate_matrix(path: Path) -> None:
    path.write_text(
        """
[[candidates]]
label = "static-a"
provider = "openai_api"
model = "static"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "static-b"
provider = "openai_api"
model = "static"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"
""".strip(),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_run_eval_suite_runs_selected_scenario_and_writes_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)

    result = await run_eval_suite(
        config,
        scenario_names=("current-position",),
        processor_factory=lambda candidate, recorder, mcp_url: PoseOnlyBackend(recorder),
        adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
    )

    assert len(result.attempts) == 1
    assert result.attempts[0].candidate_label == "static"
    assert result.attempts[0].scenario_name == "current-position"
    assert result.attempts[0].passed is True
    assert result.attempts[0].model_turn_count == 1
    assert result.evidence_dir.exists()
    assert (result.evidence_dir / "attempts.jsonl").exists()


@pytest.mark.asyncio
async def test_run_eval_suite_rejects_unknown_scenario_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown scenario names: missing"):
        await run_eval_suite(
            _config(tmp_path),
            scenario_names=("missing",),
            processor_factory=lambda candidate, recorder, mcp_url: PoseOnlyBackend(recorder),
            adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
        )


@pytest.mark.asyncio
async def test_run_eval_suite_records_exception_attempt_and_continues(tmp_path: Path) -> None:
    backend: FailingAfterToolBackend | None = None

    def processor_factory(candidate, recorder, mcp_url):
        nonlocal backend
        backend = FailingAfterToolBackend(recorder)
        return backend

    result = await run_eval_suite(
        _config(tmp_path),
        scenario_names=("current-position",),
        processor_factory=processor_factory,
        adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
    )

    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.passed is False
    assert attempt.reason == "exception"
    assert attempt.exception == "RuntimeError: model stopped"
    assert attempt.model_turn_count == 0
    assert attempt.tool_call_count == 1
    assert attempt.tool_calls[0]["name"] == "moveit_get_current_pose"
    assert backend is not None
    assert backend.disconnected is True


@pytest.mark.asyncio
async def test_run_eval_suite_times_out_hung_attempt_and_writes_jsonl(tmp_path: Path) -> None:
    progress: list[str] = []

    config = EvalRunConfig(
        matrix_path=tmp_path / "matrix.toml",
        pack_name="core_robot_commands",
        samples=1,
        evidence_root=tmp_path / "evidence",
        attempt_timeout_s=0.01,
    )
    _write_matrix(config.matrix_path)

    result = await run_eval_suite(
        config,
        scenario_names=("current-position",),
        processor_factory=lambda candidate, recorder, mcp_url: HangingBackend(recorder),
        adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
        on_attempt=lambda attempt: progress.append(attempt.reason),
    )

    assert len(result.attempts) == 1
    attempt = result.attempts[0]
    assert attempt.passed is False
    assert attempt.reason == "timeout"
    assert attempt.exception == "TimeoutError: attempt exceeded 0.01s"
    assert attempt.elapsed_s < 1.0
    assert progress == ["timeout"]
    assert (result.evidence_dir / "attempts.jsonl").read_text(encoding="utf-8").count("\n") == 1


@pytest.mark.asyncio
async def test_run_eval_suite_observes_final_pose_after_execution(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    _write_matrix(matrix_path)
    config = EvalRunConfig(
        matrix_path=matrix_path,
        pack_name="core_robot_commands",
        samples=1,
        evidence_root=tmp_path / "evidence",
        attempt_timeout_s=120.0,
    )

    result = await run_eval_suite(
        config,
        scenario_names=("move-up-bit",),
        processor_factory=lambda candidate, recorder, mcp_url: MoveUpWithoutFinalObservationBackend(recorder),
        adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
    )

    attempt = result.attempts[0]
    assert attempt.passed is True
    assert [call["name"] for call in attempt.tool_calls] == [
        "moveit_get_current_pose",
        "moveit_plan_cartesian_motion",
        "moveit_execute_plan",
        "moveit_get_current_pose",
    ]
