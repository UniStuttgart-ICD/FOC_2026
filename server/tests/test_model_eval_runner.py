from __future__ import annotations

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
        await self._recorder.call_tool(
            "moveit_plan_and_execute_cartesian_motion",
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
        yield "Moved up a bit."


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
async def test_run_eval_suite_observes_final_pose_after_execution(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    _write_matrix(matrix_path)
    config = EvalRunConfig(
        matrix_path=matrix_path,
        pack_name="core_robot_commands",
        samples=1,
        evidence_root=tmp_path / "evidence",
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
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_get_current_pose",
    ]
