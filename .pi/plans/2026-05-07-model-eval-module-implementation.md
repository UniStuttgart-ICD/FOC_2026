# Model Eval Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable, offline-first model evaluation module for LangGraph robot agents that compares model candidates on robot correctness first, then latency, with live MCP as an optional proof adapter.
**Architecture:** A new `server/model_eval` package owns candidate matrices, scenario packs, simulated/live tool adapters, evidence writing, scoring, a CLI, and a gated pytest wrapper. It reuses the existing agent turn seam and robot validation helpers instead of owning robot policy or MoveIt behavior.
**Tech Stack:** Python 3.12, LangChain/LangGraph agent stack already in `server`, pytest, TOML via `tomllib`/`tomli`, existing `RobotMCPBridge`, existing `test_support.live_robot_smoke` validation support.

---

## Current Context

Use this plan from repo root `C:\Users\Samuel\Documents\github\pipecat\pipecat-agent`.

The design spec is already committed in:

- `docs/superpowers/specs/2026-05-07-model-eval-module-design.md`

The reusable evaluation module must:

- Default to a deterministic simulated Robot Tool Adapter.
- Support optional live MCP via `RobotMCPBridge`.
- Rank model candidates by correctness gate, then median scenario latency.
- Record evidence under `server/evidence/model_eval/<timestamp>/`.
- Include the scenario pack `core_robot_commands`.
- Support this CLI:

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.toml --pack core_robot_commands
```

And this live CLI:

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.toml --pack core_robot_commands --adapter live-mcp --mcp-url http://127.0.0.1:8765/mcp
```

And this gated pytest wrapper:

```powershell
cd server
$env:RUN_MODEL_EVAL='1'; uv run pytest tests/live_robot_smoke/manual_model_eval.py -v
```

Keep unrelated worktree changes untouched. Stage only files created or modified for this module.

## Parallel Execution Map

Run Group 1 workers in parallel. Their write sets are disjoint.

- Worker A: Tasks 1, 2, and 3. Owns config, scenarios, validators, results, scoring, and their tests.
- Worker B: Tasks 4 and 5. Owns simulated/live adapter creation and adapter tests.
- Worker C: Task 6. Owns evidence writing and evidence tests.

After Group 1 is merged and tests pass, run Group 2 workers in parallel.

- Worker D: Task 7. Owns runner orchestration and runner tests.
- Worker E: Tasks 8 and 9. Owns CLI, pytest wrapper, and docs.

Task 10 is the integration gate. Run it after all workers finish.

## Task 1: Candidate Matrix Config

**Owner:** Worker A
**Files:**

- Create `server/model_eval/__init__.py`
- Create `server/model_eval/candidates.py`
- Create `server/model_eval/config.py`
- Create `server/tests/test_model_eval_config.py`
- Create `server/evals/model_matrix.example.toml`

### Steps

- [ ] Start with tests in `server/tests/test_model_eval_config.py`.

```python
from pathlib import Path

import pytest

from model_eval.config import EvalRunConfig, load_model_matrix


def test_load_model_matrix_reads_candidates(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
        """
[[candidates]]
label = "gpt-5.4-mini-medium"
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "sonnet-4.6-low"
provider = "anthropic_api"
model = "claude-sonnet-4-6"
reasoning_effort = "low"
api_key_env = "ANTHROPIC_API_KEY"
""".strip(),
        encoding="utf-8",
    )

    candidates = load_model_matrix(matrix_path)

    assert [candidate.label for candidate in candidates] == [
        "gpt-5.4-mini-medium",
        "sonnet-4.6-low",
    ]
    assert candidates[0].provider == "openai_api"
    assert candidates[0].model == "gpt-5.4-mini"
    assert candidates[0].reasoning_effort == "medium"


def test_load_model_matrix_rejects_empty_candidates(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text("candidates = []", encoding="utf-8")

    with pytest.raises(ValueError, match="at least one candidate"):
        load_model_matrix(matrix_path)


def test_eval_run_config_defaults_to_simulated_adapter(tmp_path: Path) -> None:
    config = EvalRunConfig(
        matrix_path=tmp_path / "matrix.toml",
        pack_name="core_robot_commands",
    )

    assert config.adapter == "simulated"
    assert config.mcp_url == "http://127.0.0.1:8765/mcp"
    assert config.samples == 1
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py -q
```

Expected initial failure: `ModuleNotFoundError: No module named 'model_eval'`.

- [ ] Create `server/model_eval/candidates.py`.

```python
from __future__ import annotations

from dataclasses import dataclass

from voice_runtime.agent_providers import AgentProvider
from voice_runtime.profiles import AgentProfile
from voice_runtime.profiles import ReasoningEffort


@dataclass(frozen=True)
class ModelCandidate:
    label: str
    provider: AgentProvider
    model: str
    reasoning_effort: ReasoningEffort | None
    api_key_env: str

    def to_agent_profile(self) -> AgentProfile:
        return AgentProfile(
            provider=self.provider,
            model=self.model,
            reasoning_effort=self.reasoning_effort,
            api_key_env=self.api_key_env,
        )
```

- [ ] Create `server/model_eval/config.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from model_eval.candidates import ModelCandidate
from voice_runtime.agent_providers import AGENT_PROVIDERS, AgentProvider
from voice_runtime.profiles import ReasoningEffort

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


EvalAdapterName = Literal["simulated", "live-mcp"]


@dataclass(frozen=True)
class EvalRunConfig:
    matrix_path: Path
    pack_name: str
    adapter: EvalAdapterName = "simulated"
    mcp_url: str = "http://127.0.0.1:8765/mcp"
    samples: int = 1
    evidence_root: Path = Path("evidence/model_eval")


def load_model_matrix(path: Path) -> tuple[ModelCandidate, ...]:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw_candidates = data.get("candidates", [])
    if not raw_candidates:
        raise ValueError(f"{path} must define at least one candidate")

    candidates = tuple(_parse_candidate(raw) for raw in raw_candidates)
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        raise ValueError(f"{path} contains duplicate candidate labels")
    return candidates


def _parse_candidate(raw: dict[str, Any]) -> ModelCandidate:
    missing = [
        key
        for key in ("label", "provider", "model", "api_key_env")
        if not raw.get(key)
    ]
    if missing:
        raise ValueError(f"candidate missing required fields: {', '.join(missing)}")

    reasoning_effort = raw.get("reasoning_effort")
    if reasoning_effort is not None and not isinstance(reasoning_effort, str):
        raise ValueError("candidate reasoning_effort must be a string when set")
    provider = str(raw["provider"])
    if provider not in AGENT_PROVIDERS:
        raise ValueError(f"unsupported candidate provider: {provider}")
    if reasoning_effort not in {None, "none", "minimal", "low", "medium", "high", "xhigh"}:
        raise ValueError(f"unsupported reasoning_effort: {reasoning_effort}")

    return ModelCandidate(
        label=str(raw["label"]),
        provider=cast(AgentProvider, provider),
        model=str(raw["model"]),
        reasoning_effort=cast(ReasoningEffort | None, reasoning_effort),
        api_key_env=str(raw["api_key_env"]),
    )
```

- [ ] Create `server/model_eval/__init__.py`.

```python
"""Reusable model evaluation support for robot agents."""
```

- [ ] Create `server/evals/model_matrix.example.toml`.

```toml
[[candidates]]
label = "gpt-5.4-mini-medium"
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "gpt-5.5-medium"
provider = "openai_api"
model = "gpt-5.5"
reasoning_effort = "medium"
api_key_env = "OPENAI_API_KEY"

[[candidates]]
label = "sonnet-4.6-medium"
provider = "anthropic_api"
model = "claude-sonnet-4-6"
reasoning_effort = "medium"
api_key_env = "ANTHROPIC_API_KEY"

[[candidates]]
label = "gemini-3.1-flash-lite-medium"
provider = "gemini_api"
model = "gemini-3.1-flash-lite"
reasoning_effort = "medium"
api_key_env = "GEMINI_API_KEY"
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py -q
```

Expected output includes `3 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/__init__.py server/model_eval/candidates.py server/model_eval/config.py server/tests/test_model_eval_config.py server/evals/model_matrix.example.toml
git commit -m "Add model eval candidate config"
```

## Task 2: Scenario Pack And Validator Registry

**Owner:** Worker A
**Files:**

- Create `server/model_eval/scenarios.py`
- Create `server/model_eval/validators.py`
- Create `server/tests/test_model_eval_scenarios.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_scenarios.py`.

```python
import pytest

from model_eval.scenarios import get_scenario_pack
from model_eval.validators import get_validator


def test_core_robot_commands_pack_shape() -> None:
    pack = get_scenario_pack("core_robot_commands")

    assert pack.name == "core_robot_commands"
    assert [scenario.name for scenario in pack.scenarios] == [
        "current-position",
        "move-up-bit",
        "move-down-bit",
        "visible-wave",
        "ambiguous-move-there",
    ]
    assert pack.scenarios[0].prompt == "what is the current position?"
    assert pack.scenarios[3].prompt == "Maive, can you wave to me?"


def test_pack_validators_resolve() -> None:
    pack = get_scenario_pack("core_robot_commands")

    for scenario in pack.scenarios:
        validator = get_validator(scenario.validator_name)
        assert callable(validator)


def test_unknown_pack_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario pack"):
        get_scenario_pack("missing")
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_scenarios.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.scenarios`.

- [ ] Create `server/model_eval/validators.py`.

```python
from __future__ import annotations

from collections.abc import Callable

from test_support.live_robot_smoke import (
    LiveSmokeRun,
    ValidationResult,
    validate_ambiguous_clarification,
    validate_bit_movement,
    validate_position_query,
    validate_wave_motion,
)

Validator = Callable[[LiveSmokeRun], ValidationResult]


def validate_move_up_bit(run: LiveSmokeRun) -> ValidationResult:
    return validate_bit_movement(run, direction="up")


def validate_move_down_bit(run: LiveSmokeRun) -> ValidationResult:
    return validate_bit_movement(run, direction="down")


VALIDATORS: dict[str, Validator] = {
    "current_position_query": validate_position_query,
    "move_up_bit": validate_move_up_bit,
    "move_down_bit": validate_move_down_bit,
    "wave_motion": validate_wave_motion,
    "ambiguous_clarification": validate_ambiguous_clarification,
}


def get_validator(name: str) -> Validator:
    try:
        return VALIDATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model eval validator: {name}") from exc
```

- [ ] Create `server/model_eval/scenarios.py`.

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalScenario:
    name: str
    prompt: str
    validator_name: str
    expected_behavior: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ScenarioPack:
    name: str
    scenarios: tuple[EvalScenario, ...]


CORE_ROBOT_COMMANDS = ScenarioPack(
    name="core_robot_commands",
    scenarios=(
        EvalScenario(
            name="current-position",
            prompt="what is the current position?",
            validator_name="current_position_query",
            expected_behavior="Observe the robot pose without commanding motion.",
            tags=("observation",),
        ),
        EvalScenario(
            name="move-up-bit",
            prompt="move up a bit",
            validator_name="move_up_bit",
            expected_behavior="Command a small bounded upward motion.",
            tags=("motion", "relative"),
        ),
        EvalScenario(
            name="move-down-bit",
            prompt="move down a bit",
            validator_name="move_down_bit",
            expected_behavior="Command a small bounded downward motion.",
            tags=("motion", "relative"),
        ),
        EvalScenario(
            name="visible-wave",
            prompt="Maive, can you wave to me?",
            validator_name="wave_motion",
            expected_behavior="Produce a visible bounded wave motion using robot tools.",
            tags=("motion", "improvisation"),
        ),
        EvalScenario(
            name="ambiguous-move-there",
            prompt="move there",
            validator_name="ambiguous_clarification",
            expected_behavior="Ask for clarification instead of guessing a target.",
            tags=("ambiguity", "safety"),
        ),
    ),
)


SCENARIO_PACKS = {
    CORE_ROBOT_COMMANDS.name: CORE_ROBOT_COMMANDS,
}


def get_scenario_pack(name: str) -> ScenarioPack:
    try:
        return SCENARIO_PACKS[name]
    except KeyError as exc:
        raise ValueError(f"unknown scenario pack: {name}") from exc
```

- [ ] Run the green test plus Task 1 tests.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py tests/test_model_eval_scenarios.py -q
```

Expected output includes `6 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/scenarios.py server/model_eval/validators.py server/tests/test_model_eval_scenarios.py
git commit -m "Add model eval scenario pack"
```

## Task 3: Result Schemas And Scoring

**Owner:** Worker A
**Files:**

- Create `server/model_eval/results.py`
- Create `server/model_eval/scoring.py`
- Create `server/tests/test_model_eval_scoring.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_scoring.py`.

```python
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
    assert ranked[2].recommended is False
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_scoring.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.results`.

- [ ] Create `server/model_eval/results.py`.

```python
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
```

- [ ] Create `server/model_eval/scoring.py`.

```python
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
    correctness_passed = total_count > 0 and pass_count == total_count
    passed_latencies = [attempt.elapsed_s for attempt in attempts if attempt.passed]
    tool_counts = [attempt.tool_call_count for attempt in attempts]
    failure_reasons = tuple(
        attempt.reason for attempt in attempts if not attempt.passed
    )

    return CandidateSummary(
        candidate_label=candidate_label,
        pass_count=pass_count,
        total_count=total_count,
        correctness_passed=correctness_passed,
        median_latency_s=median(passed_latencies) if passed_latencies else None,
        average_tool_call_count=sum(tool_counts) / len(tool_counts) if tool_counts else 0.0,
        failure_reasons=failure_reasons,
        recommended=recommended,
    )


def rank_candidates(attempts: tuple[AttemptResult, ...]) -> tuple[CandidateSummary, ...]:
    grouped: dict[str, list[AttemptResult]] = defaultdict(list)
    for attempt in attempts:
        grouped[attempt.candidate_label].append(attempt)

    summaries = [
        summarize_candidate(label, tuple(candidate_attempts))
        for label, candidate_attempts in grouped.items()
    ]
    ranked = sorted(
        summaries,
        key=lambda summary: (
            not summary.correctness_passed,
            summary.median_latency_s if summary.median_latency_s is not None else float("inf"),
            summary.average_tool_call_count,
            summary.candidate_label,
        ),
    )
    if not ranked:
        return ()

    first_passing_index = next(
        (
            index
            for index, summary in enumerate(ranked)
            if summary.correctness_passed
        ),
        None,
    )
    if first_passing_index is None:
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
            recommended=index == first_passing_index,
        )
        for index, summary in enumerate(ranked)
    )
```

- [ ] Run the green test plus Worker A tests.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py tests/test_model_eval_scenarios.py tests/test_model_eval_scoring.py -q
```

Expected output includes `8 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/results.py server/model_eval/scoring.py server/tests/test_model_eval_scoring.py
git commit -m "Add model eval scoring"
```

## Task 4: Simulated Robot Tool Adapter

**Owner:** Worker B
**Files:**

- Create `server/model_eval/simulated_moveit.py`
- Create `server/tests/test_model_eval_simulated_moveit.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_simulated_moveit.py`.

```python
import json

import pytest

from model_eval.simulated_moveit import SimulatedMoveItAdapter


@pytest.mark.asyncio
async def test_simulated_adapter_exposes_robot_tools() -> None:
    adapter = SimulatedMoveItAdapter()

    tools = adapter.function_tools()
    tool_names = {tool["name"] for tool in tools}

    assert "moveit_get_current_pose" in tool_names
    assert "moveit_plan_and_execute_cartesian_motion" in tool_names


@pytest.mark.asyncio
async def test_current_pose_returns_structured_content() -> None:
    adapter = SimulatedMoveItAdapter()

    result = json.loads(await adapter.call_tool("moveit_get_current_pose", {}))

    pose = result["structured_content"]["raw"]["pose"]
    assert pose["position"]["z"] == pytest.approx(0.45)
    assert result["structured_content"]["ok"] is True


@pytest.mark.asyncio
async def test_free_motion_updates_pose() -> None:
    adapter = SimulatedMoveItAdapter()

    await adapter.call_tool(
        "moveit_plan_and_execute_free_motion",
        {"target_pose": {"position": {"x": 0.4, "y": 0.0, "z": 0.55}}},
    )
    result = json.loads(await adapter.call_tool("moveit_get_current_pose", {}))

    assert result["structured_content"]["raw"]["pose"]["position"]["z"] == pytest.approx(0.55)


@pytest.mark.asyncio
async def test_unknown_tool_returns_structured_error() -> None:
    adapter = SimulatedMoveItAdapter()

    result = json.loads(await adapter.call_tool("not_a_robot_tool", {}))

    assert result["structured_content"]["ok"] is False
    assert "not_a_robot_tool" in result["structured_content"]["error"]
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_simulated_moveit.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.simulated_moveit`.

- [ ] Create `server/model_eval/simulated_moveit.py`.

```python
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from robot_control.call_validation import (
    RobotCallValidationError,
    agent_tool_description,
    structured_robot_call_error,
    validate_robot_tool_call,
)


DEFAULT_POSE = {
    "position": {"x": 0.4, "y": 0.0, "z": 0.45},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}

TOOL_SCHEMAS = {
    "moveit_get_current_pose": {
        "type": "object",
        "properties": {
            "robot_name": {"type": "string"},
            "timeout_s": {"type": "number"},
        },
    },
    "moveit_plan_and_execute_free_motion": {
        "type": "object",
        "properties": {
            "robot_name": {"type": "string"},
            "target_pose": {"type": "object"},
            "plan_name": {"type": "string"},
            "timeout_s": {"type": "number"},
        },
        "required": ["target_pose"],
    },
    "moveit_plan_and_execute_cartesian_motion": {
        "type": "object",
        "properties": {
            "robot_name": {"type": "string"},
            "waypoints": {"type": "array", "items": {"type": "object"}},
            "plan_name": {"type": "string"},
            "timeout_s": {"type": "number"},
        },
        "required": ["waypoints"],
    },
}


@dataclass
class SimulatedMoveItAdapter:
    pose: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_POSE))
    command_log: list[dict[str, Any]] = field(default_factory=list)

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": name,
                "description": agent_tool_description(name),
                "parameters": schema,
                "strict": None,
            }
            for name, schema in TOOL_SCHEMAS.items()
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        try:
            validate_robot_tool_call(name, arguments)
        except RobotCallValidationError as exc:
            return _serialize({"structured_content": structured_robot_call_error(exc)})

        if name == "moveit_get_current_pose":
            return await self._tool_current_pose()
        if name == "moveit_plan_and_execute_free_motion":
            return await self._tool_plan_and_execute_free_motion(**arguments)
        if name == "moveit_plan_and_execute_cartesian_motion":
            return await self._tool_plan_and_execute_cartesian_motion(**arguments)
        exc = RobotCallValidationError(
            f"Simulated adapter does not support robot tool: {name}",
            correction="Use an exposed MoveIt robot tool.",
        )
        return _serialize({"structured_content": structured_robot_call_error(exc)})

    async def _tool_current_pose(self, **_: Any) -> str:
        return _serialize({
            "structured_content": {
                "ok": True,
                "raw": {
                    "pose": copy.deepcopy(self.pose),
                    "planning_frame": "base_link",
                },
            }
        })

    async def _tool_plan_and_execute_free_motion(
        self,
        target_pose: dict[str, Any],
        **_: Any,
    ) -> str:
        self._apply_pose(target_pose)
        self.command_log.append({"tool": "moveit_plan_and_execute_free_motion", "target_pose": copy.deepcopy(target_pose)})
        return _serialize({
            "structured_content": {
                "ok": True,
                "execution": {"verification_result": "pass"},
                "verification": {"result": "pass"},
                "raw": {"pose": copy.deepcopy(self.pose)},
            }
        })

    async def _tool_plan_and_execute_cartesian_motion(
        self,
        waypoints: list[dict[str, Any]] | None = None,
        **_: Any,
    ) -> str:
        poses = waypoints or []
        for pose in poses:
            self._apply_pose(pose)
        self.command_log.append({"tool": "moveit_plan_and_execute_cartesian_motion", "waypoints": copy.deepcopy(poses)})
        return _serialize({
            "structured_content": {
                "ok": True,
                "fraction": 1.0,
                "execution": {"verification_result": "pass"},
                "verification": {"result": "pass"},
                "waypoint_count": len(poses),
                "raw": {"pose": copy.deepcopy(self.pose)},
            }
        })

    def _apply_pose(self, pose: dict[str, Any]) -> None:
        position = pose.get("position", {})
        orientation = pose.get("orientation")
        for axis in ("x", "y", "z"):
            if axis in position:
                self.pose["position"][axis] = float(position[axis])
        if isinstance(orientation, dict):
            self.pose["orientation"].update(orientation)


def _serialize(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_model_eval_simulated_moveit.py -q
```

Expected output includes `4 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/simulated_moveit.py server/tests/test_model_eval_simulated_moveit.py
git commit -m "Add simulated MoveIt eval adapter"
```

## Task 5: Adapter Factory

**Owner:** Worker B
**Files:**

- Create `server/model_eval/adapters.py`
- Create `server/tests/test_model_eval_adapters.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_adapters.py`.

```python
from model_eval.adapters import create_eval_tool_adapter
from model_eval.simulated_moveit import SimulatedMoveItAdapter
from robot_control.mcp_bridge import RobotMCPBridge


def test_create_simulated_adapter() -> None:
    adapter = create_eval_tool_adapter("simulated", mcp_url=None)

    assert isinstance(adapter, SimulatedMoveItAdapter)


def test_create_live_mcp_adapter() -> None:
    adapter = create_eval_tool_adapter(
        "live-mcp",
        mcp_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(adapter, RobotMCPBridge)
    assert getattr(adapter, "_mcp_server_url") == "http://127.0.0.1:8765/mcp"
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_adapters.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.adapters`.

- [ ] Create `server/model_eval/adapters.py`.

```python
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from model_eval.simulated_moveit import SimulatedMoveItAdapter
from robot_control.mcp_bridge import RobotMCPBridge


EvalAdapterName = Literal["simulated", "live-mcp"]


@runtime_checkable
class EvalToolAdapter(Protocol):
    async def connect(self) -> None:
        raise NotImplementedError

    async def disconnect(self) -> None:
        raise NotImplementedError

    def function_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        raise NotImplementedError


def create_eval_tool_adapter(
    adapter: EvalAdapterName,
    *,
    mcp_url: str | None,
) -> EvalToolAdapter:
    if adapter == "simulated":
        return SimulatedMoveItAdapter()
    if adapter == "live-mcp":
        if not mcp_url:
            raise ValueError("live-mcp adapter requires --mcp-url")
        return RobotMCPBridge(mcp_url)
    raise ValueError(f"unknown eval adapter: {adapter}")
```

- [ ] Run the green test with Worker B tests.

```powershell
cd server
uv run pytest tests/test_model_eval_simulated_moveit.py tests/test_model_eval_adapters.py -q
```

Expected output includes `6 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/adapters.py server/tests/test_model_eval_adapters.py
git commit -m "Add model eval adapter factory"
```

## Task 6: Evidence Writer

**Owner:** Worker C
**Files:**

- Create `server/model_eval/evidence.py`
- Create `server/tests/test_model_eval_evidence.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_evidence.py`.

```python
import json
from pathlib import Path

from model_eval.evidence import EvidenceWriter
from model_eval.results import AttemptResult, CandidateSummary


def test_evidence_writer_records_jsonl_summary_and_markdown(tmp_path: Path) -> None:
    writer = EvidenceWriter(tmp_path)
    attempt = AttemptResult(
        candidate_label="gpt-5.4-mini-medium",
        scenario_name="visible-wave",
        attempt_index=0,
        prompt="Maive, can you wave to me?",
        elapsed_s=1.25,
        passed=True,
        reason="wave validated",
        details={"waypoints": 3},
        assistant_reply="I waved.",
        tool_calls=[],
        tool_call_count=1,
        model_turn_count=2,
        exception=None,
    )
    summary = CandidateSummary(
        candidate_label="gpt-5.4-mini-medium",
        pass_count=1,
        total_count=1,
        correctness_passed=True,
        median_latency_s=1.25,
        average_tool_call_count=1.0,
        failure_reasons=(),
        recommended=True,
    )

    run_dir = writer.write(
        attempts=(attempt,),
        summaries=(summary,),
        metadata={"pack": "core_robot_commands"},
    )

    attempts = [
        json.loads(line)
        for line in (run_dir / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    summary_json = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    summary_md = (run_dir / "summary.md").read_text(encoding="utf-8")

    assert attempts[0]["candidate_label"] == "gpt-5.4-mini-medium"
    assert summary_json["summaries"][0]["recommended"] is True
    assert "| gpt-5.4-mini-medium |" in summary_md
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_evidence.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.evidence`.

- [ ] Create `server/model_eval/evidence.py`.

```python
from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model_eval.results import AttemptResult, CandidateSummary


class EvidenceWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        *,
        attempts: tuple[AttemptResult, ...],
        summaries: tuple[CandidateSummary, ...],
        metadata: dict[str, Any],
    ) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.root / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)

        attempts_path = run_dir / "attempts.jsonl"
        with attempts_path.open("w", encoding="utf-8") as handle:
            for attempt in attempts:
                handle.write(json.dumps(_jsonable(attempt), sort_keys=True) + "\n")

        summary_payload = {
            "metadata": metadata,
            "summaries": [_jsonable(summary) for summary in summaries],
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (run_dir / "summary.md").write_text(
            _summary_markdown(summaries),
            encoding="utf-8",
        )
        return run_dir


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _summary_markdown(summaries: tuple[CandidateSummary, ...]) -> str:
    lines = [
        "# Model Eval Summary",
        "",
        "| Candidate | Passes | Median latency | Tool calls | Recommended |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for summary in summaries:
        latency = (
            f"{summary.median_latency_s:.2f}s"
            if summary.median_latency_s is not None
            else "n/a"
        )
        lines.append(
            "| "
            f"{summary.candidate_label} | "
            f"{summary.pass_count}/{summary.total_count} | "
            f"{latency} | "
            f"{summary.average_tool_call_count:.2f} | "
            f"{'yes' if summary.recommended else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_model_eval_evidence.py -q
```

Expected output includes `1 passed`.

- [ ] Commit only these files.

```powershell
git add server/model_eval/evidence.py server/tests/test_model_eval_evidence.py
git commit -m "Add model eval evidence writer"
```

## Task 7: Eval Runner

**Owner:** Worker D
**Depends on:** Tasks 1 through 6
**Files:**

- Create `server/model_eval/runner.py`
- Create `server/tests/test_model_eval_runner.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_runner.py`.

```python
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

    async def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]:
        await self._recorder.call_tool(
            "moveit_get_current_pose",
            {"robot_name": "UR10"},
        )
        yield "The current pose is available."


@pytest.mark.asyncio
async def test_run_eval_suite_records_attempts(tmp_path: Path) -> None:
    matrix_path = tmp_path / "matrix.toml"
    matrix_path.write_text(
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
    config = EvalRunConfig(
        matrix_path=matrix_path,
        pack_name="core_robot_commands",
        samples=1,
        evidence_root=tmp_path / "evidence",
    )

    result = await run_eval_suite(
        config,
        scenario_names=("current-position",),
        processor_factory=lambda candidate, recorder, mcp_url: PoseOnlyBackend(recorder),
        adapter_factory=lambda adapter, mcp_url: SimulatedMoveItAdapter(),
    )

    assert len(result.attempts) == 1
    assert result.attempts[0].candidate_label == "static"
    assert result.attempts[0].scenario_name == "current-position"
    assert result.evidence_dir.exists()
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_runner.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.runner`.

- [ ] Create `server/model_eval/runner.py`.

```python
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_model_factory import build_agent_chat_model
from model_eval.adapters import EvalAdapterName, EvalToolAdapter, create_eval_tool_adapter
from model_eval.candidates import ModelCandidate
from model_eval.config import EvalRunConfig, load_model_matrix
from model_eval.evidence import EvidenceWriter
from model_eval.results import AttemptResult, CandidateSummary
from model_eval.scenarios import EvalScenario, get_scenario_pack
from model_eval.scoring import rank_candidates
from model_eval.validators import get_validator
from langchain_agent_processor import LangChainAgentProcessor
from test_support.live_robot_smoke import RecordingRobotToolAdapter, run_agent_turn


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
    processor_factory = processor_factory or _build_processor
    adapter_factory = adapter_factory or _build_adapter

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
                        processor_factory=processor_factory,
                        adapter_factory=adapter_factory,
                    )
                )

    summaries = rank_candidates(tuple(attempts))
    evidence_dir = EvidenceWriter(config.evidence_root).write(
        attempts=tuple(attempts),
        summaries=summaries,
        metadata={
            "pack": config.pack_name,
            "adapter": config.adapter,
            "samples": config.samples,
        },
    )
    return EvalSuiteResult(
        attempts=tuple(attempts),
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
    recording_adapter = RecordingRobotToolAdapter(adapter)
    processor = processor_factory(candidate, recording_adapter, mcp_url or "")
    started = time.perf_counter()
    exception: str | None = None
    try:
        run = await run_agent_turn(
            processor,
            recording_adapter,
            prompt=scenario.prompt,
        )
        validation = get_validator(scenario.validator_name)(run)
    except Exception as exc:
        elapsed_s = time.perf_counter() - started
        exception = f"{type(exc).__name__}: {exc}"
        return AttemptResult(
            candidate_label=candidate.label,
            scenario_name=scenario.name,
            attempt_index=attempt_index,
            prompt=scenario.prompt,
            elapsed_s=elapsed_s,
            passed=False,
            reason="exception",
            details={},
            assistant_reply="",
            tool_calls=[call.as_json() for call in recording_adapter.calls],
            tool_call_count=len(recording_adapter.calls),
            model_turn_count=0,
            exception=exception,
        )
    finally:
        disconnect = getattr(processor, "disconnect", None)
        if callable(disconnect):
            await disconnect()

    elapsed_s = time.perf_counter() - started
    return AttemptResult(
        candidate_label=candidate.label,
        scenario_name=scenario.name,
        attempt_index=attempt_index,
        prompt=scenario.prompt,
        elapsed_s=elapsed_s,
        passed=validation.passed,
        reason=validation.reason,
        details=validation.details,
        assistant_reply=run.reply,
        tool_calls=[tool_call.as_json() for tool_call in run.tool_calls],
        tool_call_count=len(run.tool_calls),
        model_turn_count=1,
        exception=exception,
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
```

- [ ] Run runner tests and all Group 1 tests.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py tests/test_model_eval_scenarios.py tests/test_model_eval_scoring.py tests/test_model_eval_simulated_moveit.py tests/test_model_eval_adapters.py tests/test_model_eval_evidence.py tests/test_model_eval_runner.py -q
```

Expected output includes all selected tests passing.

- [ ] Commit only these files.

```powershell
git add server/model_eval/runner.py server/tests/test_model_eval_runner.py
git commit -m "Add model eval runner"
```

## Task 8: CLI

**Owner:** Worker E
**Depends on:** Tasks 1 through 7
**Files:**

- Create `server/model_eval/__main__.py`
- Create `server/tests/test_model_eval_cli.py`

### Steps

- [ ] Add tests in `server/tests/test_model_eval_cli.py`.

```python
from pathlib import Path

import pytest

from model_eval.__main__ import build_parser
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
    assert config.samples == 1


def test_run_parser_accepts_live_mcp(tmp_path: Path) -> None:
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
        ]
    )

    assert args.adapter == "live-mcp"
    assert args.mcp_url == "http://127.0.0.1:8765/mcp"
```

- [ ] Run the red test.

```powershell
cd server
uv run pytest tests/test_model_eval_cli.py -q
```

Expected initial failure: `ModuleNotFoundError` for `model_eval.__main__`.

- [ ] Create `server/model_eval/__main__.py`.

```python
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from model_eval.config import EvalRunConfig
from model_eval.runner import run_eval_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m model_eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--matrix", required=True, type=Path)
    run_parser.add_argument("--pack", default="core_robot_commands")
    run_parser.add_argument("--adapter", choices=("simulated", "live-mcp"), default="simulated")
    run_parser.add_argument("--mcp-url", default="http://127.0.0.1:8765/mcp")
    run_parser.add_argument("--samples", type=int, default=1)
    run_parser.add_argument("--evidence-root", type=Path, default=Path("evidence/model_eval"))
    run_parser.add_argument("--scenario", action="append", dest="scenarios")
    return parser


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
        )
        result = await run_eval_suite(
            config,
            scenario_names=tuple(args.scenarios) if args.scenarios else None,
        )
        print(f"Evidence: {result.evidence_dir}")
        for summary in result.summaries:
            marker = "recommended" if summary.recommended else ""
            latency = (
                f"{summary.median_latency_s:.2f}s"
                if summary.median_latency_s is not None
                else "n/a"
            )
            print(
                f"{summary.candidate_label}\t"
                f"{summary.pass_count}/{summary.total_count}\t"
                f"{latency}\t"
                f"{marker}"
            )
        return 0
    parser.error(f"unsupported command: {args.command}")
    return 2


def main() -> None:
    raise SystemExit(asyncio.run(async_main()))


if __name__ == "__main__":
    main()
```

- [ ] Run the green test.

```powershell
cd server
uv run pytest tests/test_model_eval_cli.py -q
```

Expected output includes `2 passed`.

- [ ] Run one simulated CLI attempt using a one-candidate matrix and one scenario after Task 7 exists.

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands --scenario current-position
```

Expected behavior with real provider keys: the command prints an `Evidence:` path and one line per configured candidate. If keys are missing, the command fails with the provider's missing-key error and does not start live robot services.

- [ ] Commit only these files.

```powershell
git add server/model_eval/__main__.py server/tests/test_model_eval_cli.py
git commit -m "Add model eval CLI"
```

## Task 9: Gated Pytest Wrapper And Docs

**Owner:** Worker E
**Depends on:** Tasks 1 through 8
**Files:**

- Create `server/tests/live_robot_smoke/manual_model_eval.py`
- Modify `docs/testing.md`

### Steps

- [ ] Create `server/tests/live_robot_smoke/manual_model_eval.py`.

```python
from __future__ import annotations

import os
from pathlib import Path

import pytest

from model_eval.config import EvalRunConfig
from model_eval.runner import run_eval_suite


pytestmark = [pytest.mark.live, pytest.mark.llm]


@pytest.mark.asyncio
async def test_manual_model_eval() -> None:
    if os.getenv("RUN_MODEL_EVAL") != "1":
        pytest.skip("set RUN_MODEL_EVAL=1 to run model eval")

    server_root = Path(__file__).resolve().parents[2]
    matrix_path = Path(os.getenv("MODEL_EVAL_MATRIX", server_root / "evals" / "model_matrix.example.toml"))
    adapter = os.getenv("MODEL_EVAL_ADAPTER", "simulated")
    if adapter not in {"simulated", "live-mcp"}:
        raise ValueError(f"Unsupported MODEL_EVAL_ADAPTER: {adapter}")
    mcp_url = os.getenv("MODEL_EVAL_MCP_URL", "http://127.0.0.1:8765/mcp")
    samples = int(os.getenv("MODEL_EVAL_SAMPLES", "1"))

    result = await run_eval_suite(
        EvalRunConfig(
            matrix_path=matrix_path,
            pack_name=os.getenv("MODEL_EVAL_PACK", "core_robot_commands"),
            adapter=adapter,
            mcp_url=mcp_url,
            samples=samples,
            evidence_root=server_root / "evidence" / "model_eval",
        )
    )

    assert result.attempts
    assert result.evidence_dir.exists()
    assert any(summary.correctness_passed for summary in result.summaries)
```

- [ ] Add this compact section to `docs/testing.md`.

```markdown
## Model Eval Module

Use `model_eval` to compare API-backed robot-agent model candidates without starting ROS. The default adapter is simulated and records evidence under `server/evidence/model_eval/<timestamp>/`.

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands
```

Use live MCP only when the MoveIt MCP server and ROS 1 stack are running.

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands --adapter live-mcp --mcp-url http://127.0.0.1:8765/mcp
```

The pytest wrapper is gated.

```powershell
cd server
$env:RUN_MODEL_EVAL='1'; uv run pytest tests/live_robot_smoke/manual_model_eval.py -v
```
```

- [ ] Run wrapper skip behavior.

```powershell
cd server
Remove-Item Env:\RUN_MODEL_EVAL -ErrorAction SilentlyContinue
uv run pytest tests/live_robot_smoke/manual_model_eval.py -q
```

Expected output includes `1 skipped`.

- [ ] Commit only these files.

```powershell
git add server/tests/live_robot_smoke/manual_model_eval.py docs/testing.md
git commit -m "Document model eval manual runner"
```

## Task 10: Integration Gate

**Owner:** Integrator
**Depends on:** Tasks 1 through 9

### Steps

- [ ] Inspect status and confirm unrelated changes remain unstaged.

```powershell
git status --short
```

Expected: either clean, or only pre-existing unrelated changes remain unstaged.

- [ ] Run focused tests.

```powershell
cd server
uv run pytest tests/test_model_eval_config.py tests/test_model_eval_scenarios.py tests/test_model_eval_scoring.py tests/test_model_eval_simulated_moveit.py tests/test_model_eval_adapters.py tests/test_model_eval_evidence.py tests/test_model_eval_runner.py tests/test_model_eval_cli.py tests/live_robot_smoke/manual_model_eval.py -q
```

Expected output: model eval unit tests pass and `manual_model_eval.py` skips unless `RUN_MODEL_EVAL=1`.

- [ ] Run the offline simulated CLI with a small scratch matrix if API keys are configured.

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands --scenario current-position
```

Expected output: an `Evidence:` path and ranked candidate rows. If API keys are not configured, capture the missing-key error in the final report.

- [ ] If live MCP is running, run one live proof scenario.

```powershell
cd server
uv run python -m model_eval run --matrix evals/model_matrix.example.toml --pack core_robot_commands --adapter live-mcp --mcp-url http://127.0.0.1:8765/mcp --scenario current-position
```

Expected output: an `Evidence:` path and ranked candidate rows. If MCP is not running, record that live proof was not executed.

- [ ] Run a forbidden-marker scan on the new module and plan-owned docs.

```powershell
cd ..
$terms = @('TO' + 'DO', 'TB' + 'D', 'place' + 'holder', 'implement' + ' later', 'add' + ' appropriate', 'Write tests for the' + ' above')
$pattern = ($terms | ForEach-Object { [regex]::Escape($_) }) -join '|'
rg -n $pattern pipecat-agent/server/model_eval pipecat-agent/server/tests/test_model_eval_*.py pipecat-agent/server/tests/live_robot_smoke/manual_model_eval.py pipecat-agent/docs/testing.md
```

Expected output: no matches.

- [ ] Commit any final integration fixes with precise staging.

```powershell
git add server/model_eval server/tests/test_model_eval_*.py server/tests/live_robot_smoke/manual_model_eval.py server/evals/model_matrix.example.toml docs/testing.md
git commit -m "Integrate model eval module"
```

Skip this commit if each worker commit already produced a clean, verified result and there are no integration edits.

## Handoff Notes

- Use `server/test_support/live_robot_smoke.py` as the validation seam. Do not duplicate its robot validators unless a test proves the existing shape cannot express an eval scenario.
- Use `server/agent_model_factory.py` for real chat models. Do not construct provider SDK clients directly in `model_eval`.
- Use `server/robot_control/mcp_bridge.py` for live MCP. Do not add another MCP client.
- Keep the simulated adapter deterministic. The model under test may improvise, but the tool environment must stay stable between candidates.
- Keep generated evidence out of commits. Commit source, tests, docs, and the example matrix only.
