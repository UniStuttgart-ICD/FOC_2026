# Manual Live LLM Robot Smoke Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a manual, opt-in live smoke test suite that sends text Agent Turns through the real Codex OAuth backend and MoveIt simulation, records robot tool evidence, and keeps normal CI deterministic.

**Architecture:** Keep the pass/fail pipeline simple: deterministic tests cover the harness; manual live smoke tests live in a non-default pytest file and run only with an explicit environment gate. The live path uses `OpenAICodexAgentProcessor` with a test-only recording wrapper around the real `RobotMCPBridge`, so runtime robot behavior is unchanged while evidence is captured as JSON.

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio, OpenAI Codex OAuth backend, LangGraph-backed `OpenAICodexAgentProcessor`, MoveIt MCP via `RobotMCPBridge`, JSON evidence artifacts.

---

## Scope

This plan implements only **Live LLM Robot Smoke Tests** from `CONTEXT.md`:

- Manual only, never normal CI.
- Text commands through the `AgentTurnInput` / `AgentBackend` seam.
- Real Codex OAuth backend.
- Real MoveIt MCP simulation.
- Minimal JSON evidence.
- Simple pass/fail scenarios:
  - `what is the current position?`
  - `move up a bit`
  - `move down a bit`
  - `move there`

Out of scope for this plan:

- Wake word, STT, TTS, browser audio, or full Pipecat voice e2e.
- Wave/star pass-fail tests.
- Runtime safety or Task Policy changes.
- Production logging hooks in `RobotMCPBridge`.

## File structure

- Create: `server/test_support/__init__.py` — marks the test-support package.
- Create: `server/test_support/live_robot_smoke.py` — recording adapter, live run model, validators, and evidence writer.
- Create: `server/tests/test_live_robot_smoke_support.py` — deterministic tests for the recording adapter and validators.
- Create: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py` — explicit-path manual live pytest suite; not collected by normal `uv run pytest` because the filename does not match `test_*.py`.
- Modify: `server/pyproject.toml` — register pytest markers used by the manual live suite.
- Modify: `.gitignore` — ignore generated live smoke evidence under `server/evidence/live_smoke/`.
- Create: `docs/testing.md` — concise testing pipeline documentation.
- Modify: `README.md` — link to `docs/testing.md`.

---

## Task 1: Add deterministic tests for the live smoke support layer

**Files:**
- Create: `server/tests/test_live_robot_smoke_support.py`

- [ ] **Step 1: Write the failing support tests**

Create `server/tests/test_live_robot_smoke_support.py` with this full content:

```python
import json

import pytest

from test_support.live_robot_smoke import (
    LiveSmokeRun,
    RecordedToolCall,
    RecordingRobotToolAdapter,
    validate_ambiguous_clarification,
    validate_bit_movement,
    validate_position_query,
)


class FakeRobotToolAdapter:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.calls: list[tuple[str, dict]] = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    def function_tools(self) -> list[dict]:
        return [
            {"type": "function", "name": "moveit_get_current_pose", "parameters": {"type": "object"}},
            {
                "type": "function",
                "name": "moveit_plan_and_execute_free_motion",
                "parameters": {"type": "object"},
            },
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return json.dumps({"structured_content": {"ok": True, "tool": name}})


@pytest.mark.asyncio
async def test_recording_adapter_delegates_and_records_json_output() -> None:
    delegate = FakeRobotToolAdapter()
    recorder = RecordingRobotToolAdapter(delegate)

    await recorder.connect()
    output = await recorder.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})
    await recorder.disconnect()

    assert delegate.connected is True
    assert delegate.disconnected is True
    assert output == json.dumps({"structured_content": {"ok": True, "tool": "moveit_get_current_pose"}})
    assert delegate.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert [call.name for call in recorder.calls] == ["moveit_get_current_pose"]
    assert recorder.calls[0].arguments == {"robot_name": "UR10"}
    assert recorder.calls[0].output_json == {
        "structured_content": {"ok": True, "tool": "moveit_get_current_pose"}
    }


def test_position_query_requires_pose_observation_and_no_motion() -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="The current pose is x=0.1, y=0.2, z=0.3.",
        tool_calls=[pose_call(z=0.3)],
    )

    result = validate_position_query(run)

    assert result.passed is True
    assert result.reason == "position query observed current pose without movement"


def test_position_query_rejects_motion_tools() -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="Moved.",
        tool_calls=[pose_call(z=0.3), verified_execution_call()],
    )

    result = validate_position_query(run)

    assert result.passed is False
    assert "unexpected motion tools" in result.reason


def test_move_up_bit_accepts_verified_plus_z_motion() -> None:
    run = LiveSmokeRun(
        prompt="move up a bit",
        reply="Moved up 50 mm.",
        tool_calls=[pose_call(z=0.30), verified_execution_call(), pose_call(z=0.35)],
    )

    result = validate_bit_movement(run, direction="up")

    assert result.passed is True
    assert result.details["delta_z_m"] == pytest.approx(0.05)


def test_move_up_bit_rejects_wrong_direction() -> None:
    run = LiveSmokeRun(
        prompt="move up a bit",
        reply="Moved up 50 mm.",
        tool_calls=[pose_call(z=0.30), verified_execution_call(), pose_call(z=0.25)],
    )

    result = validate_bit_movement(run, direction="up")

    assert result.passed is False
    assert "expected +Z movement" in result.reason


def test_move_down_bit_accepts_verified_minus_z_motion() -> None:
    run = LiveSmokeRun(
        prompt="move down a bit",
        reply="Moved down 50 mm.",
        tool_calls=[pose_call(z=0.35), verified_execution_call(), pose_call(z=0.30)],
    )

    result = validate_bit_movement(run, direction="down")

    assert result.passed is True
    assert result.details["delta_z_m"] == pytest.approx(-0.05)


def test_ambiguous_command_accepts_clarification_without_motion() -> None:
    run = LiveSmokeRun(
        prompt="move there",
        reply="Where would you like me to move?",
        tool_calls=[pose_call(z=0.30)],
    )

    result = validate_ambiguous_clarification(run)

    assert result.passed is True
    assert result.reason == "ambiguous command asked for clarification without movement"


def test_ambiguous_command_rejects_motion_execution() -> None:
    run = LiveSmokeRun(
        prompt="move there",
        reply="I moved there.",
        tool_calls=[pose_call(z=0.30), verified_execution_call()],
    )

    result = validate_ambiguous_clarification(run)

    assert result.passed is False
    assert "unexpected motion tools" in result.reason


def pose_call(*, z: float, x: float = 0.10, y: float = 0.20) -> RecordedToolCall:
    output_json = {
        "structured_content": {
            "ok": True,
            "robot": "UR10",
            "raw": {
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
        }
    }
    return RecordedToolCall(
        name="moveit_get_current_pose",
        arguments={"robot_name": "UR10"},
        output_text=json.dumps(output_json),
        output_json=output_json,
    )


def verified_execution_call() -> RecordedToolCall:
    output_json = {
        "structured_content": {
            "ok": True,
            "verification": {"result": "pass"},
        }
    }
    return RecordedToolCall(
        name="moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "target_pose": {"x": 0.1, "y": 0.2, "z": 0.35}},
        output_text=json.dumps(output_json),
        output_json=output_json,
    )
```

- [ ] **Step 2: Run the support tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_live_robot_smoke_support.py -v
```

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'test_support'`.

---

## Task 2: Implement the live smoke support layer

**Files:**
- Create: `server/test_support/__init__.py`
- Create: `server/test_support/live_robot_smoke.py`
- Test: `server/tests/test_live_robot_smoke_support.py`

- [ ] **Step 1: Create the test-support package marker**

Create `server/test_support/__init__.py` with this content:

```python
"""Support code for tests and manual eval harnesses."""
```

- [ ] **Step 2: Implement recording, validators, and evidence writing**

Create `server/test_support/live_robot_smoke.py` with this full content:

```python
from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from voice_runtime.agent_turn import AgentTurnInput

EXPECTED_BIT_DELTA_M = 0.05
BIT_DELTA_TOLERANCE_M = 0.03
XY_DRIFT_MAX_M = 0.05
DEFAULT_EVIDENCE_DIR = Path("evidence/live_smoke")
MOTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
}
EXECUTION_TOOL_NAMES = {
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
}


class RobotToolAdapterLike(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def function_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class AgentTurnBackendLike(Protocol):
    def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]: ...


@dataclass(frozen=True)
class RecordedToolCall:
    name: str
    arguments: dict[str, Any]
    output_text: str
    output_json: Any

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output_text": self.output_text,
            "output_json": self.output_json,
        }


@dataclass(frozen=True)
class LiveSmokeRun:
    prompt: str
    reply: str
    tool_calls: list[RecordedToolCall]

    def as_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reply": self.reply,
            "tool_calls": [call.as_json() for call in self.tool_calls],
        }


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


class RecordingRobotToolAdapter:
    """Records live robot tool calls while delegating to the real robot tool adapter."""

    def __init__(self, delegate: RobotToolAdapterLike) -> None:
        self._delegate = delegate
        self._calls: list[RecordedToolCall] = []

    @property
    def calls(self) -> list[RecordedToolCall]:
        return list(self._calls)

    def clear(self) -> None:
        self._calls.clear()

    async def connect(self) -> None:
        await self._delegate.connect()

    async def disconnect(self) -> None:
        await self._delegate.disconnect()

    def function_tools(self) -> list[dict[str, Any]]:
        return self._delegate.function_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        output_text = await self._delegate.call_tool(name, arguments)
        self._calls.append(
            RecordedToolCall(
                name=name,
                arguments=dict(arguments),
                output_text=output_text,
                output_json=_json_or_text(output_text),
            )
        )
        return output_text


async def run_agent_turn(
    backend: AgentTurnBackendLike,
    recorder: RecordingRobotToolAdapter,
    prompt: str,
) -> LiveSmokeRun:
    recorder.clear()
    turn = AgentTurnInput(user_text=prompt, messages=[{"role": "user", "content": prompt}])
    chunks = [chunk async for chunk in backend.run_turn(turn)]
    return LiveSmokeRun(prompt=prompt, reply="".join(chunks).strip(), tool_calls=recorder.calls)


def validate_position_query(run: LiveSmokeRun) -> ValidationResult:
    if not _has_call(run.tool_calls, "moveit_get_current_pose"):
        return ValidationResult(False, "position query did not observe current pose")
    unexpected_motion = _called_motion_tools(run.tool_calls)
    if unexpected_motion:
        return ValidationResult(
            False,
            f"position query used unexpected motion tools: {unexpected_motion}",
            {"motion_tools": unexpected_motion},
        )
    return ValidationResult(True, "position query observed current pose without movement")


def validate_bit_movement(
    run: LiveSmokeRun,
    *,
    direction: Literal["up", "down"],
) -> ValidationResult:
    if not _has_call(run.tool_calls, "moveit_get_current_pose"):
        return ValidationResult(False, f"move {direction} did not observe current pose")
    if not _has_verified_execution(run.tool_calls):
        return ValidationResult(False, f"move {direction} did not record verified execution")

    poses = [_pose_position(call) for call in run.tool_calls if call.name == "moveit_get_current_pose"]
    positions = [pose for pose in poses if pose is not None]
    if len(positions) < 2:
        return ValidationResult(False, f"move {direction} did not record start and final poses")

    start = positions[0]
    final = positions[-1]
    delta_x = final["x"] - start["x"]
    delta_y = final["y"] - start["y"]
    delta_z = final["z"] - start["z"]
    expected_sign = 1.0 if direction == "up" else -1.0
    signed_delta_z = expected_sign * delta_z
    details = {
        "start": start,
        "final": final,
        "delta_x_m": delta_x,
        "delta_y_m": delta_y,
        "delta_z_m": delta_z,
        "expected_delta_m": expected_sign * EXPECTED_BIT_DELTA_M,
    }

    lower = EXPECTED_BIT_DELTA_M - BIT_DELTA_TOLERANCE_M
    upper = EXPECTED_BIT_DELTA_M + BIT_DELTA_TOLERANCE_M
    if signed_delta_z < lower or signed_delta_z > upper:
        sign_label = "+Z" if direction == "up" else "-Z"
        return ValidationResult(
            False,
            f"expected {sign_label} movement around {EXPECTED_BIT_DELTA_M} m, got delta_z={delta_z:.4f} m",
            details,
        )
    if abs(delta_x) > XY_DRIFT_MAX_M or abs(delta_y) > XY_DRIFT_MAX_M:
        return ValidationResult(
            False,
            f"movement drifted too far in X/Y: delta_x={delta_x:.4f} m, delta_y={delta_y:.4f} m",
            details,
        )
    return ValidationResult(True, f"move {direction} executed verified bounded movement", details)


def validate_ambiguous_clarification(run: LiveSmokeRun) -> ValidationResult:
    unexpected_motion = _called_motion_tools(run.tool_calls)
    if unexpected_motion:
        return ValidationResult(
            False,
            f"ambiguous command used unexpected motion tools: {unexpected_motion}",
            {"motion_tools": unexpected_motion},
        )
    reply = run.reply.lower()
    clarification_terms = ("?", "where", "which", "clarify", "location", "target")
    if not any(term in reply for term in clarification_terms):
        return ValidationResult(False, "ambiguous command did not ask a recognizable clarification")
    return ValidationResult(True, "ambiguous command asked for clarification without movement")


def write_evidence(
    *,
    evidence_dir: Path,
    case_name: str,
    run: LiveSmokeRun,
    validation: ValidationResult,
) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = evidence_dir / f"{timestamp}-{_slug(case_name)}.json"
    payload = {
        "case": case_name,
        "run": run.as_json(),
        "validator": validation.as_json(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _has_call(calls: list[RecordedToolCall], name: str) -> bool:
    return any(call.name == name for call in calls)


def _called_motion_tools(calls: list[RecordedToolCall]) -> list[str]:
    return [call.name for call in calls if call.name in MOTION_TOOL_NAMES]


def _has_verified_execution(calls: list[RecordedToolCall]) -> bool:
    for call in calls:
        if call.name not in EXECUTION_TOOL_NAMES:
            continue
        structured = _structured_content(call.output_json)
        if not isinstance(structured, dict) or structured.get("ok") is not True:
            continue
        verification = structured.get("verification")
        if isinstance(verification, dict) and verification.get("result") == "pass":
            return True
        execution = structured.get("execution")
        if isinstance(execution, dict) and execution.get("verification_result") == "pass":
            return True
    return False


def _pose_position(call: RecordedToolCall) -> dict[str, float] | None:
    structured = _structured_content(call.output_json)
    if not isinstance(structured, dict):
        return None
    raw = structured.get("raw")
    if not isinstance(raw, dict):
        return None
    pose = raw.get("pose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    if not isinstance(position, dict):
        return None
    try:
        return {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _structured_content(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    return value.get("structured_content")


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "case"
```

- [ ] **Step 3: Run the support tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_live_robot_smoke_support.py -v
```

Expected: all tests in `tests/test_live_robot_smoke_support.py` pass.

- [ ] **Step 4: Commit Task 2**

Run from the repository root:

```bash
git add server/test_support/__init__.py server/test_support/live_robot_smoke.py server/tests/test_live_robot_smoke_support.py
git commit -m "test: add live robot smoke support"
```

---

## Task 3: Add the explicit-path manual live smoke suite

**Files:**
- Create: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- Modify: `server/pyproject.toml`
- Test: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`

- [ ] **Step 1: Register pytest markers**

In `server/pyproject.toml`, replace this block:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
```

with this block:

```toml
[tool.pytest.ini_options]
pythonpath = ["."]
markers = [
    "live: manual tests that require live credentials or external services",
    "llm: manual tests that call the real Codex OAuth backend",
    "robot_sim: manual tests that require the MoveIt robot simulation stack",
]
```

- [ ] **Step 2: Create the manual live smoke test file**

Create directory `server/tests/live_robot_smoke/` and create `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py` with this full content:

```python
from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_asyncio

from openai_codex_agent_processor import OpenAICodexAgentProcessor
from robot_mcp_bridge import RobotMCPBridge
from test_support.live_robot_smoke import (
    DEFAULT_EVIDENCE_DIR,
    LiveSmokeRun,
    RecordingRobotToolAdapter,
    ValidationResult,
    run_agent_turn,
    validate_ambiguous_clarification,
    validate_bit_movement,
    validate_position_query,
    write_evidence,
)

RUN_ENV = "RUN_LIVE_LLM_ROBOT_SMOKE"
MCP_URL_ENV = "LIVE_LLM_ROBOT_MCP_URL"
MODEL_ENV = "LIVE_LLM_ROBOT_MODEL"
EVIDENCE_DIR_ENV = "LIVE_LLM_ROBOT_EVIDENCE_DIR"
DEFAULT_MCP_URL = "http://127.0.0.1:8765/mcp"
DEFAULT_MODEL = "gpt-5.4-mini"

if os.getenv(RUN_ENV) != "1":
    pytest.skip(
        f"manual live robot smoke tests require {RUN_ENV}=1, Codex OAuth login, and MoveIt MCP",
        allow_module_level=True,
    )

pytestmark = [pytest.mark.asyncio, pytest.mark.live, pytest.mark.llm, pytest.mark.robot_sim]


@pytest_asyncio.fixture
async def live_agent() -> tuple[OpenAICodexAgentProcessor, RecordingRobotToolAdapter]:
    mcp_url = os.getenv(MCP_URL_ENV, DEFAULT_MCP_URL)
    model = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    recorder = RecordingRobotToolAdapter(RobotMCPBridge(mcp_url))
    processor = OpenAICodexAgentProcessor(
        mcp_url,
        model=model,
        tool_bridge=recorder,
    )
    try:
        yield processor, recorder
    finally:
        await processor.disconnect()


async def test_manual_live_llm_robot_smoke_suite(
    live_agent: tuple[OpenAICodexAgentProcessor, RecordingRobotToolAdapter],
) -> None:
    processor, recorder = live_agent
    evidence_dir = Path(os.getenv(EVIDENCE_DIR_ENV, str(DEFAULT_EVIDENCE_DIR)))
    cases: list[tuple[str, str, Callable[[LiveSmokeRun], ValidationResult]]] = [
        ("current-position", "what is the current position?", validate_position_query),
        ("move-up-bit", "move up a bit", lambda run: validate_bit_movement(run, direction="up")),
        ("move-down-bit", "move down a bit", lambda run: validate_bit_movement(run, direction="down")),
        ("ambiguous-move-there", "move there", validate_ambiguous_clarification),
    ]

    failures: list[str] = []
    for case_name, prompt, validator in cases:
        try:
            run = await run_agent_turn(processor, recorder, prompt)
            validation = validator(run)
        except Exception as exc:
            run = LiveSmokeRun(prompt=prompt, reply="", tool_calls=recorder.calls)
            validation = ValidationResult(False, f"case raised {type(exc).__name__}: {exc}")
        evidence_path = write_evidence(
            evidence_dir=evidence_dir,
            case_name=case_name,
            run=run,
            validation=validation,
        )
        if not validation.passed:
            failures.append(f"{case_name}: {validation.reason}; evidence={evidence_path}")

    assert not failures, "\n".join(failures)
```

- [ ] **Step 3: Verify explicit run without the live gate skips**

Run from `server/`:

```bash
uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -q
```

Expected: `1 skipped` and no Codex or MCP network calls.

- [ ] **Step 4: Verify normal pytest does not collect the manual live file**

Run from `server/`:

```bash
uv run pytest --collect-only -q | rg "manual_live_llm_robot_smoke"
```

Expected: `rg` exits with code 1 and prints no collected test path for `manual_live_llm_robot_smoke.py`.

- [ ] **Step 5: Commit Task 3**

Run from the repository root:

```bash
git add server/pyproject.toml server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py
git commit -m "test: add manual live LLM robot smoke suite"
```

---

## Task 4: Ignore generated evidence and document the manual workflow

**Files:**
- Modify: `.gitignore`
- Create: `docs/testing.md`
- Modify: `README.md`

- [ ] **Step 1: Ignore generated live smoke evidence**

In `.gitignore`, replace this block:

```gitignore
# Runtime voice metrics
server/logs/
```

with this block:

```gitignore
# Runtime voice metrics
server/logs/

# Manual live smoke evidence
server/evidence/live_smoke/
```

- [ ] **Step 2: Create testing documentation**

Create `docs/testing.md` with this full content:

```markdown
# Testing

## Default tests

Run deterministic tests from `server/`:

```bash
uv run pytest
```

Default tests must not require Codex OAuth, MoveIt MCP, STT/TTS providers, wake-word models beyond existing unit-test fixtures, browser audio, or robot simulation infrastructure.

## Manual live LLM robot smoke tests

Manual live smoke tests send text through the Agent Turn seam:

```text
AgentTurnInput -> OpenAICodexAgentProcessor -> Codex OAuth backend -> RobotMCPBridge -> MoveIt simulation
```

They do not exercise wake, STT, TTS, browser audio, or the full Pipecat voice pipeline.

### Prerequisites

- Pi is logged in with the `openai-codex` OAuth profile.
- The MoveIt MCP server is reachable.
- The UR10 simulation is running in safe simulation mode.

### Run

From `server/`:

```bash
RUN_LIVE_LLM_ROBOT_SMOKE=1 uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -v
```

Optional overrides:

```bash
LIVE_LLM_ROBOT_MCP_URL=http://127.0.0.1:8765/mcp
LIVE_LLM_ROBOT_MODEL=gpt-5.4-mini
LIVE_LLM_ROBOT_EVIDENCE_DIR=evidence/live_smoke
```

### Scenarios

The v1 smoke suite covers:

1. `what is the current position?` — observes pose and does not move.
2. `move up a bit` — observes pose, executes verified bounded +Z movement.
3. `move down a bit` — observes pose, executes verified bounded -Z movement.
4. `move there` — asks for clarification and does not move.

Each case writes minimal JSON evidence under `server/evidence/live_smoke/` by default.

## Exploratory gesture evals

Prompts such as `wave to me` and `draw a star` are exploratory evals. They are useful for behavior review, but they are not part of the pass/fail testing pipeline until their assertions become deterministic and actionable.
```

- [ ] **Step 3: Link testing docs from README**

In `README.md`, replace this section:

```markdown
## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)
```

with this section:

```markdown
## Testing

See [Testing](docs/testing.md) for deterministic tests and manual live LLM robot smoke tests.

## Learn More

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat GitHub](https://github.com/pipecat-ai/pipecat)
- [Pipecat Examples](https://github.com/pipecat-ai/pipecat-examples)
- [Discord Community](https://discord.gg/pipecat)
```

- [ ] **Step 4: Run deterministic docs-related checks**

Run from `server/`:

```bash
uv run pytest tests/test_live_robot_smoke_support.py -v
uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -q
```

Expected:

- Support tests pass.
- Manual live smoke file reports `1 skipped` without `RUN_LIVE_LLM_ROBOT_SMOKE=1`.

- [ ] **Step 5: Commit Task 4**

Run from the repository root:

```bash
git add .gitignore README.md docs/testing.md
git commit -m "docs: document manual live robot smoke tests"
```

---

## Task 5: Final verification and manual live acceptance

**Files:**
- Verify only; no source edits expected.

- [ ] **Step 1: Run the full deterministic test suite**

Run from `server/`:

```bash
uv run pytest -q
```

Expected: all deterministic tests pass. The manual live smoke file is not collected by the default command.

- [ ] **Step 2: Run lint**

Run from `server/`:

```bash
uv run ruff check .
```

Expected: `All checks passed!`.

- [ ] **Step 3: Run type checks**

Run from `server/`:

```bash
uv run pyright .
```

Expected: `0 errors`.

- [ ] **Step 4: Verify the manual live suite skips without the gate**

Run from `server/`:

```bash
uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -q
```

Expected: `1 skipped`.

- [ ] **Step 5: Run the manual live smoke suite with prepared live prerequisites**

Run from `server/` with Pi Codex OAuth login, MoveIt MCP, and UR10 simulation ready:

```bash
RUN_LIVE_LLM_ROBOT_SMOKE=1 uv run pytest tests/live_robot_smoke/manual_live_llm_robot_smoke.py -v
```

Expected: the single manual smoke suite test passes and writes four JSON evidence files under `server/evidence/live_smoke/`.

- [ ] **Step 6: Inspect evidence files**

Run from `server/`:

```bash
find evidence/live_smoke -type f -name '*.json' -print | sort | tail -4
```

Expected: four recent JSON files, one per smoke scenario.

- [ ] **Step 7: Confirm generated evidence is ignored**

Run from the repository root:

```bash
git status --short --ignored server/evidence/live_smoke
```

Expected: generated files under `server/evidence/live_smoke/` appear as ignored (`!!`) and are not staged.

- [ ] **Step 8: Commit any verification-only documentation fixes**

If verification reveals a typo in docs or command text, fix it and commit with:

```bash
git add README.md docs/testing.md .gitignore server/pyproject.toml server/test_support server/tests
git commit -m "docs: clarify live robot smoke workflow"
```

Expected: no commit is created when there are no documentation fixes.

---

## Self-review

**Spec coverage:**

- Manual-only gate: Task 3 explicit env gate and non-default filename.
- Agent Turn seam: Task 3 constructs `OpenAICodexAgentProcessor` and calls `run_agent_turn()` with `AgentTurnInput`.
- Real Codex + real MoveIt MCP: Task 3 uses default processor backend and real `RobotMCPBridge`.
- Recording wrapper: Task 2 implements `RecordingRobotToolAdapter` and Task 1 tests it.
- Minimal JSON evidence: Task 2 implements `write_evidence()` and Task 3 writes evidence per case.
- V1 scenarios: Task 3 lists all four agreed prompts and validators.
- Wave/star excluded from pass/fail: Task 4 documents exploratory gesture evals as outside the pipeline.

**Placeholder scan:**

- No placeholder markers or open-ended implementation steps remain.
- Code blocks define every helper used by later tasks.

**Type consistency:**

- `LiveSmokeRun`, `RecordedToolCall`, `RecordingRobotToolAdapter`, and `ValidationResult` are defined in Task 2 and imported consistently in Tasks 1 and 3.
- Validator names match across support tests, support implementation, and manual live test file.
