# Minimal Task Policy Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a minimal deterministic **Task Policy Layer** that blocks obvious robot-step precondition failures before individual MoveIt tool calls reach **Robot Call Validation** and MoveIt MCP.

**Architecture:** Keep Codex/LangGraph responsible for open-ended intent and task planning. Add `server/robot_control/task_policy.py` as a pure Robot Control module. The policy runs before the existing Robot Call Validation/MCP path: fresh pose before motion, no blind `moveit_execute_plan`, and no `moveit_attach_object` unless the gripper is recently known closed. Keep `voice_runtime.robot_safety` as the current legacy placement for Robot Call Validation until the broader `robot_control` extraction.

**Tech Stack:** Python 3.12, Pipecat, LangGraph, pytest, ruff, pyright.

---

## Architecture constraints

- Follow `ARCHITECTURE.md` and `CONTEXT.md` terminology.
- Do not put Task Policy in `voice_runtime/`.
- Create `server/robot_control/task_policy.py`.
- `robot_control.task_policy` must not import `voice_runtime`, Pipecat, MCP, Codex, LangGraph, or app composition modules.
- Use a Protocol for the policy-readable context so the current legacy `RobotContextStore` can satisfy it structurally.
- Do not describe this as movement safety. Movement safety is delegated to MoveIt planning/execution and the robot simulation stack.
- Task Policy v1 checks only obvious pre-tool preconditions. It does not prove semantic task safety, object/world state, arbitrary pick/place workflows, or emergency stop.

## Scope

Implement only these generic policies:

1. **Fresh observation before motion**: motion/planning/execution tools require a recent successful `moveit_get_current_pose` observation.
2. **No blind execute**: `moveit_execute_plan` requires a recently recorded executable `plan_name` from a successful planning result.
3. **Basic attach ordering**: `moveit_attach_object` requires a non-empty `object_name` and gripper state recently known as `closed`.

Explicitly out of scope:

- object perception
- full pick/place workflows
- proving “is holding the thing”
- arbitrary semantic task safety
- emergency stop
- policy per possible LLM task
- changing the browser/Pipecat audio pipeline
- extracting all Robot Control modules in this plan
- moving `voice_runtime.robot_safety` or `voice_runtime.robot_context` in this plan

## Files and responsibilities

- Modify: `server/voice_runtime/robot_context.py`
  - Add policy-readable state APIs for recent observation, executable plans, and recent gripper state.
  - Keep advisory instruction rendering unchanged.
  - This remains a legacy placement until the broader Robot Control extraction.
- Modify: `server/tests/test_robot_context.py`
  - Add tests for recent observation, executable plan memory, and gripper state timestamps.
- Create: `server/robot_control/__init__.py`
  - Introduce the Robot Control package.
- Create: `server/robot_control/task_policy.py`
  - Add deterministic task precondition checks and structured error serialization.
- Create: `server/tests/test_robot_task_policy.py`
  - Unit-test policy decisions without Pipecat, MCP, Codex, or `voice_runtime` imports.
- Create/modify: `server/tests/test_robot_control_imports.py`
  - Enforce that pure Robot Control policy code does not import Voice Runtime or app adapters.
- Modify: `server/langgraph_robot_agent.py`
  - Run Task Policy before `RobotMCPBridge.call_tool()`.
  - Record executable plans before auto-executing them.
  - Route auto-execution through the same policy-checked helper.
- Modify: `server/tests/test_langgraph_robot_agent.py`
  - Add integration tests proving policy failures become tool outputs sent back to Codex.
- Optional docs: `CONTEXT.md`
  - Update only if implementation changes settle new terms. Current context already defines **Task Policy Layer** and **Task Policy Decision**.

---

## Task 1: Extend RobotContextStore with policy-readable state

**Files:**
- Modify: `server/voice_runtime/robot_context.py`
- Modify: `server/tests/test_robot_context.py`

- [ ] **Step 1: Add failing context tests**

Append to `server/tests/test_robot_context.py`:

```python
def test_robot_context_reports_recent_and_stale_pose_observations() -> None:
    now = 100.0
    store = RobotContextStore(time_fn=lambda: now)

    assert store.has_recent_robot_observation(max_age_s=15.0) is False

    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
            }
        }
    )
    store.update_from_tool_result("moveit_get_current_pose", output)

    assert store.has_recent_robot_observation(max_age_s=15.0) is True

    now = 116.0
    assert store.has_recent_robot_observation(max_age_s=15.0) is False


def test_robot_context_remembers_recent_executable_plan_names() -> None:
    now = 200.0
    store = RobotContextStore(time_fn=lambda: now)

    store.remember_executable_plan("plan-1")

    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is True
    assert store.has_recent_executable_plan("missing", max_age_s=60.0) is False

    now = 261.0
    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is False


def test_robot_context_tracks_recent_gripper_state_from_gripper_tools() -> None:
    now = 300.0
    store = RobotContextStore(time_fn=lambda: now)
    ok_output = json.dumps({"structured_content": {"ok": True}})

    assert store.gripper_state() is None
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is False

    store.update_from_tool_result("moveit_close_gripper", ok_output)
    assert store.gripper_state() == "closed"
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is True

    now = 331.0
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is False

    store.update_from_tool_result("moveit_open_gripper", ok_output)
    assert store.gripper_state() == "open"
    assert store.has_recent_gripper_state("open", max_age_s=30.0) is True
```

- [ ] **Step 2: Run the new tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py::test_robot_context_reports_recent_and_stale_pose_observations tests/test_robot_context.py::test_robot_context_remembers_recent_executable_plan_names tests/test_robot_context.py::test_robot_context_tracks_recent_gripper_state_from_gripper_tools -v
```

Expected: failures because `RobotContextStore` does not yet accept `time_fn` or expose the new methods.

- [ ] **Step 3: Implement the minimal context APIs**

In `server/voice_runtime/robot_context.py`, update imports:

```python
from collections.abc import Callable
from dataclasses import dataclass, field
```

Update `RobotContextSnapshot`:

```python
@dataclass
class RobotContextSnapshot:
    observed_at_s: float | None = None
    robot_name: str | None = None
    tcp_pose: dict[str, Any] | None = None
    gripper_state: str | None = None
    gripper_observed_at_s: float | None = None
    last_execution_result: str | None = None
    executable_plan_observed_at_s: dict[str, float] = field(default_factory=dict)
```

Update `RobotContextStore.__init__` and add policy-readable methods:

```python
class RobotContextStore:
    def __init__(self, *, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._snapshot = RobotContextSnapshot()
        self._time_fn = time_fn

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        observed_at_s = self._snapshot.observed_at_s
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s

    def remember_executable_plan(self, plan_name: str) -> None:
        if plan_name:
            self._snapshot.executable_plan_observed_at_s[plan_name] = self._time_fn()

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
        observed_at_s = self._snapshot.executable_plan_observed_at_s.get(plan_name)
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s

    def gripper_state(self) -> str | None:
        return self._snapshot.gripper_state

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool:
        if self._snapshot.gripper_state != state:
            return False
        observed_at_s = self._snapshot.gripper_observed_at_s
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s
```

In `update_from_tool_result`, replace `time.monotonic()` with `self._time_fn()`:

```python
self._snapshot.observed_at_s = self._time_fn()
```

Also update gripper state after confirming `structured_content.ok is True`:

```python
        if tool_name == "moveit_close_gripper":
            self._snapshot.gripper_state = "closed"
            self._snapshot.gripper_observed_at_s = self._time_fn()
            return
        if tool_name == "moveit_open_gripper":
            self._snapshot.gripper_state = "open"
            self._snapshot.gripper_observed_at_s = self._time_fn()
            return
```

When parsing a gripper state from an observation/status result, also set `gripper_observed_at_s` to `self._snapshot.observed_at_s`.

Keep existing parsing for `moveit_get_current_pose` and legacy `moveit_get_robot_status` intact.

- [ ] **Step 4: Run context tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py -v
```

Expected: all robot context tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add server/voice_runtime/robot_context.py server/tests/test_robot_context.py
git commit -m "feat: expose robot context for task policy"
```

---

## Task 2: Add pure Robot Control task policy module

**Files:**
- Create: `server/robot_control/__init__.py`
- Create: `server/robot_control/task_policy.py`
- Create: `server/tests/test_robot_task_policy.py`
- Create/modify: `server/tests/test_robot_control_imports.py`

- [ ] **Step 1: Write failing policy tests**

Create `server/tests/test_robot_task_policy.py`:

```python
from dataclasses import dataclass, field

from robot_control.task_policy import (
    DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    DEFAULT_GRIPPER_STATE_MAX_AGE_S,
    TaskPolicyDecision,
    structured_task_policy_error,
    validate_task_step,
)

VALID_TARGET_POSE = {
    "position": {"x": 0.1, "y": 0.2, "z": 0.3},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


@dataclass
class FakeTaskPolicyContext:
    recent_pose: bool = False
    executable_plans: set[str] = field(default_factory=set)
    gripper: str | None = None
    recent_gripper: bool = False

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        return self.recent_pose

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
        return plan_name in self.executable_plans

    def gripper_state(self) -> str | None:
        return self.gripper

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool:
        return self.gripper == state and self.recent_gripper


def test_policy_allows_observation_without_existing_context() -> None:
    decision = validate_task_step(
        "moveit_get_current_pose",
        {"robot_name": "UR10"},
        FakeTaskPolicyContext(),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_motion_without_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(),
    )

    assert decision.ok is False
    assert decision.error == "Fresh robot pose is required before motion."
    assert decision.correction == "Call moveit_get_current_pose, then retry the motion."
    assert decision.suggested_next_tool == "moveit_get_current_pose"


def test_policy_allows_motion_after_recent_pose_observation() -> None:
    decision = validate_task_step(
        "moveit_plan_free_motion",
        {"robot_name": "UR10", "target_pose": VALID_TARGET_POSE},
        FakeTaskPolicyContext(recent_pose=True),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_uses_configured_pose_freshness_window() -> None:
    context = FakeTaskPolicyContext(recent_pose=True)

    decision = validate_task_step(
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "waypoints": [VALID_TARGET_POSE]},
        context,
        fresh_observation_max_age_s=DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    )

    assert decision.ok is True


def test_policy_rejects_execute_plan_when_plan_was_not_returned_by_planning() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "invented-plan"},
        FakeTaskPolicyContext(recent_pose=True),
    )

    assert decision.ok is False
    assert decision.error == "Cannot execute an unknown or stale plan."
    assert decision.correction == "Plan first, then execute the returned plan_name."
    assert decision.suggested_next_tool == "moveit_plan_free_motion"


def test_policy_allows_execute_plan_when_plan_was_recently_recorded() -> None:
    decision = validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        FakeTaskPolicyContext(recent_pose=True, executable_plans={"plan-1"}),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_passes_executable_plan_freshness_window_to_context() -> None:
    class CapturingContext(FakeTaskPolicyContext):
        seen_max_age_s: float | None = None

        def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
            self.seen_max_age_s = max_age_s
            return super().has_recent_executable_plan(plan_name, max_age_s=max_age_s)

    context = CapturingContext(recent_pose=True, executable_plans={"plan-1"})

    validate_task_step(
        "moveit_execute_plan",
        {"robot_name": "UR10", "plan_name": "plan-1"},
        context,
        executable_plan_max_age_s=DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    )

    assert context.seen_max_age_s == DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S


def test_policy_rejects_attach_when_gripper_state_is_unknown() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach object before the gripper is known closed."
    assert decision.correction == "Close the gripper or observe gripper state before attaching."
    assert decision.suggested_next_tool == "moveit_close_gripper"


def test_policy_rejects_attach_when_gripper_state_is_stale() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=False),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach object before the gripper is known closed."


def test_policy_allows_attach_when_gripper_is_recently_closed() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": "cube"},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=True),
    )

    assert decision == TaskPolicyDecision(ok=True)


def test_policy_rejects_attach_without_object_name() -> None:
    decision = validate_task_step(
        "moveit_attach_object",
        {"robot_name": "UR10", "object_name": ""},
        FakeTaskPolicyContext(gripper="closed", recent_gripper=True),
    )

    assert decision.ok is False
    assert decision.error == "Cannot attach an unnamed object."
    assert decision.correction == "Retry with the object_name to attach."
    assert decision.suggested_next_tool is None


def test_structured_task_policy_error_shape() -> None:
    payload = structured_task_policy_error(
        TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before motion.",
            correction="Call moveit_get_current_pose, then retry the motion.",
            suggested_next_tool="moveit_get_current_pose",
        )
    )

    assert payload == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }
```

- [ ] **Step 2: Add failing import-direction test**

Create `server/tests/test_robot_control_imports.py`:

```python
import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
ROBOT_CONTROL_DIR = SERVER_DIR / "robot_control"

PURE_ROBOT_CONTROL_MODULES = {"task_policy.py"}
PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS = {
    "agent_control",
    "agents",
    "langgraph",
    "mcp",
    "openai",
    "pipecat",
    "voice_runtime",
}


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])
    return roots


def test_pure_robot_control_modules_do_not_import_voice_runtime_or_adapters() -> None:
    for name in PURE_ROBOT_CONTROL_MODULES:
        path = ROBOT_CONTROL_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports forbidden module(s): {sorted(forbidden)}"
```

- [ ] **Step 3: Run policy/import tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_robot_task_policy.py tests/test_robot_control_imports.py -v
```

Expected: import failures because `robot_control.task_policy` does not exist.

- [ ] **Step 4: Implement `robot_control.task_policy`**

Create `server/robot_control/__init__.py`:

```python
"""Robot Control modules for policy, validation, context, and tool adapters."""
```

Create `server/robot_control/task_policy.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

DEFAULT_FRESH_OBSERVATION_MAX_AGE_S = 15.0
DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S = 120.0
DEFAULT_GRIPPER_STATE_MAX_AGE_S = 30.0

MOTION_TOOL_NAMES = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_and_execute_free_motion",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_execute_plan",
    }
)


class TaskPolicyContext(Protocol):
    def has_recent_robot_observation(self, *, max_age_s: float) -> bool: ...

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool: ...

    def gripper_state(self) -> str | None: ...

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool: ...


@dataclass(frozen=True)
class TaskPolicyDecision:
    ok: bool
    error: str | None = None
    correction: str | None = None
    retryable: bool = True
    suggested_next_tool: str | None = None


def validate_task_step(
    name: str,
    arguments: dict[str, Any],
    context: TaskPolicyContext,
    *,
    fresh_observation_max_age_s: float = DEFAULT_FRESH_OBSERVATION_MAX_AGE_S,
    executable_plan_max_age_s: float = DEFAULT_EXECUTABLE_PLAN_MAX_AGE_S,
    gripper_state_max_age_s: float = DEFAULT_GRIPPER_STATE_MAX_AGE_S,
) -> TaskPolicyDecision:
    if name in MOTION_TOOL_NAMES and not context.has_recent_robot_observation(
        max_age_s=fresh_observation_max_age_s
    ):
        return TaskPolicyDecision(
            ok=False,
            error="Fresh robot pose is required before motion.",
            correction="Call moveit_get_current_pose, then retry the motion.",
            suggested_next_tool="moveit_get_current_pose",
        )

    if name == "moveit_execute_plan":
        plan_name = arguments.get("plan_name")
        if not isinstance(plan_name, str) or not context.has_recent_executable_plan(
            plan_name,
            max_age_s=executable_plan_max_age_s,
        ):
            return TaskPolicyDecision(
                ok=False,
                error="Cannot execute an unknown or stale plan.",
                correction="Plan first, then execute the returned plan_name.",
                suggested_next_tool="moveit_plan_free_motion",
            )

    if name == "moveit_attach_object":
        object_name = arguments.get("object_name")
        if not isinstance(object_name, str) or not object_name.strip():
            return TaskPolicyDecision(
                ok=False,
                error="Cannot attach an unnamed object.",
                correction="Retry with the object_name to attach.",
                suggested_next_tool=None,
            )
        if not context.has_recent_gripper_state("closed", max_age_s=gripper_state_max_age_s):
            return TaskPolicyDecision(
                ok=False,
                error="Cannot attach object before the gripper is known closed.",
                correction="Close the gripper or observe gripper state before attaching.",
                suggested_next_tool="moveit_close_gripper",
            )

    return TaskPolicyDecision(ok=True)


def structured_task_policy_error(decision: TaskPolicyDecision) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": decision.error or "Task policy rejected the robot step.",
        "correction": decision.correction or "Revise the robot step and retry.",
        "retryable": decision.retryable,
    }
    if decision.suggested_next_tool is not None:
        payload["suggested_next_tool"] = decision.suggested_next_tool
    return payload
```

- [ ] **Step 5: Run policy/import tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_task_policy.py tests/test_robot_control_imports.py -v
```

Expected: all task policy and import tests pass.

- [ ] **Step 6: Run pure-module quality checks**

Run from `server/`:

```bash
uv run ruff check robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py
uv run pyright robot_control/task_policy.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py
```

Expected: ruff passes and pyright reports 0 errors.

- [ ] **Step 7: Commit Task 2**

```bash
git add server/robot_control server/tests/test_robot_task_policy.py server/tests/test_robot_control_imports.py
git commit -m "feat: add minimal robot task policy"
```

---

## Task 3: Enforce task policy inside LangGraph robot tool execution

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Add failing LangGraph policy integration tests**

Append to `server/tests/test_langgraph_robot_agent.py`:

```python
@pytest.mark.asyncio
async def test_graph_sends_policy_failure_as_tool_output_when_motion_lacks_fresh_observation() -> None:
    class NoObservationBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_plan_free_motion",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            return json.dumps({"structured_content": {"ok": True}})

    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    tool = tool_call("moveit_plan_free_motion", arguments=plan_args)
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="I need a fresh pose before moving."),
        ],
        bridge=NoObservationBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a fresh pose before moving."
    assert fixture.bridge.calls == []
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_blocks_blind_execute_plan_even_after_fresh_pose() -> None:
    execute = tool_call(
        "moveit_execute_plan",
        arguments={"robot_name": "UR10", "plan_name": "invented-plan"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[execute],
                output_items=[output_item("moveit_execute_plan", arguments=execute.arguments)],
            ),
            CodexResponseResult(text="I need to plan before executing."),
        ]
    )

    text = await fixture.graph.run_turn(turn("execute the last plan"))

    assert text == "I need to plan before executing."
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output == {
        "ok": False,
        "error": "Cannot execute an unknown or stale plan.",
        "correction": "Plan first, then execute the returned plan_name.",
        "retryable": True,
        "suggested_next_tool": "moveit_plan_free_motion",
    }


@pytest.mark.asyncio
async def test_graph_allows_auto_execute_after_recording_executable_plan() -> None:
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    plan = tool_call("moveit_plan_free_motion", arguments=plan_args)
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[plan],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "plan-1"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
```

- [ ] **Step 2: Run new integration tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_sends_policy_failure_as_tool_output_when_motion_lacks_fresh_observation tests/test_langgraph_robot_agent.py::test_graph_blocks_blind_execute_plan_even_after_fresh_pose tests/test_langgraph_robot_agent.py::test_graph_allows_auto_execute_after_recording_executable_plan -v
```

Expected: at least the policy-failure tests fail because LangGraph does not yet call Task Policy.

- [ ] **Step 3: Import Task Policy into LangGraph**

In `server/langgraph_robot_agent.py`, add imports:

```python
from robot_control.task_policy import structured_task_policy_error, validate_task_step
```

- [ ] **Step 4: Add a policy-checked bridge helper**

Inside `LangGraphRobotAgent`, add:

```python
    async def _call_policy_checked_tool(self, name: str, arguments: dict[str, Any]) -> str:
        decision = validate_task_step(name, arguments, self._robot_context)
        if not decision.ok:
            return json.dumps(structured_task_policy_error(decision), ensure_ascii=False)
        output = await self._tool_bridge.call_tool(name, arguments)
        self._robot_context.update_from_tool_result(name, output)
        return output
```

- [ ] **Step 5: Route `_execute_tool` through Task Policy**

Replace the first two lines inside `_execute_tool`'s `try` block:

```python
            output = await self._tool_bridge.call_tool(name, arguments)
            self._robot_context.update_from_tool_result(name, output)
```

with:

```python
            output = await self._call_policy_checked_tool(name, arguments)
```

Then, after `plan_name = executable_plan_name(output)`, record valid plan names before auto-execution:

```python
            if name in PLAN_TOOL_NAMES and plan_name:
                self._robot_context.remember_executable_plan(plan_name)
                execution_output = await self._call_policy_checked_tool(
                    "moveit_execute_plan",
                    {"robot_name": VIZOR_ROBOT_NAME, "plan_name": plan_name},
                )
```

Do not call `self._tool_bridge.call_tool("moveit_execute_plan", ...)` directly in auto-execution anymore.

- [ ] **Step 6: Run LangGraph tests and verify policy behavior**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py -v
```

Expected: all LangGraph tests pass.

- [ ] **Step 7: Commit Task 3**

```bash
git add server/langgraph_robot_agent.py server/tests/test_langgraph_robot_agent.py
git commit -m "feat: enforce task policy before robot tools"
```

---

## Task 4: Add attach-ordering integration coverage

**Files:**
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Add failing attach ordering integration tests**

Append to `server/tests/test_langgraph_robot_agent.py`:

```python
@pytest.mark.asyncio
async def test_graph_blocks_attach_before_gripper_is_closed() -> None:
    class AttachBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_get_current_pose",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_attach_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    attach = tool_call(
        "moveit_attach_object",
        arguments={"robot_name": "UR10", "object_name": "cube"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[attach],
                output_items=[output_item("moveit_attach_object", arguments=attach.arguments)],
            ),
            CodexResponseResult(text="I need to close the gripper before attaching."),
        ],
        bridge=AttachBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "I need to close the gripper before attaching."
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output == {
        "ok": False,
        "error": "Cannot attach object before the gripper is known closed.",
        "correction": "Close the gripper or observe gripper state before attaching.",
        "retryable": True,
        "suggested_next_tool": "moveit_close_gripper",
    }


@pytest.mark.asyncio
async def test_graph_allows_attach_after_close_gripper_tool_result() -> None:
    class GripperBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_get_current_pose",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_close_gripper",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_attach_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            return json.dumps({"structured_content": {"ok": True}})

    close = tool_call("moveit_close_gripper", call_id="call-1", arguments={"robot_name": "UR10"})
    attach = tool_call(
        "moveit_attach_object",
        call_id="call-2",
        arguments={"robot_name": "UR10", "object_name": "cube"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[close, attach],
                output_items=[
                    output_item("moveit_close_gripper", call_id="call-1", arguments=close.arguments),
                    output_item("moveit_attach_object", call_id="call-2", arguments=attach.arguments),
                ],
            ),
            CodexResponseResult(text="Attached the cube."),
        ],
        bridge=GripperBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "Attached the cube."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_close_gripper", {"robot_name": "UR10"}),
        ("moveit_attach_object", {"robot_name": "UR10", "object_name": "cube"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
```

- [ ] **Step 2: Run attach tests and verify they fail before Task 3 implementation or pass after Task 3**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py::test_graph_blocks_attach_before_gripper_is_closed tests/test_langgraph_robot_agent.py::test_graph_allows_attach_after_close_gripper_tool_result -v
```

Expected after Task 3: both tests pass. If either fails, fix only the policy integration or gripper context update needed for these tests.

- [ ] **Step 3: Commit Task 4**

```bash
git add server/tests/test_langgraph_robot_agent.py
git commit -m "test: cover task policy attach ordering"
```

---

## Task 5: Update domain docs and verify full suite

**Files:**
- Optional modify: `CONTEXT.md`
- Verify: all modified code/tests

- [ ] **Step 1: Verify glossary terms are still aligned**

Confirm `CONTEXT.md` contains these terms and relationships:

- **Task Policy Layer**
- **Task Policy Decision**
- **Robot Call Validation**
- **MoveIt Safety Boundary**
- **Task Policy Layer** runs before **Robot Call Validation**
- **Task Policy Decision** returns structured tool feedback, not a movement-safety claim

Only edit `CONTEXT.md` if the implementation settles a new term or changes one of these relationships.

- [ ] **Step 2: Run targeted tests**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_robot_mcp_bridge.py tests/test_voice_runtime_robot_safety.py -v
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run full verification**

Run from `server/`:

```bash
uv run pytest -q
uv run ruff check .
uv run pyright .
```

Expected: pytest reports all tests passed, ruff reports `All checks passed!`, and pyright reports `0 errors`.

- [ ] **Step 4: Review final diff for scope**

Run from repo root:

```bash
git diff --stat
git diff -- server/robot_control/task_policy.py server/voice_runtime/robot_context.py server/langgraph_robot_agent.py CONTEXT.md
```

Expected:

- No Pipecat pipeline ordering changes.
- No new provider dependencies.
- No changes to STT/TTS/wake behavior.
- `robot_control.task_policy` is pure and does not import `voice_runtime`.
- `voice_runtime.robot_safety` remains Robot Call Validation in its legacy placement.
- `task_policy` contains only generic precondition checks from this plan.

- [ ] **Step 5: Commit final docs if changed**

```bash
git add CONTEXT.md
git commit -m "docs: align task policy glossary"
```

Skip this commit if `CONTEXT.md` did not change.

---

## Completion checklist

- [ ] `RobotContextStore` exposes recent pose, executable plan memory, and recent gripper state.
- [ ] `robot_control.task_policy` exists and has unit tests.
- [ ] `robot_control.task_policy` does not import Voice Runtime, Pipecat, MCP, Codex, or LangGraph.
- [ ] Motion tools are blocked without recent `moveit_get_current_pose`.
- [ ] `moveit_execute_plan` is blocked unless the plan name was recently returned by planning.
- [ ] Auto-execution records the plan name before calling `moveit_execute_plan`.
- [ ] `moveit_attach_object` is blocked unless gripper state is recently known closed.
- [ ] Policy failures are returned as structured tool outputs to Codex.
- [ ] Existing `voice_runtime.robot_safety` still validates tool shape, robot name, workspace, timeout, and plan-name structure.
- [ ] Full pytest, ruff, and pyright verification pass.
