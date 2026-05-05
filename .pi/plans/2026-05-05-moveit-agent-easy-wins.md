# MoveIt Agent Easy Wins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the voice robot agent reliably use the current MoveIt tools, robot feedback, structured corrections, and compact robot context before the LangGraph migration.

**Architecture:** Keep Pipecat and the Codex agent backend in place. Improve the prompt, robot safety/tool contracts, bridge serialization, and agent-turn context injection behind the existing `RobotMCPBridge` and `OpenAICodexAgentProcessor` seams. New tool support is advertised only when the backing MoveIt MCP server exposes the corresponding legacy or canonical tool name.

**Tech Stack:** Python 3.10+, Pipecat, OpenAI Codex backend adapter, MCP `CallToolResult`, pytest, ruff, pyright.

---

## Parallelization map

These tasks are intentionally split for parallel agents.

- **Task 1** touches prompt and prompt tests only.
- **Task 2** touches robot safety contracts and safety tests.
- **Task 3** touches bridge serialization and bridge tests.
- **Task 4** touches Codex processor context injection and processor tests.
- **Task 5** adds behavior-contract eval tests after Tasks 1-4 are merged.
- **Task 6** updates docs/instructions and runs final validation.

Avoid running Task 5 until Tasks 1-4 are integrated because it checks cross-cutting behavior.

## Files and responsibilities

- Modify: `AGENTS.md` — agent instruction file. Already updated with MoveIt/prompt/context guidance before this plan was written.
- Modify: `server/prompts.py` — user-visible robot agent behavior and canonical tool usage rules.
- Create: `server/tests/test_prompts.py` — verifies prompt/tool-name alignment and required behavior rules.
- Modify: `server/voice_runtime/robot_safety.py` — canonical tool names, argument validation, tool descriptions, executable-plan helpers, structured safety failures.
- Modify: `server/tests/test_voice_runtime_robot_safety.py` — safety contract coverage for new tools and structured failures.
- Modify: `server/robot_mcp_bridge.py` — canonical tool advertisement, description overrides, validation failure serialization, tool result normalization.
- Modify: `server/tests/test_robot_mcp_bridge.py` — bridge behavior for canonical tools, new tools, and structured errors.
- Create: `server/voice_runtime/robot_context.py` — compact last-known robot context state and instruction block rendering.
- Modify: `server/openai_codex_agent_processor.py` — inject compact robot context into instructions and update context from tool results.
- Modify: `server/tests/test_openai_codex_agent_processor.py` — context injection and context update tests.
- Create: `server/tests/test_moveit_agent_behavior_contracts.py` — integration-style behavior-contract tests with fakes.

---

## Task 1: Align prompt with real MoveIt tools

**Files:**
- Modify: `server/prompts.py`
- Create: `server/tests/test_prompts.py`

- [ ] **Step 1: Write failing prompt alignment tests**

Create `server/tests/test_prompts.py`:

```python
from prompts import SYSTEM_PROMPT

CANONICAL_TOOLS = {
    "moveit_get_robot_status",
    "moveit_plan_free_motion",
    "moveit_plan_linear_motion",
    "moveit_execute_plan",
    "moveit_open_gripper",
    "moveit_close_gripper",
}

STALE_TOOLS = {
    "connect_robot",
    "disconnect_robot",
    "get_joints",
    "get_tcp_pose",
    "move_to_position",
    "move_to_pose",
    "move_linear",
    "move_joints",
    "control_gripper",
    "control_gripper_position",
}


def test_prompt_lists_only_canonical_moveit_tools() -> None:
    for tool_name in CANONICAL_TOOLS:
        assert tool_name in SYSTEM_PROMPT

    for tool_name in STALE_TOOLS:
        assert tool_name not in SYSTEM_PROMPT


def test_prompt_requires_observe_plan_execute_verify_for_robot_actions() -> None:
    assert "observe" in SYSTEM_PROMPT.lower()
    assert "plan before" in SYSTEM_PROMPT.lower()
    assert "execute only" in SYSTEM_PROMPT.lower()
    assert "verify" in SYSTEM_PROMPT.lower()


def test_prompt_requires_fresh_status_for_state_dependent_actions() -> None:
    prompt = SYSTEM_PROMPT.lower()
    assert "moveit_get_robot_status" in prompt
    assert "relative" in prompt
    assert "fresh" in prompt
    assert "last-known context is advisory" in prompt
```

- [ ] **Step 2: Run the prompt tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: at least one failure because `server/prompts.py` currently lists stale tools such as `move_to_position` and `get_tcp_pose`.

- [ ] **Step 3: Replace `SYSTEM_PROMPT` with the MoveIt-aligned prompt**

In `server/prompts.py`, replace the existing `SYSTEM_PROMPT` value with:

```python
SYSTEM_PROMPT = """You are a voice-controlled robot agent for a Universal Robot UR10 arm running in simulation.

Users speak commands to you via voice. Respond conversationally but briefly, usually 1 sentence.

# Goal
Safely translate user intent into MoveIt tool calls. For robot actions, observe when current state matters, plan before execution, execute only returned valid plans, verify results, then respond briefly.

# Available MoveIt tools
- moveit_get_robot_status: inspect current robot state, TCP pose, joints, gripper, planning state, and recent execution state.
- moveit_plan_free_motion: plan a non-linear MoveIt motion to a target pose.
- moveit_plan_linear_motion: plan a straight TCP path to a target pose.
- moveit_execute_plan: execute a valid plan returned by a planning tool.
- moveit_open_gripper: open the gripper.
- moveit_close_gripper: close the gripper.

# Robot and safety constraints
- This version is simulation-only.
- The only allowed robot_name is "UR10".
- There is no HoloLens, gaze target, world model, or user-position data.
- If the user says "that", "this", "there", "bring it here", or another ambiguous reference without enough context, ask a clarifying question instead of guessing.

# Tool-use rules
- For movement, gripper, retry, and safety-sensitive actions, use MoveIt tools instead of answering from memory.
- Last-known context is advisory only. For movement, relative commands, retries, or safety-sensitive actions, call moveit_get_robot_status first for fresh state.
- Plan before execution. Use moveit_execute_plan only with a plan_name returned by a successful planning tool.
- Use moveit_plan_free_motion for ordinary point-to-point movement.
- Use moveit_plan_linear_motion only when a straight TCP path matters.
- Call tools one at a time and wait for each result.
- If a tool returns retryable=true, apply the correction once. If the same action fails twice, stop and explain the blocker.

# Coordinates and magnitudes
- +X: forward from the base.
- +Y: left from the base.
- +Z: up.
- "up" means +Z, "down" means -Z.
- "a bit" or "slightly" means 0.05 m.
- No modifier means 0.10 m.
- "a lot" or "far" means 0.30 m.

# Response style
- Keep responses to 1 short sentence unless the user asks for detail.
- Report movement distances in mm to the user.
- No emojis.
"""
```

- [ ] **Step 4: Run prompt tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_prompts.py -v
```

Expected: all tests in `tests/test_prompts.py` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add server/prompts.py server/tests/test_prompts.py
git commit -m "fix: align robot prompt with MoveIt tools"
```

---

## Task 2: Define and validate agent-friendly MoveIt tool contracts

**Files:**
- Modify: `server/voice_runtime/robot_safety.py`
- Modify: `server/tests/test_voice_runtime_robot_safety.py`

- [ ] **Step 1: Add failing tests for new canonical tools and descriptions**

Append to `server/tests/test_voice_runtime_robot_safety.py`:

```python
from voice_runtime.robot_safety import (
    agent_tool_description,
    structured_robot_error,
)


def test_accepts_relative_motion_arguments() -> None:
    validate_robot_tool_call(
        "moveit_plan_relative_motion",
        {
            "robot_name": "UR10",
            "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
            "motion_type": "free",
            "timeout_s": 10.0,
        },
    )


def test_rejects_relative_motion_outside_delta_limit() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call(
            "moveit_plan_relative_motion",
            {
                "robot_name": "UR10",
                "delta": {"x": 0.0, "y": 0.0, "z": 2.0},
                "motion_type": "free",
            },
        )

    assert str(exc.value) == "Relative motion is outside safe delta range"
    assert "within +/-0.30 m" in exc.value.correction


def test_rejects_unknown_relative_motion_type() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call(
            "moveit_plan_relative_motion",
            {
                "robot_name": "UR10",
                "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
                "motion_type": "diagonal",
            },
        )

    assert str(exc.value) == "motion_type must be free or linear"


def test_accepts_named_pose_tools() -> None:
    validate_robot_tool_call("moveit_list_named_poses", {"robot_name": "UR10"})
    validate_robot_tool_call(
        "moveit_plan_named_pose",
        {"robot_name": "UR10", "pose_name": "home", "timeout_s": 10.0},
    )


def test_rejects_empty_named_pose() -> None:
    with pytest.raises(RobotSafetyError) as exc:
        validate_robot_tool_call("moveit_plan_named_pose", {"robot_name": "UR10", "pose_name": ""})

    assert str(exc.value) == "Expected a non-empty pose_name"


def test_structured_robot_error_shape() -> None:
    err = RobotSafetyError("bad target", correction="Use a safe target.")

    assert structured_robot_error(err) == {
        "ok": False,
        "error": "bad target",
        "correction": "Use a safe target.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_robot_status",
    }


def test_agent_tool_descriptions_are_high_signal() -> None:
    assert "fresh robot state" in agent_tool_description("moveit_get_robot_status")
    assert "relative" in agent_tool_description("moveit_plan_relative_motion")
    assert "named" in agent_tool_description("moveit_plan_named_pose")
```

- [ ] **Step 2: Run the safety tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_voice_runtime_robot_safety.py -v
```

Expected: failures for missing `agent_tool_description`, `structured_robot_error`, and unrecognized new tool names.

- [ ] **Step 3: Extend robot safety constants and descriptions**

In `server/voice_runtime/robot_safety.py`, update constants near the top:

```python
RELATIVE_DELTA_ABS_LIMIT_M = 0.30

AGENT_TO_LEGACY_MCP_TOOL_NAMES = {
    "moveit_plan_free_motion": "plan_free_motion",
    "moveit_plan_linear_motion": "plan_linear_motion",
    "moveit_execute_plan": "execute_plan",
    "moveit_open_gripper": "open_gripper",
    "moveit_close_gripper": "close_gripper",
    "moveit_get_robot_status": "get_robot_status",
    "moveit_plan_relative_motion": "plan_relative_motion",
    "moveit_list_named_poses": "list_named_poses",
    "moveit_plan_named_pose": "plan_named_pose",
}

_AGENT_TOOL_DESCRIPTIONS = {
    "moveit_get_robot_status": "Get fresh robot state: connection/planning status, TCP pose, joints, gripper state, safety state, and recent plan/execution summaries. Call before movement, relative commands, retries, or safety-sensitive actions.",
    "moveit_plan_free_motion": "Plan a point-to-point MoveIt motion to an absolute target pose. Use for ordinary movement when a straight TCP path is not required.",
    "moveit_plan_linear_motion": "Plan a straight TCP path to an absolute target pose. Use only when the user asks for a straight or linear motion.",
    "moveit_execute_plan": "Execute a valid plan_name returned by a successful MoveIt planning tool. Do not invent plan names.",
    "moveit_open_gripper": "Open the UR10 gripper in simulation.",
    "moveit_close_gripper": "Close the UR10 gripper in simulation.",
    "moveit_plan_relative_motion": "Plan a relative movement from the current TCP pose using a small x/y/z delta in meters. Use for voice commands such as move up a bit, go left, lower slightly, or back up.",
    "moveit_list_named_poses": "List named robot poses available for the UR10, such as home or ready, when exposed by the MoveIt MCP server.",
    "moveit_plan_named_pose": "Plan motion to a named robot pose such as home, ready, or reset. Use after list_named_poses when the requested pose name is uncertain.",
}

_ALLOWED_ARGUMENTS: dict[str, set[str]] = {
    "moveit_plan_free_motion": {"robot_name", "position", "timeout_s"},
    "moveit_plan_linear_motion": {"robot_name", "position", "timeout_s"},
    "moveit_execute_plan": {"robot_name", "plan_name", "timeout_s"},
    "moveit_open_gripper": {"robot_name", "timeout_s"},
    "moveit_close_gripper": {"robot_name", "timeout_s"},
    "moveit_get_robot_status": {"robot_name"},
    "moveit_plan_relative_motion": {"robot_name", "delta", "motion_type", "timeout_s"},
    "moveit_list_named_poses": {"robot_name"},
    "moveit_plan_named_pose": {"robot_name", "pose_name", "timeout_s"},
}
```

- [ ] **Step 4: Add helper functions and validators**

In `server/voice_runtime/robot_safety.py`, add these functions below `canonical_mcp_tool_name`:

```python
def agent_tool_description(agent_tool_name: str) -> str:
    try:
        return _AGENT_TOOL_DESCRIPTIONS[agent_tool_name]
    except KeyError as exc:
        raise RobotSafetyError(
            f"Tool is not allowed: {agent_tool_name}",
            correction="Use one of the allowed MoveIt robot tools.",
        ) from exc


def structured_robot_error(
    exc: RobotSafetyError,
    *,
    retryable: bool = True,
    suggested_next_tool: str | None = "moveit_get_robot_status",
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "error": str(exc),
        "correction": exc.correction,
        "retryable": retryable,
    }
    if suggested_next_tool is not None:
        payload["suggested_next_tool"] = suggested_next_tool
    return payload
```

In `validate_robot_tool_call`, after the gripper branch, add:

```python
    if name == "moveit_plan_relative_motion":
        _validate_delta(arguments.get("delta"))
        motion_type = arguments.get("motion_type", "free")
        if motion_type not in {"free", "linear"}:
            raise RobotSafetyError(
                "motion_type must be free or linear",
                correction='Retry with motion_type="free" or motion_type="linear".',
            )
        _validate_timeout(arguments.get("timeout_s"))
        return

    if name == "moveit_plan_named_pose":
        pose_name = arguments.get("pose_name")
        if not isinstance(pose_name, str) or not pose_name.strip():
            raise RobotSafetyError(
                "Expected a non-empty pose_name",
                correction="Retry with a named pose returned by moveit_list_named_poses.",
            )
        _validate_timeout(arguments.get("timeout_s"))
        return
```

Add `_validate_delta` near `_validate_pose`:

```python
def _validate_delta(value: Any) -> None:
    if not isinstance(value, dict):
        raise RobotSafetyError(
            "Expected delta coordinates",
            correction="Retry with delta x, y, and z coordinates in meters.",
        )
    for axis in ("x", "y", "z"):
        coordinate = _finite_float(value.get(axis))
        if coordinate is None or abs(coordinate) > RELATIVE_DELTA_ABS_LIMIT_M:
            raise RobotSafetyError(
                "Relative motion is outside safe delta range",
                correction=f"Retry with x/y/z deltas within +/-{RELATIVE_DELTA_ABS_LIMIT_M:.2f} m.",
            )
```

- [ ] **Step 5: Run the safety tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_voice_runtime_robot_safety.py -v
```

Expected: all tests in `tests/test_voice_runtime_robot_safety.py` pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add server/voice_runtime/robot_safety.py server/tests/test_voice_runtime_robot_safety.py
git commit -m "feat: define agent-friendly MoveIt tool contracts"
```

---

## Task 3: Normalize bridge tool descriptions and structured errors

**Files:**
- Modify: `server/robot_mcp_bridge.py`
- Modify: `server/tests/test_robot_mcp_bridge.py`

- [ ] **Step 1: Add failing bridge tests for descriptions, new tools, and structured failures**

Append to `server/tests/test_robot_mcp_bridge.py`:

```python
class FakeExpandedServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(name="get_robot_status", description="Raw status", inputSchema={"type": "object"}),
            Tool(name="plan_relative_motion", description="Raw relative", inputSchema={"type": "object"}),
            Tool(name="list_named_poses", description="Raw names", inputSchema={"type": "object"}),
            Tool(name="plan_named_pose", description="Raw named", inputSchema={"type": "object"}),
        ]


@pytest.mark.asyncio
async def test_bridge_advertises_agent_friendly_descriptions():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeExpandedServer())

    await bridge.connect()

    tools = {tool["name"]: tool for tool in bridge.function_tools()}
    assert "fresh robot state" in tools["moveit_get_robot_status"]["description"]
    assert "relative movement" in tools["moveit_plan_relative_motion"]["description"]
    assert "named robot poses" in tools["moveit_list_named_poses"]["description"]
    assert "named robot pose" in tools["moveit_plan_named_pose"]["description"]


@pytest.mark.asyncio
async def test_validation_failure_returns_structured_retry_guidance_without_mcp_call():
    server = FakeExpandedServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_robot_status", {"robot_name": "UR5"})

    assert json.loads(output) == {
        "ok": False,
        "error": "Only Vizor robot UR10 is allowed",
        "correction": 'Retry with robot_name="UR10".',
        "retryable": True,
        "suggested_next_tool": "moveit_get_robot_status",
    }
    assert server.called == []


@pytest.mark.asyncio
async def test_calls_new_relative_motion_tool_when_advertised():
    server = FakeExpandedServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    await bridge.call_tool(
        "moveit_plan_relative_motion",
        {
            "robot_name": "UR10",
            "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
            "motion_type": "free",
        },
    )

    assert server.called == [
        (
            "plan_relative_motion",
            {
                "robot_name": "UR10",
                "delta": {"x": 0.0, "y": 0.0, "z": 0.05},
                "motion_type": "free",
            },
        )
    ]
```

- [ ] **Step 2: Run bridge tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_robot_mcp_bridge.py -v
```

Expected: failures because bridge descriptions still pass through raw MCP descriptions and validation failures use the old shape.

- [ ] **Step 3: Use safety descriptions and structured errors in the bridge**

In `server/robot_mcp_bridge.py`, update imports:

```python
from voice_runtime.robot_safety import (
    AGENT_TO_LEGACY_MCP_TOOL_NAMES,
    RobotSafetyError,
    agent_tool_description,
    structured_robot_error,
    validate_robot_tool_call,
)
```

In `function_tools`, replace the description field:

```python
"description": agent_tool_description(agent_name),
```

Replace `_serialize_validation_failure` with:

```python
def _serialize_validation_failure(exc: RobotSafetyError) -> str:
    return json.dumps(structured_robot_error(exc), ensure_ascii=False)
```

- [ ] **Step 4: Update the existing old-shape bridge test**

In `server/tests/test_robot_mcp_bridge.py`, update `test_validation_failure_returns_compatible_error_json_without_mcp_call` expected output to:

```python
assert json.loads(output) == {
    "ok": False,
    "error": "Only Vizor robot UR10 is allowed",
    "correction": 'Retry with robot_name="UR10".',
    "retryable": True,
    "suggested_next_tool": "moveit_get_robot_status",
}
```

- [ ] **Step 5: Run bridge tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_mcp_bridge.py -v
```

Expected: all bridge tests pass.

- [ ] **Step 6: Commit Task 3**

```bash
git add server/robot_mcp_bridge.py server/tests/test_robot_mcp_bridge.py
git commit -m "feat: return agent-friendly MoveIt tool feedback"
```

---

## Task 4: Add compact robot context injection

**Files:**
- Create: `server/voice_runtime/robot_context.py`
- Modify: `server/openai_codex_agent_processor.py`
- Modify: `server/tests/test_openai_codex_agent_processor.py`

- [ ] **Step 1: Write failing unit tests for robot context rendering**

Create `server/tests/test_robot_context.py`:

```python
import json

from voice_runtime.robot_context import RobotContextStore


def test_empty_robot_context_renders_advisory_block() -> None:
    store = RobotContextStore()

    text = store.render_instruction_block()

    assert "Last-known robot context" in text
    assert "No robot status has been observed yet" in text
    assert "advisory only" in text
    assert "moveit_get_robot_status" in text


def test_robot_context_updates_from_status_tool_output() -> None:
    store = RobotContextStore()
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot_name": "UR10",
                "tcp_pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
                "gripper": {"state": "open"},
                "last_execution": {"result": "pass"},
            }
        }
    )

    store.update_from_tool_result("moveit_get_robot_status", output)

    text = store.render_instruction_block()
    assert "UR10" in text
    assert "x=0.100" in text
    assert "y=0.200" in text
    assert "z=0.300" in text
    assert "gripper: open" in text
    assert "last execution: pass" in text
```

- [ ] **Step 2: Run context tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py -v
```

Expected: import failure because `voice_runtime.robot_context` does not exist.

- [ ] **Step 3: Implement `RobotContextStore`**

Create `server/voice_runtime/robot_context.py`:

```python
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class RobotContextSnapshot:
    observed_at_s: float | None = None
    robot_name: str | None = None
    tcp_pose: dict[str, Any] | None = None
    gripper_state: str | None = None
    last_execution_result: str | None = None


class RobotContextStore:
    def __init__(self) -> None:
        self._snapshot = RobotContextSnapshot()

    def render_instruction_block(self) -> str:
        age = self._status_age_text()
        lines = [
            "Last-known robot context:",
            "- This context is advisory only.",
            "- For movement, relative commands, retries, or safety-sensitive actions, call moveit_get_robot_status first.",
            f"- status age: {age}",
        ]
        if self._snapshot.robot_name is None:
            lines.append("- No robot status has been observed yet.")
            return "\n".join(lines)

        lines.append(f"- robot: {self._snapshot.robot_name}")
        pose_text = self._tcp_pose_text()
        if pose_text:
            lines.append(f"- tcp pose: {pose_text}")
        if self._snapshot.gripper_state:
            lines.append(f"- gripper: {self._snapshot.gripper_state}")
        if self._snapshot.last_execution_result:
            lines.append(f"- last execution: {self._snapshot.last_execution_result}")
        return "\n".join(lines)

    def update_from_tool_result(self, tool_name: str, output: str) -> None:
        if tool_name != "moveit_get_robot_status":
            return
        structured_content = _structured_content(output)
        if not isinstance(structured_content, dict) or structured_content.get("ok") is not True:
            return

        self._snapshot.observed_at_s = time.monotonic()
        robot_name = structured_content.get("robot_name")
        if isinstance(robot_name, str):
            self._snapshot.robot_name = robot_name
        tcp_pose = structured_content.get("tcp_pose")
        if isinstance(tcp_pose, dict):
            self._snapshot.tcp_pose = tcp_pose
        gripper = structured_content.get("gripper")
        if isinstance(gripper, dict) and isinstance(gripper.get("state"), str):
            self._snapshot.gripper_state = gripper["state"]
        last_execution = structured_content.get("last_execution")
        if isinstance(last_execution, dict) and isinstance(last_execution.get("result"), str):
            self._snapshot.last_execution_result = last_execution["result"]

    def _status_age_text(self) -> str:
        if self._snapshot.observed_at_s is None:
            return "unknown"
        return f"{time.monotonic() - self._snapshot.observed_at_s:.1f}s"

    def _tcp_pose_text(self) -> str | None:
        pose = self._snapshot.tcp_pose
        if not isinstance(pose, dict):
            return None
        position = pose.get("position")
        if not isinstance(position, dict):
            return None
        try:
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        except (KeyError, TypeError, ValueError):
            return None
        return f"x={x:.3f}, y={y:.3f}, z={z:.3f}"


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")
```

- [ ] **Step 4: Run context tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py -v
```

Expected: all context tests pass.

- [ ] **Step 5: Add failing Codex processor tests for context injection and updates**

Append to `server/tests/test_openai_codex_agent_processor.py`:

```python
@pytest.mark.asyncio
async def test_injects_compact_robot_context_into_codex_instructions():
    backend = FakeBackend([CodexResponseResult(text="ok")])
    bridge = FakeBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await _run_turn(processor, "what can you do?")

    instructions = backend.requests[0]["instructions"]
    assert "Last-known robot context" in instructions
    assert "advisory only" in instructions
    assert "moveit_get_robot_status" in instructions


@pytest.mark.asyncio
async def test_updates_robot_context_after_status_tool_result():
    status_call = CodexToolCall(
        call_id="call-1",
        item_id="item-1",
        name="moveit_get_robot_status",
        arguments={"robot_name": "UR10"},
        raw_arguments='{"robot_name":"UR10"}',
    )
    backend = FakeBackend(
        [
            CodexResponseResult(
                tool_calls=[status_call],
                output_items=[
                    {
                        "type": "function_call",
                        "id": "item-1",
                        "call_id": "call-1",
                        "name": "moveit_get_robot_status",
                        "arguments": '{"robot_name":"UR10"}',
                    }
                ],
            ),
            CodexResponseResult(text="Robot is ready."),
        ]
    )

    class StatusBridge(FakeBridge):
        async def call_tool(self, name, arguments):
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot_name": "UR10",
                        "tcp_pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
                        "gripper": {"state": "open"},
                    }
                }
            )

    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=StatusBridge(),
    )

    await _run_turn(processor, "status")

    followup_backend = FakeBackend([CodexResponseResult(text="ok")])
    processor._backend_client = followup_backend
    await _run_turn(processor, "where is it?")

    instructions = followup_backend.requests[0]["instructions"]
    assert "robot: UR10" in instructions
    assert "x=0.100" in instructions
    assert "gripper: open" in instructions
```

- [ ] **Step 6: Run Codex processor tests and verify they fail**

Run from `server/`:

```bash
uv run pytest tests/test_openai_codex_agent_processor.py -v
```

Expected: failures because instructions do not yet include robot context.

- [ ] **Step 7: Inject context into Codex instructions and update it from tool results**

In `server/openai_codex_agent_processor.py`, add import:

```python
from voice_runtime.robot_context import RobotContextStore
```

In `OpenAICodexAgentProcessor.__init__`, add:

```python
self._robot_context = RobotContextStore()
```

Add a method to the class:

```python
def _instructions(self) -> str:
    return f"{SYSTEM_PROMPT}\n\n{self._robot_context.render_instruction_block()}"
```

Replace both `instructions=SYSTEM_PROMPT` arguments in `backend_client.create_response(...)` with:

```python
instructions=self._instructions(),
```

In `_call_robot_tool`, after `output = await tool_bridge.call_tool(name, arguments)`, add:

```python
self._robot_context.update_from_tool_result(name, output)
```

- [ ] **Step 8: Run targeted tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py tests/test_openai_codex_agent_processor.py -v
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 4**

```bash
git add server/voice_runtime/robot_context.py server/tests/test_robot_context.py server/openai_codex_agent_processor.py server/tests/test_openai_codex_agent_processor.py
git commit -m "feat: inject compact robot context into Codex turns"
```

---

## Task 5: Add behavior-contract eval tests

**Files:**
- Create: `server/tests/test_moveit_agent_behavior_contracts.py`

- [ ] **Step 1: Write behavior-contract tests**

Create `server/tests/test_moveit_agent_behavior_contracts.py`:

```python
import json

import pytest

from codex_auth import CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput


class Store:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class ScriptedBackend:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def create_response(self, credentials, *, model, instructions, input_items, tools):
        self.requests.append(
            {
                "model": model,
                "instructions": instructions,
                "input_items": list(input_items),
                "tools": list(tools),
            }
        )
        return self.results.pop(0)

    async def close(self):
        pass


class BehaviorBridge:
    def __init__(self):
        self.calls = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    def function_tools(self):
        return [
            {"type": "function", "name": "moveit_get_robot_status", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_free_motion", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_execute_plan", "parameters": {"type": "object"}, "strict": None},
            {"type": "function", "name": "moveit_plan_relative_motion", "parameters": {"type": "object"}, "strict": None},
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "moveit_get_robot_status":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot_name": "UR10",
                        "tcp_pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
                        "gripper": {"state": "open"},
                    }
                }
            )
        if name == "moveit_plan_free_motion":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": "plan-1"},
                    }
                }
            )
        if name == "moveit_execute_plan":
            return json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
        return json.dumps({"structured_content": {"ok": True}})


async def run_processor(processor, text):
    turn = AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])
    return [chunk async for chunk in processor.run_turn(turn)]


def tool_call(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return CodexToolCall(
        call_id=call_id,
        item_id=item_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def output_item(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


@pytest.mark.asyncio
async def test_relative_movement_behavior_observes_before_answering():
    status = tool_call("moveit_get_robot_status")
    backend = ScriptedBackend(
        [
            CodexResponseResult(tool_calls=[status], output_items=[output_item("moveit_get_robot_status")]),
            CodexResponseResult(text="I checked the robot and can plan the relative move."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls[0] == ("moveit_get_robot_status", {"robot_name": "UR10"})
    assert chunks == ["I checked the robot and can plan the relative move."]


@pytest.mark.asyncio
async def test_plan_tool_is_auto_executed_once_plan_is_executable():
    plan_args = {
        "robot_name": "UR10",
        "position": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    plan = tool_call("moveit_plan_free_motion", arguments=plan_args)
    backend = ScriptedBackend(
        [
            CodexResponseResult(
                tool_calls=[plan],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls == [
        ("moveit_plan_free_motion", plan_args),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "plan-1"}),
    ]
    assert chunks == ["Moved up 50 mm."]
```

- [ ] **Step 2: Run behavior-contract tests**

Run from `server/`:

```bash
uv run pytest tests/test_moveit_agent_behavior_contracts.py -v
```

Expected: all tests pass after Tasks 1-4 are integrated.

- [ ] **Step 3: Commit Task 5**

```bash
git add server/tests/test_moveit_agent_behavior_contracts.py
git commit -m "test: add MoveIt agent behavior contracts"
```

---

## Task 6: Documentation/instruction update and final validation

**Files:**
- Verify: `AGENTS.md`
- Verify: `.pi/plans/2026-05-05-moveit-agent-easy-wins.md`
- Optional modify: `README.md` only if runtime usage changed for users

- [ ] **Step 1: Verify AGENTS.md contains targeted MoveIt guidance**

Run from repo root:

```bash
grep -n "agent-friendly workflow tools" AGENTS.md
grep -n "stale tools" AGENTS.md
grep -n "Last-known" AGENTS.md
```

Expected: each command prints one matching line.

- [ ] **Step 2: Run all targeted tests**

Run from `server/`:

```bash
uv run pytest tests/test_prompts.py tests/test_voice_runtime_robot_safety.py tests/test_robot_mcp_bridge.py tests/test_robot_context.py tests/test_openai_codex_agent_processor.py tests/test_moveit_agent_behavior_contracts.py -v
```

Expected: all targeted tests pass.

- [ ] **Step 3: Run full verification**

Run from `server/`:

```bash
uv run pytest
uv run ruff check .
uv run pyright .
```

Expected: all commands exit successfully. If pre-existing failures unrelated to this plan appear, capture the exact command output and file a separate cleanup issue instead of masking failures.

- [ ] **Step 4: Final commit if AGENTS.md changed during execution**

The current AGENTS.md guidance was updated before this plan. If an implementation worker changes it further, commit it:

```bash
git add AGENTS.md README.md
git commit -m "docs: document MoveIt agent guidance"
```

Skip this commit if there are no docs changes.

---

## Completion checklist

- [ ] Prompt lists only canonical real MoveIt tools.
- [ ] Safety layer recognizes relative and named-pose tool contracts.
- [ ] Bridge exposes agent-friendly descriptions.
- [ ] Validation failures return structured retry guidance.
- [ ] Codex instructions include compact advisory robot context.
- [ ] Fresh status rule is present in prompt and context block.
- [ ] Behavior-contract tests cover status-before-relative and plan-before-execute behavior.
- [ ] Full tests, ruff, and pyright are run or blockers are documented.
