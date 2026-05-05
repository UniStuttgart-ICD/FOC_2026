# Robot Control Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract robot-side control code from legacy top-level and `voice_runtime` placements into the target `server/robot_control/` package.

**Architecture:** `voice_runtime` stays Pipecat/audio-only. `robot_control` owns Robot Context, Robot Call Validation, Robot Tool Adapter, and Task Policy. This plan performs the Robot Control extraction before the later `agent_control` extraction.

**Tech Stack:** Python 3.12, pytest, ruff, pyright, MCP `CallToolResult`, LangGraph tests with fakes.

---

## Prerequisite

Run this after `.pi/plans/2026-05-05-minimal-task-policy-layer.md` has landed. That plan introduces `server/robot_control/task_policy.py` and the first Robot Control import guard. This plan completes the Robot Control package by moving context, call validation, and MCP bridge modules into it.

## Architecture constraints

- Follow `ARCHITECTURE.md` and `CONTEXT.md` terminology.
- Do not keep compatibility shims in `voice_runtime` that import `robot_control`; that would violate the import-direction invariant.
- `voice_runtime` must not import `robot_control`.
- `robot_control` must not import `voice_runtime`, `agent_control`, or Pipecat.
- `robot_control.mcp_bridge` may import MCP adapter dependencies (`agents`, `mcp`) because it is the Robot Tool Adapter.
- Pure Robot Control modules (`call_validation.py`, `context.py`, `task_policy.py`) must not import MCP, Codex, LangGraph, Pipecat, `voice_runtime`, or app composition modules.
- Use **Robot Call Validation**, not “Robot Safety”, for local tool-call validation language.
- Movement safety remains delegated to MoveIt planning/execution and the robot simulation stack.

## Files and responsibilities

- Create/modify: `server/robot_control/__init__.py`
  - Package marker and package docstring.
- Move: `server/voice_runtime/robot_safety.py` -> `server/robot_control/call_validation.py`
  - Robot Call Validation constants, validation errors, tool descriptions, executable-plan parsing, and result text helpers.
- Move: `server/voice_runtime/robot_context.py` -> `server/robot_control/context.py`
  - Robot Context store and tool-result parsing.
- Move: `server/robot_mcp_bridge.py` -> `server/robot_control/mcp_bridge.py`
  - Robot MCP tool discovery, tool advertisement, validation failure serialization, and MCP execution.
- Delete: `server/voice_runtime/robot_safety.py`
- Delete: `server/voice_runtime/robot_context.py`
- Delete: `server/robot_mcp_bridge.py`
- Rename/modify: `server/tests/test_voice_runtime_robot_safety.py` -> `server/tests/test_robot_call_validation.py`
- Modify: `server/tests/test_robot_context.py`
- Modify: `server/tests/test_robot_mcp_bridge.py`
- Modify: `server/tests/test_robot_control_imports.py`
- Modify: `server/tests/test_orthogonal_imports.py`
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/openai_codex_agent_processor.py`
- Modify tests importing moved modules: `server/tests/test_langgraph_robot_agent.py`, `server/tests/test_openai_codex_agent_processor.py`
- Modify docs/instructions: `AGENTS.md`, `CONTEXT.md`, `ARCHITECTURE.md`, `README.md`, `docs/architecture.md`

---

## Task 1: Strengthen Robot Control import guards

**Files:**
- Modify: `server/tests/test_robot_control_imports.py`

- [ ] **Step 1: Replace the import guard with the target Robot Control guard**

Replace `server/tests/test_robot_control_imports.py` with:

```python
import ast
from pathlib import Path

SERVER_DIR = Path(__file__).resolve().parents[1]
ROBOT_CONTROL_DIR = SERVER_DIR / "robot_control"

PURE_ROBOT_CONTROL_MODULES = {
    "call_validation.py",
    "context.py",
    "task_policy.py",
}
ROBOT_CONTROL_FORBIDDEN_ROOTS = {
    "agent_control",
    "pipecat",
    "voice_runtime",
}
PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS = ROBOT_CONTROL_FORBIDDEN_ROOTS | {
    "agents",
    "langgraph",
    "mcp",
    "openai",
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


def test_robot_control_modules_do_not_import_voice_runtime_or_agent_control() -> None:
    for path in ROBOT_CONTROL_DIR.glob("*.py"):
        imported = _import_roots(path)
        forbidden = imported & ROBOT_CONTROL_FORBIDDEN_ROOTS
        assert not forbidden, f"{path.name} imports forbidden module(s): {sorted(forbidden)}"


def test_pure_robot_control_modules_do_not_import_runtime_adapters() -> None:
    for name in PURE_ROBOT_CONTROL_MODULES:
        path = ROBOT_CONTROL_DIR / name
        imported = _import_roots(path)
        forbidden = imported & PURE_ROBOT_CONTROL_FORBIDDEN_ROOTS
        assert not forbidden, f"{name} imports forbidden module(s): {sorted(forbidden)}"
```

- [ ] **Step 2: Run the import guard and verify it fails**

Run from `server/`:

```bash
uv run pytest tests/test_robot_control_imports.py -v
```

Expected: failure because `robot_control/call_validation.py` and `robot_control/context.py` do not exist yet.

- [ ] **Step 3: Commit the failing guard**

```bash
git add server/tests/test_robot_control_imports.py
git commit -m "test: define robot control import invariants"
```

---

## Task 2: Move Robot Call Validation into robot_control

**Files:**
- Move: `server/voice_runtime/robot_safety.py` -> `server/robot_control/call_validation.py`
- Move/modify: `server/tests/test_voice_runtime_robot_safety.py` -> `server/tests/test_robot_call_validation.py`

- [ ] **Step 1: Move the module and tests**

Run from repo root:

```bash
git mv server/voice_runtime/robot_safety.py server/robot_control/call_validation.py
git mv server/tests/test_voice_runtime_robot_safety.py server/tests/test_robot_call_validation.py
```

Expected: both files are staged as renames.

- [ ] **Step 2: Rename validation error and structured error terms**

In `server/robot_control/call_validation.py`, replace:

```python
class RobotSafetyError(ValueError):
    """Raised when a robot tool call violates local validation policy."""

    def __init__(self, message: str, *, correction: str):
        super().__init__(message)
        self.correction = correction
```

with:

```python
class RobotCallValidationError(ValueError):
    """Raised when a robot tool call violates local validation policy."""

    def __init__(self, message: str, *, correction: str):
        super().__init__(message)
        self.correction = correction
```

Then replace every `RobotSafetyError` in `server/robot_control/call_validation.py` with `RobotCallValidationError`.

Rename this function:

```python
def structured_robot_error(
    exc: RobotCallValidationError,
    *,
    retryable: bool = True,
    suggested_next_tool: str | None = "moveit_get_current_pose",
) -> dict[str, Any]:
```

into:

```python
def structured_robot_call_error(
    exc: RobotCallValidationError,
    *,
    retryable: bool = True,
    suggested_next_tool: str | None = "moveit_get_current_pose",
) -> dict[str, Any]:
```

Do not change the JSON payload shape.

- [ ] **Step 3: Update the call validation tests**

In `server/tests/test_robot_call_validation.py`, replace the import block with:

```python
from robot_control.call_validation import (
    RobotCallValidationError,
    agent_tool_description,
    canonical_mcp_tool_name,
    executable_plan_name,
    execution_result_text,
    structured_robot_call_error,
    validate_robot_tool_call,
)
```

Then replace:

```python
RobotSafetyError
```

with:

```python
RobotCallValidationError
```

Replace:

```python
structured_robot_error(err)
```

with:

```python
structured_robot_call_error(err)
```

- [ ] **Step 4: Run focused call validation tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_call_validation.py -v
```

Expected: all call validation tests pass.

- [ ] **Step 5: Run the import guard and verify call_validation no longer fails**

Run from `server/`:

```bash
uv run pytest tests/test_robot_control_imports.py -v
```

Expected: still fails because `robot_control/context.py` does not exist yet; it should not fail on `call_validation.py` imports.

- [ ] **Step 6: Commit Robot Call Validation extraction**

```bash
git add server/robot_control/call_validation.py server/tests/test_robot_call_validation.py server/voice_runtime/robot_safety.py server/tests/test_voice_runtime_robot_safety.py
git commit -m "refactor: move robot call validation into robot control"
```

---

## Task 3: Move Robot Context into robot_control

**Files:**
- Move: `server/voice_runtime/robot_context.py` -> `server/robot_control/context.py`
- Modify: `server/tests/test_robot_context.py`

- [ ] **Step 1: Move the context module**

Run from repo root:

```bash
git mv server/voice_runtime/robot_context.py server/robot_control/context.py
```

Expected: file is staged as a rename.

- [ ] **Step 2: Update Robot Context tests to import from robot_control**

In `server/tests/test_robot_context.py`, replace:

```python
from voice_runtime.robot_context import RobotContextStore
```

with:

```python
from robot_control.context import RobotContextStore
```

- [ ] **Step 3: Run Robot Context tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_context.py -v
```

Expected: all Robot Context tests pass.

- [ ] **Step 4: Run Robot Control import guard and verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_robot_control_imports.py -v
```

Expected: all Robot Control import tests pass.

- [ ] **Step 5: Commit Robot Context extraction**

```bash
git add server/robot_control/context.py server/tests/test_robot_context.py server/voice_runtime/robot_context.py
git commit -m "refactor: move robot context into robot control"
```

---

## Task 4: Move Robot MCP Bridge into robot_control

**Files:**
- Move: `server/robot_mcp_bridge.py` -> `server/robot_control/mcp_bridge.py`
- Modify: `server/tests/test_robot_mcp_bridge.py`

- [ ] **Step 1: Move the MCP bridge module**

Run from repo root:

```bash
git mv server/robot_mcp_bridge.py server/robot_control/mcp_bridge.py
```

Expected: file is staged as a rename.

- [ ] **Step 2: Update MCP bridge imports**

In `server/robot_control/mcp_bridge.py`, replace the Robot Call Validation import block with:

```python
from robot_control.call_validation import (
    AGENT_TO_LEGACY_MCP_TOOL_NAMES,
    ALLOWED_ROBOT_TOOLS,
    RobotCallValidationError,
    agent_tool_description,
    structured_robot_call_error,
    validate_robot_tool_call,
)
```

Replace:

```python
except RobotSafetyError as exc:
```

with:

```python
except RobotCallValidationError as exc:
```

Replace:

```python
def _serialize_validation_failure(exc: RobotSafetyError) -> str:
    return json.dumps(structured_robot_error(exc), ensure_ascii=False)
```

with:

```python
def _serialize_validation_failure(exc: RobotCallValidationError) -> str:
    return json.dumps(structured_robot_call_error(exc), ensure_ascii=False)
```

Replace the `RobotMCPBridge` class docstring:

```python
"""Converts robot MCP tools to Codex function tools and executes safe calls."""
```

with:

```python
"""Converts robot MCP tools to Codex function tools and executes validated calls."""
```

- [ ] **Step 3: Update MCP bridge tests**

In `server/tests/test_robot_mcp_bridge.py`, replace:

```python
from robot_mcp_bridge import RobotMCPBridge, RobotMCPError
from voice_runtime.robot_safety import agent_tool_description
```

with:

```python
from robot_control.call_validation import agent_tool_description
from robot_control.mcp_bridge import RobotMCPBridge, RobotMCPError
```

Also rename the out-of-bounds motion argument test to:

```python
def test_rejects_out_of_bounds_motion_arguments_before_mcp_call():
```

- [ ] **Step 4: Run MCP bridge tests and verify they pass**

Run from `server/`:

```bash
uv run pytest tests/test_robot_mcp_bridge.py -v
```

Expected: all MCP bridge tests pass.

- [ ] **Step 5: Run Robot Control import guard and verify it passes**

Run from `server/`:

```bash
uv run pytest tests/test_robot_control_imports.py -v
```

Expected: all Robot Control import tests pass. `mcp_bridge.py` may import `agents` and `mcp`, but must not import `voice_runtime`, `agent_control`, or Pipecat.

- [ ] **Step 6: Commit Robot MCP Bridge extraction**

```bash
git add server/robot_control/mcp_bridge.py server/tests/test_robot_mcp_bridge.py server/robot_mcp_bridge.py
git commit -m "refactor: move robot mcp bridge into robot control"
```

---

## Task 5: Update Agent Orchestration and Agent Backend imports

**Files:**
- Modify: `server/langgraph_robot_agent.py`
- Modify: `server/openai_codex_agent_processor.py`
- Modify: `server/tests/test_langgraph_robot_agent.py`
- Modify: `server/tests/test_openai_codex_agent_processor.py`

- [ ] **Step 1: Update LangGraph imports**

In `server/langgraph_robot_agent.py`, replace:

```python
from robot_mcp_bridge import RobotMCPError
from voice_runtime.robot_context import RobotContextStore
from voice_runtime.robot_safety import (
    RobotSafetyError,
    executable_plan_name,
    execution_result_text,
    structured_robot_error,
)
```

with:

```python
from robot_control.call_validation import (
    RobotCallValidationError,
    executable_plan_name,
    execution_result_text,
    structured_robot_call_error,
)
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPError
```

If Task Policy from `.pi/plans/2026-05-05-minimal-task-policy-layer.md` is already integrated, keep this import in the same file:

```python
from robot_control.task_policy import structured_task_policy_error, validate_task_step
```

- [ ] **Step 2: Update LangGraph exception conversion**

In `server/langgraph_robot_agent.py`, replace:

```python
            safety_error = RobotSafetyError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_error(safety_error), ensure_ascii=False)
```

with:

```python
            validation_error = RobotCallValidationError(
                str(exc),
                correction="Check the robot control server, then retry the robot action.",
            )
            return json.dumps(structured_robot_call_error(validation_error), ensure_ascii=False)
```

- [ ] **Step 3: Update Codex Agent Backend imports**

In `server/openai_codex_agent_processor.py`, replace:

```python
from robot_mcp_bridge import RobotMCPBridge
from voice_runtime.robot_context import RobotContextStore
```

with:

```python
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPBridge
```

- [ ] **Step 4: Update tests importing Robot Context or MCP error**

In `server/tests/test_langgraph_robot_agent.py`, replace:

```python
from voice_runtime.robot_context import RobotContextStore
```

with:

```python
from robot_control.context import RobotContextStore
```

Replace the local import:

```python
from robot_mcp_bridge import RobotMCPError
```

with:

```python
from robot_control.mcp_bridge import RobotMCPError
```

If `server/tests/test_openai_codex_agent_processor.py` imports moved modules, update them to `robot_control.context` or `robot_control.mcp_bridge`.

- [ ] **Step 5: Run Agent Orchestration and Agent Backend tests**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit Agent Control import updates**

```bash
git add server/langgraph_robot_agent.py server/openai_codex_agent_processor.py server/tests/test_langgraph_robot_agent.py server/tests/test_openai_codex_agent_processor.py
git commit -m "refactor: use robot control modules from agent orchestration"
```

---

## Task 6: Update structural import tests and remove legacy references

**Files:**
- Modify: `server/tests/test_orthogonal_imports.py`
- Verify: moved/deleted legacy files

- [ ] **Step 1: Update Voice Runtime import guard**

In `server/tests/test_orthogonal_imports.py`, add `robot_control` to `APP_MODULE_ROOTS`:

```python
APP_MODULE_ROOTS = {
    "agent_processor_factory",
    "bot",
    "codex_auth",
    "codex_backend_client",
    "config",
    "metrics",
    "openai_codex_agent_processor",
    "pipeline_builder",
    "prompts",
    "providers",
    "robot_control",
    "wake",
}
```

Remove `robot_safety.py` from `PURE_MODULES` so the set becomes:

```python
PURE_MODULES = {
    "contracts.py",
    "profiles.py",
    "voice_metrics.py",
    "assembly.py",
}
```

- [ ] **Step 2: Verify legacy modules are gone**

Run from repo root:

```bash
test ! -e server/voice_runtime/robot_safety.py
test ! -e server/voice_runtime/robot_context.py
test ! -e server/robot_mcp_bridge.py
```

Expected: all commands exit 0.

- [ ] **Step 3: Search for stale imports and terminology in code/tests**

Run from repo root:

```bash
rg -n "voice_runtime\.robot_safety|voice_runtime\.robot_context|from robot_mcp_bridge|RobotSafetyError|structured_robot_error|test_voice_runtime_robot_safety" server
```

Expected: no matches.

- [ ] **Step 4: Run structural tests**

Run from `server/`:

```bash
uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py -v
```

Expected: all structural tests pass.

- [ ] **Step 5: Commit structural cleanup**

```bash
git add server/tests/test_orthogonal_imports.py
git commit -m "test: enforce robot control extraction imports"
```

---

## Task 7: Update docs and agent guidance after extraction

**Files:**
- Modify: `AGENTS.md`
- Modify: `CONTEXT.md`
- Modify: `ARCHITECTURE.md`
- Modify: `README.md`
- Modify: `docs/architecture.md`

- [ ] **Step 1: Update `AGENTS.md` project map**

In `AGENTS.md`, replace the Robot Control legacy entries with:

```markdown
- `server/robot_control/` - Robot Control Modules: Task Policy, Robot Call Validation, Robot Tool Adapter, and Robot Context.
```

Remove these legacy-placement bullets if present:

```markdown
- `server/robot_mcp_bridge.py` - Robot MCP tool Adapter used by Codex; target home is `server/robot_control/`.
- `server/voice_runtime/robot_safety.py` - Legacy placement for Robot Call Validation; target home is `server/robot_control/call_validation.py`.
- `server/voice_runtime/robot_context.py` - Legacy placement for Robot Context; target home is `server/robot_control/context.py`.
```

In the Robot Call Validation block, replace:

```markdown
- Target home is `server/robot_control/call_validation.py`; `voice_runtime.robot_safety` is legacy placement.
```

with:

```markdown
- Implementation lives in `server/robot_control/call_validation.py`.
```

In the Robot Context block, replace:

```markdown
- Target home is `server/robot_control/context.py`; `voice_runtime.robot_context` is legacy placement.
```

with:

```markdown
- Implementation lives in `server/robot_control/context.py`.
```

- [ ] **Step 2: Update `CONTEXT.md` flagged ambiguity**

In `CONTEXT.md`, replace:

```markdown
- `voice_runtime.robot_safety` and `voice_runtime.robot_context` are legacy placements; resolved target home is the **Robot Control Module**.
```

with:

```markdown
- `voice_runtime.robot_safety` and `voice_runtime.robot_context` were legacy placements; resolved implementation home is the **Robot Control Module**.
```

- [ ] **Step 3: Update `ARCHITECTURE.md` robot_control section**

In `ARCHITECTURE.md`, replace:

```markdown
`voice_runtime.robot_safety` and `voice_runtime.robot_context` are legacy placements. Their target home is `robot_control`.

Extract `robot_control` before extracting `agent_control`, then clean up legacy top-level placements.
```

with:

```markdown
Robot Call Validation, Robot Context, Task Policy, and the Robot Tool Adapter live under `robot_control`.

After `robot_control` extraction, extract `agent_control`, then clean up any remaining legacy top-level placements.
```

- [ ] **Step 4: Update `README.md` and `docs/architecture.md` if they mention legacy modules**

Run from repo root:

```bash
rg -n "voice_runtime\.robot_safety|voice_runtime\.robot_context|robot_mcp_bridge|Robot Safety|locally enforced|locally validated" README.md docs/architecture.md
```

Expected before edits: only current references that need updating.

Replace stale references with:

```markdown
Robot movement safety is delegated to MoveIt planning/execution and the robot simulation stack. Local Robot Call Validation lives in `robot_control.call_validation` and exists for clearer errors, not as the source of movement safety.
```

- [ ] **Step 5: Run doc search and verify stale terms are gone from current docs**

Run from repo root:

```bash
rg -n "voice_runtime\.robot_safety|voice_runtime\.robot_context|from robot_mcp_bridge|Robot Safety coverage|locally enforced|locally validated" AGENTS.md CONTEXT.md ARCHITECTURE.md README.md docs/architecture.md
```

Expected: no matches.

- [ ] **Step 6: Commit docs update**

```bash
git add AGENTS.md CONTEXT.md ARCHITECTURE.md README.md docs/architecture.md
git commit -m "docs: align robot control extraction guidance"
```

---

## Task 8: Full verification and final scope review

**Files:**
- Verify: all modified files

- [ ] **Step 1: Run targeted Robot Control tests**

Run from `server/`:

```bash
uv run pytest tests/test_robot_call_validation.py tests/test_robot_context.py tests/test_robot_mcp_bridge.py tests/test_robot_task_policy.py tests/test_robot_control_imports.py -v
```

Expected: all targeted Robot Control tests pass.

- [ ] **Step 2: Run Agent/Voice integration tests affected by imports**

Run from `server/`:

```bash
uv run pytest tests/test_langgraph_robot_agent.py tests/test_openai_codex_agent_processor.py tests/test_agent_processor_factory.py tests/test_pipeline_builder.py tests/test_orthogonal_imports.py -v
```

Expected: all integration and structural tests pass.

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
git diff -- server/pipeline_builder.py server/bot.py server/voice_runtime/assembly.py server/voice_runtime/wake_command.py server/voice_runtime/agent_turn.py
```

Expected:

- `pipeline_builder.py` unchanged except import paths only if necessary.
- `bot.py` unchanged.
- `voice_runtime/assembly.py` unchanged.
- `voice_runtime/wake_command.py` unchanged.
- `voice_runtime/agent_turn.py` unchanged.
- No Pipecat processor-ordering changes.
- No STT/TTS/wake behavior changes.
- No new provider dependencies.

- [ ] **Step 5: Search final tree for stale Robot Control placement**

Run from repo root:

```bash
rg -n "voice_runtime\.robot_safety|voice_runtime\.robot_context|from robot_mcp_bridge|RobotSafetyError|structured_robot_error|test_voice_runtime_robot_safety" server AGENTS.md CONTEXT.md ARCHITECTURE.md README.md docs/architecture.md
```

Expected: no matches.

- [ ] **Step 6: Final commit if any verification-only fixes were needed**

If Step 3 or Step 5 required small fixes, commit them:

```bash
git add server AGENTS.md CONTEXT.md ARCHITECTURE.md README.md docs/architecture.md
git commit -m "chore: finish robot control extraction"
```

Skip this commit if no files changed after Task 7.

---

## Completion checklist

- [ ] `server/robot_control/call_validation.py` owns Robot Call Validation.
- [ ] `server/robot_control/context.py` owns Robot Context.
- [ ] `server/robot_control/mcp_bridge.py` owns Robot Tool Adapter.
- [ ] `server/robot_control/task_policy.py` remains pure and inside Robot Control.
- [ ] `server/voice_runtime/robot_safety.py` is deleted.
- [ ] `server/voice_runtime/robot_context.py` is deleted.
- [ ] `server/robot_mcp_bridge.py` is deleted.
- [ ] Runtime imports use `robot_control.*` modules.
- [ ] Voice Runtime does not import Robot Control.
- [ ] Robot Control does not import Voice Runtime or Agent Control.
- [ ] Current docs no longer describe Robot Control modules as legacy placements.
- [ ] Full pytest, ruff, and pyright verification pass.
