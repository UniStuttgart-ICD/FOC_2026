# Deepen Agent Control Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Agent Control implementation behind a real `server/agent_control/` Module while keeping `pipeline_builder.py` as the app composition root.

**Architecture:** `agent_control` owns the LangChain API Backend, Agent Orchestration, Robot Agent Prompt, and chat-model construction. `pipeline_builder.py` may import the Agent Control factory, but Voice Runtime and Robot Control must not import Agent Control. Remove legacy root Agent Control files after imports and tests move.

**Tech Stack:** Python 3.12, pytest, ruff, pyright, LangChain, LangGraph, Pipecat, local structural import tests.

---

## Current State

- `server/agent_control/` already exists in the working tree with `prompts.py` and prompt parts.
- `server/prompts.py` is deleted in the working tree.
- `server/langgraph_robot_agent.py` and `server/tests/test_prompts.py` already import `agent_control.prompts`.
- Root Agent Control files still exist: `agent_model_factory.py`, `agent_processor_factory.py`, `langchain_agent_processor.py`, `langgraph_robot_agent.py`.
- `pipeline_builder.py`, `bot.py`, `config.py`, `runtime_profiles.toml`, `pyproject.toml`, and `uv.lock` stay at app/project root.

## File Map

- Keep: `server/pipeline_builder.py` as the app composition root.
- Keep: `server/bot.py` as the runner/lifecycle shell.
- Keep: `server/config.py` as app config facade over `voice_runtime.profiles`.
- Keep: `server/runtime_profiles.toml` as concrete app configuration.
- Existing: `server/agent_control/__init__.py` package marker.
- Existing: `server/agent_control/prompts.py` prompt renderer.
- Existing: `server/agent_control/prompt_parts/*.md` prompt content.
- Move: `server/agent_model_factory.py` -> `server/agent_control/model_factory.py`.
- Move: `server/langchain_agent_processor.py` -> `server/agent_control/langchain_agent_processor.py`.
- Move: `server/langgraph_robot_agent.py` -> `server/agent_control/langgraph_robot_agent.py`.
- Move: `server/agent_processor_factory.py` -> `server/agent_control/factory.py`.
- Modify: tests and manual scripts to import from `agent_control.*`.
- Modify: `server/tests/test_orthogonal_imports.py` to enforce the new Module seam.
- Modify: `AGENTS.md` and `ARCHITECTURE.md` to remove "target home" language for Agent Control.

## Target Interface

The external Agent Control seam for the app composition root is:

```python
from agent_control.factory import create_agent_processor
```

The test and eval seams are:

```python
from agent_control.model_factory import build_agent_chat_model
from agent_control.langchain_agent_processor import LangChainAgentProcessor
from agent_control.langgraph_robot_agent import LangGraphRobotAgent
from agent_control.prompts import SYSTEM_PROMPT
```

---

### Task 1: Lock Agent Control Package Shape With a Failing Structural Test

**Files:**
- Modify: `server/tests/test_orthogonal_imports.py`

- [ ] **Step 1: Write the failing test**

Add these constants near the existing path constants:

```python
AGENT_CONTROL_DIR = SERVER_DIR / "agent_control"

REQUIRED_AGENT_CONTROL_MODULES = {
    "__init__.py",
    "factory.py",
    "langchain_agent_processor.py",
    "langgraph_robot_agent.py",
    "model_factory.py",
    "prompts.py",
}

DELETED_LEGACY_AGENT_CONTROL_ROOTS = {
    SERVER_DIR / "agent_model_factory.py",
    SERVER_DIR / "agent_processor_factory.py",
    SERVER_DIR / "langchain_agent_processor.py",
    SERVER_DIR / "langgraph_robot_agent.py",
    SERVER_DIR / "prompts.py",
}
```

Add these tests:

```python
def test_agent_control_package_contains_agent_control_modules() -> None:
    missing = [
        name
        for name in sorted(REQUIRED_AGENT_CONTROL_MODULES)
        if not (AGENT_CONTROL_DIR / name).exists()
    ]

    assert not missing, f"agent_control is missing module file(s): {missing}"


def test_legacy_agent_control_root_modules_are_deleted() -> None:
    remaining = [str(path.relative_to(SERVER_DIR)) for path in DELETED_LEGACY_AGENT_CONTROL_ROOTS if path.exists()]

    assert not remaining, f"legacy root Agent Control module(s) still exist: {remaining}"
```

- [ ] **Step 2: Run the test and verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_orthogonal_imports.py::test_agent_control_package_contains_agent_control_modules tests/test_orthogonal_imports.py::test_legacy_agent_control_root_modules_are_deleted -q
```

Expected: FAIL because `factory.py`, `langchain_agent_processor.py`, `langgraph_robot_agent.py`, and `model_factory.py` are not all inside `agent_control`, and root legacy files still exist.

- [ ] **Step 3: Do not implement in this task**

Leave the test failing. The next tasks move one Module at a time and turn this structural test green at the end.

---

### Task 2: Move Chat Model Construction Into Agent Control

**Files:**
- Move: `server/agent_model_factory.py` -> `server/agent_control/model_factory.py`
- Modify: `server/tests/test_agent_model_factory.py`
- Modify: `server/model_eval/runner.py`
- Modify: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- Modify: `server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py`
- Modify: `server/tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py`

- [ ] **Step 1: Write the failing import test**

In `server/tests/test_agent_model_factory.py`, change:

```python
from agent_model_factory import build_agent_chat_model
```

to:

```python
from agent_control.model_factory import build_agent_chat_model
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_control.model_factory'`.

- [ ] **Step 3: Move the implementation**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git mv server/agent_model_factory.py server/agent_control/model_factory.py
```

- [ ] **Step 4: Update all chat model factory imports**

Change these imports:

```python
from agent_model_factory import build_agent_chat_model
```

to:

```python
from agent_control.model_factory import build_agent_chat_model
```

Apply this in:

- `server/model_eval/runner.py`
- `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- `server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py`
- `server/tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py`

- [ ] **Step 5: Run focused tests and verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py tests/test_model_eval_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add server/agent_control/model_factory.py server/tests/test_agent_model_factory.py server/model_eval/runner.py server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py server/tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py
git commit -m "refactor: move agent model factory into agent control"
```

---

### Task 3: Move LangGraph Agent Orchestration Into Agent Control

**Files:**
- Move: `server/langgraph_robot_agent.py` -> `server/agent_control/langgraph_robot_agent.py`
- Modify: `server/agent_control/langchain_agent_processor.py` after Task 4 if it already exists
- Modify: `server/langchain_agent_processor.py` if Task 4 has not run yet
- Modify: `server/tests/test_langgraph_robot_agent.py`

- [ ] **Step 1: Write failing import tests**

In `server/tests/test_langgraph_robot_agent.py`, replace each local import:

```python
from langgraph_robot_agent import LangGraphRobotAgent
```

with:

```python
from agent_control.langgraph_robot_agent import LangGraphRobotAgent
```

Replace monkeypatch targets:

```python
monkeypatch.setattr("langgraph_robot_agent.logger", fake_logger)
```

with:

```python
monkeypatch.setattr("agent_control.langgraph_robot_agent.logger", fake_logger)
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langgraph_robot_agent.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_control.langgraph_robot_agent'`.

- [ ] **Step 3: Move the implementation**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git mv server/langgraph_robot_agent.py server/agent_control/langgraph_robot_agent.py
```

- [ ] **Step 4: Update backend imports**

Wherever the backend imports the graph, change:

```python
from langgraph_robot_agent import LangGraphRobotAgent
```

to:

```python
from agent_control.langgraph_robot_agent import LangGraphRobotAgent
```

Apply this in the current backend file:

- `server/langchain_agent_processor.py`, or
- `server/agent_control/langchain_agent_processor.py` if Task 4 has already moved it.

Keep this existing prompt import:

```python
from agent_control.prompts import SYSTEM_PROMPT
```

- [ ] **Step 5: Run focused tests and verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langgraph_robot_agent.py tests/test_langchain_agent_processor.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add server/agent_control/langgraph_robot_agent.py server/tests/test_langgraph_robot_agent.py server/langchain_agent_processor.py server/agent_control/langchain_agent_processor.py
git commit -m "refactor: move langgraph robot agent into agent control"
```

If one of the staged backend paths does not exist, omit it from `git add`.

---

### Task 4: Move LangChain API Backend Into Agent Control

**Files:**
- Move: `server/langchain_agent_processor.py` -> `server/agent_control/langchain_agent_processor.py`
- Modify: `server/tests/test_langchain_agent_processor.py`
- Modify: `server/tests/test_agent_processor_factory.py`
- Modify: `server/tests/test_moveit_agent_behavior_contracts.py`
- Modify: `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- Modify: `server/model_eval/runner.py`

- [ ] **Step 1: Write failing import tests**

In `server/tests/test_langchain_agent_processor.py`, change:

```python
from langchain_agent_processor import LangChainAgentProcessor
```

to:

```python
from agent_control.langchain_agent_processor import LangChainAgentProcessor
```

Change monkeypatch targets:

```python
monkeypatch.setattr("langchain_agent_processor.RobotMCPBridge", CreatedBridge)
monkeypatch.setattr("langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
```

to:

```python
monkeypatch.setattr("agent_control.langchain_agent_processor.RobotMCPBridge", CreatedBridge)
monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langchain_agent_processor.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_control.langchain_agent_processor'`.

- [ ] **Step 3: Move the implementation**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git mv server/langchain_agent_processor.py server/agent_control/langchain_agent_processor.py
```

- [ ] **Step 4: Update all backend imports**

Change:

```python
from langchain_agent_processor import LangChainAgentProcessor
```

to:

```python
from agent_control.langchain_agent_processor import LangChainAgentProcessor
```

Apply this in:

- `server/tests/test_agent_processor_factory.py`
- `server/tests/test_moveit_agent_behavior_contracts.py`
- `server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py`
- `server/model_eval/runner.py`

- [ ] **Step 5: Run focused tests and verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langchain_agent_processor.py tests/test_moveit_agent_behavior_contracts.py tests/test_model_eval_runner.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add server/agent_control/langchain_agent_processor.py server/tests/test_langchain_agent_processor.py server/tests/test_agent_processor_factory.py server/tests/test_moveit_agent_behavior_contracts.py server/tests/live_robot_smoke/manual_live_llm_robot_smoke.py server/model_eval/runner.py
git commit -m "refactor: move langchain backend into agent control"
```

---

### Task 5: Move Agent Processor Factory Into Agent Control

**Files:**
- Move: `server/agent_processor_factory.py` -> `server/agent_control/factory.py`
- Modify: `server/pipeline_builder.py`
- Modify: `server/tests/test_agent_processor_factory.py`
- Modify: `server/tests/test_pipeline_builder.py`
- Modify: `server/tests/test_orthogonal_imports.py`

- [ ] **Step 1: Write failing factory import test**

In `server/tests/test_agent_processor_factory.py`, change:

```python
from agent_processor_factory import create_agent_processor
```

to:

```python
from agent_control.factory import create_agent_processor
```

Change monkeypatch targets:

```python
"agent_processor_factory.build_agent_chat_model"
"agent_processor_factory.LangChainAgentProcessor"
"agent_processor_factory.AgentTurnProcessor"
```

to:

```python
"agent_control.factory.build_agent_chat_model"
"agent_control.factory.LangChainAgentProcessor"
"agent_control.factory.AgentTurnProcessor"
```

- [ ] **Step 2: Run the focused test and verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_processor_factory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_control.factory'`.

- [ ] **Step 3: Move the implementation**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
git mv server/agent_processor_factory.py server/agent_control/factory.py
```

- [ ] **Step 4: Update factory imports inside the moved file**

In `server/agent_control/factory.py`, change:

```python
from agent_model_factory import build_agent_chat_model
from langchain_agent_processor import LangChainAgentProcessor
```

to:

```python
from agent_control.model_factory import build_agent_chat_model
from agent_control.langchain_agent_processor import LangChainAgentProcessor
```

- [ ] **Step 5: Update app composition root**

In `server/pipeline_builder.py`, change:

```python
from agent_processor_factory import create_agent_processor
```

to:

```python
from agent_control.factory import create_agent_processor
```

- [ ] **Step 6: Update pipeline builder monkeypatch targets**

In `server/tests/test_pipeline_builder.py`, change:

```python
"pipeline_builder.create_agent_processor"
```

only if the tests patch that imported symbol. Keep the target as:

```python
"pipeline_builder.create_agent_processor"
```

because `pipeline_builder.py` still imports the symbol into its own module namespace.

- [ ] **Step 7: Run focused tests and verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_processor_factory.py tests/test_pipeline_builder.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git add server/agent_control/factory.py server/pipeline_builder.py server/tests/test_agent_processor_factory.py server/tests/test_pipeline_builder.py
git commit -m "refactor: move agent processor factory into agent control"
```

---

### Task 6: Enforce Agent Control Import Locality

**Files:**
- Modify: `server/tests/test_orthogonal_imports.py`
- Modify: `server/tests/test_robot_control_imports.py`

- [ ] **Step 1: Write failing Agent Control import locality test**

In `server/tests/test_orthogonal_imports.py`, add:

```python
AGENT_CONTROL_FORBIDDEN_ROOTS = {
    "bot",
    "config",
    "metrics",
    "pipeline_builder",
    "providers",
    "wake",
    "wake_tuning",
}
```

Add:

```python
def test_agent_control_modules_do_not_import_app_or_voice_runtime_adapters() -> None:
    for path in AGENT_CONTROL_DIR.glob("*.py"):
        imported = _import_roots(path)
        forbidden = imported & AGENT_CONTROL_FORBIDDEN_ROOTS

        assert not forbidden, f"{path.name} imports forbidden module(s): {sorted(forbidden)}"
```

- [ ] **Step 2: Run locality tests and verify red if imports still leak**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py -q
```

Expected before cleanup: FAIL if any Agent Control file still imports an app root or if legacy root Agent Control modules remain.

- [ ] **Step 3: Fix imports minimally**

Allowed Agent Control imports:

```python
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.agent_turn import AgentTurnProcessor
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPBridge
from robot_control.call_validation import RobotCallValidationError
from robot_control.task_policy import validate_task_step
from process_trace import NoopProcessTracer, ProcessTracer
```

Disallowed Agent Control imports:

```python
from config import ...
from pipeline_builder import ...
from providers import ...
from metrics import ...
from wake import ...
```

If a disallowed import appears, move the construction responsibility back to `pipeline_builder.py` or `agent_control.factory.py` and pass plain config/data into the lower-level Module.

- [ ] **Step 4: Run locality tests and verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add server/tests/test_orthogonal_imports.py server/tests/test_robot_control_imports.py
git commit -m "test: enforce agent control module locality"
```

---

### Task 7: Remove All Legacy Root Import References

**Files:**
- Modify: any file found by the search below.

- [ ] **Step 1: Run stale-reference search**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
rg -n "from (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)|import (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)" server AGENTS.md ARCHITECTURE.md CONTEXT.md README.md docs
```

Expected before cleanup: any remaining stale root imports are listed.

- [ ] **Step 2: Replace stale imports**

Use these replacements:

```python
from agent_model_factory import build_agent_chat_model
```

becomes:

```python
from agent_control.model_factory import build_agent_chat_model
```

```python
from agent_processor_factory import create_agent_processor
```

becomes:

```python
from agent_control.factory import create_agent_processor
```

```python
from langchain_agent_processor import LangChainAgentProcessor
```

becomes:

```python
from agent_control.langchain_agent_processor import LangChainAgentProcessor
```

```python
from langgraph_robot_agent import LangGraphRobotAgent
```

becomes:

```python
from agent_control.langgraph_robot_agent import LangGraphRobotAgent
```

```python
from prompts import SYSTEM_PROMPT
```

becomes:

```python
from agent_control.prompts import SYSTEM_PROMPT
```

- [ ] **Step 3: Re-run stale-reference search**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
rg -n "from (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)|import (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)" server AGENTS.md ARCHITECTURE.md CONTEXT.md README.md docs
```

Expected: no matches except historical docs under `docs/plans/2026-03-19-voice-robot-agent-plan.md` or archived implementation plans. If historical docs match, leave them unchanged.

- [ ] **Step 4: Run broad affected tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py tests/test_agent_processor_factory.py tests/test_langchain_agent_processor.py tests/test_langgraph_robot_agent.py tests/test_pipeline_builder.py tests/test_model_eval_runner.py tests/test_moveit_agent_behavior_contracts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add server AGENTS.md ARCHITECTURE.md CONTEXT.md README.md docs
git commit -m "refactor: remove legacy root agent control imports"
```

---

### Task 8: Update Architecture Documentation

**Files:**
- Modify: `AGENTS.md`
- Modify: `ARCHITECTURE.md`
- Modify: `CONTEXT.md` only if domain terms change.

- [ ] **Step 1: Write the doc expectation check**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
rg -n "target home is `server/agent_control/`|Current API-key LangChain Agent Backend adapter|Current LangGraph Agent Orchestration" AGENTS.md ARCHITECTURE.md CONTEXT.md
```

Expected before doc cleanup: matches in `AGENTS.md` or `ARCHITECTURE.md`.

- [ ] **Step 2: Update `AGENTS.md` project map**

Replace the old Agent Control bullets with:

```markdown
- `server/agent_control/` - Agent Control Module: native LangChain API Backend, LangGraph Agent Orchestration, Robot Agent Prompt, and Agent Turn factory.
```

Keep:

```markdown
- `server/pipeline_builder.py` - App composition root for concrete adapters and pipeline task assembly.
```

- [ ] **Step 3: Update `ARCHITECTURE.md` Agent Control section**

Ensure the Agent Control section says:

```markdown
`agent_control` is the Module for API-key-backed LangChain Agent Orchestration.

It contains:

- **LangChain API Backend**: builds native LangChain chat models and satisfies the Agent Turn backend seam.
- **Agent Orchestration**: the LangGraph loop that calls the model, executes robot tools through Robot Control, observes Robot Context, and repeats until done or blocked.
- **Robot Agent Prompt**: the prompt renderer and prompt parts aligned with Robot Call Validation and Robot Tool Adapter feedback.

**API Boundary:** `agent_control` satisfies `voice_runtime.AgentBackend` and depends on `robot_control` for robot execution. It must not own Pipecat transport, audio frames, wake handling, STT/TTS, interruption behavior, or pipeline ordering.
```

- [ ] **Step 4: Re-run doc expectation check**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
rg -n "target home is `server/agent_control/`|Current API-key LangChain Agent Backend adapter|Current LangGraph Agent Orchestration" AGENTS.md ARCHITECTURE.md CONTEXT.md
```

Expected: no matches.

- [ ] **Step 5: Commit**

Run:

```powershell
git add AGENTS.md ARCHITECTURE.md CONTEXT.md
git commit -m "docs: document agent control module extraction"
```

---

### Task 9: Full Verification

**Files:**
- All changed files.

- [ ] **Step 1: Run focused Agent Control tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py tests/test_agent_processor_factory.py tests/test_langchain_agent_processor.py tests/test_langgraph_robot_agent.py tests/test_prompts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run architecture/import tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_orthogonal_imports.py tests/test_robot_control_imports.py tests/test_pipeline_builder.py -q
```

Expected: PASS.

- [ ] **Step 3: Run model eval tests affected by import moves**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_model_eval_runner.py tests/test_model_eval_cli.py tests/test_model_eval_evidence.py -q
```

Expected: PASS.

- [ ] **Step 4: Run full server test suite**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest -q
```

Expected: PASS, with live tests skipped unless explicitly enabled.

- [ ] **Step 5: Run lint and type checks**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run ruff check .
uv run pyright .
```

Expected: PASS.

- [ ] **Step 6: Run final stale-reference search**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
rg -n "from (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)|import (agent_model_factory|agent_processor_factory|langchain_agent_processor|langgraph_robot_agent|prompts)" server AGENTS.md ARCHITECTURE.md CONTEXT.md README.md docs
```

Expected: no matches outside historical archived plans.

- [ ] **Step 7: Inspect final file placement**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent
Get-ChildItem server/agent_control -Recurse | Select-Object FullName
Test-Path server/agent_model_factory.py
Test-Path server/agent_processor_factory.py
Test-Path server/langchain_agent_processor.py
Test-Path server/langgraph_robot_agent.py
Test-Path server/prompts.py
```

Expected: Agent Control files are under `server/agent_control`; each `Test-Path` prints `False`.

## Acceptance Criteria

- `server/agent_control/` owns LangChain API Backend, Agent Orchestration, Robot Agent Prompt, and chat-model construction.
- `pipeline_builder.py` imports `create_agent_processor` from `agent_control.factory`.
- No production code imports root `agent_model_factory`, `agent_processor_factory`, `langchain_agent_processor`, `langgraph_robot_agent`, or `prompts`.
- Root legacy Agent Control files are deleted.
- Structural import tests enforce Agent Control Locality.
- Voice Runtime does not import Agent Control or Robot Control.
- Robot Control does not import Agent Control or Voice Runtime.
- Docs describe `agent_control` as implemented, not as a target home.
- Focused tests, full pytest, ruff, and pyright pass.

## Out Of Scope

- Moving `providers.py` into Voice Runtime.
- Changing wake tuning settings or wake tuning logs.
- Changing model/provider behaviour in `runtime_profiles.toml`.
- Changing Robot Control policy, validation, MCP, or MoveIt behaviour.
- Refactoring `metrics.py` into Voice Runtime.
