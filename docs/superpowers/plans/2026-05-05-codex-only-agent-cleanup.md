# Codex-Only Agent Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove Claude agent support and make OpenAI Codex OAuth the only supported agent backend.

**Architecture:** Keep Pipecat as the runtime and pipeline owner. Preserve `AgentTurnProcessor` and `AgentBackend` as the agent seam, but make `OpenAICodexAgentProcessor` the only concrete backend. Convert local debug profiles to Codex so later LangGraph work starts from a clean Codex-only baseline.

**Tech Stack:** Python, Pipecat, OpenAI Codex OAuth backend, MCP robot bridge, pytest, ruff, pyright, uv.

---

## File structure

- Delete: `server/claude_agent_processor.py` — removed legacy Claude backend Adapter.
- Modify: `server/agent_processor_factory.py` — remove Claude import/branch; keep Codex factory only.
- Modify: `server/voice_runtime/profiles.py` — remove `claude` provider literal and parser default.
- Modify: `server/runtime_profiles.toml` — convert `local_current` and `no_wake_debug` to Codex.
- Modify: `server/pyproject.toml` — remove `claude-agent-sdk` dependency.
- Modify: `server/uv.lock` — refresh after dependency removal.
- Modify: `server/.env.example` — remove Claude auth/model variables.
- Modify: `README.md` — document Codex-only auth and safety posture.
- Modify: `server/tests/test_agent_processor_factory.py` — remove Claude factory test.
- Modify: `server/tests/test_voice_runtime_profiles.py` — make fixtures Codex-only and add rejection coverage for legacy `claude`.
- Modify: `server/tests/test_config.py` — remove `claude` from config fixtures so unrelated validation tests still exercise their intended errors.

Do not edit `server/.env`; it may contain local secrets and is git-ignored.

---

### Task 1: Update tests to describe Codex-only behavior

**Files:**
- Modify: `server/tests/test_agent_processor_factory.py`
- Modify: `server/tests/test_voice_runtime_profiles.py`
- Modify: `server/tests/test_config.py`

- [ ] **Step 1: Replace the factory tests with Codex-only coverage**

Replace the contents of `server/tests/test_agent_processor_factory.py` with:

```python
from pipecat.processors.frame_processor import FrameProcessor

from agent_processor_factory import create_agent_processor
from config import AgentConfig
from voice_runtime.agent_turn import AgentTurnProcessor


def test_creates_openai_codex_agent_turn_processor():
    processor = create_agent_processor(
        AgentConfig(provider="openai_codex_oauth", model="gpt-5.5"),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor, FrameProcessor)
```

- [ ] **Step 2: Update the shared profile fixture to use Codex for `no_wake_debug`**

In `server/tests/test_voice_runtime_profiles.py`, inside `_write_profiles()`, replace:

```toml
[profiles.no_wake_debug.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
```

with:

```toml
[profiles.no_wake_debug.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

- [ ] **Step 3: Add explicit rejection coverage for legacy Claude provider**

Add this test to `server/tests/test_voice_runtime_profiles.py` after `test_local_profile_has_no_cloud_stt_tts_env_requirements`:

```python
def test_legacy_claude_agent_provider_is_rejected(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.legacy]
category = "local_debug"
[profiles.legacy.wake]
provider = "none"
[profiles.legacy.emergency_stop]
enabled = false
[profiles.legacy.stt]
provider = "whisper"
model = "base"
[profiles.legacy.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.legacy.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
[profiles.legacy.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.legacy.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="provider must be one of"):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="legacy",
        )
```

- [ ] **Step 4: Replace remaining non-rejection `claude` fixtures in profile tests**

In `server/tests/test_voice_runtime_profiles.py`, replace each `claude` agent block in tests that are not `test_legacy_claude_agent_provider_is_rejected`:

```toml
[profiles.bad.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
```

with:

```toml
[profiles.bad.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

This keeps wake/emergency/threshold tests focused on their intended validation errors instead of failing earlier on the agent provider.

- [ ] **Step 5: Replace `claude` fixtures in config tests**

In `server/tests/test_config.py`, replace every test fixture block like this:

```toml
[profiles.local_current.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
```

with:

```toml
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

Also replace any same-shaped block under another profile name, preserving the profile prefix.

- [ ] **Step 6: Run the focused tests and verify they fail for production-code reasons**

Run:

```bash
cd server
uv run pytest tests/test_agent_processor_factory.py tests/test_voice_runtime_profiles.py tests/test_config.py -q
```

Expected: failures because production code still accepts/imports Claude and profile literals still include `claude`. No syntax errors should appear.

- [ ] **Step 7: Commit the test changes**

```bash
git add server/tests/test_agent_processor_factory.py server/tests/test_voice_runtime_profiles.py server/tests/test_config.py
git commit -m "test: define codex-only agent behavior"
```

---

### Task 2: Remove Claude provider support from code

**Files:**
- Delete: `server/claude_agent_processor.py`
- Modify: `server/agent_processor_factory.py`
- Modify: `server/voice_runtime/profiles.py`

- [ ] **Step 1: Delete the Claude backend Adapter**

Run:

```bash
git rm server/claude_agent_processor.py
```

Expected: file is staged for deletion.

- [ ] **Step 2: Replace the agent factory with Codex-only construction**

Replace `server/agent_processor_factory.py` with:

```python
from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor

from config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnProcessor


def create_agent_processor(config: AgentConfig, *, mcp_server_url: str) -> FrameProcessor:
    if config.provider != "openai_codex_oauth":
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=OpenAICodexAgentProcessor(mcp_server_url=mcp_server_url, model=config.model)
    )
```

- [ ] **Step 3: Remove `claude` from profile provider typing**

In `server/voice_runtime/profiles.py`, replace:

```python
AgentProvider = Literal["claude", "openai_codex_oauth"]
```

with:

```python
AgentProvider = Literal["openai_codex_oauth"]
```

Then replace:

```python
_AGENT_PROVIDERS = {"claude", "openai_codex_oauth"}
```

with:

```python
_AGENT_PROVIDERS = {"openai_codex_oauth"}
```

- [ ] **Step 4: Simplify the agent profile parser default**

In `server/voice_runtime/profiles.py`, replace `_parse_agent()` with:

```python
def _parse_agent(table: dict[str, Any]) -> AgentProfile:
    provider = cast(AgentProvider, _literal(table, "provider", _AGENT_PROVIDERS))
    return AgentProfile(provider=provider, model=_string(table, "model", "gpt-5.4-mini"))
```

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd server
uv run pytest tests/test_agent_processor_factory.py tests/test_voice_runtime_profiles.py tests/test_config.py -q
```

Expected: tests pass or fail only because runtime profile TOML still contains Claude entries. If failures mention `local_current` or `no_wake_debug` from `runtime_profiles.toml`, continue to Task 3.

- [ ] **Step 6: Run import ordering lint on changed Python files**

Run:

```bash
cd server
uv run ruff check agent_processor_factory.py voice_runtime/profiles.py tests/test_agent_processor_factory.py tests/test_voice_runtime_profiles.py tests/test_config.py
```

Expected: PASS. If import sorting fails, run:

```bash
uv run ruff check --fix agent_processor_factory.py voice_runtime/profiles.py tests/test_agent_processor_factory.py tests/test_voice_runtime_profiles.py tests/test_config.py
```

Then rerun the check.

- [ ] **Step 7: Commit the code removal**

```bash
git add server/agent_processor_factory.py server/voice_runtime/profiles.py
git add -u server/claude_agent_processor.py
git commit -m "refactor: remove claude agent provider"
```

---

### Task 3: Convert runtime profiles and dependencies to Codex-only

**Files:**
- Modify: `server/runtime_profiles.toml`
- Modify: `server/pyproject.toml`
- Modify: `server/uv.lock`

- [ ] **Step 1: Convert `local_current` to Codex**

In `server/runtime_profiles.toml`, replace:

```toml
[profiles.local_current.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
```

with:

```toml
[profiles.local_current.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

- [ ] **Step 2: Convert `no_wake_debug` to Codex**

In `server/runtime_profiles.toml`, replace:

```toml
[profiles.no_wake_debug.agent]
provider = "claude"
model = "claude-haiku-4-5-20251001"
```

with:

```toml
[profiles.no_wake_debug.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
```

- [ ] **Step 3: Remove the Claude SDK dependency**

In `server/pyproject.toml`, remove this dependency line:

```toml
    "claude-agent-sdk<0.1.49",
```

Keep all Codex/OpenAI dependencies.

- [ ] **Step 4: Refresh the uv lockfile**

Run:

```bash
cd server
uv lock
```

Expected: `server/uv.lock` updates and no dependency resolution error occurs.

- [ ] **Step 5: Verify current runtime profiles load**

Run:

```bash
cd server
uv run pytest tests/test_voice_runtime_profiles.py::test_default_profile_path_and_name_load_current_app_profile -q
```

Expected: PASS.

- [ ] **Step 6: Run dependency/import smoke checks**

Run:

```bash
cd server
uv run python - <<'PY'
from agent_processor_factory import create_agent_processor
from config import AgentConfig, load_runtime_config
from voice_runtime.profiles import load_runtime_profile

profile = load_runtime_profile(profile_name="local_current")
assert profile.agent.provider == "openai_codex_oauth"
processor = create_agent_processor(
    AgentConfig(provider="openai_codex_oauth", model=profile.agent.model),
    mcp_server_url=profile.mcp_robot_url,
)
assert processor is not None
print("codex-only smoke check passed")
PY
```

Expected output includes:

```text
codex-only smoke check passed
```

- [ ] **Step 7: Commit profile and dependency cleanup**

```bash
git add server/runtime_profiles.toml server/pyproject.toml server/uv.lock
git commit -m "chore: make runtime profiles codex-only"
```

---

### Task 4: Clean up docs and example environment

**Files:**
- Modify: `README.md`
- Modify: `server/.env.example`
- Verify: `AGENTS.md`

- [ ] **Step 1: Remove Claude entries from `.env.example`**

In `server/.env.example`, remove this block:

```dotenv
# Claude fallback profile
CLAUDE_MODEL="claude-haiku-4-5-20251001"
```

Also remove this block:

```dotenv
# Claude Agent SDK / Anthropic
# Auth: Uses OAuth via `claude auth login` (Claude Max subscription).
# Run the bundled CLI's auth if not already logged in:
#   .venv/Lib/site-packages/claude_agent_sdk/_bundled/claude.exe auth login
# No API key needed when using OAuth.
```

Do not edit `server/.env`.

- [ ] **Step 2: Update README local profile auth text**

In `README.md`, replace:

```markdown
`local_current` and `no_wake_debug` use local STT/TTS, but still use the Claude cloud agent. Authenticate with `claude auth login` and keep Claude plus the profile MCP URL reachable.
```

with:

```markdown
`local_current` and `no_wake_debug` use local STT/TTS with the same OpenAI Codex OAuth agent backend as the benchmark profiles. Keep Pi's `openai-codex` OAuth profile and the configured robot MCP URL reachable.
```

- [ ] **Step 3: Update README robot safety text**

In `README.md`, replace:

```markdown
Codex through `RobotMCPBridge` is locally enforced. Direct Claude MCP is prompt-only unless a safe MCP proxy Adapter is added.
```

with:

```markdown
All robot tool calls go through Codex and `RobotMCPBridge`, where canonical `moveit_*` calls are locally validated by `voice_runtime.robot_safety` before reaching the MCP server.
```

- [ ] **Step 4: Confirm project agent guidance already reflects the cleanup**

Read `AGENTS.md` and confirm it includes these points:

```markdown
- Pipecat owns transport, audio frames, wake, STT, TTS, interruption behavior, and pipeline backpressure.
- The target architecture is Codex-only. Do not add new Claude support.
- Add LangGraph behind the existing `AgentBackend` seam before considering deeper pipeline changes.
```

If `AGENTS.md` is missing any of those statements, add them inside specific `<important if="...">` blocks rather than as broad always-on rules.

- [ ] **Step 5: Search tracked docs/source for stale Claude setup language**

Run:

```bash
git grep -n -i "claude\|claude-agent-sdk" -- README.md server ':!server/.env' ':!server/.venv' ':!server/.pytest_cache' ':!server/.ruff_cache'
```

Expected: no matches in `README.md` or tracked `server/` source files. If matches remain in tests, profiles, `.env.example`, or code, remove them unless they are the dedicated rejection test for legacy `claude` provider.

- [ ] **Step 6: Commit docs cleanup**

```bash
git add README.md server/.env.example AGENTS.md
git commit -m "docs: document codex-only agent setup"
```

---

### Task 5: Final verification

**Files:**
- Verify all changed files.

- [ ] **Step 1: Run full tests**

Run:

```bash
cd server
uv run pytest
```

Expected: all tests pass.

- [ ] **Step 2: Run ruff**

Run:

```bash
cd server
uv run ruff check .
```

Expected: PASS.

- [ ] **Step 3: Run pyright**

Run:

```bash
cd server
uv run pyright .
```

Expected: 0 errors.

- [ ] **Step 4: Verify no Claude imports remain in tracked Python files**

Run:

```bash
git grep -n "claude_agent_sdk\|ClaudeAgentProcessor\|ClaudeSDKClient" -- '*.py'
```

Expected: no matches.

- [ ] **Step 5: Verify git status and commit final fixes if needed**

Run:

```bash
git status --short
```

Expected: clean tree. If files are modified because of lockfile/docs/test adjustments, inspect them and commit:

```bash
git add <changed-files>
git commit -m "fix: complete codex-only cleanup"
```

---

## Self-review

Spec coverage:
- Claude backend deletion: Task 2.
- Provider literal/config branch removal: Task 2.
- Local profiles converted to Codex: Task 3.
- Dependency removal and lock refresh: Task 3.
- README and env docs cleanup: Task 4.
- Tests for Codex-only behavior and legacy rejection: Task 1.
- Full verification: Task 5.

No placeholders remain. All commands are exact and all implementation steps include concrete file edits or code blocks.
