# Native LangChain Provider Switch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Codex OAuth chat-model path for normal robot runs with real API-key-backed LangChain providers that emit native `AIMessage.tool_calls`.

**Architecture:** Keep `LangGraphRobotAgent` as the robot orchestration layer and keep `bind_tools()` as the only model-to-tool interface. Add a provider model factory that builds `ChatOpenAI` or `ChatGoogleGenerativeAI` from `runtime_profiles.toml`, then route those models through a generic LangChain robot processor. Keep `openai_codex_oauth` only as a legacy provider until the new live smoke tests are green.

**Tech Stack:** Python 3.12, pytest, LangGraph, LangChain Core, `langchain-openai`, `langchain-google-genai`, OpenAI API keys, Google Gemini API keys.

---

## Documentation Baseline

- LangChain `ChatOpenAI` docs say the Python integration supports tool calling and structured output, and uses `langchain-openai` plus an OpenAI Platform API key: https://docs.langchain.com/oss/python/integrations/chat/openai
- LangChain `ChatGoogleGenerativeAI` docs say the Python integration supports tool calling, structured output, `bind_tools()`, `thinking_level` for Gemini 3+, and `thinking_budget` for Gemini 2.5: https://docs.langchain.com/oss/python/integrations/chat/google_generative_ai
- OpenAI function calling docs define tool/function calling as the model returning tool calls and JSON arguments for application actions: https://developers.openai.com/api/docs/guides/function-calling
- OpenAI current model docs list GPT-5.4 and GPT-5.4-mini as supporting function tools and reasoning settings including `none`, `low`, `medium`, `high`, and `xhigh`: https://developers.openai.com/api/docs/models
- Google Gemini function calling docs say Gemini can return a function call suggestion after being given function declarations: https://ai.google.dev/gemini-api/docs/function-calling
- `chub get gemini/genai --lang py` confirms the current Google SDK is `google-genai`, `GEMINI_API_KEY` is valid for direct SDK use, Gemini 2.5 uses `thinking_budget`, and Gemini 2.5 Pro cannot disable thinking.
- `chub get langchain/openai --lang py` and `chub get langchain/package --lang py` failed locally on Windows with `ENOENT: no such file or directory, mkdir ''`; use the official LangChain docs above as fallback evidence.

## File Map

- Modify `server/pyproject.toml`: add `langchain-openai` and `langchain-google-genai`.
- Modify `server/voice_runtime/profiles.py`: add real API providers, API key env config, Gemini thinking config, and validation.
- Modify `server/runtime_profiles.toml`: add switchable OpenAI and Gemini agent profiles.
- Create `server/agent_model_factory.py`: provider-specific LangChain chat model construction.
- Create `server/langchain_agent_processor.py`: generic robot agent turn backend for API-key LangChain chat models.
- Modify `server/agent_processor_factory.py`: route `openai_api` and `gemini_api` through the new generic processor.
- Modify `server/tests/test_voice_runtime_profiles.py`: config parsing and validation coverage.
- Create `server/tests/test_agent_model_factory.py`: constructor argument tests without live API calls.
- Create `server/tests/test_langchain_agent_processor.py`: generic processor behavior tests using fake chat model and fake robot bridge.
- Modify `server/tests/test_agent_processor_factory.py`: routing tests for new providers.
- Create `server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py`: gated live API-key test that proves native tool calls are emitted.
- Optionally keep `server/tests/live_robot_smoke/manual_live_codex_tool_calling_probe.py` as historical evidence, but do not use it as the success path.

## Provider Contract

Runtime profile examples:

```toml
[profiles.hybrid_low_latency.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "low"
api_key_env = "OPENAI_API_KEY"

[profiles.hybrid_gemini.agent]
provider = "gemini_api"
model = "gemini-2.5-flash"
reasoning_effort = "low"
thinking_budget = 1024
api_key_env = "GOOGLE_API_KEY"
```

Rules:

- `openai_api` defaults to `OPENAI_API_KEY`.
- `gemini_api` defaults to `GOOGLE_API_KEY`; `api_key_env = "GEMINI_API_KEY"` is also allowed.
- OpenAI passes `reasoning_effort` through to `ChatOpenAI`.
- Gemini 3+ maps `reasoning_effort` to LangChain's `thinking_level`.
- Gemini 2.5 passes `thinking_budget` when set; if only `reasoning_effort` is set, use a documented project mapping in `agent_model_factory.py`.
- Gemini 2.5 Pro rejects `thinking_budget = 0` and rejects `reasoning_effort = "none"` because Gemini docs say Pro cannot turn thinking off.
- The graph still validates robot safety and current pose before execution; the LLM fills concrete tool arguments.

---

### Task 1: Parse API-Key Agent Profiles

**Files:**
- Modify: `server/voice_runtime/profiles.py`
- Test: `server/tests/test_voice_runtime_profiles.py`

- [ ] **Step 1: Write failing tests for OpenAI and Gemini agent profile parsing**

Add tests:

```python
def test_openai_api_agent_profile_requires_openai_key_env(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.openai_api]
category = "local_debug"
[profiles.openai_api.wake]
provider = "none"
[profiles.openai_api.emergency_stop]
enabled = false
[profiles.openai_api.stt]
provider = "whisper"
[profiles.openai_api.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.openai_api.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "low"
[profiles.openai_api.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.openai_api.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="openai_api",
    )

    assert profile.agent.provider == "openai_api"
    assert profile.agent.api_key_env == "OPENAI_API_KEY"
    assert profile.agent.reasoning_effort == "low"
    assert profile.required_env_names() == ("OPENAI_API_KEY",)
```

```python
def test_gemini_api_agent_profile_accepts_thinking_budget_and_key_override(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.gemini_api]
category = "local_debug"
[profiles.gemini_api.wake]
provider = "none"
[profiles.gemini_api.emergency_stop]
enabled = false
[profiles.gemini_api.stt]
provider = "whisper"
[profiles.gemini_api.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.gemini_api.agent]
provider = "gemini_api"
model = "gemini-2.5-flash"
reasoning_effort = "medium"
thinking_budget = 1024
api_key_env = "GEMINI_API_KEY"
[profiles.gemini_api.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.gemini_api.metrics]
enabled = false
""",
    )

    profile = load_runtime_profile(
        profiles_path=profiles_path,
        server_dir=tmp_path,
        profile_name="gemini_api",
    )

    assert profile.agent.provider == "gemini_api"
    assert profile.agent.api_key_env == "GEMINI_API_KEY"
    assert profile.agent.thinking_budget == 1024
    assert profile.required_env_names() == ("GEMINI_API_KEY",)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py::test_openai_api_agent_profile_requires_openai_key_env tests/test_voice_runtime_profiles.py::test_gemini_api_agent_profile_accepts_thinking_budget_and_key_override -q
```

Expected: FAIL with `provider must be one of` or `AgentProfile has no attribute api_key_env`.

- [ ] **Step 3: Implement minimal profile support**

Change the agent typing and dataclass:

```python
AgentProvider = Literal["openai_codex_oauth", "openai_api", "gemini_api"]
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

_AGENT_PROVIDERS = {"openai_codex_oauth", "openai_api", "gemini_api"}
_DEFAULT_AGENT_KEY_ENV = {
    "openai_codex_oauth": None,
    "openai_api": "OPENAI_API_KEY",
    "gemini_api": "GOOGLE_API_KEY",
}

@dataclass(frozen=True)
class AgentProfile:
    provider: AgentProvider
    model: str
    reasoning_effort: ReasoningEffort | None = None
    api_key_env: str | None = None
    thinking_budget: int | None = None
```

Update `_parse_agent()`:

```python
def _parse_agent(table: dict[str, Any]) -> AgentProfile:
    provider = cast(AgentProvider, _literal(table, "provider", _AGENT_PROVIDERS))
    reasoning_effort = cast(
        ReasoningEffort | None,
        _optional_literal(table, "reasoning_effort", _REASONING_EFFORTS),
    )
    api_key_env = _optional_string(table, "api_key_env")
    if api_key_env is None:
        api_key_env = _DEFAULT_AGENT_KEY_ENV[provider]
    return AgentProfile(
        provider=provider,
        model=_string(table, "model", "gpt-5.4-mini"),
        reasoning_effort=reasoning_effort,
        api_key_env=api_key_env,
        thinking_budget=_optional_non_negative_int(table, "thinking_budget"),
    )
```

Add helper:

```python
def _optional_non_negative_int(table: dict[str, Any], key: str) -> int | None:
    value = table.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProfileError(f"{key} must be an integer")
    if value < 0:
        raise ProfileError(f"{key} must be non-negative")
    return value
```

Update `required_env_names()` to append `self.agent.api_key_env` when present, then de-duplicate in order.

- [ ] **Step 4: Run tests to verify green**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py::test_openai_api_agent_profile_requires_openai_key_env tests/test_voice_runtime_profiles.py::test_gemini_api_agent_profile_accepts_thinking_budget_and_key_override -q
```

Expected: PASS.

- [ ] **Step 5: Add validation tests for invalid Gemini reasoning config**

Add tests:

```python
def test_gemini_25_pro_rejects_disabled_thinking(tmp_path: Path):
    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profile(
        profiles_path,
        """
[profiles.bad_gemini]
category = "local_debug"
[profiles.bad_gemini.wake]
provider = "none"
[profiles.bad_gemini.emergency_stop]
enabled = false
[profiles.bad_gemini.stt]
provider = "whisper"
[profiles.bad_gemini.tts]
provider = "kokoro"
voice = "af_heart"
[profiles.bad_gemini.agent]
provider = "gemini_api"
model = "gemini-2.5-pro"
thinking_budget = 0
[profiles.bad_gemini.mcp.robot]
url = "http://127.0.0.1:8765/mcp"
[profiles.bad_gemini.metrics]
enabled = false
""",
    )

    with pytest.raises(ProfileError, match="gemini-2.5-pro cannot disable thinking"):
        load_runtime_profile(
            profiles_path=profiles_path,
            server_dir=tmp_path,
            profile_name="bad_gemini",
        )
```

- [ ] **Step 6: Implement the minimal validation**

In `_validate_runtime_profile()`:

```python
    if (
        profile.agent.provider == "gemini_api"
        and profile.agent.model.startswith("gemini-2.5-pro")
        and profile.agent.thinking_budget == 0
    ):
        raise ProfileError("gemini-2.5-pro cannot disable thinking")
```

- [ ] **Step 7: Run profile tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py -q
```

Expected: PASS.

---

### Task 2: Add Provider Dependencies

**Files:**
- Modify: `server/pyproject.toml`

- [ ] **Step 1: Add dependencies**

Add:

```toml
"langchain-openai>=1.1,<2",
"langchain-google-genai>=4.0,<5",
```

Keep `langchain-codex-oauth` for now so legacy tests keep passing.

- [ ] **Step 2: Sync and verify imports**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv sync
uv run python -c "from langchain_openai import ChatOpenAI; from langchain_google_genai import ChatGoogleGenerativeAI; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```powershell
git add server/pyproject.toml server/uv.lock
git commit -m "chore: add native langchain model providers"
```

---

### Task 3: Build API-Key Chat Models From Config

**Files:**
- Create: `server/agent_model_factory.py`
- Test: `server/tests/test_agent_model_factory.py`

- [ ] **Step 1: Write failing constructor tests**

Create `server/tests/test_agent_model_factory.py`:

```python
from typing import Any

import pytest

from agent_model_factory import build_agent_chat_model
from voice_runtime.profiles import AgentProfile


class CapturedChatModel:
    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs


def test_builds_chat_openai_with_reasoning_effort_and_key():
    model = build_agent_chat_model(
        AgentProfile(
            provider="openai_api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        env={"OPENAI_API_KEY": "sk-test"},
        chat_openai_cls=CapturedChatModel,
    )

    assert isinstance(model, CapturedChatModel)
    assert model.kwargs["model"] == "gpt-5.4-mini"
    assert model.kwargs["api_key"] == "sk-test"
    assert model.kwargs["reasoning_effort"] == "low"
```

```python
def test_builds_chat_google_with_thinking_budget_and_key():
    model = build_agent_chat_model(
        AgentProfile(
            provider="gemini_api",
            model="gemini-2.5-flash",
            reasoning_effort="medium",
            api_key_env="GEMINI_API_KEY",
            thinking_budget=1024,
        ),
        env={"GEMINI_API_KEY": "gem-test"},
        chat_google_cls=CapturedChatModel,
    )

    assert isinstance(model, CapturedChatModel)
    assert model.kwargs["model"] == "gemini-2.5-flash"
    assert model.kwargs["google_api_key"] == "gem-test"
    assert model.kwargs["thinking_budget"] == 1024
```

```python
def test_missing_provider_key_raises_clear_error():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        build_agent_chat_model(
            AgentProfile(provider="openai_api", model="gpt-5.4-mini", api_key_env="OPENAI_API_KEY"),
            env={},
            chat_openai_cls=CapturedChatModel,
        )
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_model_factory'`.

- [ ] **Step 3: Implement minimal model factory**

Create `server/agent_model_factory.py`:

```python
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from voice_runtime.profiles import AgentProfile

_GEMINI_25_BUDGET_BY_EFFORT = {
    "none": 0,
    "minimal": 0,
    "low": 512,
    "medium": 1024,
    "high": 4096,
    "xhigh": 8192,
}


def build_agent_chat_model(
    config: AgentProfile,
    *,
    env: Mapping[str, str] | None = None,
    chat_openai_cls: type[Any] | None = None,
    chat_google_cls: type[Any] | None = None,
) -> Any:
    env = env or os.environ
    if config.provider == "openai_api":
        return _build_openai(config, env, chat_openai_cls)
    if config.provider == "gemini_api":
        return _build_gemini(config, env, chat_google_cls)
    raise ValueError(f"Unsupported native LangChain agent provider: {config.provider}")


def _required_key(config: AgentProfile, env: Mapping[str, str]) -> str:
    if config.api_key_env is None:
        raise ValueError(f"{config.provider} requires api_key_env")
    key = env.get(config.api_key_env)
    if not key:
        raise ValueError(f"{config.api_key_env} is required for {config.provider}")
    return key


def _build_openai(
    config: AgentProfile,
    env: Mapping[str, str],
    chat_openai_cls: type[Any] | None,
) -> Any:
    if chat_openai_cls is None:
        from langchain_openai import ChatOpenAI

        chat_openai_cls = ChatOpenAI
    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": _required_key(config, env),
        "use_responses_api": True,
    }
    if config.reasoning_effort is not None:
        kwargs["reasoning_effort"] = config.reasoning_effort
    return chat_openai_cls(**kwargs)


def _build_gemini(
    config: AgentProfile,
    env: Mapping[str, str],
    chat_google_cls: type[Any] | None,
) -> Any:
    if chat_google_cls is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        chat_google_cls = ChatGoogleGenerativeAI
    kwargs: dict[str, Any] = {
        "model": config.model,
        "google_api_key": _required_key(config, env),
    }
    if config.model.startswith("gemini-3") and config.reasoning_effort is not None:
        if config.reasoning_effort in {"none", "xhigh"}:
            raise ValueError("Gemini 3 thinking_level supports minimal, low, medium, or high")
        kwargs["thinking_level"] = config.reasoning_effort
    elif config.model.startswith("gemini-2.5"):
        if config.thinking_budget is not None:
            kwargs["thinking_budget"] = config.thinking_budget
        elif config.reasoning_effort is not None:
            kwargs["thinking_budget"] = _GEMINI_25_BUDGET_BY_EFFORT[config.reasoning_effort]
    return chat_google_cls(**kwargs)
```

- [ ] **Step 4: Run factory tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_model_factory.py -q
```

Expected: PASS.

- [ ] **Step 5: Add tests for Gemini 3 thinking level**

Add:

```python
def test_builds_gemini_3_with_thinking_level():
    model = build_agent_chat_model(
        AgentProfile(
            provider="gemini_api",
            model="gemini-3.1-pro-preview",
            reasoning_effort="low",
            api_key_env="GOOGLE_API_KEY",
        ),
        env={"GOOGLE_API_KEY": "gem-test"},
        chat_google_cls=CapturedChatModel,
    )

    assert model.kwargs["thinking_level"] == "low"
    assert "thinking_budget" not in model.kwargs
```

Run:

```powershell
uv run pytest tests/test_agent_model_factory.py -q
```

Expected: PASS.

---

### Task 4: Add Generic LangChain Robot Processor

**Files:**
- Create: `server/langchain_agent_processor.py`
- Test: `server/tests/test_langchain_agent_processor.py`

- [ ] **Step 1: Write failing processor tests**

Create `server/tests/test_langchain_agent_processor.py` by copying the fake model, bridge, `ai_text()`, `ai_tool_call()`, and `_run_turn()` helpers from `test_openai_codex_agent_processor.py`, then add:

```python
import pytest

from langchain_agent_processor import LangChainAgentProcessor


@pytest.mark.asyncio
async def test_generic_processor_runs_langgraph_without_codex_credentials():
    model = FakeChatModel([ai_text("ready")])
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gpt-5.4-mini",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["ready"]
    assert bridge.connected is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert model.requests
```

```python
@pytest.mark.asyncio
async def test_generic_processor_executes_model_tool_call():
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("pose observed"),
        ]
    )
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gemini-2.5-flash",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "where are you?")

    assert result.chunks == ["pose observed"]
    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langchain_agent_processor.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'langchain_agent_processor'`.

- [ ] **Step 3: Implement processor**

Create `server/langchain_agent_processor.py`:

```python
from __future__ import annotations

from typing import Any

from loguru import logger

from langgraph_robot_agent import LangGraphRobotAgent
from robot_control.context import RobotContextStore
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.agent_turn import AgentTurnInput


class LangChainAgentProcessor:
    """Runs Agent Turns through an API-key-backed LangChain chat model."""

    def __init__(
        self,
        mcp_server_url: str,
        *,
        chat_model: Any,
        model_label: str,
        tool_bridge: Any | None = None,
    ):
        self._mcp_server_url = mcp_server_url
        self._chat_model = chat_model
        self._model_label = model_label
        self._tool_bridge = tool_bridge
        self._owns_tool_bridge = tool_bridge is None
        self._connected = False
        self._model_logged = False
        self._robot_context = RobotContextStore()
        self._thread_id = f"langchain-agent-{id(self)}"
        self._graph_agent: LangGraphRobotAgent | None = None
        self._graph_chat_model: Any | None = None
        self._graph_tool_bridge: Any | None = None

    async def connect(self) -> None:
        await self._ensure_connected()

    async def disconnect(self) -> None:
        if self._tool_bridge is not None and (self._connected or not self._owns_tool_bridge):
            await self._tool_bridge.disconnect()
        self._tool_bridge = None
        self._graph_agent = None
        self._graph_chat_model = None
        self._graph_tool_bridge = None
        self._connected = False
        logger.info("LangChain API-key agent disconnected")

    async def run_turn(self, turn: AgentTurnInput):
        logger.info("User said: {}", turn.user_text)
        try:
            await self._ensure_connected()
        except Exception as exc:
            logger.error("LangChain agent connection error: {}", exc)
            yield "I can't reach the robot control server right now."
            return

        tool_bridge = self._tool_bridge
        if tool_bridge is None:
            yield "I can't reach the robot control server right now."
            return

        if not self._model_logged:
            logger.info("LangChain model: {}", self._model_label)
            self._model_logged = True

        graph = self._graph_agent_for(self._chat_model, tool_bridge)
        try:
            yield await graph.run_turn(turn)
        except Exception as exc:
            logger.error("LangChain agent error: {}", exc)
            yield "I encountered an error. Please try again."

    async def _ensure_connected(self) -> None:
        if self._connected:
            return
        if self._tool_bridge is None:
            self._tool_bridge = RobotMCPBridge(self._mcp_server_url)
        await self._tool_bridge.connect()
        self._connected = True
        logger.info("LangChain API-key agent connected")

    def _graph_agent_for(self, chat_model: Any, tool_bridge: Any) -> LangGraphRobotAgent:
        if (
            self._graph_agent is None
            or self._graph_chat_model is not chat_model
            or self._graph_tool_bridge is not tool_bridge
        ):
            self._graph_agent = LangGraphRobotAgent(
                model=chat_model,
                tool_bridge=tool_bridge,
                robot_context=self._robot_context,
                thread_id=self._thread_id,
            )
            self._graph_chat_model = chat_model
            self._graph_tool_bridge = tool_bridge
        return self._graph_agent
```

- [ ] **Step 4: Run processor tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_langchain_agent_processor.py -q
```

Expected: PASS.

---

### Task 5: Route New Providers Through the Generic Processor

**Files:**
- Modify: `server/agent_processor_factory.py`
- Modify: `server/tests/test_agent_processor_factory.py`

- [ ] **Step 1: Write failing routing tests**

Add:

```python
from langchain_agent_processor import LangChainAgentProcessor


class FakeChatModel:
    pass


def test_creates_openai_api_agent_turn_processor(monkeypatch):
    monkeypatch.setattr(
        "agent_processor_factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
    )

    processor = create_agent_processor(
        AgentConfig(
            provider="openai_api",
            model="gpt-5.4-mini",
            reasoning_effort="low",
            api_key_env="OPENAI_API_KEY",
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor._backend, LangChainAgentProcessor)
```

```python
def test_creates_gemini_api_agent_turn_processor(monkeypatch):
    monkeypatch.setattr(
        "agent_processor_factory.build_agent_chat_model",
        lambda config: FakeChatModel(),
    )

    processor = create_agent_processor(
        AgentConfig(
            provider="gemini_api",
            model="gemini-2.5-flash",
            api_key_env="GOOGLE_API_KEY",
            thinking_budget=1024,
        ),
        mcp_server_url="http://127.0.0.1:8765/mcp",
    )

    assert isinstance(processor, AgentTurnProcessor)
    assert isinstance(processor._backend, LangChainAgentProcessor)
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_processor_factory.py -q
```

Expected: FAIL because `openai_api` and `gemini_api` are unsupported.

- [ ] **Step 3: Implement routing**

Modify imports:

```python
from agent_model_factory import build_agent_chat_model
from langchain_agent_processor import LangChainAgentProcessor
```

Modify `create_agent_processor()`:

```python
    if config.provider == "openai_codex_oauth":
        backend = OpenAICodexAgentProcessor(
            mcp_server_url=mcp_server_url,
            model=config.model,
            reasoning_effort=config.reasoning_effort,
        )
    elif config.provider in {"openai_api", "gemini_api"}:
        backend = LangChainAgentProcessor(
            mcp_server_url,
            chat_model=build_agent_chat_model(config),
            model_label=config.model,
        )
    else:
        raise ValueError(f"Unsupported agent provider: {config.provider}")
    return AgentTurnProcessor(
        backend=backend,
        on_turn_started=on_turn_started,
        on_turn_finished=on_turn_finished,
    )
```

- [ ] **Step 4: Run factory tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_agent_processor_factory.py -q
```

Expected: PASS.

---

### Task 6: Add Switchable Runtime Profiles

**Files:**
- Modify: `server/runtime_profiles.toml`
- Modify: `server/tests/test_voice_runtime_profiles.py`

- [ ] **Step 1: Write failing bundled profile tests**

Add:

```python
def test_bundled_default_profile_uses_native_openai_api_agent():
    profile = load_runtime_profile()

    assert profile.agent.provider == "openai_api"
    assert profile.agent.api_key_env == "OPENAI_API_KEY"
```

```python
def test_bundled_gemini_profile_is_available():
    server_dir = Path(__file__).resolve().parents[1]

    profile = load_runtime_profile(
        profiles_path=default_profiles_path(server_dir),
        server_dir=server_dir,
        profile_name="hybrid_gemini",
    )

    assert profile.agent.provider == "gemini_api"
    assert profile.agent.model.startswith("gemini-")
    assert profile.agent.api_key_env in {"GOOGLE_API_KEY", "GEMINI_API_KEY"}
```

- [ ] **Step 2: Run tests to verify red**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py::test_bundled_default_profile_uses_native_openai_api_agent tests/test_voice_runtime_profiles.py::test_bundled_gemini_profile_is_available -q
```

Expected: FAIL because bundled profiles still use `openai_codex_oauth` and no `hybrid_gemini` exists.

- [ ] **Step 3: Update bundled profiles**

Change default agent block:

```toml
[profiles.hybrid_low_latency.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
reasoning_effort = "low"
api_key_env = "OPENAI_API_KEY"
```

Add a Gemini switch profile by copying `hybrid_low_latency` and changing only the profile name and agent block:

```toml
[profiles.hybrid_gemini.agent]
provider = "gemini_api"
model = "gemini-2.5-flash"
reasoning_effort = "low"
thinking_budget = 1024
api_key_env = "GOOGLE_API_KEY"
```

Leave a legacy profile:

```toml
[profiles.hybrid_codex_oauth.agent]
provider = "openai_codex_oauth"
model = "gpt-5.4-mini"
reasoning_effort = "medium"
```

- [ ] **Step 4: Run profile tests**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py -q
```

Expected: PASS.

---

### Task 7: Prove Native Tool Calling With Real API Keys

**Files:**
- Create: `server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py`
- Modify: `server/pyproject.toml`

- [ ] **Step 1: Write gated live test**

Create:

```python
from __future__ import annotations

import os
from typing import Literal

import pytest
from langchain_core.tools import tool

from agent_model_factory import build_agent_chat_model
from voice_runtime.profiles import AgentProfile


pytestmark = [pytest.mark.live, pytest.mark.llm]


@tool
def report_robot_motion(
    robot_name: str,
    motion: Literal["up_down", "wave"],
    distance_m: float,
) -> str:
    """Report the exact robot motion the assistant intends to perform."""
    return f"{robot_name}:{motion}:{distance_m}"


def _live_enabled() -> bool:
    return os.getenv("RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL") == "1"


def _profile_from_env() -> AgentProfile:
    provider = os.getenv("LIVE_AGENT_PROVIDER", "openai_api")
    if provider == "gemini_api":
        return AgentProfile(
            provider="gemini_api",
            model=os.getenv("LIVE_GEMINI_MODEL", "gemini-2.5-flash"),
            reasoning_effort=os.getenv("LIVE_REASONING_EFFORT", "low"),
            thinking_budget=int(os.getenv("LIVE_GEMINI_THINKING_BUDGET", "1024")),
            api_key_env=os.getenv("LIVE_GEMINI_KEY_ENV", "GOOGLE_API_KEY"),
        )
    return AgentProfile(
        provider="openai_api",
        model=os.getenv("LIVE_OPENAI_MODEL", "gpt-5.4-mini"),
        reasoning_effort=os.getenv("LIVE_REASONING_EFFORT", "low"),
        api_key_env=os.getenv("LIVE_OPENAI_KEY_ENV", "OPENAI_API_KEY"),
    )


@pytest.mark.skipif(not _live_enabled(), reason="set RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL=1")
@pytest.mark.parametrize("attempt", range(5))
def test_live_provider_emits_native_tool_call(attempt: int):
    model = build_agent_chat_model(_profile_from_env())
    bound = model.bind_tools([report_robot_motion])

    msg = bound.invoke(
        "Call report_robot_motion for robot UR10 doing an up_down motion over 0.05 meters. "
        "Do not answer in text; use the tool."
    )

    assert msg.tool_calls, f"attempt {attempt} produced no tool_calls: {msg!r}"
    call = msg.tool_calls[0]
    assert call["name"] == "report_robot_motion"
    assert call["args"]["robot_name"] == "UR10"
    assert call["args"]["motion"] == "up_down"
    assert 0.01 <= float(call["args"]["distance_m"]) <= 0.10
```

- [ ] **Step 2: Add marker**

Add to `server/pyproject.toml` markers:

```toml
"native_llm: manual tests that call real API-key LangChain providers",
```

- [ ] **Step 3: Run default gated behavior**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py -q
```

Expected: 5 skipped.

- [ ] **Step 4: Run live OpenAI tool-call proof**

Run after `OPENAI_API_KEY` is set:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
$env:RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL="1"
$env:LIVE_AGENT_PROVIDER="openai_api"
$env:LIVE_OPENAI_MODEL="gpt-5.4-mini"
$env:LIVE_REASONING_EFFORT="low"
uv run pytest tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py -q -s
```

Expected: 5 passed. If this fails with `tool_calls=[]`, do not proceed to robot execution.

- [ ] **Step 5: Run live Gemini tool-call proof**

Run after `GOOGLE_API_KEY` or `GEMINI_API_KEY` is set:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
$env:RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL="1"
$env:LIVE_AGENT_PROVIDER="gemini_api"
$env:LIVE_GEMINI_MODEL="gemini-2.5-flash"
$env:LIVE_GEMINI_THINKING_BUDGET="1024"
uv run pytest tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py -q -s
```

Expected: 5 passed. If this fails with `tool_calls=[]`, mark Gemini unsupported for the first rollout and keep OpenAI as the default.

---

### Task 8: Prove The Real Robot Tool Schema Is Usable Without Executing Motion

**Files:**
- Create: `server/tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py`

- [ ] **Step 1: Write gated schema-only probe**

Create:

```python
from __future__ import annotations

import os

import pytest

from agent_model_factory import build_agent_chat_model
from robot_control.mcp_bridge import RobotMCPBridge
from voice_runtime.profiles import AgentProfile


pytestmark = [pytest.mark.live, pytest.mark.llm, pytest.mark.robot_sim]


@pytest.mark.skipif(
    os.getenv("RUN_LIVE_NATIVE_ROBOT_SCHEMA_PROBE") != "1",
    reason="set RUN_LIVE_NATIVE_ROBOT_SCHEMA_PROBE=1",
)
@pytest.mark.asyncio
async def test_live_model_fills_real_moveit_tool_arguments_without_execution():
    bridge = RobotMCPBridge(os.getenv("LIVE_ROBOT_MCP_URL", "http://127.0.0.1:8765/mcp"))
    await bridge.connect()
    try:
        tools = bridge.function_tools()
        motion_tool = next(t for t in tools if t["name"] == "moveit_plan_and_execute_free_motion")
        model = build_agent_chat_model(
            AgentProfile(
                provider="openai_api",
                model=os.getenv("LIVE_OPENAI_MODEL", "gpt-5.4-mini"),
                reasoning_effort=os.getenv("LIVE_REASONING_EFFORT", "low"),
                api_key_env="OPENAI_API_KEY",
            )
        )
        bound = model.bind_tools([motion_tool])
        msg = await bound.ainvoke(
            "For robot UR10, prepare a moveit_plan_and_execute_free_motion call that moves "
            "the end effector up by about 5 cm from the current pose. Only call the tool."
        )
    finally:
        await bridge.disconnect()

    assert msg.tool_calls
    call = msg.tool_calls[0]
    assert call["name"] == "moveit_plan_and_execute_free_motion"
    assert call["args"]["robot_name"] == "UR10"
    assert "target_pose" in call["args"]
```

- [ ] **Step 2: Run default gated behavior**

Run:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py -q
```

Expected: 1 skipped.

- [ ] **Step 3: Run live schema probe**

Run with MCP robot sim and `OPENAI_API_KEY` set:

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
$env:RUN_LIVE_NATIVE_ROBOT_SCHEMA_PROBE="1"
uv run pytest tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py -q -s
```

Expected: PASS and no robot motion is executed by the test.

---

### Task 9: Full Verification

**Files:**
- All modified files.

- [ ] **Step 1: Run focused tests**

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest tests/test_voice_runtime_profiles.py tests/test_agent_model_factory.py tests/test_langchain_agent_processor.py tests/test_agent_processor_factory.py -q
```

Expected: PASS.

- [ ] **Step 2: Run full test suite**

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run pytest -q
```

Expected: PASS, with gated live tests skipped unless explicitly enabled.

- [ ] **Step 3: Run lint and type checks**

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
uv run ruff check .
uv run pyright
```

Expected: PASS.

- [ ] **Step 4: Run live OpenAI proof before claiming native path works**

```powershell
cd C:\Users\Samuel\Documents\github\pipecat\pipecat-agent\server
$env:RUN_LIVE_NATIVE_LANGCHAIN_TOOL_CALL="1"
$env:LIVE_AGENT_PROVIDER="openai_api"
uv run pytest tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py -q -s
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```powershell
git add server/pyproject.toml server/uv.lock server/voice_runtime/profiles.py server/runtime_profiles.toml server/agent_model_factory.py server/langchain_agent_processor.py server/agent_processor_factory.py server/tests/test_voice_runtime_profiles.py server/tests/test_agent_model_factory.py server/tests/test_langchain_agent_processor.py server/tests/test_agent_processor_factory.py server/tests/live_robot_smoke/manual_live_native_langchain_tool_calling.py server/tests/live_robot_smoke/manual_live_native_robot_tool_schema_probe.py
git commit -m "feat: add native langchain api agent providers"
```

## Acceptance Criteria

- Default profile can use `openai_api` with `OPENAI_API_KEY`.
- A Gemini profile can be selected with `gemini_api` and either `GOOGLE_API_KEY` or configured `GEMINI_API_KEY`.
- Unit tests prove provider config, env requirements, and provider constructor arguments.
- The generic processor runs the existing LangGraph robot loop without Codex OAuth credentials.
- Gated live tests prove real API providers emit `AIMessage.tool_calls` for a synthetic tool before any robot motion path is trusted.
- Schema-only robot probe proves the model can fill the actual MoveIt tool shape without executing the tool.
- Existing Codex OAuth provider remains available for comparison but is not the default success path.

## Out Of Scope

- Reducing repeated `GET /client` browser polling spam. Track that separately so this change stays focused on model/tool-call reliability.
- Deterministic motion fallback behavior changes. Fallback can stay as a safety net, but the goal of this plan is to make it unnecessary for normal tool-worthy intents.
