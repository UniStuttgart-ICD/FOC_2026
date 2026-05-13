# OpenAI Codex OAuth Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the hard-wired Claude Agent SDK path with a config-selectable LLM provider path that can use the same ChatGPT/Codex subscription OAuth credentials Pi stores in `~/.pi/agent/auth.json`.

**Architecture:** Keep the Pipecat pipeline stable and introduce a small provider boundary: `bot.py` loads config, `agent_processor_factory.py` selects a concrete `FrameProcessor`, and provider processors own their SDK/auth details. The first new provider is `openai_codex_oauth`, which reads Pi's `openai-codex` OAuth profile, refreshes it, and uses OpenAI Agents SDK with a Codex Responses backend client and the existing robot MCP server.

**Tech Stack:** Python 3.10+, Pipecat, `claude-agent-sdk`, OpenAI Agents SDK (`openai-agents`), OpenAI Python SDK, `httpx`, `pytest`, TOML config via `tomllib`/`tomli`.

---

## Important assumptions and corrections

1. **“OpenAI OAuth” means Codex/ChatGPT subscription OAuth, not normal OpenAI Platform API auth.** Pi and OpenClaw both treat subscription auth as `openai-codex` / Codex Responses, separate from `openai` / `OPENAI_API_KEY`.
2. **This will reuse Pi's auth file by default.** Your current Pi auth file already has an `openai-codex` entry shaped like `{ type: "oauth", access, refresh, expires, accountId }`. The app will read that profile from `PI_CODING_AGENT_DIR/auth.json` or `~/.pi/agent/auth.json`.
3. **This is a local/personal subscription route.** It is not the recommended production/multi-user OpenAI Platform path. For production, add and select `openai_api_key` later.
4. **OpenAI Agents SDK is official, but Codex backend usage through it needs live validation.** The SDK supports custom `AsyncOpenAI` clients and local MCP servers. If Codex rejects an Agents SDK request shape, keep the auth/config layer and replace only the OpenAI processor's model call implementation with a direct Codex Responses loop.

## File structure

- Modify: `pipecat-agent/server/pyproject.toml` — add OpenAI/pytest/tomli dependencies.
- Create: `pipecat-agent/server/config.example.toml` — checked-in provider config template.
- Create: `pipecat-agent/server/agent_config.py` — parse and validate config.
- Create: `pipecat-agent/server/codex_auth.py` — read/refresh Pi Codex OAuth credentials.
- Create: `pipecat-agent/server/openai_codex_agent_processor.py` — Pipecat processor backed by OpenAI Agents SDK + local MCP.
- Create: `pipecat-agent/server/agent_processor_factory.py` — select `ClaudeAgentProcessor` or `OpenAICodexAgentProcessor` from config.
- Modify: `pipecat-agent/server/bot.py` — load config and use factory.
- Modify: `pipecat-agent/server/.env.example` — document config path and auth modes.
- Modify: `pipecat-agent/README.md` — document provider selection.
- Create: `pipecat-agent/server/tests/test_agent_config.py` — config loader tests.
- Create: `pipecat-agent/server/tests/test_codex_auth.py` — auth profile and refresh tests.
- Create: `pipecat-agent/server/tests/test_agent_processor_factory.py` — provider selection tests.

---

### Task 1: Add dependencies and config loader

**Files:**
- Modify: `pipecat-agent/server/pyproject.toml`
- Create: `pipecat-agent/server/config.example.toml`
- Create: `pipecat-agent/server/agent_config.py`
- Create: `pipecat-agent/server/tests/test_agent_config.py`

- [ ] **Step 1: Add dependencies**

Modify `pipecat-agent/server/pyproject.toml` dependencies to include OpenAI Agents SDK, OpenAI SDK, httpx, and tomli fallback. Add pytest to dev dependencies.

```toml
[project]
name = "pipecat-agent"
version = "0.1.0"
description = "Voice-controlled UR robot agent"
requires-python = ">=3.10,<3.13"
dependencies = [
    "pipecat-ai[kokoro,runner,silero,webrtc,whisper]",
    "claude-agent-sdk<0.1.49",
    "openai-agents>=0.14.0,<1",
    "openai>=2.29.0,<3",
    "httpx>=0.28.0,<1",
    "tomli>=2.0.0; python_version < '3.11'",
]

[dependency-groups]
dev = [
    "pyright>=1.1.404,<2",
    "ruff>=0.12.11,<1",
    "pytest>=8.0.0,<9",
    "pytest-asyncio>=0.24.0,<1",
]

[tool.ruff]
line-length = 100
[tool.ruff.lint]
select = ["I"]
```

- [ ] **Step 2: Install dependencies**

Run:

```bash
cd pipecat-agent/server
uv sync
```

Expected: `uv` updates `.venv` and `uv.lock` without dependency resolution errors.

- [ ] **Step 3: Write config tests first**

Create `pipecat-agent/server/tests/test_agent_config.py`:

```python
from pathlib import Path

import pytest

from agent_config import AgentConfig, ConfigError, load_agent_config


def test_loads_openai_codex_config(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
[llm]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[openai_codex]
base_url = "https://chatgpt.com/backend-api/codex"
pi_auth_file = "C:/Users/Samuel/.pi/agent/auth.json"
session_db = ":memory:"

[mcp.robot]
url = "http://127.0.0.1:8765/mcp"
""".strip(),
        encoding="utf-8",
    )

    config = load_agent_config(path)

    assert config.llm.provider == "openai_codex_oauth"
    assert config.llm.model == "gpt-5.5"
    assert config.openai_codex.base_url == "https://chatgpt.com/backend-api/codex"
    assert config.openai_codex.pi_auth_file == "C:/Users/Samuel/.pi/agent/auth.json"
    assert config.openai_codex.session_db == ":memory:"
    assert config.mcp.robot_url == "http://127.0.0.1:8765/mcp"


def test_defaults_to_claude_when_config_missing(tmp_path: Path):
    config = load_agent_config(tmp_path / "missing.toml")

    assert config == AgentConfig.default()
    assert config.llm.provider == "claude"
    assert config.llm.model == "claude-haiku-4-5-20251001"
    assert config.mcp.robot_url == "http://127.0.0.1:8765/mcp"


def test_rejects_unknown_provider(tmp_path: Path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nprovider = "bogus"\n', encoding="utf-8")

    with pytest.raises(ConfigError, match="Unsupported llm.provider"):
        load_agent_config(path)
```

- [ ] **Step 4: Run tests and verify they fail**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_agent_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_config'`.

- [ ] **Step 5: Implement config loader**

Create `pipecat-agent/server/agent_config.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[no-redef]

ProviderName = Literal["claude", "openai_codex_oauth"]
SUPPORTED_PROVIDERS = {"claude", "openai_codex_oauth"}


class ConfigError(ValueError):
    """Raised when agent config is present but invalid."""


@dataclass(frozen=True)
class LLMConfig:
    provider: ProviderName
    model: str


@dataclass(frozen=True)
class OpenAICodexConfig:
    base_url: str
    pi_auth_file: str | None
    session_db: str
    auth_profile: str


@dataclass(frozen=True)
class MCPConfig:
    robot_url: str


@dataclass(frozen=True)
class AgentConfig:
    llm: LLMConfig
    openai_codex: OpenAICodexConfig
    mcp: MCPConfig

    @staticmethod
    def default() -> "AgentConfig":
        return AgentConfig(
            llm=LLMConfig(
                provider="claude",
                model=os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            ),
            openai_codex=OpenAICodexConfig(
                base_url="https://chatgpt.com/backend-api/codex",
                pi_auth_file=None,
                session_db=":memory:",
                auth_profile="openai-codex",
            ),
            mcp=MCPConfig(
                robot_url=os.getenv("MCP_SERVER_URL", "http://127.0.0.1:8765/mcp"),
            ),
        )


def default_config_path() -> Path:
    return Path(os.getenv("AGENT_CONFIG", "config.toml"))


def _table(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"[{key}] must be a TOML table")
    return value


def _string(table: dict[str, Any], key: str, default: str | None = None) -> str | None:
    value = table.get(key, default)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{key} must be a non-empty string")
    return value.strip()


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    config_path = Path(path) if path is not None else default_config_path()
    default = AgentConfig.default()
    if not config_path.exists():
        return default

    with config_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ConfigError("Config root must be a TOML table")

    llm = _table(data, "llm")
    provider = _string(llm, "provider", default.llm.provider)
    if provider not in SUPPORTED_PROVIDERS:
        raise ConfigError(
            f"Unsupported llm.provider '{provider}'. Supported providers: {sorted(SUPPORTED_PROVIDERS)}"
        )
    model_default = default.llm.model if provider == "claude" else "gpt-5.5"
    model = _string(llm, "model", model_default)

    openai_codex = _table(data, "openai_codex")
    mcp = _table(data, "mcp")
    robot = _table(mcp, "robot")

    return AgentConfig(
        llm=LLMConfig(provider=provider, model=model or model_default),  # type: ignore[arg-type]
        openai_codex=OpenAICodexConfig(
            base_url=_string(
                openai_codex,
                "base_url",
                default.openai_codex.base_url,
            )
            or default.openai_codex.base_url,
            pi_auth_file=_string(openai_codex, "pi_auth_file", default.openai_codex.pi_auth_file),
            session_db=_string(openai_codex, "session_db", default.openai_codex.session_db)
            or default.openai_codex.session_db,
            auth_profile=_string(openai_codex, "auth_profile", default.openai_codex.auth_profile)
            or default.openai_codex.auth_profile,
        ),
        mcp=MCPConfig(
            robot_url=_string(robot, "url", default.mcp.robot_url) or default.mcp.robot_url,
        ),
    )
```

- [ ] **Step 6: Add config template**

Create `pipecat-agent/server/config.example.toml`:

```toml
# Copy to config.toml and edit locally.

[llm]
# claude = current Claude Agent SDK path
# openai_codex_oauth = ChatGPT/Codex subscription OAuth path using Pi auth
provider = "openai_codex_oauth"
model = "gpt-5.5"

[openai_codex]
# Codex Responses backend used by Pi/OpenClaw-style subscription auth.
base_url = "https://chatgpt.com/backend-api/codex"
# Optional. Defaults to $PI_CODING_AGENT_DIR/auth.json or ~/.pi/agent/auth.json.
# pi_auth_file = "C:/Users/Samuel/.pi/agent/auth.json"
auth_profile = "openai-codex"
# Use :memory: for one WebRTC session only, or a file path for persisted history.
session_db = ":memory:"

[mcp.robot]
url = "http://127.0.0.1:8765/mcp"
```

- [ ] **Step 7: Run config tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_agent_config.py -v
```

Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
cd pipecat-agent
git add server/pyproject.toml server/uv.lock server/agent_config.py server/config.example.toml server/tests/test_agent_config.py
git commit -m "feat: add provider config loader"
```

---

### Task 2: Add Pi Codex OAuth credential reader and refresher

**Files:**
- Create: `pipecat-agent/server/codex_auth.py`
- Create: `pipecat-agent/server/tests/test_codex_auth.py`

- [ ] **Step 1: Write auth tests first**

Create `pipecat-agent/server/tests/test_codex_auth.py`:

```python
import base64
import json
import time
from pathlib import Path

import httpx
import pytest

from codex_auth import CodexAuthError, PiCodexCredentialStore


def _jwt(payload: dict) -> str:
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    return f"header.{encoded}.signature"


def test_reads_existing_pi_openai_codex_profile(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expires = int(time.time() * 1000) + 60_000
    access = _jwt({"exp": int(time.time()) + 60, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-1"}})
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": access,
                    "refresh": "refresh-token",
                    "expires": expires,
                    "accountId": "acct-1",
                }
            }
        ),
        encoding="utf-8",
    )

    store = PiCodexCredentialStore(auth_file=auth_file)
    credentials = store.get_credentials()

    assert credentials.access == access
    assert credentials.refresh == "refresh-token"
    assert credentials.account_id == "acct-1"


def test_missing_auth_profile_explains_pi_login(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")

    store = PiCodexCredentialStore(auth_file=auth_file)

    with pytest.raises(CodexAuthError, match="Run `pi`, then `/login`, then select ChatGPT Plus/Pro"):
        store.get_credentials()


def test_refreshes_expired_token_and_persists_result(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expired_access = _jwt({"exp": int(time.time()) - 60})
    refreshed_access = _jwt(
        {"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-2"}}
    )
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": expired_access,
                    "refresh": "refresh-token",
                    "expires": 1,
                    "accountId": "acct-1",
                }
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://auth.openai.com/oauth/token"
        assert "grant_type=refresh_token" in request.content.decode()
        assert "client_id=app_EMoamEEZ73f0CkXaXp7hrann" in request.content.decode()
        return httpx.Response(
            200,
            json={
                "access_token": refreshed_access,
                "refresh_token": "new-refresh-token",
                "expires_in": 3600,
            },
        )

    store = PiCodexCredentialStore(
        auth_file=auth_file,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    credentials = store.get_credentials()

    assert credentials.access == refreshed_access
    assert credentials.refresh == "new-refresh-token"
    assert credentials.account_id == "acct-2"

    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["openai-codex"]["access"] == refreshed_access
    assert saved["openai-codex"]["refresh"] == "new-refresh-token"
    assert saved["openai-codex"]["accountId"] == "acct-2"
```

- [ ] **Step 2: Run auth tests and verify they fail**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_auth.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'codex_auth'`.

- [ ] **Step 3: Implement credential store**

Create `pipecat-agent/server/codex_auth.py`:

```python
from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

OPENAI_CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
DEFAULT_PROFILE = "openai-codex"
REFRESH_SKEW_MS = 60_000


class CodexAuthError(RuntimeError):
    """Raised when Codex OAuth credentials cannot be loaded or refreshed."""


@dataclass(frozen=True)
class CodexCredentials:
    access: str
    refresh: str
    expires: int
    account_id: str


class PiCodexCredentialStore:
    def __init__(
        self,
        auth_file: str | Path | None = None,
        profile: str = DEFAULT_PROFILE,
        client: httpx.Client | None = None,
    ):
        self._auth_file = Path(auth_file) if auth_file else self.default_auth_file()
        self._profile = profile
        self._client = client or httpx.Client(timeout=30)

    @staticmethod
    def default_auth_file() -> Path:
        agent_dir = os.getenv("PI_CODING_AGENT_DIR")
        if agent_dir:
            return Path(agent_dir) / "auth.json"
        return Path.home() / ".pi" / "agent" / "auth.json"

    def get_credentials(self) -> CodexCredentials:
        data = self._read_auth_file()
        entry = data.get(self._profile)
        if not isinstance(entry, dict) or entry.get("type") != "oauth":
            raise CodexAuthError(
                f"Pi Codex OAuth profile '{self._profile}' was not found in {self._auth_file}. "
                "Run `pi`, then `/login`, then select ChatGPT Plus/Pro (Codex)."
            )

        credentials = self._credentials_from_entry(entry)
        if credentials.expires > int(time.time() * 1000) + REFRESH_SKEW_MS:
            return credentials
        return self._refresh_and_save(data, credentials)

    def _read_auth_file(self) -> dict[str, Any]:
        if not self._auth_file.exists():
            raise CodexAuthError(
                f"Pi auth file was not found at {self._auth_file}. "
                "Run `pi`, then `/login`, then select ChatGPT Plus/Pro (Codex)."
            )
        with self._auth_file.open("r", encoding="utf-8") as f:
            parsed = json.load(f)
        if not isinstance(parsed, dict):
            raise CodexAuthError(f"Pi auth file {self._auth_file} is not a JSON object.")
        return parsed

    def _credentials_from_entry(self, entry: dict[str, Any]) -> CodexCredentials:
        access = _require_string(entry, "access")
        refresh = _require_string(entry, "refresh")
        expires = _require_int(entry, "expires")
        account_id = _optional_string(entry, "accountId") or _account_id_from_access_token(access)
        if not account_id:
            raise CodexAuthError("Pi Codex OAuth profile is missing accountId and access token claims.")
        return CodexCredentials(access=access, refresh=refresh, expires=expires, account_id=account_id)

    def _refresh_and_save(
        self,
        data: dict[str, Any],
        credentials: CodexCredentials,
    ) -> CodexCredentials:
        response = self._client.post(
            OPENAI_TOKEN_URL,
            data={
                "grant_type": "refresh_token",
                "refresh_token": credentials.refresh,
                "client_id": OPENAI_CODEX_CLIENT_ID,
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        if response.status_code >= 400:
            raise CodexAuthError(
                f"OpenAI Codex OAuth refresh failed with HTTP {response.status_code}: {response.text}"
            )
        payload = response.json()
        access = _require_string(payload, "access_token")
        refresh = _optional_string(payload, "refresh_token") or credentials.refresh
        expires_in = _optional_int(payload, "expires_in")
        expires = int(time.time() * 1000) + int(expires_in or 3600) * 1000
        account_id = _account_id_from_access_token(access) or credentials.account_id

        updated = CodexCredentials(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=account_id,
        )
        data[self._profile] = {
            "type": "oauth",
            "access": updated.access,
            "refresh": updated.refresh,
            "expires": updated.expires,
            "accountId": updated.account_id,
        }
        self._write_auth_file(data)
        return updated

    def _write_auth_file(self, data: dict[str, Any]) -> None:
        tmp = self._auth_file.with_suffix(self._auth_file.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._auth_file)
        try:
            self._auth_file.chmod(0o600)
        except OSError:
            pass


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"Expected non-empty string field '{key}' in Codex OAuth data.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _require_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, int):
        return value
    raise CodexAuthError(f"Expected integer field '{key}' in Codex OAuth data.")


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) else None


def _decode_jwt_payload(token: str) -> dict[str, Any] | None:
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        padded = parts[1] + "=" * ((4 - len(parts[1]) % 4) % 4)
        decoded = base64.urlsafe_b64decode(padded.encode()).decode()
        payload = json.loads(decoded)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _account_id_from_access_token(token: str) -> str | None:
    payload = _decode_jwt_payload(token)
    auth = payload.get("https://api.openai.com/auth") if payload else None
    if isinstance(auth, dict):
        account_id = auth.get("chatgpt_account_id")
        if isinstance(account_id, str) and account_id:
            return account_id
    return None
```

- [ ] **Step 4: Run auth tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_codex_auth.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd pipecat-agent
git add server/codex_auth.py server/tests/test_codex_auth.py
git commit -m "feat: read pi codex oauth credentials"
```

---

### Task 3: Add OpenAI Codex Agents SDK processor

**Files:**
- Create: `pipecat-agent/server/openai_codex_agent_processor.py`

- [ ] **Step 1: Create processor using OpenAI Agents SDK and local MCP**

Create `pipecat-agent/server/openai_codex_agent_processor.py`:

```python
"""Pipecat processor that runs OpenAI Agents SDK through Codex OAuth."""

from __future__ import annotations

import uuid

from agents import Agent, ModelSettings, Runner, SQLiteSession, set_default_openai_client
from agents.mcp import MCPServerStreamableHttp
from loguru import logger
from openai import AsyncOpenAI
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from agent_config import OpenAICodexConfig
from codex_auth import CodexAuthError, PiCodexCredentialStore
from prompts import SYSTEM_PROMPT


class OpenAICodexAgentProcessor(FrameProcessor):
    """Routes user turns through OpenAI Agents SDK using Pi's Codex OAuth profile."""

    def __init__(self, mcp_server_url: str, model: str, codex_config: OpenAICodexConfig, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model
        self._codex_config = codex_config
        self._credential_store = PiCodexCredentialStore(
            auth_file=codex_config.pi_auth_file,
            profile=codex_config.auth_profile,
        )
        self._mcp_server: MCPServerStreamableHttp | None = None
        self._agent: Agent | None = None
        self._session = SQLiteSession(f"voice-robot-{uuid.uuid4()}", codex_config.session_db)
        self._model_logged = False

    async def connect(self):
        """Initialize Codex credentials, OpenAI client, MCP connection, and agent."""
        if self._agent:
            return

        credentials = self._credential_store.get_credentials()
        client = AsyncOpenAI(
            base_url=self._codex_config.base_url,
            api_key=lambda: self._credential_store.get_credentials().access,
            default_headers={
                "chatgpt-account-id": credentials.account_id,
                "OpenAI-Beta": "responses=experimental",
                "originator": "pipecat-agent",
            },
        )
        set_default_openai_client(client)

        self._mcp_server = MCPServerStreamableHttp(
            name="robot",
            params={
                "url": self._mcp_server_url,
                "timeout": 10,
                "sse_read_timeout": 300,
            },
            cache_tools_list=True,
            max_retry_attempts=2,
        )
        await self._mcp_server.__aenter__()

        self._agent = Agent(
            name="Voice Robot Agent",
            instructions=SYSTEM_PROMPT,
            model=self._model,
            mcp_servers=[self._mcp_server],
            model_settings=ModelSettings(tool_choice="auto"),
        )
        logger.info("OpenAI Codex agent connected")

    async def disconnect(self):
        """Shut down MCP connection."""
        if self._mcp_server:
            await self._mcp_server.__aexit__(None, None, None)
            self._mcp_server = None
        self._agent = None
        logger.info("OpenAI Codex agent disconnected")

    async def _process_with_agent(self, user_text: str):
        if not self._agent:
            await self.push_frame(LLMTextFrame(text="Agent not connected."))
            return

        try:
            result = await Runner.run(self._agent, user_text, session=self._session)
        except CodexAuthError as e:
            logger.error(f"OpenAI Codex auth error: {e}")
            await self.push_frame(
                LLMTextFrame(text="OpenAI Codex authentication failed. Run pi, use /login, and select ChatGPT Plus/Pro (Codex).")
            )
            return
        except Exception as e:
            logger.error(f"OpenAI Codex Agents SDK error: {e}")
            await self.push_frame(LLMTextFrame(text="I encountered an OpenAI Codex agent error. Please try again."))
            return

        if not self._model_logged:
            logger.info(f"OpenAI Codex model: {self._model}")
            self._model_logged = True

        text = str(result.final_output or "").strip()
        if text:
            await self.push_frame(LLMTextFrame(text=text))
        else:
            await self.push_frame(LLMTextFrame(text="I completed the action but have nothing to report."))

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            await self.disconnect()
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMContextFrame):
            messages = frame.context.messages if frame.context else []
            user_text = None
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str) and content.strip():
                        user_text = content.strip()
                        break
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                user_text = part["text"].strip()
                                break
                        if user_text:
                            break

            if user_text:
                logger.info(f"User said: {user_text}")
                await self.push_frame(LLMFullResponseStartFrame())
                await self._process_with_agent(user_text)
                await self.push_frame(LLMFullResponseEndFrame())
            else:
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)
```

- [ ] **Step 2: Run import check**

Run:

```bash
cd pipecat-agent/server
uv run python - <<'PY'
from agent_config import AgentConfig
from openai_codex_agent_processor import OpenAICodexAgentProcessor
cfg = AgentConfig.default()
processor = OpenAICodexAgentProcessor(
    mcp_server_url=cfg.mcp.robot_url,
    model="gpt-5.5",
    codex_config=cfg.openai_codex,
)
print(type(processor).__name__)
PY
```

Expected: prints `OpenAICodexAgentProcessor`.

- [ ] **Step 3: Commit**

```bash
cd pipecat-agent
git add server/openai_codex_agent_processor.py
git commit -m "feat: add openai codex agent processor"
```

---

### Task 4: Add processor factory and wire bot.py

**Files:**
- Create: `pipecat-agent/server/agent_processor_factory.py`
- Create: `pipecat-agent/server/tests/test_agent_processor_factory.py`
- Modify: `pipecat-agent/server/bot.py`
- Modify: `pipecat-agent/server/claude_agent_processor.py`

- [ ] **Step 1: Update Claude processor to accept model from config**

Modify `pipecat-agent/server/claude_agent_processor.py` constructor only:

```python
    def __init__(self, mcp_server_url: str, model: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._mcp_server_url = mcp_server_url
        self._model = model or os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001")
        self._client: ClaudeSDKClient | None = None
        self._model_logged = False
```

- [ ] **Step 2: Write factory tests first**

Create `pipecat-agent/server/tests/test_agent_processor_factory.py`:

```python
from agent_config import AgentConfig, LLMConfig
from agent_processor_factory import create_agent_processor
from claude_agent_processor import ClaudeAgentProcessor
from openai_codex_agent_processor import OpenAICodexAgentProcessor


def test_creates_claude_processor():
    config = AgentConfig.default()
    processor = create_agent_processor(config)

    assert isinstance(processor, ClaudeAgentProcessor)


def test_creates_openai_codex_processor():
    config = AgentConfig(
        llm=LLMConfig(provider="openai_codex_oauth", model="gpt-5.5"),
        openai_codex=AgentConfig.default().openai_codex,
        mcp=AgentConfig.default().mcp,
    )

    processor = create_agent_processor(config)

    assert isinstance(processor, OpenAICodexAgentProcessor)
```

- [ ] **Step 3: Run factory tests and verify they fail**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_agent_processor_factory.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'agent_processor_factory'`.

- [ ] **Step 4: Implement factory**

Create `pipecat-agent/server/agent_processor_factory.py`:

```python
from __future__ import annotations

from pipecat.processors.frame_processor import FrameProcessor

from agent_config import AgentConfig
from claude_agent_processor import ClaudeAgentProcessor
from openai_codex_agent_processor import OpenAICodexAgentProcessor


def create_agent_processor(config: AgentConfig) -> FrameProcessor:
    if config.llm.provider == "claude":
        return ClaudeAgentProcessor(
            mcp_server_url=config.mcp.robot_url,
            model=config.llm.model,
        )
    if config.llm.provider == "openai_codex_oauth":
        return OpenAICodexAgentProcessor(
            mcp_server_url=config.mcp.robot_url,
            model=config.llm.model,
            codex_config=config.openai_codex,
        )
    raise ValueError(f"Unsupported provider: {config.llm.provider}")
```

- [ ] **Step 5: Wire bot.py**

Modify imports in `pipecat-agent/server/bot.py`:

```python
from agent_config import load_agent_config
from agent_processor_factory import create_agent_processor
```

Remove:

```python
from claude_agent_processor import ClaudeAgentProcessor
```

Replace processor creation in `run_bot`:

```python
    config = load_agent_config()
    logger.info(f"LLM provider: {config.llm.provider}, model: {config.llm.model}")

    agent_processor = create_agent_processor(config)
```

Replace the pipeline entry:

```python
            agent_processor,
```

Replace event handlers:

```python
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        logger.info("Client connected")
        await agent_processor.connect()

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        await agent_processor.disconnect()
        await task.cancel()
```

- [ ] **Step 6: Run tests**

Run:

```bash
cd pipecat-agent/server
uv run pytest tests/test_agent_config.py tests/test_codex_auth.py tests/test_agent_processor_factory.py -v
```

Expected: all tests pass.

- [ ] **Step 7: Run static checks**

Run:

```bash
cd pipecat-agent/server
uv run ruff check .
uv run pyright .
```

Expected: no ruff errors. Pyright may report third-party missing stubs for SDKs; if it does, add narrow `# type: ignore` comments only on imports that lack stubs and rerun.

- [ ] **Step 8: Commit**

```bash
cd pipecat-agent
git add server/bot.py server/claude_agent_processor.py server/agent_processor_factory.py server/tests/test_agent_processor_factory.py
git commit -m "feat: select agent processor from config"
```

---

### Task 5: Update docs and local config examples

**Files:**
- Modify: `pipecat-agent/server/.env.example`
- Modify: `pipecat-agent/README.md`

- [ ] **Step 1: Update `.env.example`**

Replace the Claude auth block in `pipecat-agent/server/.env.example` with:

```dotenv
# Agent config
# Copy config.example.toml to config.toml and choose provider/model there.
# AGENT_CONFIG can point to a different TOML config path.
# AGENT_CONFIG=config.toml

# Claude Agent SDK / Anthropic fallback provider
# Use this only when [llm].provider = "claude" in config.toml.
CLAUDE_MODEL=claude-haiku-4-5-20251001

# OpenAI Codex OAuth provider
# Use this when [llm].provider = "openai_codex_oauth" in config.toml.
# Auth is read from Pi's auth file by default:
#   C:/Users/Samuel/.pi/agent/auth.json
# To create it, run `pi`, then `/login`, then select ChatGPT Plus/Pro (Codex).
# This is Codex/ChatGPT subscription auth, not OPENAI_API_KEY Platform billing.
```

Leave the existing Whisper, Kokoro, and MCP variables below the new block.

- [ ] **Step 2: Update README setup**

Modify `pipecat-agent/README.md` setup step 3 to include:

```markdown
3. **Configure environment and agent provider**:

   ```bash
   cp .env.example .env
   cp config.example.toml config.toml
   ```

   For ChatGPT/Codex subscription auth, first authenticate Pi:

   ```bash
   pi
   /login
   # Select ChatGPT Plus/Pro (Codex)
   ```

   Then set `config.toml`:

   ```toml
   [llm]
   provider = "openai_codex_oauth"
   model = "gpt-5.5"
   ```

   This reuses Pi's `~/.pi/agent/auth.json` `openai-codex` OAuth profile.
   It does not use `OPENAI_API_KEY` billing.

   To use the existing Claude path instead:

   ```toml
   [llm]
   provider = "claude"
   model = "claude-haiku-4-5-20251001"
   ```
```

- [ ] **Step 3: Commit docs**

```bash
cd pipecat-agent
git add README.md server/.env.example
git commit -m "docs: document configurable llm providers"
```

---

### Task 6: Live validation and fallback decision

**Files:**
- May modify: `pipecat-agent/server/openai_codex_agent_processor.py`
- May modify: `pipecat-agent/server/config.example.toml`
- May create: `pipecat-agent/server/scripts/smoke_codex_agent.py`

- [ ] **Step 1: Verify Pi auth profile exists without printing secrets**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path.home() / '.pi' / 'agent' / 'auth.json'
data = json.loads(p.read_text())
entry = data.get('openai-codex')
print('has_openai_codex_profile=', isinstance(entry, dict) and entry.get('type') == 'oauth')
print('has_account_id=', bool(entry and entry.get('accountId')))
print('expires=', entry.get('expires') if isinstance(entry, dict) else None)
PY
```

Expected: `has_openai_codex_profile= True` and `has_account_id= True`.

- [ ] **Step 2: Create a smoke script**

Create `pipecat-agent/server/scripts/smoke_codex_agent.py`:

```python
import asyncio

from agent_config import load_agent_config
from openai_codex_agent_processor import OpenAICodexAgentProcessor


async def main():
    config = load_agent_config()
    processor = OpenAICodexAgentProcessor(
        mcp_server_url=config.mcp.robot_url,
        model=config.llm.model,
        codex_config=config.openai_codex,
    )
    await processor.connect()
    print("connected")
    await processor.disconnect()
    print("disconnected")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 3: Run smoke script with OpenAI Codex config**

Ensure `pipecat-agent/server/config.toml` contains:

```toml
[llm]
provider = "openai_codex_oauth"
model = "gpt-5.5"

[openai_codex]
base_url = "https://chatgpt.com/backend-api/codex"
auth_profile = "openai-codex"
session_db = ":memory:"

[mcp.robot]
url = "http://127.0.0.1:8765/mcp"
```

Run while robot MCP server is running:

```bash
cd pipecat-agent/server
uv run python scripts/smoke_codex_agent.py
```

Expected: prints `connected` then `disconnected`.

- [ ] **Step 4: Run one full WebRTC session manually**

Run:

```bash
cd pipecat-agent/server
uv run bot.py
```

Then connect via the existing SmallWebRTC frontend/runner flow and say: “What robot tools can you use?”

Expected:
- Log includes `LLM provider: openai_codex_oauth, model: gpt-5.5`.
- Log includes `OpenAI Codex agent connected`.
- Assistant responds with a concise description of robot/MCP capabilities.

- [ ] **Step 5: Decide fallback only if live validation fails**

If the failure message shows Codex rejected OpenAI Agents SDK request fields, keep `agent_config.py`, `codex_auth.py`, and factory wiring. Replace `OpenAICodexAgentProcessor._process_with_agent()` with a direct `AsyncOpenAI.responses.create(...)` loop and convert MCP tools manually only after capturing the exact rejected field from logs. Do not change the auth design unless refresh/token loading is the failure.

- [ ] **Step 6: Final verification**

Run:

```bash
cd pipecat-agent/server
uv run pytest -v
uv run ruff check .
uv run pyright .
```

Expected: tests pass, ruff passes, pyright has no project-code errors.

- [ ] **Step 7: Commit live validation helper or remove it**

If `scripts/smoke_codex_agent.py` was useful and contains no secrets, keep it:

```bash
cd pipecat-agent
git add server/scripts/smoke_codex_agent.py
git commit -m "test: add openai codex smoke script"
```

If it is only a temporary local helper, delete it and commit any remaining changes:

```bash
cd pipecat-agent
rm server/scripts/smoke_codex_agent.py
git add -A
git commit -m "chore: finalize openai codex provider"
```

---

## Self-review

- **Spec coverage:** The plan adds config-based model/provider selection, reuses Pi's `openai-codex` OAuth subscription profile, keeps Claude as fallback, adds OpenAI Agents SDK + MCP processor, and documents the distinction between Codex OAuth and OpenAI API keys.
- **Placeholder scan:** No implementation step depends on unspecified function names or omitted config fields. Live fallback is conditional on a concrete captured SDK/backend error.
- **Type consistency:** `AgentConfig`, `OpenAICodexConfig`, `PiCodexCredentialStore`, and `OpenAICodexAgentProcessor` names match across tests, factory, and processor code.
