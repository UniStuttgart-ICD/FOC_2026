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
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth", "access": access, "refresh": "refresh-token", "expires": expires, "accountId": "acct-1"}}), encoding="utf-8")

    credentials = PiCodexCredentialStore(auth_file=auth_file).get_credentials()

    assert credentials.access == access
    assert credentials.refresh == "refresh-token"
    assert credentials.account_id == "acct-1"


def test_missing_auth_profile_explains_pi_login(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    auth_file.write_text("{}", encoding="utf-8")

    with pytest.raises(CodexAuthError, match="Run `pi`, then `/login`, then select ChatGPT Plus/Pro"):
        PiCodexCredentialStore(auth_file=auth_file).get_credentials()


def test_refreshes_expired_token_and_persists_result(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expired_access = _jwt({"exp": int(time.time()) - 60})
    refreshed_access = _jwt({"exp": int(time.time()) + 3600, "https://api.openai.com/auth": {"chatgpt_account_id": "acct-2"}})
    auth_file.write_text(json.dumps({"openai-codex": {"type": "oauth", "access": expired_access, "refresh": "refresh-token", "expires": 1, "accountId": "acct-1"}}), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://auth.openai.com/oauth/token"
        assert "grant_type=refresh_token" in request.content.decode()
        return httpx.Response(200, json={"access_token": refreshed_access, "refresh_token": "new-refresh-token", "expires_in": 3600})

    store = PiCodexCredentialStore(auth_file=auth_file, client=httpx.Client(transport=httpx.MockTransport(handler)))
    credentials = store.get_credentials()

    assert credentials.access == refreshed_access
    assert credentials.refresh == "new-refresh-token"
    assert credentials.account_id == "acct-2"
    saved = json.loads(auth_file.read_text(encoding="utf-8"))
    assert saved["openai-codex"]["access"] == refreshed_access


def test_refresh_invalid_json_response_raises_codex_auth_error(tmp_path: Path):
    auth_file = tmp_path / "auth.json"
    expired_access = _jwt({"exp": int(time.time()) - 60})
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": expired_access,
                    "refresh": "refresh-token",
                    "expires": 1,
                }
            }
        ),
        encoding="utf-8",
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not-json")

    store = PiCodexCredentialStore(
        auth_file=auth_file,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(CodexAuthError) as exc_info:
        store.get_credentials()

    message = str(exc_info.value)
    assert "OpenAI Codex OAuth token refresh returned invalid JSON" in message
    assert "Run `pi`, then `/login`, then select ChatGPT Plus/Pro" in message
    assert "not-json" not in message
