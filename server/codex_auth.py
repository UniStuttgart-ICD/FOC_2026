from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

CODEX_PROFILE = "openai-codex"
TOKEN_URL = "https://auth.openai.com/oauth/token"
LOGIN_GUIDANCE = "Run `pi`, then `/login`, then select ChatGPT Plus/Pro."


class CodexAuthError(RuntimeError):
    """Raised when Pi OpenAI Codex OAuth credentials are missing or invalid."""


@dataclass(frozen=True)
class CodexCredentials:
    access: str
    refresh: str
    account_id: str | None


class PiCodexCredentialStore:
    """Reads and refreshes Pi's OpenAI Codex OAuth credentials."""

    def __init__(
        self,
        *,
        auth_file: str | Path | None = None,
        profile: str = CODEX_PROFILE,
        client: httpx.Client | None = None,
    ):
        self._auth_file = Path(auth_file) if auth_file is not None else _default_auth_file()
        self._profile = profile
        self._client = client

    def get_credentials(self) -> CodexCredentials:
        data = self._read_auth_file()
        profile = self._profile_data(data)
        access = _required_string(profile, "access")
        refresh = _required_string(profile, "refresh")
        account_id = _optional_string(profile, "accountId") or _account_id_from_jwt(access)

        if not self._is_expired(profile, access):
            return CodexCredentials(access=access, refresh=refresh, account_id=account_id)

        return self._refresh_credentials(data, profile, refresh)

    def _read_auth_file(self) -> dict[str, Any]:
        if not self._auth_file.exists():
            raise CodexAuthError(f"OpenAI Codex OAuth credentials not found. {LOGIN_GUIDANCE}")
        try:
            data = json.loads(self._auth_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CodexAuthError(f"OpenAI Codex OAuth auth file is invalid JSON. {LOGIN_GUIDANCE}") from exc
        if not isinstance(data, dict):
            raise CodexAuthError(f"OpenAI Codex OAuth auth file must contain profiles. {LOGIN_GUIDANCE}")
        return data

    def _profile_data(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = data.get(self._profile)
        if not isinstance(profile, dict) or profile.get("type") != "oauth":
            raise CodexAuthError(f"OpenAI Codex OAuth profile '{self._profile}' not found. {LOGIN_GUIDANCE}")
        return profile

    @staticmethod
    def _is_expired(profile: dict[str, Any], access: str) -> bool:
        expires = profile.get("expires")
        now_ms = int(time.time() * 1000)
        if isinstance(expires, (int, float)) and int(expires) <= now_ms:
            return True

        jwt_exp = _jwt_exp(access)
        if jwt_exp is not None and jwt_exp <= int(time.time()):
            return True

        return False

    def _refresh_credentials(
        self,
        data: dict[str, Any],
        profile: dict[str, Any],
        refresh: str,
    ) -> CodexCredentials:
        client = self._client or httpx.Client(timeout=30)
        close_client = self._client is None
        try:
            response = client.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh,
                },
            )
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as exc:
            raise CodexAuthError(f"OpenAI Codex OAuth token refresh failed. {LOGIN_GUIDANCE}") from exc
        finally:
            if close_client:
                client.close()

        if not isinstance(token_data, dict):
            raise CodexAuthError(f"OpenAI Codex OAuth token refresh returned invalid JSON. {LOGIN_GUIDANCE}")

        access = _required_string(token_data, "access_token")
        new_refresh = _optional_string(token_data, "refresh_token") or refresh
        account_id = _account_id_from_jwt(access) or _optional_string(profile, "accountId")
        expires = _expires_ms(token_data, access)

        profile["access"] = access
        profile["refresh"] = new_refresh
        if expires is not None:
            profile["expires"] = expires
        if account_id is not None:
            profile["accountId"] = account_id

        self._persist(data)
        return CodexCredentials(access=access, refresh=new_refresh, account_id=account_id)

    def _persist(self, data: dict[str, Any]) -> None:
        self._auth_file.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps(data, indent=2, sort_keys=True)
        self._auth_file.write_text(f"{serialized}\n", encoding="utf-8")


def _default_auth_file() -> Path:
    return Path.home() / ".pi" / "agent" / "auth.json"


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"OpenAI Codex OAuth credential field '{key}' is missing. {LOGIN_GUIDANCE}")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"OpenAI Codex OAuth credential field '{key}' is invalid. {LOGIN_GUIDANCE}")
    return value


def _expires_ms(token_data: dict[str, Any], access: str) -> int | None:
    expires_in = token_data.get("expires_in")
    if isinstance(expires_in, (int, float)):
        return int((time.time() + float(expires_in)) * 1000)
    jwt_exp = _jwt_exp(access)
    return jwt_exp * 1000 if jwt_exp is not None else None


def _jwt_exp(token: str) -> int | None:
    payload = _jwt_payload(token)
    exp = payload.get("exp")
    return int(exp) if isinstance(exp, (int, float)) else None


def _account_id_from_jwt(token: str) -> str | None:
    auth_claim = _jwt_payload(token).get("https://api.openai.com/auth")
    if not isinstance(auth_claim, dict):
        return None
    account_id = auth_claim.get("chatgpt_account_id")
    return account_id if isinstance(account_id, str) and account_id else None


def _jwt_payload(token: str) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    padding = "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(f"{payload}{padding}")
        data = json.loads(decoded)
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
