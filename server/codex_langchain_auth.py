from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codex_oauth.exceptions import NotAuthenticatedError
from codex_oauth.store import AuthStore, OAuthCredentials

from codex_auth import CODEX_PROFILE, CodexAuthError, _account_id_from_jwt, _default_auth_file


class PiLangChainCodexAuthStore(AuthStore):
    """AuthStore-compatible adapter over Pi's ~/.pi/agent/auth.json Codex profile."""

    def __init__(self, *, auth_file: str | Path | None = None, profile: str = CODEX_PROFILE):
        self.auth_path = Path(auth_file) if auth_file is not None else _default_auth_file()
        self._profile = profile

    def load(self) -> OAuthCredentials:
        data = self._read_auth_file()
        profile = self._profile_data(data)
        access = _required_string(profile, "access")
        refresh = _required_string(profile, "refresh")
        expires = _required_int(profile, "expires")
        account_id = _optional_string(profile, "accountId") or _account_id_from_jwt(access)
        if not account_id:
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth account id is missing. Re-run Pi login.")
        return OAuthCredentials(
            access=access,
            refresh=refresh,
            expires=expires,
            account_id=account_id,
        )

    def save(self, creds: OAuthCredentials) -> None:
        data = self._read_auth_file() if self.auth_path.exists() else {}
        data[self._profile] = {
            "type": "oauth",
            "access": creds.access,
            "refresh": creds.refresh,
            "expires": creds.expires,
            "accountId": creds.account_id,
        }
        self.auth_path.parent.mkdir(parents=True, exist_ok=True)
        self.auth_path.write_text(f"{json.dumps(data, indent=2, sort_keys=True)}\n", encoding="utf-8")

    def _read_auth_file(self) -> dict[str, Any]:
        if not self.auth_path.exists():
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth credentials not found. Run `pi`, then `/login`.")
        try:
            data = json.loads(self.auth_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CodexAuthError("Pi OpenAI Codex OAuth auth file is invalid JSON.") from exc
        if not isinstance(data, dict):
            raise CodexAuthError("Pi OpenAI Codex OAuth auth file must contain profiles.")
        return data

    def _profile_data(self, data: dict[str, Any]) -> dict[str, Any]:
        profile = data.get(self._profile)
        if not isinstance(profile, dict) or profile.get("type") != "oauth":
            raise NotAuthenticatedError("Pi OpenAI Codex OAuth profile not found. Run `pi`, then `/login`.")
        return profile


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is missing.")
    return value


def _optional_string(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is invalid.")
    return value


def _required_int(data: dict[str, Any], key: str) -> int:
    value = data.get(key)
    if isinstance(value, bool):
        raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is invalid.")
    if isinstance(value, (int, float)):
        return int(value)
    raise CodexAuthError(f"Pi OpenAI Codex OAuth field {key!r} is missing.")
