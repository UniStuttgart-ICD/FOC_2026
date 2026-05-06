import json
from pathlib import Path

from codex_langchain_auth import PiLangChainCodexAuthStore


def test_pi_langchain_auth_store_loads_pi_oauth_profile(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "access-token",
                    "refresh": "refresh-token",
                    "expires": 9_999_999_999_999,
                    "accountId": "account-id",
                }
            }
        ),
        encoding="utf-8",
    )

    store = PiLangChainCodexAuthStore(auth_file=auth_file)

    creds = store.load()

    assert creds.access == "access-token"
    assert creds.refresh == "refresh-token"
    assert creds.expires == 9_999_999_999_999
    assert creds.account_id == "account-id"


def test_pi_langchain_auth_store_saves_back_to_pi_profile(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "openai-codex": {
                    "type": "oauth",
                    "access": "old-access",
                    "refresh": "old-refresh",
                    "expires": 1,
                    "accountId": "old-account",
                }
            }
        ),
        encoding="utf-8",
    )
    store = PiLangChainCodexAuthStore(auth_file=auth_file)
    creds_type = type(store.load())

    store.save(
        creds_type(
            access="new-access",
            refresh="new-refresh",
            expires=2,
            account_id="new-account",
        )
    )

    data = json.loads(auth_file.read_text(encoding="utf-8"))
    assert data["openai-codex"] == {
        "type": "oauth",
        "access": "new-access",
        "refresh": "new-refresh",
        "expires": 2,
        "accountId": "new-account",
    }
