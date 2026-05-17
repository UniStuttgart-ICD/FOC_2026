from __future__ import annotations

import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class DashboardSecurity:
    token: str

    @classmethod
    def generate(cls) -> "DashboardSecurity":
        return cls(token=secrets.token_urlsafe(32))

    def is_valid(self, token: str | None) -> bool:
        return bool(token) and secrets.compare_digest(token, self.token)

    def require(self, token: str | None) -> None:
        if not self.is_valid(token):
            raise PermissionError("invalid dashboard token")
