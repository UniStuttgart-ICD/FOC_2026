import pytest

from operator_dashboard.security import DashboardSecurity


def test_generate_creates_valid_random_token() -> None:
    security = DashboardSecurity.generate()

    assert len(security.token) >= 32
    assert security.is_valid(security.token)
    assert not security.is_valid("wrong")


def test_explicit_token_validation_is_exact() -> None:
    security = DashboardSecurity(token="abc123")

    assert security.is_valid("abc123")
    assert not security.is_valid("ABC123")
    assert not security.is_valid("")


def test_require_rejects_invalid_token() -> None:
    security = DashboardSecurity(token="abc123")

    with pytest.raises(PermissionError, match="invalid dashboard token"):
        security.require("wrong")
