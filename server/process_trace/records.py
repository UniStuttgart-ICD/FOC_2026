from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

SECRET_KEY_MARKERS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "credential",
    "password",
    "secret",
    "token",
)


@dataclass(frozen=True)
class TraceOptions:
    include_text: bool = True
    include_tool_payloads: bool = True


class TraceWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None:
        ...


class MemoryTraceWriter:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def write(self, record: dict[str, Any]) -> None:
        self.records.append(record)


def sanitize_value(value: Any, *, key: str | None = None) -> Any:
    if key is not None and _is_secret_key(key):
        return "[REDACTED]"
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, bytearray):
        return f"<bytearray len={len(value)}>"
    if isinstance(value, dict):
        return {
            str(item_key): sanitize_value(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_value(item) for item in value]
    return repr(value)


def sanitize_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    if not attributes:
        return {}
    return {str(key): sanitize_value(value, key=str(key)) for key, value in attributes.items()}


def _is_secret_key(key: str) -> bool:
    normalized_key = "".join(char for char in key.lower() if char.isalnum())
    return any(_normalize_marker(marker) in normalized_key for marker in SECRET_KEY_MARKERS)


def _normalize_marker(marker: str) -> str:
    return "".join(char for char in marker if char.isalnum())
