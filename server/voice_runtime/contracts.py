from __future__ import annotations

from typing import Protocol


class VoiceRuntimeError(RuntimeError):
    """Raised when a reusable Voice Runtime Module cannot satisfy its Interface."""


class AsyncLifecycle(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...
