import pytest

from voice_runtime.contracts import AsyncLifecycle, VoiceRuntimeError


class FakeLifecycle:
    def __init__(self):
        self.events = []

    async def connect(self) -> None:
        self.events.append("connect")

    async def disconnect(self) -> None:
        self.events.append("disconnect")


@pytest.mark.asyncio
async def test_async_lifecycle_protocol_is_structural():
    lifecycle: AsyncLifecycle = FakeLifecycle()

    await lifecycle.connect()
    await lifecycle.disconnect()

    assert lifecycle.events == ["connect", "disconnect"]


def test_voice_runtime_error_is_runtime_specific():
    error = VoiceRuntimeError("bad profile")

    assert str(error) == "bad profile"
