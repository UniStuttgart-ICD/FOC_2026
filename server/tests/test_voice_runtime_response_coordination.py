from __future__ import annotations

from typing import Any

import pytest
from pipecat.frames.frames import (
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection

from voice_runtime.response_coordination import (
    BotResponseCoordinator,
    BotSpeechOutputCoordinator,
)


class CapturingOutputCoordinator(BotSpeechOutputCoordinator):
    def __init__(self, coordinator: BotResponseCoordinator, **kwargs: Any) -> None:
        super().__init__(coordinator=coordinator, **kwargs)
        self.pushed: list[Frame] = []

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        self.pushed.append(frame)


@pytest.mark.asyncio
async def test_output_coordinator_releases_after_response_end_and_tts_stop() -> None:
    coordinator = BotResponseCoordinator()
    await coordinator.begin_response()
    events: list[str] = []
    processor = CapturingOutputCoordinator(
        coordinator,
        on_response_started=lambda: events.append("started"),
        on_response_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(TTSStartedFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(
        TTSAudioRawFrame(audio=b"\0\0", sample_rate=24000, num_channels=1),
        FrameDirection.DOWNSTREAM,
    )
    await processor.process_frame(TTSStoppedFrame(), FrameDirection.DOWNSTREAM)

    assert coordinator.is_response_active is True
    assert events == ["started"]

    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert coordinator.is_response_active is False
    assert events == ["started", "finished"]


@pytest.mark.asyncio
async def test_output_coordinator_releases_empty_response_on_response_end() -> None:
    coordinator = BotResponseCoordinator()
    await coordinator.begin_response()
    events: list[str] = []
    processor = CapturingOutputCoordinator(
        coordinator,
        on_response_started=lambda: events.append("started"),
        on_response_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(LLMFullResponseEndFrame(), FrameDirection.DOWNSTREAM)

    assert coordinator.is_response_active is False
    assert events == ["started", "finished"]


@pytest.mark.asyncio
async def test_output_coordinator_resets_on_end_frame() -> None:
    coordinator = BotResponseCoordinator()
    await coordinator.begin_response()
    events: list[str] = []
    processor = CapturingOutputCoordinator(
        coordinator,
        on_response_started=lambda: events.append("started"),
        on_response_finished=lambda: events.append("finished"),
    )

    await processor.process_frame(LLMFullResponseStartFrame(), FrameDirection.DOWNSTREAM)
    await processor.process_frame(EndFrame(), FrameDirection.DOWNSTREAM)

    assert coordinator.is_response_active is False
    assert events == ["started", "finished"]
