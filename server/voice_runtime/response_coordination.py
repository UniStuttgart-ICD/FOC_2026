from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    Frame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor


class BotResponseCoordinator:
    """Serializes assistant responses until the speech output path is done."""

    def __init__(self) -> None:
        self._response_lock = asyncio.Lock()
        self._active = False

    @property
    def is_response_active(self) -> bool:
        return self._active

    async def begin_response(self) -> None:
        await self._response_lock.acquire()
        self._active = True

    def finish_response(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._response_lock.locked():
            self._response_lock.release()

    def reset_response(self) -> None:
        self._active = False
        if self._response_lock.locked():
            self._response_lock.release()


class BotSpeechOutputCoordinator(FrameProcessor):
    """Keeps response/wake state aligned with TTS output completion."""

    def __init__(
        self,
        *,
        coordinator: BotResponseCoordinator,
        on_response_started: Callable[[], None] | None = None,
        on_response_finished: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._coordinator = coordinator
        self._on_response_started = on_response_started
        self._on_response_finished = on_response_finished
        self._output_active = False
        self._response_open = False
        self._tts_active = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, (CancelFrame, EndFrame)):
            self._finish_output(reset=True)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, LLMFullResponseStartFrame):
            self._start_output()
            self._response_open = True
        elif isinstance(frame, TTSStartedFrame):
            self._start_output()
            self._tts_active = True
        elif isinstance(frame, TTSAudioRawFrame):
            self._start_output()
            self._tts_active = True

        await self.push_frame(frame, direction)

        if isinstance(frame, LLMFullResponseEndFrame):
            self._response_open = False
            self._finish_if_complete()
        elif isinstance(frame, (TTSStoppedFrame, BotStoppedSpeakingFrame)):
            self._tts_active = False
            self._finish_if_complete()

    def _start_output(self) -> None:
        if self._output_active:
            return
        self._output_active = True
        if self._on_response_started is not None:
            self._on_response_started()

    def _finish_if_complete(self) -> None:
        if self._response_open or self._tts_active:
            return
        self._finish_output(reset=False)

    def _finish_output(self, *, reset: bool) -> None:
        if self._output_active and self._on_response_finished is not None:
            self._on_response_finished()
        self._output_active = False
        self._response_open = False
        self._tts_active = False
        if reset:
            self._coordinator.reset_response()
        else:
            self._coordinator.finish_response()
