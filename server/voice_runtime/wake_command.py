from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from pipecat.frames.frames import Frame, InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

_WAKE_PATTERN = re.compile(
    r"^\s*(?:hey\s+)?mae?ve(?:\b|[\s,;:!?.-])[\s,;:!?.-]*", re.IGNORECASE
)


class WakeWordDetector(Protocol):
    def detected(self, pcm16: NDArray[np.int16]) -> tuple[bool, str | None, float]: ...


def strip_mave_wake_phrase(text: str) -> str:
    """Remove a leading Mave wake phrase from a transcript."""
    stripped = text.strip()
    return _WAKE_PATTERN.sub("", stripped, count=1).strip()


@dataclass
class WakeDetectedFrame(Frame):
    wake_phrase: str
    model_name: str | None
    score: float


class MaveVoiceCommandAudioGate(FrameProcessor):
    """Blocks user audio until Mave is detected, then replays buffered audio."""

    def __init__(
        self,
        detector: WakeWordDetector,
        *,
        pre_buffer_s: float = 1.5,
        rearm_delay_s: float = 0.75,
        max_awake_s: float = 8.0,
        candidate_log_threshold: float = 0.3,
        required_hits: int = 1,
        wake_threshold: float | None = None,
        time_fn: Callable[[], float] = time.monotonic,
        wake_phrase: str = "mave",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if required_hits < 1:
            raise ValueError("required_hits must be at least 1")
        self._detector = detector
        self._pre_buffer_s = pre_buffer_s
        self._rearm_delay_s = rearm_delay_s
        self._max_awake_s = max_awake_s
        self._candidate_log_threshold = candidate_log_threshold
        self._required_hits = required_hits
        self._consecutive_hits = 0
        self._wake_threshold = wake_threshold
        self._time_fn = time_fn
        self._wake_phrase = wake_phrase
        self._ring: deque[InputAudioRawFrame] = deque()
        self._ring_samples = 0
        self._awake = False
        self._wake_started_at: float | None = None
        self._rearm_until = 0.0

    @property
    def is_awake(self) -> bool:
        return self._awake

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if isinstance(frame, InputAudioRawFrame):
            await self._process_audio_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    async def _process_audio_frame(
        self, frame: InputAudioRawFrame, direction: FrameDirection
    ) -> None:
        now = self._time_fn()
        if self._awake and self._awake_timed_out(now):
            logger.info("Wake command window timed out; rearming audio gate")
            self._reset(now=now)

        if self._awake:
            await self.push_frame(frame, direction)
            return

        if now < self._rearm_until:
            return

        self._append_ring(frame)
        pcm16 = self._to_mono_int16(frame)
        detected, model_name, score = self._detector.detected(pcm16)
        rms, peak = _audio_levels(pcm16)
        if not detected:
            self._consecutive_hits = 0
            if model_name is not None and score >= self._candidate_log_threshold:
                logger.debug(
                    self._diagnostic_message(
                        "Wake candidate",
                        model_name=model_name,
                        score=score,
                        rms=rms,
                        peak=peak,
                        gate_open=False,
                    )
                )
            return

        self._consecutive_hits += 1
        gate_open = self._consecutive_hits >= self._required_hits
        if gate_open:
            log_message = self._diagnostic_message(
                "Wake word detected",
                model_name=model_name,
                score=score,
                rms=rms,
                peak=peak,
                gate_open=True,
            )
            logger.info(log_message)
        elif model_name is not None and score >= self._candidate_log_threshold:
            logger.debug(
                self._diagnostic_message(
                    "Wake candidate",
                    model_name=model_name,
                    score=score,
                    rms=rms,
                    peak=peak,
                    gate_open=False,
                )
            )

        if self._consecutive_hits < self._required_hits:
            return

        event_wake_phrase = model_name or self._wake_phrase
        self._awake = True
        self._wake_started_at = now
        buffered = list(self._ring)
        self._ring.clear()
        self._ring_samples = 0

        await self.push_frame(
            WakeDetectedFrame(
                wake_phrase=event_wake_phrase,
                model_name=model_name,
                score=score,
            ),
            direction,
        )
        for buffered_frame in buffered:
            await self.push_frame(buffered_frame, direction)

    def _awake_timed_out(self, now: float) -> bool:
        return self._wake_started_at is not None and now - self._wake_started_at >= self._max_awake_s

    def _append_ring(self, frame: InputAudioRawFrame) -> None:
        self._ring.append(frame)
        self._ring_samples += len(frame.audio) // 2 // max(frame.num_channels, 1)
        max_samples = int(frame.sample_rate * self._pre_buffer_s)
        while self._ring and self._ring_samples > max_samples:
            old = self._ring.popleft()
            self._ring_samples -= len(old.audio) // 2 // max(old.num_channels, 1)

    def reset(self) -> None:
        self._reset(now=self._time_fn())

    def _reset(self, *, now: float) -> None:
        self._awake = False
        self._wake_started_at = None
        self._ring.clear()
        self._ring_samples = 0
        self._rearm_until = now + self._rearm_delay_s
        self._consecutive_hits = 0

    def _diagnostic_message(
        self,
        prefix: str,
        *,
        model_name: str | None,
        score: float,
        rms: float,
        peak: int,
        gate_open: bool,
    ) -> str:
        threshold = "n/a" if self._wake_threshold is None else f"{self._wake_threshold:.3f}"
        model = model_name or self._wake_phrase
        return (
            f"{prefix}: model={model} score={score:.3f} threshold={threshold} "
            f"hits={self._consecutive_hits}/{self._required_hits} rms={rms:.1f} "
            f"peak={peak} gate_open={str(gate_open).lower()}"
        )

    @staticmethod
    def _to_mono_int16(frame: InputAudioRawFrame) -> NDArray[np.int16]:
        pcm = np.frombuffer(frame.audio, dtype=np.int16)
        if frame.num_channels <= 1:
            return pcm
        mono = pcm.reshape(-1, frame.num_channels).mean(axis=1).astype(np.int16)
        return np.asarray(mono, dtype=np.int16)


def _audio_levels(pcm16: NDArray[np.int16]) -> tuple[float, int]:
    if pcm16.size == 0:
        return 0.0, 0
    samples = pcm16.astype(np.float64)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = int(np.max(np.abs(pcm16.astype(np.int32))))
    return rms, peak


class MaveVoiceCommandTranscriptAdapter(FrameProcessor):
    """Removes the leading wake phrase and rearms the audio gate after a command."""

    def __init__(
        self,
        *,
        on_finalized_transcription: Callable[[], None] | None = None,
        single_command: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._on_finalized_transcription = on_finalized_transcription
        self._single_command = single_command

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)

        if not isinstance(frame, TranscriptionFrame):
            await self.push_frame(frame, direction)
            return

        finalized_transcription = frame.finalized
        cleaned_text = strip_mave_wake_phrase(frame.text)
        if cleaned_text:
            await self.push_frame(
                TranscriptionFrame(
                    text=cleaned_text,
                    user_id=frame.user_id,
                    timestamp=frame.timestamp,
                    language=frame.language,
                    result=frame.result,
                    finalized=frame.finalized,
                ),
                direction,
            )

        if (
            finalized_transcription
            and self._single_command
            and self._on_finalized_transcription is not None
        ):
            self._on_finalized_transcription()


@dataclass(frozen=True)
class MaveVoiceCommandProcessors:
    audio_gate: MaveVoiceCommandAudioGate
    transcript_adapter: MaveVoiceCommandTranscriptAdapter


def build_mave_voice_command_processors(
    *,
    detector: WakeWordDetector,
    pre_buffer_s: float = 1.5,
    rearm_delay_s: float = 0.75,
    max_awake_s: float = 8.0,
    single_command: bool = True,
    candidate_log_threshold: float = 0.3,
    required_hits: int = 1,
    wake_threshold: float | None = None,
    time_fn: Callable[[], float] = time.monotonic,
) -> MaveVoiceCommandProcessors:
    audio_gate = MaveVoiceCommandAudioGate(
        detector=detector,
        pre_buffer_s=pre_buffer_s,
        rearm_delay_s=rearm_delay_s,
        max_awake_s=max_awake_s,
        candidate_log_threshold=candidate_log_threshold,
        required_hits=required_hits,
        wake_threshold=wake_threshold,
        time_fn=time_fn,
    )
    transcript_adapter = MaveVoiceCommandTranscriptAdapter(
        on_finalized_transcription=audio_gate.reset,
        single_command=single_command,
    )
    return MaveVoiceCommandProcessors(
        audio_gate=audio_gate,
        transcript_adapter=transcript_adapter,
    )
