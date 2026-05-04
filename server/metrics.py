from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    InputAudioRawFrame,
    LLMFullResponseEndFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection


def _perf_counter() -> float:
    return time.perf_counter()


def _duration_ms(start: float, end: float | None) -> float | None:
    if end is None:
        return None
    return round((end - start) * 1000, 2)


@dataclass
class TurnMetrics:
    turn_id: str
    started_at: float = field(default_factory=_perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    wake_phrase: str = ""
    transcript: str = ""
    response: str = ""

    def mark(self, name: str) -> None:
        self.marks[name] = time.perf_counter()

    def elapsed_ms(self, mark: str) -> float | None:
        return _duration_ms(self.started_at, self.marks.get(mark))

    def duration_ms(self, start_mark: str, end_mark: str) -> float | None:
        start = self.marks.get(start_mark)
        if start is None:
            return None
        return _duration_ms(start, self.marks.get(end_mark))


class VoiceMetricsRecorder:
    def __init__(self, *, profile: str, category: str, path: Path, include_text: bool):
        self._profile = profile
        self._category = category
        self._path = path
        self._include_text = include_text
        self._turns: dict[str, TurnMetrics] = {}
        self._disabled = False

    def start_turn(self, turn_id: str) -> TurnMetrics:
        turn = TurnMetrics(turn_id=turn_id)
        self._turns[turn_id] = turn
        return turn

    def get_turn(self, turn_id: str) -> TurnMetrics | None:
        return self._turns.get(turn_id)

    def mark(self, turn_id: str, name: str) -> TurnMetrics | None:
        turn = self._turns.get(turn_id)
        if turn is None:
            return None
        turn.mark(name)
        return turn

    def finish_turn(self, turn_id: str) -> None:
        turn = self._turns.pop(turn_id, None)
        if turn is None:
            return
        speech_captured = turn.marks.get("speech_captured")
        tts_first_audio = turn.marks.get("tts_first_audio")
        wake_detected = turn.marks.get("wake_detected")
        speech_start = wake_detected if wake_detected is not None else turn.started_at
        record: dict[str, Any] = {
            "timestamp_unix": time.time(),
            "profile": self._profile,
            "category": self._category,
            "turn_id": turn.turn_id,
            "wake_phrase": turn.wake_phrase,
            "wake_latency_ms": turn.elapsed_ms("wake_detected"),
            "speech_captured_ms": _duration_ms(speech_start, speech_captured),
            "stt_latency_ms": turn.duration_ms("speech_captured", "stt_done"),
            "agent_latency_ms": turn.duration_ms("stt_done", "agent_done"),
            "tts_first_audio_ms": turn.duration_ms("agent_done", "tts_first_audio"),
            "tts_done_ms": turn.duration_ms("tts_first_audio", "tts_done"),
            "total_to_first_audio_ms": _duration_ms(turn.started_at, tts_first_audio),
            "total_turn_ms": _duration_ms(turn.started_at, time.perf_counter()),
        }
        if self._include_text:
            record["transcript"] = turn.transcript
            record["response"] = turn.response
        self._write(record)
        logger.info(
            "Voice metrics profile={} turn={} total={}ms transcript={!r}",
            self._profile,
            turn.turn_id,
            record["total_turn_ms"],
            turn.transcript[:120],
        )

    def _write(self, record: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            self._disabled = True
            logger.warning(f"Disabling voice metrics after write failure: {exc}")


class VoiceMetricsObserver(BaseObserver):
    """Records turn metrics from Pipecat frame flow."""

    def __init__(self, recorder: VoiceMetricsRecorder, **kwargs):
        super().__init__(**kwargs)
        self._recorder = recorder
        self._current_turn_id: str | None = None
        self._turn_count = 0
        self._wake_marked = False
        self._stt_marked = False
        self._tts_first_audio_marked = False

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        frame = data.frame
        if isinstance(frame, InputAudioRawFrame):
            if data.source.__class__.__name__ == "MaveWakeWordGate":
                self._mark_wake_detected()
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._ensure_turn()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._mark("speech_captured")
        elif isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._record_transcription(frame)
        elif isinstance(frame, LLMTextFrame):
            self._append_response(frame.text)
        elif isinstance(frame, LLMFullResponseEndFrame):
            self._mark("agent_done")
        elif isinstance(frame, TTSAudioRawFrame):
            self._mark_tts_first_audio()
        elif isinstance(frame, TTSStoppedFrame | BotStoppedSpeakingFrame):
            self._finish_turn()

    def _ensure_turn(self) -> TurnMetrics:
        if self._current_turn_id is not None:
            turn = self._recorder.get_turn(self._current_turn_id)
            if turn is not None:
                return turn

        self._turn_count += 1
        self._current_turn_id = f"turn-{self._turn_count}"
        self._wake_marked = False
        self._stt_marked = False
        self._tts_first_audio_marked = False
        return self._recorder.start_turn(self._current_turn_id)

    def _current_turn(self) -> TurnMetrics | None:
        if self._current_turn_id is None:
            return None
        return self._recorder.get_turn(self._current_turn_id)

    def _mark(self, name: str) -> None:
        if self._current_turn_id is None:
            self._ensure_turn()
        if self._current_turn_id is not None:
            self._recorder.mark(self._current_turn_id, name)

    def _mark_wake_detected(self) -> None:
        turn = self._ensure_turn()
        if self._wake_marked:
            return
        turn.wake_phrase = "mave"
        self._mark("wake_detected")
        self._wake_marked = True

    def _record_transcription(self, frame: TranscriptionFrame) -> None:
        turn = self._ensure_turn()
        turn.transcript = frame.text
        if not self._stt_marked:
            self._mark("stt_done")
            self._stt_marked = True

    def _append_response(self, text: str) -> None:
        turn = self._current_turn()
        if turn is None:
            return
        turn.response = f"{turn.response}{text}"

    def _mark_tts_first_audio(self) -> None:
        if self._tts_first_audio_marked:
            return
        self._mark("tts_first_audio")
        self._tts_first_audio_marked = True

    def _finish_turn(self) -> None:
        if self._current_turn_id is None:
            return
        self._mark("tts_done")
        self._recorder.finish_turn(self._current_turn_id)
        self._current_turn_id = None
        self._wake_marked = False
        self._stt_marked = False
        self._tts_first_audio_marked = False
