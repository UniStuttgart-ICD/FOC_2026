from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
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

from voice_runtime.voice_metrics import VoiceTurnTimeline
from voice_runtime.wake_command import WakeDetectedFrame


def _perf_counter() -> float:
    return time.perf_counter()


class TurnMetrics:
    def __init__(self, *, profile: str, category: str, turn_id: str):
        self.turn_id = turn_id
        self._timeline = VoiceTurnTimeline(
            profile=profile,
            category=category,
            turn_id=turn_id,
            started_at=_perf_counter(),
            now_fn=time.perf_counter,
            wall_time_fn=time.time,
        )
        self._wake_phrase = ""
        self._transcript = ""
        self._response = ""

    @property
    def wake_phrase(self) -> str:
        return self._wake_phrase

    @wake_phrase.setter
    def wake_phrase(self, value: str) -> None:
        self._wake_phrase = value

    @property
    def transcript(self) -> str:
        return self._transcript

    @transcript.setter
    def transcript(self, value: str) -> None:
        self._transcript = value

    @property
    def response(self) -> str:
        return self._response

    @response.setter
    def response(self, value: str) -> None:
        self._response = value

    def mark(self, name: str) -> None:
        if name == "wake_detected":
            self._timeline.wake_detected(self._wake_phrase)
        elif name == "speech_captured":
            self._timeline.speech_captured()
        elif name == "stt_done":
            self._timeline.stt_done(self._transcript)
        elif name == "agent_done":
            self._timeline.agent_done()
        elif name == "tts_first_audio":
            self._timeline.tts_audio_started()
        elif name == "tts_done":
            self._timeline.tts_done()

    def append_response(self, text: str) -> None:
        self._response = f"{self._response}{text}"
        self._timeline.append_agent_text(text)

    def record(self, *, finished_at: float, include_text: bool) -> dict[str, Any]:
        if self._transcript:
            self._timeline.stt_done(
                self._transcript,
                at=self._timeline._marks.get("stt_done"),
            )
        if self._response and not self._timeline._response:
            self._timeline.append_agent_text(self._response)
        return self._timeline.to_record(finished_at=finished_at, include_text=include_text)


class VoiceMetricsRecorder:
    def __init__(self, *, profile: str, category: str, path: Path, include_text: bool):
        self._profile = profile
        self._category = category
        self._path = path
        self._include_text = include_text
        self._turns: dict[str, TurnMetrics] = {}
        self._disabled = False

    def start_turn(self, turn_id: str) -> TurnMetrics:
        turn = TurnMetrics(profile=self._profile, category=self._category, turn_id=turn_id)
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
        record = turn.record(finished_at=time.perf_counter(), include_text=self._include_text)
        self._write(record)
        logger.info(
            "Voice metrics profile={} turn={} total={}ms transcript={!r}",
            self._profile,
            turn.turn_id,
            record["total_turn_ms"],
            turn.transcript[:120],
        )

    def discard_turn(self, turn_id: str) -> None:
        self._turns.pop(turn_id, None)

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
        if isinstance(frame, WakeDetectedFrame):
            self._mark_wake_detected(frame.wake_phrase)
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

    def _mark_wake_detected(self, wake_phrase: str) -> None:
        turn = self._ensure_turn()
        if self._wake_marked:
            return
        turn.wake_phrase = wake_phrase
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
        turn.append_response(text)

    def _mark_tts_first_audio(self) -> None:
        if self._tts_first_audio_marked:
            return
        if self._current_turn() is None:
            return
        self._mark("tts_first_audio")
        self._tts_first_audio_marked = True

    def _finish_turn(self) -> None:
        if self._current_turn_id is None:
            return
        turn = self._current_turn()
        if turn is not None and not turn.transcript and not turn.response:
            self._recorder.discard_turn(self._current_turn_id)
        else:
            self._mark("tts_done")
            self._recorder.finish_turn(self._current_turn_id)
        self._current_turn_id = None
        self._wake_marked = False
        self._stt_marked = False
        self._tts_first_audio_marked = False
