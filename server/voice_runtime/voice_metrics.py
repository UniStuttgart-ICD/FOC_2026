from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _duration_ms(start: float, end: float | None) -> float | None:
    if end is None:
        return None
    return round((end - start) * 1000, 2)


class VoiceTurnTimeline:
    def __init__(
        self,
        *,
        profile: str,
        category: str,
        turn_id: str,
        started_at: float,
        now_fn: Callable[[], float],
        wall_time_fn: Callable[[], float],
    ):
        self._profile = profile
        self._category = category
        self._turn_id = turn_id
        self._started_at = started_at
        self._now_fn = now_fn
        self._wall_time_fn = wall_time_fn
        self._marks: dict[str, float] = {}
        self._wake_phrase = ""
        self._transcript = ""
        self._response = ""

    def wake_detected(self, wake_phrase: str, at: float | None = None) -> None:
        self._wake_phrase = wake_phrase
        self._mark("wake_detected", at)

    def speech_captured(self, at: float | None = None) -> None:
        self._mark("speech_captured", at)

    def stt_done(self, transcript: str, at: float | None = None) -> None:
        self._transcript = transcript
        self._mark("stt_done", at)

    def agent_done(self, at: float | None = None) -> None:
        self._mark("agent_done", at)

    def append_agent_text(self, text: str) -> None:
        self._response = f"{self._response}{text}"

    def tts_audio_started(self, at: float | None = None) -> None:
        if "tts_first_audio" in self._marks:
            return
        self._mark("tts_first_audio", at)

    def tts_done(self, at: float | None = None) -> None:
        self._mark("tts_done", at)

    def to_record(
        self, *, finished_at: float | None = None, include_text: bool = True
    ) -> dict[str, Any]:
        speech_captured = self._marks.get("speech_captured")
        tts_first_audio = self._marks.get("tts_first_audio")
        wake_detected = self._marks.get("wake_detected")
        speech_start = wake_detected if wake_detected is not None else self._started_at
        record: dict[str, Any] = {
            "timestamp_unix": self._wall_time_fn(),
            "profile": self._profile,
            "category": self._category,
            "turn_id": self._turn_id,
            "wake_phrase": self._wake_phrase,
            "wake_latency_ms": self._elapsed_ms("wake_detected"),
            "speech_captured_ms": _duration_ms(speech_start, speech_captured),
            "stt_latency_ms": self._duration_between_ms("speech_captured", "stt_done"),
            "agent_latency_ms": self._duration_between_ms("stt_done", "agent_done"),
            "tts_first_audio_ms": self._duration_between_ms("agent_done", "tts_first_audio"),
            "tts_done_ms": self._duration_between_ms("tts_first_audio", "tts_done"),
            "total_to_first_audio_ms": _duration_ms(self._started_at, tts_first_audio),
            "total_turn_ms": _duration_ms(self._started_at, self._finished_at(finished_at)),
        }
        if include_text:
            record["transcript"] = self._transcript
            record["response"] = self._response
        return record

    def _mark(self, name: str, at: float | None) -> None:
        self._marks[name] = at if at is not None else self._now_fn()

    def _elapsed_ms(self, mark: str) -> float | None:
        return _duration_ms(self._started_at, self._marks.get(mark))

    def _duration_between_ms(self, start_mark: str, end_mark: str) -> float | None:
        start = self._marks.get(start_mark)
        if start is None:
            return None
        return _duration_ms(start, self._marks.get(end_mark))

    def _finished_at(self, finished_at: float | None) -> float:
        return finished_at if finished_at is not None else self._now_fn()
