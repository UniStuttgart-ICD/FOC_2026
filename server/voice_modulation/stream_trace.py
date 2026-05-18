from __future__ import annotations

import json
import logging
import math
import struct
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

VOICE_STREAM_UTTERANCE_ID = "voice_stream_utterance_id"
VOICE_STREAM_CHUNK_SEQ = "voice_stream_chunk_seq"
VOICE_STREAM_SOURCE = "voice_stream_source"
VOICE_STREAM_TRACE_BASE_PATH = Path("logs/voice_modulation_stream_trace.jsonl")

LOGGER = logging.getLogger(__name__)


class VoiceStreamTraceWriter(Protocol):
    def write(self, record: dict[str, Any]) -> None:
        ...


class VoiceStreamTracerProtocol(Protocol):
    def event(self, event: str, **attributes: Any) -> None:
        ...


class JsonlVoiceStreamTraceWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._disabled = False
        self._warned = False

    @property
    def path(self) -> Path:
        return self._path

    def write(self, record: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with self._path.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
        except OSError as exc:
            self._disabled = True
            if not self._warned:
                self._warned = True
                LOGGER.warning("Disabling voice stream trace writer after write failure: %s", exc)


class VoiceStreamTracer:
    def __init__(
        self,
        writer: VoiceStreamTraceWriter,
        *,
        session_id: str,
        clock: Callable[[], float] | None = None,
        utc_now: Callable[[], datetime] | None = None,
    ) -> None:
        self._writer = writer
        self._session_id = session_id
        self._clock = clock or time.perf_counter
        self._utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._started_at = self._clock()
        self._sequence = 0

    @property
    def session_id(self) -> str:
        return self._session_id

    def event(self, event: str, **attributes: Any) -> None:
        self._sequence += 1
        timestamp = self._utc_now().astimezone(timezone.utc)
        record = {
            "event": event,
            "sequence": self._sequence,
            "session_id": self._session_id,
            "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
            "elapsed_ms": round((self._clock() - self._started_at) * 1000.0, 3),
        }
        record.update(_sanitize_attributes(attributes))
        self._writer.write(record)


def pcm16_audio_metrics(
    audio: bytes,
    *,
    sample_rate: int,
    num_channels: int,
) -> dict[str, Any]:
    sample_count = len(audio) // 2
    duration_ms = 0.0
    if sample_rate > 0 and num_channels > 0:
        duration_ms = (sample_count / (sample_rate * num_channels)) * 1000.0
    rms, peak = _pcm16_levels(audio)
    return {
        "audio_bytes": len(audio),
        "duration_ms": round(duration_ms, 3),
        "rms": round(rms, 3),
        "peak": peak,
    }


def _pcm16_levels(audio: bytes) -> tuple[float, int]:
    usable = audio[: len(audio) - (len(audio) % 2)]
    if not usable:
        return 0.0, 0

    total_square = 0
    peak = 0
    count = 0
    for (sample,) in struct.iter_unpack("<h", usable):
        value = int(sample)
        absolute = abs(value)
        peak = max(peak, absolute)
        total_square += value * value
        count += 1
    if count == 0:
        return 0.0, 0
    return math.sqrt(total_square / count), peak


def _sanitize_attributes(attributes: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _sanitize_value(value) for key, value in attributes.items()}


def _sanitize_value(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    if isinstance(value, bytearray):
        return f"<bytearray len={len(value)}>"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _sanitize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    return repr(value)
