from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection

from process_trace.context import TraceContext
from process_trace.trace import ProcessTracer
from voice_runtime.wake_command import WakeDetectedFrame

_MODULE = "voice_runtime"


@dataclass(frozen=True)
class _OpenSpan:
    started_at_unix_ns: int
    context: TraceContext


class ProcessTraceObserver(BaseObserver):
    """Adapts Pipecat voice frames into process trace records."""

    def __init__(
        self,
        tracer: ProcessTracer,
        *,
        session_context: TraceContext,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._tracer = tracer
        self._session_context = session_context
        self._turn_context: TraceContext | None = None
        self._speech_capture_span: _OpenSpan | None = None
        self._stt_span: _OpenSpan | None = None
        self._tts_span: _OpenSpan | None = None
        self._tts_first_audio_recorded = False
        self._recorded_wake_frame_ids: set[int] = set()
        self._recorded_stt_frame_ids: set[int] = set()

    async def on_push_frame(self, data: FramePushed):
        if data.direction != FrameDirection.DOWNSTREAM:
            return

        frame = data.frame
        if isinstance(frame, WakeDetectedFrame):
            self._record_wake(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._start_speech_capture()
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._stop_speech_capture_start_stt()
        elif isinstance(frame, TranscriptionFrame) and frame.finalized:
            self._stop_stt(frame)
        elif isinstance(frame, LLMTextFrame):
            self._start_tts()
        elif isinstance(frame, TTSAudioRawFrame):
            self._record_tts_first_audio(frame)
        elif isinstance(frame, TTSStoppedFrame | BotStoppedSpeakingFrame):
            self._finish_tts_turn()

    def _ensure_turn(self) -> TraceContext:
        if self._turn_context is not None:
            return self._turn_context

        self._turn_context = self._tracer.start_turn(context=self._session_context)
        return self._turn_context

    def _record_wake(self, frame: WakeDetectedFrame) -> None:
        frame_id = id(frame)
        if frame_id in self._recorded_wake_frame_ids:
            return
        self._recorded_wake_frame_ids.add(frame_id)

        context = self._ensure_turn()
        attributes: dict[str, Any] = {
            "score": frame.score,
        }
        if self._tracer.options.include_text:
            attributes["wake_phrase"] = frame.wake_phrase
        if frame.model_name is not None:
            attributes["model_name"] = frame.model_name
        self._tracer.event("voice.wake", _MODULE, attributes=attributes, context=context)

    def _start_speech_capture(self) -> None:
        if self._speech_capture_span is None:
            self._speech_capture_span = self._open_span()

    def _stop_speech_capture_start_stt(self) -> None:
        ended_at_unix_ns = time.time_ns()
        if self._speech_capture_span is not None:
            self._tracer.record_span(
                "voice.speech_capture",
                _MODULE,
                started_at_unix_ns=self._speech_capture_span.started_at_unix_ns,
                ended_at_unix_ns=ended_at_unix_ns,
                context=self._speech_capture_span.context,
            )
            self._speech_capture_span = None
        self._stt_span = _OpenSpan(ended_at_unix_ns, self._ensure_turn())

    def _stop_stt(self, frame: TranscriptionFrame) -> None:
        frame_id = id(frame)
        if frame_id in self._recorded_stt_frame_ids:
            return
        self._recorded_stt_frame_ids.add(frame_id)

        span = self._stt_span or self._open_span()
        attributes = {}
        if self._tracer.options.include_text:
            attributes["transcript"] = frame.text
        self._tracer.record_span(
            "voice.stt",
            _MODULE,
            started_at_unix_ns=span.started_at_unix_ns,
            ended_at_unix_ns=time.time_ns(),
            attributes=attributes,
            context=span.context,
        )
        self._stt_span = None

    def _start_tts(self) -> None:
        if self._tts_span is None:
            self._tts_span = self._open_span()

    def _record_tts_first_audio(self, frame: TTSAudioRawFrame) -> None:
        if self._tts_first_audio_recorded or self._turn_context is None:
            return
        self._tracer.event(
            "voice.tts_first_audio",
            _MODULE,
            attributes={"audio_bytes": len(frame.audio)},
            context=self._turn_context,
        )
        self._tts_first_audio_recorded = True

    def _finish_tts_turn(self) -> None:
        if self._tts_span is not None:
            self._tracer.record_span(
                "voice.tts",
                _MODULE,
                started_at_unix_ns=self._tts_span.started_at_unix_ns,
                ended_at_unix_ns=time.time_ns(),
                context=self._tts_span.context,
            )
        self._reset_turn()

    def _open_span(self) -> _OpenSpan:
        return _OpenSpan(started_at_unix_ns=time.time_ns(), context=self._ensure_turn())

    def _reset_turn(self) -> None:
        self._turn_context = None
        self._speech_capture_span = None
        self._stt_span = None
        self._tts_span = None
        self._tts_first_audio_recorded = False
        self._recorded_wake_frame_ids.clear()
        self._recorded_stt_frame_ids.clear()
