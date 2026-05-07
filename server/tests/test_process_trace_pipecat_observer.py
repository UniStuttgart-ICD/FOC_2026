from __future__ import annotations

import pytest
from pipecat.frames.frames import (
    BotStoppedSpeakingFrame,
    LLMTextFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from process_trace import MemoryTraceWriter, ProcessTracer, TraceContext, TraceOptions
from process_trace.pipecat_observer import ProcessTraceObserver
from voice_runtime.wake_command import WakeDetectedFrame


def _pushed(frame):
    return FramePushed(
        source=FrameProcessor(),
        destination=FrameProcessor(),
        frame=frame,
        direction=FrameDirection.DOWNSTREAM,
        timestamp=0,
    )


def _observer(*, include_text: bool = True):
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer, TraceOptions(include_text=include_text))
    observer = ProcessTraceObserver(
        tracer,
        session_context=TraceContext(trace_id="trace", session_id="session"),
    )
    return observer, writer


def _records(writer: MemoryTraceWriter, name: str) -> list[dict[str, object]]:
    return [record for record in writer.records if record["name"] == name]


@pytest.mark.asyncio
async def test_observer_emits_voice_wake_event() -> None:
    observer, writer = _observer()

    await observer.on_push_frame(
        _pushed(WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91))
    )

    wake = _records(writer, "voice.wake")[0]
    assert wake["record_type"] == "event"
    assert wake["trace_id"] == "trace"
    assert wake["session_id"] == "session"
    assert wake["turn_id"]
    assert wake["attributes"] == {
        "wake_phrase": "mave",
        "model_name": "mave",
        "score": 0.91,
    }


@pytest.mark.asyncio
async def test_observer_emits_new_turn_for_each_wake_speech_tts_cycle() -> None:
    observer, writer = _observer()

    for text in ["move up", "move down"]:
        await observer.on_push_frame(
            _pushed(WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91))
        )
        await observer.on_push_frame(_pushed(UserStartedSpeakingFrame()))
        await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
        await observer.on_push_frame(
            _pushed(TranscriptionFrame(text=text, user_id="u", timestamp="t", finalized=True))
        )
        await observer.on_push_frame(_pushed(LLMTextFrame(text="Moving.")))
        await observer.on_push_frame(_pushed(TTSStoppedFrame()))

    turn_starts = _records(writer, "trace.turn_start")
    turn_ids = [record["turn_id"] for record in turn_starts]
    assert len(turn_starts) == 2
    assert len(set(turn_ids)) == 2


@pytest.mark.asyncio
async def test_observer_dedupes_same_wake_frame_object() -> None:
    observer, writer = _observer()
    frame = WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91)

    await observer.on_push_frame(_pushed(frame))
    await observer.on_push_frame(_pushed(frame))

    assert len(_records(writer, "voice.wake")) == 1


@pytest.mark.asyncio
async def test_observer_clears_wake_dedupe_after_turn_completion() -> None:
    observer, writer = _observer()
    frame = WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91)

    await observer.on_push_frame(_pushed(frame))
    await observer.on_push_frame(_pushed(LLMTextFrame(text="Moving.")))
    await observer.on_push_frame(_pushed(TTSStoppedFrame()))
    await observer.on_push_frame(_pushed(frame))

    assert len(_records(writer, "voice.wake")) == 2


@pytest.mark.asyncio
async def test_observer_omits_wake_phrase_when_include_text_false() -> None:
    observer, writer = _observer(include_text=False)

    await observer.on_push_frame(
        _pushed(WakeDetectedFrame(wake_phrase="mave", model_name="mave", score=0.91))
    )

    wake = _records(writer, "voice.wake")[0]
    assert wake["attributes"] == {
        "model_name": "mave",
        "score": 0.91,
    }


@pytest.mark.asyncio
async def test_observer_emits_speech_capture_span_from_user_start_stop() -> None:
    observer, writer = _observer()

    await observer.on_push_frame(_pushed(UserStartedSpeakingFrame()))
    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))

    spans = _records(writer, "voice.speech_capture")
    assert len(spans) == 1
    assert spans[0]["record_type"] == "span"
    assert spans[0]["turn_id"]


@pytest.mark.asyncio
async def test_observer_emits_stt_span_with_transcript_only_when_include_text_true() -> None:
    observer, writer = _observer(include_text=True)

    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(
        _pushed(TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True))
    )

    stt = _records(writer, "voice.stt")[0]
    assert stt["record_type"] == "span"
    assert stt["attributes"] == {"transcript": "move up"}

    private_observer, private_writer = _observer(include_text=False)
    await private_observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await private_observer.on_push_frame(
        _pushed(TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True))
    )

    private_stt = _records(private_writer, "voice.stt")[0]
    assert private_stt["attributes"] == {}


@pytest.mark.asyncio
async def test_observer_dedupes_same_finalized_transcription_frame_object() -> None:
    observer, writer = _observer()
    frame = TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True)

    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(frame))
    await observer.on_push_frame(_pushed(frame))

    assert len(_records(writer, "voice.stt")) == 1


@pytest.mark.asyncio
async def test_observer_clears_stt_dedupe_after_turn_completion() -> None:
    observer, writer = _observer()
    frame = TranscriptionFrame(text="move up", user_id="u", timestamp="t", finalized=True)

    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(frame))
    await observer.on_push_frame(_pushed(LLMTextFrame(text="Moving.")))
    await observer.on_push_frame(_pushed(TTSStoppedFrame()))
    await observer.on_push_frame(_pushed(UserStoppedSpeakingFrame()))
    await observer.on_push_frame(_pushed(frame))

    assert len(_records(writer, "voice.stt")) == 2


@pytest.mark.asyncio
async def test_observer_emits_tts_span_and_first_audio_byte_count_without_raw_audio() -> None:
    observer, writer = _observer()

    await observer.on_push_frame(_pushed(LLMTextFrame(text="Moving.")))
    await observer.on_push_frame(
        _pushed(TTSAudioRawFrame(audio=b"\x00\x01\x02", sample_rate=16000, num_channels=1))
    )
    await observer.on_push_frame(_pushed(TTSStoppedFrame()))
    await observer.on_push_frame(
        _pushed(TTSAudioRawFrame(audio=b"\x03\x04", sample_rate=16000, num_channels=1))
    )
    await observer.on_push_frame(_pushed(BotStoppedSpeakingFrame()))

    tts = _records(writer, "voice.tts")[0]
    first_audio = _records(writer, "voice.tts_first_audio")[0]
    assert tts["record_type"] == "span"
    assert first_audio["record_type"] == "event"
    assert first_audio["attributes"] == {"audio_bytes": 3}
    assert "audio" not in first_audio["attributes"]
    assert len(_records(writer, "voice.tts_first_audio")) == 1
