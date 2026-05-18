from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport

from config import (
    AgentConfig,
    EmergencyStopConfig,
    MetricsConfig,
    ProcessTraceConfig,
    RobotExecutionConfig,
    RuntimeConfig,
    STTConfig,
    TTSConfig,
    WakeConfig,
)
from metrics import VoiceMetricsObserver
from pipeline_builder import _create_voice_modulation_processor, build_pipeline
from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.profiles import TTSProfile
from voice_runtime.response_coordination import BotResponseCoordinator, BotSpeechOutputCoordinator
from voice_runtime.wake_command import MaveVoiceCommandAudioGate, MaveVoiceCommandTranscriptAdapter
from voice_runtime.wake_tone import WakeToneProcessor


def test_pipeline_builder_uses_voice_runtime_provider_factories():
    import pipeline_builder
    import voice_runtime.providers

    assert pipeline_builder.create_stt_service is voice_runtime.providers.create_stt_service
    assert pipeline_builder.create_tts_service is voice_runtime.providers.create_tts_service


class FakePipeline:
    def __init__(self, processors):
        self.processors = processors


class FakePipelineTask:
    def __init__(self, pipeline, *, params, observers, rtvi_processor=None):
        self.pipeline = pipeline
        self.params = params
        self.observers = observers
        self.rtvi_processor = rtvi_processor


class FakeJsonlTraceWriter:
    def __init__(self, path: Path):
        self.path = path


class FakeJsonlVoiceStreamTraceWriter:
    def __init__(self, path: Path):
        self.path = path


class FakeVoiceStreamTracer:
    def __init__(self, writer, *, session_id: str):
        self.writer = writer
        self.session_id = session_id
        self.records: list[dict[str, Any]] = []

    def event(self, event: str, **attributes: Any) -> None:
        self.records.append({"event": event, **attributes})


class FakeProcessTracer:
    def __init__(self, writer, options):
        self.writer = writer
        self.options = options
        self.started_sessions: list[tuple[str, str, str | None]] = []

    def start_session(self, profile: str, category: str, *, session_id: str | None = None):
        self.started_sessions.append((profile, category, session_id))
        return "session-context"


class FakeNoopProcessTracer:
    def __init__(self, options=None):
        self.options = options
        self.started_sessions: list[tuple[str, str, str | None]] = []

    def start_session(self, profile: str, category: str, *, session_id: str | None = None):
        self.started_sessions.append((profile, category, session_id))
        return "noop-session-context"


class FakeTraceOptions:
    def __init__(self, *, include_text: bool, include_tool_payloads: bool):
        self.include_text = include_text
        self.include_tool_payloads = include_tool_payloads


class FakeProcessTraceObserver:
    def __init__(self, tracer, *, session_context):
        self.tracer = tracer
        self.session_context = session_context


class FakeTransport:
    def input(self):
        return FrameProcessor()

    def output(self):
        return FrameProcessor()


def _config(
    tmp_path: Path,
    *,
    metrics_enabled: bool,
    process_trace_enabled: bool = False,
    wake_enabled: bool = False,
    voice_modulation: object | None = None,
    tts: TTSConfig | None = None,
    robot_execution: RobotExecutionConfig | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        profile_name="no_wake_debug",
        category="local_debug",
        wake=WakeConfig(
            provider="openwakeword" if wake_enabled else "none",
            model_path=tmp_path / "mave.onnx" if wake_enabled else None,
            vad_threshold=0.3,
            required_hits=2,
            min_wake_rms=50.0,
            min_wake_peak=150,
            rearm_delay_s=6.0,
            pre_buffer_s=2.0,
            single_command=False,
            candidate_log_threshold=0.4,
        ),
        emergency_stop=EmergencyStopConfig(enabled=False),
        stt=STTConfig(provider="whisper", model="base", device="cpu"),
        tts=tts or TTSConfig(provider="kokoro", voice="af_heart"),
        agent=AgentConfig(provider="openai_api", model="gpt-5.4-mini"),
        mcp_robot_url="http://127.0.0.1:8765/mcp",
        metrics=MetricsConfig(
            enabled=metrics_enabled,
            path=tmp_path / "metrics.jsonl",
            include_text=True,
        ),
        process_trace=ProcessTraceConfig(
            enabled=process_trace_enabled,
            path=tmp_path / "process_trace.jsonl",
            include_text=True,
            include_tool_payloads=False,
        ),
        robot_execution=robot_execution or RobotExecutionConfig(),
        voice_modulation=voice_modulation,
        server_dir=tmp_path,
    )


def _patch_pipeline_dependencies(monkeypatch, *, agent_processor_kwargs: dict[str, Any] | None = None):
    monkeypatch.setattr("pipeline_builder.create_stt_service", lambda config: FrameProcessor())
    monkeypatch.setattr("pipeline_builder.create_tts_service", lambda config, **kwargs: FrameProcessor())

    def fake_agent_processor(config, *, mcp_server_url, **kwargs):
        if agent_processor_kwargs is not None:
            agent_processor_kwargs.update(kwargs)
        return FrameProcessor()

    monkeypatch.setattr(
        "pipeline_builder.create_agent_processor",
        fake_agent_processor,
    )
    monkeypatch.setattr(
        "pipeline_builder.LLMContextAggregatorPair",
        lambda context, user_params: (FrameProcessor(), FrameProcessor()),
    )
    monkeypatch.setattr("pipeline_builder.Pipeline", FakePipeline)
    monkeypatch.setattr("pipeline_builder.PipelineTask", FakePipelineTask)
    monkeypatch.setattr("pipeline_builder.JsonlTraceWriter", FakeJsonlTraceWriter, raising=False)
    monkeypatch.setattr(
        "pipeline_builder.JsonlVoiceStreamTraceWriter",
        FakeJsonlVoiceStreamTraceWriter,
        raising=False,
    )
    monkeypatch.setattr("pipeline_builder.ProcessTracer", FakeProcessTracer, raising=False)
    monkeypatch.setattr("pipeline_builder.NoopProcessTracer", FakeNoopProcessTracer, raising=False)
    monkeypatch.setattr("pipeline_builder.TraceOptions", FakeTraceOptions, raising=False)
    monkeypatch.setattr("pipeline_builder.VoiceStreamTracer", FakeVoiceStreamTracer, raising=False)
    monkeypatch.setattr(
        "pipeline_builder._create_process_trace_observer",
        lambda tracer, session_context: FakeProcessTraceObserver(
            tracer,
            session_context=session_context,
        ),
        raising=False,
    )


def test_metrics_observer_is_wired_when_metrics_enabled(monkeypatch, tmp_path: Path):
    _patch_pipeline_dependencies(monkeypatch)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=True),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert built.metrics is not None
    assert any(isinstance(observer, VoiceMetricsObserver) for observer in task.observers)


def test_metrics_observer_is_not_wired_when_metrics_disabled(monkeypatch, tmp_path: Path):
    _patch_pipeline_dependencies(monkeypatch)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert built.metrics is None
    assert task.observers == []


def test_process_trace_observer_and_tracer_are_wired_when_enabled(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setattr(
        "pipeline_builder._utc_now",
        lambda: datetime(2026, 5, 7, 12, 34, 56, tzinfo=timezone.utc),
        raising=False,
    )
    monkeypatch.setattr(
        "pipeline_builder._new_session_id",
        lambda: "0123456789abcdef0123456789abcdef",
        raising=False,
    )

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False, process_trace_enabled=True),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert isinstance(built.process_tracer, FakeProcessTracer)
    assert built.process_tracer.writer.path == (
        tmp_path
        / "process_trace"
        / "process_trace-20260507T123456Z-01234567.jsonl"
    )
    assert built.process_tracer.options.include_text is True
    assert built.process_tracer.options.include_tool_payloads is False
    assert built.process_tracer.started_sessions == [
        ("no_wake_debug", "local_debug", "0123456789abcdef0123456789abcdef")
    ]
    assert seen_agent_kwargs["tracer"] is built.process_tracer
    assert len(task.observers) == 1
    observer = task.observers[0]
    assert isinstance(observer, FakeProcessTraceObserver)
    assert observer.tracer is built.process_tracer
    assert observer.session_context == "session-context"


def test_process_trace_disabled_uses_noop_tracer_and_no_observer(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setattr(
        "pipeline_builder._new_session_id",
        lambda: "0123456789abcdef0123456789abcdef",
        raising=False,
    )

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False, process_trace_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert isinstance(built.process_tracer, FakeNoopProcessTracer)
    assert built.process_tracer.started_sessions == [
        ("no_wake_debug", "local_debug", "0123456789abcdef0123456789abcdef")
    ]
    assert seen_agent_kwargs["tracer"] is built.process_tracer
    assert task.observers == []


def test_pipeline_passes_vizor_mcp_env_options_to_agent_processor(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setenv("MCP_VIZOR_URL", "http://127.0.0.1:8001/mcp")
    monkeypatch.setenv("USER_SENSING_MAX_AGE_S", "3.5")

    build_pipeline(
        _config(tmp_path, metrics_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["mcp_vizor_url"] == "http://127.0.0.1:8001/mcp"
    assert seen_agent_kwargs["user_sensing_max_age_s"] == 3.5


def test_pipeline_passes_verified_execution_env_to_agent_processor(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setenv("VERIFIED_EXECUTION_URL", "http://127.0.0.1:8770")

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            robot_execution=RobotExecutionConfig(simulation_only=False),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["verified_execution_url"] == "http://127.0.0.1:8770"


def test_pipeline_simulation_only_blocks_verified_execution_env(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setenv("VERIFIED_EXECUTION_URL", "http://127.0.0.1:8770")

    build_pipeline(
        _config(tmp_path, metrics_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["verified_execution_url"] is None


def test_pipeline_uses_profile_verified_execution_url_when_real_execution_enabled(
    monkeypatch,
    tmp_path: Path,
):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            robot_execution=RobotExecutionConfig(
                simulation_only=False,
                verified_execution_url="http://127.0.0.1:8770",
            ),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["verified_execution_url"] == "http://127.0.0.1:8770"


def test_pipeline_allows_env_override_for_simulation_only(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setenv("ROBOT_EXECUTION_SIMULATION_ONLY", "true")

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            robot_execution=RobotExecutionConfig(
                simulation_only=False,
                verified_execution_url="http://127.0.0.1:8770",
            ),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["verified_execution_url"] is None


def test_pipeline_applies_modular_speech_delivery_to_gemini_live_tts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from agent_control.prompts import SPEECH_DELIVERY_STYLE

    seen_tts_configs: list[TTSProfile] = []
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        "pipeline_builder.create_tts_service",
        lambda config: seen_tts_configs.append(config) or FrameProcessor(),
    )

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            tts=TTSConfig(provider="gemini_live", model="gemini-3.1-flash-live-preview", voice="Kore"),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_tts_configs[0].instructions == SPEECH_DELIVERY_STYLE


def test_pipeline_uses_gemini_live_rtvi_processor_for_gemini_live_tts(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from voice_runtime.rtvi import GeminiLiveConversationRTVIProcessor

    _patch_pipeline_dependencies(monkeypatch)

    built = build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            tts=TTSConfig(provider="gemini_live", model="gemini-3.1-flash-live-preview", voice="Kore"),
        ),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert isinstance(task.rtvi_processor, GeminiLiveConversationRTVIProcessor)


def test_pipeline_keeps_default_rtvi_processor_for_other_tts(monkeypatch, tmp_path: Path) -> None:
    _patch_pipeline_dependencies(monkeypatch)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False, tts=TTSConfig(provider="kokoro", voice="af_heart")),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert task.rtvi_processor is None


def test_pipeline_preserves_explicit_tts_instructions(monkeypatch, tmp_path: Path) -> None:
    seen_tts_configs: list[TTSProfile] = []
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        "pipeline_builder.create_tts_service",
        lambda config: seen_tts_configs.append(config) or FrameProcessor(),
    )

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            tts=TTSConfig(
                provider="gemini_live",
                model="gemini-3.1-flash-live-preview",
                voice="Kore",
                instructions="Use custom lab delivery.",
            ),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_tts_configs[0].instructions == "Use custom lab delivery."


def test_pipeline_disables_vizor_mcp_when_user_sensing_disabled(monkeypatch, tmp_path: Path):
    seen_agent_kwargs: dict[str, Any] = {}
    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setenv("MCP_VIZOR_URL", "http://127.0.0.1:8001/mcp")
    monkeypatch.setenv("USER_SENSING_ENABLED", "false")

    build_pipeline(
        _config(tmp_path, metrics_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_agent_kwargs["mcp_vizor_url"] is None


def test_logs_use_session_scoped_paths(monkeypatch, tmp_path: Path):
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        "pipeline_builder._utc_now",
        lambda: datetime(2026, 5, 7, 12, 34, 56, tzinfo=timezone.utc),
        raising=False,
    )
    monkeypatch.setattr(
        "pipeline_builder._new_session_id",
        lambda: "0123456789abcdef0123456789abcdef",
        raising=False,
    )

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=True, process_trace_enabled=True),
        cast(BaseTransport, FakeTransport()),
    )
    tracer = cast(FakeProcessTracer, built.process_tracer)

    assert tracer.writer.path == (
        tmp_path
        / "process_trace"
        / "process_trace-20260507T123456Z-01234567.jsonl"
    )
    assert built.metrics is not None
    assert built.metrics._path == tmp_path / "metrics" / "metrics-20260507T123456Z-01234567.jsonl"
    assert tracer.started_sessions == [
        ("no_wake_debug", "local_debug", "0123456789abcdef0123456789abcdef")
    ]


def test_voice_modulation_stream_trace_uses_session_path_and_shared_tracer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr(
        "pipeline_builder._utc_now",
        lambda: datetime(2026, 5, 7, 12, 34, 56, tzinfo=timezone.utc),
        raising=False,
    )
    monkeypatch.setattr(
        "pipeline_builder._new_session_id",
        lambda: "0123456789abcdef0123456789abcdef",
        raising=False,
    )
    tts = FrameProcessor()
    seen_tts_tracers: list[FakeVoiceStreamTracer | None] = []

    def fake_create_tts_service(config, *, voice_stream_tracer=None):
        seen_tts_tracers.append(voice_stream_tracer)
        return tts

    monkeypatch.setattr("pipeline_builder.create_tts_service", fake_create_tts_service)

    built = build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            voice_modulation=VoiceModulationSettings(enabled=True, gain_db=3.0),
        ),
        cast(BaseTransport, FakeTransport()),
    )
    processors = cast(FakePipeline, built.pipeline).processors
    voice_modulation = next(
        processor
        for processor in processors
        if isinstance(processor, VoiceModulationProcessor)
    )
    stream_tracer = seen_tts_tracers[0]

    assert isinstance(stream_tracer, FakeVoiceStreamTracer)
    assert stream_tracer.writer.path == (
        tmp_path
        / "logs"
        / "voice_modulation_stream_trace"
        / "voice_modulation_stream_trace-20260507T123456Z-01234567.jsonl"
    )
    assert stream_tracer.session_id == "0123456789abcdef0123456789abcdef"
    assert voice_modulation._voice_stream_tracer is stream_tracer


def test_voice_modulation_stream_trace_is_not_created_when_modulation_disabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_pipeline_dependencies(monkeypatch)
    seen_tts_kwargs: list[dict[str, Any]] = []

    def fake_create_tts_service(config, **kwargs):
        seen_tts_kwargs.append(kwargs)
        return FrameProcessor()

    monkeypatch.setattr("pipeline_builder.create_tts_service", fake_create_tts_service)

    build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            voice_modulation=VoiceModulationSettings(enabled=False),
        ),
        cast(BaseTransport, FakeTransport()),
    )

    assert seen_tts_kwargs == [{}]


def test_wake_enabled_uses_two_voice_command_adapters_around_stt(monkeypatch, tmp_path: Path):
    stt = FrameProcessor()
    tts = FrameProcessor()
    transport_output = FrameProcessor()
    seen_detector_kwargs = {}
    seen_agent_kwargs: dict[str, Any] = {}
    wake_config_logs: list[str] = []

    _patch_pipeline_dependencies(monkeypatch, agent_processor_kwargs=seen_agent_kwargs)
    monkeypatch.setattr("pipeline_builder.create_stt_service", lambda config: stt)
    monkeypatch.setattr("pipeline_builder.create_tts_service", lambda config, **kwargs: tts)
    monkeypatch.setattr(
        "pipeline_builder.logger",
        Mock(info=lambda message, *args: wake_config_logs.append(message.format(*args))),
        raising=False,
    )

    class WakeToneTransport(FakeTransport):
        def output(self):
            return transport_output

    def fake_detector(model_path, *, threshold, vad_threshold):
        seen_detector_kwargs["model_path"] = model_path
        seen_detector_kwargs["threshold"] = threshold
        seen_detector_kwargs["vad_threshold"] = vad_threshold
        detector = Mock()
        detector.detected.return_value = (False, None, 0.0)
        return detector

    monkeypatch.setattr("pipeline_builder.OpenWakeWordDetector", fake_detector)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False, wake_enabled=True),
        cast(BaseTransport, WakeToneTransport()),
    )
    processors = cast(FakePipeline, built.pipeline).processors
    stt_index = processors.index(stt)
    wake_tone_index = next(
        index for index, processor in enumerate(processors) if isinstance(processor, WakeToneProcessor)
    )
    bot_speech_output_index = next(
        index
        for index, processor in enumerate(processors)
        if isinstance(processor, BotSpeechOutputCoordinator)
    )

    assert isinstance(processors[stt_index - 1], MaveVoiceCommandAudioGate)
    assert isinstance(processors[stt_index + 1], MaveVoiceCommandTranscriptAdapter)
    assert processors[wake_tone_index - 1] is tts
    assert processors[wake_tone_index + 1] is transport_output
    assert processors[bot_speech_output_index - 1] is transport_output
    assert processors[stt_index - 1]._wake_threshold == 0.5
    assert processors[stt_index - 1]._pre_buffer_s == 2.0
    assert processors[stt_index - 1]._candidate_log_threshold == 0.4
    assert processors[stt_index - 1]._required_hits == 2
    assert processors[stt_index - 1]._min_wake_rms == 50.0
    assert processors[stt_index - 1]._min_wake_peak == 150
    assert processors[stt_index - 1]._rearm_delay_s == 6.0
    assert processors[stt_index + 1]._single_command is False
    assert isinstance(seen_agent_kwargs["response_coordinator"], BotResponseCoordinator)
    assert seen_detector_kwargs == {
        "model_path": tmp_path / "mave.onnx",
        "threshold": 0.5,
        "vad_threshold": 0.3,
    }
    assert callable(seen_agent_kwargs["on_turn_started"])
    assert "on_turn_finished" not in seen_agent_kwargs
    assert any(
        "Wake config" in message
        and "threshold=0.5" in message
        and "vad_threshold=0.3" in message
        and "min_wake_rms=50.0" in message
        and "min_wake_peak=150" in message
        and "required_hits=2" in message
        and "rearm_delay_s=6.0" in message
        for message in wake_config_logs
    )


def test_wake_disabled_does_not_wire_wake_tone(monkeypatch, tmp_path: Path):
    _patch_pipeline_dependencies(monkeypatch)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False, wake_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )
    processors = cast(FakePipeline, built.pipeline).processors

    assert not any(isinstance(processor, WakeToneProcessor) for processor in processors)
    assert not any(isinstance(processor, BotSpeechOutputCoordinator) for processor in processors)


def test_create_voice_modulation_processor_returns_none_for_missing_disabled_or_unknown_settings() -> None:
    assert _create_voice_modulation_processor(None) is None
    assert _create_voice_modulation_processor(object()) is None
    assert _create_voice_modulation_processor(VoiceModulationSettings(enabled=False)) is None


def test_create_voice_modulation_processor_returns_processor_for_enabled_settings() -> None:
    settings = VoiceModulationSettings(enabled=True, gain_db=3.0)

    processor = _create_voice_modulation_processor(settings)

    assert isinstance(processor, VoiceModulationProcessor)


def test_enabled_voice_modulation_is_wired_between_tts_and_transport_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_pipeline_dependencies(monkeypatch)
    tts = FrameProcessor()
    transport_output = FrameProcessor()
    monkeypatch.setattr("pipeline_builder.create_tts_service", lambda config, **kwargs: tts)

    class VoiceModulationTransport(FakeTransport):
        def output(self):
            return transport_output

    built = build_pipeline(
        _config(
            tmp_path,
            metrics_enabled=False,
            voice_modulation=VoiceModulationSettings(enabled=True, gain_db=3.0),
        ),
        cast(BaseTransport, VoiceModulationTransport()),
    )
    processors = cast(FakePipeline, built.pipeline).processors
    tts_index = processors.index(tts)

    assert isinstance(processors[tts_index + 1], VoiceModulationProcessor)
    assert processors[tts_index + 2] is transport_output
