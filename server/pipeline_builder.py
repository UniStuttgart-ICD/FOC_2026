from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.observers.base_observer import BaseObserver
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport

from agent_control.factory import create_agent_processor
from config import RuntimeConfig
from metrics import VoiceMetricsObserver, VoiceMetricsRecorder
from process_trace import (
    JsonlTraceWriter,
    NoopProcessTracer,
    ProcessTracer,
    TraceContext,
    TraceOptions,
)
from voice_modulation.processor import VoiceModulationProcessor
from voice_modulation.settings import VoiceModulationSettings
from voice_runtime.assembly import VoiceRuntimeParts, ordered_voice_runtime_processors
from voice_runtime.providers import create_stt_service, create_tts_service
from voice_runtime.wake_command import build_mave_voice_command_processors
from wake.openwakeword_detector import OpenWakeWordDetector


@dataclass
class BuiltPipeline:
    pipeline: Pipeline
    task: PipelineTask
    agent_processor: FrameProcessor
    user_aggregator: FrameProcessor
    assistant_aggregator: FrameProcessor
    metrics: VoiceMetricsRecorder | None
    process_tracer: ProcessTracer | NoopProcessTracer


def build_pipeline(config: RuntimeConfig, transport: BaseTransport) -> BuiltPipeline:
    stt = create_stt_service(config.stt)
    tts = create_tts_service(config.tts)
    voice_modulation = _create_voice_modulation_processor(config.voice_modulation)
    session_id = _new_session_id()
    session_started_at = _utc_now()
    process_tracer = _build_process_tracer(config, session_id, session_started_at)
    session_context = process_tracer.start_session(
        config.profile_name,
        config.category,
        session_id=session_id,
    )

    voice_command_audio = None
    voice_command_transcript = None
    if config.wake.provider == "openwakeword":
        assert config.wake.model_path is not None
        detector = OpenWakeWordDetector(
            config.wake.model_path,
            threshold=config.wake.threshold,
            vad_threshold=config.wake.vad_threshold,
        )
        voice_command_processors = build_mave_voice_command_processors(
            detector=detector,
            pre_buffer_s=config.wake.pre_buffer_s,
            rearm_delay_s=config.wake.rearm_delay_s,
            single_command=config.wake.single_command,
            candidate_log_threshold=config.wake.candidate_log_threshold,
            required_hits=config.wake.required_hits,
            min_wake_rms=config.wake.min_wake_rms,
            min_wake_peak=config.wake.min_wake_peak,
            wake_threshold=config.wake.threshold,
        )
        voice_command_audio = voice_command_processors.audio_gate
        voice_command_transcript = voice_command_processors.transcript_adapter
        logger.info(
            "Wake config detector={} threshold={} vad_threshold={} candidate_log_threshold={} "
            "required_hits={} min_wake_rms={} min_wake_peak={} rearm_delay_s={}",
            config.wake.provider,
            config.wake.threshold,
            config.wake.vad_threshold,
            config.wake.candidate_log_threshold,
            config.wake.required_hits,
            config.wake.min_wake_rms,
            config.wake.min_wake_peak,
            config.wake.rearm_delay_s,
        )

    agent_kwargs = {}
    if voice_command_audio is not None:
        agent_kwargs = {
            "on_turn_started": voice_command_audio.suppress,
            "on_turn_finished": voice_command_audio.unsuppress,
        }
    agent_processor = create_agent_processor(
        config.agent,
        mcp_server_url=config.mcp_robot_url,
        tracer=process_tracer,
        **agent_kwargs,
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    processors = cast(
        list[FrameProcessor],
        ordered_voice_runtime_processors(
            VoiceRuntimeParts(
                transport_input=transport.input(),
                voice_command_audio=voice_command_audio,
                stt=stt,
                voice_command_transcript=voice_command_transcript,
                user_aggregator=user_aggregator,
                agent_turn=agent_processor,
                tts=tts,
                voice_modulation=voice_modulation,
                transport_output=transport.output(),
                assistant_aggregator=assistant_aggregator,
            )
        ),
    )

    pipeline = Pipeline(processors)
    metrics = None
    observers: list[BaseObserver] = []
    if config.metrics.enabled:
        metrics = VoiceMetricsRecorder(
            profile=config.profile_name,
            category=config.category,
            path=session_log_path(config.metrics.path, session_started_at, session_id),
            include_text=config.metrics.include_text,
        )
        observers.append(VoiceMetricsObserver(metrics))
    if isinstance(process_tracer, ProcessTracer):
        observers.append(_create_process_trace_observer(process_tracer, session_context))

    task = PipelineTask(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        observers=observers,
    )
    return BuiltPipeline(
        pipeline=pipeline,
        task=task,
        agent_processor=agent_processor,
        user_aggregator=user_aggregator,
        assistant_aggregator=assistant_aggregator,
        metrics=metrics,
        process_tracer=process_tracer,
    )


def session_log_path(base_path: Path, started_at: datetime, session_id: str) -> Path:
    timestamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = base_path.suffix or ".jsonl"
    stem = base_path.stem or "session"
    session_token = session_id[:8]
    return base_path.parent / stem / f"{stem}-{timestamp}-{session_token}{suffix}"


def _create_voice_modulation_processor(
    settings: object | None,
) -> VoiceModulationProcessor | None:
    if not isinstance(settings, VoiceModulationSettings) or not settings.enabled:
        return None
    return VoiceModulationProcessor(settings=settings)


def _new_session_id() -> str:
    return uuid.uuid4().hex


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _build_process_tracer(
    config: RuntimeConfig,
    session_id: str,
    session_started_at: datetime,
) -> ProcessTracer | NoopProcessTracer:
    options = TraceOptions(
        include_text=config.process_trace.include_text,
        include_tool_payloads=config.process_trace.include_tool_payloads,
    )
    if not config.process_trace.enabled:
        return NoopProcessTracer(options)
    return ProcessTracer(
        JsonlTraceWriter(session_log_path(config.process_trace.path, session_started_at, session_id)),
        options,
    )


def _create_process_trace_observer(
    process_tracer: ProcessTracer,
    session_context: TraceContext,
) -> BaseObserver:
    from process_trace.pipecat_observer import ProcessTraceObserver

    return ProcessTraceObserver(process_tracer, session_context=session_context)
