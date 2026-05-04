from __future__ import annotations

from dataclasses import dataclass

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

from agent_processor_factory import create_agent_processor
from config import RuntimeConfig
from metrics import VoiceMetricsObserver, VoiceMetricsRecorder
from providers import create_stt_service, create_tts_service
from wake.openwakeword_detector import OpenWakeWordDetector
from wake.transcript_cleanup import WakePhraseTranscriptCleaner
from wake.wake_gate import MaveWakeWordGate


@dataclass
class BuiltPipeline:
    pipeline: Pipeline
    task: PipelineTask
    agent_processor: FrameProcessor
    user_aggregator: FrameProcessor
    assistant_aggregator: FrameProcessor
    metrics: VoiceMetricsRecorder | None


def build_pipeline(config: RuntimeConfig, transport: BaseTransport) -> BuiltPipeline:
    stt = create_stt_service(config.stt)
    tts = create_tts_service(config.tts)
    agent_processor = create_agent_processor(config.agent, mcp_server_url=config.mcp_robot_url)

    wake_gate = None
    transcript_cleaner = None
    if config.wake.provider == "openwakeword":
        assert config.wake.model_path is not None
        detector = OpenWakeWordDetector(config.wake.model_path, threshold=config.wake.threshold)
        wake_gate = MaveWakeWordGate(detector=detector, pre_buffer_s=config.wake.pre_buffer_s)
        transcript_cleaner = WakePhraseTranscriptCleaner(on_finalized_transcription=wake_gate.reset)

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    processors: list[FrameProcessor] = [transport.input()]
    if wake_gate is not None:
        processors.append(wake_gate)
    processors.append(stt)
    if transcript_cleaner is not None:
        processors.append(transcript_cleaner)
    processors.extend(
        [
            user_aggregator,
            agent_processor,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )

    pipeline = Pipeline(processors)
    metrics = None
    observers: list[BaseObserver] = []
    if config.metrics.enabled:
        metrics = VoiceMetricsRecorder(
            profile=config.profile_name,
            category=config.category,
            path=config.metrics.path,
            include_text=config.metrics.include_text,
        )
        observers.append(VoiceMetricsObserver(metrics))

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
    )
