from pathlib import Path
from typing import Any, cast
from unittest.mock import Mock

from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.base_transport import BaseTransport

from config import (
    AgentConfig,
    EmergencyStopConfig,
    MetricsConfig,
    RuntimeConfig,
    STTConfig,
    TTSConfig,
    WakeConfig,
)
from metrics import VoiceMetricsObserver
from pipeline_builder import build_pipeline
from voice_runtime.wake_command import MaveVoiceCommandAudioGate, MaveVoiceCommandTranscriptAdapter


class FakePipeline:
    def __init__(self, processors):
        self.processors = processors


class FakePipelineTask:
    def __init__(self, pipeline, *, params, observers):
        self.pipeline = pipeline
        self.params = params
        self.observers = observers


class FakeTransport:
    def input(self):
        return FrameProcessor()

    def output(self):
        return FrameProcessor()


def _config(tmp_path: Path, *, metrics_enabled: bool, wake_enabled: bool = False) -> RuntimeConfig:
    return RuntimeConfig(
        profile_name="no_wake_debug",
        category="local_debug",
        wake=WakeConfig(
            provider="openwakeword" if wake_enabled else "none",
            model_path=tmp_path / "mave.onnx" if wake_enabled else None,
            vad_threshold=0.3,
            required_hits=2,
            pre_buffer_s=2.0,
            single_command=False,
            candidate_log_threshold=0.4,
        ),
        emergency_stop=EmergencyStopConfig(enabled=False),
        stt=STTConfig(provider="whisper", model="base", device="cpu"),
        tts=TTSConfig(provider="kokoro", voice="af_heart"),
        agent=AgentConfig(provider="openai_codex_oauth", model="gpt-5.4-mini"),
        mcp_robot_url="http://127.0.0.1:8765/mcp",
        metrics=MetricsConfig(
            enabled=metrics_enabled,
            path=tmp_path / "metrics.jsonl",
            include_text=True,
        ),
        server_dir=tmp_path,
    )


def _patch_pipeline_dependencies(monkeypatch):
    monkeypatch.setattr("pipeline_builder.create_stt_service", lambda config: FrameProcessor())
    monkeypatch.setattr("pipeline_builder.create_tts_service", lambda config: FrameProcessor())
    monkeypatch.setattr(
        "pipeline_builder.create_agent_processor",
        lambda config, *, mcp_server_url: FrameProcessor(),
    )
    monkeypatch.setattr(
        "pipeline_builder.LLMContextAggregatorPair",
        lambda context, user_params: (FrameProcessor(), FrameProcessor()),
    )
    monkeypatch.setattr("pipeline_builder.Pipeline", FakePipeline)
    monkeypatch.setattr("pipeline_builder.PipelineTask", FakePipelineTask)


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


def test_wake_enabled_uses_two_voice_command_adapters_around_stt(monkeypatch, tmp_path: Path):
    stt = FrameProcessor()
    seen_detector_kwargs = {}

    _patch_pipeline_dependencies(monkeypatch)
    monkeypatch.setattr("pipeline_builder.create_stt_service", lambda config: stt)

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
        cast(BaseTransport, FakeTransport()),
    )
    processors = cast(FakePipeline, built.pipeline).processors
    stt_index = processors.index(stt)

    assert isinstance(processors[stt_index - 1], MaveVoiceCommandAudioGate)
    assert isinstance(processors[stt_index + 1], MaveVoiceCommandTranscriptAdapter)
    assert processors[stt_index - 1]._wake_threshold == 0.5
    assert processors[stt_index - 1]._pre_buffer_s == 2.0
    assert processors[stt_index - 1]._candidate_log_threshold == 0.4
    assert processors[stt_index - 1]._required_hits == 2
    assert processors[stt_index + 1]._single_command is False
    assert seen_detector_kwargs == {
        "model_path": tmp_path / "mave.onnx",
        "threshold": 0.5,
        "vad_threshold": 0.3,
    }
