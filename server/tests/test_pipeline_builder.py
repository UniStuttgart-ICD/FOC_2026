from pathlib import Path
from typing import Any, cast

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


def _config(tmp_path: Path, *, metrics_enabled: bool) -> RuntimeConfig:
    return RuntimeConfig(
        profile_name="no_wake_debug",
        category="local_debug",
        wake=WakeConfig(provider="none", model_path=None),
        emergency_stop=EmergencyStopConfig(enabled=False),
        stt=STTConfig(provider="whisper", model="base", device="cpu"),
        tts=TTSConfig(provider="kokoro", voice="af_heart"),
        agent=AgentConfig(provider="claude", model="claude-haiku-4-5-20251001"),
        mcp_robot_url="http://127.0.0.1:8765/mcp",
        metrics=MetricsConfig(
            enabled=metrics_enabled,
            path=tmp_path / "metrics.jsonl",
            include_text=True,
        ),
        server_dir=tmp_path,
    )


def test_metrics_observer_is_wired_when_metrics_enabled(monkeypatch, tmp_path: Path):
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
    monkeypatch.setattr("pipeline_builder.PipelineTask", FakePipelineTask)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=True),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert built.metrics is not None
    assert any(isinstance(observer, VoiceMetricsObserver) for observer in task.observers)


def test_metrics_observer_is_not_wired_when_metrics_disabled(monkeypatch, tmp_path: Path):
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
    monkeypatch.setattr("pipeline_builder.PipelineTask", FakePipelineTask)

    built = build_pipeline(
        _config(tmp_path, metrics_enabled=False),
        cast(BaseTransport, FakeTransport()),
    )
    task = cast(Any, built.task)

    assert built.metrics is None
    assert task.observers == []
