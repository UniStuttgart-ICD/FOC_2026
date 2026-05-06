from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pipecat.frames.frames import InputAudioRawFrame, TranscriptionFrame
from pipecat.processors.frame_processor import FrameDirection

from wake.openwakeword_detector import OpenWakeWordDetector, OpenWakeWordResourceError
from wake.wake_gate import MaveWakeWordGate


class CapturingGate(MaveWakeWordGate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pushed = []

    async def push_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
        self.pushed.append((frame, direction))


def _frame(value: int, samples: int = 1600):
    audio = np.full(samples, value, dtype=np.int16).tobytes()
    return InputAudioRawFrame(audio=audio, sample_rate=16000, num_channels=1)


def test_openwakeword_bootstraps_required_resources_before_model(monkeypatch, tmp_path):
    model_path = tmp_path / "mave.onnx"
    model_path.write_bytes(b"custom wake model")
    resource_dir = tmp_path / "resources"
    resources = {
        "melspectrogram.onnx": resource_dir / "melspectrogram.onnx",
        "embedding_model.onnx": resource_dir / "embedding_model.onnx",
        "silero_vad.onnx": resource_dir / "silero_vad.onnx",
    }
    calls: list[tuple[str, str]] = []
    model_constructed = False

    def fake_download_file(url: str, target_directory: str):
        assert not model_constructed
        calls.append((Path(url).name, target_directory))
        resources[Path(url).name].parent.mkdir(parents=True, exist_ok=True)
        resources[Path(url).name].write_bytes(b"resource")

    def fake_model(*args, **kwargs):
        nonlocal model_constructed
        model_constructed = True
        return Mock()

    monkeypatch.setattr("wake.openwakeword_detector._openwakeword_resource_dir", lambda: resource_dir)
    monkeypatch.setattr("wake.openwakeword_detector.download_file", fake_download_file)
    monkeypatch.setattr("wake.openwakeword_detector.Model", fake_model)

    OpenWakeWordDetector(model_path)

    assert [call[0] for call in calls] == [
        "melspectrogram.onnx",
        "embedding_model.onnx",
        "silero_vad.onnx",
    ]
    assert model_constructed is True


def test_openwakeword_resource_download_failure_is_clear(monkeypatch, tmp_path):
    model_path = tmp_path / "mave.onnx"
    model_path.write_bytes(b"custom wake model")
    resource_dir = tmp_path / "resources"

    def fake_download_file(url: str, target_directory: str):
        raise OSError("network unavailable")

    model = Mock()
    monkeypatch.setattr("wake.openwakeword_detector._openwakeword_resource_dir", lambda: resource_dir)
    monkeypatch.setattr("wake.openwakeword_detector.download_file", fake_download_file)
    monkeypatch.setattr("wake.openwakeword_detector.Model", model)

    with pytest.raises(OpenWakeWordResourceError, match="melspectrogram.onnx"):
        OpenWakeWordDetector(model_path)

    model.assert_not_called()


def test_openwakeword_predict_normalizes_scores(monkeypatch, tmp_path):
    model_path = tmp_path / "mave.onnx"
    model_path.write_bytes(b"custom wake model")
    model = Mock()
    model.predict.return_value = ({"mave": np.float32(0.75)}, {"unused": {}})
    monkeypatch.setattr("wake.openwakeword_detector._ensure_openwakeword_resources", Mock())
    monkeypatch.setattr("wake.openwakeword_detector.Model", Mock(return_value=model))

    detector = OpenWakeWordDetector(model_path)

    assert detector.predict(np.zeros(1600, dtype=np.int16)) == {"mave": 0.75}


def test_openwakeword_passes_vad_threshold_to_model(monkeypatch, tmp_path):
    model_path = tmp_path / "mave.onnx"
    model_path.write_bytes(b"custom wake model")
    model_factory = Mock(return_value=Mock())
    monkeypatch.setattr("wake.openwakeword_detector._ensure_openwakeword_resources", Mock())
    monkeypatch.setattr("wake.openwakeword_detector.Model", model_factory)

    OpenWakeWordDetector(model_path, vad_threshold=0.3)

    assert model_factory.call_args.kwargs["vad_threshold"] == 0.3


def test_openwakeword_exposes_last_vad_score(monkeypatch, tmp_path):
    model_path = tmp_path / "mave.onnx"
    model_path.write_bytes(b"custom wake model")
    model = Mock()
    model.vad_threshold = 0.3
    model.vad = SimpleNamespace(prediction_buffer=[0.1, 0.2, 0.7, 0.4, 0.9, 0.2, 0.1])
    monkeypatch.setattr("wake.openwakeword_detector._ensure_openwakeword_resources", Mock())
    monkeypatch.setattr("wake.openwakeword_detector.Model", Mock(return_value=model))

    detector = OpenWakeWordDetector(model_path, vad_threshold=0.3)

    assert detector.vad_enabled is True
    assert detector.last_vad_score() == 0.7


@pytest.mark.asyncio
async def test_blocks_audio_until_wake_detected():
    detector = Mock()
    detector.detected.return_value = (False, None, 0.0)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)

    assert gate.pushed == []


@pytest.mark.asyncio
async def test_replays_prebuffer_on_wake():
    detector = Mock()
    detector.detected.side_effect = [
        (False, None, 0.0),
        (True, "mave", 0.9),
    ]
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(1), FrameDirection.DOWNSTREAM)
    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)

    pushed_audio = [item[0] for item in gate.pushed if isinstance(item[0], InputAudioRawFrame)]
    assert len(pushed_audio) == 2
    assert np.frombuffer(pushed_audio[0].audio, dtype=np.int16)[0] == 1
    assert np.frombuffer(pushed_audio[1].audio, dtype=np.int16)[0] == 2


@pytest.mark.asyncio
async def test_transcriptions_pass_through_gate_without_resetting():
    detector = Mock()
    detector.detected.return_value = (True, "mave", 0.9)
    gate = CapturingGate(detector=detector, pre_buffer_s=1.5)

    await gate.process_frame(_frame(2), FrameDirection.DOWNSTREAM)
    await gate.process_frame(
        TranscriptionFrame(text="Mave, move up", user_id="u", timestamp="t", finalized=True),
        FrameDirection.DOWNSTREAM,
    )

    transcription = [item[0] for item in gate.pushed if isinstance(item[0], TranscriptionFrame)][0]
    assert transcription.text == "Mave, move up"
    assert gate.is_awake is True
