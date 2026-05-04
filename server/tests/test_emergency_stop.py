from pathlib import Path

import pytest

from config import EmergencyStopConfig
from wake.emergency_stop import EmergencyStopDetector, build_emergency_stop_detector


def test_disabled_emergency_stop_returns_none():
    detector = build_emergency_stop_detector(EmergencyStopConfig(enabled=False))

    assert detector is None


def test_enabled_without_model_fails():
    with pytest.raises(ValueError, match="Emergency stop model is required"):
        build_emergency_stop_detector(EmergencyStopConfig(enabled=True, provider="openwakeword"))


def test_detector_interface_reports_no_detection_by_default(tmp_path: Path):
    model = tmp_path / "stop.onnx"
    model.write_bytes(b"fake")
    detector = EmergencyStopDetector(model_path=model, threshold=0.5)

    assert detector.command_text == "stop"
