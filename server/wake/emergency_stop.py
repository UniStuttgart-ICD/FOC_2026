from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from config import EmergencyStopConfig


@dataclass(frozen=True)
class EmergencyStopDetector:
    """Configuration holder for future local emergency stop detection."""

    model_path: Path
    threshold: float
    command_text: str = "stop"


def build_emergency_stop_detector(config: EmergencyStopConfig) -> EmergencyStopDetector | None:
    if not config.enabled:
        return None
    if config.model_path is None:
        raise ValueError("Emergency stop model is required when emergency stop is enabled")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Emergency stop model not found: {config.model_path}")
    return EmergencyStopDetector(model_path=config.model_path, threshold=config.threshold)
