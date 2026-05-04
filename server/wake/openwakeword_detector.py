from __future__ import annotations

from pathlib import Path

import numpy as np
from openwakeword.model import Model


class OpenWakeWordDetector:
    """Small wrapper around OpenWakeWord for one or more ONNX wake models."""

    def __init__(self, model_path: Path, *, threshold: float = 0.5):
        if not model_path.exists():
            raise FileNotFoundError(f"Wake model not found: {model_path}")
        self._threshold = threshold
        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")

    def predict(self, pcm16: np.ndarray) -> dict[str, float]:
        if pcm16.dtype != np.int16:
            raise TypeError("OpenWakeWordDetector expects int16 PCM")
        return self._model.predict(pcm16)

    def detected(self, pcm16: np.ndarray) -> tuple[bool, str | None, float]:
        scores = self.predict(pcm16)
        if not scores:
            return False, None, 0.0
        name, score = max(scores.items(), key=lambda item: item[1])
        return score >= self._threshold, name, float(score)
