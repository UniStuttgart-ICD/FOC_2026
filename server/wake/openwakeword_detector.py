from __future__ import annotations

from pathlib import Path

import numpy as np
import openwakeword
from openwakeword.model import Model
from openwakeword.utils import download_file


class OpenWakeWordResourceError(RuntimeError):
    """Raised when required openWakeWord runtime resources cannot be prepared."""


_RESOURCE_URLS = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
    "melspectrogram.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.tflite",
    "embedding_model.tflite": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.tflite",
    "silero_vad.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/silero_vad.onnx",
}


def _openwakeword_resource_dir() -> Path:
    return Path(openwakeword.__file__).resolve().parent / "resources" / "models"


def _required_resource_paths(inference_framework: str) -> list[Path]:
    resource_dir = _openwakeword_resource_dir()
    if inference_framework == "onnx":
        feature_names = ["melspectrogram.onnx", "embedding_model.onnx"]
    elif inference_framework == "tflite":
        feature_names = ["melspectrogram.tflite", "embedding_model.tflite"]
    else:
        raise ValueError(f"Unsupported openWakeWord inference framework: {inference_framework}")

    return [*(resource_dir / name for name in feature_names), resource_dir / "silero_vad.onnx"]


def _ensure_openwakeword_resources(inference_framework: str = "onnx") -> None:
    required_paths = _required_resource_paths(inference_framework)
    missing_paths = [path for path in required_paths if not path.exists()]
    if not missing_paths:
        return

    target_directory = missing_paths[0].parent
    target_directory.mkdir(parents=True, exist_ok=True)

    for path in missing_paths:
        url = _RESOURCE_URLS.get(path.name)
        if url is None:
            raise OpenWakeWordResourceError(
                f"No openWakeWord resource download URL configured for {path.name}"
            )
        try:
            download_file(url, str(target_directory))
        except Exception as exc:
            raise OpenWakeWordResourceError(
                f"Failed to download required openWakeWord resource {path.name}: {exc}"
            ) from exc

    still_missing = [path.name for path in required_paths if not path.exists()]
    if still_missing:
        missing = ", ".join(still_missing)
        raise OpenWakeWordResourceError(
            f"Required openWakeWord resources are missing after download: {missing}"
        )


class OpenWakeWordDetector:
    """Small wrapper around OpenWakeWord for one or more ONNX wake models."""

    def __init__(self, model_path: Path, *, threshold: float = 0.5, vad_threshold: float = 0.0):
        if not model_path.exists():
            raise FileNotFoundError(f"Wake model not found: {model_path}")
        _ensure_openwakeword_resources("onnx")
        self._threshold = threshold
        self._model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            vad_threshold=vad_threshold,
        )

    @property
    def vad_enabled(self) -> bool:
        return self._model.vad_threshold > 0

    def last_vad_score(self) -> float | None:
        if not self.vad_enabled:
            return None
        vad = getattr(self._model, "vad", None)
        prediction_buffer = getattr(vad, "prediction_buffer", None)
        if prediction_buffer is None:
            return 0.0
        vad_frames = list(prediction_buffer)[-7:-4]
        if not vad_frames:
            return 0.0
        return float(np.max(vad_frames))

    def predict(self, pcm16: np.ndarray) -> dict[str, float]:
        if pcm16.dtype != np.int16:
            raise TypeError("OpenWakeWordDetector expects int16 PCM")

        raw_result: object = self._model.predict(pcm16)
        if isinstance(raw_result, tuple):
            raw_result = raw_result[0]
        if not isinstance(raw_result, dict):
            raise TypeError("OpenWakeWord predict() returned an unsupported result type")

        scores: dict[str, float] = {}
        for name, score in raw_result.items():
            scores[str(name)] = float(score)
        return scores

    def detected(self, pcm16: np.ndarray) -> tuple[bool, str | None, float]:
        scores = self.predict(pcm16)
        if not scores:
            return False, None, 0.0
        name, score = max(scores.items(), key=lambda item: item[1])
        return score >= self._threshold, name, float(score)
