from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from wake_tuning.settings import WakeTuningSettings


@dataclass(frozen=True)
class DetectionResult:
    detected: bool
    model_name: str | None
    score: float
    rms: float
    peak: int
    hits: int
    required_hits: int
    decision: str
    threshold_hit: bool
    level_hit: bool


class WakeDecisionTracker:
    def __init__(self, settings: WakeTuningSettings):
        self._settings = settings
        self._hits = 0

    def evaluate(self, scores: dict[str, float], pcm16: NDArray[np.int16]) -> DetectionResult:
        rms, peak = audio_levels(pcm16)
        if scores:
            model_name, score = max(scores.items(), key=lambda item: item[1])
        else:
            model_name, score = None, 0.0

        threshold_hit = score >= self._settings.threshold
        level_hit = rms >= self._settings.min_wake_rms and peak >= self._settings.min_wake_peak
        if threshold_hit and level_hit:
            self._hits += 1
        else:
            self._hits = 0

        detected = self._hits >= self._settings.required_hits
        if detected:
            decision = "triggered"
        elif not scores:
            decision = "no_score"
        elif not threshold_hit:
            decision = "below_threshold"
        elif not level_hit:
            decision = "audio_level"
        else:
            decision = "waiting_for_hits"

        if detected:
            self._hits = 0

        return DetectionResult(
            detected=detected,
            model_name=model_name,
            score=float(score),
            rms=rms,
            peak=peak,
            hits=self._hits if not detected else self._settings.required_hits,
            required_hits=self._settings.required_hits,
            decision=decision,
            threshold_hit=threshold_hit,
            level_hit=level_hit,
        )


def audio_levels(pcm16: NDArray[np.int16]) -> tuple[float, int]:
    if pcm16.size == 0:
        return 0.0, 0
    samples = pcm16.astype(np.float64)
    rms = float(np.sqrt(np.mean(samples * samples)))
    peak = int(np.max(np.abs(pcm16.astype(np.int32))))
    return rms, peak
