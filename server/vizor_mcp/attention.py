from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GazeSample:
    target: str
    at_s: float


@dataclass
class _TargetStats:
    target: str
    dwell_s: float = 0.0
    weighted_score: float = 0.0
    last_seen_s: float = 0.0


class GazeAttentionTracker:
    def __init__(
        self,
        *,
        window_s: float = 8.0,
        history_s: float = 30.0,
        flicker_s: float = 0.25,
        stable_s: float = 0.5,
        stale_after_s: float = 2.0,
        recency_half_life_s: float = 4.0,
    ) -> None:
        self.window_s = float(window_s)
        self.history_s = float(history_s)
        self.flicker_s = float(flicker_s)
        self.stable_s = float(stable_s)
        self.stale_after_s = float(stale_after_s)
        self.recency_half_life_s = float(recency_half_life_s)
        self._samples: deque[GazeSample] = deque()

    def record(self, target: str | None, *, at_s: float) -> None:
        clean_target = (target or "").strip()
        if not clean_target:
            return

        sample = GazeSample(target=clean_target, at_s=float(at_s))
        if self._samples and sample.at_s < self._samples[-1].at_s:
            self._samples.append(sample)
            self._samples = deque(sorted(self._samples, key=lambda item: item.at_s))
        else:
            self._samples.append(sample)
        self._prune(sample.at_s)

    def summarize(
        self,
        *,
        now_s: float,
        window_s: float | None = None,
        stale_after_s: float | None = None,
    ) -> dict[str, Any]:
        now = float(now_s)
        window = float(self.window_s if window_s is None else window_s)
        stale_after = float(self.stale_after_s if stale_after_s is None else stale_after_s)
        self._prune(now)

        latest = self._samples[-1] if self._samples else None
        fresh = latest is not None and max(0.0, now - latest.at_s) <= stale_after
        runs = self._runs(now_s=now, window_s=window)
        stats = self._stats_from_runs(runs, now)
        ranked = self._ranked_targets(stats, now, fresh)
        if not ranked and latest is not None:
            ranked = [
                {
                    "target": latest.target,
                    "score": 1.0,
                    "dwell_s": 0.0,
                    "last_seen_age_s": max(0.0, now - latest.at_s),
                    "confidence": "low",
                }
            ]

        return {
            "available": latest is not None,
            "window_s": window,
            "fresh": fresh,
            "current_target": latest.target if latest is not None else None,
            "dominant_target": ranked[0]["target"] if ranked else None,
            "last_stable_target": self._last_stable_target(runs),
            "ranked_targets": ranked,
        }

    def _prune(self, now_s: float) -> None:
        cutoff = float(now_s) - self.history_s
        while len(self._samples) > 1 and self._samples[1].at_s < cutoff:
            self._samples.popleft()

    def _runs(self, *, now_s: float, window_s: float) -> list[tuple[str, float, float, float]]:
        if not self._samples:
            return []

        start_s = now_s - window_s
        samples = list(self._samples)
        first_index = 0
        for index, sample in enumerate(samples):
            if sample.at_s >= start_s:
                first_index = max(0, index - 1)
                break
        else:
            first_index = len(samples) - 1

        segments: list[tuple[str, float, float, float]] = []
        relevant = samples[first_index:]
        for index, sample in enumerate(relevant):
            segment_start = max(sample.at_s, start_s)
            next_at = relevant[index + 1].at_s if index + 1 < len(relevant) else now_s
            segment_end = min(next_at, now_s)
            if segment_end <= segment_start:
                continue
            self._append_segment(
                segments,
                sample.target,
                segment_start,
                segment_end,
                sample.at_s,
            )

        return [
            (target, started_at, ended_at, last_observed_at)
            for target, started_at, ended_at, last_observed_at in segments
            if ended_at - started_at >= self.flicker_s
        ]

    def _append_segment(
        self,
        segments: list[tuple[str, float, float, float]],
        target: str,
        start_s: float,
        end_s: float,
        observed_at_s: float,
    ) -> None:
        if segments and segments[-1][0] == target:
            previous = segments[-1]
            segments[-1] = (target, previous[1], end_s, max(previous[3], observed_at_s))
            return
        segments.append((target, start_s, end_s, observed_at_s))

    def _stats_from_runs(
        self,
        runs: list[tuple[str, float, float, float]],
        now_s: float,
    ) -> dict[str, _TargetStats]:
        stats: dict[str, _TargetStats] = {}
        for target, started_at, ended_at, last_observed_at in runs:
            dwell_s = ended_at - started_at
            age_s = max(0.0, now_s - last_observed_at)
            recency = 0.5 ** (age_s / self.recency_half_life_s)
            item = stats.setdefault(target, _TargetStats(target=target))
            item.dwell_s += dwell_s
            item.weighted_score += dwell_s * recency
            item.last_seen_s = max(item.last_seen_s, last_observed_at)
        return stats

    def _ranked_targets(
        self,
        stats: dict[str, _TargetStats],
        now_s: float,
        fresh: bool,
    ) -> list[dict[str, Any]]:
        total_score = sum(item.weighted_score for item in stats.values())
        if total_score <= 0.0:
            return []

        ranked_stats = sorted(
            stats.values(),
            key=lambda item: (item.weighted_score, item.dwell_s, item.last_seen_s),
            reverse=True,
        )
        return [
            {
                "target": item.target,
                "score": item.weighted_score / total_score,
                "dwell_s": item.dwell_s,
                "last_seen_age_s": max(0.0, now_s - item.last_seen_s),
                "confidence": self._confidence(item.weighted_score / total_score, fresh),
            }
            for item in ranked_stats
        ]

    def _confidence(self, score: float, fresh: bool) -> str:
        if not fresh:
            return "low"
        if math.isclose(score, 1.0) or score >= 0.65:
            return "high"
        if score >= 0.3:
            return "medium"
        return "low"

    def _last_stable_target(self, runs: list[tuple[str, float, float, float]]) -> str | None:
        for target, started_at, ended_at, _last_observed_at in reversed(runs):
            if ended_at - started_at >= self.stable_s:
                return target
        return None
