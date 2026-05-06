from __future__ import annotations

import time


def monotonic_s() -> float:
    return time.monotonic()


def elapsed_ms_since(start_s: float, *, now: float | None = None) -> float:
    current_s = monotonic_s() if now is None else now
    return round((current_s - start_s) * 1000.0, 2)
