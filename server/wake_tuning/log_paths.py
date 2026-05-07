from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WakeTuningLogPaths:
    stdout: Path
    stderr: Path


def default_log_dir(server_dir: Path | None = None) -> Path:
    root = server_dir or Path(__file__).resolve().parents[1]
    return root / "logs" / "wake_tuning"


def log_paths(label: str = "server", *, server_dir: Path | None = None) -> WakeTuningLogPaths:
    safe_label = _safe_label(label)
    log_dir = default_log_dir(server_dir)
    basename = f"wake_tuning_{safe_label}"
    return WakeTuningLogPaths(
        stdout=log_dir / f"{basename}.out.log",
        stderr=log_dir / f"{basename}.err.log",
    )


def _safe_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", label.strip()).strip("_").lower()
    return normalized or "server"
