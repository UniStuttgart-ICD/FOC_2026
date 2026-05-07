from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from model_eval.results import AttemptResult, CandidateSummary


class EvidenceWriter:
    def __init__(self, root: Path) -> None:
        self.root = root

    def write(
        self,
        *,
        attempts: tuple[AttemptResult, ...],
        summaries: tuple[CandidateSummary, ...],
        metadata: dict[str, Any],
    ) -> Path:
        created_at = _utc_timestamp()
        evidence_dir = self.root / _directory_timestamp(created_at)
        evidence_dir.mkdir(parents=True, exist_ok=False)

        _write_json(
            evidence_dir / "metadata.json",
            {
                **metadata,
                "created_at": created_at,
            },
        )
        _write_json(
            evidence_dir / "attempts.json",
            [asdict(attempt) for attempt in attempts],
        )
        _write_json(
            evidence_dir / "summary.json",
            [asdict(summary) for summary in summaries],
        )
        return evidence_dir


def _utc_timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _directory_timestamp(created_at: str) -> str:
    return created_at.replace(":", "").replace("-", "")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
