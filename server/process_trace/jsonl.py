from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)


class JsonlTraceWriter:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._disabled = False
        self._warned = False

    def write(self, record: dict[str, Any]) -> None:
        if self._disabled:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(record, ensure_ascii=False, sort_keys=True)
            with self._path.open("a", encoding="utf-8") as file:
                file.write(f"{line}\n")
        except OSError as exc:
            self._disabled = True
            if not self._warned:
                self._warned = True
                LOGGER.warning("Disabling process trace JSONL writer after write failure: %s", exc)
