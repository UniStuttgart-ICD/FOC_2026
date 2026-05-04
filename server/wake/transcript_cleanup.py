from __future__ import annotations

import re

_WAKE_PATTERN = re.compile(r"^\s*(?:hey\s+)?mave[\s,;:!?.-]*", re.IGNORECASE)


def strip_wake_phrase(text: str) -> str:
    """Remove a leading Mave wake phrase from a transcript."""
    cleaned = _WAKE_PATTERN.sub("", text, count=1).strip()
    return cleaned or text.strip()
