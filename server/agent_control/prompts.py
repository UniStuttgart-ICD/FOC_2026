"""System prompt for the simulation-only voice robot agent."""

from pathlib import Path

_PROMPT_PARTS_DIR = Path(__file__).with_name("prompt_parts")
_PROMPT_PARTS = (
    "mave_embodiment.md",
    "robot_contract.md",
    "examples.md",
    "response_style.md",
)


def _load_prompt_part(filename: str) -> str:
    return (_PROMPT_PARTS_DIR / filename).read_text(encoding="utf-8").strip()


def _compose_prompt(parts: tuple[str, ...]) -> str:
    return "\n\n".join(_load_prompt_part(part) for part in parts) + "\n"


SYSTEM_PROMPT = _compose_prompt(_PROMPT_PARTS)
