"""System prompt for the simulation-only voice robot agent."""

import re
from pathlib import Path

_PROMPT_PARTS_DIR = Path(__file__).with_name("prompt_parts")
_PROMPT_PARTS = (
    "mave_embodiment.md",
    "reasoning_agent_persona.md",
    "robot_contract.md",
    "examples.md",
    "speech_tag_examples.md",
    "response_style.md",
)
_SPEECH_DELIVERY_PARTS = ("speech_delivery_style.md",)
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)


def _load_prompt_part(filename: str) -> str:
    text = (_PROMPT_PARTS_DIR / filename).read_text(encoding="utf-8")
    return _HTML_COMMENT_PATTERN.sub("", text).strip()


def _compose_prompt(parts: tuple[str, ...]) -> str:
    return "\n\n".join(_load_prompt_part(part) for part in parts) + "\n"


SYSTEM_PROMPT = _compose_prompt(_PROMPT_PARTS)
SPEAKING_AGENT_PERSONA = _load_prompt_part("reasoning_agent_persona.md")
SPEECH_DELIVERY_STYLE = _compose_prompt(_SPEECH_DELIVERY_PARTS).strip()
