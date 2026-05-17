from __future__ import annotations

GEMINI_LIVE_VOICES: tuple[str, ...] = (
    "Zephyr",
    "Puck",
    "Charon",
    "Kore",
    "Fenrir",
    "Leda",
    "Orus",
    "Aoede",
    "Callirrhoe",
    "Autonoe",
    "Enceladus",
    "Iapetus",
    "Umbriel",
    "Algieba",
    "Despina",
    "Erinome",
    "Algenib",
    "Rasalgethi",
    "Laomedeia",
    "Achernar",
    "Alnilam",
    "Schedar",
    "Gacrux",
    "Pulcherrima",
    "Achird",
    "Zubenelgenubi",
    "Vindemiatrix",
    "Sadachbia",
    "Sadaltager",
    "Sulafat",
)


def gemini_live_voice_options() -> list[dict[str, str]]:
    return [{"name": voice, "label": voice} for voice in GEMINI_LIVE_VOICES]


def is_gemini_live_voice(value: str) -> bool:
    return value in GEMINI_LIVE_VOICES
