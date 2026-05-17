from pathlib import Path

import pytest

from voice_modulation.gemini_voices import GEMINI_LIVE_VOICES, is_gemini_live_voice
from voice_modulation.profile_editor import (
    save_gemini_tts_voice,
    save_voice_modulation_default,
)
from voice_modulation.settings import VoiceModulationSettings


def _profile_toml(path: Path) -> None:
    path.write_text(
        """[profiles.hybrid_gemini_live_tts]
category = "benchmark_streaming"

[profiles.hybrid_gemini_live_tts.tts]
provider = "gemini_live"
model = "gemini-3.1-flash-live-preview"
voice = "Sadaltager"

[profiles.hybrid_gemini_live_tts.agent]
provider = "gemini_api"
model = "gemini-3.1-flash-lite-preview"
""",
        encoding="utf-8",
    )


def test_gemini_live_voice_allowlist_contains_current_profile_voice() -> None:
    assert "Sadaltager" in GEMINI_LIVE_VOICES
    assert "Kore" in GEMINI_LIVE_VOICES
    assert "Puck" in GEMINI_LIVE_VOICES
    assert is_gemini_live_voice("Sadaltager")
    assert not is_gemini_live_voice("NotARealGeminiVoice")


def test_save_gemini_tts_voice_updates_only_tts_voice(tmp_path: Path) -> None:
    path = tmp_path / "runtime_profiles.toml"
    _profile_toml(path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'voice = "Sadaltager"',
            '# committed voice choice\nvoice = "Sadaltager"',
        ),
        encoding="utf-8",
    )

    result = save_gemini_tts_voice(path, "hybrid_gemini_live_tts", "Kore")

    text = path.read_text(encoding="utf-8")
    assert result.voice == "Kore"
    assert "# committed voice choice" in text
    assert 'voice = "Kore"' in text
    assert 'model = "gemini-3.1-flash-lite-preview"' in text
    assert 'provider = "gemini_api"' in text


def test_save_gemini_tts_voice_rejects_unknown_voice(tmp_path: Path) -> None:
    path = tmp_path / "runtime_profiles.toml"
    _profile_toml(path)

    with pytest.raises(ValueError, match="Unsupported Gemini Live voice"):
        save_gemini_tts_voice(path, "hybrid_gemini_live_tts", "Nope")


def test_save_voice_modulation_default_writes_profile_table(tmp_path: Path) -> None:
    path = tmp_path / "runtime_profiles.toml"
    _profile_toml(path)

    save_voice_modulation_default(
        path,
        "hybrid_gemini_live_tts",
        VoiceModulationSettings(enabled=True, preset_name="profile_default", gain_db=2.5),
    )

    text = path.read_text(encoding="utf-8")
    assert "[profiles.hybrid_gemini_live_tts.voice_modulation]" in text
    assert "enabled = true" in text
    assert 'preset_name = "profile_default"' in text
    assert "gain_db = 2.5" in text
