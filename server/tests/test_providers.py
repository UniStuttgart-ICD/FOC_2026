from unittest.mock import Mock, patch

from config import STTConfig, TTSConfig
from voice_runtime.providers import create_stt_service, create_tts_service


def test_creates_whisper_stt():
    with patch("voice_runtime.providers.WhisperSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="whisper", model="base", device="cuda"))

    service.Settings.assert_called_once_with(model="base")
    service.assert_called_once_with(device="cuda", settings="settings")


def test_creates_kokoro_tts():
    with patch("voice_runtime.providers.KokoroTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="kokoro", voice="af_heart"))

    service.Settings.assert_called_once_with(voice="af_heart")
    service.assert_called_once_with(settings="settings")


def test_creates_deepgram_flux_stt(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("voice_runtime.providers.DeepgramFluxSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="deepgram_flux", model="flux-general-en"))

    service.Settings.assert_called_once_with(model="flux-general-en")
    service.assert_called_once_with(api_key="dg", settings="settings")


def test_creates_openai_realtime_stt(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime"))

    service.Settings.assert_called_once_with(
        model="gpt-realtime-whisper", noise_reduction="near_field"
    )
    service.assert_called_once_with(api_key="oa", settings="settings")


def test_creates_openai_realtime_stt_with_configured_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAIRealtimeSTTService") as service:
        service.Settings = Mock(return_value="settings")
        create_stt_service(STTConfig(provider="openai_realtime", model="custom-transcribe"))

    service.Settings.assert_called_once_with(
        model="custom-transcribe", noise_reduction="near_field"
    )
    service.assert_called_once_with(api_key="oa", settings="settings")


def test_creates_cartesia_tts(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    with patch("voice_runtime.providers.CartesiaTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="cartesia", model="sonic-3", voice="voice-id"))

    service.Settings.assert_called_once_with(model="sonic-3", voice="voice-id")
    service.assert_called_once_with(api_key="ct", settings="settings")


def test_creates_cartesia_tts_with_default_voice_id(monkeypatch):
    monkeypatch.setenv("CARTESIA_API_KEY", "ct")
    monkeypatch.delenv("CARTESIA_VOICE_ID", raising=False)
    with patch("voice_runtime.providers.CartesiaTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="cartesia", model="sonic-3"))

    service.Settings.assert_called_once_with(
        model="sonic-3", voice="47c38ca4-5f35-497b-b1a3-415245fb35e1"
    )
    service.assert_called_once_with(api_key="ct", settings="settings")


def test_creates_openai_tts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "oa")
    with patch("voice_runtime.providers.OpenAITTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="openai", model="gpt-4o-mini-tts", voice="coral"))

    service.Settings.assert_called_once_with(model="gpt-4o-mini-tts", voice="coral")
    service.assert_called_once_with(api_key="oa", settings="settings")


def test_creates_deepgram_tts(monkeypatch):
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg")
    with patch("voice_runtime.providers.DeepgramTTSService") as service:
        service.Settings = Mock(return_value="settings")
        create_tts_service(TTSConfig(provider="deepgram", model="aura-2", voice="aura-2-andromeda-en"))

    service.Settings.assert_called_once_with(model="aura-2", voice="aura-2-andromeda-en")
    service.assert_called_once_with(api_key="dg", settings="settings")


def test_creates_gemini_live_tts(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "gg")
    with patch("voice_runtime.providers.GeminiLiveSpeechRendererService") as service:
        create_tts_service(
            TTSConfig(
                provider="gemini_live",
                model="gemini-3.1-flash-live-preview",
                voice="Kore",
                instructions="Speak the transcript exactly.",
            )
        )

    service.assert_called_once_with(
        api_key="gg",
        model="gemini-3.1-flash-live-preview",
        voice="Kore",
        instructions="Speak the transcript exactly.",
    )
