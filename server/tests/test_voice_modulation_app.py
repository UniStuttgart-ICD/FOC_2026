from __future__ import annotations

import base64
import sys
import types
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from voice_modulation.settings import BUILT_IN_PRESETS
from voice_runtime.profiles import TTSProfile


def _pcm16() -> bytes:
    return b"\x00\x00\x00\x40\x00\xc0"


def _write_profiles(path: Path) -> None:
    path.write_text(
        """
[profiles.local_current]
category = "local_debug"

[profiles.local_current.wake]
provider = "none"

[profiles.local_current.emergency_stop]
enabled = false

[profiles.local_current.stt]
provider = "whisper"
model = "base"
device = "cpu"

[profiles.local_current.tts]
provider = "kokoro"
voice = "af_heart"

[profiles.local_current.agent]
provider = "openai_api"
model = "gpt-5.4-mini"
api_key_env = "OPENAI_API_KEY"

[profiles.local_current.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.local_current.metrics]
enabled = false
path = "logs/metrics.jsonl"
include_text = false

[profiles.local_current.process_trace]
enabled = false
path = "logs/trace.jsonl"

[profiles.cartesia_stream]
category = "benchmark_streaming"

[profiles.cartesia_stream.wake]
provider = "none"

[profiles.cartesia_stream.emergency_stop]
enabled = false

[profiles.cartesia_stream.stt]
provider = "deepgram_flux"
model = "flux-general-en"

[profiles.cartesia_stream.tts]
provider = "cartesia"
model = "sonic-3"
voice = "voice-123"

[profiles.cartesia_stream.agent]
provider = "gemini_api"
model = "gemini-2.5-flash"
api_key_env = "GOOGLE_API_KEY"

[profiles.cartesia_stream.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.cartesia_stream.metrics]
enabled = false
path = "logs/metrics.jsonl"
include_text = false

[profiles.cartesia_stream.process_trace]
enabled = false
path = "logs/trace.jsonl"
""".lstrip(),
        encoding="utf-8",
    )


def test_wav_preview_helpers_round_trip_pcm16() -> None:
    from voice_modulation.preview import (
        AudioBytes,
        decode_preview,
        encode_preview,
        pcm16_to_wav_bytes,
        wav_bytes_to_pcm16,
    )

    audio = AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)
    wav = pcm16_to_wav_bytes(audio)
    decoded = wav_bytes_to_pcm16(wav)
    preview = decode_preview(encode_preview(audio))

    assert wav.startswith(b"RIFF")
    assert decoded == audio
    assert preview == audio


def test_render_effect_preview_passes_channel_count_to_real_dsp() -> None:
    from voice_modulation.preview import AudioBytes, render_effect_preview

    stereo = AudioBytes(pcm16=b"\x00\x00\x00\x10\x00\x20\x00\x30", sample_rate=16000, channels=2)

    rendered = render_effect_preview(stereo, BUILT_IN_PRESETS["clean"])

    assert rendered == stereo


def test_profiles_route_lists_runtime_profiles_with_missing_env(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/profiles")

    assert response.status_code == 200
    profiles = {profile["name"]: profile for profile in response.json()["profiles"]}
    assert profiles["local_current"]["category"] == "local_debug"
    assert profiles["local_current"]["tts"] == {
        "provider": "kokoro",
        "model": None,
        "voice": "af_heart",
    }
    assert profiles["local_current"]["missing_env"] == ["OPENAI_API_KEY"]
    assert profiles["cartesia_stream"]["missing_env"] == [
        "DEEPGRAM_API_KEY",
        "CARTESIA_API_KEY",
        "GOOGLE_API_KEY",
    ]


def test_profiles_route_loads_server_env_file(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "OPENAI_API_KEY=dummy-openai",
                "DEEPGRAM_API_KEY=dummy-deepgram",
                "CARTESIA_API_KEY=dummy-cartesia",
                "GOOGLE_API_KEY=dummy-google",
            ]
        ),
        encoding="utf-8",
    )
    for env_name in ("OPENAI_API_KEY", "DEEPGRAM_API_KEY", "CARTESIA_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)

    client = TestClient(create_app(server_dir=tmp_path))
    response = client.get("/api/profiles")

    assert response.status_code == 200
    profiles = {profile["name"]: profile for profile in response.json()["profiles"]}
    assert profiles["local_current"]["missing_env"] == []
    assert profiles["cartesia_stream"]["missing_env"] == []


def test_presets_route_lists_built_in_preset_names(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/presets")

    assert response.status_code == 200
    names = {preset["name"] for preset in response.json()["presets"]}
    assert names == set(BUILT_IN_PRESETS)


def test_settings_routes_load_defaults_save_and_reload(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    initial = client.get("/api/settings/local_current")
    saved = client.post(
        "/api/settings/local_current",
        json=BUILT_IN_PRESETS["robot"].to_dict() | {"gain_db": 5.0},
    )
    reloaded = client.get("/api/settings/local_current")

    assert initial.status_code == 200
    assert initial.json()["saved"] is False
    assert initial.json()["settings"]["preset_name"] == "clean"
    assert saved.status_code == 200
    assert saved.json()["ok"] is True
    assert reloaded.json()["saved"] is True
    assert reloaded.json()["settings"]["gain_db"] == 5.0


def test_settings_post_rejects_out_of_range_values(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/settings/local_current",
        json=BUILT_IN_PRESETS["robot"].to_dict() | {"wet_mix": 1.5},
    )

    assert response.status_code == 400
    assert "wet_mix must be between 0.0 and 1.0" in response.json()["detail"]


def test_effect_preview_rejects_invalid_base64_audio(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/preview/effect",
        json={
            "audio": {
                "pcm16_base64": "not base64",
                "sample_rate": 16000,
                "channels": 1,
            },
            "settings": BUILT_IN_PRESETS["robot"].to_dict(),
        },
    )

    assert response.status_code == 400
    assert "Invalid preview audio" in response.json()["detail"]


def test_effect_preview_rejects_channel_misaligned_pcm(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/preview/effect",
        json={
            "audio": {
                "pcm16_base64": base64.b64encode(b"\x00\x00").decode("ascii"),
                "sample_rate": 16000,
                "channels": 2,
            },
            "settings": BUILT_IN_PRESETS["robot"].to_dict(),
        },
    )

    assert response.status_code == 400
    assert "audio length must align" in response.json()["detail"]


def test_tts_preview_rejects_unknown_profile(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/preview/tts",
        json={"profile_name": "missing", "text": "Status report."},
    )

    assert response.status_code == 404
    assert "Unknown profile" in response.json()["detail"]


def test_tts_preview_rejects_invalid_synthesized_audio(tmp_path) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        return AudioBytes(pcm16=b"\x00\x00", sample_rate=16000, channels=2)

    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/tts",
        json={"profile_name": "local_current", "text": "Status report."},
    )

    assert response.status_code == 400
    assert "audio length must align" in response.json()["detail"]


def test_tts_preview_route_uses_injected_synthesizer(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    calls: list[tuple[TTSProfile, str]] = []

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        calls.append((tts, text))
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    dsp = types.ModuleType("voice_modulation.dsp")
    cast(Any, dsp).process_pcm16 = lambda pcm16, *, sample_rate, num_channels, settings: pcm16
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/tts",
        json={"profile_name": "local_current", "text": "Status report."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["audio"]["sample_rate"] == 16000
    assert body["audio"]["channels"] == 1
    assert base64.b64decode(body["audio"]["wav_base64"]).startswith(b"RIFF")
    assert calls == [(TTSProfile(provider="kokoro", voice="af_heart"), "Status report.")]


def test_effect_preview_route_uses_dsp_process_pcm16(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes, encode_preview

    _write_profiles(tmp_path / "runtime_profiles.toml")
    dsp = types.ModuleType("voice_modulation.dsp")
    calls = []

    def process_pcm16(pcm16, *, sample_rate, num_channels, settings):
        calls.append((pcm16, sample_rate, num_channels, settings.preset_name))
        return b"\x01\x00\x02\x00"

    cast(Any, dsp).process_pcm16 = process_pcm16
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/preview/effect",
        json={
            "audio": asdict(encode_preview(AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1))),
            "settings": BUILT_IN_PRESETS["radio"].to_dict(),
        },
    )

    assert response.status_code == 200
    output = base64.b64decode(response.json()["audio"]["pcm16_base64"])
    assert output == b"\x01\x00\x02\x00"
    assert calls == [(_pcm16(), 16000, 1, "radio")]


def test_index_page_serves_voice_mod_lab_workbench(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "Voice Mod Lab" in response.text
    assert "profileSelect" in response.text
    assert "low_battery" in response.text


def test_tts_synthesizer_reports_missing_provider_env(monkeypatch) -> None:
    from voice_modulation.preview import VoicePreviewError, synthesize_tts_reference

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(VoicePreviewError, match="OPENAI_API_KEY"):
        synthesize_tts_reference(TTSProfile(provider="openai"), "hello")
