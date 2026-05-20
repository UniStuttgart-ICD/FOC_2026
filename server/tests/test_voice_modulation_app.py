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

[profiles.gemini_live_preview]
category = "local_debug"

[profiles.gemini_live_preview.wake]
provider = "none"

[profiles.gemini_live_preview.emergency_stop]
enabled = false

[profiles.gemini_live_preview.stt]
provider = "deepgram_flux"
model = "flux-general-en"

[profiles.gemini_live_preview.tts]
provider = "gemini_live"
model = "gemini-3.1-flash-live-preview"
voice = "Kore"

[profiles.gemini_live_preview.agent]
provider = "gemini_api"
model = "gemini-3.1-flash-preview"
api_key_env = "GOOGLE_API_KEY"

[profiles.gemini_live_preview.mcp.robot]
url = "http://127.0.0.1:8765/mcp"

[profiles.gemini_live_preview.metrics]
enabled = false
path = "logs/metrics.jsonl"
include_text = false

[profiles.gemini_live_preview.process_trace]
enabled = false
path = "logs/trace.jsonl"
""".lstrip(),
        encoding="utf-8",
    )


def _write_prompt_parts(prompt_dir: Path) -> None:
    prompt_dir.mkdir(parents=True)
    for filename in (
        "mave_embodiment.md",
        "reasoning_agent_persona.md",
        "speech_delivery_style.md",
        "speech_tag_examples.md",
        "behavior_examples.md",
        "examples.md",
    ):
        (prompt_dir / filename).write_text(f"# {filename}\n", encoding="utf-8")


def _write_persona_template(
    server_dir: Path,
    template_id: str = "independent_agent",
) -> Path:
    template_dir = server_dir / "agent_control" / "persona_templates" / template_id
    template_dir.mkdir(parents=True)
    for filename in (
        "mave_embodiment.md",
        "reasoning_agent_persona.md",
        "speech_delivery_style.md",
        "speech_tag_examples.md",
        "behavior_examples.md",
    ):
        (template_dir / filename).write_text(f"# Template {filename}\n", encoding="utf-8")
    return template_dir


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


def test_run_agent_status_route_is_removed(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/run-agent/status")

    assert response.status_code == 404


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


def test_cartesia_voices_route_reports_missing_key(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app

    monkeypatch.delenv("CARTESIA_API_KEY", raising=False)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/cartesia/voices")

    assert response.status_code == 200
    body = response.json()
    assert body["available"] is False
    assert body["voices"] == []
    assert "CARTESIA_API_KEY" in body["reason"]
    assert body["voice_library_url"] == "https://play.cartesia.ai/voices"


def test_cartesia_voices_route_uses_injected_fetcher(tmp_path) -> None:
    from voice_modulation.app import create_app

    def fake_fetcher() -> dict[str, object]:
        return {
            "voices": [
                {
                    "id": "voice-1",
                    "name": "Ronald",
                    "language": "en",
                    "description": "Stable voice agent voice",
                }
            ],
            "has_more": True,
        }

    client = TestClient(create_app(server_dir=tmp_path, cartesia_voice_fetcher=fake_fetcher))

    response = client.get("/api/cartesia/voices")

    assert response.status_code == 200
    assert response.json() == {
        "available": True,
        "voices": [
            {
                "id": "voice-1",
                "name": "Ronald",
                "language": "en",
                "description": "Stable voice agent voice",
            }
        ],
        "has_more": True,
        "reason": None,
        "voice_library_url": "https://play.cartesia.ai/voices",
    }


def test_gemini_voices_route_lists_documented_voices(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/gemini/voices")

    assert response.status_code == 200
    voices = response.json()["voices"]
    names = {voice["name"] for voice in voices}
    assert {"Kore", "Sadaltager", "Puck"}.issubset(names)


def test_presets_route_lists_built_in_preset_names(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/presets")

    assert response.status_code == 200
    names = {preset["name"] for preset in response.json()["presets"]}
    assert names == set(BUILT_IN_PRESETS)


def test_presets_route_exposes_focused_preset_set(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/presets")

    assert response.status_code == 200
    names = [preset["name"] for preset in response.json()["presets"]]
    assert names == [
        "clean",
        "protocol_droid",
        "masked_breather",
        "helmet_comms",
        "damaged_droid",
        "ai_core",
        "titan_mech",
        "hologram",
    ]


def test_persona_route_loads_prompt_parts(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    (prompt_dir / "reasoning_agent_persona.md").write_text(
        "# Reasoning agent persona\nTest persona text.\n",
        encoding="utf-8",
    )
    (prompt_dir / "speech_delivery_style.md").write_text(
        "# Speech delivery style\nSpeak the transcript exactly.\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/persona")

    assert response.status_code == 200
    body = response.json()
    assert "Test persona text" in body["speaking_persona"]
    assert "Speak the transcript exactly" in body["speech_delivery"]
    assert body["sources"] == {
        "speaking_persona": "reasoning_agent_persona.md",
        "speech_delivery": "speech_delivery_style.md",
    }


def test_persona_route_reads_prompt_parts_fresh(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    persona_path = prompt_dir / "reasoning_agent_persona.md"
    persona_path.write_text("# Reasoning agent persona\nFirst persona.\n", encoding="utf-8")
    client = TestClient(create_app(server_dir=tmp_path))

    first = client.get("/api/persona").json()
    persona_path.write_text("# Reasoning agent persona\nSecond persona.\n", encoding="utf-8")
    second = client.get("/api/persona").json()

    assert "First persona" in first["speaking_persona"]
    assert "Second persona" in second["speaking_persona"]


def test_persona_parts_route_exposes_allowlisted_parts(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/persona/parts")

    assert response.status_code == 200
    parts = response.json()["parts"]
    by_id = {part["id"]: part for part in parts}
    assert by_id["mave_embodiment"]["editable"] is True
    assert by_id["canonical_motion_examples"]["editable"] is False


def test_persona_part_save_route_writes_allowlisted_part(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/persona/parts/behavior_examples",
        json={"content": "# Behavior examples\n- Keep it brief.\n"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["restart_required"] is True
    assert "Git has source changes" not in body
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8").startswith(
        "# Behavior examples"
    )


def test_persona_part_save_route_preserves_markdown_content_exactly(
    tmp_path: Path,
) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    client = TestClient(create_app(server_dir=tmp_path))
    content = "\n# Behavior examples\n\n- Keep it brief.\n\n"

    response = client.post(
        "/api/persona/parts/behavior_examples",
        json={"content": content},
    )

    assert response.status_code == 200
    assert response.json()["part"]["content"] == content
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8") == content


def test_persona_template_part_save_route_updates_loaded_template_folder(
    tmp_path: Path,
) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    _write_persona_template(tmp_path, "Persona 1")
    persona_2 = _write_persona_template(tmp_path, "Persona 2")
    (persona_2 / "behavior_examples.md").write_text(
        "# Behavior examples\n- Persona 2 text.\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(server_dir=tmp_path))
    saved_content = "# Behavior examples\n- Persona 1 edited text.\n"

    response = client.post(
        "/api/persona/templates/Persona%201/parts/behavior_examples",
        json={"content": saved_content},
    )
    assert response.status_code == 200
    assert response.json()["template_source_changed"] is True
    client.post("/api/persona/templates/Persona%202/load")
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8") == (
        "# Behavior examples\n- Persona 2 text.\n"
    )

    response = client.post("/api/persona/templates/Persona%201/load")

    assert response.status_code == 200
    persona_1_part = (
        tmp_path
        / "agent_control"
        / "persona_templates"
        / "Persona 1"
        / "behavior_examples.md"
    )
    assert persona_1_part.read_text(encoding="utf-8") == saved_content
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8") == saved_content


def test_persona_templates_route_lists_templates(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    _write_persona_template(tmp_path, "Bobby Fused")
    _write_persona_template(tmp_path, "kibbitz_separate")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/persona/templates")

    assert response.status_code == 200
    templates = {template["id"]: template for template in response.json()["templates"]}
    assert templates["Bobby Fused"]["available"] is True
    assert templates["Bobby Fused"]["label"] == "Bobby Fused"
    assert templates["kibbitz_separate"]["available"] is True
    assert templates["kibbitz_separate"]["label"] == "Kibbitz Separate"


def test_persona_template_load_route_writes_editable_parts(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    _write_persona_template(tmp_path, "Bobby Fused")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post("/api/persona/templates/Bobby%20Fused/load")

    assert response.status_code == 200
    body = response.json()
    assert body["restart_required"] is True
    assert {part["id"] for part in body["parts"]} >= {"mave_embodiment", "behavior_examples"}
    assert (prompt_dir / "behavior_examples.md").read_text(encoding="utf-8").startswith(
        "# Template behavior_examples.md"
    )


def test_robot_embodied_template_load_route_writes_editable_parts(tmp_path: Path) -> None:
    from voice_modulation.app import create_app

    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    _write_persona_template(tmp_path, "robot_embodied_agent")
    template_dir = (
        tmp_path / "agent_control" / "persona_templates" / "robot_embodied_agent"
    )
    (template_dir / "mave_embodiment.md").write_text(
        "# MAVE embodiment\nRobot body template.\n",
        encoding="utf-8",
    )
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post("/api/persona/templates/robot_embodied_agent/load")

    assert response.status_code == 200
    body = response.json()
    assert body["restart_required"] is True
    assert {part["id"] for part in body["parts"]} >= {
        "mave_embodiment",
        "speech_delivery_style",
    }
    assert (prompt_dir / "mave_embodiment.md").read_text(encoding="utf-8") == (
        "# MAVE embodiment\nRobot body template.\n"
    )


def test_settings_routes_load_defaults_save_and_reload(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    initial = client.get("/api/settings/local_current")
    saved = client.post(
        "/api/settings/local_current",
            json=BUILT_IN_PRESETS["protocol_droid"].to_dict() | {"gain_db": 5.0},
    )
    reloaded = client.get("/api/settings/local_current")

    assert initial.status_code == 200
    assert initial.json()["saved"] is False
    assert initial.json()["settings"]["preset_name"] == "clean"
    assert saved.status_code == 200
    assert saved.json()["ok"] is True
    assert reloaded.json()["saved"] is True
    assert reloaded.json()["settings"]["gain_db"] == 5.0


def test_settings_route_uses_profile_voice_modulation_default_without_local_override(
    tmp_path,
) -> None:
    from voice_modulation.app import create_app

    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    profiles_path.write_text(
        profiles_path.read_text(encoding="utf-8")
        + """

[profiles.local_current.voice_modulation]
enabled = true
preset_name = "profile_default"
gain_db = 2.0
""",
        encoding="utf-8",
    )
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/api/settings/local_current")

    assert response.status_code == 200
    body = response.json()
    assert body["saved"] is False
    assert body["settings"]["enabled"] is True
    assert body["settings"]["preset_name"] == "profile_default"
    assert body["settings"]["gain_db"] == 2.0


def test_settings_post_rejects_out_of_range_values(tmp_path) -> None:
    from voice_modulation.app import create_app

    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/settings/local_current",
            json=BUILT_IN_PRESETS["protocol_droid"].to_dict() | {"wet_mix": 1.5},
    )

    assert response.status_code == 400
    assert "wet_mix must be between 0.0 and 1.0" in response.json()["detail"]


def test_tts_voice_save_route_updates_runtime_profile_toml(tmp_path) -> None:
    from voice_modulation.app import create_app

    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/profiles/gemini_live_preview/tts/voice",
        json={"voice": "Kore"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["voice"] == "Kore"
    assert body["restart_required"] is True
    assert body["source_path"] == str(profiles_path)
    assert 'voice = "Kore"' in profiles_path.read_text(encoding="utf-8")


def test_tts_voice_save_route_rejects_non_gemini_voice(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/profiles/gemini_live_preview/tts/voice",
        json={"voice": "UnknownVoice"},
    )

    assert response.status_code == 400
    assert "Unsupported Gemini Live voice" in response.json()["detail"]


def test_voice_modulation_default_route_updates_runtime_profile_toml(tmp_path) -> None:
    from voice_modulation.app import create_app

    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.post(
        "/api/profiles/local_current/voice-modulation-default",
        json={"enabled": True, "preset_name": "profile_default", "gain_db": 2.0},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["restart_required"] is True
    assert body["source_path"] == str(profiles_path)
    text = profiles_path.read_text(encoding="utf-8")
    assert "[profiles.local_current.voice_modulation]" in text
    assert 'preset_name = "profile_default"' in text


def test_embodiment_routes_load_and_save_runtime_profile_toml(tmp_path) -> None:
    from voice_modulation.app import create_app

    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    client = TestClient(create_app(server_dir=tmp_path))

    initial = client.get("/api/embodiment/local_current")
    saved = client.post(
        "/api/profiles/local_current/embodiment",
        json={
            "enabled": True,
            "rosbridge_host": "127.0.0.1",
            "rosbridge_port": 9090,
            "animation_topic": "/HOLO1_AnimSignal",
            "animation_topic_type": "std_msgs/String",
            "start_blink_on_connect": True,
            "stop_blink_on_disconnect": True,
            "wave_duration_s": 0.5,
            "move_duration_s": 1.2,
            "motions": {
                "nod": {"start_signal": "start_nod", "stop_signal": "stop_nod"},
            },
            "touch_trigger": {
                "enabled": True,
                "topic": "/HOLO1_TouchSignal",
                "topic_type": "std_msgs/String",
                "link_name": "hand_link",
                "motion": "move",
                "cooldown_s": 2.0,
            },
        },
    )
    reloaded = client.get("/api/embodiment/local_current")

    assert initial.status_code == 200
    assert initial.json()["settings"]["enabled"] is False
    assert saved.status_code == 200
    assert saved.json()["restart_required"] is True
    assert reloaded.json()["settings"]["enabled"] is True
    assert reloaded.json()["settings"]["motions"]["nod"]["start_signal"] == "start_nod"
    assert reloaded.json()["settings"]["touch_trigger"]["link_name"] == "hand_link"
    text = profiles_path.read_text(encoding="utf-8")
    assert "[profiles.local_current.embodiment]" in text
    assert 'animation_topic = "/HOLO1_AnimSignal"' in text
    assert 'start_signal = "start_nod"' in text


def test_embodiment_test_route_uses_injected_controller(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    profiles_path = tmp_path / "runtime_profiles.toml"
    profiles_path.write_text(
        profiles_path.read_text(encoding="utf-8")
        + """

[profiles.local_current.embodiment]
enabled = true
""",
        encoding="utf-8",
    )
    calls: list[tuple[str, str | None]] = []

    class FakeController:
        async def start_animation(
            self, motion: str, *, side: str | None = None
        ) -> dict[str, object]:
            calls.append((f"start:{motion}", side))
            return {"ok": True}

        async def stop_animation(
            self, motion: str, *, side: str | None = None
        ) -> dict[str, object]:
            calls.append((f"stop:{motion}", side))
            return {"ok": True}

        async def stop(self) -> None:
            return None

    client = TestClient(
        create_app(
            server_dir=tmp_path,
            embodiment_controller_factory=lambda _settings: cast(Any, FakeController()),
        )
    )

    response = client.post(
        "/api/profiles/local_current/embodiment/test",
        json={"motion": "wave", "action": "start", "side": "left"},
    )

    assert response.status_code == 200
    assert calls == [("start:wave", "left")]


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
                "settings": BUILT_IN_PRESETS["protocol_droid"].to_dict(),
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
                "settings": BUILT_IN_PRESETS["protocol_droid"].to_dict(),
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


def test_tts_preview_uses_profile_voice_modulation_default_without_local_override(
    tmp_path,
    monkeypatch,
) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    profiles_path = tmp_path / "runtime_profiles.toml"
    _write_profiles(profiles_path)
    profiles_path.write_text(
        profiles_path.read_text(encoding="utf-8")
        + """

[profiles.local_current.voice_modulation]
enabled = true
preset_name = "profile_default"
gain_db = 2.0
""",
        encoding="utf-8",
    )

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    seen_settings: list[Any] = []
    dsp = types.ModuleType("voice_modulation.dsp")

    def fake_process_pcm16(pcm16: bytes, *, sample_rate: int, num_channels: int, settings: Any) -> bytes:
        seen_settings.append(settings)
        return pcm16

    cast(Any, dsp).process_pcm16 = fake_process_pcm16
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/tts",
        json={"profile_name": "local_current", "text": "Status report."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["settings"]["enabled"] is True
    assert body["settings"]["preset_name"] == "profile_default"
    assert body["settings"]["gain_db"] == 2.0
    assert seen_settings
    assert seen_settings[0].preset_name == "profile_default"


def test_source_preview_route_returns_clean_audio_without_modulation(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    calls: list[tuple[TTSProfile, str]] = []

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        calls.append((tts, text))
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    dsp = types.ModuleType("voice_modulation.dsp")
    cast(Any, dsp).process_pcm16 = lambda *args, **kwargs: pytest.fail("source route modulated")
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/source",
        json={"profile_name": "local_current", "text": "Status report."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["profile"] == "local_current"
    assert "modulated" not in body
    assert body["audio"]["sample_rate"] == 16000
    assert base64.b64decode(body["audio"]["wav_base64"]).startswith(b"RIFF")
    assert calls == [(TTSProfile(provider="kokoro", voice="af_heart"), "Status report.")]


def test_source_preview_route_accepts_cartesia_voice_override(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    calls: list[tuple[TTSProfile, str]] = []

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        calls.append((tts, text))
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    dsp = types.ModuleType("voice_modulation.dsp")
    cast(Any, dsp).process_pcm16 = lambda *args, **kwargs: pytest.fail("source route modulated")
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/source",
        json={"profile_name": "cartesia_stream", "text": "Status report.", "voice_id": "voice-override"},
    )

    assert response.status_code == 200
    assert calls == [
        (TTSProfile(provider="cartesia", model="sonic-3", voice="voice-override"), "Status report.")
    ]


def test_source_preview_route_accepts_gemini_voice_override(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    _write_prompt_parts(tmp_path / "agent_control" / "prompt_parts")
    calls: list[tuple[TTSProfile, str]] = []

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        calls.append((tts, text))
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    dsp = types.ModuleType("voice_modulation.dsp")
    cast(Any, dsp).process_pcm16 = lambda *args, **kwargs: pytest.fail("source route modulated")
    monkeypatch.setitem(sys.modules, "voice_modulation.dsp", dsp)
    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/source",
        json={"profile_name": "gemini_live_preview", "text": "Status report.", "voice_id": "Kore"},
    )

    assert response.status_code == 200
    assert calls[0][0].voice == "Kore"


def test_tts_preview_route_accepts_gemini_voice_override(tmp_path, monkeypatch) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    _write_prompt_parts(tmp_path / "agent_control" / "prompt_parts")
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
        json={"profile_name": "gemini_live_preview", "text": "Status report.", "voice_id": "Kore"},
    )

    assert response.status_code == 200
    assert calls[0][0].voice == "Kore"


def test_source_preview_route_applies_gemini_live_speech_delivery(tmp_path) -> None:
    from voice_modulation.app import create_app
    from voice_modulation.preview import AudioBytes

    _write_profiles(tmp_path / "runtime_profiles.toml")
    prompt_dir = tmp_path / "agent_control" / "prompt_parts"
    _write_prompt_parts(prompt_dir)
    (prompt_dir / "speech_delivery_style.md").write_text(
        "Use current modulation delivery.",
        encoding="utf-8",
    )
    calls: list[TTSProfile] = []

    def fake_synthesizer(tts: TTSProfile, text: str) -> AudioBytes:
        calls.append(tts)
        return AudioBytes(pcm16=_pcm16(), sample_rate=16000, channels=1)

    client = TestClient(create_app(server_dir=tmp_path, preview_synthesizer=fake_synthesizer))

    response = client.post(
        "/api/preview/source",
        json={"profile_name": "gemini_live_preview", "text": "Status report."},
    )

    assert response.status_code == 200
    assert calls
    assert calls[0].provider == "gemini_live"
    assert calls[0].instructions is not None
    assert "Use current modulation delivery." in calls[0].instructions

    (prompt_dir / "speech_delivery_style.md").write_text(
        "Use updated modulation delivery.",
        encoding="utf-8",
    )
    second_response = client.post(
        "/api/preview/source",
        json={"profile_name": "gemini_live_preview", "text": "Status report."},
    )

    assert second_response.status_code == 200
    assert calls[1].instructions is not None
    assert "Use updated modulation delivery." in calls[1].instructions


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
                "settings": BUILT_IN_PRESETS["helmet_comms"].to_dict(),
        },
    )

    assert response.status_code == 200
    output = base64.b64decode(response.json()["audio"]["pcm16_base64"])
    assert output == b"\x01\x00\x02\x00"
    assert calls == [(_pcm16(), 16000, 1, "helmet_comms")]


def test_index_page_serves_voice_mod_lab_workbench(tmp_path) -> None:
    from voice_modulation.app import create_app

    _write_profiles(tmp_path / "runtime_profiles.toml")
    client = TestClient(create_app(server_dir=tmp_path))

    response = client.get("/")

    assert response.status_code == 200
    assert "Agent Persona Lab" in response.text
    assert 'rel="icon" type="image/svg+xml"' in response.text
    assert "data:image/svg+xml" in response.text
    assert "profileSelect" in response.text
    assert "geminiVoiceSelect" in response.text
    assert 'data-tab="voiceTab"' not in response.text
    assert 'data-tab="modulationTab">Modulation' in response.text
    assert 'data-tab="embodimentTab">Embodiment' in response.text
    assert "speechDeliveryEditor" in response.text
    assert "saveSpeechDeliveryBtn" in response.text
    assert response.text.index("speechDeliveryEditor") < response.text.index("Character bay")
    assert "voiceIdInput" in response.text
    assert "https://play.cartesia.ai/voices" in response.text
    assert "https://docs.cloud.google.com/text-to-speech/docs/gemini-tts" in response.text
    assert "personaParts" in response.text
    assert "saveModulationBtn" in response.text
    assert "sourceBtn" in response.text
    assert "renderBtn" in response.text
    assert 'data-tab="runAgentTab">RUN AGENT' not in response.text
    assert "touchLinkName" in response.text
    assert "motionControlGrid" in response.text
    assert "addMotionBtn" in response.text
    assert "openDashboardBtn" not in response.text
    assert "openPipecatClientBtn" not in response.text
    assert "runAgentStatus" not in response.text
    assert "/api/run-agent/status" not in response.text
    assert "runAgentFrame" not in response.text
    assert "agent-frame" not in response.text
    assert "<iframe" not in response.text
    assert "window.open" not in response.text
    assert "justify-content: center" in response.text
    assert "window.location.assign(status.dashboard_url)" not in response.text
    assert "window.location.assign(status.pipecat_client_url)" not in response.text
    assert "Pipecat is not running yet." not in response.text
    assert "protocol_droid" in response.text
    assert "masked_breather" in response.text
    assert "Character bay" in response.text
    for label in ["Voice size", "Robot edge", "Radio filter", "Glitch", "Space", "Mask breath"]:
        assert label in response.text
    for expert_label in [
        "Ring modulation",
        "Tremolo depth",
        "Echo feedback",
        "Bit depth",
        "Limiter",
    ]:
        assert expert_label not in response.text


def test_tts_synthesizer_reports_missing_provider_env(monkeypatch) -> None:
    from voice_modulation.preview import VoicePreviewError, synthesize_tts_reference

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(VoicePreviewError, match="OPENAI_API_KEY"):
        synthesize_tts_reference(TTSProfile(provider="openai"), "hello")


@pytest.mark.asyncio
async def test_tts_reference_preview_primes_sample_rate_before_run_tts(monkeypatch) -> None:
    from pipecat.frames.frames import TTSAudioRawFrame

    from voice_modulation import preview

    class FakeTTSService:
        def __init__(self) -> None:
            self._sample_rate = 0
            self.stopped = False

        @property
        def sample_rate(self) -> int:
            return self._sample_rate

        async def stop(self, frame) -> None:
            self.stopped = True

        async def run_tts(self, text: str, context_id: str):
            assert self.sample_rate == 24000
            yield TTSAudioRawFrame(
                audio=b"\x00\x00",
                sample_rate=self.sample_rate,
                num_channels=1,
                context_id=context_id,
            )

    service = FakeTTSService()

    async def fake_create_tts_service(tts: TTSProfile):
        return service, None

    monkeypatch.setattr(preview, "_create_tts_service", fake_create_tts_service)

    audio = await preview._synthesize_tts_reference(TTSProfile(provider="deepgram"), "hello")

    assert service.stopped is True
    assert audio.sample_rate == 24000
