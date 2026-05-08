import numpy as np
from fastapi.testclient import TestClient

from wake_tuning import app as wake_tuning_app


def test_wake_tuning_page_and_settings_api_load(monkeypatch, tmp_path):
    monkeypatch.setenv("WAKE_TUNING_SETTINGS_PATH", str(tmp_path / "settings.json"))
    client = TestClient(wake_tuning_app.app)

    page = client.get("/")
    favicon = client.get("/favicon.ico")
    settings = client.get("/api/settings?profile=hybrid_low_latency")

    assert page.status_code == 200
    assert "Mave Wake Lab" in page.text
    assert "Apply sliders" in page.text
    assert "Minimum model score required to trigger" in page.text
    assert "Audio kept before wake and replayed into STT" in page.text
    assert "OpenWakeWord Silero VAD gate" in page.text
    assert "Would replay on trigger" in page.text
    assert 'value="hybrid_openai_stt"' in page.text
    assert favicon.status_code == 204
    assert settings.status_code == 200
    assert settings.json()["profile"] == "hybrid_low_latency"
    assert settings.json()["settings"]["threshold"] > 0


def test_wake_tuning_save_then_loads_saved_profile_settings(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("WAKE_TUNING_SETTINGS_PATH", str(settings_path))
    client = TestClient(wake_tuning_app.app)

    response = client.post(
        "/api/settings",
        json={
            "profile": "hybrid_low_latency",
            "settings": {
                "threshold": 0.41,
                "vad_threshold": 0.0,
                "candidate_log_threshold": 0.3,
                "required_hits": 1,
                "min_wake_rms": 0.0,
                "min_wake_peak": 0,
                "rearm_delay_s": 0.5,
                "pre_buffer_s": 0.2,
            },
        },
    )
    reloaded = client.get("/api/settings?profile=hybrid_low_latency")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert reloaded.json()["saved"] is True
    assert reloaded.json()["settings"]["threshold"] == 0.41


def test_wake_tuning_websocket_reports_detection(monkeypatch, tmp_path):
    settings_path = tmp_path / "settings.json"
    monkeypatch.setenv("WAKE_TUNING_SETTINGS_PATH", str(settings_path))

    class FakeDetector:
        threshold = None
        vad_enabled = False

        def __init__(self, *args, **kwargs):
            self.__class__.threshold = kwargs["threshold"]

        def predict(self, pcm16):
            assert pcm16.dtype == np.int16
            return {"mave": 0.95}

        def last_vad_score(self):
            return None

    monkeypatch.setattr(wake_tuning_app, "OpenWakeWordDetector", FakeDetector)
    client = TestClient(wake_tuning_app.app)
    audio = np.full(1280, 50, dtype=np.int16).tobytes()
    settings = (
        "%7B%22threshold%22%3A0.37%2C%22vad_threshold%22%3A0.0%2C"
        "%22candidate_log_threshold%22%3A0.3%2C%22required_hits%22%3A1%2C"
        "%22min_wake_rms%22%3A0.0%2C%22min_wake_peak%22%3A0%2C"
        "%22rearm_delay_s%22%3A0.5%2C%22pre_buffer_s%22%3A0.2%7D"
    )

    with client.websocket_connect(
        f"/ws/detect?profile=hybrid_low_latency&settings={settings}"
    ) as websocket:
        ready = websocket.receive_json()
        websocket.send_bytes(audio)
        detection = websocket.receive_json()

    assert ready["type"] == "ready"
    assert detection["type"] == "detection"
    assert detection["detected"] is True
    assert detection["model_name"] == "mave"
    assert detection["score"] == 0.95
    assert detection["decision"] == "triggered"
    assert detection["vad_enabled"] is False
    assert detection["vad_score"] is None
    assert FakeDetector.threshold == 0.37


def test_settings_api_reports_default_local_state_path(monkeypatch):
    monkeypatch.delenv("WAKE_TUNING_SETTINGS_PATH", raising=False)
    client = TestClient(wake_tuning_app.app)

    response = client.get("/api/settings?profile=hybrid_low_latency")

    assert response.status_code == 200
    assert response.json()["settings_path"].endswith("state/wake_tuning_settings.json")
