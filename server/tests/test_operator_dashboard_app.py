from __future__ import annotations

import sys
from pathlib import Path

import psutil
from fastapi.testclient import TestClient

import operator_dashboard.app as dashboard_app_module
from operator_dashboard.app import create_app
from operator_dashboard.config import load_dashboard_config
from operator_dashboard.models import DashboardConfig
from operator_dashboard.security import DashboardSecurity


def _config(tmp_path: Path) -> Path:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.echo]
label = "Echo Service"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-c", "print('hello')"]
ready_checks = [
  {{ type = "process", label = "Process" }},
]
""",
        encoding="utf-8",
    )
    return config_path


def _write_dashboard_config(tmp_path: Path) -> DashboardConfig:
    return load_dashboard_config(_config(tmp_path))


def test_status_requires_token(tmp_path: Path) -> None:
    config = _write_dashboard_config(tmp_path)
    app = create_app(config, DashboardSecurity(token="secret"))
    client = TestClient(app)

    response = client.get("/api/status")

    assert response.status_code == 403


def test_status_returns_services_with_token(tmp_path: Path) -> None:
    config = _write_dashboard_config(tmp_path)
    app = create_app(config, DashboardSecurity(token="secret"))
    client = TestClient(app)

    response = client.get("/api/status?token=secret")

    assert response.status_code == 200
    body = response.json()
    assert body["services"]["echo"]["label"] == "Echo Service"
    assert body["services"]["echo"]["state"] == "stopped"
    assert body["services"]["echo"]["include_in_global_actions"] is True
    ready_checks = body["services"]["echo"]["ready_checks"]
    assert ready_checks
    assert ready_checks[0]["type"] == "process"
    assert ready_checks[0]["ok"] is False


def test_start_timeout_returns_degraded_status_with_ready_checks(
    tmp_path: Path,
) -> None:
    script = tmp_path / "long_running.py"
    script.write_text(
        """
import time

while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.echo]
label = "Echo Service"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-u", {str(script)!r}]
startup_timeout_s = 0.05
ready_checks = [
  {{ type = "tcp", label = "Closed TCP", host = "127.0.0.1", port = 9, timeout_s = 0.01 }},
]
""",
        encoding="utf-8",
    )
    app = create_app(
        load_dashboard_config(config_path), DashboardSecurity(token="secret")
    )

    with TestClient(app) as client:
        try:
            response = client.post("/api/services/echo/start?token=secret")

            assert response.status_code != 500
            assert response.status_code in {200, 202, 409}
            body = response.json()
            assert body["state"] == "degraded"
            assert body["pid"] is not None
            assert body["ready_checks"]
            assert body["ready_checks"][0]["ok"] is False
            assert "127.0.0.1:9" in body["ready_checks"][0]["detail"]
        finally:
            client.portal.call(app.state.manager.stop_all)


def test_start_errors_are_clear_non_500(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.bad_cwd]
label = "Bad CWD"
cwd = {str(missing_cwd)!r}
command = [{sys.executable!r}, "-c", "print('never')"]
ready_checks = [{{ type = "process" }}]

[services.bad_executable]
label = "Bad Executable"
cwd = {str(tmp_path)!r}
command = ["definitely-missing-dashboard-executable", "--version"]
ready_checks = [{{ type = "process" }}]
""",
        encoding="utf-8",
    )
    app = create_app(
        load_dashboard_config(config_path), DashboardSecurity(token="secret")
    )
    client = TestClient(app)

    cwd_response = client.post("/api/services/bad_cwd/start?token=secret")
    executable_response = client.post("/api/services/bad_executable/start?token=secret")
    unknown_response = client.post("/api/services/unknown/start?token=secret")

    assert cwd_response.status_code != 500
    assert cwd_response.status_code in {400, 404}
    assert "cwd" in cwd_response.json()["detail"].lower()
    assert executable_response.status_code != 500
    assert executable_response.status_code in {400, 404}
    assert "executable" in executable_response.json()["detail"].lower()
    assert unknown_response.status_code != 500
    assert unknown_response.status_code in {400, 404}
    assert "unknown" in unknown_response.json()["detail"].lower()


def test_lifespan_shutdown_stops_tracked_process(tmp_path: Path) -> None:
    script = tmp_path / "long_running.py"
    script.write_text(
        """
import time

while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.echo]
label = "Echo Service"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-u", {str(script)!r}]
ready_checks = [{{ type = "process" }}]
""",
        encoding="utf-8",
    )
    app = create_app(
        load_dashboard_config(config_path), DashboardSecurity(token="secret")
    )

    with TestClient(app) as client:
        response = client.post("/api/services/echo/start?token=secret")
        assert response.status_code == 200
        pid = response.json()["pid"]
        assert pid is not None
        assert psutil.pid_exists(pid)

    assert app.state.manager.status("echo").state == "stopped"
    assert not psutil.pid_exists(pid)


def test_start_all_skips_auxiliary_services(tmp_path: Path) -> None:
    script = tmp_path / "long_running.py"
    script.write_text(
        """
import time

while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.core]
label = "Core Service"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-u", {str(script)!r}]
ready_checks = [{{ type = "process" }}]

[services.wake_tuning]
label = "Wake Word Tuning"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-u", {str(script)!r}]
include_in_global_actions = false
ready_checks = [{{ type = "process" }}]
""",
        encoding="utf-8",
    )
    app = create_app(
        load_dashboard_config(config_path), DashboardSecurity(token="secret")
    )

    with TestClient(app) as client:
        try:
            response = client.post("/api/start-all?token=secret")

            assert response.status_code == 200
            body = response.json()
            assert body["services"]["core"]["pid"] is not None
            assert body["services"]["core"]["state"] == "ready"
            assert body["services"]["wake_tuning"]["pid"] is None
            assert body["services"]["wake_tuning"]["state"] == "stopped"
            assert body["services"]["wake_tuning"]["include_in_global_actions"] is False
        finally:
            client.portal.call(app.state.manager.stop_all)


def test_index_contains_dashboard_shell(tmp_path: Path) -> None:
    config = _write_dashboard_config(tmp_path)
    app = create_app(config, DashboardSecurity(token="secret"))
    client = TestClient(app)

    response = client.get("/?token=secret")

    assert response.status_code == 200
    assert "Operator Dashboard" in response.text


def test_index_includes_dashboard_ui_hooks(tmp_path: Path) -> None:
    app = create_app(
        load_dashboard_config(_config(tmp_path)), DashboardSecurity(token="secret")
    )
    client = TestClient(app)
    response = client.get("/?token=secret")
    assert 'id="start-all"' in response.text
    assert 'id="stop-all"' in response.text
    assert ">Start system<" in response.text
    assert ">Stop system<" in response.text
    assert 'id="service-grid"' in response.text
    assert 'id="auxiliary-service-grid"' in response.text
    assert 'href="/static/styles.css?v=' in response.text
    assert 'src="/static/app.js?v=' in response.text
    assert "Core system" in response.text
    assert "Auxiliary processes" in response.text
    assert 'id="log-panel"' in response.text
    assert 'id="robot-home"' in response.text
    assert 'id="gripper-open"' in response.text
    assert 'id="gripper-close"' in response.text


def test_dashboard_script_renders_auxiliary_processes_separately() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"
    source = script.read_text(encoding="utf-8")

    assert "auxiliaryServiceGrid" in source
    assert "auxiliaryServiceCount" in source
    assert "auxiliaryServiceIds" in source
    assert "const coreEntries = entries.filter" in source
    assert "const auxiliaryEntries = entries.filter" in source
    assert "nodes.auxiliaryServiceGrid.replaceChildren" in source


def test_index_has_no_embedded_voice_tuning_controls(tmp_path: Path) -> None:
    app = create_app(
        load_dashboard_config(_config(tmp_path)), DashboardSecurity(token="secret")
    )
    client = TestClient(app)

    response = client.get("/?token=secret")

    assert "view-tab" not in response.text
    assert 'id="independent-tools"' not in response.text
    assert 'data-independent-tool="wake-word"' not in response.text
    assert 'data-independent-tool="voice-modulation"' not in response.text
    assert 'id="wake-threshold"' not in response.text
    assert 'id="voice-pitch"' not in response.text
    assert 'id="voice-timbre"' not in response.text


def test_dashboard_script_does_not_embed_voice_tuning_controls() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"
    source = script.read_text(encoding="utf-8")

    assert "viewButtons" not in source
    assert "showView" not in source
    assert "updateWakeWordTuning" not in source
    assert "updateVoiceModulation" not in source
    assert "wakeThreshold" not in source
    assert "voicePitch" not in source


def test_service_card_keydown_ignores_child_controls() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"

    assert "if (card && event.target === card)" in script.read_text(encoding="utf-8")


def test_service_action_buttons_have_service_specific_aria_labels() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"
    source = script.read_text(encoding="utf-8")

    assert "const serviceName = service.label || serviceId;" in source
    assert '"aria-label": `Start ${serviceName}`' in source
    assert '"aria-label": `Restart ${serviceName}`' in source
    assert '"aria-label": `Stop ${serviceName}`' in source


def test_dashboard_script_wires_robot_rtde_controls() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"
    source = script.read_text(encoding="utf-8")

    assert "robotHome" in source
    assert "gripperOpen" in source
    assert "gripperClose" in source
    assert "/api/robot/home" in source
    assert "/api/robot/gripper/open" in source
    assert "/api/robot/gripper/close" in source


def test_robot_home_proxy_posts_to_verified_execution_server(
    tmp_path: Path, monkeypatch
) -> None:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        f"""
[dashboard]
host = "127.0.0.1"
port = 8787

[services.verified_execution]
label = "Verified Execution"
cwd = {str(tmp_path)!r}
command = [{sys.executable!r}, "-c", "print('server')"]
ready_checks = [
  {{ type = "http", url = "http://127.0.0.1:8770/health", label = "Verified execution" }},
]
""",
        encoding="utf-8",
    )
    requests = []

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return (
                b'{"ok":true,"robot_name":"UR10","command":"home",'
                b'"status":"homed","error":null,"correction":null}'
            )

    def fake_urlopen(request, timeout: float):
        requests.append((request.full_url, request.data, timeout))
        return FakeResponse()

    monkeypatch.setattr(
        dashboard_app_module.urllib.request,
        "urlopen",
        fake_urlopen,
    )
    app = create_app(
        load_dashboard_config(config_path), DashboardSecurity(token="secret")
    )
    client = TestClient(app)

    response = client.post("/api/robot/home?token=secret")

    assert response.status_code == 200
    assert response.json()["status"] == "homed"
    assert requests == [
        (
            "http://127.0.0.1:8770/home",
            b'{"robot_name": "UR10", "timeout_s": 30.0}',
            31.0,
        )
    ]


def test_service_refresh_preserves_grid_focus() -> None:
    script = Path(__file__).resolve().parents[1] / "operator_dashboard/static/app.js"
    source = script.read_text(encoding="utf-8")

    assert "function captureServiceGridFocus()" in source
    assert "function restoreServiceGridFocus(focusState)" in source
    assert "function serviceGridNodes()" in source
    assert "function serviceGridFor(node)" in source
    assert "serviceGridFor(activeElement)" in source
    assert "CSS.escape" in source
    assert "const focusState = captureServiceGridFocus();" in source
    assert "nodes.serviceGrid.replaceChildren(" in source
    assert "nodes.auxiliaryServiceGrid.replaceChildren(" in source
    assert "restoreServiceGridFocus(focusState);" in source
