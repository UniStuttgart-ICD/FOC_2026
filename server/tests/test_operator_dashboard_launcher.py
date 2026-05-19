from __future__ import annotations

import sys

from operator_dashboard.models import DashboardConfig, ServiceConfig
from operator_dashboard.security import DashboardSecurity


def test_launcher_configures_graceful_shutdown_timeout(monkeypatch) -> None:
    import scripts.run_operator_dashboard as launcher

    captured: dict[str, object] = {}
    config = DashboardConfig(
        services={
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=".",
                command=[sys.executable, "-c", "print('fake')"],
            )
        }
    )

    monkeypatch.setattr(launcher, "load_dashboard_config", lambda path: config)
    monkeypatch.setattr(launcher, "_dashboard_port_in_use", lambda host, port: False)
    monkeypatch.setattr(
        launcher.DashboardSecurity,
        "generate",
        staticmethod(lambda: DashboardSecurity(token="secret")),
    )
    monkeypatch.setattr(launcher, "create_app", lambda cfg, security: "app")
    monkeypatch.setattr(sys, "argv", ["run_operator_dashboard.py", "--no-open-browser"])
    monkeypatch.setattr(
        launcher.uvicorn,
        "run",
        lambda app, **kwargs: captured.update({"app": app, **kwargs}),
    )

    launcher.main()

    assert captured["app"] == "app"
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8787
    assert captured["timeout_graceful_shutdown"] >= 30


def test_launcher_rejects_used_port_before_generating_token(monkeypatch, capsys) -> None:
    import scripts.run_operator_dashboard as launcher

    config = DashboardConfig(
        services={
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=".",
                command=[sys.executable, "-c", "print('fake')"],
            )
        }
    )

    monkeypatch.setattr(launcher, "load_dashboard_config", lambda path: config)
    monkeypatch.setattr(launcher, "_dashboard_port_in_use", lambda host, port: True)
    monkeypatch.setattr(
        launcher.DashboardSecurity,
        "generate",
        staticmethod(lambda: (_ for _ in ()).throw(AssertionError("token generated"))),
    )
    monkeypatch.setattr(
        launcher,
        "create_app",
        lambda cfg, security: (_ for _ in ()).throw(AssertionError("app created")),
    )
    monkeypatch.setattr(
        launcher.webbrowser,
        "open",
        lambda url: (_ for _ in ()).throw(AssertionError("browser opened")),
    )
    monkeypatch.setattr(
        launcher.uvicorn,
        "run",
        lambda app, **kwargs: (_ for _ in ()).throw(AssertionError("uvicorn started")),
    )
    monkeypatch.setattr(sys, "argv", ["run_operator_dashboard.py", "--no-open-browser"])

    try:
        launcher.main()
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("launcher did not exit")

    captured = capsys.readouterr()
    assert "already in use" in captured.err
    assert "Operator Dashboard:" not in captured.out
