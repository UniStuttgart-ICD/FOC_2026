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
