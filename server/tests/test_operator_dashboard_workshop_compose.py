from __future__ import annotations

import subprocess

import operator_dashboard.workshop_compose as workshop_compose


def _completed(command: list[str], returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout="", stderr=stderr)


def test_up_pulls_images_before_starting_compose(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(workshop_compose, "_run", fake_run)

    assert workshop_compose.main(["up"]) == 0

    assert commands == [
        [
            "docker",
            "compose",
            "-f",
            str(workshop_compose.COMPOSE_FILE),
            "pull",
            "--quiet",
        ],
        [
            "docker",
            "compose",
            "-f",
            str(workshop_compose.COMPOSE_FILE),
            "up",
            "--detach",
            "--remove-orphans",
        ],
    ]


def test_up_reports_registry_auth_failure(monkeypatch, capsys) -> None:
    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return _completed(
            command,
            returncode=1,
            stderr="denied: requested access to the resource is denied\nunauthorized: authentication required",
        )

    monkeypatch.setattr(workshop_compose, "_run", fake_run)

    assert workshop_compose.main(["up"]) == 1

    captured = capsys.readouterr()
    assert "Docker registry authorization failed while pulling workshop images." in captured.err
    assert "docker logout ghcr.io" in captured.err
    assert str(workshop_compose.COMPOSE_FILE) in captured.err


def test_down_removes_orphans(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command: list[str]) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return _completed(command)

    monkeypatch.setattr(workshop_compose, "_run", fake_run)

    assert workshop_compose.main(["down"]) == 0

    assert commands == [
        [
            "docker",
            "compose",
            "-f",
            str(workshop_compose.COMPOSE_FILE),
            "down",
            "--remove-orphans",
        ]
    ]
