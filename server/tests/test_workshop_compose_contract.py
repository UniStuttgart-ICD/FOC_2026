from __future__ import annotations

from pathlib import Path


def test_workshop_compose_uses_pulled_images_without_local_builds() -> None:
    compose = Path(__file__).resolve().parents[2] / "workshop.compose.yml"

    contents = compose.read_text(encoding="utf-8")

    assert "build:" not in contents
    assert "dockerfile:" not in contents
    assert "samulienko/noetic-vizor-rviz:latest" in contents
    assert "ghcr.io/samulko/noetic-vizor-local:latest" in contents
    assert contents.count("ghcr.io/samulko/01-docker-multi-actor-mcp:latest") == 2


def test_workshop_compose_preserves_runtime_ports_and_services() -> None:
    compose = Path(__file__).resolve().parents[2] / "workshop.compose.yml"

    contents = compose.read_text(encoding="utf-8")

    for service_name in ("vizor-demo", "ros-core", "vizor-mcp", "moveit-mcp"):
        assert f"{service_name}:" in contents
    assert "container_name:" not in contents
    for port in (
        "6080",
        "5901",
        "9090",
        "10000",
        "10001",
        "10002",
        "10003",
        "11311",
        "8001",
        "8765",
    ):
        assert f'"{port}:{port}"' in contents
    assert (
        "./server/logs/moveit_planning:/root/catkin_ws/logs/moveit_planning"
        in contents
    )
