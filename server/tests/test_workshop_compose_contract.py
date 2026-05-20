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


def test_workshop_compose_gates_ros_dependents_on_healthchecks() -> None:
    compose = Path(__file__).resolve().parents[2] / "workshop.compose.yml"

    contents = compose.read_text(encoding="utf-8")

    assert "socket.create_connection(('127.0.0.1', 11311), 2)" in contents
    assert "socket.create_connection(('127.0.0.1', 9090), 2)" in contents
    assert "socket.create_connection(('127.0.0.1', 6080), 2)" in contents
    assert (
        "vizor-demo:\n"
        "    image: samulienko/noetic-vizor-rviz:latest\n"
        "    tty: true\n"
        "    environment:\n"
        "      - ROS_HOSTNAME=vizor-demo\n"
        "      - ROS_MASTER_URI=http://ros-core:11311"
    ) in contents
    assert (
        "    depends_on:\n"
        "      ros-core:\n"
        "        condition: service_healthy"
    ) in contents
    assert contents.count("condition: service_healthy") == 3
