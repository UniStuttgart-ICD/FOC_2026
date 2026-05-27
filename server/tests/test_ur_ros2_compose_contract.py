from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_ur_ros2_compose_is_isolated_state_only_driver() -> None:
    compose = ROOT / "ur-ros2.compose.yml"
    dockerfile = ROOT / "docker" / "ur-ros2-humble" / "Dockerfile"

    assert compose.exists()
    assert dockerfile.exists()

    contents = compose.read_text(encoding="utf-8")

    assert "ur-ros2-driver:" in contents
    assert "network_mode: host" in contents
    assert "ports:" not in contents
    assert 'ROS_DOMAIN_ID: "42"' in contents
    assert "RMW_IMPLEMENTATION: rmw_fastrtps_cpp" in contents
    assert "source /ur_ws/install/setup.bash" in contents
    assert "ros2 launch ur_robot_driver ur_control.launch.py" in contents
    assert "ur_type:=ur10e" in contents
    assert "robot_ip:=169.254.130.206" in contents
    assert "reverse_ip:=169.254.130.5" in contents
    assert "headless_mode:=true" in contents
    assert "activate_joint_controller:=false" in contents
    assert "launch_rviz:=false" in contents


def test_ur_ros2_dockerfile_builds_patched_source_overlay() -> None:
    dockerfile = ROOT / "docker" / "ur-ros2-humble" / "Dockerfile"

    assert dockerfile.exists()

    contents = dockerfile.read_text(encoding="utf-8")

    assert "FROM ros:humble-ros-base-jammy" in contents
    assert "UR_ROS2_DRIVER_REF" in contents
    assert "UR_CLIENT_LIBRARY_REF" in contents
    assert "Universal_Robots_ROS2_Driver.git" in contents
    assert "Universal_Robots_Client_Library.git" in contents
    assert "std::chrono::milliseconds timeout(10000)" in contents
    assert "--dependency-types build" in contents
    assert "--dependency-types test" not in contents
    assert "--allow-overriding" not in contents
    assert "colcon build" in contents
    for package in (
        "ros-humble-ur",
        "ros-humble-ros2-control",
        "ros-humble-ros2-controllers",
        "ros-humble-control-msgs",
        "ros-humble-rmw-fastrtps-cpp",
        "ros-humble-ros2controlcli",
        "ros-humble-controller-manager",
        "iproute2",
        "netcat-openbsd",
        "procps",
    ):
        assert package in contents


def test_workshop_compose_stays_on_ros1_workshop_stack() -> None:
    workshop = ROOT / "workshop.compose.yml"

    contents = workshop.read_text(encoding="utf-8")

    assert "ur-ros2-driver" not in contents
    assert "network_mode: host" not in contents
    assert "ur_control.launch.py" not in contents
    assert "ros-core:" in contents
    assert "vizor-demo:" in contents


def test_ur_ros2_state_test_progress_is_documented() -> None:
    operator_note = ROOT / "docs" / "ur-ros2-state-test.md"
    readme = ROOT / "README.md"

    assert operator_note.exists()

    contents = operator_note.read_text(encoding="utf-8")
    readme_contents = readme.read_text(encoding="utf-8")

    assert "UR ROS 2 Humble Real Robot State Test" in contents
    assert "169.254.130.206" in contents
    assert "activate_joint_controller:=false" in contents
    assert "state-only" in contents
    assert "Do not use this sidecar for robot motion" in contents
    assert "calibration mismatch" in contents
    assert "docs/ur-ros2-state-test.md" in readme_contents
