import importlib.util
from pathlib import Path

import pytest
import tomllib
from pydantic import ValidationError

from operator_dashboard.config import default_config_path, load_dashboard_config
from operator_dashboard.models import CheckType, ServiceState
from robot_control.shared_geometry.modeltracker_sync_server import (
    DEFAULT_PORT as MODELTRACKER_SYNC_PORT,
)


def test_loads_dashboard_config_with_services(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        """
[dashboard]
host = "127.0.0.1"
port = 8787

[services.echo]
label = "Echo Service"
cwd = "."
command = ["python", "-c", "print('hello')"]
ready_checks = [
  { type = "tcp", host = "127.0.0.1", port = 8765, label = "MCP port" },
]
links = [
  { label = "Open", url = "http://localhost:8765" },
]
""",
        encoding="utf-8",
    )

    config = load_dashboard_config(config_path)

    assert config.dashboard.host == "127.0.0.1"
    assert config.dashboard.port == 8787
    assert list(config.services) == ["echo"]
    service = config.services["echo"]
    assert service.label == "Echo Service"
    assert service.command == ["python", "-c", "print('hello')"]
    assert service.include_in_global_actions is True
    assert service.ready_checks[0].type is CheckType.TCP
    assert service.links[0].url == "http://localhost:8765"
    assert ServiceState.READY.value == "ready"


def test_known_auxiliary_services_default_out_of_global_actions(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        """
[dashboard]
host = "127.0.0.1"
port = 8787

[services.wake_tuning]
label = "Wake Word Tuning"
cwd = "."
command = ["python", "-c", "print('wake')"]

[services.voice_modulation]
label = "Agent Persona Lab"
cwd = "."
command = ["python", "-c", "print('voice')"]
""",
        encoding="utf-8",
    )

    config = load_dashboard_config(config_path)

    assert config.services["wake_tuning"].include_in_global_actions is False
    assert config.services["voice_modulation"].include_in_global_actions is False


def test_operator_dashboard_and_moveit_mcp_are_packaged() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["name"] == "pipecat-agent"
    assert "ur-rtde>=1.6.2,<2" not in pyproject["project"]["dependencies"]
    assert pyproject["project"]["optional-dependencies"]["robot"] == [
        "ur-rtde>=1.6.2,<2"
    ]
    assert importlib.util.find_spec("operator_dashboard") is not None
    assert importlib.util.find_spec("moveit_mcp") is not None
    assert importlib.util.find_spec("vizor_mcp") is not None
    assert importlib.util.find_spec("verified_execution_server") is not None


def test_example_config_starts_vizor_first_and_uses_default_pipecat_command() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = default_config_path(repo_root)
    config = load_dashboard_config(config_path)

    assert (
        config_path
        == repo_root / "server" / "operator_dashboard" / "default_config.toml"
    )
    assert list(config.services) == [
        "vizor",
        "modeltracker_sync",
        "verified_execution",
        "pipecat",
        "wake_tuning",
        "voice_modulation",
    ]
    vizor = config.services["vizor"]
    assert vizor.cwd == str(repo_root)
    assert vizor.require_running_process is False
    assert vizor.command == [
        "uv",
        "run",
        "--directory",
        "server",
        "python",
        "-m",
        "operator_dashboard.workshop_compose",
        "up",
    ]
    assert vizor.stop_command == [
        "uv",
        "run",
        "--directory",
        "server",
        "python",
        "-m",
        "operator_dashboard.workshop_compose",
        "down",
    ]
    assert (
        vizor.links[0].url
        == "http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote"
    )
    assert (
        next(check.url for check in vizor.ready_checks if check.type is CheckType.HTTP)
        == "http://127.0.0.1:6080/vnc_auto.html?host=127.0.0.1&port=6080&path=websockify&autoconnect=true&resize=remote"
    )
    assert [check.port for check in vizor.ready_checks if check.type is CheckType.TCP] == [
        9090,
        5901,
        8001,
        8765,
    ]
    modeltracker_sync = config.services["modeltracker_sync"]
    assert modeltracker_sync.cwd == str(repo_root / "server")
    assert modeltracker_sync.command == [
        "uv",
        "run",
        "python",
        "-m",
        "robot_control.shared_geometry.modeltracker_sync_server",
    ]
    assert MODELTRACKER_SYNC_PORT == 8788
    assert (
        modeltracker_sync.ready_checks[0].url
        == f"http://127.0.0.1:{MODELTRACKER_SYNC_PORT}/health"
    )
    execution = config.services["verified_execution"]
    assert execution.cwd == str(repo_root / "server")
    assert execution.command == [
        "uv",
        "run",
        "--extra",
        "robot",
        "python",
        "-m",
        "verified_execution_server",
    ]
    assert execution.env["UR_ROBOT_IP"] == "169.254.130.206"
    assert execution.env["UR_SKIP_GRIPPER"] == "false"
    assert execution.ready_checks[0].url == "http://127.0.0.1:8770/health"
    pipecat = config.services["pipecat"]
    assert pipecat.command == ["uv", "run", "bot.py"]
    assert pipecat.env["MCP_VIZOR_URL"] == "http://127.0.0.1:8001/mcp"
    assert pipecat.env["VERIFIED_EXECUTION_URL"] == "http://127.0.0.1:8770"
    assert pipecat.env["ROBOT_JOB_MONITOR_HOST"] == "127.0.0.1"
    assert pipecat.env["ROBOT_JOB_MONITOR_PORT"] == "8898"
    assert [check.url for check in pipecat.ready_checks] == [
        "http://localhost:7860/client/",
        "http://127.0.0.1:8898",
    ]
    assert all(check.type is CheckType.HTTP for check in pipecat.ready_checks)
    assert [check.required for check in pipecat.ready_checks] == [True, False]
    assert [link.url for link in pipecat.links] == [
        "http://localhost:7860/client/",
        "http://127.0.0.1:8898",
    ]
    wake_tuning = config.services["wake_tuning"]
    assert wake_tuning.include_in_global_actions is False
    assert wake_tuning.command == ["uv", "run", "python", "-m", "wake_tuning.app"]
    assert wake_tuning.ready_checks[0].url == "http://127.0.0.1:9010"
    assert wake_tuning.links[0].url == "http://127.0.0.1:9010"
    voice_modulation = config.services["voice_modulation"]
    assert voice_modulation.label == "Agent Persona Lab"
    assert voice_modulation.include_in_global_actions is False
    assert voice_modulation.command == [
        "uv",
        "run",
        "uvicorn",
        "voice_modulation.app:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8897",
    ]
    assert voice_modulation.ready_checks[0].url == "http://127.0.0.1:8897"
    assert voice_modulation.ready_checks[0].label == "Agent Persona Lab"
    assert voice_modulation.links[0].label == "Open Agent Persona"
    assert voice_modulation.links[0].url == "http://127.0.0.1:8897"


def test_rejects_non_localhost_dashboard_host(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        """
[dashboard]
host = "0.0.0.0"
port = 8787

[services.echo]
label = "Echo Service"
cwd = "."
command = ["python", "-c", "print('hello')"]
""",
        encoding="utf-8",
    )

    with pytest.raises(
        ValidationError, match="dashboard host must be 127.0.0.1 or localhost"
    ):
        load_dashboard_config(config_path)


def test_rejects_empty_service_command(tmp_path: Path) -> None:
    config_path = tmp_path / "dashboard.toml"
    config_path.write_text(
        """
[dashboard]
host = "127.0.0.1"
port = 8787

[services.bad]
label = "Bad"
cwd = "."
command = []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="command must contain at least one item"):
        load_dashboard_config(config_path)
