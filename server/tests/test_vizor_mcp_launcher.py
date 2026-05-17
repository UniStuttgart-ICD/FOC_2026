from pathlib import Path

from scripts.run_vizor_mcp_server import build_server_command, parse_args


def test_build_vizor_mcp_server_command_uses_root_uv_module() -> None:
    cmd = build_server_command(
        host="127.0.0.1",
        port=8001,
        rosbridge_host="localhost",
        rosbridge_port=9090,
        enable_holo1_tracking_on_startup=True,
        attention_window_s=8.0,
        holo1_tracking_keepalive_s=10.0,
    )

    assert cmd[:4] == ["uv", "run", "python", "-m"]
    assert "vizor_mcp" in cmd
    assert "--enable-holo1-tracking-on-startup" in cmd
    assert "--attention-window-s" in cmd
    assert "--holo1-tracking-keepalive-s" in cmd


def test_vizor_mcp_launcher_defaults_to_repo_root() -> None:
    args = parse_args([])

    assert args.port == 8001
    assert args.rosbridge_port == 9090
    assert Path(args.cwd).name == "server"
