from __future__ import annotations

import argparse
import socket
import sys
import webbrowser
from pathlib import Path

import uvicorn

from operator_dashboard.app import create_app
from operator_dashboard.config import default_config_path, load_dashboard_config
from operator_dashboard.security import DashboardSecurity

REPO_ROOT = Path(__file__).resolve().parents[2]
GRACEFUL_SHUTDOWN_TIMEOUT_S = 45
PORT_CHECK_TIMEOUT_S = 0.25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the operator dashboard")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to dashboard TOML config",
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Print the dashboard URL without opening a browser tab",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path = args.config or default_config_path(REPO_ROOT)
    config = load_dashboard_config(config_path)

    host = config.dashboard.host
    port = config.dashboard.port
    if _dashboard_port_in_use(host, port):
        print(
            f"Operator dashboard port {host}:{port} is already in use. "
            "Stop the existing dashboard before starting a new one.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1)

    security = DashboardSecurity.generate()
    app = create_app(config, security)

    url = f"http://{host}:{port}/?token={security.token}"
    print(f"Operator Dashboard: {url}", flush=True)
    print("Keep this window open while the workshop stack is running.", flush=True)
    if not args.no_open_browser:
        webbrowser.open(url)

    uvicorn.run(
        app,
        host=host,
        port=port,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_TIMEOUT_S,
    )


def _dashboard_port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=PORT_CHECK_TIMEOUT_S):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
