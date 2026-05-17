from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

import uvicorn

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_DIR = REPO_ROOT / "server"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from operator_dashboard.app import create_app
from operator_dashboard.config import default_config_path, load_dashboard_config
from operator_dashboard.security import DashboardSecurity

GRACEFUL_SHUTDOWN_TIMEOUT_S = 45


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
    security = DashboardSecurity.generate()
    app = create_app(config, security)

    host = config.dashboard.host
    port = config.dashboard.port
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


if __name__ == "__main__":
    main()
