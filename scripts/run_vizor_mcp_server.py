#!/usr/bin/env python3
"""Restarting supervisor for the standalone Vizor MCP server."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path


def default_server_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "server"


def build_server_command(
    *,
    host: str,
    port: int,
    rosbridge_host: str,
    rosbridge_port: int,
    enable_holo1_tracking_on_startup: bool,
    attention_window_s: float,
    holo1_tracking_keepalive_s: float,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "vizor_mcp",
        "--rosbridge-host",
        rosbridge_host,
        "--rosbridge-port",
        str(rosbridge_port),
        "--transport",
        "streamable-http",
        "--http-host",
        host,
        "--http-port",
        str(port),
    ]
    if enable_holo1_tracking_on_startup:
        cmd.append("--enable-holo1-tracking-on-startup")
    cmd.extend(["--attention-window-s", str(attention_window_s)])
    cmd.extend(["--holo1-tracking-keepalive-s", str(holo1_tracking_keepalive_s)])
    return cmd


def supervise(
    cmd: Sequence[str],
    *,
    cwd: str,
    popen_factory: Callable[..., subprocess.Popen] = subprocess.Popen,
    restart_delay: float = 1.0,
    max_restart_delay: float = 30.0,
    max_restarts: int | None = None,
) -> int:
    restarts = 0
    last_exit_code = 0
    current_delay = restart_delay

    while True:
        print(f"[Supervisor] Starting: {' '.join(cmd)}", file=sys.stderr)
        process = popen_factory(cmd, cwd=cwd)
        try:
            last_exit_code = process.wait()
        except KeyboardInterrupt:
            print("[Supervisor] Interrupted; stopping child...", file=sys.stderr)
            terminate = getattr(process, "terminate", None)
            if callable(terminate):
                terminate()
            return 130

        if last_exit_code == 0:
            print("[Supervisor] Server exited cleanly; not restarting.", file=sys.stderr)
            return 0

        if max_restarts is not None and restarts >= max_restarts:
            print("[Supervisor] Restart limit reached; exiting.", file=sys.stderr)
            return last_exit_code

        print(
            f"[Supervisor] Server exited with code {last_exit_code}; "
            f"restarting in {current_delay:.1f}s...",
            file=sys.stderr,
        )
        restarts += 1
        time.sleep(current_delay)
        current_delay = min(current_delay * 2, max_restart_delay)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Restarting Vizor MCP server supervisor")
    parser.add_argument("--cwd", default=str(default_server_dir()))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--rosbridge-host", default="localhost")
    parser.add_argument("--rosbridge-port", type=int, default=9090)
    parser.add_argument("--attention-window-s", type=float, default=8.0)
    parser.add_argument("--holo1-tracking-keepalive-s", type=float, default=10.0)
    parser.add_argument("--enable-holo1-tracking-on-startup", action="store_true")
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument("--max-restart-delay", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cmd = build_server_command(
        host=args.host,
        port=args.port,
        rosbridge_host=args.rosbridge_host,
        rosbridge_port=args.rosbridge_port,
        enable_holo1_tracking_on_startup=args.enable_holo1_tracking_on_startup,
        attention_window_s=args.attention_window_s,
        holo1_tracking_keepalive_s=args.holo1_tracking_keepalive_s,
    )
    return supervise(
        cmd,
        cwd=args.cwd,
        restart_delay=args.restart_delay,
        max_restart_delay=args.max_restart_delay,
    )


if __name__ == "__main__":
    raise SystemExit(main())
