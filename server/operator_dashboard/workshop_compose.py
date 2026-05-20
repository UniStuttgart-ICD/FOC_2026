from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "workshop.compose.yml"

_AUTH_FAILURE_MARKERS = (
    "unauthorized",
    "authentication required",
    "requested access to the resource is denied",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start or stop the workshop compose stack")
    parser.add_argument("action", choices=["up", "down"])
    args = parser.parse_args(argv)

    if args.action == "down":
        return _run_and_report(_compose_command("down", "--remove-orphans")).returncode

    pull = _run_and_report(_compose_command("pull", "--quiet"))
    if pull.returncode != 0:
        _report_auth_failure_if_needed(pull)
        return pull.returncode

    up = _run_and_report(_compose_command("up", "--detach", "--remove-orphans"))
    if up.returncode != 0:
        _report_auth_failure_if_needed(up)
    return up.returncode


def _compose_command(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


def _run_and_report(command: list[str]) -> subprocess.CompletedProcess[str]:
    result = _run(command)
    _write_completed_process_output(result)
    return result


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(
            command,
            127,
            stdout="",
            stderr=f"required executable not found: {exc.filename}\n",
        )


def _write_completed_process_output(result: subprocess.CompletedProcess[str]) -> None:
    if result.stdout:
        print(result.stdout, end="", flush=True)
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr, flush=True)


def _report_auth_failure_if_needed(result: subprocess.CompletedProcess[str]) -> None:
    output = f"{result.stdout}\n{result.stderr}".lower()
    if not any(marker in output for marker in _AUTH_FAILURE_MARKERS):
        return

    print(
        "\nDocker registry authorization failed while pulling workshop images.\n"
        f"Compose file: {COMPOSE_FILE}\n"
        "The current workshop images are public. On a student PC this is usually "
        "stale Docker credentials or Docker Desktop registry state.\n"
        "Try these commands, then click Start again:\n"
        "  docker logout ghcr.io\n"
        "  docker logout\n"
        f"  docker compose -f {COMPOSE_FILE} pull\n",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
