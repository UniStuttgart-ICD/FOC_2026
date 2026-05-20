from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
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

    with _anonymous_docker_config_env() as docker_env:
        print("Using temporary anonymous Docker config for workshop image pulls.", flush=True)
        pull = _run_and_report(_compose_command("pull", "--quiet"), env=docker_env)
        if pull.returncode != 0:
            _report_auth_failure_if_needed(pull)
            return pull.returncode

        up = _run_and_report(
            _compose_command("up", "--detach", "--remove-orphans"),
            env=docker_env,
        )
    if up.returncode != 0:
        _report_auth_failure_if_needed(up)
    return up.returncode


def _compose_command(*args: str) -> list[str]:
    return ["docker", "compose", "-f", str(COMPOSE_FILE), *args]


@contextmanager
def _anonymous_docker_config_env() -> Iterator[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="mave-docker-config-") as config_dir:
        env = os.environ.copy()
        env["DOCKER_CONFIG"] = config_dir
        yield env


def _run_and_report(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = _run(command, env=env)
    _write_completed_process_output(result)
    return result


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    print(f"$ {shlex.join(command)}", flush=True)
    try:
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
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
        "The dashboard already used a temporary anonymous Docker config, so saved "
        "student Docker/GitHub credentials were ignored.\n"
        "Try these commands, then click Start again:\n"
        "  docker logout ghcr.io\n"
        "  docker logout\n"
        f"  docker compose -f {COMPOSE_FILE} pull\n",
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
