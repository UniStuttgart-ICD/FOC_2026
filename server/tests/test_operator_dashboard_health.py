from __future__ import annotations

import socket
import sys
import urllib.error
import urllib.request
from http.client import HTTPMessage
from pathlib import Path

import pytest

from operator_dashboard.health import HealthChecker
from operator_dashboard.models import (
    CheckType,
    ReadyCheckConfig,
    ServiceConfig,
    ServiceState,
)
from operator_dashboard.service_manager import ServiceManager


def _long_running_command(tmp_path: Path, marker: str | None = None) -> list[str]:
    script = tmp_path / f"service_{marker or 'plain'}.py"
    if marker is None:
        script.write_text(
            """
import time

while True:
    time.sleep(0.1)
""",
            encoding="utf-8",
        )
    else:
        order_file = tmp_path / "start_order.txt"
        script.write_text(
            f"""
import time
from pathlib import Path

with Path({str(order_file)!r}).open("a", encoding="utf-8") as handle:
    handle.write({marker!r} + "\\n")
    handle.flush()

while True:
    time.sleep(0.1)
""",
            encoding="utf-8",
        )
    return [sys.executable, "-u", str(script)]


@pytest.mark.asyncio
async def test_process_check_reports_running_process(tmp_path: Path) -> None:
    manager = ServiceManager(
        {
            "worker": ServiceConfig(
                label="Worker",
                cwd=str(tmp_path),
                command=_long_running_command(tmp_path),
                ready_checks=[ReadyCheckConfig(type=CheckType.PROCESS)],
            )
        }
    )
    checker = HealthChecker(manager)

    await manager.start("worker")
    try:
        statuses = await checker.check_service("worker")
    finally:
        await manager.stop("worker")

    assert len(statuses) == 1
    assert statuses[0].ok is True
    assert statuses[0].label == "process"


@pytest.mark.asyncio
async def test_tcp_check_reports_closed_localhost_port(tmp_path: Path) -> None:
    manager = ServiceManager(
        {
            "closed": ServiceConfig(
                label="Closed Port",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('unused')"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.TCP,
                        host="127.0.0.1",
                        port=9,
                        timeout_s=0.05,
                    )
                ],
            )
        }
    )
    checker = HealthChecker(manager)

    statuses = await checker.check_service("closed")

    assert len(statuses) == 1
    assert statuses[0].ok is False
    assert "127.0.0.1:9" in statuses[0].detail


@pytest.mark.asyncio
async def test_http_check_reports_ok_for_status_200(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def urlopen(url: str, timeout: float) -> Response:
        assert url == "http://127.0.0.1/health"
        assert timeout == 0.05
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    manager = ServiceManager(
        {
            "web": ServiceConfig(
                label="Web",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('unused')"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.HTTP,
                        url="http://127.0.0.1/health",
                        timeout_s=0.05,
                    )
                ],
            )
        }
    )
    checker = HealthChecker(manager)

    statuses = await checker.check_service("web")

    assert len(statuses) == 1
    assert statuses[0].type is CheckType.HTTP
    assert statuses[0].ok is True
    assert statuses[0].detail == "http http://127.0.0.1/health returned 200"


@pytest.mark.asyncio
async def test_http_check_reports_verified_execution_robot_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        status = 200

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"robot":{"robot_connected":true,'
                b'"gripper_connected":true,"gripper_position":17}}'
            )

    def urlopen(url: str, timeout: float) -> Response:
        assert url == "http://127.0.0.1:8770/health"
        assert timeout == 0.05
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    manager = ServiceManager(
        {
            "verified_execution": ServiceConfig(
                label="Verified Execution",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('unused')"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.HTTP,
                        url="http://127.0.0.1:8770/health",
                        timeout_s=0.05,
                    )
                ],
            )
        }
    )
    checker = HealthChecker(manager)

    statuses = await checker.check_service("verified_execution")

    assert statuses[0].detail == (
        "http http://127.0.0.1:8770/health returned 200; "
        "robot ready; gripper ready"
    )


@pytest.mark.asyncio
async def test_http_check_reports_ok_for_status_304_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(url: str, timeout: float) -> None:
        assert url == "http://127.0.0.1/health"
        assert timeout == 0.05
        raise urllib.error.HTTPError(
            url,
            304,
            "Not Modified",
            hdrs=HTTPMessage(),
            fp=None,
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    manager = ServiceManager(
        {
            "web": ServiceConfig(
                label="Web",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('unused')"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.HTTP,
                        url="http://127.0.0.1/health",
                        timeout_s=0.05,
                    )
                ],
            )
        }
    )
    checker = HealthChecker(manager)

    statuses = await checker.check_service("web")

    assert len(statuses) == 1
    assert statuses[0].type is CheckType.HTTP
    assert statuses[0].ok is True
    assert statuses[0].detail == "http http://127.0.0.1/health returned 304"


@pytest.mark.asyncio
async def test_http_check_reports_not_ok_for_status_500(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Response:
        status = 500

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            return None

    def urlopen(url: str, timeout: float) -> Response:
        assert url == "http://127.0.0.1/health"
        assert timeout == 0.05
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    manager = ServiceManager(
        {
            "web": ServiceConfig(
                label="Web",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('unused')"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.HTTP,
                        url="http://127.0.0.1/health",
                        timeout_s=0.05,
                    )
                ],
            )
        }
    )
    checker = HealthChecker(manager)

    statuses = await checker.check_service("web")

    assert len(statuses) == 1
    assert statuses[0].type is CheckType.HTTP
    assert statuses[0].ok is False
    assert statuses[0].detail == "http http://127.0.0.1/health returned 500"


@pytest.mark.asyncio
async def test_ready_patterns_report_ok_from_recent_logs(tmp_path: Path) -> None:
    marker = "DASHBOARD_READY_MARKER"
    manager = ServiceManager(
        {
            "logger": ServiceConfig(
                label="Logger",
                cwd=str(tmp_path),
                command=[
                    sys.executable,
                    "-u",
                    "-c",
                    (
                        "import time; "
                        f"print({marker!r}, flush=True); "
                        "time.sleep(60)"
                    ),
                ],
                ready_patterns=[marker],
            )
        }
    )
    checker = HealthChecker(manager)

    await manager.start("logger")
    try:
        await manager.wait_for_log_pattern("logger", marker, timeout_s=1.0)
        statuses = await checker.check_service("logger")
    finally:
        await manager.stop("logger")

    assert len(statuses) == 1
    assert statuses[0].type is CheckType.LOG_PATTERN
    assert statuses[0].label == marker
    assert statuses[0].ok is True
    assert statuses[0].detail == f"found log pattern: {marker}"


@pytest.mark.asyncio
async def test_exited_process_is_not_ready_when_tcp_check_matches_other_process(
    tmp_path: Path,
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    manager = ServiceManager(
        {
            "colliding": ServiceConfig(
                label="Colliding",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "raise SystemExit(7)"],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.TCP,
                        host="127.0.0.1",
                        port=port,
                        timeout_s=0.05,
                    )
                ],
                startup_timeout_s=0.3,
            )
        }
    )
    checker = HealthChecker(manager)

    try:
        await manager.start("colliding")
        with pytest.raises(TimeoutError, match="before becoming ready"):
            await checker.wait_until_ready("colliding", timeout_s=0.3)

        assert manager.status("colliding").state in {
            ServiceState.DEGRADED,
            ServiceState.FAILED,
        }
    finally:
        listener.close()
        await manager.stop_all()


@pytest.mark.asyncio
async def test_start_all_starts_in_order_and_stops_on_timeout(
    tmp_path: Path,
) -> None:
    manager = ServiceManager(
        {
            "first": ServiceConfig(
                label="First",
                cwd=str(tmp_path),
                command=_long_running_command(tmp_path, "first"),
                ready_checks=[ReadyCheckConfig(type=CheckType.PROCESS)],
                startup_timeout_s=0.3,
            ),
            "second": ServiceConfig(
                label="Second",
                cwd=str(tmp_path),
                command=_long_running_command(tmp_path, "second"),
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.TCP,
                        host="127.0.0.1",
                        port=9,
                        timeout_s=0.05,
                    )
                ],
                startup_timeout_s=0.3,
            ),
        }
    )
    checker = HealthChecker(manager)

    try:
        with pytest.raises(TimeoutError):
            await checker.start_all()

        assert (tmp_path / "start_order.txt").read_text(
            encoding="utf-8"
        ).splitlines() == [
            "first",
            "second",
        ]
        assert manager.status("first").state is ServiceState.READY
        assert manager.status("second").state in {
            ServiceState.DEGRADED,
            ServiceState.FAILED,
        }
    finally:
        await manager.stop_all()
