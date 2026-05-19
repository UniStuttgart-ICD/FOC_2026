from __future__ import annotations

import asyncio
import os
import socket
import sys
import time
from pathlib import Path

import psutil
import pytest

import operator_dashboard.service_manager as service_manager_module
from operator_dashboard.models import (
    CheckType,
    ReadyCheckConfig,
    ServiceConfig,
    ServiceState,
)
from operator_dashboard.service_manager import ServiceManager


async def _force_cleanup_service(manager: ServiceManager, service_id: str) -> None:
    service = manager._service(service_id)
    process = service.process
    if process is not None and process.returncode is None:
        manager._terminate_process_tree(process.pid)
        await asyncio.wait_for(process.wait(), timeout=1.0)
    if service.reader_tasks:
        await asyncio.gather(*service.reader_tasks, return_exceptions=True)


def test_all_statuses_wraps_services_and_service_ids_returns_ids(
    tmp_path: Path,
) -> None:
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    assert "fake" in manager.all_statuses().services
    assert manager.service_ids() == ["fake"]


def test_known_auxiliary_service_ids_are_never_global_actions(tmp_path: Path) -> None:
    manager = ServiceManager(
        {
            "moveit_mcp": ServiceConfig(
                label="MoveIt MCP",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('core')"],
            ),
            "wake_tuning": ServiceConfig(
                label="Wake Word Tuning",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('wake')"],
                include_in_global_actions=True,
            ),
            "voice_modulation": ServiceConfig(
                label="Agent Persona Lab",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('voice')"],
                include_in_global_actions=True,
            ),
        }
    )

    assert manager.global_action_service_ids() == ["moveit_mcp"]
    assert manager.status("wake_tuning").include_in_global_actions is False
    assert manager.status("voice_modulation").include_in_global_actions is False


@pytest.mark.asyncio
async def test_stop_all_stops_services_concurrently(tmp_path: Path) -> None:
    manager = ServiceManager(
        {
            "first": ServiceConfig(
                label="First Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('first')"],
            ),
            "second": ServiceConfig(
                label="Second Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('second')"],
            ),
        }
    )
    stopped: list[str] = []

    async def fake_stop(service_id: str) -> None:
        stopped.append(service_id)
        await asyncio.sleep(0.2)

    manager.stop = fake_stop  # type: ignore[method-assign]

    start = time.monotonic()
    await manager.stop_all()

    assert time.monotonic() - start < 0.35
    assert set(stopped) == {"first", "second"}


@pytest.mark.asyncio
async def test_stop_process_tree_requests_graceful_group_stop_before_kill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 1234
        returncode: int | None = None

        async def wait(self) -> int:
            self.returncode = 0
            return 0

    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('fake')"],
            )
        }
    )
    requested_pids: list[int] = []

    monkeypatch.setattr(
        manager,
        "_request_graceful_process_group_stop",
        lambda pid: requested_pids.append(pid) or True,
    )
    monkeypatch.setattr(
        manager,
        "_terminate_process_tree",
        lambda pid: pytest.fail("hard kill should be a fallback"),
    )

    assert await manager._stop_process_tree(FakeProcess())
    assert requested_pids == [1234]


@pytest.mark.asyncio
async def test_stop_cancels_reader_task_when_stdout_pipe_stays_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 1235
        returncode: int | None = None

    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('fake')"],
            )
        }
    )
    service = manager._service("fake")
    process = FakeProcess()
    service.process = process  # type: ignore[assignment]

    async def never_finishes() -> None:
        await asyncio.Event().wait()

    reader_task = asyncio.create_task(never_finishes())
    service.reader_tasks.append(reader_task)

    async def fake_stop_process_tree(process: FakeProcess) -> bool:
        process.returncode = 0
        return True

    monkeypatch.setattr(manager, "_stop_process_tree", fake_stop_process_tree)

    status = await asyncio.wait_for(manager.stop("fake"), timeout=2.0)

    assert status.state is ServiceState.STOPPED
    assert service.reader_tasks == []
    assert reader_task.cancelled()


@pytest.mark.asyncio
async def test_stop_closes_asyncio_subprocess_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeTransport:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    class FakeProcess:
        pid = 1236
        returncode: int | None = None

        def __init__(self) -> None:
            self._transport = FakeTransport()

    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('fake')"],
            )
        }
    )
    service = manager._service("fake")
    process = FakeProcess()
    service.process = process  # type: ignore[assignment]

    async def fake_stop_process_tree(process: FakeProcess) -> bool:
        process.returncode = 0
        return True

    monkeypatch.setattr(manager, "_stop_process_tree", fake_stop_process_tree)

    await manager.stop("fake")

    assert process._transport.close_calls == 1


@pytest.mark.asyncio
async def test_wait_for_log_pattern_raises_on_timeout(tmp_path: Path) -> None:
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    with pytest.raises(TimeoutError):
        await manager.wait_for_log_pattern("fake", "missing", timeout_s=0.01)


@pytest.mark.asyncio
async def test_concurrent_start_only_creates_one_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode = None
            self.stdout = None

        async def wait(self) -> int:
            return 0

    created_processes: list[FakeProcess] = []

    async def fake_create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> FakeProcess:
        await asyncio.sleep(0)
        process = FakeProcess(pid=1000 + len(created_processes))
        created_processes.append(process)
        return process

    monkeypatch.setattr(
        service_manager_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    first_status, second_status = await asyncio.gather(
        manager.start("fake"), manager.start("fake")
    )

    assert len(created_processes) == 1
    assert first_status.pid == created_processes[0].pid
    assert second_status.pid == created_processes[0].pid
    assert manager.status("fake").pid == created_processes[0].pid


@pytest.mark.asyncio
async def test_stale_log_reader_does_not_fail_replacement_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeStdout:
        def __init__(self) -> None:
            self._lines: asyncio.Queue[bytes] = asyncio.Queue()

        async def readline(self) -> bytes:
            return await self._lines.get()

        def feed_eof(self) -> None:
            self._lines.put_nowait(b"")

    class FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid
            self.returncode: int | None = None
            self.stdout = FakeStdout()

        async def wait(self) -> int:
            return self.returncode or 0

    created_processes: list[FakeProcess] = []

    async def fake_create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> FakeProcess:
        process = FakeProcess(pid=2000 + len(created_processes))
        created_processes.append(process)
        return process

    monkeypatch.setattr(
        service_manager_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    await manager.start("fake")
    service = manager._service("fake")
    old_process = created_processes[0]
    old_reader_task = service.reader_tasks[0]
    await asyncio.sleep(0)

    old_process.returncode = 9
    await manager.start("fake")
    new_process = created_processes[1]

    old_process.stdout.feed_eof()
    try:
        await asyncio.wait_for(
            asyncio.gather(old_reader_task, return_exceptions=True), timeout=1.0
        )
        status = manager.status("fake")

        assert status.pid == new_process.pid
        assert status.state is not ServiceState.FAILED
        assert status.last_exit_code is None
        assert status.last_error is None
    finally:
        for task in service.reader_tasks:
            task.cancel()
        await asyncio.gather(*service.reader_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_start_sets_utf8_python_environment_for_child_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 2222
        returncode = None
        stdout = None

        async def wait(self) -> int:
            return 0

    captured_env: dict[str, str] = {}

    async def fake_create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> FakeProcess:
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(
        service_manager_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    await manager.start("fake")

    assert captured_env["PYTHONIOENCODING"] == "utf-8"
    assert captured_env["PYTHONUTF8"] == "1"
    if "VIRTUAL_ENV" in os.environ:
        assert "VIRTUAL_ENV" not in captured_env


@pytest.mark.asyncio
async def test_start_merges_service_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeProcess:
        pid = 3333
        returncode = None
        stdout = None

        async def wait(self) -> int:
            return 0

    captured_env: dict[str, str] = {}

    async def fake_create_subprocess_exec(
        *args: object, **kwargs: object
    ) -> FakeProcess:
        captured_env.update(kwargs["env"])
        return FakeProcess()

    monkeypatch.setattr(
        service_manager_module.asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-c", "print('never')"],
                env={"MCP_VIZOR_URL": "http://127.0.0.1:8001/mcp"},
            )
        }
    )

    await manager.start("fake")

    assert captured_env["MCP_VIZOR_URL"] == "http://127.0.0.1:8001/mcp"


@pytest.mark.asyncio
async def test_start_captures_ready_logs_and_stop_marks_stopped(
    tmp_path: Path,
) -> None:
    script = tmp_path / "fake_service.py"
    script.write_text(
        """
import time

print('ready marker http://127.0.0.1:4321', flush=True)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-u", str(script)],
            )
        }
    )

    await manager.start("fake")
    try:
        assert await manager.wait_for_log_pattern("fake", "ready marker", timeout_s=5.0)

        status = manager.status("fake")
        assert status.state in {ServiceState.STARTING, ServiceState.READY}
        assert status.pid is not None
        assert any("ready marker" in line for line in status.recent_logs)
        assert "http://127.0.0.1:4321" in status.detected_urls
        assert "fake" in manager.all_statuses().services
    finally:
        await manager.stop("fake")

    assert manager.status("fake").state is ServiceState.STOPPED


@pytest.mark.asyncio
async def test_stop_marks_failed_when_ready_port_remains_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = listener.getsockname()[1]
    monkeypatch.setattr(
        service_manager_module, "_PORT_RELEASE_TIMEOUT_S", 0.2, raising=False
    )

    script = tmp_path / "long_running.py"
    script.write_text(
        """
import time

while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    manager = ServiceManager(
        {
            "leaky": ServiceConfig(
                label="Leaky Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-u", str(script)],
                ready_checks=[
                    ReadyCheckConfig(
                        type=CheckType.TCP,
                        host="127.0.0.1",
                        port=port,
                        timeout_s=0.01,
                    )
                ],
            )
        }
    )

    await manager.start("leaky")
    try:
        status = await manager.stop("leaky")
    finally:
        listener.close()
        await _force_cleanup_service(manager, "leaky")

    assert status.state is ServiceState.FAILED
    assert status.last_error is not None
    assert f"127.0.0.1:{port}" in status.last_error


@pytest.mark.asyncio
async def test_stop_command_timeout_cleans_up_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service_script = tmp_path / "long_running_service.py"
    service_script.write_text(
        """
import time

while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    stop_pid_file = tmp_path / "stop_command.pid"
    stop_script = tmp_path / "hung_stop_command.py"
    stop_script.write_text(
        """
import os
import sys
import time
from pathlib import Path

Path(sys.argv[1]).write_text(str(os.getpid()), encoding="utf-8")
print("hung stop command started", flush=True)
time.sleep(60)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(service_manager_module, "_STOP_TIMEOUT_S", 0.2)
    manager = ServiceManager(
        {
            "fake": ServiceConfig(
                label="Fake Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-u", str(service_script)],
                stop_command=[
                    sys.executable,
                    "-u",
                    str(stop_script),
                    str(stop_pid_file),
                ],
            )
        }
    )

    await manager.start("fake")
    try:
        status = await manager.stop("fake")
    finally:
        await _force_cleanup_service(manager, "fake")

    stop_pid = int(stop_pid_file.read_text(encoding="utf-8"))
    assert status.state is ServiceState.STOPPED
    assert "stop command timed out" in status.recent_logs
    assert not psutil.pid_exists(stop_pid)


@pytest.mark.asyncio
async def test_start_rejects_missing_cwd_and_sets_failed(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"
    manager = ServiceManager(
        {
            "bad": ServiceConfig(
                label="Bad Service",
                cwd=str(missing_cwd),
                command=[sys.executable, "-c", "print('never')"],
            )
        }
    )

    with pytest.raises(FileNotFoundError):
        await manager.start("bad")

    status = manager.status("bad")
    assert status.state is ServiceState.FAILED
    assert status.last_error is not None
    assert str(missing_cwd) in status.last_error


@pytest.mark.asyncio
async def test_restart_stops_then_starts_and_changes_pid(tmp_path: Path) -> None:
    script = tmp_path / "restart_service.py"
    script.write_text(
        """
import os
import time

print(f'pid marker {os.getpid()}', flush=True)
while True:
    time.sleep(0.1)
""",
        encoding="utf-8",
    )
    manager = ServiceManager(
        {
            "restartable": ServiceConfig(
                label="Restartable Service",
                cwd=str(tmp_path),
                command=[sys.executable, "-u", str(script)],
            )
        }
    )

    await manager.start("restartable")
    try:
        assert await manager.wait_for_log_pattern(
            "restartable", "pid marker", timeout_s=5.0
        )
        first_pid = manager.status("restartable").pid
        assert first_pid is not None

        await manager.restart("restartable")
        assert await manager.wait_for_log_pattern(
            "restartable", "pid marker", timeout_s=5.0
        )
        second_pid = manager.status("restartable").pid

        assert second_pid is not None
        assert second_pid != first_pid
    finally:
        await manager.stop("restartable")
