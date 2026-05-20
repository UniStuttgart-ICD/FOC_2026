from __future__ import annotations

import asyncio
import os
import re
import signal
import socket
import subprocess
import sys
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import psutil

from operator_dashboard.models import (
    AUXILIARY_SERVICE_IDS,
    DashboardStatus,
    ServiceConfig,
    ServiceState,
    ServiceStatus,
)

_MAX_LOG_LINES = 500
_GRACEFUL_STOP_TIMEOUT_S = 3.0
_STOP_TIMEOUT_S = 10.0
_REAP_TIMEOUT_S = 1.0
_READER_DRAIN_TIMEOUT_S = 1.0
_PORT_RELEASE_TIMEOUT_S = 5.0
_PORT_RELEASE_POLL_S = 0.1
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


@dataclass
class ManagedService:
    config: ServiceConfig
    process: asyncio.subprocess.Process | None = None
    state: ServiceState = ServiceState.STOPPED
    logs: deque[str] = field(default_factory=lambda: deque(maxlen=_MAX_LOG_LINES))
    last_exit_code: int | None = None
    last_error: str | None = None
    detected_urls: list[str] = field(default_factory=list)
    reader_tasks: list[asyncio.Task[None]] = field(default_factory=list)
    log_reader_generation: int = 0
    lifecycle_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def is_running(self) -> bool:
        return self.process is not None and self.process.returncode is None


class ServiceManager:
    def __init__(self, services: dict[str, ServiceConfig]) -> None:
        self._services = {
            service_id: ManagedService(config=config)
            for service_id, config in services.items()
        }

    def service_ids(self) -> list[str]:
        return list(self._services)

    def global_action_service_ids(self) -> list[str]:
        return [
            service_id
            for service_id, service in self._services.items()
            if self._include_in_global_actions(service_id, service)
        ]

    def status(self, service_id: str) -> ServiceStatus:
        service = self._service(service_id)
        return self._status(service_id, service)

    def all_statuses(self) -> DashboardStatus:
        return DashboardStatus(
            services={
                service_id: self._status(service_id, service)
                for service_id, service in self._services.items()
            }
        )

    async def start(self, service_id: str) -> ServiceStatus:
        service = self._service(service_id)
        async with service.lifecycle_lock:
            return await self._start_unlocked(service_id, service)

    async def stop(self, service_id: str) -> ServiceStatus:
        service = self._service(service_id)
        async with service.lifecycle_lock:
            return await self._stop_unlocked(service_id, service)

    async def restart(self, service_id: str) -> ServiceStatus:
        service = self._service(service_id)
        async with service.lifecycle_lock:
            await self._stop_unlocked(service_id, service)
            return await self._start_unlocked(service_id, service)

    async def stop_all(self, service_ids: Iterable[str] | None = None) -> None:
        ids = list(service_ids) if service_ids is not None else self.service_ids()
        await asyncio.gather(*(self.stop(service_id) for service_id in reversed(ids)))

    async def wait_for_log_pattern(
        self, service_id: str, pattern: str, timeout_s: float
    ) -> bool:
        service = self._service(service_id)
        deadline = asyncio.get_running_loop().time() + timeout_s
        while True:
            if any(pattern in line for line in service.logs):
                return True
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(
                    f"timed out after {timeout_s:g}s waiting for log pattern: {pattern}"
                )
            await asyncio.sleep(0.05)

    def set_state(self, service_id: str, state: ServiceState) -> None:
        self._service(service_id).state = state

    def _service(self, service_id: str) -> ManagedService:
        try:
            return self._services[service_id]
        except KeyError as exc:
            raise KeyError(f"unknown service: {service_id}") from exc

    async def _start_unlocked(
        self, service_id: str, service: ManagedService
    ) -> ServiceStatus:
        if service.is_running():
            return self._status(service_id, service)

        cwd = Path(service.config.cwd)
        if not cwd.exists():
            service.state = ServiceState.FAILED
            service.last_error = f"service cwd does not exist: {cwd}"
            raise FileNotFoundError(service.last_error)

        await self._cancel_reader_tasks(service)
        service.log_reader_generation += 1
        service.logs.clear()
        service.detected_urls.clear()
        service.last_error = None
        service.last_exit_code = None
        service.state = ServiceState.STARTING

        kwargs: dict[str, Any] = {
            "cwd": str(cwd),
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "env": self._subprocess_env(service.config.env),
        }
        kwargs.update(self._subprocess_process_group_kwargs())

        try:
            process = await asyncio.create_subprocess_exec(
                *service.config.command,
                **cast(Any, kwargs),
            )
        except Exception as exc:
            service.state = ServiceState.FAILED
            service.last_error = str(exc)
            raise

        service.process = process
        service.reader_tasks.append(
            asyncio.create_task(
                self._read_logs(
                    service_id,
                    service,
                    process,
                    service.log_reader_generation,
                )
            )
        )
        return self._status(service_id, service)

    async def _stop_unlocked(
        self, service_id: str, service: ManagedService
    ) -> ServiceStatus:
        service.state = ServiceState.STOPPING

        if service.config.stop_command:
            await self._run_stop_command(service)

        process = service.process
        if process is not None and process.returncode is None:
            if not await self._stop_process_tree(process):
                await self._drain_or_cancel_reader_tasks(service)
                self._close_process_transport(process)
                service.state = ServiceState.FAILED
                service.last_error = (
                    f"timed out after {_STOP_TIMEOUT_S:g}s stopping service"
                )
                return self._status(service_id, service)

        if process is not None:
            service.last_exit_code = process.returncode

        await self._drain_or_cancel_reader_tasks(service)
        if process is not None:
            self._close_process_transport(process)

        service.process = None
        service.state = ServiceState.STOPPED
        leaked_ports = await self._wait_for_service_ports_to_close(service)
        if leaked_ports:
            service.state = ServiceState.FAILED
            service.last_error = (
                "ports still listening after stop: " + ", ".join(leaked_ports)
            )
            self._append_log(service, service.last_error)
        return self._status(service_id, service)

    def _status(self, service_id: str, service: ManagedService) -> ServiceStatus:
        process = service.process
        pid = (
            process.pid if process is not None and process.returncode is None else None
        )
        return ServiceStatus(
            id=service_id,
            label=service.config.label,
            state=service.state,
            pid=pid,
            last_exit_code=service.last_exit_code,
            command=list(service.config.command),
            links=list(service.config.links),
            recent_logs=list(service.logs),
            last_error=service.last_error,
            detected_urls=list(service.detected_urls),
            include_in_global_actions=self._include_in_global_actions(
                service_id, service
            ),
        )

    def _include_in_global_actions(
        self, service_id: str, service: ManagedService
    ) -> bool:
        return (
            service.config.include_in_global_actions
            and service_id not in AUXILIARY_SERVICE_IDS
        )

    async def _read_logs(
        self,
        service_id: str,
        service: ManagedService,
        process: asyncio.subprocess.Process,
        generation: int,
    ) -> None:
        if process.stdout is None:
            return

        while True:
            chunk = await process.stdout.readline()
            if not chunk:
                break
            if not self._is_current_reader(service, process, generation):
                break
            line = chunk.decode(errors="replace").rstrip("\r\n")
            self._append_log(service, line)
            self._detect_urls(service, line)
            if (
                service.state is ServiceState.STARTING
                and service.config.ready_patterns
                and any(pattern in line for pattern in service.config.ready_patterns)
            ):
                service.state = ServiceState.READY

        returncode = await process.wait()
        if not self._is_current_reader(service, process, generation):
            return

        service.last_exit_code = returncode
        if (
            returncode != 0
            and service.config.require_running_process
            and service.state not in {
                ServiceState.STOPPING,
                ServiceState.STOPPED,
            }
        ):
            service.state = ServiceState.FAILED
            service.last_error = (
                service.last_error
                or f"service exited with code {returncode}: {service_id}"
            )
        elif (
            returncode == 0
            and service.config.require_running_process
            and service.state not in {
                ServiceState.STOPPING,
                ServiceState.STOPPED,
            }
        ):
            service.state = ServiceState.STOPPED

    async def _cancel_reader_tasks(self, service: ManagedService) -> None:
        if not service.reader_tasks:
            return

        for task in service.reader_tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*service.reader_tasks, return_exceptions=True)
        service.reader_tasks.clear()

    async def _drain_or_cancel_reader_tasks(self, service: ManagedService) -> None:
        if not service.reader_tasks:
            return

        try:
            await asyncio.wait_for(
                asyncio.gather(*service.reader_tasks, return_exceptions=True),
                timeout=_READER_DRAIN_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            for task in service.reader_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*service.reader_tasks, return_exceptions=True)
        finally:
            service.reader_tasks.clear()

    def _is_current_reader(
        self,
        service: ManagedService,
        process: asyncio.subprocess.Process,
        generation: int,
    ) -> bool:
        return (
            service.process is process and service.log_reader_generation == generation
        )

    def _subprocess_env(
        self, service_env: dict[str, str] | None = None
    ) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"
        env.pop("VIRTUAL_ENV", None)
        if service_env:
            env.update(service_env)
        return env

    def _subprocess_process_group_kwargs(self) -> dict[str, object]:
        if sys.platform == "win32":
            return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        return {"start_new_session": True}

    async def _run_stop_command(self, service: ManagedService) -> None:
        stop_command = service.config.stop_command
        if not stop_command:
            return

        kwargs: dict[str, Any] = {
            "cwd": service.config.cwd,
            "stdout": asyncio.subprocess.PIPE,
            "stderr": asyncio.subprocess.STDOUT,
            "env": self._subprocess_env(service.config.env),
        }
        kwargs.update(self._subprocess_process_group_kwargs())

        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *stop_command,
                **cast(Any, kwargs),
            )
            stdout, _ = await asyncio.wait_for(
                process.communicate(), timeout=_STOP_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            if process is not None:
                self._terminate_process_tree(process.pid)
                try:
                    await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_S)
                except asyncio.TimeoutError:
                    self._close_process_transport(process)
                    self._append_log(
                        service, "stop command timed out; cleanup did not exit"
                    )
                    return
                self._close_process_transport(process)
            self._append_log(service, "stop command timed out")
            return
        except Exception as exc:
            self._append_log(service, f"stop command failed: {exc}")
            return

        if process is not None:
            self._close_process_transport(process)
        if stdout:
            for raw_line in stdout.decode(errors="replace").splitlines():
                self._append_log(service, raw_line)
                self._detect_urls(service, raw_line)

    def _close_process_transport(self, process: asyncio.subprocess.Process) -> None:
        transport = getattr(process, "_transport", None)
        close = getattr(transport, "close", None)
        if callable(close):
            close()

    async def _stop_process_tree(self, process: asyncio.subprocess.Process) -> bool:
        if process.returncode is not None:
            return True

        if self._request_graceful_process_group_stop(process.pid):
            try:
                await asyncio.wait_for(
                    process.wait(), timeout=_GRACEFUL_STOP_TIMEOUT_S
                )
                return True
            except asyncio.TimeoutError:
                pass

        self._terminate_process_tree(process.pid)
        try:
            await asyncio.wait_for(process.wait(), timeout=_REAP_TIMEOUT_S)
        except asyncio.TimeoutError:
            return False
        return True

    def _request_graceful_process_group_stop(self, pid: int) -> bool:
        try:
            if sys.platform == "win32":
                os.kill(pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(pid, signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            return False
        return True

    def _terminate_process_tree(self, pid: int) -> None:
        try:
            root = psutil.Process(pid)
        except psutil.NoSuchProcess:
            return

        try:
            processes = root.children(recursive=True) + [root]
        except psutil.NoSuchProcess:
            return

        for process in processes:
            try:
                process.terminate()
            except psutil.NoSuchProcess:
                continue

        _, alive = psutil.wait_procs(processes, timeout=_STOP_TIMEOUT_S)
        for process in alive:
            try:
                process.kill()
            except psutil.NoSuchProcess:
                continue

        if alive:
            psutil.wait_procs(alive, timeout=_REAP_TIMEOUT_S)

    async def _wait_for_service_ports_to_close(
        self, service: ManagedService
    ) -> list[str]:
        targets = self._service_ready_ports(service)
        if not targets:
            return []

        deadline = asyncio.get_running_loop().time() + _PORT_RELEASE_TIMEOUT_S
        open_targets: list[tuple[str, int]] = []
        while True:
            open_targets = await self._open_tcp_targets(targets)
            if not open_targets:
                return []
            if asyncio.get_running_loop().time() >= deadline:
                return [self._format_target(host, port) for host, port in open_targets]
            await asyncio.sleep(_PORT_RELEASE_POLL_S)

    async def _open_tcp_targets(
        self, targets: list[tuple[str, int]]
    ) -> list[tuple[str, int]]:
        results = await asyncio.gather(
            *(
                asyncio.to_thread(self._tcp_port_is_open, host, port)
                for host, port in targets
            )
        )
        return [
            target
            for target, is_open in zip(targets, results, strict=True)
            if is_open
        ]

    def _service_ready_ports(self, service: ManagedService) -> list[tuple[str, int]]:
        targets: list[tuple[str, int]] = []
        for check in service.config.ready_checks:
            if check.type.value == "tcp" and check.host and check.port is not None:
                targets.append((check.host, check.port))
            elif check.type.value == "http" and check.url:
                target = self._http_url_target(check.url)
                if target is not None:
                    targets.append(target)
        return list(dict.fromkeys(targets))

    def _http_url_target(self, url: str) -> tuple[str, int] | None:
        parsed = urlparse(url)
        if not parsed.hostname:
            return None
        if parsed.port is not None:
            return parsed.hostname, parsed.port
        if parsed.scheme == "http":
            return parsed.hostname, 80
        if parsed.scheme == "https":
            return parsed.hostname, 443
        return None

    def _tcp_port_is_open(self, host: str, port: int) -> bool:
        try:
            if self._tcp_port_is_listening(host, port):
                return True
        except (OSError, psutil.Error):
            pass

        try:
            with socket.create_connection((host, port), timeout=0.1):
                return True
        except OSError:
            return False

    def _tcp_port_is_listening(self, host: str, port: int) -> bool:
        host_addresses = self._resolve_host_addresses(host)
        for connection in psutil.net_connections(kind="inet"):
            if connection.status != psutil.CONN_LISTEN:
                continue
            local_address = connection.laddr
            if not local_address or local_address.port != port:
                continue
            if self._address_matches_host(local_address.ip, host_addresses):
                return True
        return False

    def _resolve_host_addresses(self, host: str) -> set[str]:
        addresses = {host}
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            if family in {socket.AF_INET, socket.AF_INET6}:
                addresses.add(str(sockaddr[0]))
        return addresses

    def _address_matches_host(self, address: str, host_addresses: set[str]) -> bool:
        return address in {"0.0.0.0", "::"} or address in host_addresses

    def _format_target(self, host: str, port: int) -> str:
        return f"{host}:{port}"

    def _append_log(self, service: ManagedService, line: str) -> None:
        service.logs.append(line)

    def _detect_urls(self, service: ManagedService, line: str) -> None:
        for url in _URL_RE.findall(line):
            if url not in service.detected_urls:
                service.detected_urls.append(url)
