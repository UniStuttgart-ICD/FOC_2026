from __future__ import annotations

import asyncio
import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterable

from operator_dashboard.models import (
    CheckType,
    ReadyCheckConfig,
    ReadyCheckStatus,
    ServiceState,
)
from operator_dashboard.service_manager import ManagedService, ServiceManager

_POLL_INTERVAL_S = 0.25
_PROCESS_EXIT_POLL_S = 0.02
_READY_EXIT_SETTLE_S = 0.15


class HealthChecker:
    def __init__(self, manager: ServiceManager) -> None:
        self.manager = manager

    async def check_service(self, service_id: str) -> list[ReadyCheckStatus]:
        service = self.manager._service(service_id)
        checks = list(service.config.ready_checks)
        checks.extend(
            ReadyCheckConfig(
                type=CheckType.LOG_PATTERN,
                pattern=pattern,
                label=pattern,
            )
            for pattern in service.config.ready_patterns
        )
        return [await self._check(service, check) for check in checks]

    async def wait_until_ready(
        self, service_id: str, timeout_s: float
    ) -> list[ReadyCheckStatus]:
        deadline = asyncio.get_running_loop().time() + timeout_s
        statuses: list[ReadyCheckStatus] = []

        while True:
            service = self.manager._service(service_id)
            await self._raise_if_process_exited_before_ready(service_id, service)

            statuses = await self.check_service(service_id)
            if all(status.ok for status in statuses):
                await self._raise_if_process_exited_before_ready(
                    service_id, service, timeout_s=_READY_EXIT_SETTLE_S
                )
                self.manager.set_state(service_id, ServiceState.READY)
                return statuses

            if asyncio.get_running_loop().time() >= deadline:
                self.manager.set_state(service_id, ServiceState.DEGRADED)
                raise TimeoutError(
                    f"timed out after {timeout_s:g}s waiting for {service_id} "
                    f"ready checks: {self._format_statuses(statuses)}"
                )

            await asyncio.sleep(_POLL_INTERVAL_S)

    async def _raise_if_process_exited_before_ready(
        self,
        service_id: str,
        service: ManagedService,
        timeout_s: float = _PROCESS_EXIT_POLL_S,
    ) -> None:
        process = service.process
        if process is None or service.state is ServiceState.READY:
            return
        if process.returncode is None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(process.wait()), timeout=timeout_s
                )
            except asyncio.TimeoutError:
                return

        service.last_exit_code = process.returncode
        if process.returncode == 0:
            service.state = ServiceState.DEGRADED
            service.last_error = f"service exited before becoming ready: {service_id}"
        else:
            service.state = ServiceState.FAILED
            service.last_error = f"service exited with code {process.returncode} before becoming ready: {service_id}"
        raise TimeoutError(service.last_error)

    async def start_all(
        self, service_ids: Iterable[str] | None = None
    ) -> dict[str, list[ReadyCheckStatus]]:
        ids = (
            list(service_ids) if service_ids is not None else self.manager.service_ids()
        )
        ready_statuses: dict[str, list[ReadyCheckStatus]] = {}

        for service_id in ids:
            await self.manager.start(service_id)
            service = self.manager._service(service_id)
            ready_statuses[service_id] = await self.wait_until_ready(
                service_id, service.config.startup_timeout_s
            )

        return ready_statuses

    async def _check(
        self, service: ManagedService, check: ReadyCheckConfig
    ) -> ReadyCheckStatus:
        if check.type is CheckType.PROCESS:
            return self._check_process(service, check)
        if check.type is CheckType.LOG_PATTERN:
            return self._check_log_pattern(service, check)
        if check.type is CheckType.TCP:
            return await self._check_tcp(check)
        if check.type is CheckType.HTTP:
            return await self._check_http(check)
        raise ValueError(f"unsupported ready check type: {check.type}")

    def _check_process(
        self, service: ManagedService, check: ReadyCheckConfig
    ) -> ReadyCheckStatus:
        process = service.process
        ok = process is not None and process.returncode is None
        if process is None:
            detail = "process not started"
        elif process.returncode is None:
            detail = f"process running with pid {process.pid}"
        else:
            detail = f"process exited with code {process.returncode}"

        return ReadyCheckStatus(
            type=check.type,
            label=self._label(check),
            ok=ok,
            detail=detail,
        )

    def _check_log_pattern(
        self, service: ManagedService, check: ReadyCheckConfig
    ) -> ReadyCheckStatus:
        pattern = check.pattern or ""
        ok = bool(pattern) and any(pattern in line for line in service.logs)
        detail = (
            f"found log pattern: {pattern}" if ok else f"missing log pattern: {pattern}"
        )
        return ReadyCheckStatus(
            type=check.type,
            label=self._label(check),
            ok=ok,
            detail=detail,
        )

    async def _check_tcp(self, check: ReadyCheckConfig) -> ReadyCheckStatus:
        host = check.host or ""
        port = check.port or 0
        target = f"{host}:{port}"
        try:
            await asyncio.to_thread(self._open_tcp, host, port, check.timeout_s)
        except Exception as exc:
            return ReadyCheckStatus(
                type=check.type,
                label=self._label(check),
                ok=False,
                detail=f"tcp {target} unavailable: {exc}",
            )

        return ReadyCheckStatus(
            type=check.type,
            label=self._label(check),
            ok=True,
            detail=f"tcp {target} reachable",
        )

    async def _check_http(self, check: ReadyCheckConfig) -> ReadyCheckStatus:
        url = check.url or ""
        try:
            response_status, response_body = await asyncio.to_thread(
                self._open_http, url, check.timeout_s
            )
        except urllib.error.HTTPError as exc:
            response_status = exc.code
            response_body = None
        except Exception as exc:
            return ReadyCheckStatus(
                type=check.type,
                label=self._label(check),
                ok=False,
                detail=f"http {url} unavailable: {exc}",
            )

        return ReadyCheckStatus(
            type=check.type,
            label=self._label(check),
            ok=200 <= response_status < 400,
            detail=self._http_detail(url, response_status, response_body),
        )

    def _open_tcp(self, host: str, port: int, timeout_s: float) -> None:
        with socket.create_connection((host, port), timeout_s):
            return

    def _open_http(self, url: str, timeout_s: float) -> tuple[int, dict | None]:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            body = None
            read = getattr(response, "read", None)
            if callable(read):
                raw = read()
                if isinstance(raw, bytes):
                    body = self._json_body(raw)
            return response.status, body

    def _json_body(self, raw: bytes) -> dict | None:
        if not raw:
            return None
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return body if isinstance(body, dict) else None

    def _http_detail(
        self,
        url: str,
        response_status: int,
        response_body: dict | None,
    ) -> str:
        detail = f"http {url} returned {response_status}"
        if not response_body:
            return detail
        robot = response_body.get("robot")
        if not isinstance(robot, dict):
            return detail

        robot_label = "robot ready" if robot.get("robot_connected") else "robot unavailable"
        gripper_label = (
            "gripper ready"
            if robot.get("gripper_connected")
            else "gripper unavailable"
        )
        return f"{detail}; {robot_label}; {gripper_label}"

    def _label(self, check: ReadyCheckConfig) -> str:
        return check.label or check.type.value

    def _format_statuses(self, statuses: list[ReadyCheckStatus]) -> str:
        if not statuses:
            return "no checks configured"
        return "; ".join(
            f"{status.label}={'ok' if status.ok else 'failed'} ({status.detail})"
            for status in statuses
        )
