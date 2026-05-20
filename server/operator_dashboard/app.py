from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

from operator_dashboard.health import HealthChecker
from operator_dashboard.models import (
    CheckType,
    DashboardConfig,
    DashboardStatus,
    ServiceState,
    ServiceStatus,
)
from operator_dashboard.security import DashboardSecurity
from operator_dashboard.service_manager import ServiceManager

STATIC_DIR = Path(__file__).resolve().parent / "static"
logger = logging.getLogger(__name__)


def create_app(config: DashboardConfig, security: DashboardSecurity) -> FastAPI:
    manager = ServiceManager(config.services)
    health_checker = HealthChecker(manager)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            try:
                await manager.stop_all()
            except Exception:
                logger.exception("failed to stop dashboard services during shutdown")

    app = FastAPI(title="Operator Dashboard", lifespan=lifespan)
    app.state.manager = manager
    app.state.health_checker = health_checker

    def require_token(token: Annotated[str | None, Query()] = None) -> None:
        security.require(token)

    @app.exception_handler(PermissionError)
    async def permission_error_handler(
        request: Request, exc: PermissionError
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    async def status_with_ready_checks(service_id: str) -> ServiceStatus:
        ready_checks = await health_checker.check_service(service_id)
        service = manager._service(service_id)
        status = manager.status(service_id)
        if not service.config.require_running_process:
            required_checks = [check for check in ready_checks if check.required]
            if required_checks and all(check.ok for check in required_checks):
                service.last_error = None
                manager.set_state(service_id, ServiceState.READY)
                status = manager.status(service_id)
            elif status.state is ServiceState.READY:
                manager.set_state(service_id, ServiceState.DEGRADED)
                status = manager.status(service_id)
        return status.model_copy(update={"ready_checks": ready_checks})

    async def dashboard_status_with_ready_checks() -> DashboardStatus:
        return DashboardStatus(
            services={
                service_id: await status_with_ready_checks(service_id)
                for service_id in manager.service_ids()
            }
        )

    def service_http_error(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=str(exc))

        detail = str(exc)
        if isinstance(exc, FileNotFoundError):
            if "cwd" not in detail.lower() and "executable" not in detail.lower():
                detail = f"service executable not found: {detail}"
            return HTTPException(status_code=400, detail=detail)

        if isinstance(exc, OSError):
            return HTTPException(
                status_code=400,
                detail=f"service executable failed to start: {detail}",
            )

        return HTTPException(status_code=500, detail=detail)

    def verified_execution_base_url() -> str:
        service = config.services.get("verified_execution")
        if service is None:
            raise HTTPException(
                status_code=404,
                detail="verified_execution service is not configured",
            )

        for check in service.ready_checks:
            if check.type is CheckType.HTTP and check.url:
                parsed = urllib.parse.urlparse(check.url)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    return urllib.parse.urlunparse(
                        parsed._replace(path="", params="", query="", fragment="")
                    )

        raise HTTPException(
            status_code=400,
            detail="verified_execution service has no HTTP ready check URL",
        )

    async def post_verified_execution(
        path: str,
        payload: dict[str, object],
        *,
        timeout_s: float,
    ) -> dict:
        return await asyncio.to_thread(
            _post_json,
            f"{verified_execution_base_url().rstrip('/')}{path}",
            payload,
            timeout_s,
        )

    async def wait_for_ready_status(service_id: str) -> ServiceStatus:
        service = manager._service(service_id)
        try:
            await health_checker.wait_until_ready(
                service_id, service.config.startup_timeout_s
            )
        except TimeoutError:
            return await status_with_ready_checks(service_id)
        return await status_with_ready_checks(service_id)

    @app.get("/", response_class=FileResponse)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])

    @api.get("/status", response_model=DashboardStatus)
    async def status() -> DashboardStatus:
        return await dashboard_status_with_ready_checks()

    @api.post("/services/{service_id}/start", response_model=ServiceStatus)
    async def start_service(service_id: str) -> ServiceStatus:
        try:
            await manager.start(service_id)
            return await wait_for_ready_status(service_id)
        except (KeyError, FileNotFoundError, OSError) as exc:
            raise service_http_error(exc) from exc

    @api.post("/services/{service_id}/stop", response_model=ServiceStatus)
    async def stop_service(service_id: str) -> ServiceStatus:
        try:
            await manager.stop(service_id)
            return await status_with_ready_checks(service_id)
        except KeyError as exc:
            raise service_http_error(exc) from exc

    @api.post("/services/{service_id}/restart", response_model=ServiceStatus)
    async def restart_service(service_id: str) -> ServiceStatus:
        try:
            await manager.restart(service_id)
            return await wait_for_ready_status(service_id)
        except (KeyError, FileNotFoundError, OSError) as exc:
            raise service_http_error(exc) from exc

    @api.post("/start-all", response_model=DashboardStatus)
    async def start_all() -> DashboardStatus:
        try:
            for service_id in manager.global_action_service_ids():
                await manager.start(service_id)
                status = await wait_for_ready_status(service_id)
                if status.state is ServiceState.DEGRADED:
                    break
        except (KeyError, FileNotFoundError, OSError) as exc:
            raise service_http_error(exc) from exc
        return await dashboard_status_with_ready_checks()

    @api.post("/stop-all", response_model=DashboardStatus)
    async def stop_all() -> DashboardStatus:
        await manager.stop_all(manager.global_action_service_ids())
        return await dashboard_status_with_ready_checks()

    @api.post("/robot/home")
    async def home_robot() -> dict:
        return await post_verified_execution(
            "/home",
            {"robot_name": "UR10", "timeout_s": 60.0},
            timeout_s=61.0,
        )

    @api.post("/robot/sync-state")
    async def sync_robot_state() -> dict:
        return await post_verified_execution(
            "/sync_state",
            {"robot_name": "UR10", "timeout_s": 10.0},
            timeout_s=11.0,
        )

    @api.post("/robot/gripper/{action}")
    async def control_gripper(action: str) -> dict:
        if action not in {"open", "close"}:
            raise HTTPException(status_code=422, detail="gripper action must be open or close")
        return await post_verified_execution(
            f"/gripper/{action}",
            {"robot_name": "UR10", "timeout_s": 10.0},
            timeout_s=11.0,
        )

    app.include_router(api)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    return app


def _post_json(url: str, payload: dict[str, object], timeout_s: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return _json_body(response.read())
    except urllib.error.HTTPError as exc:
        raise HTTPException(status_code=exc.code, detail=_verified_error_detail(exc)) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"verified execution server unavailable: {exc.reason}",
        ) from exc


def _json_body(raw: bytes) -> dict:
    if not raw:
        return {}
    try:
        body = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError:
        return {"raw": raw.decode("utf-8", errors="replace")}
    return body if isinstance(body, dict) else {"value": body}


def _verified_error_detail(exc: urllib.error.HTTPError) -> object:
    body = _json_body(exc.read())
    if "detail" in body:
        return body["detail"]
    if body:
        return body
    return exc.reason
