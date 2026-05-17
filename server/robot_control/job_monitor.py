from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from loguru import logger

from robot_control.job_board import RobotJobBoard


def create_app(board: RobotJobBoard) -> Any:
    from fastapi import FastAPI, Query
    from fastapi.responses import HTMLResponse, Response

    app = FastAPI(title="Robot Job Blackboard Monitor")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _static_index_path().read_text(encoding="utf-8")

    @app.get("/api/robot-jobs")
    async def robot_jobs(max_events: int = Query(default=50, ge=0, le=200)) -> dict[str, Any]:
        return await board.snapshot(max_events=max_events)

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    return app


@dataclass
class RobotJobMonitorServer:
    url: str
    _server: Any
    _task: asyncio.Task[None]

    async def stop(self) -> None:
        self._server.should_exit = True
        await self._task


async def start_robot_job_monitor_from_env(
    board: RobotJobBoard | None,
    *,
    env: Mapping[str, str] | None = None,
) -> RobotJobMonitorServer | None:
    settings = os.environ if env is None else env
    raw_port = settings.get("ROBOT_JOB_MONITOR_PORT")
    if raw_port is None or not raw_port.strip():
        return None
    if board is None:
        logger.warning("ROBOT_JOB_MONITOR_PORT is set, but no Robot Job Blackboard is available")
        return None
    try:
        port = int(raw_port)
    except ValueError:
        logger.warning("Invalid ROBOT_JOB_MONITOR_PORT={!r}; monitor disabled", raw_port)
        return None
    if port <= 0 or port > 65535:
        logger.warning("ROBOT_JOB_MONITOR_PORT={} is outside the valid range; monitor disabled", port)
        return None

    host = settings.get("ROBOT_JOB_MONITOR_HOST", "127.0.0.1").strip() or "127.0.0.1"
    import uvicorn

    app = create_app(board)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    setattr(server, "install_signal_handlers", lambda: None)
    task = asyncio.create_task(server.serve())
    url = f"http://{host}:{port}"
    logger.info("Robot Job Blackboard monitor listening on {}", url)
    return RobotJobMonitorServer(url=url, _server=server, _task=task)


def _static_index_path() -> Path:
    return Path(__file__).resolve().parent / "static" / "job_monitor.html"
