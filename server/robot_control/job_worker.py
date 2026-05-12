from __future__ import annotations

import asyncio
from typing import Any, Protocol

from robot_control.job_board import RobotJobBoard


class RobotToolBridgeLike(Protocol):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class RobotJobWorker:
    def __init__(self, *, board: RobotJobBoard, tool_bridge: RobotToolBridgeLike) -> None:
        self._board = board
        self._tool_bridge = tool_bridge
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stopping.clear()
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self) -> None:
        self._stopping.set()
        if self._task is None:
            return
        await self._task
        self._task = None

    async def run_once(self) -> bool:
        job = await self._board.claim_next()
        if job is None:
            return False
        try:
            result = await self._tool_bridge.call_tool(job.tool_name, job.arguments)
        except Exception as exc:
            await self._board.fail(job.job_id, str(exc))
        else:
            await self._board.complete(job.job_id, result)
        return True

    async def _run_loop(self) -> None:
        while not self._stopping.is_set():
            ran = await self.run_once()
            if not ran:
                await asyncio.sleep(0.05)
