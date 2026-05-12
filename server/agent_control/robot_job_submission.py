from __future__ import annotations

import json
from typing import Any

from robot_control.job_board import RobotJobBoard, SubmitRobotJob

QUEUEABLE_ROBOT_ACTION_TOOLS = frozenset(
    {
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_and_execute_free_motion",
        "moveit_plan_and_execute_cartesian_motion",
        "moveit_execute_plan",
        "moveit_open_gripper",
        "moveit_close_gripper",
        "moveit_attach_object",
    }
)


class RobotJobSubmitter:
    def __init__(self, job_board: RobotJobBoard) -> None:
        self._job_board = job_board

    async def submit_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        requested_by_turn_id: str | None = None,
    ) -> str:
        job = await self._job_board.submit(
            SubmitRobotJob(
                tool_name=tool_name,
                arguments=arguments,
                requested_by_turn_id=requested_by_turn_id,
            )
        )
        return json.dumps(
            {
                "content": [
                    f"Queued robot job {job.job_id} for {tool_name}. "
                    "The robot worker will report completion or failure."
                ],
                "structured_content": {
                    "ok": True,
                    "status": job.status.value,
                    "job_id": job.job_id,
                    "tool_name": job.tool_name,
                },
                "is_error": False,
            },
            ensure_ascii=False,
        )
