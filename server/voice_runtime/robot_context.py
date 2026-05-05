from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class RobotContextSnapshot:
    observed_at_s: float | None = None
    robot_name: str | None = None
    tcp_pose: dict[str, Any] | None = None
    gripper_state: str | None = None
    last_execution_result: str | None = None


class RobotContextStore:
    def __init__(self, *, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._snapshot = RobotContextSnapshot()
        self._time_fn = time_fn

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        observed_at_s = self._snapshot.observed_at_s
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s

    def render_instruction_block(self) -> str:
        age = self._status_age_text()
        lines = [
            "Last-known robot context:",
            "- This context is advisory only.",
            "- For movement, relative commands, retries, or safety-sensitive actions, call moveit_get_current_pose first.",
            f"- status age: {age}",
        ]
        if self._snapshot.robot_name is None:
            lines.append("- No robot status has been observed yet.")
            return "\n".join(lines)

        lines.append(f"- robot: {self._snapshot.robot_name}")
        pose_text = self._tcp_pose_text()
        if pose_text:
            lines.append(f"- tcp pose: {pose_text}")
        if self._snapshot.gripper_state:
            lines.append(f"- gripper: {self._snapshot.gripper_state}")
        if self._snapshot.last_execution_result:
            lines.append(f"- last execution: {self._snapshot.last_execution_result}")
        return "\n".join(lines)

    def latest_tcp_pose(self) -> dict[str, Any] | None:
        if self._snapshot.tcp_pose is None:
            return None
        return dict(self._snapshot.tcp_pose)

    def update_from_tool_result(self, tool_name: str, output: str) -> None:
        if tool_name not in {"moveit_get_current_pose", "moveit_get_robot_status"}:
            return
        structured_content = _structured_content(output)
        if not isinstance(structured_content, dict) or structured_content.get("ok") is not True:
            return

        self._snapshot.observed_at_s = self._time_fn()
        robot_name = structured_content.get("robot_name", structured_content.get("robot"))
        if isinstance(robot_name, str):
            self._snapshot.robot_name = robot_name
        tcp_pose = structured_content.get("tcp_pose")
        raw = structured_content.get("raw")
        if not isinstance(tcp_pose, dict) and isinstance(raw, dict):
            tcp_pose = raw.get("pose")
        if isinstance(tcp_pose, dict):
            self._snapshot.tcp_pose = tcp_pose
        gripper = structured_content.get("gripper")
        if isinstance(gripper, dict) and isinstance(gripper.get("state"), str):
            self._snapshot.gripper_state = gripper["state"]
        last_execution = structured_content.get("last_execution")
        if isinstance(last_execution, dict) and isinstance(last_execution.get("result"), str):
            self._snapshot.last_execution_result = last_execution["result"]

    def _status_age_text(self) -> str:
        if self._snapshot.observed_at_s is None:
            return "unknown"
        return f"{self._time_fn() - self._snapshot.observed_at_s:.1f}s"

    def _tcp_pose_text(self) -> str | None:
        pose = self._snapshot.tcp_pose
        if not isinstance(pose, dict):
            return None
        position = pose.get("position")
        if not isinstance(position, dict):
            return None
        try:
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        except (KeyError, TypeError, ValueError):
            return None
        return f"x={x:.3f}, y={y:.3f}, z={z:.3f}"


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")
