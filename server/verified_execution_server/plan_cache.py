from __future__ import annotations

import asyncio
import time
from threading import RLock
from typing import Any, Protocol

from verified_execution_server.models import CachedPlan


class PlanCache(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def get_plan(self, robot_name: str, plan_name: str) -> CachedPlan | None: ...

    def sync_joint_state(
        self,
        robot_name: str,
        *,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> bool: ...

    def size(self) -> int: ...

    def is_connected(self) -> bool: ...


class RosPlanCache:
    def __init__(
        self,
        *,
        robot_name: str = "UR10",
        host: str = "127.0.0.1",
        port: int = 9090,
        time_fn: Any = time.monotonic,
    ) -> None:
        self.robot_name = robot_name
        self.host = host
        self.port = port
        self._time_fn = time_fn
        self._lock = RLock()
        self._plans: dict[tuple[str, str], CachedPlan] = {}
        self._roslibpy: Any | None = None
        self._client: Any | None = None
        self._subscriber: Any | None = None
        self._joint_state_publishers: dict[str, Any] = {}
        self._connected = False

    async def start(self) -> None:
        await asyncio.to_thread(self._connect)

    async def stop(self) -> None:
        await asyncio.to_thread(self._close)

    def get_plan(self, robot_name: str, plan_name: str) -> CachedPlan | None:
        with self._lock:
            return self._plans.get((robot_name, plan_name))

    def size(self) -> int:
        with self._lock:
            return len(self._plans)

    def is_connected(self) -> bool:
        client = self._client
        return bool(self._connected and client is not None and client.is_connected)

    def _connect(self) -> None:
        if self._connected:
            return
        import roslibpy

        self._roslibpy = roslibpy
        self._client = roslibpy.Ros(host=self.host, port=self.port)
        topic = f"/{self.robot_name}/request/planned_path"
        self._subscriber = roslibpy.Topic(
            self._client,
            topic,
            "vizor_package/PlannedTrajectory",
        )
        self._subscriber.subscribe(self._record_planned_path)
        self._client.run()
        self._connected = True

    def _close(self) -> None:
        if self._subscriber is not None:
            try:
                self._subscriber.unsubscribe()
            except Exception:
                pass
            self._subscriber = None
        if self._client is not None:
            try:
                self._client.terminate()
            except Exception:
                pass
            self._client = None
        self._joint_state_publishers = {}
        self._connected = False

    def sync_joint_state(
        self,
        robot_name: str,
        *,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> bool:
        client = self._client
        roslibpy = self._roslibpy
        if (
            roslibpy is None
            or client is None
            or not client.is_connected
            or len(joint_names) != len(joint_positions)
            or not joint_names
        ):
            return False
        topic = f"/{robot_name}/move_group/fake_controller_joint_states"
        publisher = self._joint_state_publishers.get(topic)
        if publisher is None:
            publisher = roslibpy.Topic(client, topic, "sensor_msgs/JointState")
            self._joint_state_publishers[topic] = publisher
        timestamp = float(self._time_fn())
        seconds = int(timestamp)
        payload = {
            "header": {
                "stamp": {
                    "secs": seconds,
                    "nsecs": int((timestamp - seconds) * 1_000_000_000),
                },
                "frame_id": "",
            },
            "name": joint_names,
            "position": joint_positions,
            "velocity": [],
            "effort": [],
        }
        try:
            publisher.publish(roslibpy.Message(payload))
        except Exception:
            return False
        return True

    def _record_planned_path(self, message: dict[str, Any]) -> None:
        plan_name = message.get("name")
        if not isinstance(plan_name, str) or not plan_name:
            return
        frames = _trajectory_frames(message)
        if not frames:
            return
        plan = CachedPlan(
            robot_name=self.robot_name,
            plan_name=plan_name,
            frames=frames,
            joint_names=_joint_names(message),
            observed_at_s=float(self._time_fn()),
        )
        with self._lock:
            self._plans[(self.robot_name, plan_name)] = plan


def _trajectory_frames(message: dict[str, Any]) -> list[dict]:
    raw_points = ((message.get("joint_trajectory") or {}).get("points") or [])
    frames: list[dict] = []
    for point in raw_points:
        if not isinstance(point, dict):
            continue
        positions = _float_list(point.get("positions"))
        if not positions:
            continue
        frame: dict[str, Any] = {"positions": positions}
        velocities = _float_list(point.get("velocities"))
        if velocities:
            frame["velocities"] = velocities
        accelerations = _float_list(point.get("accelerations"))
        if accelerations:
            frame["accelerations"] = accelerations
        time_from_start_s = _duration_seconds(point.get("time_from_start"))
        if time_from_start_s is not None:
            frame["time_from_start_s"] = time_from_start_s
        frames.append(frame)
    return frames


def _joint_names(message: dict[str, Any]) -> list[str] | None:
    raw_names = ((message.get("joint_trajectory") or {}).get("joint_names") or [])
    if not isinstance(raw_names, list) or not raw_names:
        return None
    names = [name for name in raw_names if isinstance(name, str) and name]
    return names or None


def _float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [float(item) for item in value]


def _duration_seconds(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, dict):
        return None
    seconds = value.get("secs", value.get("sec", 0))
    nanoseconds = value.get("nsecs", value.get("nanosec", 0))
    return float(seconds) + float(nanoseconds) / 1_000_000_000.0
