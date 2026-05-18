from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Protocol

from verified_execution_server.models import CachedPlan

PLANNING_SCENE_COMPONENTS = 4 | 8 | 16 | 512
COLLISION_OBJECT_ADD = 0
COLLISION_OBJECT_REMOVE = 1


@dataclass(frozen=True)
class AttachedObjectReleaseResult:
    ok: bool
    status: str
    checked: bool
    attached_objects_before_release: list[str] = field(default_factory=list)
    attached_objects_released: list[str] = field(default_factory=list)
    published: bool = False
    verified: bool = False
    topic_or_service: str = ""
    error: str | None = None
    correction: str | None = None


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

    def sync_gripper_joint_state(
        self,
        robot_name: str,
        *,
        joint_name: str,
        joint_position: float,
    ) -> bool: ...

    def release_attached_objects(
        self,
        robot_name: str,
        *,
        timeout_s: float,
    ) -> AttachedObjectReleaseResult: ...

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
        topic = f"/{robot_name}/move_group/fake_controller_joint_states"
        return self._publish_joint_state(
            topic,
            joint_names=joint_names,
            joint_positions=joint_positions,
        )

    def sync_gripper_joint_state(
        self,
        robot_name: str,
        *,
        joint_name: str,
        joint_position: float,
    ) -> bool:
        topic = f"/{robot_name}/gripper_joint_states"
        return self._publish_joint_state(
            topic,
            joint_names=[joint_name],
            joint_positions=[joint_position],
        )

    def release_attached_objects(
        self,
        robot_name: str,
        *,
        timeout_s: float,
    ) -> AttachedObjectReleaseResult:
        with self._lock:
            service_name = _apply_planning_scene_service_name(robot_name)
            payload = self._read_planning_scene(robot_name, timeout_s)
            if not isinstance(payload, dict):
                return AttachedObjectReleaseResult(
                    ok=False,
                    status="planning_scene_unavailable",
                    checked=True,
                    topic_or_service=service_name,
                    error="MoveIt planning scene service did not return scene geometry.",
                    correction=f"Check /{robot_name}/get_planning_scene and retry state sync.",
                )

            scene = _scene_from_payload(payload)
            attached_objects = _attached_collision_objects(scene)
            object_names = [_attached_object_name(attached) for attached in attached_objects]
            object_names = [name for name in object_names if name is not None]
            if not object_names:
                return AttachedObjectReleaseResult(
                    ok=True,
                    status="no_attached_objects",
                    checked=True,
                    verified=True,
                    topic_or_service=service_name,
                )

            planning_scene_diff = _release_attached_objects_diff(attached_objects)
            if not self._apply_planning_scene(robot_name, planning_scene_diff, timeout_s):
                return AttachedObjectReleaseResult(
                    ok=False,
                    status="planning_scene_apply_failed",
                    checked=True,
                    attached_objects_before_release=object_names,
                    attached_objects_released=object_names,
                    published=False,
                    verified=False,
                    topic_or_service=service_name,
                    error="MoveIt apply_planning_scene did not confirm attached object release.",
                    correction=f"Check {service_name} and retry state sync.",
                )

            verified_payload = self._read_planning_scene(robot_name, timeout_s)
            verified = (
                isinstance(verified_payload, dict)
                and _objects_released(_scene_from_payload(verified_payload), object_names)
            )
            if not verified:
                return AttachedObjectReleaseResult(
                    ok=False,
                    status="released_collision_object_unverified",
                    checked=True,
                    attached_objects_before_release=object_names,
                    attached_objects_released=object_names,
                    published=True,
                    verified=False,
                    topic_or_service=service_name,
                    error=(
                        "MoveIt apply_planning_scene did not verify "
                        f"{', '.join(object_names)} released."
                    ),
                    correction=(
                        f"Check /{robot_name}/get_planning_scene and {service_name}, "
                        "then retry state sync."
                    ),
                )

            return AttachedObjectReleaseResult(
                ok=True,
                status="released_collision_objects_verified",
                checked=True,
                attached_objects_before_release=object_names,
                attached_objects_released=object_names,
                published=True,
                verified=True,
                topic_or_service=service_name,
            )

    def _publish_joint_state(
        self,
        topic: str,
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

    def _read_planning_scene(self, robot_name: str, timeout_s: float) -> dict[str, Any] | None:
        client = self._client
        roslibpy = self._roslibpy
        if roslibpy is None or client is None or not client.is_connected:
            return None
        try:
            service = roslibpy.Service(
                client,
                f"/{robot_name}/get_planning_scene",
                "moveit_msgs/GetPlanningScene",
            )
            request = roslibpy.ServiceRequest(
                {"components": {"components": PLANNING_SCENE_COMPONENTS}}
            )
            response = service.call(request, timeout=timeout_s)
        except Exception:
            return None
        if isinstance(response, dict):
            return dict(response)
        if hasattr(response, "items"):
            return dict(response.items())
        return None

    def _apply_planning_scene(
        self,
        robot_name: str,
        payload: dict[str, Any],
        timeout_s: float,
    ) -> bool:
        client = self._client
        roslibpy = self._roslibpy
        if roslibpy is None or client is None or not client.is_connected:
            return False
        try:
            service = roslibpy.Service(
                client,
                _apply_planning_scene_service_name(robot_name),
                "moveit_msgs/ApplyPlanningScene",
            )
            request = roslibpy.ServiceRequest({"scene": payload})
            response = service.call(request, timeout=timeout_s)
        except Exception:
            return False
        success = response.get("success") if hasattr(response, "get") else getattr(response, "success", False)
        return success is True

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


def _apply_planning_scene_service_name(robot_name: str) -> str:
    return f"/{robot_name}/apply_planning_scene"


def _scene_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    scene = payload.get("scene")
    return scene if isinstance(scene, dict) else payload


def _attached_collision_objects(scene: dict[str, Any]) -> list[dict[str, Any]]:
    robot_state_value = scene.get("robot_state")
    robot_state = robot_state_value if isinstance(robot_state_value, dict) else {}
    attached_objects: list[dict[str, Any]] = []
    for attached in robot_state.get("attached_collision_objects") or []:
        if not isinstance(attached, dict):
            continue
        if _attached_object_name(attached) is None:
            continue
        attached_objects.append(attached)
    return attached_objects


def _attached_object_name(attached: dict[str, Any]) -> str | None:
    collision_object = attached.get("object")
    if not isinstance(collision_object, dict):
        return None
    object_name = collision_object.get("id")
    return object_name if isinstance(object_name, str) and object_name else None


def _release_attached_objects_diff(attached_objects: list[dict[str, Any]]) -> dict[str, Any]:
    remove_attached_objects: list[dict[str, Any]] = []
    world_objects: list[dict[str, Any]] = []
    for attached in attached_objects:
        object_name = _attached_object_name(attached)
        if object_name is None:
            continue
        link_name = attached.get("link_name")
        if not isinstance(link_name, str) or not link_name:
            link_name = ""
        touch_links = attached.get("touch_links")
        touch = [str(link) for link in touch_links] if isinstance(touch_links, list) else []
        remove_attached_objects.append(
            {
                "link_name": link_name,
                "object": {"id": object_name, "operation": COLLISION_OBJECT_REMOVE},
                "touch_links": touch,
            }
        )
        collision_object = deepcopy(attached["object"])
        collision_object["operation"] = COLLISION_OBJECT_ADD
        world_objects.append(collision_object)
    return {
        "name": "",
        "robot_state": {
            "is_diff": True,
            "attached_collision_objects": remove_attached_objects,
        },
        "world": {"collision_objects": world_objects},
        "is_diff": True,
    }


def _objects_released(scene: dict[str, Any], object_names: list[str]) -> bool:
    attached_names = set(_attached_object_ids(scene))
    world_names = set(_world_collision_object_ids(scene))
    return all(name not in attached_names and name in world_names for name in object_names)


def _attached_object_ids(scene: dict[str, Any]) -> list[str]:
    return [
        name
        for name in (_attached_object_name(attached) for attached in _attached_collision_objects(scene))
        if name is not None
    ]


def _world_collision_object_ids(scene: dict[str, Any]) -> list[str]:
    world_value = scene.get("world")
    world = world_value if isinstance(world_value, dict) else {}
    object_ids: list[str] = []
    for collision_object in world.get("collision_objects") or []:
        if not isinstance(collision_object, dict):
            continue
        object_id = collision_object.get("id")
        if isinstance(object_id, str) and object_id:
            object_ids.append(object_id)
    return object_ids


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
