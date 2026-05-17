from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from queue import Empty, Queue
from threading import RLock
from typing import Any, Callable, Protocol

from moveit_mcp.scene import (
    PLANNING_SCENE_COMPONENTS,
    available_object_names,
    object_context,
    summarize_planning_scene,
)

SUCCESS_STATUSES = {"success", "success! "}
PHYSICAL_PARAM = "/vizor_robot_control/physical"
AGENT_PATH_NAME = "AgentPath"
MTC_PICK_TASK_SERVICE = "/vizor_mtc/plan_pick_task"
MTC_PICK_TASK_REQUEST_PARAM = f"{MTC_PICK_TASK_SERVICE}/request"
MTC_COMPOUND_TASK_SERVICE = "/vizor_mtc/plan_compound_task"
MTC_COMPOUND_TASK_REQUEST_PARAM = f"{MTC_COMPOUND_TASK_SERVICE}/request"
DEFAULT_ATTACH_LINK = "tool0"
COLLISION_OBJECT_ADD = 0
COLLISION_OBJECT_REMOVE = 1
DEFAULT_TOUCH_LINKS = [
    "tool0",
    "wrist_3_link",
    "robotiq_arg2f_base_link",
    "left_outer_knuckle",
    "right_outer_knuckle",
    "left_inner_knuckle",
    "right_inner_knuckle",
    "left_outer_finger",
    "right_outer_finger",
    "left_inner_finger",
    "right_inner_finger",
    "left_inner_finger_pad",
    "right_inner_finger_pad",
]


@dataclass(frozen=True)
class Pose:
    position: dict[str, float]
    orientation: dict[str, float]

    @classmethod
    def position_only(cls, x: float, y: float, z: float) -> "Pose":
        return cls(
            position={"x": float(x), "y": float(y), "z": float(z)},
            orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        )

    @classmethod
    def from_input(cls, value: dict[str, Any]) -> "Pose":
        if "position" in value or "orientation" in value:
            position = value.get("position")
            orientation = value.get("orientation", {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0})
            if not isinstance(position, dict) or not isinstance(orientation, dict):
                raise ValueError("Pose input must contain position and orientation objects")
        else:
            position = value
            orientation = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}

        pose = cls(
            position={axis: float(position[axis]) for axis in ("x", "y", "z")},
            orientation={axis: float(orientation[axis]) for axis in ("x", "y", "z", "w")},
        )
        pose._validate_quaternion()
        return pose

    def _validate_quaternion(self) -> None:
        norm = math.sqrt(sum(self.orientation[axis] ** 2 for axis in ("x", "y", "z", "w")))
        if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(f"Quaternion orientation must be normalized; norm={norm:.6f}")

    def to_msg(self) -> dict[str, Any]:
        return {"position": dict(self.position), "orientation": dict(self.orientation)}


@dataclass(frozen=True)
class PlanFeedback:
    robot: str
    name: str
    status: str
    trajectory_points: int
    can_execute: bool
    raw_path: dict[str, Any] | None
    final_joint_positions: list[float] | None


@dataclass(frozen=True)
class ExecuteFeedback:
    robot: str
    name: str
    status: str
    physical_mode: bool | None
    command_published: bool
    observed_joint_state: list[float] | None
    expected_joint_state: list[float] | None
    final_positions_match: bool
    observed_joint_names: list[str] | None = None
    expected_joint_names: list[str] | None = None


@dataclass(frozen=True)
class CurrentPoseFeedback:
    robot: str
    ok: bool
    status: str
    planning_frame: str | None
    pose: Pose | None
    source: str
    message: str


@dataclass(frozen=True)
class RobotStateFeedback:
    robot: str
    ok: bool
    status: str
    planning_frame: str | None
    pose: Pose | None
    physical_mode: bool | None
    joint_state: list[float] | None
    source: str
    message: str


@dataclass(frozen=True)
class GripperFeedback:
    robot: str
    state: str
    action_name: str
    action_type: str
    joint_state_topic: str
    goal_position_m: float
    speed_mps: float
    force: float
    expected_joint_position: float
    observed_joint_position: float | None
    action_result: dict[str, Any] | None
    command_sent: bool
    ok: bool


@dataclass(frozen=True)
class SceneObjectsFeedback:
    robot: str
    ok: bool
    status: str
    planning_frame: str | None
    objects: list[dict[str, Any]]
    source: str
    message: str


@dataclass(frozen=True)
class ObjectContextFeedback:
    robot: str
    ok: bool
    status: str
    planning_frame: str | None
    object_context: dict[str, Any] | None
    available_objects: list[str]
    source: str
    message: str


@dataclass(frozen=True)
class AttachSceneFeedback:
    robot: str
    object_name: str
    ok: bool
    status: str
    planning_frame: str | None
    link_name: str
    touch_links: list[str]
    scene_update_published: bool
    source: str
    message: str


@dataclass(frozen=True)
class DetachSceneFeedback:
    robot: str
    object_name: str
    ok: bool
    status: str
    planning_frame: str | None
    link_name: str
    scene_update_published: bool
    source: str
    message: str


@dataclass(frozen=True)
class RemoveSceneFeedback:
    robot: str
    object_name: str
    ok: bool
    status: str
    planning_frame: str | None
    scene_update_published: bool
    source: str
    message: str


class RosbridgeTransport(Protocol):
    def publish(self, topic: str, message_type: str, payload: dict[str, Any]) -> None: ...
    def apply_planning_scene(self, robot: str, payload: dict[str, Any], timeout_s: float) -> bool: ...
    def prepare_for_plan(self, status_topic: str, path_topic: str) -> None: ...
    def wait_for_status(self, topic: str, timeout_s: float) -> str | None: ...
    def wait_for_planned_path(self, topic: str, name: str, timeout_s: float) -> dict[str, Any] | None: ...
    def prepare_for_execute(self, joint_state_topic: str) -> None: ...
    def prepare_for_gripper(self, status_topic: str, gripper_topic: str) -> None: ...
    def wait_for_joint_state(self, topic: str, timeout_s: float) -> list[float] | None: ...
    def wait_for_bool(self, topic: str, timeout_s: float) -> bool | None: ...
    def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None: ...
    def read_physical_mode(self, param: str = PHYSICAL_PARAM) -> bool | None: ...
    def read_current_pose(self, robot: str, timeout_s: float) -> dict[str, Any] | None: ...
    def read_planning_scene(self, robot: str, timeout_s: float) -> dict[str, Any] | None: ...
    def plan_mtc_pick_task(self, robot: str, object_name: str, grasp_face: str | None, timeout_s: float) -> dict[str, Any] | None: ...
    def plan_mtc_compound_task(
        self,
        robot: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: list[str] | None,
        backend: str,
        timeout_s: float,
    ) -> dict[str, Any] | None: ...
    def send_action_goal(self, action_name: str, action_type: str, goal: dict[str, Any], timeout_s: float) -> dict[str, Any] | None: ...


class FakeRosbridgeTransport:
    def __init__(self, *, physical_mode: bool | None = False) -> None:
        self.published: list[tuple[str, dict[str, Any]]] = []
        self.applied_planning_scenes: list[dict[str, Any]] = []
        self.action_goals: list[tuple[str, str, dict[str, Any]]] = []
        self.events: list[tuple[Any, ...]] = []
        self._physical_mode = physical_mode
        self._statuses: dict[str, Queue[str]] = {}
        self._paths: dict[str, Queue[dict[str, Any]]] = {}
        self._joint_states: dict[str, Queue[dict[str, Any]]] = {}
        self._bools: dict[str, Queue[bool]] = {}
        self._action_results: dict[str, Queue[dict[str, Any]]] = {}
        self._mtc_pick_task_results: Queue[dict[str, Any] | None] = Queue()
        self._mtc_compound_task_results: Queue[dict[str, Any] | None] = Queue()
        self._planning_scenes: dict[str, dict[str, Any]] = {}
        self._after_publish_statuses: dict[str, Queue[str]] = {}
        self._after_publish_paths: dict[str, Queue[dict[str, Any]]] = {}
        self._after_publish_joint_states: dict[str, Queue[dict[str, Any]]] = {}
        self._after_publish_bools: dict[str, Queue[bool]] = {}
        self._after_action_joint_states: dict[str, Queue[dict[str, Any]]] = {}
        self._current_poses: dict[str, dict[str, Any]] = {}

    def set_physical_mode(self, value: bool | None) -> None:
        self._physical_mode = value

    def set_current_pose(self, robot: str, pose: dict[str, Any], *, planning_frame: str = "base_link") -> None:
        self._current_poses[robot] = {
            "ok": True,
            "robot": robot,
            "planning_frame": planning_frame,
            "pose": pose,
        }

    def set_planning_scene(self, robot: str, scene: dict[str, Any], *, planning_frame: str = "base_link") -> None:
        self._planning_scenes[robot] = {**scene, "planning_frame": planning_frame}

    def publish(self, topic: str, message_type: str, payload: dict[str, Any]) -> None:
        self.events.append(("publish", topic))
        self.published.append((topic, payload))
        self._release_after_publish_messages(published_name=payload.get("name"))

    def apply_planning_scene(self, robot: str, payload: dict[str, Any], timeout_s: float) -> bool:
        self.events.append(("apply_planning_scene", robot, timeout_s))
        self.applied_planning_scenes.append(deepcopy(payload))
        current = self._planning_scenes.get(robot)
        if not isinstance(current, dict):
            return False
        self._planning_scenes[robot] = _apply_planning_scene_diff(current, payload)
        return True

    def prepare_for_plan(self, status_topic: str, path_topic: str) -> None:
        self.events.append(("prepare_for_plan", status_topic, path_topic))
        self._drain(self._statuses.setdefault(status_topic, Queue()))
        self._drain(self._paths.setdefault(path_topic, Queue()))

    def prepare_for_execute(self, joint_state_topic: str) -> None:
        self.events.append(("prepare_for_execute", joint_state_topic))
        self._drain(self._joint_states.setdefault(joint_state_topic, Queue()))

    def prepare_for_gripper(self, status_topic: str, gripper_topic: str) -> None:
        self.events.append(("prepare_for_gripper", status_topic, gripper_topic))
        self._drain(self._statuses.setdefault(status_topic, Queue()))
        self._drain(self._bools.setdefault(gripper_topic, Queue()))

    def queue_status(self, topic: str, status: str) -> None:
        self._statuses.setdefault(topic, Queue()).put(status)

    def queue_status_after_publish(self, topic: str, status: str) -> None:
        self._after_publish_statuses.setdefault(topic, Queue()).put(status)

    def queue_planned_path(
        self,
        topic: str,
        *,
        name: str,
        points: int,
        final_positions: list[float] | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        self._paths.setdefault(topic, Queue()).put(_planned_path_msg(name, points, final_positions, joint_names))

    def queue_planned_path_after_publish(
        self,
        topic: str,
        *,
        name: str,
        points: int,
        final_positions: list[float] | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        self._after_publish_paths.setdefault(topic, Queue()).put(_planned_path_msg(name, points, final_positions, joint_names))

    def queue_joint_state(self, topic: str, positions: list[float], *, names: list[str] | None = None) -> None:
        self._joint_states.setdefault(topic, Queue()).put(_joint_state_msg(positions, names))

    def queue_joint_state_after_publish(self, topic: str, positions: list[float], *, names: list[str] | None = None) -> None:
        self._after_publish_joint_states.setdefault(topic, Queue()).put(_joint_state_msg(positions, names))

    def queue_bool(self, topic: str, value: bool) -> None:
        self._bools.setdefault(topic, Queue()).put(bool(value))

    def queue_bool_after_publish(self, topic: str, value: bool) -> None:
        self._after_publish_bools.setdefault(topic, Queue()).put(bool(value))

    def queue_action_result(self, action_name: str, result: dict[str, Any]) -> None:
        self._action_results.setdefault(action_name, Queue()).put(dict(result))

    def queue_mtc_pick_task_result(self, result: dict[str, Any] | None) -> None:
        self._mtc_pick_task_results.put(deepcopy(result) if isinstance(result, dict) else None)

    def queue_mtc_compound_task_result(self, result: dict[str, Any] | None) -> None:
        self._mtc_compound_task_results.put(deepcopy(result) if isinstance(result, dict) else None)

    def queue_joint_state_after_action(self, topic: str, positions: list[float], *, names: list[str] | None = None) -> None:
        self._after_action_joint_states.setdefault(topic, Queue()).put(_joint_state_msg(positions, names))

    def wait_for_status(self, topic: str, timeout_s: float) -> str | None:
        try:
            return self._statuses.setdefault(topic, Queue()).get(timeout=timeout_s)
        except Empty:
            return None

    def wait_for_planned_path(self, topic: str, name: str, timeout_s: float) -> dict[str, Any] | None:
        queue = self._paths.setdefault(topic, Queue())
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = queue.get(timeout=remaining)
            except Empty:
                return None
            if msg.get("name") == name:
                return msg

    def wait_for_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
        msg = self.wait_for_joint_state_message(topic, timeout_s)
        return _joint_state_positions(msg) if msg is not None else None

    def wait_for_joint_state_message(self, topic: str, timeout_s: float) -> dict[str, Any] | None:
        try:
            return self._joint_states.setdefault(topic, Queue()).get(timeout=timeout_s)
        except Empty:
            return None

    def wait_for_bool(self, topic: str, timeout_s: float) -> bool | None:
        try:
            return self._bools.setdefault(topic, Queue()).get(timeout=timeout_s)
        except Empty:
            return None

    def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
        self.events.append(("read_joint_state", topic, timeout_s))
        return self.wait_for_joint_state(topic, timeout_s)

    def read_physical_mode(self, param: str = PHYSICAL_PARAM) -> bool | None:
        self.events.append(("read_physical_mode", param))
        return self._physical_mode

    def read_current_pose(self, robot: str, timeout_s: float) -> dict[str, Any] | None:
        self.events.append(("read_current_pose", robot, timeout_s))
        return self._current_poses.get(robot)

    def read_planning_scene(self, robot: str, timeout_s: float) -> dict[str, Any] | None:
        self.events.append(("read_planning_scene", robot, timeout_s))
        return self._planning_scenes.get(robot)

    def plan_mtc_pick_task(self, robot: str, object_name: str, grasp_face: str | None, timeout_s: float) -> dict[str, Any] | None:
        self.events.append(("plan_mtc_pick_task", robot, object_name, grasp_face, timeout_s))
        try:
            result = self._mtc_pick_task_results.get_nowait()
        except Empty:
            return None
        return deepcopy(result) if isinstance(result, dict) else None

    def plan_mtc_compound_task(
        self,
        robot: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: list[str] | None,
        backend: str,
        timeout_s: float,
    ) -> dict[str, Any] | None:
        self.events.append(
            (
                "plan_mtc_compound_task",
                robot,
                deepcopy(requirements),
                deepcopy(preferences) if isinstance(preferences, dict) else None,
                tuple(stage_intents or []),
                backend,
                timeout_s,
            )
        )
        try:
            result = self._mtc_compound_task_results.get_nowait()
        except Empty:
            return None
        return deepcopy(result) if isinstance(result, dict) else None

    def send_action_goal(self, action_name: str, action_type: str, goal: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        self.events.append(("send_action_goal", action_name, action_type, timeout_s))
        self.action_goals.append((action_name, action_type, dict(goal)))
        self._release_after_action_messages()
        try:
            return self._action_results.setdefault(action_name, Queue()).get(timeout=timeout_s)
        except Empty:
            return None

    def _release_after_publish_messages(self, *, published_name: str | None) -> None:
        for topic, queue in self._after_publish_statuses.items():
            if published_name is None:
                continue
            if not queue.empty():
                self._statuses.setdefault(topic, Queue()).put(queue.get_nowait())
        for topic, queue in self._after_publish_paths.items():
            if published_name is None:
                continue
            while not queue.empty():
                msg = queue.get_nowait()
                self._paths.setdefault(topic, Queue()).put(msg)
                if msg.get("name") == published_name:
                    break
        for topic, queue in self._after_publish_joint_states.items():
            if published_name is not None:
                continue
            if _pending_after_publish_plan_feedback(
                self._after_publish_statuses,
                self._after_publish_paths,
            ):
                if not queue.empty():
                    self._joint_states.setdefault(topic, Queue()).put(queue.get_nowait())
                continue
            while not queue.empty():
                self._joint_states.setdefault(topic, Queue()).put(queue.get_nowait())
        for topic, queue in self._after_publish_bools.items():
            if not queue.empty():
                self._bools.setdefault(topic, Queue()).put(queue.get_nowait())

    def _release_after_action_messages(self) -> None:
        for topic, queue in self._after_action_joint_states.items():
            while not queue.empty():
                self._joint_states.setdefault(topic, Queue()).put(queue.get_nowait())

    @staticmethod
    def _drain(queue: Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return


class RoslibpyTransport:
    def __init__(self, *, host: str = "localhost", port: int = 9090) -> None:
        import roslibpy

        self.roslibpy = roslibpy
        self.client = roslibpy.Ros(host=host, port=port)
        self._queues: dict[str, Queue[Any]] = {}
        self._subscribed: dict[str, Any] = {}
        self._latest_messages: dict[str, Any] = {}

    def connect(self) -> None:
        self.client.run()

    def close(self) -> None:
        try:
            self.client.terminate()
        except AttributeError as exc:
            if "_thread" not in str(exc):
                raise

    def publish(self, topic: str, message_type: str, payload: dict[str, Any]) -> None:
        topic_obj = self.roslibpy.Topic(self.client, topic, message_type)
        topic_obj.publish(self.roslibpy.Message(payload))

    def apply_planning_scene(self, robot: str, payload: dict[str, Any], timeout_s: float) -> bool:
        for service_name in _apply_planning_scene_service_names(robot):
            try:
                service = self.roslibpy.Service(self.client, service_name, "moveit_msgs/ApplyPlanningScene")
                request = self.roslibpy.ServiceRequest({"scene": payload})
                response: Any = service.call(request, timeout=timeout_s)
            except Exception:
                continue
            success = response.get("success") if hasattr(response, "get") else getattr(response, "success", False)
            if success is True:
                return True
        return False

    def prepare_for_plan(self, status_topic: str, path_topic: str) -> None:
        self._subscribe_once(status_topic, "std_msgs/String")
        self._subscribe_once(path_topic, "vizor_package/PlannedTrajectory")
        self._drain(self._queues[status_topic])
        self._drain(self._queues[path_topic])

    def prepare_for_execute(self, joint_state_topic: str) -> None:
        self._subscribe_once(joint_state_topic, "sensor_msgs/JointState")
        self._drain(self._queues[joint_state_topic])

    def prepare_for_gripper(self, status_topic: str, gripper_topic: str) -> None:
        self._subscribe_once(status_topic, "std_msgs/String")
        self._subscribe_once(gripper_topic, "std_msgs/Bool")
        self._drain(self._queues[status_topic])
        self._drain(self._queues[gripper_topic])

    def wait_for_status(self, topic: str, timeout_s: float) -> str | None:
        queue = self._subscribe_once(topic, "std_msgs/String")
        try:
            return queue.get(timeout=timeout_s).get("data")
        except Empty:
            return None

    def wait_for_planned_path(self, topic: str, name: str, timeout_s: float) -> dict[str, Any] | None:
        queue = self._subscribe_once(topic, "vizor_package/PlannedTrajectory")
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                msg = queue.get(timeout=remaining)
            except Empty:
                return None
            if msg.get("name") == name:
                return msg

    def wait_for_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
        msg = self.wait_for_joint_state_message(topic, timeout_s)
        return _joint_state_positions(msg) if msg is not None else None

    def wait_for_joint_state_message(self, topic: str, timeout_s: float) -> dict[str, Any] | None:
        queue = self._subscribe_once(topic, "sensor_msgs/JointState")
        try:
            return queue.get(timeout=timeout_s)
        except Empty:
            return None

    def wait_for_bool(self, topic: str, timeout_s: float) -> bool | None:
        queue = self._subscribe_once(topic, "std_msgs/Bool")
        try:
            msg = queue.get(timeout=timeout_s)
        except Empty:
            return None
        return _bool_value(msg)

    def read_joint_state(self, topic: str, timeout_s: float) -> list[float] | None:
        self._subscribe_once(topic, "sensor_msgs/JointState")
        latest = self._latest_messages.get(topic)
        if latest is not None:
            return _joint_state_positions(latest)
        return self.wait_for_joint_state(topic, timeout_s)

    def read_physical_mode(self, param: str = PHYSICAL_PARAM) -> bool | None:
        try:
            service = self.roslibpy.Service(self.client, "/rosapi/get_param", "rosapi/GetParam")
            request = self.roslibpy.ServiceRequest({"name": param, "default": "__unknown__"})
            response: Any = service.call(request)
        except Exception:
            return None
        value = response.get("value") if hasattr(response, "get") else getattr(response, "value", None)
        return _parse_bool_param_value(value)

    def read_current_pose(self, robot: str, timeout_s: float) -> dict[str, Any] | None:
        try:
            service = self.roslibpy.Service(self.client, f"/{robot}/get_current_pose", "std_srvs/Trigger")
            response: Any = service.call(self.roslibpy.ServiceRequest(), timeout=timeout_s)
        except Exception:
            return None
        success = response.get("success") if hasattr(response, "get") else getattr(response, "success", False)
        message = response.get("message") if hasattr(response, "get") else getattr(response, "message", "")
        if not success or not isinstance(message, str):
            return None
        try:
            return json.loads(message)
        except json.JSONDecodeError:
            return None

    def read_planning_scene(self, robot: str, timeout_s: float) -> dict[str, Any] | None:
        try:
            service = self.roslibpy.Service(self.client, f"/{robot}/get_planning_scene", "moveit_msgs/GetPlanningScene")
            request = self.roslibpy.ServiceRequest({"components": {"components": PLANNING_SCENE_COMPONENTS}})
            response: Any = service.call(request, timeout=timeout_s)
        except Exception:
            return None
        if isinstance(response, dict):
            return dict(response)
        if hasattr(response, "items"):
            return dict(response.items())
        return None

    def plan_mtc_pick_task(self, robot: str, object_name: str, grasp_face: str | None, timeout_s: float) -> dict[str, Any] | None:
        request_payload = {
            "robot_name": robot,
            "object_name": object_name,
            "grasp_face": grasp_face,
        }
        return self._call_mtc_trigger_service(
            service_name=MTC_PICK_TASK_SERVICE,
            request_param=MTC_PICK_TASK_REQUEST_PARAM,
            request_payload=request_payload,
            timeout_s=timeout_s,
        )

    def plan_mtc_compound_task(
        self,
        robot: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None = None,
        stage_intents: list[str] | None = None,
        backend: str = "mtc",
        timeout_s: float = 10.0,
    ) -> dict[str, Any] | None:
        request_payload = {
            "robot_name": robot,
            "requirements": deepcopy(requirements),
            "preferences": deepcopy(preferences) if isinstance(preferences, dict) else {},
            "backend": backend,
        }
        if stage_intents is not None:
            request_payload["stage_intents"] = list(stage_intents)
        return self._call_mtc_trigger_service(
            service_name=MTC_COMPOUND_TASK_SERVICE,
            request_param=MTC_COMPOUND_TASK_REQUEST_PARAM,
            request_payload=request_payload,
            timeout_s=timeout_s,
        )

    def _call_mtc_trigger_service(
        self,
        *,
        service_name: str,
        request_param: str,
        request_payload: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any] | None:
        try:
            set_param = self.roslibpy.Service(self.client, "/rosapi/set_param", "rosapi/SetParam")
            set_param.call(
                self.roslibpy.ServiceRequest(
                    {
                        "name": request_param,
                        "value": json.dumps(request_payload, sort_keys=True),
                    }
                ),
                timeout=timeout_s,
            )
            service = self.roslibpy.Service(self.client, service_name, "std_srvs/Trigger")
            response: Any = service.call(self.roslibpy.ServiceRequest(), timeout=timeout_s)
        except Exception:
            return None
        message = response.get("message") if hasattr(response, "get") else getattr(response, "message", "")
        if not isinstance(message, str):
            return None
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    def send_action_goal(self, action_name: str, action_type: str, goal: dict[str, Any], timeout_s: float) -> dict[str, Any] | None:
        result_topic = f"{action_name}/result"
        result_type = f"{action_type}Result"
        goal_topic = f"{action_name}/goal"
        goal_type = f"{action_type}Goal"
        result_queue = self._subscribe_once(result_topic, result_type)
        self._drain(result_queue)
        goal_topic_obj = self.roslibpy.Topic(self.client, goal_topic, goal_type)
        goal_id = f"moveit_mcp_{time.time_ns()}"
        now = time.time()
        secs = int(now)
        stamp = {"secs": secs, "nsecs": int((now - secs) * 1_000_000_000)}
        action_goal = {
            "header": {"seq": 0, "stamp": stamp, "frame_id": ""},
            "goal_id": {"stamp": stamp, "id": goal_id},
            "goal": goal,
        }
        try:
            goal_topic_obj.advertise()
            time.sleep(0.1)
            goal_topic_obj.publish(self.roslibpy.Message(action_goal))
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                result_msg = result_queue.get(timeout=remaining)
                status = result_msg.get("status", {}) if isinstance(result_msg, dict) else {}
                result_goal_id = status.get("goal_id", {}).get("id")
                if result_goal_id == goal_id:
                    result = result_msg.get("result", {})
                    return dict(result or {})
        except Exception:
            return None
        finally:
            try:
                goal_topic_obj.unadvertise()
            except Exception:
                pass

    def _subscribe_once(self, topic: str, message_type: str) -> Queue[Any]:
        queue = self._queues.setdefault(topic, Queue())
        if topic not in self._subscribed:
            topic_obj = self.roslibpy.Topic(self.client, topic, message_type)
            topic_obj.subscribe(lambda msg, q=queue, t=topic: self._record_message(t, q, msg))
            self._subscribed[topic] = topic_obj
        return queue

    def _record_message(self, topic: str, queue: Queue[Any], msg: Any) -> None:
        self._latest_messages[topic] = msg
        queue.put(msg)

    @staticmethod
    def _drain(queue: Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return


def _apply_planning_scene_service_names(robot: str) -> tuple[str, ...]:
    robot_name = robot.strip("/")
    if not robot_name:
        return ("/apply_planning_scene",)
    return (f"/{robot_name}/apply_planning_scene", "/apply_planning_scene")


def _normalize_mtc_compound_payload(
    payload: dict[str, Any],
    *,
    robot: str,
    requirements: dict[str, Any],
    preferences: dict[str, Any] | None,
    stage_intents: list[str] | None,
) -> dict[str, Any]:
    result = dict(payload)
    normalized_requirements = _dict_value(result.get("requirements")) or dict(requirements)
    normalized_preferences = _dict_value(result.get("preferences")) or dict(preferences or {})
    object_name = str(normalized_requirements.get("object_name") or result.get("object_name") or "")
    task_goal = str(normalized_requirements.get("goal") or result.get("task_goal") or "")
    target_pose = result.get("target_pose")
    if target_pose is None:
        target_pose = normalized_requirements.get("target_pose")
    if not isinstance(target_pose, dict):
        target_pose = None
    target_position = result.get("target_position")
    if target_position is None:
        target_position = normalized_requirements.get("target_position")
    if not isinstance(target_position, dict):
        target_position = None
    result.setdefault("ok", False)
    result.setdefault("backend", "mtc")
    result.setdefault("task_kind", "compound")
    result.setdefault("robot_name", robot)
    result.setdefault("object_name", object_name)
    result.setdefault("task_goal", task_goal)
    result["requirements"] = normalized_requirements
    result["preferences"] = normalized_preferences
    result.setdefault("stage_intents", list(stage_intents or []))
    result.setdefault("target_pose", target_pose)
    result.setdefault("target_position", target_position)
    result["task_stages"] = _dict_list(result.get("task_stages") or result.get("stage_summaries"))
    result["candidate_attempts"] = _dict_list(result.get("candidate_attempts"))
    result.setdefault("candidate_count", len(result["candidate_attempts"]))
    result.setdefault("selected_cost", None)
    result.setdefault("failed_stage", None)
    result.setdefault("blocker", None)
    result.setdefault("correction", "")
    result["scene_snapshot"] = _dict_value(result.get("scene_snapshot"))
    result["object_context"] = _dict_value(result.get("object_context"))
    result["selected_stage_evidence"] = _list_value(result.get("selected_stage_evidence"))
    result["selected_grasp_evidence"] = _dict_value(result.get("selected_grasp_evidence"))
    result["selected_place_evidence"] = _dict_value(result.get("selected_place_evidence"))
    if result.get("ok") is not True:
        result.pop("task_solution_id", None)
        result["execution_contract"] = _non_executable_mtc_compound_contract(result.get("execution_contract"))
        return result
    stage_debug_names = _agent_path_stage_debug_names(result["task_stages"], result["stage_intents"])
    preview = _dict_value(result.get("preview"))
    preview.setdefault("public_name", AGENT_PATH_NAME)
    preview.setdefault("stage_debug_names", stage_debug_names)
    result["preview"] = preview
    execution_contract_value = result.get("execution_contract")
    if isinstance(execution_contract_value, dict):
        execution_contract = dict(execution_contract_value)
        execution_contract.setdefault("target_kind", "task_solution")
        execution_contract.setdefault("requires_explicit_approval", True)
        execution_contract.setdefault("agent_path_name", AGENT_PATH_NAME)
        execution_contract.setdefault(
            "approval_signal",
            {"topic": f"/{robot}/command/execute", "payload": AGENT_PATH_NAME},
        )
        result["execution_contract"] = execution_contract
    elif isinstance(execution_contract_value, list):
        result["execution_contract"] = list(execution_contract_value)
    else:
        result["execution_contract"] = {
            "target_kind": "task_solution",
            "requires_explicit_approval": True,
            "can_execute": False,
        }
    return result


def _non_executable_mtc_compound_contract(value: Any) -> dict[str, Any]:
    contract = dict(value) if isinstance(value, dict) else {}
    target_kind = contract.get("target_kind")
    requires_explicit_approval = contract.get("requires_explicit_approval")
    return {
        "target_kind": target_kind if isinstance(target_kind, str) and target_kind else "task_solution",
        "requires_explicit_approval": (
            requires_explicit_approval if isinstance(requires_explicit_approval, bool) else True
        ),
        "can_execute": False,
    }


def _dict_value(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _list_value(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _agent_path_stage_debug_names(stages: list[dict[str, Any]], stage_intents: list[str]) -> list[str]:
    source: list[Any] = stages if stages else stage_intents
    names: list[str] = []
    for index, item in enumerate(source, start=1):
        raw_name: Any
        if isinstance(item, dict):
            raw_name = item.get("name") or item.get("intent") or item.get("kind")
        else:
            raw_name = item
        names.append(f"{AGENT_PATH_NAME}:{index:02d}_{_agent_path_slug(str(raw_name or 'stage'))}")
    return names


def _agent_path_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "_" for char in value.strip()]
    slug = "_".join(part for part in "".join(chars).split("_") if part)
    return slug or "stage"


class VizorClient:
    def __init__(
        self,
        *,
        transport: RosbridgeTransport,
        joint_tolerance: float = 1e-3,
        task_id_factory: Callable[[], int] | None = None,
    ) -> None:
        self.transport = transport
        self.joint_tolerance = joint_tolerance
        self._task_id_factory = task_id_factory or _new_task_id
        self._locks: dict[str, RLock] = {}
        self._planned_final_positions: dict[tuple[str, str], list[float] | None] = {}
        self._planned_joint_names: dict[tuple[str, str], list[str] | None] = {}
        self._active_agent_paths: dict[str, dict[str, Any]] = {}
        self._invalidated_agent_paths: set[str] = set()

    def _lock_for(self, robot: str) -> RLock:
        self._locks.setdefault(robot, RLock())
        return self._locks[robot]

    def get_current_pose(self, *, robot: str, timeout_s: float = 2.0) -> CurrentPoseFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_current_pose"
            payload = self.transport.read_current_pose(robot, timeout_s)
            if not isinstance(payload, dict):
                return CurrentPoseFeedback(robot, False, "current pose unavailable", None, None, source, "Current pose service did not return a pose")
            try:
                pose = Pose.from_input(payload["pose"])
            except (KeyError, TypeError, ValueError) as exc:
                return CurrentPoseFeedback(robot, False, "invalid current pose", None, None, source, str(exc))
            planning_frame = payload.get("planning_frame")
            if not isinstance(planning_frame, str) or not planning_frame:
                planning_frame = "base_link"
            return CurrentPoseFeedback(robot, True, "current pose observed", planning_frame, pose, source, "Current MoveIt pose observed")

    def get_robot_state(self, *, robot: str, timeout_s: float = 2.0) -> RobotStateFeedback:
        with self._lock_for(robot):
            pose_feedback = self.get_current_pose(robot=robot, timeout_s=timeout_s)
            physical_mode = self.transport.read_physical_mode(PHYSICAL_PARAM)
            joint_topic = f"/{robot}/move_group/fake_controller_joint_states"
            joint_state = self.transport.read_joint_state(joint_topic, timeout_s)
            ok = pose_feedback.ok and physical_mode is not None and joint_state is not None
            status = "robot state observed" if ok else "robot state incomplete"
            return RobotStateFeedback(
                robot=robot,
                ok=ok,
                status=status,
                planning_frame=pose_feedback.planning_frame,
                pose=pose_feedback.pose,
                physical_mode=physical_mode,
                joint_state=joint_state,
                source=f"{pose_feedback.source}; {PHYSICAL_PARAM}; {joint_topic}",
                message="Robot state observed" if ok else "Robot state is missing pose, physical-mode, or joint-state feedback",
            )

    def list_scene_objects(self, *, robot: str, timeout_s: float = 2.0) -> SceneObjectsFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_planning_scene"
            payload = self.transport.read_planning_scene(robot, timeout_s)
            if not isinstance(payload, dict):
                return SceneObjectsFeedback(
                    robot=robot,
                    ok=False,
                    status="planning scene unavailable",
                    planning_frame=None,
                    objects=[],
                    source=source,
                    message="MoveIt planning scene service did not return scene geometry",
                )
            scene = summarize_planning_scene(payload)
            objects = scene["objects"]
            return SceneObjectsFeedback(
                robot=robot,
                ok=True,
                status="planning scene observed",
                planning_frame=scene["planning_frame"],
                objects=objects,
                source=source,
                message=f"Observed {len(objects)} planning-scene objects",
            )

    def get_object_context(self, *, robot: str, object_name: str, timeout_s: float = 2.0) -> ObjectContextFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_planning_scene"
            payload = self.transport.read_planning_scene(robot, timeout_s)
            if not isinstance(payload, dict):
                return ObjectContextFeedback(
                    robot=robot,
                    ok=False,
                    status="planning scene unavailable",
                    planning_frame=None,
                    object_context=None,
                    available_objects=[],
                    source=source,
                    message="MoveIt planning scene service did not return scene geometry",
                )
            scene = summarize_planning_scene(payload)
            context = object_context(scene, object_name)
            names = available_object_names(scene)
            if context is None:
                return ObjectContextFeedback(
                    robot=robot,
                    ok=False,
                    status="object not found",
                    planning_frame=scene["planning_frame"],
                    object_context=None,
                    available_objects=names,
                    source=source,
                    message=f"Object {object_name!r} was not found in the MoveIt planning scene",
                )
            return ObjectContextFeedback(
                robot=robot,
                ok=True,
                status="object context observed",
                planning_frame=scene["planning_frame"],
                object_context=context,
                available_objects=names,
                source=source,
                message=f"Object context observed for {object_name}",
            )

    def plan_mtc_pick_task(
        self,
        *,
        robot: str,
        object_name: str,
        grasp_face: str | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        with self._lock_for(robot):
            payload = self.transport.plan_mtc_pick_task(robot, object_name, grasp_face, timeout_s)
            if not isinstance(payload, dict):
                return {
                    "ok": False,
                    "task_solution_id": "",
                    "failed_stage": "mtc_service_unavailable",
                    "message": f"{MTC_PICK_TASK_SERVICE} did not return a structured response.",
                    "blocker": f"{MTC_PICK_TASK_SERVICE} is unavailable or returned non-JSON data.",
                    "correction": "Start the Vizor MTC backend service and retry; do not treat this as a solved pick.",
                    "stage_summaries": [],
                    "candidate_attempts": [],
                    "candidate_count": 0,
                    "selected_cost": None,
                    "selected_grasp_face": grasp_face,
                    "robot_name": robot,
                    "object_name": object_name,
                    "grasp_face": grasp_face,
                    "backend": "mtc",
                    "gripper_responsibility": {
                        "open": "not_planned",
                        "close": "not_planned",
                        "verification": "not_planned",
                    },
                    "attach_responsibility": {
                        "attach": "not_planned",
                        "detach": "not_planned",
                        "verification": "not_planned",
                    },
                }
            result = dict(payload)
            result.setdefault("robot_name", robot)
            result.setdefault("object_name", object_name)
            result.setdefault("grasp_face", grasp_face)
            result.setdefault("backend", "mtc")
            result.setdefault("stage_summaries", [])
            result.setdefault("candidate_attempts", [])
            result.setdefault("candidate_count", len(result["candidate_attempts"]))
            result.setdefault("selected_cost", None)
            result.setdefault("selected_grasp_face", grasp_face)
            result.setdefault("gripper_responsibility", {})
            result.setdefault("attach_responsibility", {})
            return result

    def plan_mtc_compound_task(
        self,
        *,
        robot: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None = None,
        stage_intents: list[str] | None = None,
        backend: str = "mtc",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        normalized_requirements = dict(requirements)
        normalized_preferences = dict(preferences or {})
        intents = [str(intent) for intent in (stage_intents or [])]
        if backend != "mtc":
            return self._mtc_compound_unavailable_result(
                robot=robot,
                requirements=normalized_requirements,
                preferences=normalized_preferences,
                stage_intents=intents,
                failed_stage="mtc_backend_required",
                blocker=f"Compound MTC planning requires backend=\"mtc\"; got {backend!r}.",
                correction="Retry compound task planning with backend=\"mtc\".",
            )
        with self._lock_for(robot):
            payload = self.transport.plan_mtc_compound_task(
                robot,
                normalized_requirements,
                normalized_preferences,
                intents,
                backend,
                timeout_s,
            )
            if not isinstance(payload, dict):
                return self._mtc_compound_unavailable_result(
                    robot=robot,
                    requirements=normalized_requirements,
                    preferences=normalized_preferences,
                    stage_intents=intents,
                )
            result = _normalize_mtc_compound_payload(
                payload,
                robot=robot,
                requirements=normalized_requirements,
                preferences=normalized_preferences,
                stage_intents=intents,
            )
            if result.get("ok") is not True:
                result.pop("task_solution_id", None)
            return result

    @staticmethod
    def _mtc_compound_unavailable_result(
        *,
        robot: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any],
        stage_intents: list[str],
        failed_stage: str = "mtc_service_unavailable",
        blocker: str | None = None,
        correction: str | None = None,
    ) -> dict[str, Any]:
        object_name = str(requirements.get("object_name") or "")
        task_goal = str(requirements.get("goal") or "")
        target_pose = requirements.get("target_pose") if isinstance(requirements.get("target_pose"), dict) else None
        target_position = (
            requirements.get("target_position")
            if isinstance(requirements.get("target_position"), dict)
            else None
        )
        return {
            "ok": False,
            "backend": "mtc",
            "task_kind": "compound",
            "failed_stage": failed_stage,
            "message": f"{MTC_COMPOUND_TASK_SERVICE} did not return a structured response.",
            "blocker": blocker or f"{MTC_COMPOUND_TASK_SERVICE} is unavailable or returned non-JSON data.",
            "correction": correction or "Start the Vizor MTC compound backend service and retry; do not treat this as a solved task.",
            "robot_name": robot,
            "object_name": object_name,
            "task_goal": task_goal,
            "requirements": dict(requirements),
            "preferences": dict(preferences),
            "stage_intents": list(stage_intents),
            "target_pose": target_pose,
            "target_position": target_position,
            "task_stages": [],
            "candidate_attempts": [],
            "candidate_count": 0,
            "selected_cost": None,
            "scene_snapshot": {},
            "object_context": {},
            "selected_stage_evidence": [],
            "selected_grasp_evidence": {},
            "selected_place_evidence": {},
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": False,
            },
        }

    def attach_object(
        self,
        *,
        robot: str,
        object_name: str,
        link_name: str = DEFAULT_ATTACH_LINK,
        touch_links: list[str] | None = None,
        timeout_s: float = 2.0,
    ) -> AttachSceneFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_planning_scene; /{robot}/apply_planning_scene"
            payload = self.transport.read_planning_scene(robot, timeout_s)
            if not isinstance(payload, dict):
                return AttachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="planning scene unavailable",
                    planning_frame=None,
                    link_name=link_name,
                    touch_links=list(touch_links or DEFAULT_TOUCH_LINKS),
                    scene_update_published=False,
                    source=source,
                    message="MoveIt planning scene service did not return scene geometry",
                )

            scene = payload.get("scene", payload)
            collision_object = _find_world_collision_object(scene, object_name)
            planning_frame = payload.get("planning_frame")
            if not isinstance(planning_frame, str) or not planning_frame:
                planning_frame = _collision_object_frame(collision_object) if collision_object is not None else "base_link"
            if collision_object is None:
                return AttachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="object not found",
                    planning_frame=planning_frame,
                    link_name=link_name,
                    touch_links=list(touch_links or DEFAULT_TOUCH_LINKS),
                    scene_update_published=False,
                    source=source,
                    message=f"Object {object_name!r} was not found in the MoveIt planning scene",
                )

            touch = list(touch_links or DEFAULT_TOUCH_LINKS)
            attached_object = deepcopy(collision_object)
            attached_object["operation"] = 0
            remove_object = {
                "id": object_name,
                "header": {"frame_id": _collision_object_frame(collision_object) or planning_frame or "base_link"},
                "operation": COLLISION_OBJECT_REMOVE,
            }
            planning_scene_diff = {
                "name": "",
                "robot_state": {
                    "is_diff": True,
                    "attached_collision_objects": [
                        {
                            "link_name": link_name,
                            "object": attached_object,
                            "touch_links": touch,
                        }
                    ],
                },
                "world": {"collision_objects": [remove_object]},
                "is_diff": True,
            }
            scene_applied = self.transport.apply_planning_scene(robot, planning_scene_diff, timeout_s)
            verified_payload = self.transport.read_planning_scene(robot, timeout_s)
            verified_context: dict[str, Any] | None = None
            if isinstance(verified_payload, dict):
                verified_scene = summarize_planning_scene(verified_payload)
                verified_context = object_context(verified_scene, object_name)
            scene_verified = (
                isinstance(verified_context, dict)
                and verified_context.get("state") == "attached"
                and verified_context.get("attached_to") == link_name
            )
            if not scene_verified:
                apply_status = "confirmed" if scene_applied else "not confirmed"
                return AttachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="attached collision object unverified",
                    planning_frame=planning_frame,
                    link_name=link_name,
                    touch_links=touch,
                    scene_update_published=False,
                    source=source,
                    message=(
                        f"MoveIt apply_planning_scene was {apply_status}, but /{robot}/get_planning_scene "
                        f"did not verify {object_name} attached to {link_name}"
                    ),
                )
            return AttachSceneFeedback(
                robot=robot,
                object_name=object_name,
                ok=True,
                status="attached collision object verified",
                planning_frame=planning_frame,
                link_name=link_name,
                touch_links=touch,
                scene_update_published=True,
                source=source,
                message=f"Verified MoveIt planning-scene attachment of {object_name} to {link_name}",
            )

    def detach_object(
        self,
        *,
        robot: str,
        object_name: str,
        object_pose: Pose,
        link_name: str = DEFAULT_ATTACH_LINK,
        timeout_s: float = 2.0,
    ) -> DetachSceneFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_planning_scene; /{robot}/apply_planning_scene"
            payload = self.transport.read_planning_scene(robot, timeout_s)
            if not isinstance(payload, dict):
                return DetachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="planning scene unavailable",
                    planning_frame=None,
                    link_name=link_name,
                    scene_update_published=False,
                    source=source,
                    message="MoveIt planning scene service did not return scene geometry",
                )

            scene = payload.get("scene", payload)
            attached = _find_attached_collision_object(scene, object_name)
            planning_frame = payload.get("planning_frame")
            if not isinstance(planning_frame, str) or not planning_frame:
                attached_object_value = attached.get("object") if isinstance(attached, dict) else None
                attached_object = (
                    attached_object_value if isinstance(attached_object_value, dict) else None
                )
                planning_frame = _collision_object_frame(attached_object) or "base_link"
            if attached is None:
                return DetachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="attached object not found",
                    planning_frame=planning_frame,
                    link_name=link_name,
                    scene_update_published=False,
                    source=source,
                    message=f"Object {object_name!r} was not attached in the MoveIt planning scene",
                )

            attached_link = attached.get("link_name")
            if not isinstance(attached_link, str) or not attached_link:
                attached_link = link_name
            collision_object_value = attached.get("object")
            collision_object: dict[str, Any] = (
                collision_object_value if isinstance(collision_object_value, dict) else {}
            )
            world_object = _released_collision_object(collision_object, object_pose, planning_frame)
            remove_attached_object = {
                "link_name": attached_link,
                "object": {"id": object_name, "operation": COLLISION_OBJECT_REMOVE},
                "touch_links": list(attached.get("touch_links") or []),
            }
            planning_scene_diff = {
                "name": "",
                "robot_state": {
                    "is_diff": True,
                    "attached_collision_objects": [remove_attached_object],
                },
                "world": {"collision_objects": [world_object]},
                "is_diff": True,
            }
            scene_applied = self.transport.apply_planning_scene(robot, planning_scene_diff, timeout_s)
            verified_payload = self.transport.read_planning_scene(robot, timeout_s)
            verified_context: dict[str, Any] | None = None
            if isinstance(verified_payload, dict):
                verified_scene = summarize_planning_scene(verified_payload)
                verified_context = object_context(verified_scene, object_name)
            scene_verified = isinstance(verified_context, dict) and verified_context.get("state") == "free"
            if not scene_verified:
                apply_status = "confirmed" if scene_applied else "not confirmed"
                return DetachSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="released collision object unverified",
                    planning_frame=planning_frame,
                    link_name=attached_link,
                    scene_update_published=False,
                    source=source,
                    message=(
                        f"MoveIt apply_planning_scene was {apply_status}, but /{robot}/get_planning_scene "
                        f"did not verify {object_name} released as a free object"
                    ),
                )
            return DetachSceneFeedback(
                robot=robot,
                object_name=object_name,
                ok=True,
                status="released collision object verified",
                planning_frame=planning_frame,
                link_name=attached_link,
                scene_update_published=True,
                source=source,
                message=f"Verified MoveIt planning-scene release of {object_name} from {attached_link}",
            )

    def remove_scene_object(
        self,
        *,
        robot: str,
        object_name: str,
        timeout_s: float = 2.0,
    ) -> RemoveSceneFeedback:
        with self._lock_for(robot):
            source = f"/{robot}/get_planning_scene; /{robot}/apply_planning_scene"
            payload = self.transport.read_planning_scene(robot, timeout_s)
            if not isinstance(payload, dict):
                return RemoveSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="planning scene unavailable",
                    planning_frame=None,
                    scene_update_published=False,
                    source=source,
                    message="MoveIt planning scene service did not return scene geometry",
                )

            scene = payload.get("scene", payload)
            collision_object = _find_world_collision_object(scene, object_name)
            attached = _find_attached_collision_object(scene, object_name)
            planning_frame = payload.get("planning_frame")
            if not isinstance(planning_frame, str) or not planning_frame:
                planning_frame = _collision_object_frame(collision_object) or "base_link"
            if attached is not None:
                return RemoveSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="object attached",
                    planning_frame=planning_frame,
                    scene_update_published=False,
                    source=source,
                    message=(
                        f"Object {object_name!r} is attached; release and verify it before "
                        "removing it from the planning scene"
                    ),
                )
            if collision_object is None:
                return RemoveSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="object not found",
                    planning_frame=planning_frame,
                    scene_update_published=False,
                    source=source,
                    message=f"Object {object_name!r} was not found as a free MoveIt planning-scene object",
                )

            remove_object = {
                "id": object_name,
                "header": {"frame_id": _collision_object_frame(collision_object) or planning_frame or "base_link"},
                "operation": COLLISION_OBJECT_REMOVE,
            }
            planning_scene_diff = {
                "name": "",
                "world": {"collision_objects": [remove_object]},
                "is_diff": True,
            }
            scene_applied = self.transport.apply_planning_scene(robot, planning_scene_diff, timeout_s)
            verified_payload = self.transport.read_planning_scene(robot, timeout_s)
            scene_verified = False
            if isinstance(verified_payload, dict):
                verified_scene = summarize_planning_scene(verified_payload)
                scene_verified = object_name not in available_object_names(verified_scene)
            if not scene_verified:
                apply_status = "confirmed" if scene_applied else "not confirmed"
                return RemoveSceneFeedback(
                    robot=robot,
                    object_name=object_name,
                    ok=False,
                    status="scene object removal unverified",
                    planning_frame=planning_frame,
                    scene_update_published=False,
                    source=source,
                    message=(
                        f"MoveIt apply_planning_scene was {apply_status}, but /{robot}/get_planning_scene "
                        f"still listed {object_name}"
                    ),
                )
            return RemoveSceneFeedback(
                robot=robot,
                object_name=object_name,
                ok=True,
                status="scene object removed",
                planning_frame=planning_frame,
                scene_update_published=True,
                source=source,
                message=f"Verified removal of planning-scene object {object_name}",
            )

    def plan_free_motion(self, *, robot: str, name: str, pose: Pose, timeout_s: float = 10.0) -> PlanFeedback:
        with self._lock_for(robot):
            status_topic = f"/{robot}/request/status"
            path_topic = f"/{robot}/request/planned_path"
            self.transport.prepare_for_plan(status_topic, path_topic)
            self.transport.publish(
                f"/{robot}/request/free",
                "vizor_package/PlanningFree",
                {"name": name, "target_pose": pose.to_msg()},
            )
            return self._wait_for_plan(robot=robot, name=name, timeout_s=timeout_s)

    def plan_cartesian_motion(self, *, robot: str, name: str, poses: list[Pose], timeout_s: float = 10.0) -> PlanFeedback:
        with self._lock_for(robot):
            status_topic = f"/{robot}/request/status"
            path_topic = f"/{robot}/request/planned_path"
            self.transport.prepare_for_plan(status_topic, path_topic)
            self.transport.publish(
                f"/{robot}/request/cartesian",
                "vizor_package/PlanningCartesian",
                {"name": name, "poses": [pose.to_msg() for pose in poses]},
            )
            return self._wait_for_plan(robot=robot, name=name, timeout_s=timeout_s)

    def plan_sampled_motion(self, *, robot: str, name: str, poses: list[Pose], timeout_s: float = 10.0) -> PlanFeedback:
        with self._lock_for(robot):
            status_topic = f"/{robot}/request/status"
            path_topic = f"/{robot}/request/planned_path"
            self.transport.prepare_for_plan(status_topic, path_topic)
            self.transport.publish(
                f"/{robot}/request/sampled",
                "vizor_package/PlanningCartesian",
                {"name": name, "poses": [pose.to_msg() for pose in poses]},
            )
            return self._wait_for_plan(robot=robot, name=name, timeout_s=timeout_s)

    def register_agent_path(
        self,
        *,
        robot: str,
        task_solution_id: str,
        stage_debug_names: list[str],
        final_joint_positions: list[float] | None = None,
        joint_names: list[str] | None = None,
    ) -> None:
        with self._lock_for(robot):
            self._active_agent_paths[robot] = {
                "name": AGENT_PATH_NAME,
                "task_solution_id": task_solution_id,
                "stage_debug_names": list(stage_debug_names),
            }
            self._planned_final_positions[(robot, AGENT_PATH_NAME)] = (
                list(final_joint_positions) if final_joint_positions is not None else None
            )
            self._planned_joint_names[(robot, AGENT_PATH_NAME)] = (
                list(joint_names) if joint_names is not None else None
            )
            self._invalidated_agent_paths.discard(robot)

    def stop_agent_path(self, *, robot: str) -> dict[str, Any]:
        with self._lock_for(robot):
            self.transport.publish(f"/{robot}/command/stop", "std_msgs/String", {"data": AGENT_PATH_NAME})
            self._active_agent_paths.pop(robot, None)
            self._planned_final_positions.pop((robot, AGENT_PATH_NAME), None)
            self._planned_joint_names.pop((robot, AGENT_PATH_NAME), None)
            self._invalidated_agent_paths.add(robot)
            return {
                "ok": True,
                "robot": robot,
                "name": AGENT_PATH_NAME,
                "status": "agent path invalidated",
                "requires_reobserve_replan": True,
            }

    def execute_plan(self, *, robot: str, name: str, timeout_s: float = 10.0) -> ExecuteFeedback:
        with self._lock_for(robot):
            if name == AGENT_PATH_NAME and (
                robot in self._invalidated_agent_paths or robot not in self._active_agent_paths
            ):
                return ExecuteFeedback(robot, name, "AgentPath requires re-observe/replan", None, False, None, None, False)
            physical_mode = self.transport.read_physical_mode(PHYSICAL_PARAM)
            if physical_mode is None:
                return ExecuteFeedback(robot, name, "physical mode unknown", None, False, None, self._planned_final_positions.get((robot, name)), False)
            if physical_mode is True:
                return ExecuteFeedback(robot, name, "physical mode enabled", True, False, None, self._planned_final_positions.get((robot, name)), False)

            expected = self._planned_final_positions.get((robot, name))
            if expected is None:
                return ExecuteFeedback(robot, name, "plan final state unavailable", False, False, None, None, False)
            expected_names = self._planned_joint_names.get((robot, name))

            joint_topic = f"/{robot}/move_group/fake_controller_joint_states"
            self.transport.prepare_for_execute(joint_topic)
            self.transport.publish(f"/{robot}/command/execute", "std_msgs/String", {"data": name})
            observed, matches, observed_names = self._wait_for_matching_joint_state(
                topic=joint_topic,
                expected=expected,
                expected_names=expected_names,
                timeout_s=timeout_s,
            )
            status = "final joint state matched" if matches else "execution unverified"
            return ExecuteFeedback(
                robot,
                name,
                status,
                False,
                True,
                observed,
                expected,
                matches,
                observed_names,
                expected_names,
            )

    def command_gripper(self, *, robot: str, state: str, timeout_s: float = 5.0) -> GripperFeedback:
        if state not in {"open", "closed"}:
            raise ValueError(f"Unsupported gripper state: {state}")

        with self._lock_for(robot):
            goal_position_m = 0.0 if state == "closed" else 0.085
            expected_joint_position = 0.8 if state == "closed" else 0.0
            speed_mps = 0.05
            force = 50.0
            action_name = f"/{robot}/command_robotiq_action"
            action_type = "robotiq_2f_gripper_msgs/CommandRobotiqGripperAction"
            joint_state_topic = f"/{robot}/gripper_joint_states"
            goal = {
                "emergency_release": False,
                "emergency_release_dir": 0,
                "stop": False,
                "position": goal_position_m,
                "speed": speed_mps,
                "force": force,
            }

            self.transport.prepare_for_execute(joint_state_topic)
            action_result = self.transport.send_action_goal(action_name, action_type, goal, timeout_s)
            observed_joint_state, joint_matches, _ = self._wait_for_matching_joint_state(
                topic=joint_state_topic,
                expected=[expected_joint_position],
                expected_names=None,
                timeout_s=timeout_s,
            )
            observed_joint_position = observed_joint_state[0] if observed_joint_state else None
            ok = action_result is not None and joint_matches
            return GripperFeedback(
                robot=robot,
                state=state,
                action_name=action_name,
                action_type=action_type,
                joint_state_topic=joint_state_topic,
                goal_position_m=goal_position_m,
                speed_mps=speed_mps,
                force=force,
                expected_joint_position=expected_joint_position,
                observed_joint_position=observed_joint_position,
                action_result=action_result,
                command_sent=True,
                ok=ok,
            )

    def _wait_for_gripper_state(self, *, topic: str, expected: bool, deadline: float) -> bool | None:
        last_observed: bool | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return last_observed
            observed = self.transport.wait_for_bool(topic, remaining)
            if observed is None:
                return last_observed
            last_observed = observed
            if observed is expected:
                return observed

    def _wait_for_task_status(self, *, topic: str, task_id: int, deadline: float) -> str | None:
        prefix = f"{task_id}_"
        last_observed: str | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return last_observed
            observed = self.transport.wait_for_status(topic, remaining)
            if observed is None:
                return last_observed
            last_observed = observed
            if observed.startswith(prefix):
                return observed

    def _wait_for_matching_joint_state(
        self,
        *,
        topic: str,
        expected: list[float],
        expected_names: list[str] | None,
        timeout_s: float,
    ) -> tuple[list[float] | None, bool, list[str] | None]:
        deadline = time.monotonic() + timeout_s
        last_observed: list[float] | None = None
        last_names: list[str] | None = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return last_observed, False, last_names
            msg = _wait_for_joint_state_message(self.transport, topic, remaining)
            if msg is None:
                return last_observed, False, last_names
            observed = _joint_state_positions(msg)
            observed_names = _joint_state_names(msg)
            last_observed = observed
            last_names = observed_names
            if _positions_match(expected, observed, self.joint_tolerance, expected_names, observed_names):
                return observed, True, observed_names

    def _wait_for_plan(self, *, robot: str, name: str, timeout_s: float) -> PlanFeedback:
        path = self.transport.wait_for_planned_path(f"/{robot}/request/planned_path", name, timeout_s)
        status = self.transport.wait_for_status(f"/{robot}/request/status", timeout_s)
        points_list = ((path or {}).get("joint_trajectory") or {}).get("points") or []
        points = len(points_list)
        final_positions = _final_positions(path)
        joint_names = _joint_names(path)
        status_value = status or "timed out"
        can_execute = status_value in SUCCESS_STATUSES and points > 0 and final_positions is not None
        self._planned_final_positions[(robot, name)] = final_positions if can_execute else None
        self._planned_joint_names[(robot, name)] = joint_names if can_execute else None
        return PlanFeedback(
            robot=robot,
            name=name,
            status=status_value,
            trajectory_points=points,
            can_execute=can_execute,
            raw_path=path,
            final_joint_positions=final_positions,
        )


def _parse_bool_param_value(value: Any) -> bool | None:
    if value in {True, "true", "True", "1", "yes"}:
        return True
    if value in {False, "false", "False", "0", "no"}:
        return False
    return None


def _pending_after_publish_plan_feedback(
    statuses: dict[str, Queue[str]],
    paths: dict[str, Queue[dict[str, Any]]],
) -> bool:
    return any(not queue.empty() for queue in statuses.values()) or any(
        not queue.empty() for queue in paths.values()
    )


def _find_world_collision_object(scene: dict[str, Any], object_name: str) -> dict[str, Any] | None:
    world_value = scene.get("world")
    world = world_value if isinstance(world_value, dict) else {}
    for collision_object in world.get("collision_objects") or []:
        if isinstance(collision_object, dict) and collision_object.get("id") == object_name:
            return collision_object
    return None


def _find_attached_collision_object(scene: dict[str, Any], object_name: str) -> dict[str, Any] | None:
    robot_state_value = scene.get("robot_state")
    robot_state = robot_state_value if isinstance(robot_state_value, dict) else {}
    for attached in robot_state.get("attached_collision_objects") or []:
        if not isinstance(attached, dict):
            continue
        collision_object = attached.get("object")
        if isinstance(collision_object, dict) and collision_object.get("id") == object_name:
            return attached
    return None


def _released_collision_object(
    collision_object: dict[str, Any],
    object_pose: Pose,
    planning_frame: str,
) -> dict[str, Any]:
    released = deepcopy(collision_object)
    released["id"] = str(collision_object.get("id") or "")
    released["header"] = {"frame_id": planning_frame}
    released["operation"] = COLLISION_OBJECT_ADD
    delta = _release_translation(collision_object, object_pose)
    if released.get("primitives"):
        released["primitive_poses"] = [
            _translated_shape_pose(collision_object.get("primitive_poses"), index, delta)
            for index, _ in enumerate(released.get("primitives", []))
        ]
    if released.get("meshes"):
        released["mesh_poses"] = [
            _translated_shape_pose(collision_object.get("mesh_poses"), index, delta)
            for index, _ in enumerate(released.get("meshes", []))
        ]
    if released.get("planes"):
        released["plane_poses"] = [
            _translated_shape_pose(collision_object.get("plane_poses"), index, delta)
            for index, _ in enumerate(released.get("planes", []))
        ]
    return released


def _release_translation(collision_object: dict[str, Any], object_pose: Pose) -> dict[str, float]:
    center = _collision_object_center(collision_object) or {"x": 0.0, "y": 0.0, "z": 0.0}
    return {
        axis: float(object_pose.position[axis]) - float(center[axis])
        for axis in ("x", "y", "z")
    }


def _collision_object_center(collision_object: dict[str, Any]) -> dict[str, float] | None:
    scene = summarize_planning_scene(
        {
            "scene": {
                "world": {"collision_objects": [collision_object]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        }
    )
    objects = scene.get("objects") or []
    if not objects:
        return None
    bounds = objects[0].get("bounds")
    center = bounds.get("center") if isinstance(bounds, dict) else None
    if not isinstance(center, dict):
        return None
    return {axis: float(center[axis]) for axis in ("x", "y", "z")}


def _translated_shape_pose(poses: Any, index: int, delta: dict[str, float]) -> dict[str, Any]:
    pose = _shape_pose_at(poses, index)
    return {
        "position": {
            axis: float(pose["position"][axis]) + delta[axis]
            for axis in ("x", "y", "z")
        },
        "orientation": dict(pose["orientation"]),
    }


def _shape_pose_at(poses: Any, index: int) -> dict[str, Any]:
    if isinstance(poses, list) and index < len(poses) and isinstance(poses[index], dict):
        raw_pose: dict[str, Any] = poses[index]
    else:
        raw_pose = {}
    position_raw = raw_pose.get("position")
    position_value: dict[str, Any] = position_raw if isinstance(position_raw, dict) else {}
    orientation_raw = raw_pose.get("orientation")
    orientation_value: dict[str, Any] = orientation_raw if isinstance(orientation_raw, dict) else {}
    return {
        "position": {
            "x": float(position_value.get("x", 0.0)),
            "y": float(position_value.get("y", 0.0)),
            "z": float(position_value.get("z", 0.0)),
        },
        "orientation": {
            "x": float(orientation_value.get("x", 0.0)),
            "y": float(orientation_value.get("y", 0.0)),
            "z": float(orientation_value.get("z", 0.0)),
            "w": float(orientation_value.get("w", 1.0)),
        },
    }


def _apply_planning_scene_diff(current_payload: dict[str, Any], diff_payload: dict[str, Any]) -> dict[str, Any]:
    updated = deepcopy(current_payload)
    scene_value = updated.get("scene")
    scene: dict[str, Any] = scene_value if isinstance(scene_value, dict) else updated
    diff_value = diff_payload.get("scene")
    diff: dict[str, Any] = diff_value if isinstance(diff_value, dict) else diff_payload

    world_value = scene.get("world")
    world: dict[str, Any] = world_value if isinstance(world_value, dict) else {}
    scene["world"] = world
    collision_objects = [
        deepcopy(item)
        for item in world.get("collision_objects", [])
        if isinstance(item, dict)
    ]
    diff_world = diff.get("world")
    diff_collision_objects = (
        diff_world.get("collision_objects") if isinstance(diff_world, dict) else []
    )
    for collision_object in diff_collision_objects or []:
        if not isinstance(collision_object, dict):
            continue
        object_id = collision_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            continue
        operation = collision_object.get("operation", COLLISION_OBJECT_ADD)
        collision_objects = [item for item in collision_objects if item.get("id") != object_id]
        if operation != COLLISION_OBJECT_REMOVE:
            collision_objects.append(deepcopy(collision_object))
    world["collision_objects"] = collision_objects

    robot_state_value = scene.get("robot_state")
    robot_state: dict[str, Any] = robot_state_value if isinstance(robot_state_value, dict) else {}
    scene["robot_state"] = robot_state
    attached_objects = [
        deepcopy(item)
        for item in robot_state.get("attached_collision_objects", [])
        if isinstance(item, dict)
    ]
    diff_robot_state = diff.get("robot_state")
    diff_attached_objects = (
        diff_robot_state.get("attached_collision_objects")
        if isinstance(diff_robot_state, dict)
        else []
    )
    for attached in diff_attached_objects or []:
        if not isinstance(attached, dict):
            continue
        collision_object_value = attached.get("object")
        if not isinstance(collision_object_value, dict):
            continue
        collision_object: dict[str, Any] = collision_object_value
        object_id = collision_object.get("id")
        if not isinstance(object_id, str) or not object_id:
            continue
        attached_objects = [
            item
            for item in attached_objects
            if not (isinstance(item.get("object"), dict) and item["object"].get("id") == object_id)
        ]
        operation = collision_object.get("operation", COLLISION_OBJECT_ADD)
        if operation != COLLISION_OBJECT_REMOVE:
            attached_objects.append(deepcopy(attached))
    robot_state["attached_collision_objects"] = attached_objects

    return updated


def _collision_object_frame(collision_object: dict[str, Any] | None) -> str | None:
    if collision_object is None:
        return None
    header_value = collision_object.get("header")
    header = header_value if isinstance(header_value, dict) else {}
    frame = header.get("frame_id")
    return frame if isinstance(frame, str) and frame else None


def _planned_path_msg(
    name: str,
    points: int,
    final_positions: list[float] | None,
    joint_names: list[str] | None = None,
) -> dict[str, Any]:
    trajectory_points: list[dict[str, Any]] = []
    for index in range(points):
        positions = final_positions if index == points - 1 and final_positions is not None else []
        trajectory_points.append({"positions": list(positions)})
    return {
        "name": name,
        "platform_name": "",
        "joint_trajectory": {"joint_names": list(joint_names or []), "points": trajectory_points},
    }


def _final_positions(path: dict[str, Any] | None) -> list[float] | None:
    points = (((path or {}).get("joint_trajectory") or {}).get("points") or [])
    if not points:
        return None
    positions = points[-1].get("positions") or []
    if not positions:
        return None
    return [float(value) for value in positions]


def _joint_names(path: dict[str, Any] | None) -> list[str] | None:
    names = (((path or {}).get("joint_trajectory") or {}).get("joint_names") or [])
    if not names:
        return None
    return [str(name) for name in names]


def _positions_match(
    expected: list[float] | None,
    observed: list[float] | None,
    tolerance: float,
    expected_names: list[str] | None = None,
    observed_names: list[str] | None = None,
) -> bool:
    if expected is None or observed is None:
        return False
    if (
        expected_names is not None
        and observed_names is not None
        and len(expected_names) == len(expected)
        and len(observed_names) == len(observed)
    ):
        observed_by_name = dict(zip(observed_names, observed, strict=True))
        if len(observed_by_name) == len(observed_names) and all(name in observed_by_name for name in expected_names):
            return all(abs(expected[index] - observed_by_name[name]) <= tolerance for index, name in enumerate(expected_names))
    if len(expected) != len(observed):
        return False
    return all(abs(a - b) <= tolerance for a, b in zip(expected, observed))


def _joint_state_msg(positions: list[float], names: list[str] | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {"position": list(positions)}
    if names is not None:
        payload["name"] = list(names)
    return payload


def _wait_for_joint_state_message(transport: RosbridgeTransport, topic: str, timeout_s: float) -> dict[str, Any] | None:
    wait_message = getattr(transport, "wait_for_joint_state_message", None)
    if callable(wait_message):
        msg = wait_message(topic, timeout_s)
        return msg if isinstance(msg, dict) else None
    positions = transport.wait_for_joint_state(topic, timeout_s)
    return {"position": positions} if positions is not None else None


def _joint_state_positions(msg: Any) -> list[float]:
    if isinstance(msg, list):
        return [float(value) for value in msg]
    return [float(value) for value in (msg.get("position") or msg.get("positions") or [])]


def _joint_state_names(msg: Any) -> list[str] | None:
    if not isinstance(msg, dict):
        return None
    names = msg.get("name") or msg.get("names") or []
    return [str(value) for value in names] if names else None


def _bool_value(msg: Any) -> bool | None:
    if isinstance(msg, bool):
        return msg
    value = msg.get("data") if hasattr(msg, "get") else getattr(msg, "data", None)
    return value if isinstance(value, bool) else None


def _new_task_id() -> int:
    return int(time.time() * 1000) % 2_000_000_000


def _gripper_task_payload(*, task_id: int, robot: str, task_name: str) -> dict[str, Any]:
    return {
        "id": task_id,
        "target": robot,
        "type": "sequential_individual",
        "skill": "gripper",
        "deadline": 0,
        "name": task_name,
        "instruction": task_name,
        "trajectory": {
            "joint_trajectory": {
                "header": {"seq": 0, "stamp": {"secs": 0, "nsecs": 0}, "frame_id": ""},
                "joint_names": [],
                "points": [],
            },
            "mesh_trajectory": {"triangles": [], "vertices": []},
            "platform_name": robot,
        },
        "zone": {"identifier": "", "zone_ids": [], "alert_distance": 0.0, "boundaries": []},
        "content": {
            "operation": "",
            "layer": "",
            "name": "",
            "geometries": [],
            "wires": [],
            "texts": [],
            "LoD": 0,
        },
    }
