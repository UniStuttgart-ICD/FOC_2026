from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import verified_execution_server.server as server_module
from verified_execution_server.models import CachedPlan, ExecutePlanRequest
from verified_execution_server.plan_cache import RosPlanCache
from verified_execution_server.server import create_app, create_default_app
from verified_execution_server.ur_executor import URRTDETrajectoryExecutor


class FakePlanCache:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.plans: dict[tuple[str, str], CachedPlan] = {}
        self.sync_calls: list[tuple[str, list[str], list[float]]] = []
        self.gripper_sync_calls: list[tuple[str, str, float]] = []
        self.release_calls: list[tuple[str, float]] = []
        self.sync_result = True
        self.gripper_sync_result = True
        self.release_result = SimpleNamespace(
            ok=True,
            status="no_attached_objects",
            error=None,
            correction=None,
            checked=True,
            attached_objects_before_release=[],
            attached_objects_released=[],
            published=False,
            verified=True,
            topic_or_service="/UR10/apply_planning_scene",
        )

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    def get_plan(self, robot_name: str, plan_name: str) -> CachedPlan | None:
        return self.plans.get((robot_name, plan_name))

    def size(self) -> int:
        return len(self.plans)

    def is_connected(self) -> bool:
        return True

    def sync_joint_state(
        self,
        robot_name: str,
        *,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> bool:
        self.sync_calls.append((robot_name, joint_names, joint_positions))
        return self.sync_result

    def sync_gripper_joint_state(
        self,
        robot_name: str,
        *,
        joint_name: str,
        joint_position: float,
    ) -> bool:
        self.gripper_sync_calls.append((robot_name, joint_name, joint_position))
        return self.gripper_sync_result

    def release_attached_objects(
        self,
        robot_name: str,
        *,
        timeout_s: float,
    ) -> SimpleNamespace:
        self.release_calls.append((robot_name, timeout_s))
        return self.release_result


class FailingPlanCache(FakePlanCache):
    async def start(self) -> None:
        self.started = True
        raise RuntimeError("rosbridge unavailable")

    def is_connected(self) -> bool:
        return False


class FakeExecutor:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []
        self.home_calls: list[str] = []
        self.state_calls: list[str] = []
        self.gripper_calls: list[tuple[str, str]] = []
        self.startup_check_calls: list[str] = []
        self.execute_result: dict[str, object] | None = None
        self.home_result: dict[str, object] | None = None
        self.state_result: dict[str, object] | None = None

    def execute(self, robot_name: str, frames: list[dict]) -> dict[str, object] | None:
        self.calls.append((robot_name, frames))
        return self.execute_result

    def startup_check(self, robot_name: str) -> dict[str, object]:
        self.startup_check_calls.append(robot_name)
        return {
            "robot_name": robot_name,
            "robot_connected": True,
            "gripper_connected": True,
            "gripper_position": 0,
        }

    def go_home(self, robot_name: str) -> dict[str, object] | None:
        self.home_calls.append(robot_name)
        return self.home_result

    def read_state(self, robot_name: str) -> dict[str, object] | None:
        self.state_calls.append(robot_name)
        return self.state_result

    def control_gripper(self, robot_name: str, action: str) -> None:
        self.gripper_calls.append((robot_name, action))


class FakeScriptSender:
    def __init__(self) -> None:
        self.programs: list[str] = []

    def send_program(self, program: str) -> None:
        self.programs.append(program)


class FakeRTDEReceive:
    def __init__(self, host: str, samples: list[list[float]]) -> None:
        self.host = host
        self.samples = list(samples)
        self.calls = 0
        self.disconnected = False

    def getActualQ(self) -> list[float]:
        self.calls += 1
        if len(self.samples) > 1:
            return self.samples.pop(0)
        return self.samples[0]

    def disconnect(self) -> None:
        self.disconnected = True


def test_default_app_reads_rtde_completion_wait_settings_from_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_kwargs: dict[str, object] = {}

    class CapturingExecutor:
        def __init__(self, **kwargs: object) -> None:
            executor_kwargs.update(kwargs)

    class DummyPlanCache(FakePlanCache):
        def __init__(self, **_: object) -> None:
            super().__init__()

    monkeypatch.setattr(server_module, "URRTDETrajectoryExecutor", CapturingExecutor)
    monkeypatch.setattr(server_module, "RosPlanCache", DummyPlanCache)
    monkeypatch.setenv("UR_COMPLETION_TIMEOUT_S", "42.5")
    monkeypatch.setenv("UR_COMPLETION_POLL_INTERVAL_S", "0.25")
    monkeypatch.setenv("UR_JOINT_TOLERANCE_RAD", "0.015")
    monkeypatch.setenv("UR_COMPLETION_STABLE_SAMPLES", "4")

    create_default_app()

    assert executor_kwargs["completion_timeout_s"] == 42.5
    assert executor_kwargs["completion_poll_interval_s"] == 0.25
    assert executor_kwargs["joint_tolerance_rad"] == 0.015
    assert executor_kwargs["completion_stable_samples"] == 4


def test_default_app_uses_sixty_second_rtde_completion_wait_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executor_kwargs: dict[str, object] = {}

    class CapturingExecutor:
        def __init__(self, **kwargs: object) -> None:
            executor_kwargs.update(kwargs)

    class DummyPlanCache(FakePlanCache):
        def __init__(self, **_: object) -> None:
            super().__init__()

    monkeypatch.setattr(server_module, "URRTDETrajectoryExecutor", CapturingExecutor)
    monkeypatch.setattr(server_module, "RosPlanCache", DummyPlanCache)
    monkeypatch.delenv("UR_COMPLETION_TIMEOUT_S", raising=False)

    create_default_app()

    assert executor_kwargs["completion_timeout_s"] == 60.0


def test_execute_plan_request_defaults_to_sixty_second_timeout() -> None:
    assert ExecutePlanRequest(plan_name="plan-1").timeout_s == 60.0


def test_health_reports_ros_cache_state() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    cache.plans[("UR10", "plan-1")] = CachedPlan(
        robot_name="UR10",
        plan_name="plan-1",
        frames=[{"positions": [0, -1.57, 1.57, 0, 0, 0]}],
        observed_at_s=10.0,
    )

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "ros_connected": True,
        "cached_plans": 1,
        "robot": {
            "robot_name": "UR10",
            "robot_connected": True,
            "gripper_connected": True,
            "robot_error": None,
            "gripper_error": None,
            "gripper_position": 0,
        },
    }
    assert cache.started is True
    assert cache.stopped is True
    assert executor.startup_check_calls == ["UR10"]


def test_home_still_runs_when_ros_plan_cache_start_fails() -> None:
    cache = FailingPlanCache()
    executor = FakeExecutor()

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        health = client.get("/health")
        home = client.post("/home", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert health.status_code == 200
    assert health.json()["ros_connected"] is False
    assert home.status_code == 200
    assert home.json()["status"] == "homed"
    assert executor.home_calls == ["UR10"]


def test_execute_plan_runs_cached_trajectory_through_executor() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    frames = [
        {"positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]},
        {"positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0]},
    ]
    cache.plans[("UR10", "plan-1")] = CachedPlan(
        robot_name="UR10",
        plan_name="plan-1",
        frames=frames,
        observed_at_s=10.0,
    )

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post(
            "/execute",
            json={"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "robot_name": "UR10",
        "plan_name": "plan-1",
        "status": "executed",
        "trajectory_points": 2,
        "verification_result": "pass",
    }
    assert executor.calls == [("UR10", frames)]
    assert cache.sync_calls == []


def test_execute_plan_syncs_fake_controller_state_after_physical_result() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    frames = [
        {"positions": [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]},
        {"positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0]},
    ]
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    final_positions = [0.101, -1.471, 1.481, 0.101, 0.001, -0.001]
    target_positions = [0.1, -1.47, 1.48, 0.1, 0.0, 0.0]
    cache.plans[("UR10", "plan-1")] = CachedPlan(
        robot_name="UR10",
        plan_name="plan-1",
        frames=frames,
        joint_names=joint_names,
        observed_at_s=10.0,
    )
    executor.execute_result = {
        "target_joint_positions": target_positions,
        "final_joint_positions": final_positions,
        "max_joint_error": 0.001,
        "joint_tolerance_rad": 0.01,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post(
            "/execute",
            json={"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "executed"
    assert body["target_joint_positions"] == target_positions
    assert body["final_joint_positions"] == final_positions
    assert body["max_joint_error"] == 0.001
    assert body["joint_tolerance_rad"] == 0.01
    assert body["state_sync_published"] is True
    assert cache.sync_calls == [("UR10", joint_names, final_positions)]


def test_execute_plan_fails_when_fake_controller_state_sync_fails() -> None:
    cache = FakePlanCache()
    cache.sync_result = False
    executor = FakeExecutor()
    joint_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    final_positions = [0.101, -1.471, 1.481, 0.101, 0.001, -0.001]
    cache.plans[("UR10", "plan-1")] = CachedPlan(
        robot_name="UR10",
        plan_name="plan-1",
        frames=[{"positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0]}],
        joint_names=joint_names,
        observed_at_s=10.0,
    )
    executor.execute_result = {
        "target_joint_positions": [0.1, -1.47, 1.48, 0.1, 0.0, 0.0],
        "final_joint_positions": final_positions,
        "max_joint_error": 0.001,
        "joint_tolerance_rad": 0.01,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post(
            "/execute",
            json={"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
        )

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "state_sync_failed"
    assert body["verification_result"] == "fail"
    assert body["final_joint_positions"] == final_positions
    assert body["state_sync_published"] is False
    assert "fake controller" in body["error"]
    assert cache.sync_calls == [("UR10", joint_names, final_positions)]


def test_execute_plan_rejects_unknown_cached_plan() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post(
            "/execute",
            json={"robot_name": "UR10", "plan_name": "missing-plan"},
        )

    assert response.status_code == 404
    assert response.json()["ok"] is False
    assert response.json()["error"] == "No cached trajectory for plan."
    assert response.json()["correction"] == "Plan again, then retry execution."
    assert executor.calls == []


def test_home_runs_robot_home_through_executor() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/home", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "robot_name": "UR10",
        "command": "home",
        "status": "homed",
        "error": None,
        "correction": None,
    }
    assert executor.home_calls == ["UR10"]


def test_home_syncs_fake_controller_state_after_physical_home_result() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    final_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    executor.home_result = {
        "target_joint_positions": final_positions,
        "final_joint_positions": final_positions,
        "max_joint_error": 0.0,
        "joint_tolerance_rad": 0.03,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/home", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "homed"
    assert body["final_joint_positions"] == final_positions
    assert body["state_sync_published"] is True
    assert cache.sync_calls == [
        (
            "UR10",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            final_positions,
        )
    ]


def test_home_reports_state_sync_failure_after_physical_home() -> None:
    cache = FakePlanCache()
    cache.sync_result = False
    executor = FakeExecutor()
    final_positions = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    executor.home_result = {"final_joint_positions": final_positions}

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/home", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "state_sync_failed"
    assert body["final_joint_positions"] == final_positions
    assert body["state_sync_published"] is False
    assert "fake controller" in body["error"]


def test_sync_state_reads_real_joints_and_publishes_fake_controller_state() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    actual_positions = [0.2, -1.4, 1.3, 0.1, 0.0, -0.2]
    actual_tcp_pose = [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]
    executor.state_result = {
        "actual_joint_positions": actual_positions,
        "actual_tcp_pose": actual_tcp_pose,
        "actual_gripper_position": 128,
        "actual_gripper_joint_position": 128.0 / 255.0 * 0.8,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "state_synced"
    assert body["actual_joint_positions"] == actual_positions
    assert body["actual_tcp_pose"] == actual_tcp_pose
    assert body["state_sync_published"] is True
    assert body["actual_gripper_position"] == 128
    assert body["actual_gripper_joint_position"] == pytest.approx(128.0 / 255.0 * 0.8)
    assert body["gripper_joint_state_published"] is True
    assert body["gripper_joint_name"] == "finger_joint"
    assert body["gripper_joint_state_topic"] == "/UR10/gripper_joint_states"
    assert executor.state_calls == ["UR10"]
    assert cache.sync_calls == [
        (
            "UR10",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "finger_joint",
            ],
            [*actual_positions, pytest.approx(128.0 / 255.0 * 0.8)],
        )
    ]
    assert cache.gripper_sync_calls == [
        ("UR10", "finger_joint", pytest.approx(128.0 / 255.0 * 0.8))
    ]
    assert body["gripper_open_threshold_position"] == 10
    assert body["gripper_considered_open"] is False
    assert body["attached_object_release_checked"] is False
    assert body["attached_objects_before_release"] == []
    assert body["attached_objects_released"] == []
    assert body["attached_object_release_published"] is False
    assert body["attached_object_release_verified"] is False
    assert body["attached_object_release_topic_or_service"] == "/UR10/apply_planning_scene"
    assert cache.release_calls == []


def test_sync_state_fails_when_rtde_returns_no_joints() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    executor.state_result = {"actual_tcp_pose": [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]}

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "missing_joint_state"
    assert body["actual_tcp_pose"] == [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]
    assert body["error"] == "RTDE receive did not return actual joint positions."
    assert cache.sync_calls == []


def test_sync_state_fails_when_fake_controller_state_publish_fails() -> None:
    cache = FakePlanCache()
    cache.sync_result = False
    executor = FakeExecutor()
    actual_positions = [0.2, -1.4, 1.3, 0.1, 0.0, -0.2]
    executor.state_result = {
        "actual_joint_positions": actual_positions,
        "actual_gripper_position": 255,
        "actual_gripper_joint_position": 0.8,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "state_sync_failed"
    assert body["actual_joint_positions"] == actual_positions
    assert body["state_sync_published"] is False
    assert body["gripper_joint_state_published"] is False
    assert "fake controller" in body["error"]
    assert cache.sync_calls == [
        (
            "UR10",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "finger_joint",
            ],
            [*actual_positions, 0.8],
        )
    ]
    assert cache.gripper_sync_calls == []


def test_sync_state_fails_when_gripper_state_is_missing() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "missing_gripper_state"
    assert body["state_sync_published"] is False
    assert body["gripper_joint_state_published"] is False
    assert "Robotiq gripper" in body["error"]
    assert cache.sync_calls == []
    assert cache.gripper_sync_calls == []


def test_sync_state_fails_when_gripper_joint_state_publish_fails() -> None:
    cache = FakePlanCache()
    cache.gripper_sync_result = False
    executor = FakeExecutor()
    actual_positions = [0.2, -1.4, 1.3, 0.1, 0.0, -0.2]
    executor.state_result = {
        "actual_joint_positions": actual_positions,
        "actual_gripper_position": 255,
        "actual_gripper_joint_position": 0.8,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "state_sync_failed"
    assert body["state_sync_published"] is True
    assert body["gripper_joint_state_published"] is False
    assert body["actual_gripper_position"] == 255
    assert body["actual_gripper_joint_position"] == 0.8
    assert "gripper joint state sync failed" in body["error"]
    assert cache.sync_calls == [
        (
            "UR10",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
                "finger_joint",
            ],
            [*actual_positions, 0.8],
        )
    ]
    assert cache.gripper_sync_calls == [("UR10", "finger_joint", 0.8)]
    assert cache.release_calls == []


def test_sync_state_releases_attached_objects_when_gripper_is_open() -> None:
    cache = FakePlanCache()
    cache.release_result = SimpleNamespace(
        ok=True,
        status="released collision objects verified",
        error=None,
        correction=None,
        checked=True,
        attached_objects_before_release=["held_part"],
        attached_objects_released=["held_part"],
        published=True,
        verified=True,
        topic_or_service="/UR10/apply_planning_scene",
    )
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 0,
        "actual_gripper_joint_position": 0.0,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "state_synced"
    assert body["state_sync_published"] is True
    assert body["gripper_joint_state_published"] is True
    assert body["gripper_open_threshold_position"] == 10
    assert body["gripper_considered_open"] is True
    assert body["attached_object_release_checked"] is True
    assert body["attached_objects_before_release"] == ["held_part"]
    assert body["attached_objects_released"] == ["held_part"]
    assert body["attached_object_release_published"] is True
    assert body["attached_object_release_verified"] is True
    assert body["attached_object_release_topic_or_service"] == "/UR10/apply_planning_scene"
    assert cache.release_calls == [("UR10", 5.0)]


def test_sync_state_succeeds_with_no_attached_objects_when_gripper_is_open() -> None:
    cache = FakePlanCache()
    cache.release_result = SimpleNamespace(
        ok=True,
        status="no_attached_objects",
        error=None,
        correction=None,
        checked=True,
        attached_objects_before_release=[],
        attached_objects_released=[],
        published=False,
        verified=True,
        topic_or_service="/UR10/apply_planning_scene",
    )
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 10,
        "actual_gripper_joint_position": 10.0 / 255.0 * 0.8,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "state_synced"
    assert body["gripper_considered_open"] is True
    assert body["attached_object_release_checked"] is True
    assert body["attached_objects_before_release"] == []
    assert body["attached_objects_released"] == []
    assert body["attached_object_release_published"] is False
    assert body["attached_object_release_verified"] is True
    assert cache.release_calls == [("UR10", 5.0)]


def test_sync_state_does_not_release_attached_objects_when_gripper_is_not_open() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 11,
        "actual_gripper_joint_position": 11.0 / 255.0 * 0.8,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "state_synced"
    assert body["gripper_considered_open"] is False
    assert body["attached_object_release_checked"] is False
    assert body["attached_objects_before_release"] == []
    assert body["attached_objects_released"] == []
    assert body["attached_object_release_published"] is False
    assert body["attached_object_release_verified"] is False
    assert cache.release_calls == []


def test_sync_state_fails_when_open_gripper_cannot_read_planning_scene() -> None:
    cache = FakePlanCache()
    cache.release_result = SimpleNamespace(
        ok=False,
        status="planning_scene_unavailable",
        error="MoveIt planning scene service did not return scene geometry.",
        correction="Check /UR10/get_planning_scene and retry state sync.",
        checked=True,
        attached_objects_before_release=[],
        attached_objects_released=[],
        published=False,
        verified=False,
        topic_or_service="/UR10/apply_planning_scene",
    )
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 0,
        "actual_gripper_joint_position": 0.0,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["ok"] is False
    assert body["status"] == "attached_object_release_failed"
    assert body["state_sync_published"] is True
    assert body["gripper_joint_state_published"] is True
    assert body["gripper_considered_open"] is True
    assert body["attached_object_release_checked"] is True
    assert body["attached_object_release_published"] is False
    assert body["attached_object_release_verified"] is False
    assert "planning scene" in body["error"]
    assert cache.release_calls == [("UR10", 5.0)]


def test_sync_state_fails_when_attached_object_release_readback_does_not_verify() -> None:
    cache = FakePlanCache()
    cache.release_result = SimpleNamespace(
        ok=False,
        status="released_collision_object_unverified",
        error="MoveIt apply_planning_scene did not verify held_part released.",
        correction="Check /UR10/get_planning_scene and /UR10/apply_planning_scene.",
        checked=True,
        attached_objects_before_release=["held_part"],
        attached_objects_released=["held_part"],
        published=True,
        verified=False,
        topic_or_service="/UR10/apply_planning_scene",
    )
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 0,
        "actual_gripper_joint_position": 0.0,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 409
    body = response.json()
    assert body["status"] == "attached_object_release_failed"
    assert body["attached_objects_before_release"] == ["held_part"]
    assert body["attached_objects_released"] == ["held_part"]
    assert body["attached_object_release_published"] is True
    assert body["attached_object_release_verified"] is False
    assert "verify held_part released" in body["error"]


def test_sync_state_releases_multiple_attached_objects_when_gripper_is_open() -> None:
    cache = FakePlanCache()
    cache.release_result = SimpleNamespace(
        ok=True,
        status="released collision objects verified",
        error=None,
        correction=None,
        checked=True,
        attached_objects_before_release=["held_a", "held_b"],
        attached_objects_released=["held_a", "held_b"],
        published=True,
        verified=True,
        topic_or_service="/UR10/apply_planning_scene",
    )
    executor = FakeExecutor()
    executor.state_result = {
        "actual_joint_positions": [0.2, -1.4, 1.3, 0.1, 0.0, -0.2],
        "actual_gripper_position": 0,
        "actual_gripper_joint_position": 0.0,
    }

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post("/sync_state", json={"robot_name": "UR10", "timeout_s": 5.0})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "state_synced"
    assert body["attached_objects_before_release"] == ["held_a", "held_b"]
    assert body["attached_objects_released"] == ["held_a", "held_b"]
    assert body["attached_object_release_verified"] is True


def test_gripper_runs_robot_gripper_action_through_executor() -> None:
    cache = FakePlanCache()
    executor = FakeExecutor()

    with TestClient(create_app(plan_cache=cache, executor=executor)) as client:
        response = client.post(
            "/gripper/open",
            json={"robot_name": "UR10", "timeout_s": 5.0},
        )

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "robot_name": "UR10",
        "command": "gripper_open",
        "status": "gripper_opened",
        "error": None,
        "correction": None,
    }
    assert executor.gripper_calls == [("UR10", "open")]


def test_ros_plan_cache_preserves_moveit_trajectory_timing() -> None:
    cache = RosPlanCache(robot_name="UR10", time_fn=lambda: 42.0)

    cache._record_planned_path(
        {
            "name": "smooth-plan",
            "joint_trajectory": {
                "points": [
                    {
                        "positions": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
                        "velocities": [0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
                        "accelerations": [0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
                        "time_from_start": {"secs": 0, "nsecs": 100_000_000},
                    },
                    {
                        "positions": [0.1, -0.9, 1.1, 0.1, 0.0, 0.0],
                        "velocities": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
                        "accelerations": [0.2, 0.0, -0.2, 0.0, 0.0, 0.0],
                        "time_from_start": {"secs": 0, "nsecs": 300_000_000},
                    },
                ]
            },
        }
    )

    plan = cache.get_plan("UR10", "smooth-plan")

    assert plan is not None
    assert plan.joint_names is None
    assert plan.frames == [
        {
            "positions": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            "velocities": [0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
            "accelerations": [0.0, 0.2, 0.2, 0.0, 0.0, 0.0],
            "time_from_start_s": 0.1,
        },
        {
            "positions": [0.1, -0.9, 1.1, 0.1, 0.0, 0.0],
            "velocities": [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            "accelerations": [0.2, 0.0, -0.2, 0.0, 0.0, 0.0],
            "time_from_start_s": 0.3,
        },
    ]


def test_ros_plan_cache_records_joint_names_from_moveit_trajectory() -> None:
    cache = RosPlanCache(robot_name="UR10", time_fn=lambda: 42.0)

    cache._record_planned_path(
        {
            "name": "named-joints-plan",
            "joint_trajectory": {
                "joint_names": ["joint_1", "joint_2"],
                "points": [{"positions": [0.0, 1.0]}],
            },
        }
    )

    plan = cache.get_plan("UR10", "named-joints-plan")

    assert plan is not None
    assert plan.joint_names == ["joint_1", "joint_2"]


def test_ros_plan_cache_syncs_gripper_joint_state_to_gripper_topic() -> None:
    published: list[tuple[str, str, dict]] = []

    class FakeTopic:
        def __init__(self, _client: object, topic: str, message_type: str) -> None:
            self.topic = topic
            self.message_type = message_type

        def publish(self, message: dict) -> None:
            published.append((self.topic, self.message_type, message))

    fake_roslibpy = SimpleNamespace(Topic=FakeTopic, Message=lambda payload: payload)
    cache = RosPlanCache(robot_name="UR10", time_fn=lambda: 42.5)
    cache._roslibpy = fake_roslibpy
    cache._client = SimpleNamespace(is_connected=True)
    cache._connected = True

    result = cache.sync_gripper_joint_state(
        "UR10",
        joint_name="finger_joint",
        joint_position=0.4,
    )

    assert result is True
    assert published == [
        (
            "/UR10/gripper_joint_states",
            "sensor_msgs/JointState",
            {
                "header": {
                    "stamp": {"secs": 42, "nsecs": 500_000_000},
                    "frame_id": "",
                },
                "name": ["finger_joint"],
                "position": [0.4],
                "velocity": [],
                "effort": [],
            },
        )
    ]


def test_ros_plan_cache_releases_attached_objects_to_world_scene() -> None:
    attached_object = {
        "id": "held_part",
        "header": {"frame_id": "tool0"},
        "primitives": [{"type": 1, "dimensions": [0.05, 0.04, 0.03]}],
        "primitive_poses": [
            {
                "position": {"x": 0.0, "y": 0.0, "z": 0.04},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        ],
        "meshes": [],
        "mesh_poses": [],
    }
    scenes = [
        {
            "scene": {
                "world": {"collision_objects": []},
                "robot_state": {
                    "attached_collision_objects": [
                        {
                            "link_name": "tool0",
                            "object": attached_object,
                            "touch_links": ["tool0"],
                        }
                    ]
                },
            }
        },
        {
            "scene": {
                "world": {"collision_objects": [{**attached_object, "operation": 0}]},
                "robot_state": {"attached_collision_objects": []},
            }
        },
    ]
    service_calls: list[tuple[str, str, dict, float]] = []

    class FakeService:
        def __init__(self, _client: object, name: str, service_type: str) -> None:
            self.name = name
            self.service_type = service_type

        def call(self, request: dict | None = None, *, timeout: float) -> dict:
            payload = dict(request or {})
            service_calls.append((self.name, self.service_type, payload, timeout))
            if self.name == "/UR10/get_planning_scene":
                return scenes.pop(0)
            if self.name == "/UR10/apply_planning_scene":
                return {"success": True}
            raise AssertionError(f"unexpected service: {self.name}")

    fake_roslibpy = SimpleNamespace(
        Service=FakeService,
        ServiceRequest=lambda payload=None: dict(payload or {}),
    )
    cache = RosPlanCache(robot_name="UR10")
    cache._roslibpy = fake_roslibpy
    cache._client = SimpleNamespace(is_connected=True)
    cache._connected = True

    result = cache.release_attached_objects("UR10", timeout_s=0.1)

    assert result.ok is True
    assert result.checked is True
    assert result.attached_objects_before_release == ["held_part"]
    assert result.attached_objects_released == ["held_part"]
    assert result.published is True
    assert result.verified is True
    apply_request = service_calls[1][2]["scene"]
    assert apply_request["robot_state"]["attached_collision_objects"] == [
        {
            "link_name": "tool0",
            "object": {"id": "held_part", "operation": 1},
            "touch_links": ["tool0"],
        }
    ]
    assert apply_request["world"]["collision_objects"] == [{**attached_object, "operation": 0}]


def test_ros_plan_cache_reports_release_unverified_when_readback_still_attached() -> None:
    attached_object = {
        "id": "held_part",
        "header": {"frame_id": "tool0"},
        "primitives": [{"type": 1, "dimensions": [0.05, 0.04, 0.03]}],
        "primitive_poses": [],
        "meshes": [],
        "mesh_poses": [],
    }
    attached_scene = {
        "scene": {
            "world": {"collision_objects": []},
            "robot_state": {
                "attached_collision_objects": [
                    {"link_name": "tool0", "object": attached_object, "touch_links": ["tool0"]}
                ]
            },
        }
    }
    scenes = [attached_scene, attached_scene]

    class FakeService:
        def __init__(self, _client: object, name: str, _service_type: str) -> None:
            self.name = name

        def call(self, request: dict | None = None, *, timeout: float) -> dict:
            if self.name == "/UR10/get_planning_scene":
                return scenes.pop(0)
            if self.name == "/UR10/apply_planning_scene":
                return {"success": True}
            raise AssertionError(f"unexpected service: {self.name}")

    fake_roslibpy = SimpleNamespace(
        Service=FakeService,
        ServiceRequest=lambda payload=None: dict(payload or {}),
    )
    cache = RosPlanCache(robot_name="UR10")
    cache._roslibpy = fake_roslibpy
    cache._client = SimpleNamespace(is_connected=True)
    cache._connected = True

    result = cache.release_attached_objects("UR10", timeout_s=0.1)

    assert result.ok is False
    assert result.checked is True
    assert result.attached_objects_before_release == ["held_part"]
    assert result.attached_objects_released == ["held_part"]
    assert result.published is True
    assert result.verified is False
    assert result.status == "released_collision_object_unverified"


@pytest.mark.parametrize(
    ("raw_position", "expected_joint_position"),
    [(0, 0.0), (255, 0.8), (128, 128.0 / 255.0 * 0.8)],
)
def test_ur_rtde_executor_read_state_maps_gripper_position_without_motion(
    raw_position: int,
    expected_joint_position: float,
) -> None:
    class FakeReceive:
        def __init__(self, host: str) -> None:
            self.host = host

        def getActualQ(self) -> list[float]:
            return [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]

        def getActualTCPPose(self) -> list[float]:
            return [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]

        def disconnect(self) -> None:
            pass

    class FakeGripper:
        def __init__(self) -> None:
            self.connect_calls: list[tuple[str, int]] = []
            self.activate_calls = 0
            self.move_calls: list[tuple[int, int, int]] = []

        def connect(self, host: str, port: int) -> None:
            self.connect_calls.append((host, port))

        def activate(self, **_: object) -> None:
            self.activate_calls += 1

        def get_current_position(self) -> int:
            return raw_position

        def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> None:
            self.move_calls.append((position, speed, force))

    gripper = FakeGripper()
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        rtde_receive_factory=FakeReceive,
        gripper_factory=lambda: gripper,
    )

    state = executor.read_state("UR10")

    assert state is not None
    assert state["actual_joint_positions"] == [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
    assert state["actual_tcp_pose"] == [0.4, -0.2, 0.3, 0.0, 3.14, 0.0]
    assert state["actual_gripper_position"] == raw_position
    assert state["actual_gripper_joint_position"] == pytest.approx(expected_joint_position)
    assert gripper.connect_calls == [("192.0.2.10", 63352)]
    assert gripper.activate_calls == 1
    assert gripper.move_calls == []


def test_ur_rtde_executor_prefers_timed_joint_trajectory() -> None:
    class FakeRobot:
        def __init__(self, **_: object) -> None:
            self.timed_calls: list[list[dict]] = []
            self.execute_calls: list[list[dict]] = []

        def execute_timed_joint_trajectory(self, frames: list[dict]) -> None:
            self.timed_calls.append(frames)

        def execute(self, frames: list[dict]) -> None:
            self.execute_calls.append(frames)

    robots: list[FakeRobot] = []

    def robot_factory(**kwargs: object) -> FakeRobot:
        robot = FakeRobot(**kwargs)
        robots.append(robot)
        return robot

    executor = URRTDETrajectoryExecutor(robot_ip="192.0.2.10", robot_factory=robot_factory)
    frames = [{"positions": [0.0] * 6, "time_from_start_s": 0.1}]

    executor.execute("UR10", frames)

    assert robots[0].timed_calls == [frames]
    assert robots[0].execute_calls == []


def test_ur_rtde_executor_runs_home_as_single_joint_move() -> None:
    class FakeRobot:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs
            self.move_j_calls: list[tuple[list[float], float, float]] = []

        def move_j(
            self,
            joint_positions: list[float],
            speed: float,
            acceleration: float,
        ) -> bool:
            self.move_j_calls.append((joint_positions, speed, acceleration))
            return True

    robots: list[FakeRobot] = []

    def robot_factory(**kwargs: object) -> FakeRobot:
        robot = FakeRobot(**kwargs)
        robots.append(robot)
        return robot

    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        joint_speed=0.5,
        joint_accel=0.75,
        robot_factory=robot_factory,
    )

    executor.go_home("UR10")

    assert robots[0].move_j_calls == [
        (
            [
                -0.05903655687441045,
                -1.5698241536486712,
                1.529440704976217,
                -0.0015873473933716298,
                1.4997673034667969,
                0.0008195281261578202,
            ],
            0.5,
            0.75,
        )
    ]
    assert robots[0].kwargs["skip_gripper"] is True


def test_ur_rtde_executor_sends_home_as_urscript_by_default() -> None:
    sender = FakeScriptSender()
    receive = FakeRTDEReceive(
        "192.0.2.10",
        [list(URRTDETrajectoryExecutor.HOME_JOINTS)] * 2,
    )
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        script_sender=sender,
        rtde_receive_factory=lambda host: receive,
    )

    result = executor.go_home("UR10")

    assert len(sender.programs) == 1
    assert "movej([" in sender.programs[0]
    assert "servoj(" not in sender.programs[0]
    assert result["target_joint_positions"] == list(URRTDETrajectoryExecutor.HOME_JOINTS)
    assert result["max_joint_error"] == 0.0
    assert receive.disconnected is True


def test_ur_rtde_executor_sends_timed_trajectory_as_one_urscript_program() -> None:
    sender = FakeScriptSender()
    receive = FakeRTDEReceive(
        "192.0.2.10",
        [[0.2, -0.8, 1.2, 0.2, 0.0, 0.0]] * 2,
    )
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        script_sender=sender,
        servo_lookahead_time=0.12,
        servo_gain=350.0,
        rtde_receive_factory=lambda host: receive,
    )

    result = executor.execute(
        "UR10",
        [
            {"positions": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0], "time_from_start_s": 0.0},
            {"positions": [0.2, -0.8, 1.2, 0.2, 0.0, 0.0], "time_from_start_s": 0.2},
        ],
    )

    assert len(sender.programs) == 1
    program = sender.programs[0]
    assert program.count("servoj(") == 2
    assert "[0.2, -0.8, 1.2, 0.2, 0, 0]" in program
    assert "0.12, 350" in program
    assert "stopj(2.0)" in program
    assert result["target_joint_positions"] == [0.2, -0.8, 1.2, 0.2, 0.0, 0.0]
    assert result["max_joint_error"] == 0.0


def test_ur_rtde_executor_waits_for_direct_urscript_final_joints() -> None:
    sender = FakeScriptSender()
    target = [0.2, -0.8, 1.2, 0.2, 0.0, 0.0]
    receive = FakeRTDEReceive(
        "192.0.2.10",
        [
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [0.15, -0.85, 1.15, 0.15, 0.0, 0.0],
            target,
            target,
        ],
    )
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        script_sender=sender,
        rtde_receive_factory=lambda host: receive,
        completion_poll_interval_s=0.0,
    )

    result = executor.execute(
        "UR10",
        [
            {"positions": [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]},
            {"positions": target},
        ],
    )

    assert len(sender.programs) == 1
    assert receive.calls == 4
    assert receive.disconnected is True
    assert result == {
        "target_joint_positions": target,
        "final_joint_positions": target,
        "max_joint_error": 0.0,
        "joint_tolerance_rad": 0.03,
    }


def test_ur_rtde_executor_raises_when_direct_urscript_target_is_not_reached() -> None:
    sender = FakeScriptSender()
    target = [0.2, -0.8, 1.2, 0.2, 0.0, 0.0]
    actual = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
    receive = FakeRTDEReceive("192.0.2.10", [actual])
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        script_sender=sender,
        rtde_receive_factory=lambda host: receive,
        completion_poll_interval_s=0.0,
        completion_timeout_s=0.0,
    )

    with pytest.raises(RuntimeError, match="URScript target not reached"):
        executor.execute("UR10", [{"positions": target}])

    assert len(sender.programs) == 1
    assert receive.disconnected is True


def test_ur_rtde_executor_waits_for_home_joints_on_direct_urscript_path() -> None:
    sender = FakeScriptSender()
    receive = FakeRTDEReceive(
        "192.0.2.10",
        [
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            list(URRTDETrajectoryExecutor.HOME_JOINTS),
            list(URRTDETrajectoryExecutor.HOME_JOINTS),
        ],
    )
    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        script_sender=sender,
        rtde_receive_factory=lambda host: receive,
        completion_poll_interval_s=0.0,
    )

    result = executor.go_home("UR10")

    assert len(sender.programs) == 1
    assert receive.calls == 3
    assert receive.disconnected is True
    assert result["target_joint_positions"] == list(URRTDETrajectoryExecutor.HOME_JOINTS)
    assert result["final_joint_positions"] == list(URRTDETrajectoryExecutor.HOME_JOINTS)


def test_ur_rtde_executor_runs_gripper_action_through_direct_socket_by_default() -> None:
    class FakeGripper:
        def __init__(self) -> None:
            self.connect_calls: list[tuple[str, int]] = []
            self.activate_calls: list[dict] = []
            self.move_calls: list[tuple[int, int, int]] = []

        def connect(self, host: str, port: int) -> None:
            self.connect_calls.append((host, port))

        def activate(self, **kwargs: object) -> None:
            self.activate_calls.append(kwargs)

        def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> None:
            self.move_calls.append((position, speed, force))

    grippers: list[FakeGripper] = []
    robot_factory_calls: list[dict] = []

    def gripper_factory() -> FakeGripper:
        gripper = FakeGripper()
        grippers.append(gripper)
        return gripper

    def robot_factory(**kwargs: object) -> object:
        robot_factory_calls.append(kwargs)
        raise AssertionError("gripper commands should not create a URRobot")

    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        robot_factory=robot_factory,
        gripper_factory=gripper_factory,
    )

    executor.control_gripper("UR10", "close")

    assert grippers[0].connect_calls == [("192.0.2.10", 63352)]
    assert grippers[0].activate_calls == [{"auto_calibrate": False}]
    assert grippers[0].move_calls == [(255, 255, 255)]
    assert robot_factory_calls == []


def test_ur_rtde_executor_does_not_fallback_to_rtde_robot_for_disabled_gripper() -> None:
    robot_factory_calls: list[dict] = []

    def robot_factory(**kwargs: object) -> object:
        robot_factory_calls.append(kwargs)
        raise AssertionError("disabled gripper commands should not create a URRobot")

    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        skip_gripper=True,
        robot_factory=robot_factory,
    )

    with pytest.raises(RuntimeError, match="gripper is disabled"):
        executor.control_gripper("UR10", "close")

    assert robot_factory_calls == []


def test_ur_rtde_executor_uses_direct_gripper_when_enabled() -> None:
    class FakeGripper:
        def __init__(self) -> None:
            self.connect_calls: list[tuple[str, int]] = []
            self.activate_calls: list[dict] = []
            self.move_calls: list[tuple[int, int, int]] = []

        def connect(self, host: str, port: int) -> None:
            self.connect_calls.append((host, port))

        def activate(self, **kwargs: object) -> None:
            self.activate_calls.append(kwargs)

        def get_current_position(self) -> int:
            return 17

        def move_and_wait_for_pos(self, position: int, speed: int, force: int) -> None:
            self.move_calls.append((position, speed, force))

    class FakeReceive:
        def __init__(self, host: str) -> None:
            self.host = host
            self.disconnected = False

        def getActualQ(self) -> list[float]:
            return [0.0] * 6

        def disconnect(self) -> None:
            self.disconnected = True

    grippers: list[FakeGripper] = []
    robot_factory_calls: list[dict] = []

    def gripper_factory() -> FakeGripper:
        gripper = FakeGripper()
        grippers.append(gripper)
        return gripper

    def robot_factory(**kwargs: object) -> object:
        robot_factory_calls.append(kwargs)
        raise AssertionError("gripper commands should not create a URRobot")

    executor = URRTDETrajectoryExecutor(
        robot_ip="192.0.2.10",
        skip_gripper=False,
        gripper_factory=gripper_factory,
        robot_factory=robot_factory,
        rtde_receive_factory=FakeReceive,
    )

    status = executor.startup_check("UR10")
    executor.control_gripper("UR10", "open")
    executor.control_gripper("UR10", "close")

    assert status == {
        "robot_name": "UR10",
        "robot_connected": True,
        "robot_error": None,
        "gripper_connected": True,
        "gripper_error": None,
        "gripper_position": 17,
    }
    assert grippers[0].connect_calls == [("192.0.2.10", 63352)]
    assert grippers[0].activate_calls == [{"auto_calibrate": False}]
    assert grippers[0].move_calls == [(0, 255, 255), (255, 255, 255)]
    assert robot_factory_calls == []
