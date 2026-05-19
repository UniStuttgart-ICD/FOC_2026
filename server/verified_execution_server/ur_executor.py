from __future__ import annotations

import socket
import time
from collections.abc import Callable
from threading import RLock
from typing import Any, Protocol, cast

GRIPPER_MAX_RAW_POSITION = 255
GRIPPER_MAX_JOINT_POSITION = 0.8


class TrajectoryExecutor(Protocol):
    def execute(self, robot_name: str, frames: list[dict]) -> dict[str, object] | None: ...
    def go_home(self, robot_name: str) -> dict[str, object] | None: ...
    def read_state(self, robot_name: str) -> dict[str, object] | None: ...
    def control_gripper(self, robot_name: str, action: str) -> None: ...


class URScriptProgramSender(Protocol):
    def send_program(self, program: str) -> None: ...


class URScriptSocketClient:
    def __init__(self, *, robot_ip: str, port: int = 30002, timeout_s: float = 3.0) -> None:
        self.robot_ip = robot_ip
        self.port = port
        self.timeout_s = timeout_s

    def send_program(self, program: str) -> None:
        payload = program.encode("utf-8")
        with socket.create_connection(
            (self.robot_ip, self.port),
            timeout=self.timeout_s,
        ) as sock:
            sock.sendall(payload)


class URRTDETrajectoryExecutor:
    HOME_JOINTS = [
        -0.05903655687441045,
        -1.5698241536486712,
        1.529440704976217,
        -0.0015873473933716298,
        1.4997673034667969,
        0.0008195281261578202,
    ]

    def __init__(
        self,
        *,
        robot_ip: str,
        robot_port: int = 30004,
        script_port: int = 30002,
        socket_timeout_s: float = 3.0,
        joint_speed: float = 1.05,
        joint_accel: float = 1.4,
        joint_blend: float = 0.02,
        servo_lookahead_time: float = 0.1,
        servo_gain: float = 300.0,
        skip_gripper: bool = False,
        gripper_port: int = 63352,
        gripper_speed: int = 255,
        gripper_force: int = 255,
        robot_factory: Callable[..., Any] | None = None,
        gripper_factory: Callable[..., Any] | None = None,
        rtde_receive_factory: Callable[..., Any] | None = None,
        script_sender: URScriptProgramSender | None = None,
        completion_timeout_s: float = 60.0,
        completion_poll_interval_s: float = 0.1,
        joint_tolerance_rad: float = 0.03,
        completion_stable_samples: int = 2,
    ) -> None:
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.script_port = script_port
        self.socket_timeout_s = socket_timeout_s
        self.joint_speed = joint_speed
        self.joint_accel = joint_accel
        self.joint_blend = joint_blend
        self.servo_lookahead_time = servo_lookahead_time
        self.servo_gain = servo_gain
        self.skip_gripper = skip_gripper
        self.gripper_port = gripper_port
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self._robot_factory = robot_factory
        self._gripper_factory = gripper_factory
        self._rtde_receive_factory = rtde_receive_factory
        self._script_sender = script_sender
        self.completion_timeout_s = completion_timeout_s
        self.completion_poll_interval_s = completion_poll_interval_s
        self.joint_tolerance_rad = joint_tolerance_rad
        self.completion_stable_samples = max(int(completion_stable_samples), 1)
        self._robots: dict[str, Any] = {}
        self._gripper: Any | None = None
        self._gripper_lock = RLock()

    def startup_check(self, robot_name: str) -> dict[str, object]:
        status: dict[str, object] = {
            "robot_name": robot_name,
        }
        status.update(self.check_robot_receive())
        if not self.skip_gripper:
            status.update(self.check_gripper())
        else:
            status.update(
                {
                    "gripper_connected": None,
                    "gripper_error": "gripper startup check skipped",
                }
            )
        return status

    def check_robot_receive(self) -> dict[str, object]:
        rtde_r = None
        try:
            rtde_r = self._rtde_receive()
            actual_q = rtde_r.getActualQ()
            return {
                "robot_connected": bool(actual_q),
                "robot_error": None if actual_q else "RTDE receive returned no joints",
            }
        except Exception as exc:
            return {"robot_connected": False, "robot_error": str(exc)}
        finally:
            disconnect = getattr(rtde_r, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def check_gripper(self) -> dict[str, object]:
        try:
            gripper = self._direct_gripper()
            position = None
            get_current_position = getattr(gripper, "get_current_position", None)
            if callable(get_current_position):
                raw_position: Any = get_current_position()
                position = int(raw_position)
            return {
                "gripper_connected": True,
                "gripper_error": None,
                "gripper_position": position,
            }
        except Exception as exc:
            self._close_direct_gripper()
            return {
                "gripper_connected": False,
                "gripper_error": str(exc),
                "gripper_position": None,
            }

    def execute(self, robot_name: str, frames: list[dict]) -> dict[str, object] | None:
        if self._robot_factory is None:
            return self._send_joint_trajectory_program(frames)
        robot = self._robot_for(robot_name)
        timed_execute = getattr(robot, "execute_timed_joint_trajectory", None)
        if callable(timed_execute):
            return cast(dict[str, object] | None, timed_execute(frames))
        return cast(dict[str, object] | None, robot.execute(frames))

    def go_home(self, robot_name: str) -> dict[str, object] | None:
        if self._robot_factory is None:
            target = list(self.HOME_JOINTS)
            self._send_movej_program([{"positions": target}])
            return self._wait_for_joint_target(target)
        robot = self._robot_for(robot_name)
        move_j = getattr(robot, "move_j", None)
        if callable(move_j):
            if move_j(list(self.HOME_JOINTS), self.joint_speed, self.joint_accel) is False:
                raise RuntimeError("Home move failed")
            return

        go_to_home = getattr(robot, "go_to_home", None)
        if callable(go_to_home):
            if go_to_home() is False:
                raise RuntimeError("Home move failed")
            return
        raise RuntimeError("Robot does not expose a home command")

    def read_state(self, robot_name: str) -> dict[str, object] | None:
        rtde_r = None
        try:
            rtde_r = self._rtde_receive()
            actual_q = _float_list(rtde_r.getActualQ())
            if not actual_q:
                raise RuntimeError("RTDE receive returned no joints")
            gripper_position = self._read_gripper_position()
            gripper_joint_position = _gripper_joint_position(gripper_position)
            state: dict[str, object] = {
                "actual_joint_positions": actual_q,
                "actual_gripper_position": gripper_position,
                "actual_gripper_joint_position": gripper_joint_position,
            }
            get_actual_tcp_pose = getattr(rtde_r, "getActualTCPPose", None)
            if callable(get_actual_tcp_pose):
                actual_tcp_pose = _float_list(get_actual_tcp_pose())
                if actual_tcp_pose:
                    state["actual_tcp_pose"] = actual_tcp_pose
            return state
        finally:
            disconnect = getattr(rtde_r, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def control_gripper(self, robot_name: str, action: str) -> None:
        if not self.skip_gripper:
            self._control_direct_gripper(action)
            return
        raise RuntimeError("Robot gripper is disabled")

    def stop(self, robot_name: str) -> None:
        if self._robot_factory is None:
            self._send_program("def verified_execution_stop():\n  stopj(2.0)\nend\n")
            return
        robot = self._robots.get(robot_name)
        if robot is None:
            return
        stop_movement = getattr(robot, "stop_movement", None)
        if callable(stop_movement):
            stop_movement()
        else:
            setattr(robot, "stop", True)

    def close(self) -> None:
        for robot in self._robots.values():
            shutdown = getattr(robot, "shutdown", None)
            if shutdown is not None:
                shutdown()
        self._robots.clear()
        self._close_direct_gripper()

    def _robot_for(self, robot_name: str):
        robot = self._robots.get(robot_name)
        if robot is not None:
            return robot

        if self._robot_factory is None:
            raise RuntimeError("robot_factory is required for adapter-backed execution")
        robot_factory = self._robot_factory

        robot = robot_factory(
            name=robot_name,
            robot_ip=self.robot_ip,
            robot_port=self.robot_port,
            joint_speed=self.joint_speed,
            joint_accel=self.joint_accel,
            joint_blend=self.joint_blend,
            servo_lookahead_time=self.servo_lookahead_time,
            servo_gain=self.servo_gain,
            skip_gripper=True,
        )
        self._robots[robot_name] = robot
        return robot

    def _rtde_receive(self):
        rtde_receive_factory = self._rtde_receive_factory
        if rtde_receive_factory is None:
            import rtde_receive

            rtde_receive_factory = rtde_receive.RTDEReceiveInterface
        return rtde_receive_factory(self.robot_ip)

    def _control_direct_gripper(self, action: str) -> None:
        if action not in {"open", "close"}:
            raise RuntimeError(f"Unsupported gripper action: {action}")
        target = 0 if action == "open" else 255
        with self._gripper_lock:
            gripper = self._direct_gripper()
            gripper.move_and_wait_for_pos(
                target,
                self.gripper_speed,
                self.gripper_force,
            )

    def _read_gripper_position(self) -> int:
        if self.skip_gripper:
            raise RuntimeError("Robot gripper is disabled")
        with self._gripper_lock:
            gripper = self._direct_gripper()
            get_current_position = getattr(gripper, "get_current_position", None)
            if not callable(get_current_position):
                raise RuntimeError("Robot gripper does not expose current position")
            raw_position = cast(Any, get_current_position())
        try:
            position = int(raw_position)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Robotiq gripper returned invalid position: {raw_position!r}") from exc
        if position < 0 or position > GRIPPER_MAX_RAW_POSITION:
            raise RuntimeError(f"Robotiq gripper returned out-of-range position: {position}")
        return position

    def _direct_gripper(self):
        with self._gripper_lock:
            if self._gripper is not None:
                return self._gripper

            gripper_factory = self._gripper_factory
            if gripper_factory is None:
                from verified_execution_server.robotiq_client import create_gripper

                gripper_factory = create_gripper
            gripper = gripper_factory()
            gripper.connect(self.robot_ip, self.gripper_port)
            activate = getattr(gripper, "activate", None)
            if callable(activate):
                try:
                    activate(auto_calibrate=False)
                except TypeError:
                    activate()
            self._gripper = gripper
            return gripper

    def _close_direct_gripper(self) -> None:
        with self._gripper_lock:
            gripper = self._gripper
            self._gripper = None
        disconnect = getattr(gripper, "disconnect", None)
        if callable(disconnect):
            disconnect()

    def _send_joint_trajectory_program(self, frames: list[dict]) -> dict[str, object]:
        trajectory = self._joint_trajectory_frames(frames)
        if not trajectory:
            raise RuntimeError("Trajectory has no joint positions")
        if any("time_from_start_s" in frame for frame in trajectory):
            self._send_servoj_program(trajectory)
        else:
            self._send_movej_program(trajectory)
        return self._wait_for_joint_target(trajectory[-1]["positions"])

    def _send_servoj_program(self, trajectory: list[dict[str, Any]]) -> None:
        lines = ["def verified_execution_trajectory():"]
        previous_time = 0.0
        for frame in trajectory:
            time_from_start_s = max(float(frame.get("time_from_start_s", previous_time)), previous_time)
            control_time_s = max(time_from_start_s - previous_time, 0.008)
            previous_time = time_from_start_s
            lines.append(
                "  servoj("
                f"{_urscript_list(frame['positions'])}, "
                "0, 0, "
                f"{_urscript_float(control_time_s)}, "
                f"{_urscript_float(self.servo_lookahead_time)}, "
                f"{_urscript_float(self.servo_gain)}"
                ")"
            )
        lines.append("  stopj(2.0)")
        lines.append("end")
        self._send_program("\n".join(lines) + "\n")

    def _send_movej_program(self, trajectory: list[dict[str, Any]]) -> None:
        lines = ["def verified_execution_trajectory():"]
        for index, frame in enumerate(trajectory):
            blend = self.joint_blend if index < len(trajectory) - 1 else 0.0
            lines.append(
                "  movej("
                f"{_urscript_list(frame['positions'])}, "
                f"a={_urscript_float(self.joint_accel)}, "
                f"v={_urscript_float(self.joint_speed)}, "
                "t=0, "
                f"r={_urscript_float(blend)}"
                ")"
            )
        lines.append("end")
        self._send_program("\n".join(lines) + "\n")

    def _send_program(self, program: str) -> None:
        sender = self._script_sender
        if sender is None:
            sender = URScriptSocketClient(
                robot_ip=self.robot_ip,
                port=self.script_port,
                timeout_s=self.socket_timeout_s,
            )
        sender.send_program(program)

    def _wait_for_joint_target(self, target_positions: list[float]) -> dict[str, object]:
        target = [float(value) for value in target_positions[:6]]
        if len(target) != 6:
            raise RuntimeError(f"Expected six target joints, got {len(target_positions)}")

        rtde_r = None
        stable_samples = 0
        actual: list[float] = []
        max_error = float("inf")
        deadline = time.monotonic() + self.completion_timeout_s
        try:
            rtde_r = self._rtde_receive()

            while True:
                actual_q = rtde_r.getActualQ()
                actual = [float(value) for value in list(actual_q)[:6]]
                if len(actual) != 6:
                    raise RuntimeError(f"RTDE receive returned {len(actual)} joints")

                max_error = max(abs(expected - observed) for expected, observed in zip(target, actual))
                if max_error <= self.joint_tolerance_rad:
                    stable_samples += 1
                    if stable_samples >= self.completion_stable_samples:
                        return {
                            "target_joint_positions": target,
                            "final_joint_positions": actual,
                            "max_joint_error": max_error,
                            "joint_tolerance_rad": self.joint_tolerance_rad,
                        }
                else:
                    stable_samples = 0

                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "URScript target not reached: "
                        f"max_joint_error={max_error:.6g}, "
                        f"joint_tolerance_rad={self.joint_tolerance_rad:.6g}, "
                        f"target={target}, actual={actual}"
                    )
                if self.completion_poll_interval_s > 0:
                    time.sleep(self.completion_poll_interval_s)
        finally:
            disconnect = getattr(rtde_r, "disconnect", None)
            if callable(disconnect):
                disconnect()

    def _joint_trajectory_frames(self, frames: list[dict]) -> list[dict[str, Any]]:
        trajectory: list[dict[str, Any]] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            positions = frame.get("positions")
            if not isinstance(positions, list) or not positions:
                continue
            cleaned = dict(frame)
            cleaned["positions"] = [float(value) for value in positions]
            trajectory.append(cleaned)
        return trajectory


def _urscript_float(value: float) -> str:
    return f"{float(value):.9g}"


def _urscript_list(values: list[float]) -> str:
    return "[" + ", ".join(_urscript_float(value) for value in values) + "]"


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _gripper_joint_position(raw_position: int) -> float:
    return float(raw_position) / GRIPPER_MAX_RAW_POSITION * GRIPPER_MAX_JOINT_POSITION
