from __future__ import annotations

import socket
import threading
import time
from collections import OrderedDict
from enum import Enum
from typing import Union


class RobotiqGripper:
    ACT = "ACT"
    GTO = "GTO"
    ATR = "ATR"
    FOR = "FOR"
    SPE = "SPE"
    POS = "POS"
    STA = "STA"
    PRE = "PRE"
    OBJ = "OBJ"
    ENCODING = "UTF-8"

    class GripperStatus(Enum):
        RESET = 0
        ACTIVATING = 1
        ACTIVE = 3

    class ObjectStatus(Enum):
        MOVING = 0
        STOPPED_OUTER_OBJECT = 1
        STOPPED_INNER_OBJECT = 2
        AT_DEST = 3

    def __init__(self) -> None:
        self.socket: socket.socket | None = None
        self.command_lock = threading.Lock()
        self._min_position = 0
        self._max_position = 255
        self._min_speed = 0
        self._max_speed = 255
        self._min_force = 0
        self._max_force = 255

    def connect(self, hostname: str, port: int, socket_timeout: float = 2.0) -> None:
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((hostname, port))
        self.socket.settimeout(socket_timeout)

    def disconnect(self) -> None:
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def activate(self, auto_calibrate: bool = True, timeout: float = 10.0) -> None:
        if not self.is_active():
            self._reset(timeout)
            self._set_var(self.ACT, 1)
            time.sleep(1.0)
            deadline = time.time() + timeout
            while self._get_var(self.ACT) != 1 or self._get_var(self.STA) != 3:
                if time.time() >= deadline:
                    raise TimeoutError(f"Gripper activation timed out after {timeout}s")
                time.sleep(0.01)

        if auto_calibrate:
            self.auto_calibrate()

    def is_active(self) -> bool:
        status = self._get_var(self.STA)
        return RobotiqGripper.GripperStatus(status) is RobotiqGripper.GripperStatus.ACTIVE

    def get_current_position(self) -> int:
        return self._get_var(self.POS)

    def auto_calibrate(self) -> None:
        position, status = self.move_and_wait_for_pos(self._min_position, 64, 1)
        if status is not RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed opening to start: {status}")

        position, status = self.move_and_wait_for_pos(self._max_position, 64, 1)
        if status is not RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        self._max_position = position

        position, status = self.move_and_wait_for_pos(self._min_position, 64, 1)
        if status is not RobotiqGripper.ObjectStatus.AT_DEST:
            raise RuntimeError(f"Calibration failed because of an object: {status}")
        self._min_position = position

    def move(self, position: int, speed: int, force: int) -> tuple[bool, int]:
        clipped_position = _clip(position, self._min_position, self._max_position)
        clipped_speed = _clip(speed, self._min_speed, self._max_speed)
        clipped_force = _clip(force, self._min_force, self._max_force)
        values: OrderedDict[str, int | float] = OrderedDict(
            [
                (self.POS, clipped_position),
                (self.SPE, clipped_speed),
                (self.FOR, clipped_force),
                (self.GTO, 1),
            ]
        )
        return self._set_vars(values), clipped_position

    def move_and_wait_for_pos(
        self,
        position: int,
        speed: int,
        force: int,
        timeout: float = 10.0,
    ) -> tuple[int, ObjectStatus]:
        set_ok, requested_position = self.move(position, speed, force)
        if not set_ok:
            raise RuntimeError("Failed to set variables for gripper move.")

        deadline = time.time() + timeout
        while self._get_var(self.PRE) != requested_position:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Gripper move timed out waiting for acknowledgement after {timeout}s"
                )
            time.sleep(0.001)

        object_status = self._get_var(self.OBJ)
        while RobotiqGripper.ObjectStatus(object_status) is RobotiqGripper.ObjectStatus.MOVING:
            if time.time() >= deadline:
                raise TimeoutError(
                    f"Gripper move timed out waiting for completion after {timeout}s"
                )
            object_status = self._get_var(self.OBJ)

        return self._get_var(self.POS), RobotiqGripper.ObjectStatus(object_status)

    def _reset(self, timeout: float) -> None:
        self._set_var(self.ACT, 0)
        self._set_var(self.ATR, 0)
        deadline = time.time() + timeout
        while self._get_var(self.ACT) != 0 or self._get_var(self.STA) != 0:
            if time.time() >= deadline:
                raise TimeoutError(f"Gripper reset timed out after {timeout}s")
            self._set_var(self.ACT, 0)
            self._set_var(self.ATR, 0)
            time.sleep(0.01)
        time.sleep(0.5)

    def _set_vars(self, values: OrderedDict[str, Union[int, float]]) -> bool:
        command = "SET" + "".join(f" {variable} {value}" for variable, value in values.items()) + "\n"
        with self.command_lock:
            sock = self._socket()
            sock.sendall(command.encode(self.ENCODING))
            response = sock.recv(1024)
        return response == b"ack"

    def _set_var(self, variable: str, value: Union[int, float]) -> bool:
        return self._set_vars(OrderedDict([(variable, value)]))

    def _get_var(self, variable: str) -> int:
        with self.command_lock:
            sock = self._socket()
            sock.sendall(f"GET {variable}\n".encode(self.ENCODING))
            response = sock.recv(1024)
        name, value = response.decode(self.ENCODING).split()
        if name != variable:
            raise ValueError(f"Unexpected gripper response: {response!r}")
        return int(value)

    def _socket(self) -> socket.socket:
        if self.socket is None:
            raise RuntimeError("Gripper is not connected")
        return self.socket


def create_gripper() -> RobotiqGripper:
    return RobotiqGripper()


def _clip(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(value, maximum))
