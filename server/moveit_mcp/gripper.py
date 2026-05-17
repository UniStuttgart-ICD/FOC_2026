from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SimulatedGripperState:
    state_by_robot: dict[str, str] = field(default_factory=dict)
    attached_by_robot: dict[str, str] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "SimulatedGripperState":
        return cls()

    def set_state(self, robot: str, state: str) -> str:
        if state not in {"open", "closed"}:
            raise ValueError(f"Unsupported gripper state: {state}")
        self.state_by_robot[robot] = state
        if state == "open":
            self.attached_by_robot.pop(robot, None)
        return state

    def get_state(self, robot: str) -> str:
        return self.state_by_robot.get(robot, "open")

    def attach(self, robot: str, object_name: str) -> bool:
        if self.get_state(robot) != "closed":
            return False
        self.attached_by_robot[robot] = object_name
        return True

    def attached_object(self, robot: str) -> str | None:
        return self.attached_by_robot.get(robot)
