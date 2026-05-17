from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CachedPlan(BaseModel):
    robot_name: str
    plan_name: str
    frames: list[dict]
    observed_at_s: float
    joint_names: list[str] | None = None


class ExecutePlanRequest(BaseModel):
    robot_name: str = "UR10"
    plan_name: str = Field(min_length=1)
    timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)


class ExecutePlanResponse(BaseModel):
    ok: bool
    robot_name: str
    plan_name: str
    status: str
    trajectory_points: int
    verification_result: str
    error: str | None = None
    correction: str | None = None
    target_joint_positions: list[float] | None = None
    final_joint_positions: list[float] | None = None
    max_joint_error: float | None = None
    joint_tolerance_rad: float | None = None
    state_sync_published: bool | None = None


class RobotCommandRequest(BaseModel):
    robot_name: str = "UR10"
    timeout_s: float = Field(default=10.0, gt=0.0, le=120.0)


class RobotCommandResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    ok: bool
    robot_name: str
    command: str
    status: str
    error: str | None = None
    correction: str | None = None


class RobotReadiness(BaseModel):
    robot_name: str
    robot_connected: bool | None = None
    gripper_connected: bool | None = None
    robot_error: str | None = None
    gripper_error: str | None = None
    gripper_position: int | None = None


class HealthResponse(BaseModel):
    ok: bool
    ros_connected: bool
    cached_plans: int
    robot: RobotReadiness | None = None
