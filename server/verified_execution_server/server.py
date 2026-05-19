from __future__ import annotations

import asyncio
import inspect
import logging
import os
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from verified_execution_server.models import (
    ExecutePlanRequest,
    ExecutePlanResponse,
    HealthResponse,
    RobotCommandRequest,
    RobotCommandResponse,
    RobotReadiness,
)
from verified_execution_server.plan_cache import (
    AttachedObjectReleaseResult,
    PlanCache,
    RosPlanCache,
)
from verified_execution_server.ur_executor import TrajectoryExecutor, URRTDETrajectoryExecutor

LOGGER = logging.getLogger(__name__)
DEFAULT_UR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
DEFAULT_GRIPPER_JOINT_NAME = "finger_joint"
GRIPPER_OPEN_THRESHOLD_POSITION = 10


def create_app(
    *,
    plan_cache: PlanCache,
    executor: TrajectoryExecutor,
    joint_names: list[str] | None = None,
) -> FastAPI:
    sync_joint_names = list(joint_names or DEFAULT_UR_JOINT_NAMES)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await plan_cache.start()
        except Exception:
            LOGGER.exception("verified_execution.plan_cache_start_failed")
        await _run_startup_robot_check(app, executor)
        try:
            yield
        finally:
            await plan_cache.stop()
            close = getattr(executor, "close", None)
            if close is not None:
                result = close()
                if inspect.isawaitable(result):
                    await result

    app = FastAPI(title="Verified Execution Server", lifespan=lifespan)
    app.state.robot_readiness = None

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            ok=True,
            ros_connected=plan_cache.is_connected(),
            cached_plans=plan_cache.size(),
            robot=app.state.robot_readiness,
        )

    @app.post(
        "/execute",
        response_model=ExecutePlanResponse,
        response_model_exclude_none=True,
    )
    async def execute_plan(request: ExecutePlanRequest):
        LOGGER.info("verified_execute.request %s", request.model_dump())
        plan = plan_cache.get_plan(request.robot_name, request.plan_name)
        if plan is None:
            response = _failed_response(
                request,
                status="missing_plan",
                error="No cached trajectory for plan.",
                correction="Plan again, then retry execution.",
            )
            _log_execute_response(response)
            return JSONResponse(
                status_code=404,
                content=_response_content(response),
            )
        if not plan.frames:
            response = _failed_response(
                request,
                status="empty_trajectory",
                error="Cached plan has no trajectory points.",
                correction="Plan again, then retry execution.",
            )
            _log_execute_response(response)
            return JSONResponse(
                status_code=409,
                content=_response_content(response),
            )
        try:
            execution_result = await asyncio.wait_for(
                asyncio.to_thread(executor.execute, request.robot_name, plan.frames),
                timeout=request.timeout_s,
            )
        except TimeoutError:
            await _stop_executor_after_timeout(executor, request.robot_name)
            response = _failed_response(
                request,
                status="execution_timeout",
                error="Trajectory execution timed out.",
                correction="Check robot state, then retry with operator supervision.",
                trajectory_points=len(plan.frames),
            )
            _log_execute_response(response)
            return JSONResponse(
                status_code=504,
                content=_response_content(response),
            )
        except Exception as exc:
            response = _failed_response(
                request,
                status="execution_failed",
                error=str(exc),
                correction="Check the URScript/RTDE Receive connection and robot state, then retry.",
                trajectory_points=len(plan.frames),
            )
            _log_execute_response(response)
            return JSONResponse(
                status_code=500,
                content=_response_content(response),
            )

        execution_metadata = _execution_metadata(execution_result)
        if execution_metadata.get("final_joint_positions") is not None:
            sync_published = False
            if plan.joint_names is not None:
                sync_published = plan_cache.sync_joint_state(
                    request.robot_name,
                    joint_names=plan.joint_names,
                    joint_positions=execution_metadata["final_joint_positions"],
                )
            if not sync_published:
                response = _failed_response(
                    request,
                    status="state_sync_failed",
                    error="Executed physical motion, but fake controller joint state sync failed.",
                    correction="Check rosbridge and fake_controller_joint_states, then resync before planning the next stage.",
                    trajectory_points=len(plan.frames),
                    **execution_metadata,
                    state_sync_published=False,
                )
                _log_execute_response(response)
                return JSONResponse(
                    status_code=409,
                    content=_response_content(response),
                )
            execution_metadata["state_sync_published"] = True

        response = ExecutePlanResponse(
            ok=True,
            robot_name=request.robot_name,
            plan_name=request.plan_name,
            status="executed",
            trajectory_points=len(plan.frames),
            verification_result="pass",
            **execution_metadata,
        )
        _log_execute_response(response)
        return response

    @app.post("/home", response_model=RobotCommandResponse)
    async def home_robot(request: RobotCommandRequest):
        return await _run_robot_command(
            executor=executor,
            plan_cache=plan_cache,
            joint_names=sync_joint_names,
            request=request,
            command="home",
            success_status="homed",
            call=lambda: executor.go_home(request.robot_name),
            sync_position_field="final_joint_positions",
        )

    @app.post("/sync_state", response_model=RobotCommandResponse)
    async def sync_robot_state(request: RobotCommandRequest):
        return await _sync_robot_state(
            executor=executor,
            plan_cache=plan_cache,
            joint_names=sync_joint_names,
            request=request,
        )

    @app.post("/gripper/{action}", response_model=RobotCommandResponse)
    async def control_gripper(
        action: Literal["open", "close"],
        request: RobotCommandRequest,
    ):
        return await _run_robot_command(
            executor=executor,
            plan_cache=None,
            joint_names=None,
            request=request,
            command=f"gripper_{action}",
            success_status="gripper_opened" if action == "open" else "gripper_closed",
            call=lambda: executor.control_gripper(request.robot_name, action),
        )

    return app


def create_default_app() -> FastAPI:
    robot_name = os.getenv("ROBOT_NAME", "UR10")
    return create_app(
        plan_cache=RosPlanCache(
            robot_name=robot_name,
            host=os.getenv("ROSBRIDGE_HOST", "127.0.0.1"),
            port=_int_from_env("ROSBRIDGE_PORT", 9090),
        ),
        executor=URRTDETrajectoryExecutor(
            robot_ip=os.getenv("UR_ROBOT_IP", "127.0.0.1"),
            robot_port=_int_from_env("UR_RTDE_PORT", 30004),
            script_port=_int_from_env("UR_SCRIPT_PORT", 30002),
            socket_timeout_s=_float_from_env("UR_SOCKET_TIMEOUT_S", 3.0),
            joint_speed=_float_from_env("UR_JOINT_SPEED", 1.05),
            joint_accel=_float_from_env("UR_JOINT_ACCEL", 1.4),
            joint_blend=_float_from_env("UR_TRAJECTORY_BLEND_RADIUS", 0.02),
            servo_lookahead_time=_float_from_env("UR_SERVO_LOOKAHEAD_TIME", 0.1),
            servo_gain=_float_from_env("UR_SERVO_GAIN", 300.0),
            completion_timeout_s=_float_from_env("UR_COMPLETION_TIMEOUT_S", 60.0),
            completion_poll_interval_s=_float_from_env(
                "UR_COMPLETION_POLL_INTERVAL_S",
                0.1,
            ),
            joint_tolerance_rad=_float_from_env("UR_JOINT_TOLERANCE_RAD", 0.03),
            completion_stable_samples=_int_from_env("UR_COMPLETION_STABLE_SAMPLES", 2),
            skip_gripper=_bool_from_env("UR_SKIP_GRIPPER", False),
            gripper_port=_int_from_env("UR_GRIPPER_PORT", 63352),
            gripper_speed=_int_from_env("UR_GRIPPER_SPEED", 255),
            gripper_force=_int_from_env("UR_GRIPPER_FORCE", 255),
        ),
        joint_names=_joint_names_from_env("UR_JOINT_NAMES"),
    )


def _failed_response(
    request: ExecutePlanRequest,
    *,
    status: str,
    error: str,
    correction: str,
    trajectory_points: int = 0,
    target_joint_positions: list[float] | None = None,
    final_joint_positions: list[float] | None = None,
    max_joint_error: float | None = None,
    joint_tolerance_rad: float | None = None,
    state_sync_published: bool | None = None,
) -> ExecutePlanResponse:
    return ExecutePlanResponse(
        ok=False,
        robot_name=request.robot_name,
        plan_name=request.plan_name,
        status=status,
        trajectory_points=trajectory_points,
        verification_result="fail",
        error=error,
        correction=correction,
        target_joint_positions=target_joint_positions,
        final_joint_positions=final_joint_positions,
        max_joint_error=max_joint_error,
        joint_tolerance_rad=joint_tolerance_rad,
        state_sync_published=state_sync_published,
    )


def _execution_metadata(execution_result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in (
        "target_joint_positions",
        "final_joint_positions",
        "max_joint_error",
        "joint_tolerance_rad",
    ):
        value = _execution_result_value(execution_result, field)
        if value is None:
            continue
        if field.endswith("_joint_positions"):
            positions = _float_list(value)
            if positions is not None:
                metadata[field] = positions
        else:
            try:
                metadata[field] = float(value)
            except (TypeError, ValueError):
                pass
    return metadata


def _command_metadata(command_result: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for field in (
        "target_joint_positions",
        "final_joint_positions",
        "actual_joint_positions",
        "actual_tcp_pose",
    ):
        value = _execution_result_value(command_result, field)
        positions = _float_list(value)
        if positions is not None:
            metadata[field] = positions
    for field in ("max_joint_error", "joint_tolerance_rad"):
        value = _execution_result_value(command_result, field)
        if value is None:
            continue
        try:
            metadata[field] = float(value)
        except (TypeError, ValueError):
            pass
    gripper_position = _int_or_none(
        _execution_result_value(command_result, "actual_gripper_position")
    )
    if gripper_position is not None:
        metadata["actual_gripper_position"] = gripper_position
    gripper_joint_position = _float_or_none(
        _execution_result_value(command_result, "actual_gripper_joint_position")
    )
    if gripper_joint_position is not None:
        metadata["actual_gripper_joint_position"] = gripper_joint_position
    return metadata


def _execution_result_value(execution_result: Any, field: str) -> Any:
    if execution_result is None:
        return None
    if isinstance(execution_result, dict):
        return execution_result.get(field)
    return getattr(execution_result, field, None)


def _float_list(value: Any) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _response_content(response: ExecutePlanResponse) -> dict[str, Any]:
    return response.model_dump(exclude_none=True)


async def _stop_executor_after_timeout(executor: TrajectoryExecutor, robot_name: str) -> None:
    stop = getattr(executor, "stop", None)
    if not callable(stop):
        return
    try:
        await asyncio.to_thread(stop, robot_name)
    except Exception:
        LOGGER.exception("verified_execute.timeout_stop_failed robot_name=%s", robot_name)


def _log_execute_response(response: ExecutePlanResponse) -> None:
    LOGGER.info("verified_execute.response %s", response.model_dump())


async def _run_startup_robot_check(app: FastAPI, executor: TrajectoryExecutor) -> None:
    startup_check = getattr(executor, "startup_check", None)
    if not callable(startup_check):
        return
    try:
        status = await asyncio.to_thread(startup_check, os.getenv("ROBOT_NAME", "UR10"))
        app.state.robot_readiness = RobotReadiness.model_validate(status)
        print(
            "[UR][Startup] "
            f"robot_connected={app.state.robot_readiness.robot_connected} "
            f"gripper_connected={app.state.robot_readiness.gripper_connected} "
            f"gripper_position={app.state.robot_readiness.gripper_position}",
            flush=True,
        )
        LOGGER.info(
            "verified_execution.startup_robot_check %s",
            app.state.robot_readiness.model_dump(),
        )
    except Exception as exc:
        app.state.robot_readiness = RobotReadiness(
            robot_name=os.getenv("ROBOT_NAME", "UR10"),
            robot_connected=False,
            gripper_connected=False,
            robot_error=str(exc),
        )
        LOGGER.exception("verified_execution.startup_robot_check_failed")


async def _run_robot_command(
    *,
    executor: TrajectoryExecutor,
    plan_cache: PlanCache | None,
    joint_names: list[str] | None,
    request: RobotCommandRequest,
    command: str,
    success_status: str,
    call,
    sync_position_field: str | None = None,
):
    LOGGER.info(
        "verified_robot_command.request %s",
        {"command": command, **request.model_dump()},
    )
    try:
        command_result = await asyncio.wait_for(asyncio.to_thread(call), timeout=request.timeout_s)
    except TimeoutError:
        await _stop_executor_after_timeout(executor, request.robot_name)
        response = _failed_command_response(
            request,
            command=command,
            status="command_timeout",
            error="Robot command timed out.",
            correction="Check robot state, then retry with operator supervision.",
        )
        _log_command_response(response)
        return JSONResponse(status_code=504, content=response.model_dump())
    except Exception as exc:
        response = _failed_command_response(
            request,
            command=command,
            status="command_failed",
            error=str(exc),
            correction="Check the URScript/RTDE Receive connection and robot state, then retry.",
        )
        _log_command_response(response)
        return JSONResponse(status_code=500, content=response.model_dump())

    metadata = _command_metadata(command_result)
    if sync_position_field is not None:
        sync_positions = metadata.get(sync_position_field)
        if isinstance(sync_positions, list) and plan_cache is not None and joint_names is not None:
            sync_published = plan_cache.sync_joint_state(
                request.robot_name,
                joint_names=joint_names,
                joint_positions=sync_positions,
            )
            metadata["state_sync_published"] = sync_published
            if not sync_published:
                response = RobotCommandResponse(
                    ok=False,
                    robot_name=request.robot_name,
                    command=command,
                    status="state_sync_failed",
                    error="Executed physical robot command, but fake controller joint state sync failed.",
                    correction="Check rosbridge and fake_controller_joint_states, then run state sync before planning again.",
                    **metadata,
                )
                _log_command_response(response)
                return JSONResponse(status_code=409, content=response.model_dump())

    response = RobotCommandResponse(
        ok=True,
        robot_name=request.robot_name,
        command=command,
        status=success_status,
        **metadata,
    )
    _log_command_response(response)
    return response


async def _sync_robot_state(
    *,
    executor: TrajectoryExecutor,
    plan_cache: PlanCache,
    joint_names: list[str],
    request: RobotCommandRequest,
):
    LOGGER.info(
        "verified_robot_command.request %s",
        {"command": "sync_state", **request.model_dump()},
    )
    try:
        state_result = await asyncio.wait_for(
            asyncio.to_thread(executor.read_state, request.robot_name),
            timeout=request.timeout_s,
        )
    except TimeoutError:
        response = _failed_command_response(
            request,
            command="sync_state",
            status="command_timeout",
            error="Robot state sync timed out.",
            correction="Check robot state, RTDE receive, and rosbridge, then retry with operator supervision.",
        )
        _log_command_response(response)
        return JSONResponse(status_code=504, content=response.model_dump())
    except Exception as exc:
        response = _failed_command_response(
            request,
            command="sync_state",
            status="command_failed",
            error=str(exc),
            correction="Check the UR RTDE receive connection, Robotiq gripper connection, and robot state, then retry.",
        )
        _log_command_response(response)
        return JSONResponse(status_code=500, content=response.model_dump())

    metadata = _command_metadata(state_result)
    metadata["state_sync_published"] = False
    metadata["gripper_joint_state_published"] = False
    metadata["gripper_joint_name"] = DEFAULT_GRIPPER_JOINT_NAME
    metadata["gripper_joint_state_topic"] = _gripper_joint_state_topic(request.robot_name)
    metadata["gripper_open_threshold_position"] = GRIPPER_OPEN_THRESHOLD_POSITION
    metadata["gripper_considered_open"] = False
    metadata["attached_object_release_checked"] = False
    metadata["attached_objects_before_release"] = []
    metadata["attached_objects_released"] = []
    metadata["attached_object_release_published"] = False
    metadata["attached_object_release_verified"] = False
    metadata["attached_object_release_topic_or_service"] = _attached_object_release_service(
        request.robot_name
    )
    actual_positions = metadata.get("actual_joint_positions")
    if not isinstance(actual_positions, list):
        response = RobotCommandResponse(
            ok=False,
            robot_name=request.robot_name,
            command="sync_state",
            status="missing_joint_state",
            error="RTDE receive did not return actual joint positions.",
            correction="Check the UR RTDE receive connection, then retry state sync.",
            **metadata,
        )
        _log_command_response(response)
        return JSONResponse(status_code=409, content=response.model_dump())

    actual_gripper_position = metadata.get("actual_gripper_position")
    actual_gripper_joint_position = metadata.get("actual_gripper_joint_position")
    if not isinstance(actual_gripper_position, int) or not isinstance(
        actual_gripper_joint_position,
        float,
    ):
        response = RobotCommandResponse(
            ok=False,
            robot_name=request.robot_name,
            command="sync_state",
            status="missing_gripper_state",
            error="Robotiq gripper state was not returned.",
            correction="Check the Robotiq gripper connection, then retry state sync.",
            **metadata,
        )
        _log_command_response(response)
        return JSONResponse(status_code=409, content=response.model_dump())

    metadata["gripper_considered_open"] = (
        actual_gripper_position <= GRIPPER_OPEN_THRESHOLD_POSITION
    )
    moveit_joint_names = [*joint_names, DEFAULT_GRIPPER_JOINT_NAME]
    moveit_joint_positions = [*actual_positions, actual_gripper_joint_position]
    sync_published = plan_cache.sync_joint_state(
        request.robot_name,
        joint_names=moveit_joint_names,
        joint_positions=moveit_joint_positions,
    )
    metadata["state_sync_published"] = sync_published
    if not sync_published:
        response = RobotCommandResponse(
            ok=False,
            robot_name=request.robot_name,
            command="sync_state",
            status="state_sync_failed",
            error="Read real robot state, but fake controller joint state sync failed.",
            correction="Check rosbridge and fake_controller_joint_states, then retry state sync before planning again.",
            **metadata,
        )
        _log_command_response(response)
        return JSONResponse(status_code=409, content=response.model_dump())

    gripper_sync_published = plan_cache.sync_gripper_joint_state(
        request.robot_name,
        joint_name=DEFAULT_GRIPPER_JOINT_NAME,
        joint_position=actual_gripper_joint_position,
    )
    metadata["gripper_joint_state_published"] = gripper_sync_published
    if not gripper_sync_published:
        response = RobotCommandResponse(
            ok=False,
            robot_name=request.robot_name,
            command="sync_state",
            status="state_sync_failed",
            error="Read real robot state, but gripper joint state sync failed.",
            correction=(
                f"Check rosbridge and {_gripper_joint_state_topic(request.robot_name)}, "
                "then retry state sync before planning again."
            ),
            **metadata,
        )
        _log_command_response(response)
        return JSONResponse(status_code=409, content=response.model_dump())

    if metadata["gripper_considered_open"] is True:
        try:
            release_result = await asyncio.to_thread(
                plan_cache.release_attached_objects,
                request.robot_name,
                timeout_s=request.timeout_s,
            )
        except Exception as exc:
            response = RobotCommandResponse(
                ok=False,
                robot_name=request.robot_name,
                command="sync_state",
                status="attached_object_release_failed",
                error=str(exc),
                correction=(
                    f"Check /{request.robot_name}/get_planning_scene and "
                    f"{_attached_object_release_service(request.robot_name)}, then retry state sync."
                ),
                **metadata,
            )
            _log_command_response(response)
            return JSONResponse(status_code=409, content=response.model_dump())

        metadata.update(_release_result_metadata(release_result))
        if not release_result.ok:
            response = RobotCommandResponse(
                ok=False,
                robot_name=request.robot_name,
                command="sync_state",
                status="attached_object_release_failed",
                error=release_result.error or "Attached object release reconciliation failed.",
                correction=(
                    release_result.correction
                    or (
                        f"Check /{request.robot_name}/get_planning_scene and "
                        f"{_attached_object_release_service(request.robot_name)}, "
                        "then retry state sync."
                    )
                ),
                **metadata,
            )
            _log_command_response(response)
            return JSONResponse(status_code=409, content=response.model_dump())

    response = RobotCommandResponse(
        ok=True,
        robot_name=request.robot_name,
        command="sync_state",
        status="state_synced",
        **metadata,
    )
    _log_command_response(response)
    return response


def _failed_command_response(
    request: RobotCommandRequest,
    *,
    command: str,
    status: str,
    error: str,
    correction: str,
) -> RobotCommandResponse:
    return RobotCommandResponse(
        ok=False,
        robot_name=request.robot_name,
        command=command,
        status=status,
        error=error,
        correction=correction,
    )


def _log_command_response(response: RobotCommandResponse) -> None:
    LOGGER.info("verified_robot_command.response %s", response.model_dump())


def _gripper_joint_state_topic(robot_name: str) -> str:
    return f"/{robot_name}/gripper_joint_states"


def _attached_object_release_service(robot_name: str) -> str:
    return f"/{robot_name}/apply_planning_scene"


def _release_result_metadata(result: AttachedObjectReleaseResult) -> dict[str, Any]:
    return {
        "attached_object_release_checked": bool(result.checked),
        "attached_objects_before_release": list(result.attached_objects_before_release),
        "attached_objects_released": list(result.attached_objects_released),
        "attached_object_release_published": bool(result.published),
        "attached_object_release_verified": bool(result.verified),
        "attached_object_release_topic_or_service": result.topic_or_service,
    }


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def _float_from_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def _bool_from_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _joint_names_from_env(name: str) -> list[str] | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    names = [part.strip() for part in raw.split(",") if part.strip()]
    return names or None
