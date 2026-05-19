from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_APPROACH_DISTANCE_M = 0.08
DEFAULT_PLACE_STANDOFF_M = 0.01
DEFAULT_RETREAT_DISTANCE_M = 0.1
PLACE_ORIENTATION_MODES = {"keep", "horizontal", "vertical", "explicit"}

PLACE_DISTANCE_CORRECTION = (
    "Retry with positive finite approach_distance_m, place_standoff_m, and retreat_distance_m values."
)
PLACE_TARGET_CORRECTION = "Retry with a target_pose or target_position in base_link."
PLACE_ORIENTATION_CORRECTION = 'Retry with orientation_mode "keep", "horizontal", "vertical", or "explicit".'

_IDENTITY_ORIENTATION = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
_GRIPPER_DOWN_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}


@dataclass(frozen=True)
class PlacePlanInputError(ValueError):
    status: str
    correction: str
    raw: dict[str, Any]

    def __str__(self) -> str:
        return self.status


def build_place_workflow(
    object_context: dict[str, Any],
    *,
    target_pose: dict[str, Any] | None = None,
    target_position: dict[str, Any] | None = None,
    current_pose: dict[str, Any] | None = None,
    orientation_mode: str = "keep",
    place_face: str | None = None,
    support_face: str | None = None,
    approach_distance_m: float = DEFAULT_APPROACH_DISTANCE_M,
    place_standoff_m: float = DEFAULT_PLACE_STANDOFF_M,
    retreat_distance_m: float = DEFAULT_RETREAT_DISTANCE_M,
) -> dict[str, Any]:
    object_name = str(object_context.get("name") or "")
    if orientation_mode not in PLACE_ORIENTATION_MODES:
        raise PlacePlanInputError(
            status="invalid orientation mode",
            correction=PLACE_ORIENTATION_CORRECTION,
            raw={"orientation_mode": orientation_mode, "available_orientation_modes": sorted(PLACE_ORIENTATION_MODES)},
        )

    for name, value in {
        "approach_distance_m": approach_distance_m,
        "place_standoff_m": place_standoff_m,
        "retreat_distance_m": retreat_distance_m,
    }.items():
        if not _positive_finite(value):
            raise PlacePlanInputError(
                status="invalid place distance",
                correction=PLACE_DISTANCE_CORRECTION,
                raw={name: value},
            )

    position, explicit_orientation = _target_components(target_pose=target_pose, target_position=target_position)
    orientation = _release_orientation(
        object_context,
        target_orientation=explicit_orientation,
        current_pose=current_pose,
        orientation_mode=orientation_mode,
    )

    release_tcp_position = _offset_z(position, place_standoff_m)
    approach = _pose(_offset_z(position, approach_distance_m), orientation)
    release = _pose(release_tcp_position, orientation)
    retreat = _pose(_offset_z(release_tcp_position, retreat_distance_m), orientation)
    release_object_pose = _pose(position, explicit_orientation or _object_orientation(object_context) or dict(_IDENTITY_ORIENTATION))

    return {
        "workflow_kind": "place",
        "object_name": object_name,
        "planning_frame": object_context.get("planning_frame"),
        "target_object_pose": release_object_pose,
        "release_tcp_pose": release,
        "waypoints": [approach, release, retreat],
        "workflow_steps": [
            {"name": "carry_approach", "kind": "motion", "waypoint_index": 0},
            {"name": "release_pose", "kind": "motion", "waypoint_index": 1},
            {"name": "open_gripper", "kind": "gripper", "tool": "moveit_open_gripper"},
            {"name": "detach_object", "kind": "scene", "object_name": object_name},
            {"name": "retreat", "kind": "motion", "waypoint_index": 2},
        ],
        "parameters": {
            "orientation_mode": orientation_mode,
            "place_face": place_face,
            "support_face": support_face,
            "approach_distance_m": approach_distance_m,
            "place_standoff_m": place_standoff_m,
            "retreat_distance_m": retreat_distance_m,
        },
        "release_after_execute": {
            "object_name": object_name,
            "object_pose": release_object_pose,
        },
    }


def _target_components(
    *,
    target_pose: dict[str, Any] | None,
    target_position: dict[str, Any] | None,
) -> tuple[dict[str, float], dict[str, float] | None]:
    if target_pose is None and target_position is None:
        raise PlacePlanInputError(status="missing place target", correction=PLACE_TARGET_CORRECTION, raw={})

    target = target_pose if target_pose is not None else target_position
    if not isinstance(target, dict):
        raise PlacePlanInputError(status="invalid place target", correction=PLACE_TARGET_CORRECTION, raw={"target": target})

    position_value = target.get("position") if "position" in target else target
    if not isinstance(position_value, dict):
        raise PlacePlanInputError(status="invalid place target", correction=PLACE_TARGET_CORRECTION, raw={"target": target})

    try:
        position = _point(position_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlacePlanInputError(status="invalid place target", correction=PLACE_TARGET_CORRECTION, raw={"details": str(exc)}) from exc

    orientation_value = target.get("orientation") if target_pose is not None else None
    if orientation_value is None:
        return position, None
    if not isinstance(orientation_value, dict):
        raise PlacePlanInputError(status="invalid target orientation", correction=PLACE_TARGET_CORRECTION, raw={"orientation": orientation_value})
    try:
        orientation = _quaternion(orientation_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise PlacePlanInputError(status="invalid target orientation", correction=PLACE_TARGET_CORRECTION, raw={"details": str(exc)}) from exc
    return position, orientation


def _release_orientation(
    object_context: dict[str, Any],
    *,
    target_orientation: dict[str, float] | None,
    current_pose: dict[str, Any] | None,
    orientation_mode: str,
) -> dict[str, float]:
    if orientation_mode == "explicit":
        if target_orientation is None:
            raise PlacePlanInputError(
                status="missing explicit orientation",
                correction="Retry with target_pose.orientation when orientation_mode is explicit.",
                raw={},
            )
        return dict(target_orientation)
    if orientation_mode == "horizontal":
        return dict(_GRIPPER_DOWN_ORIENTATION)
    if orientation_mode == "vertical":
        return dict(_IDENTITY_ORIENTATION)
    return (
        target_orientation
        or _current_orientation(current_pose)
        or _object_orientation(object_context)
        or dict(_IDENTITY_ORIENTATION)
    )


def _current_orientation(current_pose: dict[str, Any] | None) -> dict[str, float] | None:
    if not isinstance(current_pose, dict):
        return None
    orientation = current_pose.get("orientation")
    if not isinstance(orientation, dict):
        return None
    try:
        return _quaternion(orientation)
    except (KeyError, TypeError, ValueError):
        return None


def _object_orientation(object_context: dict[str, Any]) -> dict[str, float] | None:
    pose = object_context.get("pose")
    if not isinstance(pose, dict):
        return None
    orientation = pose.get("orientation")
    if not isinstance(orientation, dict):
        return None
    try:
        return _quaternion(orientation)
    except (KeyError, TypeError, ValueError):
        return None


def _pose(position: dict[str, float], orientation: dict[str, float]) -> dict[str, Any]:
    return {"position": position, "orientation": dict(orientation)}


def _offset_z(point: dict[str, float], distance: float) -> dict[str, float]:
    return {**point, "z": _clean(point["z"] + distance)}


def _point(value: dict[str, Any]) -> dict[str, float]:
    point = {axis: _clean(float(value[axis])) for axis in ("x", "y", "z")}
    if not all(math.isfinite(point[axis]) for axis in ("x", "y", "z")):
        raise ValueError("position coordinates must be finite")
    return point


def _quaternion(value: dict[str, Any]) -> dict[str, float]:
    q = {axis: _clean(float(value[axis])) for axis in ("x", "y", "z", "w")}
    norm = math.sqrt(sum(q[axis] * q[axis] for axis in ("x", "y", "z", "w")))
    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(f"Quaternion orientation must be normalized; norm={norm:.6f}")
    return q


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0.0


def _clean(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded
