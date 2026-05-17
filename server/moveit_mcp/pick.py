from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

DEFAULT_GRASP_FACE = "top"
DEFAULT_APPROACH_DISTANCE_M = 0.08
DEFAULT_GRASP_STANDOFF_M = 0.01
DEFAULT_LIFT_DISTANCE_M = 0.1
AUTO_MAX_PICK_CANDIDATES = 8
AUTO_DISTANCE_VARIANTS = (
    (1.0, 1.0),
    (1.5, 2.0),
    (0.75, 1.5),
)
SIDE_GRASP_FACES = {"front", "back", "left", "right"}

PICK_DISTANCE_CORRECTION = (
    "Retry with positive finite approach_distance_m, grasp_standoff_m, and lift_distance_m values."
)
GRASP_FACE_CORRECTION = "Call moveit_get_object_context, then retry with one raw.object.grasp_faces[].name."

_IDENTITY_ORIENTATION = {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}
_GRIPPER_DOWN_ORIENTATION = {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0}


@dataclass(frozen=True)
class PickPlanInputError(ValueError):
    status: str
    correction: str
    raw: dict[str, Any]

    def __str__(self) -> str:
        return self.status


def build_pick_workflow(
    object_context: dict[str, Any],
    *,
    grasp_face: str = DEFAULT_GRASP_FACE,
    approach_distance_m: float = DEFAULT_APPROACH_DISTANCE_M,
    grasp_standoff_m: float = DEFAULT_GRASP_STANDOFF_M,
    lift_distance_m: float = DEFAULT_LIFT_DISTANCE_M,
) -> dict[str, Any]:
    object_name = str(object_context.get("name") or "")
    faces = [face for face in object_context.get("grasp_faces") or [] if isinstance(face, dict)]
    selected = next((face for face in faces if face.get("name") == grasp_face), None)
    if selected is None:
        raise PickPlanInputError(
            status="grasp face not available",
            correction=GRASP_FACE_CORRECTION,
            raw={"available_grasp_faces": [str(face.get("name")) for face in faces if face.get("name")]},
        )

    for name, value in {
        "approach_distance_m": approach_distance_m,
        "grasp_standoff_m": grasp_standoff_m,
        "lift_distance_m": lift_distance_m,
    }.items():
        if not _positive_finite(value):
            raise PickPlanInputError(
                status="invalid pick distance",
                correction=PICK_DISTANCE_CORRECTION,
                raw={name: value},
            )

    face_center = _point(selected["center"])
    normal = _point(selected["normal"])
    alignment_axis = _optional_point(selected.get("alignment_axis"))
    orientation = _orientation_for_grasp_frame(normal, alignment_axis)
    approach = _pose(_offset(face_center, normal, approach_distance_m), orientation)
    pre_grasp = _pose(_offset(face_center, normal, grasp_standoff_m), orientation)
    lift = _pose({**pre_grasp["position"], "z": _clean(pre_grasp["position"]["z"] + lift_distance_m)}, orientation)

    return {
        "object_name": object_name,
        "planning_frame": object_context.get("planning_frame"),
        "selected_grasp_face": selected,
        "waypoints": [approach, pre_grasp, lift],
        "motion_segments": [
            {
                "name": "approach_to_pre_grasp",
                "planner": "cartesian",
                "waypoint_indexes": [0, 1],
            },
            {
                "name": "post_grasp_lift",
                "planner": "cartesian",
                "waypoint_indexes": [1, 2],
            },
        ],
        "workflow_steps": [
            {"name": "approach", "kind": "motion", "waypoint_index": 0},
            {"name": "pre_grasp", "kind": "motion", "waypoint_index": 1},
            {"name": "close_gripper", "kind": "gripper", "tool": "moveit_close_gripper"},
            {"name": "attach_object", "kind": "scene", "tool": "moveit_attach_object", "object_name": object_name},
            {"name": "lift", "kind": "motion", "waypoint_index": 2},
        ],
        "parameters": {
            "grasp_face": grasp_face,
            "approach_distance_m": approach_distance_m,
            "grasp_standoff_m": grasp_standoff_m,
            "lift_distance_m": lift_distance_m,
        },
    }


def build_pick_candidates(
    object_context: dict[str, Any],
    *,
    requested_grasp_face: str | None,
    approach_distance_m: float = DEFAULT_APPROACH_DISTANCE_M,
    grasp_standoff_m: float = DEFAULT_GRASP_STANDOFF_M,
    lift_distance_m: float = DEFAULT_LIFT_DISTANCE_M,
    max_candidates: int = AUTO_MAX_PICK_CANDIDATES,
) -> list[dict[str, Any]]:
    face_names = _ranked_face_names(object_context, requested_grasp_face=requested_grasp_face)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, float, float, float]] = set()

    for face_name in face_names:
        for approach_scale, standoff_scale in AUTO_DISTANCE_VARIANTS:
            candidate_approach = _clean(approach_distance_m * approach_scale)
            candidate_standoff = _clean(grasp_standoff_m * standoff_scale)
            key = (face_name, candidate_approach, candidate_standoff, lift_distance_m)
            if key in seen:
                continue

            seen.add(key)
            candidates.append(
                build_pick_workflow(
                    object_context,
                    grasp_face=face_name,
                    approach_distance_m=candidate_approach,
                    grasp_standoff_m=candidate_standoff,
                    lift_distance_m=lift_distance_m,
                )
            )
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def build_oriented_pick_workflow(
    object_context: dict[str, Any],
    *,
    requested_grasp_face: str | None,
    approach_distance_m: float = DEFAULT_APPROACH_DISTANCE_M,
    grasp_standoff_m: float = DEFAULT_GRASP_STANDOFF_M,
    lift_distance_m: float = DEFAULT_LIFT_DISTANCE_M,
) -> dict[str, Any]:
    face_names = _ranked_face_names(object_context, requested_grasp_face=requested_grasp_face)
    if not face_names:
        raise PickPlanInputError(
            status="grasp face not available",
            correction=GRASP_FACE_CORRECTION,
            raw={"available_grasp_faces": []},
        )
    return build_pick_workflow(
        object_context,
        grasp_face=face_names[0],
        approach_distance_m=approach_distance_m,
        grasp_standoff_m=grasp_standoff_m,
        lift_distance_m=lift_distance_m,
    )


def _ranked_face_names(object_context: dict[str, Any], *, requested_grasp_face: str | None) -> list[str]:
    faces = [face for face in object_context.get("grasp_faces") or [] if isinstance(face, dict)]
    by_name = {str(face.get("name")): face for face in faces if face.get("name")}
    if requested_grasp_face and requested_grasp_face not in by_name:
        raise PickPlanInputError(
            status="grasp face not available",
            correction=GRASP_FACE_CORRECTION,
            raw={"available_grasp_faces": list(by_name)},
        )

    allowed_names = _allowed_auto_face_names(
        object_context,
        by_name,
        requested_grasp_face=requested_grasp_face,
    )
    ranked: list[str] = []

    beam_orientation = _beam_orientation(object_context)
    if requested_grasp_face and requested_grasp_face in allowed_names:
        ranked.append(requested_grasp_face)
    elif beam_orientation == "horizontal" and "top" in allowed_names:
        ranked.append("top")
    elif "top" in allowed_names:
        ranked.append("top")

    remaining = [
        name
        for name, _face in sorted(
            by_name.items(),
            key=lambda item: _face_rank_key(item[0], item[1], beam_orientation=beam_orientation),
        )
        if name not in ranked and name in allowed_names
    ]
    ranked.extend(remaining)
    return ranked


def _allowed_auto_face_names(
    object_context: dict[str, Any],
    by_name: dict[str, dict[str, Any]],
    *,
    requested_grasp_face: str | None,
) -> set[str]:
    beam_orientation = _beam_orientation(object_context)
    if beam_orientation == "horizontal":
        if requested_grasp_face in SIDE_GRASP_FACES and requested_grasp_face in by_name:
            return {requested_grasp_face}
        allowed = {"top"} & set(by_name)
        if allowed:
            return allowed
    if beam_orientation == "vertical":
        allowed = _meaningful_vertical_side_names(by_name)
        if allowed:
            return allowed
    return {name for name in by_name if name != "bottom"}


def _meaningful_vertical_side_names(by_name: dict[str, dict[str, Any]]) -> set[str]:
    side_names = SIDE_GRASP_FACES & set(by_name)
    if not side_names:
        return set()
    outer_names = {
        name
        for name in side_names
        if by_name[name].get("beam_side_preference") == "outer"
    }
    inner_names = {
        name
        for name in side_names
        if by_name[name].get("beam_side_preference") == "inner"
    }
    if outer_names and inner_names:
        return outer_names
    return side_names


def _face_rank_key(name: str, face: dict[str, Any], *, beam_orientation: str | None) -> tuple[int, int, float, float, str]:
    if beam_orientation == "vertical":
        side_preference = str(face.get("beam_side_preference") or "unknown")
        side_rank = {"outer": 0, "unknown": 1, "inner": 2}.get(side_preference, 1)
        clearance = face.get("scene_clearance_m")
        if isinstance(clearance, int | float) and math.isfinite(float(clearance)):
            clearance_rank = 1
            clearance_value = -float(clearance)
        else:
            clearance_rank = 0
            clearance_value = 0.0
        return (side_rank, clearance_rank, clearance_value, -float(face.get("area") or 0.0), name)
    return (0, 0, 0.0, -float(face.get("area") or 0.0), name)


def _beam_orientation(object_context: dict[str, Any]) -> str | None:
    bounds = object_context.get("bounds")
    if not isinstance(bounds, dict):
        return None
    size = bounds.get("size")
    if not isinstance(size, dict):
        return None

    try:
        x = float(size["x"])
        y = float(size["y"])
        z = float(size["z"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (x, y, z)):
        return None

    horizontal_extent = max(x, y)
    if z > horizontal_extent:
        return "vertical"
    if horizontal_extent > z:
        return "horizontal"
    return None


def _pose(position: dict[str, float], orientation: dict[str, float]) -> dict[str, Any]:
    return {"position": position, "orientation": dict(orientation)}


def _orientation_for_grasp_frame(
    normal: dict[str, float],
    alignment_axis: dict[str, float] | None,
) -> dict[str, float]:
    if alignment_axis is None:
        return _orientation_for_face_normal(normal)

    nx, ny, nz = _unit_vector(normal)
    z_axis = (-nx, -ny, -nz)
    ax, ay, az = _unit_vector(alignment_axis)
    dot = ax * z_axis[0] + ay * z_axis[1] + az * z_axis[2]
    x_axis = (
        ax - dot * z_axis[0],
        ay - dot * z_axis[1],
        az - dot * z_axis[2],
    )
    try:
        x_axis = _unit_tuple(x_axis)
    except ValueError:
        return _orientation_for_face_normal(normal)

    y_axis = _unit_tuple(_cross(z_axis, x_axis))
    x_axis = _unit_tuple(_cross(y_axis, z_axis))
    return _quaternion_from_axes(x_axis, y_axis, z_axis)


def _orientation_for_face_normal(normal: dict[str, float]) -> dict[str, float]:
    nx, ny, nz = _unit_vector(normal)
    dot = -nz
    if math.isclose(dot, 1.0, abs_tol=1e-9):
        return dict(_IDENTITY_ORIENTATION)
    if math.isclose(dot, -1.0, abs_tol=1e-9):
        return dict(_GRIPPER_DOWN_ORIENTATION)

    qx = ny
    qy = -nx
    qz = 0.0
    qw = 1.0 + dot
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return {
        "x": _clean(qx / norm),
        "y": _clean(qy / norm),
        "z": _clean(qz / norm),
        "w": _clean(qw / norm),
    }


def _unit_vector(value: dict[str, float]) -> tuple[float, float, float]:
    x = float(value["x"])
    y = float(value["y"])
    z = float(value["z"])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("grasp face normal must be non-zero")
    return x / norm, y / norm, z / norm


def _unit_tuple(value: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = value
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    return x / norm, y / norm, z / norm


def _cross(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _quaternion_from_axes(
    x_axis: tuple[float, float, float],
    y_axis: tuple[float, float, float],
    z_axis: tuple[float, float, float],
) -> dict[str, float]:
    m00, m01, m02 = x_axis[0], y_axis[0], z_axis[0]
    m10, m11, m12 = x_axis[1], y_axis[1], z_axis[1]
    m20, m21, m22 = x_axis[2], y_axis[2], z_axis[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        qw = (m21 - m12) / scale
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        qw = (m02 - m20) / scale
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        qw = (m10 - m01) / scale
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
    return _normalize_quaternion({"x": qx, "y": qy, "z": qz, "w": qw})


def _normalize_quaternion(value: dict[str, float]) -> dict[str, float]:
    norm = math.sqrt(sum(value[axis] * value[axis] for axis in ("x", "y", "z", "w")))
    return {axis: _clean(value[axis] / norm) for axis in ("x", "y", "z", "w")}


def _offset(point: dict[str, float], normal: dict[str, float], distance: float) -> dict[str, float]:
    return {axis: _clean(point[axis] + normal[axis] * distance) for axis in ("x", "y", "z")}


def _point(value: dict[str, Any]) -> dict[str, float]:
    return {axis: _clean(float(value[axis])) for axis in ("x", "y", "z")}


def _optional_point(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        return _point(value)
    except (KeyError, TypeError, ValueError):
        return None


def _positive_finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and value > 0.0


def _clean(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded
