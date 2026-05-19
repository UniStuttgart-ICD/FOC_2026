from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from robot_control.shared_geometry.world_context import (
    DEFAULT_HOLOGRAM_MODEL_PATH,
    canonical_dynamic_name,
)

SNAPPY_NAME_RE = re.compile(r"^dynamic_snappy-V[^_]+_box(\d+)$")
EPSILON = 1e-9
DEFAULT_CORRECTION_RADIANS = math.pi
SNAPSHOT_TOLERANCE = 1e-5


class ModelTrackerSyncSession:
    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_HOLOGRAM_MODEL_PATH,
        correction_radians: float = DEFAULT_CORRECTION_RADIANS,
    ) -> None:
        self._model_path = Path(model_path)
        self._correction_radians = correction_radians
        self._previous_snapshot: _ModelTrackerSnapshot | None = None

    def handle_event(self, event: dict[str, Any]) -> dict[str, Any]:
        result = sync_modeltracker_event(
            event,
            model_path=self._model_path,
            correction_radians=self._correction_radians,
        )
        snapshot = _snapshot_from_event(event)
        if result.get("ok") is True:
            if snapshot is not None:
                self._previous_snapshot = snapshot
            return result

        if not _is_multi_change_error(result):
            if snapshot is not None:
                self._previous_snapshot = snapshot
            return result

        if snapshot is None:
            return result
        if self._previous_snapshot is None:
            changed = _changed_transform_indexes(snapshot.orient, snapshot.transl)
            self._previous_snapshot = snapshot
            return {
                "ok": True,
                "updated": False,
                "status": f"Captured ModelTracker baseline with {len(changed)} changed transforms.",
            }

        changed_since_previous = _snapshot_changed_indexes(self._previous_snapshot, snapshot)
        if len(changed_since_previous) != 1:
            self._previous_snapshot = snapshot
            names = [
                snapshot.names[index]
                for index in changed_since_previous
                if index < len(snapshot.names)
            ]
            return _failure(
                "Cannot resolve one changed ModelTracker element from previous snapshot: "
                f"indexes={changed_since_previous}, names={names}"
            )

        selected_event = dict(event)
        selected_event["event_index"] = changed_since_previous[0]
        selected_result = sync_modeltracker_event(
            selected_event,
            model_path=self._model_path,
            correction_radians=self._correction_radians,
        )
        self._previous_snapshot = snapshot
        return selected_result


def sync_modeltracker_event(
    event: dict[str, Any],
    *,
    model_path: str | Path = DEFAULT_HOLOGRAM_MODEL_PATH,
    correction_radians: float = DEFAULT_CORRECTION_RADIANS,
) -> dict[str, Any]:
    names = _string_list(event.get("names"))
    orient = _matrix_list(event.get("orient"))
    transl = _matrix_list(event.get("transl"))
    mesh_centers = _point_list(event.get("mesh_centers"))

    if not orient or not transl or not names or not mesh_centers:
        return {
            "ok": True,
            "updated": False,
            "status": "Waiting for orient/transl/names/mesh_centers inputs.",
        }
    if len(orient) != len(transl):
        return _failure(f"orient/transl count mismatch: {len(orient)}/{len(transl)}")

    event_index = _optional_event_index(event.get("event_index"))
    if event_index is False:
        return _failure("event_index must be an integer")

    object_result = _event_object(names, orient, transl, mesh_centers, event_index)
    if not object_result["ok"]:
        return object_result

    object_name = object_result["object_name"]
    event_index = object_result["event_index"]
    if not isinstance(object_name, str) or not isinstance(event_index, int):
        return _failure("invalid ModelTracker event selection")

    path = Path(model_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _failure(f"hologram model file not found: {path}")
    except json.JSONDecodeError:
        return _failure(f"hologram model file is not valid JSON: {path}")
    if not isinstance(data, dict):
        return _failure("hologram model must be a JSON object")

    body_result = _find_body(data, object_name)
    if not body_result["ok"]:
        return body_result
    body = body_result["body"]
    if not isinstance(body, dict):
        return _failure(f"{object_name}: invalid body")

    center = _rotate_point(mesh_centers[event_index], correction_radians)
    rotation = _corrected_rotation(orient[event_index], correction_radians)
    quat = _quaternion_from_matrix(rotation)

    update_result = _update_pose_fields(body, object_name, center, quat)
    if not update_result["ok"]:
        return update_result

    try:
        _write_json_atomic(path, data)
    except OSError as exc:
        return _failure(f"failed to write hologram model: {exc}", retryable=True)

    return {
        "ok": True,
        "updated": True,
        "object_name": object_name,
        "event_index": event_index,
        "pose": {
            "xyz": center,
            "quat_xyzw": quat,
        },
    }


def _event_object(
    raw_names: list[str],
    orient: list[list[list[float]]],
    transl: list[list[list[float]]],
    mesh_centers: list[list[float]],
    event_index: int | None,
) -> dict[str, Any]:
    names = [_canonical_modeltracker_name(name) for name in raw_names]
    if event_index is not None:
        if event_index < 0 or event_index >= len(names):
            return _failure(f"event_index {event_index} is outside names list")
        if event_index >= len(orient) or event_index >= len(mesh_centers):
            return _failure(f"event_index {event_index} is outside event data")
        return {
            "ok": True,
            "object_name": names[event_index],
            "event_index": event_index,
        }

    if len(names) == len(orient):
        changed = _changed_transform_indexes(orient, transl)
        if not changed:
            return {
                "ok": True,
                "updated": False,
                "status": "No changed ModelTracker transform.",
            }
        if len(changed) > 1:
            changed_names = [names[index] for index in changed if index < len(names)]
            return _failure(
                "Expected one changed ModelTracker element, "
                f"got {len(changed)}: indexes={changed}, names={changed_names}"
            )
        event_index = changed[0]
        if event_index >= len(mesh_centers):
            return _failure(f"{names[event_index]}: no mesh center at index {event_index}")
        return {
            "ok": True,
            "object_name": names[event_index],
            "event_index": event_index,
        }

    if len(names) == 1 and len(orient) == 1:
        if _is_identity_transform(orient[0]) and _is_identity_transform(transl[0]):
            return {
                "ok": True,
                "updated": False,
                "status": "No changed ModelTracker transform.",
            }
        if not mesh_centers:
            return _failure(f"{names[0]}: no mesh center")
        return {
            "ok": True,
            "object_name": names[0],
            "event_index": 0,
        }

    return _failure(
        "Cannot map ModelTracker event: "
        f"names={len(names)}, orient={len(orient)}, meshes={len(mesh_centers)}"
    )


class _ModelTrackerSnapshot:
    def __init__(
        self,
        *,
        names: list[str],
        orient: list[list[list[float]]],
        transl: list[list[list[float]]],
        mesh_centers: list[list[float]],
    ) -> None:
        self.names = names
        self.orient = orient
        self.transl = transl
        self.mesh_centers = mesh_centers


def _snapshot_from_event(event: dict[str, Any]) -> _ModelTrackerSnapshot | None:
    names = [_canonical_modeltracker_name(name) for name in _string_list(event.get("names"))]
    orient = _matrix_list(event.get("orient"))
    transl = _matrix_list(event.get("transl"))
    mesh_centers = _point_list(event.get("mesh_centers"))
    if not names or not orient or not transl or not mesh_centers:
        return None
    return _ModelTrackerSnapshot(
        names=names,
        orient=orient,
        transl=transl,
        mesh_centers=mesh_centers,
    )


def _snapshot_changed_indexes(
    previous: _ModelTrackerSnapshot,
    current: _ModelTrackerSnapshot,
) -> list[int]:
    count = min(
        len(previous.names),
        len(current.names),
        len(previous.orient),
        len(current.orient),
        len(previous.transl),
        len(current.transl),
        len(previous.mesh_centers),
        len(current.mesh_centers),
    )
    changed: list[int] = []
    for index in range(count):
        if previous.names[index] != current.names[index]:
            changed.append(index)
            continue
        if not _matrix_almost_equal(previous.orient[index], current.orient[index]):
            changed.append(index)
            continue
        if not _matrix_almost_equal(previous.transl[index], current.transl[index]):
            changed.append(index)
            continue
        if not _vector_almost_equal(previous.mesh_centers[index], current.mesh_centers[index]):
            changed.append(index)
    return changed


def _matrix_almost_equal(left: list[list[float]], right: list[list[float]]) -> bool:
    for row in range(4):
        for col in range(4):
            if abs(left[row][col] - right[row][col]) > SNAPSHOT_TOLERANCE:
                return False
    return True


def _vector_almost_equal(left: list[float], right: list[float]) -> bool:
    return all(abs(left[index] - right[index]) <= SNAPSHOT_TOLERANCE for index in range(len(left)))


def _is_multi_change_error(result: dict[str, Any]) -> bool:
    error = result.get("error")
    return isinstance(error, str) and error.startswith("Expected one changed ModelTracker element")


def _optional_event_index(value: Any) -> int | None | bool:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return False
    return value


def _canonical_modeltracker_name(value: str) -> str:
    text = value.strip()
    snappy_match = SNAPPY_NAME_RE.fullmatch(text)
    if snappy_match is not None:
        return f"dynamic_{int(snappy_match.group(1))}"
    return canonical_dynamic_name(text)


def _update_pose_fields(
    body: dict[str, Any],
    object_name: str,
    center: list[float],
    quat: list[float],
) -> dict[str, Any]:
    pose = body.get("pose")
    axis = body.get("axis")
    if not isinstance(pose, dict):
        return _failure(f"{object_name}: invalid pose")
    if not isinstance(axis, dict):
        return _failure(f"{object_name}: invalid axis")

    dimensions_result = _body_dimensions(body, object_name)
    if not dimensions_result["ok"]:
        return dimensions_result

    dimensions = dimensions_result["dimensions"]
    if not isinstance(dimensions, dict):
        return _failure(f"{object_name}: invalid solid dimensions")

    local_x = _normalize(_rotated_axis(quat, [1.0, 0.0, 0.0]))
    local_y = _normalize(_rotated_axis(quat, [0.0, 1.0, 0.0]))
    local_z = _normalize(_rotated_axis(quat, [0.0, 0.0, 1.0]))

    length_x = float(dimensions["x"])
    section_y = float(dimensions["y"])
    section_z = float(dimensions["z"])
    if _should_flip_axis_order(axis, local_x):
        axis_start = _offset(center, local_x, length_x / 2.0)
        axis_end = _offset(center, local_x, -length_x / 2.0)
    else:
        axis_start = _offset(center, local_x, -length_x / 2.0)
        axis_end = _offset(center, local_x, length_x / 2.0)

    pose["xyz"] = center
    pose["quat_xyzw"] = quat
    axis["start_xyz"] = axis_start
    axis["end_xyz"] = axis_end

    features = body.get("features")
    if isinstance(features, dict):
        _update_feature(features, "end_start", world_xyz=axis_start)
        _update_feature(features, "end_end", world_xyz=axis_end)
        _update_feature(
            features,
            "face_bottom",
            center_xyz=_offset(center, local_z, -section_z / 2.0),
            normal_xyz=[-local_z[0], -local_z[1], -local_z[2]],
        )
        _update_feature(
            features,
            "face_top",
            center_xyz=_offset(center, local_z, section_z / 2.0),
            normal_xyz=local_z,
        )
        _update_feature(
            features,
            "face_side_a",
            center_xyz=_offset(center, local_y, -section_y / 2.0),
            normal_xyz=[-local_y[0], -local_y[1], -local_y[2]],
        )
        _update_feature(
            features,
            "face_side_b",
            center_xyz=_offset(center, local_y, section_y / 2.0),
            normal_xyz=local_y,
        )
    return {"ok": True}


def _body_dimensions(body: dict[str, Any], object_name: str) -> dict[str, Any]:
    solid = body.get("solid")
    if not isinstance(solid, dict):
        return _failure(f"{object_name}: invalid solid")
    dimensions = solid.get("dimensions")
    if not isinstance(dimensions, dict):
        return _failure(f"{object_name}: invalid solid dimensions")

    values: dict[str, float] = {}
    for key in ("x", "y", "z"):
        number = _finite_float(dimensions.get(key))
        if number is None:
            return _failure(f"{object_name}: dimension {key} is not finite")
        values[key] = number
    return {"ok": True, "dimensions": values}


def _find_body(data: dict[str, Any], object_name: str) -> dict[str, Any]:
    bodies = data.get("bodies")
    if not isinstance(bodies, list):
        return _failure("hologram model bodies must be a list")
    matches = [
        body
        for body in bodies
        if isinstance(body, dict)
        and isinstance(body.get("id"), str)
        and canonical_dynamic_name(body["id"]) == object_name
    ]
    if not matches:
        return _failure(f"{object_name}: not found in hologram_model.json")
    if len(matches) > 1:
        return _failure(f"{object_name}: duplicate bodies in hologram_model.json")
    return {"ok": True, "body": matches[0]}


def _changed_transform_indexes(
    orient: list[list[list[float]]],
    transl: list[list[list[float]]],
) -> list[int]:
    changed: list[int] = []
    for index in range(len(orient)):
        if _is_identity_transform(orient[index]) and _is_identity_transform(transl[index]):
            continue
        changed.append(index)
    return changed


def _is_identity_transform(matrix: list[list[float]], tolerance: float = 1e-6) -> bool:
    for row in range(4):
        for col in range(4):
            expected = 1.0 if row == col else 0.0
            if abs(matrix[row][col] - expected) > tolerance:
                return False
    return True


def _corrected_rotation(
    matrix: list[list[float]],
    correction_radians: float,
) -> list[list[float]]:
    correction = _rotation_z(correction_radians)
    rotation = [[matrix[row][col] for col in range(3)] for row in range(3)]
    return _matmul3(correction, rotation)


def _rotation_z(angle: float) -> list[list[float]]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    return [
        [cos_angle, -sin_angle, 0.0],
        [sin_angle, cos_angle, 0.0],
        [0.0, 0.0, 1.0],
    ]


def _rotate_point(point: list[float], angle: float) -> list[float]:
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    x, y, z = point
    return _clean_vector(
        [
            cos_angle * x - sin_angle * y,
            sin_angle * x + cos_angle * y,
            z,
        ]
    )


def _matmul3(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][inner] * right[inner][col] for inner in range(3)) for col in range(3)]
        for row in range(3)
    ]


def _quaternion_from_matrix(matrix: list[list[float]]) -> list[float]:
    normalized = [_normalize(row) for row in matrix]
    m00, m01, m02 = normalized[0]
    m10, m11, m12 = normalized[1]
    m20, m21, m22 = normalized[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (m21 - m12) / scale
        y = (m02 - m20) / scale
        z = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / scale
        x = 0.25 * scale
        y = (m01 + m10) / scale
        z = (m02 + m20) / scale
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / scale
        x = (m01 + m10) / scale
        y = 0.25 * scale
        z = (m12 + m21) / scale
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / scale
        x = (m02 + m20) / scale
        y = (m12 + m21) / scale
        z = 0.25 * scale

    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= EPSILON:
        raise ValueError("zero-length quaternion")
    return _clean_vector([x / norm, y / norm, z / norm, w / norm])


def _rotated_axis(quat: list[float], axis: list[float]) -> list[float]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    ax, ay, az = axis
    return [
        (1.0 - 2.0 * y * y - 2.0 * z * z) * ax
        + (2.0 * x * y - 2.0 * z * w) * ay
        + (2.0 * x * z + 2.0 * y * w) * az,
        (2.0 * x * y + 2.0 * z * w) * ax
        + (1.0 - 2.0 * x * x - 2.0 * z * z) * ay
        + (2.0 * y * z - 2.0 * x * w) * az,
        (2.0 * x * z - 2.0 * y * w) * ax
        + (2.0 * y * z + 2.0 * x * w) * ay
        + (1.0 - 2.0 * x * x - 2.0 * y * y) * az,
    ]


def _normalize(vector: list[float]) -> list[float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= EPSILON:
        raise ValueError("zero-length vector")
    return [value / length for value in vector]


def _offset(center: list[float], axis: list[float], distance: float) -> list[float]:
    return _clean_vector(
        [
            center[0] + axis[0] * distance,
            center[1] + axis[1] * distance,
            center[2] + axis[2] * distance,
        ]
    )


def _should_flip_axis_order(axis: dict[str, Any], local_x: list[float]) -> bool:
    start = axis.get("start_xyz")
    end = axis.get("end_xyz")
    if not isinstance(start, list) or not isinstance(end, list) or len(start) != 3 or len(end) != 3:
        return False
    start_values = _finite_sequence(start)
    end_values = _finite_sequence(end)
    if start_values is None or end_values is None:
        return False
    direction = [end_values[index] - start_values[index] for index in range(3)]
    return sum(direction[index] * local_x[index] for index in range(3)) < 0.0


def _update_feature(
    features: dict[str, Any],
    name: str,
    *,
    world_xyz: list[float] | None = None,
    center_xyz: list[float] | None = None,
    normal_xyz: list[float] | None = None,
) -> None:
    feature = features.get(name)
    if not isinstance(feature, dict):
        return
    if world_xyz is not None and "world_xyz" in feature:
        feature["world_xyz"] = _clean_vector(world_xyz)
    if center_xyz is not None and "center_xyz" in feature:
        feature["center_xyz"] = _clean_vector(center_xyz)
    if normal_xyz is not None and "normal_xyz" in feature:
        feature["normal_xyz"] = _clean_vector(normal_xyz)


def _matrix_list(value: Any) -> list[list[list[float]]]:
    if not isinstance(value, list):
        return []
    matrices: list[list[list[float]]] = []
    for item in value:
        matrix = _matrix(item)
        if matrix is None:
            return []
        matrices.append(matrix)
    return matrices


def _matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    matrix: list[list[float]] = []
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return None
        numbers = _finite_sequence(row)
        if numbers is None:
            return None
        matrix.append(numbers)
    return matrix


def _point_list(value: Any) -> list[list[float]]:
    if not isinstance(value, list):
        return []
    points: list[list[float]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 3:
            return []
        point = _finite_sequence(item)
        if point is None:
            return []
        points.append(point)
    return points


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _finite_sequence(values: list[Any]) -> list[float] | None:
    numbers: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            return None
        numbers.append(number)
    return numbers


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _clean_vector(values: list[float]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        if abs(value) < 1e-12:
            cleaned.append(0.0)
        else:
            cleaned.append(round(float(value), 12))
    return cleaned


def _write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def _failure(error: str, *, retryable: bool = False) -> dict[str, Any]:
    return {
        "ok": False,
        "error": error,
        "retryable": retryable,
    }
