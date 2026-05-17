from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

from robot_control.shared_geometry.world_context import (
    DEFAULT_PHYSICAL_MODEL_PATH,
    canonical_dynamic_name,
)

_ALLOWED_REASONS = {
    "verified_pick_place_release",
    "verified_place_release",
    "operator_sync",
}


def update_physical_model_pose(
    object_name: str,
    reason: str,
    pose_evidence: dict[str, object],
    *,
    model_path: str | Path = DEFAULT_PHYSICAL_MODEL_PATH,
) -> dict[str, object]:
    target_name = canonical_dynamic_name(object_name) if isinstance(object_name, str) else ""
    if not target_name:
        return _failure("object_name is required", "Provide a named dynamic object.", False)
    if reason not in _ALLOWED_REASONS:
        return _failure(
            f"unsupported reason: {reason}",
            "Use verified_pick_place_release, verified_place_release, or operator_sync.",
            False,
        )

    evidence = _validated_pose_evidence(target_name, pose_evidence)
    if not evidence["ok"]:
        return evidence

    path = Path(model_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return _failure(f"physical model file not found: {path}", "Provide an existing model_path.", False)
    except json.JSONDecodeError:
        return _failure(f"physical model file is not valid JSON: {path}", "Repair the JSON model file.", False)

    if not isinstance(data, dict):
        return _failure("physical model must be a JSON object", "Provide a valid physical model.", False)

    body_result = _find_body(data, target_name)
    if not body_result["ok"]:
        return body_result
    body = body_result["body"]
    if not isinstance(body, dict):
        return _failure(f"invalid body for {target_name}", "Repair the physical model body.", False)

    dimensions_result = _body_dimensions(body, target_name)
    if not dimensions_result["ok"]:
        return dimensions_result

    center = evidence["xyz"]
    quat = evidence["quat_xyzw"]
    if not isinstance(center, list) or not isinstance(quat, list):
        return _failure("invalid pose evidence", "Provide finite position and orientation.", False)
    if not isinstance(dimensions_result["x"], float) or not isinstance(dimensions_result["z"], float):
        return _failure(f"invalid dimensions for {target_name}", "Repair solid dimensions.", False)

    pose = body.get("pose")
    axis = body.get("axis")
    if not isinstance(pose, dict):
        return _failure(f"invalid pose for {target_name}", "Repair the physical model body pose.", False)
    if not isinstance(axis, dict):
        return _failure(f"invalid axis for {target_name}", "Repair the physical model body axis.", False)

    length_x = dimensions_result["x"]
    half_z = dimensions_result["z"] / 2.0
    local_x = _rotated_axis(quat, [1.0, 0.0, 0.0])
    local_z = _rotated_axis(quat, [0.0, 0.0, 1.0])
    if _should_flip_axis_order(axis, local_x):
        axis_start = _offset(center, local_x, length_x / 2.0)
        axis_end = _offset(center, local_x, -length_x / 2.0)
    else:
        axis_start = _offset(center, local_x, -length_x / 2.0)
        axis_end = _offset(center, local_x, length_x / 2.0)
    face_bottom = _offset(center, local_z, -half_z)
    face_top = _offset(center, local_z, half_z)

    pose["xyz"] = center
    pose["quat_xyzw"] = quat
    axis["start_xyz"] = axis_start
    axis["end_xyz"] = axis_end

    features = body.get("features")
    if isinstance(features, dict):
        _update_existing_world_xyz(features, "end_start", axis_start)
        _update_existing_world_xyz(features, "end_end", axis_end)
        _update_existing_center_xyz(features, "face_bottom", face_bottom)
        _update_existing_center_xyz(features, "face_top", face_top)

    history = data.get("operation_history")
    if not isinstance(history, list):
        history = []
        data["operation_history"] = history
    history.append(
        {
            "op": "physical_model_pose_update",
            "status": "applied",
            "object_name": target_name,
            "reason": reason,
            "source": evidence["source"],
            "pose": evidence["pose"],
        }
    )

    try:
        _write_json_atomic(path, data)
    except OSError as exc:
        return _failure(
            f"failed to write physical model: {exc}",
            "Check model_path permissions and retry.",
            True,
        )

    return {
        "ok": True,
        "object_name": target_name,
        "reason": reason,
        "source": evidence["source"],
    }


def _validated_pose_evidence(
    target_name: str,
    pose_evidence: dict[str, object],
) -> dict[str, object]:
    raw_name = pose_evidence.get("object_name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        return _failure("pose evidence object_name is required", "Provide evidence for a named object.", False)
    evidence_name = canonical_dynamic_name(raw_name)
    if evidence_name != target_name:
        return _failure(
            f"pose evidence object_name {evidence_name} does not match {target_name}",
            "Provide pose evidence for the requested object.",
            False,
        )

    source = pose_evidence.get("source")
    if not isinstance(source, str) or not source.strip():
        return _failure("pose evidence source is required", "Provide the pose evidence source.", False)

    pose = pose_evidence.get("pose")
    if not isinstance(pose, dict):
        return _failure(
            "pose evidence is required; bounds-only evidence is not accepted",
            "Provide pose evidence with position and orientation.",
            False,
        )

    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict):
        return _failure("pose position is required", "Provide finite position x, y, and z.", False)
    if not isinstance(orientation, dict):
        return _failure("pose quaternion is required", "Provide finite quaternion x, y, z, and w.", False)

    xyz = _finite_named_vector(position, ("x", "y", "z"))
    if xyz is None:
        return _failure("pose position must be finite", "Provide finite position x, y, and z.", False)
    quat = _finite_named_vector(orientation, ("x", "y", "z", "w"))
    if quat is None:
        return _failure("pose quaternion must be finite", "Provide finite quaternion x, y, z, and w.", False)
    if math.isclose(math.sqrt(sum(value * value for value in quat)), 0.0, abs_tol=1e-12):
        return _failure("pose quaternion must be non-zero", "Provide a valid orientation quaternion.", False)

    normalized_pose = {
        "position": {"x": xyz[0], "y": xyz[1], "z": xyz[2]},
        "orientation": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]},
    }
    return {
        "ok": True,
        "source": source,
        "pose": normalized_pose,
        "xyz": xyz,
        "quat_xyzw": quat,
    }


def _find_body(data: dict[str, Any], target_name: str) -> dict[str, object]:
    bodies = data.get("bodies")
    if not isinstance(bodies, list):
        return _failure("physical model bodies must be a list", "Repair the physical model bodies.", False)

    matches = [
        body
        for body in bodies
        if isinstance(body, dict)
        and isinstance(body.get("id"), str)
        and canonical_dynamic_name(body["id"]) == target_name
    ]
    if not matches:
        return _failure(f"{target_name} is missing from physical model", "Sync a known physical model object.", False)
    if len(matches) > 1:
        return _failure(f"multiple bodies match {target_name}", "Remove duplicate physical model bodies.", False)
    return {"ok": True, "body": matches[0]}


def _body_dimensions(body: dict[str, Any], object_name: str) -> dict[str, object]:
    solid = body.get("solid")
    if not isinstance(solid, dict):
        return _failure(f"invalid solid for {object_name}", "Repair the physical model body solid.", False)
    dimensions = solid.get("dimensions")
    if not isinstance(dimensions, dict):
        return _failure(f"invalid solid dimensions for {object_name}", "Repair solid dimensions.", False)

    length_x = _finite_float(dimensions.get("x"))
    section_z = _finite_float(dimensions.get("z"))
    if length_x is None or section_z is None:
        return _failure(f"solid dimensions must be finite for {object_name}", "Provide finite x and z dimensions.", False)
    return {"ok": True, "x": length_x, "z": section_z}


def _rotated_axis(quat: list[float], axis: list[float]) -> list[float]:
    x, y, z, w = quat
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    ax, ay, az = axis
    return [
        (1.0 - 2.0 * y * y - 2.0 * z * z) * ax + (2.0 * x * y - 2.0 * z * w) * ay + (2.0 * x * z + 2.0 * y * w) * az,
        (2.0 * x * y + 2.0 * z * w) * ax + (1.0 - 2.0 * x * x - 2.0 * z * z) * ay + (2.0 * y * z - 2.0 * x * w) * az,
        (2.0 * x * z - 2.0 * y * w) * ax + (2.0 * y * z + 2.0 * x * w) * ay + (1.0 - 2.0 * x * x - 2.0 * y * y) * az,
    ]


def _offset(center: list[float], axis: list[float], distance: float) -> list[float]:
    return [
        center[0] + axis[0] * distance,
        center[1] + axis[1] * distance,
        center[2] + axis[2] * distance,
    ]


def _should_flip_axis_order(axis: dict[str, Any], local_x: list[float]) -> bool:
    start = axis.get("start_xyz")
    end = axis.get("end_xyz")
    if not isinstance(start, list) or not isinstance(end, list) or len(start) != 3 or len(end) != 3:
        return False
    start_values = _finite_sequence(start)
    end_values = _finite_sequence(end)
    if start_values is None or end_values is None:
        return False
    direction = [
        end_values[index] - start_values[index]
        for index in range(3)
    ]
    dot = sum(direction[index] * local_x[index] for index in range(3))
    return dot < 0.0


def _finite_sequence(values: list[Any]) -> list[float] | None:
    numbers: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            return None
        numbers.append(number)
    return numbers


def _update_existing_world_xyz(features: dict[str, Any], feature_name: str, value: list[float]) -> None:
    feature = features.get(feature_name)
    if isinstance(feature, dict) and "world_xyz" in feature:
        feature["world_xyz"] = value


def _update_existing_center_xyz(features: dict[str, Any], feature_name: str, value: list[float]) -> None:
    feature = features.get(feature_name)
    if isinstance(feature, dict) and "center_xyz" in feature:
        feature["center_xyz"] = value


def _finite_named_vector(values: dict[Any, Any], names: tuple[str, ...]) -> list[float] | None:
    vector: list[float] = []
    for name in names:
        number = _finite_float(values.get(name))
        if number is None:
            return None
        vector.append(number)
    return vector


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


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


def _failure(error: str, correction: str, retryable: bool) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "correction": correction,
        "retryable": retryable,
    }
