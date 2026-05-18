from __future__ import annotations

import math
from typing import Any

PLANNING_SCENE_COMPONENTS = 4 | 8 | 16 | 512

_IDENTITY_POSE = {
    "position": {"x": 0.0, "y": 0.0, "z": 0.0},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}


def summarize_planning_scene(payload: dict[str, Any], *, planning_frame: str | None = None) -> dict[str, Any]:
    scene = payload.get("scene", payload)
    colors = _object_colors(scene.get("object_colors") or [])
    objects: list[dict[str, Any]] = []

    world = scene.get("world") if isinstance(scene.get("world"), dict) else {}
    for collision_object in world.get("collision_objects") or []:
        summary = _summarize_collision_object(
            collision_object,
            state="free",
            color=colors.get(str(collision_object.get("id", ""))),
        )
        if summary is not None:
            objects.append(summary)

    robot_state = scene.get("robot_state") if isinstance(scene.get("robot_state"), dict) else {}
    for attached in robot_state.get("attached_collision_objects") or []:
        collision_object = attached.get("object") if isinstance(attached, dict) else None
        if not isinstance(collision_object, dict):
            continue
        summary = _summarize_collision_object(
            collision_object,
            state="attached",
            color=colors.get(str(collision_object.get("id", ""))),
        )
        if summary is None:
            continue
        link_name = attached.get("link_name")
        if isinstance(link_name, str) and link_name:
            summary["attached_to"] = link_name
        touch_links = attached.get("touch_links")
        if isinstance(touch_links, list):
            summary["touch_links"] = [str(link) for link in touch_links]
        objects.append(summary)

    frame = planning_frame or payload.get("planning_frame") or _first_frame(objects) or "base_link"
    return {
        "planning_frame": frame,
        "objects": objects,
        "object_count": len(objects),
    }


def object_context(scene_summary: dict[str, Any], object_name: str) -> dict[str, Any] | None:
    objects = scene_summary.get("objects") or []
    target = next((item for item in objects if item.get("name") == object_name), None)
    if target is None:
        return None

    context = dict(target)
    context["planning_frame"] = scene_summary.get("planning_frame")
    context["clearance"] = _clearance(target, objects)
    context["scene_relations"] = _scene_relations(target, objects)
    context["grasp_faces"] = _grasp_faces(target, objects)
    return context


def available_object_names(scene_summary: dict[str, Any]) -> list[str]:
    return [str(item["name"]) for item in scene_summary.get("objects") or [] if item.get("name")]


def _summarize_collision_object(
    collision_object: dict[str, Any],
    *,
    state: str,
    color: dict[str, float] | None,
) -> dict[str, Any] | None:
    object_id = collision_object.get("id")
    if not isinstance(object_id, str) or not object_id:
        return None

    shapes: list[dict[str, Any]] = []
    shape_bounds: list[dict[str, Any]] = []
    object_pose_value = collision_object.get("pose")
    object_pose = _pose(object_pose_value if isinstance(object_pose_value, dict) else _IDENTITY_POSE)

    primitives = collision_object.get("primitives") or []
    primitive_poses = collision_object.get("primitive_poses") or []
    for index, primitive in enumerate(primitives):
        if not isinstance(primitive, dict):
            continue
        pose = _compose_pose(object_pose, _pose_at(primitive_poses, index))
        summary = _primitive_summary(primitive, pose)
        shapes.append(summary)
        if summary.get("bounds") is not None:
            shape_bounds.append(summary["bounds"])

    meshes = collision_object.get("meshes") or []
    mesh_poses = collision_object.get("mesh_poses") or []
    for index, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            continue
        pose = _compose_pose(object_pose, _pose_at(mesh_poses, index))
        summary = _mesh_summary(mesh, pose)
        shapes.append(summary)
        if summary.get("bounds") is not None:
            shape_bounds.append(summary["bounds"])

    planes = collision_object.get("planes") or []
    plane_poses = collision_object.get("plane_poses") or []
    for index, plane in enumerate(planes):
        if not isinstance(plane, dict):
            continue
        shapes.append(_plane_summary(plane, _compose_pose(object_pose, _pose_at(plane_poses, index))))

    bounds = _merge_bounds(shape_bounds)
    header_value = collision_object.get("header")
    header = header_value if isinstance(header_value, dict) else {}
    summary: dict[str, Any] = {
        "name": object_id,
        "state": state,
        "frame": header.get("frame_id") or "base_link",
        "pose": _first_shape_pose(shapes),
        "bounds": bounds,
        "shapes": shapes,
    }
    if color is not None:
        summary["color"] = color
    subframe_names = collision_object.get("subframe_names")
    if isinstance(subframe_names, list) and subframe_names:
        summary["subframes"] = [str(name) for name in subframe_names]
    return summary


def _primitive_summary(primitive: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    dimensions = [_clean(float(value)) for value in primitive.get("dimensions") or []]
    kind = _primitive_kind(primitive.get("type"))
    summary = {
        "kind": kind,
        "dimensions": dimensions,
        "pose": pose,
        "bounds": _primitive_bounds(kind, dimensions, pose),
    }
    alignment_axis = _primitive_alignment_axis(kind, dimensions, pose)
    if alignment_axis is not None:
        summary["alignment_axis"] = alignment_axis
    return summary


def _mesh_summary(mesh: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    vertices = mesh.get("vertices") or []
    transformed = [_transform_point(_point(vertex), pose) for vertex in vertices if isinstance(vertex, dict)]
    summary = {
        "kind": "mesh",
        "vertex_count": len(vertices),
        "triangle_count": len(mesh.get("triangles") or []),
        "pose": pose,
        "bounds": _bounds_from_points(transformed),
    }
    alignment_axis = _principal_axis(transformed)
    if alignment_axis is not None:
        summary["alignment_axis"] = alignment_axis
    return summary


def _plane_summary(plane: dict[str, Any], pose: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "plane",
        "coef": [_clean(float(value)) for value in plane.get("coef") or []],
        "pose": pose,
        "bounds": None,
    }


def _primitive_bounds(kind: str, dimensions: list[float], pose: dict[str, Any]) -> dict[str, Any] | None:
    if kind == "box" and len(dimensions) >= 3:
        x, y, z = dimensions[:3]
        corners = _box_corners(x, y, z)
    elif kind == "sphere" and dimensions:
        radius = dimensions[0]
        corners = _box_corners(radius * 2, radius * 2, radius * 2)
    elif kind in {"cylinder", "cone"} and len(dimensions) >= 2:
        height, radius = dimensions[:2]
        corners = _box_corners(radius * 2, radius * 2, height)
    else:
        return None
    return _bounds_from_points([_transform_point(point, pose) for point in corners])


def _box_corners(x: float, y: float, z: float) -> list[dict[str, float]]:
    return [
        {"x": sx * x / 2, "y": sy * y / 2, "z": sz * z / 2}
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]


def _bounds_from_points(points: list[dict[str, float]]) -> dict[str, Any] | None:
    if not points:
        return None
    mins = {axis: min(point[axis] for point in points) for axis in ("x", "y", "z")}
    maxes = {axis: max(point[axis] for point in points) for axis in ("x", "y", "z")}
    return _bounds_from_min_max(mins, maxes)


def _bounds_from_min_max(mins: dict[str, float], maxes: dict[str, float]) -> dict[str, Any]:
    return {
        "min": {axis: _clean(mins[axis]) for axis in ("x", "y", "z")},
        "max": {axis: _clean(maxes[axis]) for axis in ("x", "y", "z")},
        "center": {axis: _clean((mins[axis] + maxes[axis]) / 2) for axis in ("x", "y", "z")},
        "size": {axis: _clean(maxes[axis] - mins[axis]) for axis in ("x", "y", "z")},
    }


def _merge_bounds(bounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    usable = [item for item in bounds if isinstance(item, dict)]
    if not usable:
        return None
    mins = {axis: min(item["min"][axis] for item in usable) for axis in ("x", "y", "z")}
    maxes = {axis: max(item["max"][axis] for item in usable) for axis in ("x", "y", "z")}
    return _bounds_from_min_max(mins, maxes)


def _clearance(target: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any] | None:
    target_bounds = target.get("bounds")
    if not isinstance(target_bounds, dict):
        return None
    support = next(
        (
            item
            for item in objects
            if item.get("name") == "ground_plane" and isinstance(item.get("bounds"), dict)
        ),
        None,
    )
    if support is None:
        return None
    z_m = target_bounds["min"]["z"] - support["bounds"]["max"]["z"]
    return {"reference": "ground_plane", "z_m": _clean(z_m)}


def _grasp_faces(target: Any, objects: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    bounds = target.get("bounds") if isinstance(target, dict) else target
    if not isinstance(bounds, dict):
        return []
    alignment_axis = _object_alignment_axis(target) if isinstance(target, dict) else None
    mn = bounds["min"]
    mx = bounds["max"]
    center = bounds["center"]
    size = bounds["size"]
    faces = [
        _face("right", {"x": 1.0, "y": 0.0, "z": 0.0}, {"x": mx["x"], "y": center["y"], "z": center["z"]}, size["y"], size["z"], alignment_axis),
        _face("left", {"x": -1.0, "y": 0.0, "z": 0.0}, {"x": mn["x"], "y": center["y"], "z": center["z"]}, size["y"], size["z"], alignment_axis),
        _face("front", {"x": 0.0, "y": 1.0, "z": 0.0}, {"x": center["x"], "y": mx["y"], "z": center["z"]}, size["x"], size["z"], alignment_axis),
        _face("back", {"x": 0.0, "y": -1.0, "z": 0.0}, {"x": center["x"], "y": mn["y"], "z": center["z"]}, size["x"], size["z"], alignment_axis),
        _face("top", {"x": 0.0, "y": 0.0, "z": 1.0}, {"x": center["x"], "y": center["y"], "z": mx["z"]}, size["x"], size["y"], alignment_axis),
        _face("bottom", {"x": 0.0, "y": 0.0, "z": -1.0}, {"x": center["x"], "y": center["y"], "z": mn["z"]}, size["x"], size["y"], alignment_axis),
    ]
    return [_annotated_grasp_face(face, target, objects or []) for face in faces]


def _face(
    name: str,
    normal: dict[str, float],
    center: dict[str, float],
    width: float,
    height: float,
    alignment_axis: dict[str, float] | None = None,
) -> dict[str, Any]:
    face = {
        "name": name,
        "normal": normal,
        "center": {axis: _clean(center[axis]) for axis in ("x", "y", "z")},
        "size": {"width": _clean(width), "height": _clean(height)},
        "area": _clean(width * height),
    }
    projected_axis = _project_axis_onto_face(alignment_axis, normal)
    if projected_axis is not None:
        face["alignment_axis"] = projected_axis
    return face


def _annotated_grasp_face(face: dict[str, Any], target: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any]:
    if face["name"] not in {"front", "back", "left", "right"}:
        return face
    neighbors = _neighbor_objects(target, objects)
    clearance = _face_clearance(face, target, neighbors)
    if clearance is not None:
        face["scene_clearance_m"] = _clean(clearance)
    nearest = _nearest_neighbor(target, neighbors)
    relations = _scene_relations(target, objects)
    assembly_center = relations.get("assembly_center") if isinstance(relations, dict) else None
    faces_nearest = nearest is not None and _face_points_toward(face, target, nearest["bounds"]["center"])
    faces_center = isinstance(assembly_center, dict) and _face_points_toward(face, target, assembly_center)
    if faces_nearest or faces_center:
        face["beam_side_preference"] = "inner"
    elif nearest is not None or isinstance(assembly_center, dict):
        face["beam_side_preference"] = "outer"
    else:
        face["beam_side_preference"] = "unknown"
    return face


def _scene_relations(target: dict[str, Any], objects: list[dict[str, Any]]) -> dict[str, Any]:
    neighbors = _neighbor_objects(target, objects)
    relations: dict[str, Any] = {}
    nearest = _nearest_neighbor(target, neighbors)
    if nearest is not None:
        relations["nearest_object"] = {
            "name": nearest.get("name"),
            "center": nearest["bounds"]["center"],
            "distance_m": _clean(_center_distance(target, nearest)),
        }
    if neighbors:
        center = {
            axis: _clean(sum(float(item["bounds"]["center"][axis]) for item in neighbors) / len(neighbors))
            for axis in ("x", "y", "z")
        }
        relations["assembly_center"] = center
    return relations


def _neighbor_objects(target: dict[str, Any], objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_name = target.get("name")
    neighbors: list[dict[str, Any]] = []
    for item in objects:
        if item.get("name") == target_name or item.get("name") == "ground_plane":
            continue
        if not isinstance(item.get("bounds"), dict):
            continue
        if item.get("state") == "attached":
            continue
        neighbors.append(item)
    return neighbors


def _nearest_neighbor(target: dict[str, Any], neighbors: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not neighbors:
        return None
    return min(neighbors, key=lambda item: _center_distance(target, item))


def _center_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_center = first["bounds"]["center"]
    second_center = second["bounds"]["center"]
    return math.sqrt(sum((float(first_center[axis]) - float(second_center[axis])) ** 2 for axis in ("x", "y", "z")))


def _face_points_toward(face: dict[str, Any], target: dict[str, Any], point: dict[str, Any]) -> bool:
    normal = face["normal"]
    target_center = target["bounds"]["center"]
    vector = {axis: float(point[axis]) - float(target_center[axis]) for axis in ("x", "y", "z")}
    distance = math.sqrt(sum(value * value for value in vector.values()))
    if distance <= 0.0:
        return False
    dot = sum(float(normal[axis]) * vector[axis] for axis in ("x", "y", "z")) / distance
    return dot > 0.25


def _face_clearance(
    face: dict[str, Any],
    target: dict[str, Any],
    neighbors: list[dict[str, Any]],
) -> float | None:
    if not neighbors:
        return None
    normal = face["normal"]
    axis = max(("x", "y", "z"), key=lambda item: abs(float(normal[item])))
    direction = 1.0 if float(normal[axis]) >= 0.0 else -1.0
    target_bounds = target["bounds"]
    face_center = face["center"]
    candidates: list[float] = []
    for item in neighbors:
        bounds = item["bounds"]
        perpendicular_axes = [item_axis for item_axis in ("x", "y", "z") if item_axis != axis]
        if not all(_bounds_overlap(target_bounds, bounds, item_axis) for item_axis in perpendicular_axes):
            continue
        if direction > 0.0:
            distance = float(bounds["min"][axis]) - float(face_center[axis])
        else:
            distance = float(face_center[axis]) - float(bounds["max"][axis])
        if distance >= 0.0:
            candidates.append(distance)
    return min(candidates) if candidates else None


def _bounds_overlap(first: dict[str, Any], second: dict[str, Any], axis: str) -> bool:
    return float(first["min"][axis]) <= float(second["max"][axis]) and float(second["min"][axis]) <= float(first["max"][axis])


def _object_alignment_axis(target: dict[str, Any]) -> dict[str, float] | None:
    best: tuple[float, dict[str, float]] | None = None
    for shape in target.get("shapes") or []:
        if not isinstance(shape, dict) or not isinstance(shape.get("alignment_axis"), dict):
            continue
        extent = _shape_alignment_extent(shape)
        if best is None or extent > best[0]:
            best = (extent, shape["alignment_axis"])
    if best is not None:
        return best[1]
    bounds = target.get("bounds")
    if not isinstance(bounds, dict):
        return None
    size = bounds.get("size")
    if not isinstance(size, dict):
        return None
    axis = max(("x", "y", "z"), key=lambda item: float(size.get(item, 0.0)))
    return {name: 1.0 if name == axis else 0.0 for name in ("x", "y", "z")}


def _shape_alignment_extent(shape: dict[str, Any]) -> float:
    dimensions = shape.get("dimensions")
    if isinstance(dimensions, list) and dimensions:
        return max(float(value) for value in dimensions)
    bounds = shape.get("bounds")
    if isinstance(bounds, dict) and isinstance(bounds.get("size"), dict):
        return max(float(bounds["size"].get(axis, 0.0)) for axis in ("x", "y", "z"))
    return 0.0


def _primitive_alignment_axis(kind: str, dimensions: list[float], pose: dict[str, Any]) -> dict[str, float] | None:
    if kind != "box" or len(dimensions) < 3:
        return None
    axis_index = max(range(3), key=lambda index: dimensions[index])
    local_axis = [
        {"x": 1.0, "y": 0.0, "z": 0.0},
        {"x": 0.0, "y": 1.0, "z": 0.0},
        {"x": 0.0, "y": 0.0, "z": 1.0},
    ][axis_index]
    return _canonical_axis(_rotate(local_axis, pose["orientation"]))


def _project_axis_onto_face(
    axis: dict[str, float] | None,
    normal: dict[str, float],
) -> dict[str, float] | None:
    if axis is None:
        return None
    ax, ay, az = _unit_tuple(axis)
    nx, ny, nz = _unit_tuple(normal)
    dot = ax * nx + ay * ny + az * nz
    projected = {"x": ax - dot * nx, "y": ay - dot * ny, "z": az - dot * nz}
    return _canonical_axis(projected)


def _principal_axis(points: list[dict[str, float]]) -> dict[str, float] | None:
    if len(points) < 2:
        return None
    mean = {axis: sum(point[axis] for point in points) / len(points) for axis in ("x", "y", "z")}
    centered = [{axis: point[axis] - mean[axis] for axis in ("x", "y", "z")} for point in points]
    covariance = [
        [sum(point[a] * point[b] for point in centered) for b in ("x", "y", "z")]
        for a in ("x", "y", "z")
    ]
    diagonal = [covariance[index][index] for index in range(3)]
    if max(diagonal) <= 1e-18:
        return None
    start_index = max(range(3), key=lambda index: diagonal[index])
    vector = [1.0 if index == start_index else 0.0 for index in range(3)]
    for _ in range(12):
        vector = [
            sum(covariance[row][column] * vector[column] for column in range(3))
            for row in range(3)
        ]
        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 1e-12:
            return None
        vector = [value / norm for value in vector]
    return _canonical_axis({"x": vector[0], "y": vector[1], "z": vector[2]})


def _unit_tuple(value: dict[str, float]) -> tuple[float, float, float]:
    x = float(value["x"])
    y = float(value["y"])
    z = float(value["z"])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("axis must be non-zero")
    return x / norm, y / norm, z / norm


def _canonical_axis(value: dict[str, float]) -> dict[str, float] | None:
    x = float(value["x"])
    y = float(value["y"])
    z = float(value["z"])
    norm = math.sqrt(x * x + y * y + z * z)
    if norm <= 1e-12:
        return None
    x, y, z = x / norm, y / norm, z / norm
    components = [x, y, z]
    dominant = max(range(3), key=lambda index: abs(components[index]))
    if components[dominant] < 0.0:
        components = [-component for component in components]
    return {
        "x": _clean(components[0]),
        "y": _clean(components[1]),
        "z": _clean(components[2]),
    }


def _transform_point(point: dict[str, float], pose: dict[str, Any]) -> dict[str, float]:
    rotated = _rotate(point, pose["orientation"])
    position = pose["position"]
    return {axis: _clean(rotated[axis] + position[axis]) for axis in ("x", "y", "z")}


def _rotate(point: dict[str, float], quaternion: dict[str, float]) -> dict[str, float]:
    x, y, z = point["x"], point["y"], point["z"]
    qx, qy, qz, qw = (quaternion[axis] for axis in ("x", "y", "z", "w"))
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        return dict(point)
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm

    tx = 2 * (qy * z - qz * y)
    ty = 2 * (qz * x - qx * z)
    tz = 2 * (qx * y - qy * x)
    return {
        "x": x + qw * tx + (qy * tz - qz * ty),
        "y": y + qw * ty + (qz * tx - qx * tz),
        "z": z + qw * tz + (qx * ty - qy * tx),
    }


def _pose_at(poses: list[Any], index: int) -> dict[str, Any]:
    if index < len(poses) and isinstance(poses[index], dict):
        return _pose(poses[index])
    return _pose(_IDENTITY_POSE)


def _compose_pose(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    parent_position = parent["position"]
    rotated_child = _rotate(child["position"], parent["orientation"])
    return {
        "position": {
            axis: _clean(parent_position[axis] + rotated_child[axis])
            for axis in ("x", "y", "z")
        },
        "orientation": _multiply_quaternions(parent["orientation"], child["orientation"]),
    }


def _multiply_quaternions(first: dict[str, float], second: dict[str, float]) -> dict[str, float]:
    ax, ay, az, aw = (first[axis] for axis in ("x", "y", "z", "w"))
    bx, by, bz, bw = (second[axis] for axis in ("x", "y", "z", "w"))
    return {
        "x": _clean(aw * bx + ax * bw + ay * bz - az * by),
        "y": _clean(aw * by - ax * bz + ay * bw + az * bx),
        "z": _clean(aw * bz + ax * by - ay * bx + az * bw),
        "w": _clean(aw * bw - ax * bx - ay * by - az * bz),
    }


def _pose(value: dict[str, Any]) -> dict[str, Any]:
    position_value = value.get("position")
    orientation_value = value.get("orientation")
    position = _point(position_value if isinstance(position_value, dict) else {})
    orientation = orientation_value if isinstance(orientation_value, dict) else {}
    return {
        "position": position,
        "orientation": {
            "x": _clean(float(orientation.get("x", 0.0))),
            "y": _clean(float(orientation.get("y", 0.0))),
            "z": _clean(float(orientation.get("z", 0.0))),
            "w": _clean(float(orientation.get("w", 1.0))),
        },
    }


def _point(value: dict[str, Any]) -> dict[str, float]:
    return {
        "x": _clean(float(value.get("x", 0.0))),
        "y": _clean(float(value.get("y", 0.0))),
        "z": _clean(float(value.get("z", 0.0))),
    }


def _primitive_kind(value: Any) -> str:
    return {
        1: "box",
        2: "sphere",
        3: "cylinder",
        4: "cone",
    }.get(value, str(value))


def _object_colors(colors: list[Any]) -> dict[str, dict[str, float]]:
    by_id: dict[str, dict[str, float]] = {}
    for item in colors:
        if not isinstance(item, dict) or not item.get("id"):
            continue
        color_value = item.get("color")
        color = color_value if isinstance(color_value, dict) else {}
        by_id[str(item["id"])] = {
            "r": _clean(float(color.get("r", 0.0))),
            "g": _clean(float(color.get("g", 0.0))),
            "b": _clean(float(color.get("b", 0.0))),
            "a": _clean(float(color.get("a", 0.0))),
        }
    return by_id


def _first_shape_pose(shapes: list[dict[str, Any]]) -> dict[str, Any]:
    for shape in shapes:
        pose = shape.get("pose")
        if isinstance(pose, dict):
            return pose
    return _pose(_IDENTITY_POSE)


def _first_frame(objects: list[dict[str, Any]]) -> str | None:
    for item in objects:
        frame = item.get("frame")
        if isinstance(frame, str) and frame:
            return frame
    return None


def _clean(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == -0.0 else rounded
