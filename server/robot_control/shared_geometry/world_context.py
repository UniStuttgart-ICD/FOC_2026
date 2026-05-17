from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

DEFAULT_PHYSICAL_MODEL_PATH = Path(__file__).with_name("physical_model.json")
DEFAULT_HOLOGRAM_MODEL_PATH = Path(__file__).with_name("hologram_model.json")

_DYNAMIC_NAME_RE = re.compile(r"^dynamic_(\d+)$")


class GeometryWorldContextStore:
    """Renders paired shared-geometry models for agent prompt context."""

    def __init__(
        self,
        *,
        physical_model_path: str | Path = DEFAULT_PHYSICAL_MODEL_PATH,
        hologram_model_path: str | Path = DEFAULT_HOLOGRAM_MODEL_PATH,
    ) -> None:
        self._physical_model_path = Path(physical_model_path)
        self._hologram_model_path = Path(hologram_model_path)

    def render_instruction_block(self) -> str:
        try:
            physical_model = _load_model(self._physical_model_path)
            hologram_model = _load_model(self._hologram_model_path)
            elements = _paired_elements(physical_model, hologram_model)
        except GeometryWorldContextError as exc:
            return _blocked_context(str(exc))

        lines = [
            "Geometry World Context:",
            "- This context gives semantic dynamic object identity and hologram target poses.",
            "- MoveIt/RViz remains the live source pose authority for physical objects.",
            "- Hologram target poses are desired object poses, not TCP poses.",
            f"- physical model: {_model_name(physical_model)}",
            f"- hologram model: {_model_name(hologram_model)}",
            f"- units: {hologram_model.get('units') or physical_model.get('units') or 'unknown'}",
            "- paired dynamic objects:",
            json.dumps(elements, ensure_ascii=True),
        ]
        return "\n".join(lines)


class GeometryWorldContextError(ValueError):
    pass


def canonical_dynamic_name(value: str) -> str:
    text = value.strip()
    match = _DYNAMIC_NAME_RE.fullmatch(text)
    if match is None:
        return text
    return f"dynamic_{int(match.group(1))}"


def _load_model(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GeometryWorldContextError(f"geometry model file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GeometryWorldContextError(f"geometry model file is not valid JSON: {path}") from exc
    if not isinstance(data, dict):
        raise GeometryWorldContextError(f"geometry model is not an object: {path}")
    return data


def _paired_elements(
    physical_model: dict[str, Any],
    hologram_model: dict[str, Any],
) -> list[dict[str, Any]]:
    physical_bodies = _body_map(physical_model, "physical")
    hologram_bodies = _body_map(hologram_model, "hologram")
    if not physical_bodies:
        raise GeometryWorldContextError("physical model has no bodies")

    missing_hologram = sorted(set(physical_bodies) - set(hologram_bodies))
    if missing_hologram:
        missing = ", ".join(missing_hologram)
        raise GeometryWorldContextError(f"{missing} is missing from hologram model")

    missing_physical = sorted(set(hologram_bodies) - set(physical_bodies))
    if missing_physical:
        missing = ", ".join(missing_physical)
        raise GeometryWorldContextError(f"{missing} is missing from physical model")

    physical_model_name = _model_name(physical_model)
    hologram_model_name = _model_name(hologram_model)
    elements: list[dict[str, Any]] = []
    physical_object_names = set(physical_bodies)
    for object_name in sorted(physical_bodies):
        physical_body = physical_bodies[object_name]
        hologram_body = hologram_bodies[object_name]
        elements.append(
            {
                "object_name": object_name,
                "physical_model_name": physical_model_name,
                "hologram_model_name": hologram_model_name,
                "label": str(physical_body.get("label") or ""),
                "family": str(physical_body.get("family") or ""),
                "role": _physical_role(physical_body, object_name, physical_object_names),
                "target_pose": _target_pose(hologram_body, object_name),
            }
        )
    return elements


def _body_map(model: dict[str, Any], model_role: str) -> dict[str, dict[str, Any]]:
    bodies = model.get("bodies")
    if not isinstance(bodies, list):
        raise GeometryWorldContextError(f"{model_role} model bodies must be a list")

    mapped: dict[str, dict[str, Any]] = {}
    for body in bodies:
        if not isinstance(body, dict):
            raise GeometryWorldContextError(f"{model_role} model body must be an object")
        raw_id = body.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise GeometryWorldContextError(f"{model_role} model body has no id")
        object_name = canonical_dynamic_name(raw_id)
        if object_name in mapped:
            raise GeometryWorldContextError(f"{model_role} model has duplicate body {object_name}")
        mapped[object_name] = body
    return mapped


def _target_pose(body: dict[str, Any], object_name: str) -> dict[str, dict[str, float]]:
    pose = body.get("pose")
    if not isinstance(pose, dict):
        raise GeometryWorldContextError(f"invalid hologram target pose for {object_name}")

    xyz = pose.get("xyz")
    quat = pose.get("quat_xyzw")
    if not isinstance(xyz, list) or len(xyz) != 3:
        raise GeometryWorldContextError(f"invalid hologram target pose for {object_name}")
    if not isinstance(quat, list) or len(quat) != 4:
        raise GeometryWorldContextError(f"invalid hologram target pose for {object_name}")

    position_values = _finite_vector(xyz, object_name)
    orientation_values = _finite_vector(quat, object_name)

    position = {
        "x": position_values[0],
        "y": position_values[1],
        "z": position_values[2],
    }
    orientation = {
        "x": orientation_values[0],
        "y": orientation_values[1],
        "z": orientation_values[2],
        "w": orientation_values[3],
    }
    return {"position": position, "orientation": orientation}


def _physical_role(
    body: dict[str, Any],
    object_name: str,
    physical_object_names: set[str],
) -> dict[str, Any]:
    state = body.get("state")
    if not isinstance(state, dict):
        raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")

    role = state.get("role")
    if not isinstance(role, dict):
        raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")

    role_type = role.get("type")
    if role_type == "supporting_column":
        supports = _role_dynamic_refs(
            role.get("supports"),
            object_name,
            physical_object_names,
        )
        return {"type": role_type, "supports": supports}
    if role_type == "beam_supported_by":
        supported_by = _role_dynamic_refs(
            role.get("supported_by"),
            object_name,
            physical_object_names,
        )
        return {"type": role_type, "supported_by": supported_by}
    if role_type == "unassigned":
        return {"type": role_type}
    raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")


def _role_dynamic_refs(
    values: Any,
    object_name: str,
    physical_object_names: set[str],
) -> list[str]:
    if not isinstance(values, list) or not values:
        raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")
    refs: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")
        ref = value.strip()
        if canonical_dynamic_name(ref) != ref:
            raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")
        if ref not in physical_object_names:
            raise GeometryWorldContextError(f"invalid physical role payload for {object_name}")
        refs.append(ref)
    return refs


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _finite_vector(values: list[Any], object_name: str) -> list[float]:
    numbers: list[float] = []
    for value in values:
        number = _finite_float(value)
        if number is None:
            raise GeometryWorldContextError(f"invalid hologram target pose for {object_name}")
        numbers.append(number)
    return numbers


def _model_name(model: dict[str, Any]) -> str:
    return str(model.get("name") or "unnamed")


def _blocked_context(reason: str) -> str:
    return "\n".join(
        [
            "Geometry World Context:",
            f"- BLOCKED: {reason}.",
            "- Geometry-grounded pick-place must not infer a fallback target from the physical model or current object pose.",
        ]
    )
