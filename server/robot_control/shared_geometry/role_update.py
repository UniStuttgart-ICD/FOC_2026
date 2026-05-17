from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from robot_control.shared_geometry.world_context import (
    DEFAULT_PHYSICAL_MODEL_PATH,
    canonical_dynamic_name,
)

_SUPPORTED_ROLE_TYPES = {"unassigned", "supporting_column", "beam_supported_by"}
_VIEW_DEPENDENT_ROLE_WORDS = {"left", "right"}
_REJECTED_ROLE_TYPES = {"inventory", "built"}


def update_dynamic_role(
    object_name: str,
    role: object,
    reason: str,
    *,
    model_path: str | Path = DEFAULT_PHYSICAL_MODEL_PATH,
) -> dict[str, object]:
    path = Path(model_path)
    try:
        model = _load_model(path)
        bodies = _body_map(model)
        canonical_object_name = canonical_dynamic_name(object_name)
        body = bodies.get(canonical_object_name)
        if body is None:
            return _failure(f"unknown object {canonical_object_name}")

        normalized_role = _normalize_role(role, bodies)
        if _is_failure(normalized_role):
            return normalized_role

        operation_history = model.get("operation_history")
        if not isinstance(operation_history, list):
            return _failure("physical model operation_history must be a list")

        state = body.get("state")
        if not isinstance(state, dict):
            return _failure(f"{canonical_object_name} state must be an object")

        state["role"] = normalized_role
        operation_history.append(
            {
                "op": "dynamic_role_update",
                "status": "applied",
                "object_name": canonical_object_name,
                "reason": reason,
                "role": normalized_role,
            }
        )
        _atomic_write_model(path, model)
        return {
            "ok": True,
            "object_name": canonical_object_name,
            "role": normalized_role,
            "physical_model_updated": True,
        }
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return _failure(str(exc))


def _normalize_role(role: object, bodies: dict[str, dict[str, Any]]) -> dict[str, object]:
    if not isinstance(role, dict):
        return _failure("role must be one of the structured dynamic role payloads")

    role_type = role.get("type")
    if not isinstance(role_type, str) or not role_type.strip():
        return _failure("role.type must be one of the structured dynamic role types")

    role_type = role_type.strip()
    if role_type in _REJECTED_ROLE_TYPES:
        return _failure(f"{role_type} is not a dynamic role")
    if _is_view_dependent_role(role_type):
        return _failure(f"{role_type} is view-dependent; use structural roles instead")
    if role_type not in _SUPPORTED_ROLE_TYPES:
        return _failure(f"unsupported dynamic role type {role_type}")

    if role_type == "unassigned":
        if set(role) != {"type"}:
            return _failure("unassigned role only accepts type")
        return {"type": "unassigned"}

    if role_type == "supporting_column":
        if set(role) != {"type", "supports"}:
            return _failure("supporting_column role requires only supports")
        references = _normalize_references(role.get("supports"), bodies)
        if isinstance(references, dict):
            return references
        return {"type": "supporting_column", "supports": references}

    if set(role) != {"type", "supported_by"}:
        return _failure("beam_supported_by role requires only supported_by")
    references = _normalize_references(role.get("supported_by"), bodies)
    if isinstance(references, dict):
        return references
    return {"type": "beam_supported_by", "supported_by": references}


def _normalize_references(value: object, bodies: dict[str, dict[str, Any]]) -> list[str] | dict[str, object]:
    if not isinstance(value, list) or not value:
        return _failure("role references must be a non-empty list")

    references: list[str] = []
    for raw_reference in value:
        if not isinstance(raw_reference, str) or not raw_reference.strip():
            return _failure("role references must be dynamic object names")
        reference = canonical_dynamic_name(raw_reference)
        if reference not in bodies:
            return _failure(f"unknown reference {reference}")
        references.append(reference)
    return references


def _load_model(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("physical model must be a JSON object")
    return data


def _body_map(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    bodies = model.get("bodies")
    if not isinstance(bodies, list):
        raise ValueError("physical model bodies must be a list")

    mapped: dict[str, dict[str, Any]] = {}
    for body in bodies:
        if not isinstance(body, dict):
            raise ValueError("physical model body must be an object")
        raw_id = body.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise ValueError("physical model body has no id")
        object_name = canonical_dynamic_name(raw_id)
        if object_name in mapped:
            raise ValueError(f"physical model has duplicate body {object_name}")
        mapped[object_name] = body
    return mapped


def _atomic_write_model(path: Path, model: dict[str, Any]) -> None:
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(model, temp_file, ensure_ascii=True, indent=2)
            temp_file.write("\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _is_view_dependent_role(role_type: str) -> bool:
    words = set(role_type.lower().replace("-", "_").split("_"))
    return bool(words & _VIEW_DEPENDENT_ROLE_WORDS)


def _is_failure(value: object) -> bool:
    return isinstance(value, dict) and value.get("ok") is False


def _failure(error: str) -> dict[str, object]:
    return {
        "ok": False,
        "error": error,
        "correction": "Use a structured dynamic role with valid dynamic object references from physical_model.json.",
        "retryable": True,
    }
