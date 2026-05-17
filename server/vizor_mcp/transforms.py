from __future__ import annotations

from typing import Any

ZERO_OFFSET_M = {"x": 0.0, "y": 0.0, "z": 0.0}
DF2025_ARCHIVED_OFFSET_M = {"x": -0.173, "y": 0.051, "z": 0.103}
DF2025_OFFSET_M = DF2025_ARCHIVED_OFFSET_M
ACTIVE_TRANSFORM_NAME = "unity_axis_swap_to_base_link_no_offset"
DF2025_CALIBRATION_NAME = "df2025_archived_offset"


def unity_position_to_robot(
    position: dict[str, Any],
    *,
    offset_m: dict[str, float] | None = None,
) -> dict[str, float]:
    """Convert a Vizor user/manual position into MoveIt's base_link frame."""
    offset = ZERO_OFFSET_M if offset_m is None else offset_m
    unity_x = float(position["x"])
    unity_y = float(position["y"])
    unity_z = float(position["z"])
    return {
        "x": unity_y + float(offset["x"]),
        "y": unity_z + float(offset["y"]),
        "z": unity_x + float(offset["z"]),
    }


def unity_position_to_archived_robot_base(
    position: dict[str, Any],
    *,
    offset_m: dict[str, float] | None = None,
) -> dict[str, float]:
    """Archived DF2025 source-to-robot-base mapping kept for calibration checks."""
    offset = ZERO_OFFSET_M if offset_m is None else offset_m
    unity_x = float(position["x"])
    unity_y = float(position["y"])
    unity_z = float(position["z"])
    return {
        "x": -unity_y + float(offset["x"]),
        "y": -unity_z + float(offset["y"]),
        "z": unity_x + float(offset["z"]),
    }
