from __future__ import annotations

from collections.abc import Callable

from test_support.live_robot_smoke import (
    LiveSmokeRun,
    ValidationResult,
    validate_ambiguous_clarification,
    validate_bit_movement,
    validate_position_query,
    validate_wave_motion,
)

Validator = Callable[[LiveSmokeRun], ValidationResult]


def validate_move_up_bit(run: LiveSmokeRun) -> ValidationResult:
    return validate_bit_movement(run, direction="up")


def validate_move_down_bit(run: LiveSmokeRun) -> ValidationResult:
    return validate_bit_movement(run, direction="down")


VALIDATORS: dict[str, Validator] = {
    "current_position_query": validate_position_query,
    "move_up_bit": validate_move_up_bit,
    "move_down_bit": validate_move_down_bit,
    "wave_motion": validate_wave_motion,
    "ambiguous_clarification": validate_ambiguous_clarification,
}


def get_validator(name: str) -> Validator:
    try:
        return VALIDATORS[name]
    except KeyError as exc:
        raise ValueError(f"unknown model eval validator: {name}") from exc
