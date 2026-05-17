import pytest

from model_eval.scenarios import get_scenario_pack
from model_eval.validators import get_validator


def test_core_robot_commands_pack_shape() -> None:
    pack = get_scenario_pack("core_robot_commands")

    assert pack.name == "core_robot_commands"
    assert [scenario.name for scenario in pack.scenarios] == [
        "current-position",
        "move-up-bit",
        "move-down-bit",
        "visible-wave",
        "ambiguous-move-there",
    ]
    assert [scenario.prompt for scenario in pack.scenarios] == [
        "what is the current position?",
        "move up a bit",
        "move down a bit",
        "Maive, can you wave to me?",
        "move there",
    ]
    assert [scenario.validator_name for scenario in pack.scenarios] == [
        "current_position_query",
        "move_up_bit",
        "move_down_bit",
        "wave_motion",
        "ambiguous_clarification",
    ]


def test_pack_validators_resolve() -> None:
    pack = get_scenario_pack("core_robot_commands")

    for scenario in pack.scenarios:
        validator = get_validator(scenario.validator_name)
        assert callable(validator)


def test_unknown_pack_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown scenario pack"):
        get_scenario_pack("missing")


def test_unknown_validator_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model eval validator"):
        get_validator("missing")
