from robot_control.execution_intent import (
    explicit_execute_requested,
    should_auto_execute_successful_plan,
)


def test_auto_execute_intent_rejects_plain_motion_request() -> None:
    assert should_auto_execute_successful_plan("reach in my direction") is False


def test_auto_execute_intent_accepts_explicit_confirmation() -> None:
    assert should_auto_execute_successful_plan("go ahead and reach in my direction") is True


def test_auto_execute_intent_rejects_planning_only_request() -> None:
    assert should_auto_execute_successful_plan("plan a move up but do not execute") is False


def test_explicit_execute_intent_does_not_match_inside_other_words() -> None:
    assert explicit_execute_requested("look at the plan") is False
    assert explicit_execute_requested("ok, execute it") is True
