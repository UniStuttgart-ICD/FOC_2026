from agent_control.status_replies import (
    ACTION_COMPLETE_REPLIES,
    EXECUTION_COMPLETE_REPLIES,
    PHYSICAL_EXECUTION_FAILED_REPLIES,
    PHYSICAL_STATUS_UNAVAILABLE_REPLIES,
    PLAN_READY_REPLIES,
    action_complete_reply,
    execution_complete_reply,
    physical_execution_failed_reply,
    physical_status_unavailable_reply,
    plan_ready_reply,
)


def test_plan_ready_replies_keep_execution_gate_clear() -> None:
    text = plan_ready_reply("dynamic_5")

    assert text in PLAN_READY_REPLIES
    assert "not moved" in text.lower() or "no motion" in text.lower()
    assert "approve" in text.lower() or "approval" in text.lower()
    assert text != "Plan ready."


def test_execution_complete_replies_are_more_than_bare_status() -> None:
    text = execution_complete_reply("dynamic_5")

    assert text in EXECUTION_COMPLETE_REPLIES
    assert "complete" in text.lower() or "done" in text.lower()
    assert text != "Execution complete."


def test_physical_caveat_replies_keep_caveat_visible() -> None:
    failed = physical_execution_failed_reply("task-1")
    unavailable = physical_status_unavailable_reply("task-1")

    assert failed in PHYSICAL_EXECUTION_FAILED_REPLIES
    assert unavailable in PHYSICAL_STATUS_UNAVAILABLE_REPLIES
    assert "physical" in failed.lower()
    assert "failed" in failed.lower()
    assert "physical" in unavailable.lower()
    assert "unavailable" in unavailable.lower()


def test_action_complete_reply_stays_generic() -> None:
    text = action_complete_reply("open-gripper")

    assert text in ACTION_COMPLETE_REPLIES
    assert "complete" in text.lower() or "done" in text.lower()


def test_reply_selection_is_deterministic_for_seed() -> None:
    assert plan_ready_reply("same-seed") == plan_ready_reply("same-seed")
    assert execution_complete_reply("same-seed") == execution_complete_reply("same-seed")
