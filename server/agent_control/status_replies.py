"""Persona-shaped robot status replies."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

PLAN_READY_REPLIES = (
    "Hmmmmmm. The plan is ready; the robot has not moved yet. "
    "Approve execution, and I will set it to work.",
    "The plan is ready. No motion yet; I need explicit approval before "
    "the arm gets ideas.",
    "A workable plan is prepared. It has not moved anything; approve it "
    "when you want the arm to act.",
)

EXECUTION_COMPLETE_REPLIES = (
    "Execution is complete. The arm did the job cleanly and came back "
    "from its little errand.",
    "Done. The verified motion completed cleanly; the robot may stop "
    "looking so important.",
    "The action is complete. The motion passed verification, so the small "
    "ceremony is over.",
)

PHYSICAL_EXECUTION_FAILED_REPLIES = (
    "Execution is complete in AR/RViz, but physical execution failed.",
    "AR/RViz execution completed, but the physical robot reported that it failed.",
)

PHYSICAL_STATUS_UNAVAILABLE_REPLIES = (
    "Execution is complete in AR/RViz; physical status is unavailable.",
    "AR/RViz execution completed; physical robot status is unavailable.",
)

ACTION_COMPLETE_REPLIES = (
    "Action complete. The robot did its small bit.",
    "Done. That robot action completed.",
)


def plan_ready_reply(seed: object | None = None) -> str:
    return _select_reply(PLAN_READY_REPLIES, seed)


def execution_complete_reply(seed: object | None = None) -> str:
    return _select_reply(EXECUTION_COMPLETE_REPLIES, seed)


def physical_execution_failed_reply(seed: object | None = None) -> str:
    return _select_reply(PHYSICAL_EXECUTION_FAILED_REPLIES, seed)


def physical_status_unavailable_reply(seed: object | None = None) -> str:
    return _select_reply(PHYSICAL_STATUS_UNAVAILABLE_REPLIES, seed)


def action_complete_reply(seed: object | None = None) -> str:
    return _select_reply(ACTION_COMPLETE_REPLIES, seed)


def _select_reply(replies: Sequence[str], seed: object | None) -> str:
    if not replies:
        raise ValueError("at least one reply is required")
    rendered_seed = "" if seed is None else str(seed)
    digest = hashlib.sha256(rendered_seed.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], byteorder="big") % len(replies)
    return replies[index]
