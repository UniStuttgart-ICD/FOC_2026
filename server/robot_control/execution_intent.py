from __future__ import annotations

import re

ROBOT_ACTION_TERMS = (
    "move",
    "go",
    "raise",
    "lower",
    "lift",
    "drop",
    "wave",
    "draw",
    "point",
    "pick",
    "place",
    "put",
    "gesture",
    "open",
    "close",
    "grab",
    "release",
    "reach",
)
EXECUTE_REQUEST_TERMS = (
    "execute",
    "run the plan",
    "run it",
    "send it",
    "send the plan",
    "start the motion",
    "perform the motion",
    "go ahead",
    "proceed",
    "confirm",
    "confirmed",
    "do it",
    "yes",
    "yeah",
    "yep",
    "ok",
    "okay",
)
PLANNING_ONLY_TERMS = (
    "do not execute",
    "don't execute",
    "dont execute",
    "without executing",
    "plan only",
    "only plan",
    "just plan",
)


def explicit_execute_requested(user_text: str | None) -> bool:
    if user_text is None:
        return False
    return any(_contains_phrase(user_text, term) for term in EXECUTE_REQUEST_TERMS)


def looks_like_robot_action_request(text: str) -> bool:
    normalized = text.casefold()
    return any(term in normalized for term in ROBOT_ACTION_TERMS)


def should_auto_execute_successful_plan(user_text: str | None) -> bool:
    if user_text is None:
        return False
    if any(_contains_phrase(user_text, term) for term in PLANNING_ONLY_TERMS):
        return False
    return explicit_execute_requested(user_text)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = phrase.casefold().split()
    if not normalized_phrase:
        return False
    pattern = r"(?<!\w)" + r"\s+".join(re.escape(part) for part in normalized_phrase) + r"(?!\w)"
    return re.search(pattern, text.casefold()) is not None
