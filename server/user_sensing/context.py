from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from robot_control.shared_geometry import canonical_dynamic_name


@dataclass
class UserSensingSnapshot:
    observed_at_s: float | None = None
    context: dict[str, Any] | None = None


class UserSensingContextStore:
    def __init__(self, *, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._snapshot = UserSensingSnapshot()
        self._time_fn = time_fn

    def render_instruction_block(self) -> str:
        lines = [
            "User sensing context:",
            "- This context is advisory only.",
            "- Use fresh gaze/user/target data to resolve this, that, there, or near me.",
            "- If relevant sensing data is missing or stale, ask a clarifying question instead of guessing.",
            f"- status age: {self._status_age_text()}",
        ]
        if self._snapshot.context is None:
            lines.append("- No user sensing has been observed yet.")
            return "\n".join(lines)

        context = self._snapshot.context
        lines.append(f"- attention target: {self._attention_text(context.get('attention'))}")
        lines.append(f"- gaze target: {self._gaze_text(context.get('gaze'))}")
        gaze_object_candidate = self._gaze_object_candidate_text(context.get("gaze"))
        if gaze_object_candidate is not None:
            lines.append(f"- gaze object candidate: {gaze_object_candidate}")
        lines.append(f"- user position: {self._pose_text(context.get('user'))}")
        lines.append(f"- manual target: {self._pose_text(context.get('manual_target'))}")
        return "\n".join(lines)

    def update_from_tool_result(self, output: str) -> None:
        structured_content = _structured_content(output)
        if not isinstance(structured_content, dict):
            return
        self._snapshot.context = structured_content
        self._snapshot.observed_at_s = self._time_fn()

    def summary_attributes(self) -> dict[str, Any]:
        context = self._snapshot.context
        if not isinstance(context, dict):
            return {"context.available": False}

        attributes: dict[str, Any] = {
            "context.available": True,
            "freshness.stale": _nested_value(context, "freshness", "stale"),
        }
        attributes.update(_field_attributes("attention", context.get("attention")))
        attributes.update(_field_attributes("gaze", context.get("gaze")))
        attributes.update(_field_attributes("user", context.get("user")))
        attributes.update(_field_attributes("manual_target", context.get("manual_target")))
        return {key: value for key, value in attributes.items() if value is not None}

    def summary_text(self) -> str:
        attributes = self.summary_attributes()
        if attributes.get("context.available") is not True:
            return "unavailable"
        return (
            "attention={attention} attention_fresh={attention_fresh} "
            "gaze={gaze} gaze_stale={gaze_stale} gaze_age_s={gaze_age} "
            "user_available={user_available} user_stale={user_stale} "
            "manual_target_available={manual_available} manual_target_stale={manual_stale}"
        ).format(
            attention=attributes.get("attention.dominant_target"),
            attention_fresh=attributes.get("attention.fresh"),
            gaze=attributes.get("gaze.target"),
            gaze_stale=attributes.get("gaze.stale"),
            gaze_age=attributes.get("gaze.age_s"),
            user_available=attributes.get("user.available"),
            user_stale=attributes.get("user.stale"),
            manual_available=attributes.get("manual_target.available"),
            manual_stale=attributes.get("manual_target.stale"),
        )

    def fresh_user_position(self, *, max_age_s: float) -> dict[str, float] | None:
        if self._snapshot.observed_at_s is None:
            return None
        if self._time_fn() - self._snapshot.observed_at_s > max_age_s:
            return None
        context = self._snapshot.context
        if not isinstance(context, dict):
            return None
        user = context.get("user")
        if not isinstance(user, dict) or user.get("available") is not True:
            return None
        if user.get("stale") is True:
            return None
        frame = user.get("frame")
        if isinstance(frame, str) and frame and frame != "base_link":
            return None
        position = user.get("position")
        if not isinstance(position, dict):
            return None
        try:
            return {
                "x": float(position["x"]),
                "y": float(position["y"]),
                "z": float(position["z"]),
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _status_age_text(self) -> str:
        if self._snapshot.observed_at_s is None:
            return "unknown"
        return f"{self._time_fn() - self._snapshot.observed_at_s:.1f}s"

    @staticmethod
    def _gaze_text(value: Any) -> str:
        if not isinstance(value, dict) or value.get("available") is not True:
            return "unavailable"
        target = value.get("target")
        if not isinstance(target, str) or not target:
            return "unavailable"
        return _with_stale_suffix(target, value)

    @staticmethod
    def _gaze_object_candidate_text(value: Any) -> str | None:
        candidate = _gaze_object_candidate(value)
        if candidate is None or not isinstance(value, dict):
            return None
        return _with_stale_suffix(candidate, value)

    @staticmethod
    def _attention_text(value: Any) -> str:
        if not isinstance(value, dict) or value.get("available") is not True:
            return "unavailable"
        target = value.get("dominant_target") or value.get("last_stable_target")
        if not isinstance(target, str) or not target:
            return "unavailable"

        ranked_targets = value.get("ranked_targets")
        top = ranked_targets[0] if isinstance(ranked_targets, list) and ranked_targets else {}
        confidence = top.get("confidence") if isinstance(top, dict) else None
        dwell_s = top.get("dwell_s") if isinstance(top, dict) else None
        details: list[str] = []
        if isinstance(confidence, str) and confidence:
            details.append(f"{confidence} confidence")
        if isinstance(dwell_s, (int, float)):
            details.append(f"dwell {dwell_s:.1f}s")
        if value.get("fresh") is False:
            details.append("stale")
        if not details:
            return target
        return f"{target} ({', '.join(details)})"

    @staticmethod
    def _pose_text(value: Any) -> str:
        if not isinstance(value, dict) or value.get("available") is not True:
            return "unavailable"
        position = value.get("position")
        if not isinstance(position, dict):
            return "unavailable"
        try:
            text = "x={:.3f}, y={:.3f}, z={:.3f}".format(
                float(position["x"]),
                float(position["y"]),
                float(position["z"]),
            )
        except (KeyError, TypeError, ValueError):
            return "unavailable"
        frame = value.get("frame")
        if isinstance(frame, str) and frame:
            text = f"{text}, frame={frame}"
        return _with_stale_suffix(text, value)


def _with_stale_suffix(text: str, value: dict[str, Any]) -> str:
    if value.get("stale") is not True:
        return text
    age = value.get("age_s")
    if isinstance(age, (int, float)):
        return f"{text} (stale, age {age:.1f}s)"
    return f"{text} (stale)"


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content", payload)


def _nested_value(value: dict[str, Any], key: str, nested_key: str) -> Any:
    nested = value.get(key)
    if not isinstance(nested, dict):
        return None
    return nested.get(nested_key)


def _field_attributes(prefix: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {f"{prefix}.available": False}
    attributes = {
        f"{prefix}.available": value.get("available"),
        f"{prefix}.stale": value.get("stale"),
        f"{prefix}.age_s": value.get("age_s"),
    }
    if prefix == "attention":
        attributes[f"{prefix}.fresh"] = value.get("fresh")
        attributes[f"{prefix}.dominant_target"] = value.get("dominant_target")
        attributes[f"{prefix}.last_stable_target"] = value.get("last_stable_target")
    elif prefix == "gaze":
        attributes[f"{prefix}.target"] = value.get("target")
        attributes[f"{prefix}.raw_target"] = value.get("raw_target")
        attributes[f"{prefix}.object_candidate"] = _gaze_object_candidate(value)
    return attributes


def _gaze_object_candidate(value: Any) -> str | None:
    if not isinstance(value, dict) or value.get("available") is not True:
        return None
    raw_target = value.get("raw_target")
    if isinstance(raw_target, str) and raw_target.strip():
        return raw_target.strip()
    target = value.get("target")
    if isinstance(target, str) and target.strip().isdigit():
        return canonical_dynamic_name(f"dynamic_{target.strip()}")
    return None
