from __future__ import annotations

import time
from typing import Any

from vizor_mcp.attention import GazeAttentionTracker
from vizor_mcp.ros_client import (
    GAZE_TOPIC,
    MANUAL_TARGET_TOPIC,
    USER_TRANSFORM_TOPIC,
    TopicReading,
    VizorSensorTransport,
)
from vizor_mcp.transforms import (
    ACTIVE_TRANSFORM_NAME,
    DF2025_ARCHIVED_OFFSET_M,
    DF2025_CALIBRATION_NAME,
    ZERO_OFFSET_M,
    unity_position_to_robot,
)

ROBOT_FRAME = "base_link"


class VizorMcpTools:
    def __init__(
        self,
        *,
        transport: VizorSensorTransport,
        time_fn: Any = time.monotonic,
        attention: GazeAttentionTracker | None = None,
    ) -> None:
        self.transport = transport
        self._time_fn = time_fn
        self._attention = attention or GazeAttentionTracker()
        self._seed_attention()
        self.transport.add_listener(self._record_attention)

    @classmethod
    def with_transport(
        cls,
        transport: VizorSensorTransport,
        *,
        time_fn: Any | None = None,
        attention: GazeAttentionTracker | None = None,
    ) -> "VizorMcpTools":
        if time_fn is None and hasattr(transport, "now_s"):
            def transport_time() -> float:
                return float(getattr(transport, "now_s"))

            time_fn = transport_time
        return cls(
            transport=transport,
            time_fn=time_fn or time.monotonic,
            attention=attention,
        )

    def get_sensor_context(
        self,
        *,
        max_age_s: float = 2.0,
        include_raw: bool = False,
        attention_window_s: float = 8.0,
    ) -> dict[str, Any]:
        now_s = self._time_fn()
        connected = self.transport.is_connected()
        gaze = self._gaze(max_age_s=max_age_s)
        user = self._user_transform(max_age_s=max_age_s, include_raw=include_raw)
        manual_target = self._manual_target(max_age_s=max_age_s)
        stale = any(field.get("stale") is True for field in (gaze, user, manual_target))
        context: dict[str, Any] = {
            "ok": connected,
            "tool": "vizor_get_sensor_context",
            "source": "rosbridge",
            "retryable": not connected,
            "rosbridge": {
                "connected": connected,
                "host": self.transport.host,
                "port": self.transport.port,
            },
            "freshness": {
                "max_age_s": float(max_age_s),
                "stale": stale,
            },
            "gaze": gaze,
            "attention": self._attention.summarize(
                now_s=now_s,
                window_s=attention_window_s,
                stale_after_s=max_age_s,
            ),
            "user": user,
            "manual_target": manual_target,
            "calibration": {
                "name": ACTIVE_TRANSFORM_NAME,
                "offset_enabled": False,
                "active_offset_m": dict(ZERO_OFFSET_M),
                "archived_calibration_name": DF2025_CALIBRATION_NAME,
                "archived_offset_m": dict(DF2025_ARCHIVED_OFFSET_M),
            },
        }
        if connected:
            context.pop("retryable")
        return context

    def get_status(self) -> dict[str, Any]:
        return {
            "ok": self.transport.is_connected(),
            "tool": "vizor_get_status",
            "source": "rosbridge",
            "rosbridge": {
                "connected": self.transport.is_connected(),
                "host": self.transport.host,
                "port": self.transport.port,
            },
            "topics": {
                "gaze": GAZE_TOPIC,
                "user_transform": USER_TRANSFORM_TOPIC,
                "manual_target": MANUAL_TARGET_TOPIC,
            },
        }

    def _seed_attention(self) -> None:
        reading = self.transport.latest_message(GAZE_TOPIC)
        if reading is not None:
            self._record_attention(reading)

    def _record_attention(self, reading: TopicReading) -> None:
        if reading.topic != GAZE_TOPIC:
            return
        raw_target = _message_data(reading.payload)
        target = _normalize_gaze_target(raw_target)
        self._attention.record(target, at_s=reading.received_at_s)

    def _gaze(self, *, max_age_s: float) -> dict[str, Any]:
        reading = self.transport.latest_message(GAZE_TOPIC)
        if reading is None:
            return _missing_field(GAZE_TOPIC, "std_msgs/String")
        raw_target = _message_data(reading.payload)
        target = _normalize_gaze_target(raw_target)
        return {
            "available": target is not None,
            "target": target,
            "raw_target": raw_target,
            **_freshness(reading, self._time_fn(), max_age_s),
        }

    def _user_transform(self, *, max_age_s: float, include_raw: bool) -> dict[str, Any]:
        reading = self.transport.latest_message(USER_TRANSFORM_TOPIC)
        if reading is None:
            return _missing_pose_field(USER_TRANSFORM_TOPIC, "geometry_msgs/Pose")
        payload = reading.payload
        position = _position(payload)
        orientation = _orientation(payload)
        field = {
            "available": position is not None,
            "position": unity_position_to_robot(position) if position is not None else None,
            "orientation": orientation,
            "frame": ROBOT_FRAME,
            **_freshness(reading, self._time_fn(), max_age_s),
        }
        if include_raw or position is not None:
            field["raw_unity_pose"] = {"position": position, "orientation": orientation}
        return field

    def _manual_target(self, *, max_age_s: float) -> dict[str, Any]:
        reading = self.transport.latest_message(MANUAL_TARGET_TOPIC)
        if reading is None:
            return _missing_pose_field(MANUAL_TARGET_TOPIC, "geometry_msgs/Pose")
        payload = reading.payload
        position = _position(payload)
        orientation = _orientation(payload)
        return {
            "available": position is not None,
            "position": unity_position_to_robot(position) if position is not None else None,
            "orientation": orientation,
            "frame": ROBOT_FRAME,
            **_freshness(reading, self._time_fn(), max_age_s),
        }


def _message_data(payload: dict[str, Any]) -> Any:
    return payload.get("data")


def _normalize_gaze_target(raw_target: Any) -> str | None:
    if not isinstance(raw_target, str):
        return None
    target = raw_target.strip()
    if not target:
        return None
    return target.removeprefix("dynamic_")


def _position(payload: dict[str, Any]) -> dict[str, float] | None:
    position = payload.get("position")
    if not isinstance(position, dict):
        return None
    try:
        return {axis: float(position[axis]) for axis in ("x", "y", "z")}
    except (KeyError, TypeError, ValueError):
        return None


def _orientation(payload: dict[str, Any]) -> dict[str, float] | None:
    orientation = payload.get("orientation")
    if not isinstance(orientation, dict):
        return None
    try:
        return {axis: float(orientation[axis]) for axis in ("x", "y", "z", "w")}
    except (KeyError, TypeError, ValueError):
        return None


def _freshness(reading: TopicReading, now_s: float, max_age_s: float) -> dict[str, Any]:
    age_s = max(0.0, float(now_s) - reading.received_at_s)
    return {
        "age_s": age_s,
        "stale": age_s > max_age_s,
        "source_topic": reading.topic,
        "message_type": reading.message_type,
    }


def _missing_field(topic: str, message_type: str) -> dict[str, Any]:
    return {
        "available": False,
        "target": None,
        "raw_target": None,
        "age_s": None,
        "stale": True,
        "source_topic": topic,
        "message_type": message_type,
    }


def _missing_pose_field(topic: str, message_type: str) -> dict[str, Any]:
    return {
        "available": False,
        "position": None,
        "orientation": None,
        "frame": ROBOT_FRAME,
        "age_s": None,
        "stale": True,
        "source_topic": topic,
        "message_type": message_type,
    }
