from __future__ import annotations

import json
import math
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol

from voice_runtime.agent_turn import AgentTurnInput

EXPECTED_BIT_DELTA_M = 0.05
BIT_DELTA_TOLERANCE_M = 0.03
XY_DRIFT_MAX_M = 0.05
EXPECTED_WAVE_LATERAL_SPAN_M = 0.20
MIN_WAVE_LATERAL_SPAN_M = 0.18
MIN_WAVE_VERTICAL_LIFT_M = 0.06
DEFAULT_EVIDENCE_DIR = Path("evidence/live_smoke")
CURRENT_POSE_TOOL_NAME = "moveit_get_current_pose"
MOTION_TOOL_NAMES = {
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
}
EXECUTION_TOOL_NAMES = {
    "moveit_plan_and_execute_free_motion",
    "moveit_plan_and_execute_cartesian_motion",
    "moveit_execute_plan",
}


class RobotToolAdapterLike(Protocol):
    async def connect(self) -> None: ...

    async def disconnect(self) -> None: ...

    def function_tools(self) -> list[dict[str, Any]]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str: ...


class AgentTurnBackendLike(Protocol):
    def run_turn(self, turn: AgentTurnInput) -> AsyncIterator[str]: ...


@dataclass(frozen=True)
class RecordedToolCall:
    name: str
    arguments: dict[str, Any]
    output_text: str
    output_json: Any

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "output_text": self.output_text,
            "output_json": self.output_json,
        }


@dataclass(frozen=True)
class LiveSmokeRun:
    prompt: str
    reply: str
    tool_calls: list[RecordedToolCall]

    def as_json(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "reply": self.reply,
            "tool_calls": [call.as_json() for call in self.tool_calls],
        }


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "details": self.details,
        }


class RecordingRobotToolAdapter:
    """Records live robot tool calls while delegating to the real robot tool adapter."""

    def __init__(self, delegate: RobotToolAdapterLike) -> None:
        self._delegate = delegate
        self._calls: list[RecordedToolCall] = []

    @property
    def calls(self) -> list[RecordedToolCall]:
        return list(self._calls)

    def clear(self) -> None:
        self._calls.clear()

    async def connect(self) -> None:
        await self._delegate.connect()

    async def disconnect(self) -> None:
        await self._delegate.disconnect()

    def function_tools(self) -> list[dict[str, Any]]:
        return self._delegate.function_tools()

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        output_text = await self._delegate.call_tool(name, arguments)
        self._calls.append(
            RecordedToolCall(
                name=name,
                arguments=dict(arguments),
                output_text=output_text,
                output_json=_json_or_text(output_text),
            )
        )
        return output_text


async def run_agent_turn(
    backend: AgentTurnBackendLike,
    recorder: RecordingRobotToolAdapter,
    prompt: str,
) -> LiveSmokeRun:
    recorder.clear()
    turn = AgentTurnInput(user_text=prompt, messages=[{"role": "user", "content": prompt}])
    chunks = [chunk async for chunk in backend.run_turn(turn)]
    return LiveSmokeRun(prompt=prompt, reply="".join(chunks).strip(), tool_calls=recorder.calls)


def validate_position_query(run: LiveSmokeRun) -> ValidationResult:
    if not _has_successful_pose_observation(run.tool_calls):
        return ValidationResult(False, "position query did not observe a successful parseable current pose")
    unexpected_tools = _called_unexpected_no_action_tools(run.tool_calls)
    if unexpected_tools:
        return ValidationResult(
            False,
            f"position query used unexpected robot tools: {unexpected_tools}",
            {"robot_tools": unexpected_tools},
        )
    return ValidationResult(True, "position query observed current pose without movement")


def validate_bit_movement(
    run: LiveSmokeRun,
    *,
    direction: Literal["up", "down"],
) -> ValidationResult:
    if not _has_call(run.tool_calls, "moveit_get_current_pose"):
        return ValidationResult(False, f"move {direction} did not observe current pose")
    if not _has_verified_execution(run.tool_calls):
        return ValidationResult(False, f"move {direction} did not record verified execution")

    poses = [_pose_position(call) for call in run.tool_calls if call.name == "moveit_get_current_pose"]
    positions = [pose for pose in poses if pose is not None]
    if len(positions) < 2:
        return ValidationResult(False, f"move {direction} did not record start and final poses")

    start = positions[0]
    final = positions[-1]
    delta_x = final["x"] - start["x"]
    delta_y = final["y"] - start["y"]
    delta_z = final["z"] - start["z"]
    expected_sign = 1.0 if direction == "up" else -1.0
    signed_delta_z = expected_sign * delta_z
    details = {
        "start": start,
        "final": final,
        "delta_x_m": delta_x,
        "delta_y_m": delta_y,
        "delta_z_m": delta_z,
        "expected_delta_m": expected_sign * EXPECTED_BIT_DELTA_M,
    }

    lower = EXPECTED_BIT_DELTA_M - BIT_DELTA_TOLERANCE_M
    upper = EXPECTED_BIT_DELTA_M + BIT_DELTA_TOLERANCE_M
    if signed_delta_z < lower or signed_delta_z > upper:
        sign_label = "+Z" if direction == "up" else "-Z"
        return ValidationResult(
            False,
            f"expected {sign_label} movement around {EXPECTED_BIT_DELTA_M} m, got delta_z={delta_z:.4f} m",
            details,
        )
    if abs(delta_x) > XY_DRIFT_MAX_M or abs(delta_y) > XY_DRIFT_MAX_M:
        return ValidationResult(
            False,
            f"movement drifted too far in X/Y: delta_x={delta_x:.4f} m, delta_y={delta_y:.4f} m",
            details,
        )
    return ValidationResult(True, f"move {direction} executed verified bounded movement", details)


def validate_wave_motion(run: LiveSmokeRun) -> ValidationResult:
    start = next(
        (
            position
            for call in run.tool_calls
            if call.name == CURRENT_POSE_TOOL_NAME
            for position in [_pose_position(call)]
            if position is not None
        ),
        None,
    )
    if start is None:
        return ValidationResult(False, "wave did not observe a successful parseable current pose")

    cartesian_calls = [
        call for call in run.tool_calls if call.name == "moveit_plan_and_execute_cartesian_motion"
    ]
    if not cartesian_calls:
        return ValidationResult(False, "wave did not use moveit_plan_and_execute_cartesian_motion")

    execution_call = next((call for call in cartesian_calls if _is_verified_execution(call)), None)
    if execution_call is None:
        return ValidationResult(False, "wave did not record verified cartesian execution")

    raw_waypoints = execution_call.arguments.get("waypoints")
    if not isinstance(raw_waypoints, list):
        return ValidationResult(False, "wave cartesian execution did not include waypoints")
    if len(raw_waypoints) < 4:
        return ValidationResult(False, "expected at least 4 wave waypoints")

    waypoints = [_waypoint_position(waypoint) for waypoint in raw_waypoints]
    positions = [position for position in waypoints if position is not None]
    if len(positions) != len(raw_waypoints):
        return ValidationResult(False, "wave waypoints must include finite x/y/z positions")

    lateral_span_m = max(position["y"] for position in positions) - min(position["y"] for position in positions)
    vertical_lift_m = max(position["z"] - start["z"] for position in positions)
    details = {
        "start": start,
        "waypoint_count": len(positions),
        "lateral_span_m": lateral_span_m,
        "vertical_lift_m": vertical_lift_m,
        "expected_lateral_span_m": EXPECTED_WAVE_LATERAL_SPAN_M,
    }

    if lateral_span_m < MIN_WAVE_LATERAL_SPAN_M:
        return ValidationResult(
            False,
            f"expected at least {MIN_WAVE_LATERAL_SPAN_M} m lateral wave span, got {lateral_span_m:.4f} m",
            details,
        )
    if vertical_lift_m < MIN_WAVE_VERTICAL_LIFT_M:
        return ValidationResult(
            False,
            f"expected at least {MIN_WAVE_VERTICAL_LIFT_M} m vertical wave lift, got {vertical_lift_m:.4f} m",
            details,
        )
    return ValidationResult(True, "wave executed visible verified cartesian sweep", details)


def validate_ambiguous_clarification(run: LiveSmokeRun) -> ValidationResult:
    unexpected_tools = _called_unexpected_no_action_tools(run.tool_calls)
    if unexpected_tools:
        return ValidationResult(
            False,
            f"ambiguous command used unexpected robot tools: {unexpected_tools}",
            {"robot_tools": unexpected_tools},
        )
    reply = run.reply.lower()
    clarification_terms = ("?", "where", "which", "clarify", "location", "target")
    if not any(term in reply for term in clarification_terms):
        return ValidationResult(False, "ambiguous command did not ask a recognizable clarification")
    return ValidationResult(True, "ambiguous command asked for clarification without movement")


def write_evidence(
    *,
    evidence_dir: Path,
    case_name: str,
    run: LiveSmokeRun,
    validation: ValidationResult,
) -> Path:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = evidence_dir / f"{timestamp}-{_slug(case_name)}.json"
    payload = {
        "case": case_name,
        "run": run.as_json(),
        "validator": validation.as_json(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _has_call(calls: list[RecordedToolCall], name: str) -> bool:
    return any(call.name == name for call in calls)


def _called_unexpected_no_action_tools(calls: list[RecordedToolCall]) -> list[str]:
    return [call.name for call in calls if call.name.startswith("moveit_") and call.name != CURRENT_POSE_TOOL_NAME]


def _has_successful_pose_observation(calls: list[RecordedToolCall]) -> bool:
    return any(call.name == CURRENT_POSE_TOOL_NAME and _pose_position(call) is not None for call in calls)


def _has_verified_execution(calls: list[RecordedToolCall]) -> bool:
    return any(_is_verified_execution(call) for call in calls if call.name in EXECUTION_TOOL_NAMES)


def _is_verified_execution(call: RecordedToolCall) -> bool:
    structured = _structured_content(call.output_json)
    if not isinstance(structured, dict) or structured.get("ok") is not True:
        return False
    verification = structured.get("verification")
    if isinstance(verification, dict) and verification.get("result") == "pass":
        return True
    execution = structured.get("execution")
    return isinstance(execution, dict) and execution.get("verification_result") == "pass"


def _pose_position(call: RecordedToolCall) -> dict[str, float] | None:
    structured = _structured_content(call.output_json)
    if not isinstance(structured, dict) or structured.get("ok") is not True:
        return None
    raw = structured.get("raw")
    if not isinstance(raw, dict):
        return None
    pose = raw.get("pose")
    if not isinstance(pose, dict):
        return None
    return _waypoint_position(pose)


def _waypoint_position(waypoint: Any) -> dict[str, float] | None:
    if not isinstance(waypoint, dict):
        return None
    position = waypoint.get("position") if isinstance(waypoint.get("position"), dict) else waypoint
    if not isinstance(position, dict):
        return None
    x = _finite_float(position.get("x"))
    y = _finite_float(position.get("y"))
    z = _finite_float(position.get("z"))
    if x is None or y is None or z is None:
        return None
    return {"x": x, "y": y, "z": z}


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return number


def _structured_content(value: Any) -> Any:
    if not isinstance(value, dict):
        return None
    return value.get("structured_content")


def _json_or_text(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "case"
