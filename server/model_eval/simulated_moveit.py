from __future__ import annotations

import copy
import json
from typing import Any

from robot_control.call_validation import agent_tool_description

ROBOT_NAME = "UR10"
SCENE_SNAPSHOT_ID = "scene_20260515_001"
TASK_SOLUTION_ID = "pick_task_dynamic_5_001"
TASK_LEVEL_PICK_TOOL_SEQUENCE = [
    "moveit_list_scene_objects",
    "moveit_get_object_context",
    "moveit_get_current_pose",
    "moveit_plan_pick_task",
    "approval_recorded",
    "moveit_execute_task_solution",
    "moveit_verify_attached_object",
]
INITIAL_POSE: dict[str, Any] = {
    "position": {"x": 0.4, "y": 0.1, "z": 0.3},
    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
}
SCENE_OBJECTS: list[dict[str, Any]] = [
    {
        "name": "dynamic_5",
        "type": "box",
        "scene_snapshot_id": SCENE_SNAPSHOT_ID,
        "planning_frame": "base_link",
        "pose": {
            "position": {"x": 0.52, "y": 0.04, "z": 0.12},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
]

_SUPPORTED_TOOLS = (
    "moveit_get_current_pose",
    "moveit_plan_cartesian_motion",
    "moveit_execute_plan",
    "moveit_list_available_robots",
)

_DESCRIPTIONS = {
    "moveit_list_available_robots": "List robots available in the simulated MoveIt scene.",
}


def task_level_pick_replay_scenario() -> dict[str, Any]:
    """Return a compact replay artifact for the task-level pick loop."""
    approval_payload = {
        "target_kind": "task_solution",
        "task_solution_id": TASK_SOLUTION_ID,
        "source_tool": "moveit_plan_pick_task",
        "object_name": "dynamic_5",
        "expected_movement": "approach grasp, close gripper, attach object, lift object",
        "scene_snapshot_id": SCENE_SNAPSHOT_ID,
        "approval_turn_id": "turn_001",
        "approved_at": "2026-05-15T17:45:00Z",
    }
    execution_result = {
        "ok": True,
        "task_solution_id": TASK_SOLUTION_ID,
        "executed_stages": [
            "observe_current_state",
            "connect_to_pre_grasp",
            "approach_grasp",
            "close_gripper",
            "attach_object",
            "lift_object",
            "verify_attached_object",
        ],
    }
    verification_result = {
        "ok": True,
        "object_name": "dynamic_5",
        "attached": True,
        "evidence": [{"kind": "attached_object", "object_name": "dynamic_5"}],
    }
    return {
        "utterance": "pick up dynamic_5",
        "mocked_robot_state": {
            "robot_name": ROBOT_NAME,
            "planning_frame": "base_link",
            "pose": copy.deepcopy(INITIAL_POSE),
        },
        "mocked_scene_objects": copy.deepcopy(SCENE_OBJECTS),
        "expected_tool_sequence": list(TASK_LEVEL_PICK_TOOL_SEQUENCE),
        "typed_tool_outputs": [
            {
                "tool_name": "moveit_list_scene_objects",
                "structured_content": {
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "raw": {
                        "scene_snapshot_id": SCENE_SNAPSHOT_ID,
                        "objects": copy.deepcopy(SCENE_OBJECTS),
                    },
                },
            },
            {
                "tool_name": "moveit_get_object_context",
                "structured_content": {
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "raw": {
                        "scene_snapshot_id": SCENE_SNAPSHOT_ID,
                        "object": copy.deepcopy(SCENE_OBJECTS[0]),
                    },
                },
            },
            {
                "tool_name": "moveit_get_current_pose",
                "structured_content": {
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "raw": {"pose": copy.deepcopy(INITIAL_POSE)},
                },
            },
            {
                "tool_name": "moveit_plan_pick_task",
                "structured_content": _task_solution_content(approval_payload),
            },
            {
                "tool_name": "moveit_execute_task_solution",
                "structured_content": {
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "raw": copy.deepcopy(execution_result),
                },
            },
            {
                "tool_name": "moveit_verify_attached_object",
                "structured_content": copy.deepcopy(verification_result),
            },
        ],
        "policy_decisions": [
            {"tool_name": "moveit_plan_pick_task", "decision": "allow"},
            {"tool_name": "moveit_execute_task_solution", "decision": "allow"},
        ],
        "validation_results": [
            {"tool_name": "moveit_plan_pick_task", "ok": True},
            {"tool_name": "moveit_execute_task_solution", "ok": True},
        ],
        "approval_payload": approval_payload,
        "execution_result": execution_result,
        "verification_result": verification_result,
        "terminal_job_event": {
            "event_type": "robot_job_completed",
            "tool_name": "moveit_verify_attached_object",
            "ok": True,
        },
    }


def negative_pick_replay_scenarios() -> dict[str, dict[str, Any]]:
    return {
        "partial_legacy_pick": {
            "utterance": "pick up dynamic_5",
            "tool_output": {
                "structured_content": {
                    "ok": False,
                    "error": "pick_segment_planning_failed",
                    "failed_stage": "local_cartesian_pick",
                    "failed_segment": "local_cartesian_pick",
                    "retryable": True,
                    "suggested_next_tool": "moveit_explain_motion_failure",
                    "feedback": {"can_execute": False},
                    "raw": {
                        "stage_report": [
                            {"name": "connect_to_pre_grasp", "stage_type": "motion_plan", "status": "solved"},
                            {"name": "local_cartesian_pick", "stage_type": "motion_plan", "status": "failed"},
                        ],
                        "candidate_attempts": 12,
                        "blocker": "local cartesian approach failed after preposition",
                        "scene_snapshot_id": SCENE_SNAPSHOT_ID,
                        "partial_plan": {
                            "kind": "preposition",
                            "plan_name": "pick_dynamic_5_preposition",
                        },
                    },
                },
            },
            "execution_attempted": False,
        },
        "missing_approval": {
            "tool_name": "moveit_execute_task_solution",
            "arguments": {"robot_name": ROBOT_NAME, "task_solution_id": TASK_SOLUTION_ID},
            "policy_decision": {
                "tool_name": "moveit_execute_task_solution",
                "decision": "block",
                "reason": "missing_approval",
            },
            "execution_attempted": False,
        },
        "stale_scene_snapshot_id": {
            "tool_name": "moveit_execute_task_solution",
            "arguments": {"robot_name": ROBOT_NAME, "task_solution_id": TASK_SOLUTION_ID},
            "approval_payload": {
                "task_solution_id": TASK_SOLUTION_ID,
                "scene_snapshot_id": "scene_20260515_stale",
            },
            "current_scene_snapshot_id": SCENE_SNAPSHOT_ID,
            "policy_decision": {
                "tool_name": "moveit_execute_task_solution",
                "decision": "block",
                "reason": "stale_scene_snapshot_id",
            },
            "execution_attempted": False,
        },
        "attachment_verification_failure": {
            "execution_result": {
                "ok": True,
                "task_solution_id": TASK_SOLUTION_ID,
            },
            "verification_result": {
                "ok": False,
                "object_name": "dynamic_5",
                "attached": False,
                "error": "object_not_attached",
            },
            "success_claim_allowed": False,
            "terminal_job_event": {
                "event_type": "robot_job_failed",
                "tool_name": "moveit_verify_attached_object",
                "ok": False,
            },
        },
    }


def _task_solution_content(approval_payload: dict[str, Any]) -> dict[str, Any]:
    stages = [
        {"name": "observe_current_state", "stage_type": "observation", "status": "solved"},
        {"name": "connect_to_pre_grasp", "stage_type": "motion_plan", "status": "solved"},
        {"name": "approach_grasp", "stage_type": "motion_plan", "status": "solved"},
        {"name": "close_gripper", "stage_type": "gripper", "status": "solved"},
        {"name": "attach_object", "stage_type": "planning_scene", "status": "solved"},
        {"name": "lift_object", "stage_type": "motion_plan", "status": "solved"},
        {"name": "verify_attached_object", "stage_type": "verification", "status": "solved"},
    ]
    return {
        "ok": True,
        "robot": ROBOT_NAME,
        "feedback": {
            "can_execute": True,
            "execution_target": "task_solution",
        },
        "raw": {
            "task_solution_id": TASK_SOLUTION_ID,
            "task_kind": "pick",
            "backend": "emulated",
            "object_name": "dynamic_5",
            "robot_name": ROBOT_NAME,
            "created_from_tool": "moveit_plan_pick_task",
            "scene_snapshot_id": SCENE_SNAPSHOT_ID,
            "planning_frame": "base_link",
            "object_pose_age_s": 0.24,
            "solver": "emulated_mtc_stages",
            "selected_cost": 1.42,
            "clearance_m": 0.018,
            "stages": stages,
            "stage_report": stages,
            "candidate_attempts": 1,
            "approval": {
                "required": True,
                **copy.deepcopy(approval_payload),
            },
            "evidence": [
                {"kind": "scene_snapshot", "id": SCENE_SNAPSHOT_ID},
                {"kind": "stage_report", "count": len(stages)},
            ],
        },
    }


class SimulatedMoveItAdapter:
    """Deterministic offline Robot Tool Adapter for model evaluation."""

    def __init__(self) -> None:
        self._pose = copy.deepcopy(INITIAL_POSE)
        self._plans: dict[str, dict[str, Any]] = {}
        self._connected = False

    async def connect(self) -> None:
        self._connected = True

    async def disconnect(self) -> None:
        self._connected = False

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": name,
                "description": _tool_description(name),
                "parameters": _tool_parameters(name),
                "strict": None,
            }
            for name in _SUPPORTED_TOOLS
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_get_current_pose":
            return _tool_output(
                content=[f"{ROBOT_NAME} pose is x={self._pose['position']['x']}, y={self._pose['position']['y']}, z={self._pose['position']['z']}."],
                structured_content={
                    "ok": True,
                    "robot_name": ROBOT_NAME,
                    "planning_frame": "base_link",
                    "raw": {"pose": copy.deepcopy(self._pose)},
                },
            )
        if name == "moveit_plan_cartesian_motion":
            return self._plan_cartesian(arguments)
        if name == "moveit_execute_plan":
            return self._execute_plan(arguments)
        if name == "moveit_list_available_robots":
            return _tool_output(
                content=[f"Available robots: {ROBOT_NAME}."],
                structured_content={
                    "ok": True,
                    "robots": [{"name": ROBOT_NAME, "planning_frame": "base_link"}],
                    "raw": {"robots": [ROBOT_NAME]},
                },
            )
        return _error_output(f"Tool is not supported by simulated MoveIt adapter: {name}")

    def _plan_cartesian(self, arguments: dict[str, Any]) -> str:
        waypoints = arguments.get("waypoints", arguments.get("positions", arguments.get("points")))
        if not isinstance(waypoints, list) or not waypoints:
            return _error_output("Expected one or more Cartesian waypoints.")
        final_pose = _pose_from_value(waypoints[-1], fallback_orientation=self._pose["orientation"])
        if final_pose is None:
            return _error_output("Expected final waypoint with finite x/y/z position.")
        plan_name = str(arguments.get("plan_name") or "simulated_cartesian_plan")
        self._plans[plan_name] = {
            "pose": final_pose,
            "waypoint_count": len(waypoints),
        }
        return _plan_output(
            content=["Cartesian motion planned."],
            raw={
                "plan_name": plan_name,
                "waypoint_count": len(waypoints),
                "pose": copy.deepcopy(final_pose),
            },
        )

    def _execute_plan(self, arguments: dict[str, Any]) -> str:
        plan_name = str(arguments.get("plan_name") or "")
        plan = self._plans.get(plan_name)
        if plan is None:
            return _error_output(f"Unknown simulated plan: {plan_name}")
        self._pose = copy.deepcopy(plan["pose"])
        return _motion_output(
            content=[f"Plan {plan_name} executed and verified."],
            raw={
                "plan_name": plan_name,
                "waypoint_count": plan["waypoint_count"],
                "pose": copy.deepcopy(self._pose),
            },
        )


def _tool_description(name: str) -> str:
    if name in _DESCRIPTIONS:
        return _DESCRIPTIONS[name]
    return agent_tool_description(name)


def _tool_parameters(name: str) -> dict[str, Any]:
    if name == "moveit_get_current_pose":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "timeout_s": {"type": "number"},
            },
        }
    if name == "moveit_plan_cartesian_motion":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "waypoints": {"type": "array", "items": {"type": "object"}},
                "plan_name": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["waypoints"],
        }
    if name == "moveit_execute_plan":
        return {
            "type": "object",
            "properties": {
                "robot_name": {"type": "string", "const": ROBOT_NAME},
                "plan_name": {"type": "string"},
                "timeout_s": {"type": "number"},
            },
            "required": ["plan_name"],
        }
    return {"type": "object", "properties": {}}


def _pose_from_value(value: Any, *, fallback_orientation: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    position = value.get("position") if isinstance(value.get("position"), dict) else value
    if not isinstance(position, dict):
        return None
    coordinates: dict[str, float] = {}
    for axis in ("x", "y", "z"):
        coordinate = position.get(axis)
        if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
            return None
        coordinates[axis] = float(coordinate)
    orientation = value.get("orientation")
    if not isinstance(orientation, dict):
        orientation = fallback_orientation
    return {"position": coordinates, "orientation": copy.deepcopy(orientation)}


def _motion_output(*, content: list[str], raw: dict[str, Any]) -> str:
    return _tool_output(
        content=content,
        structured_content={
            "ok": True,
            "feedback": {"can_execute": True},
            "verification": {"result": "pass"},
            "execution": {"verification_result": "pass"},
            "raw": raw,
        },
    )


def _plan_output(*, content: list[str], raw: dict[str, Any]) -> str:
    return _tool_output(
        content=content,
        structured_content={
            "ok": True,
            "feedback": {"can_execute": True},
            "plan": {
                "ok": True,
                "plan_name": raw["plan_name"],
                "can_execute": True,
            },
            "raw": raw,
        },
    )


def _error_output(error: str) -> str:
    return _tool_output(
        content=[error],
        structured_content={"ok": False, "error": error, "retryable": False},
        is_error=True,
    )


def _tool_output(
    *,
    content: list[str],
    structured_content: dict[str, Any],
    is_error: bool = False,
) -> str:
    return json.dumps(
        {
            "content": content,
            "structured_content": structured_content,
            "is_error": is_error,
        },
        ensure_ascii=False,
    )
