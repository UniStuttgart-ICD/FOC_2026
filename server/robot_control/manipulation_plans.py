from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ManipulationFollowUpAction:
    tool: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ExecutableManipulationPlan:
    tool_name: str
    plan_name: str
    robot_name: str | None
    can_execute: bool
    raw: dict[str, Any]
    feedback: dict[str, Any]
    structured_content: dict[str, Any]
    after_success: ManipulationFollowUpAction | None = None


@dataclass(frozen=True)
class TaskSolutionResult:
    tool_name: str
    task_solution_id: str
    task_kind: str
    backend: str
    object_name: str
    robot_name: str | None
    scene_snapshot_id: str | None
    approval_required: bool
    raw: dict[str, Any]
    feedback: dict[str, Any]
    structured_content: dict[str, Any]


def parse_executable_plan_result(tool_name: str, output: str) -> ExecutableManipulationPlan | None:
    structured = _structured_content(output)
    if not isinstance(structured, dict) or structured.get("ok") is not True:
        return None

    feedback = structured.get("feedback")
    if not isinstance(feedback, dict) or feedback.get("can_execute") is not True:
        return None

    raw = structured.get("raw")
    if not isinstance(raw, dict):
        return None

    plan_name = raw.get("plan_name")
    if not isinstance(plan_name, str) or not plan_name:
        return None

    robot_name = structured.get("robot", structured.get("robot_name"))
    return ExecutableManipulationPlan(
        tool_name=tool_name,
        plan_name=plan_name,
        robot_name=robot_name if isinstance(robot_name, str) and robot_name else None,
        can_execute=True,
        raw=dict(raw),
        feedback=dict(feedback),
        structured_content=dict(structured),
        after_success=_after_success_action(raw),
    )


def parse_task_solution_result(tool_name: str, output: str) -> TaskSolutionResult | None:
    structured = _structured_content(output)
    if not isinstance(structured, dict) or structured.get("ok") is not True:
        return None

    feedback = structured.get("feedback")
    if not isinstance(feedback, dict) or feedback.get("can_execute") is not True:
        return None
    if feedback.get("execution_target") != "task_solution":
        return None

    raw = structured.get("raw")
    if not isinstance(raw, dict):
        return None

    created_from_tool = raw.get("created_from_tool")
    if isinstance(created_from_tool, str) and created_from_tool and created_from_tool != tool_name:
        return None

    task_solution_id = raw.get("task_solution_id")
    task_kind = raw.get("task_kind")
    backend = raw.get("backend")
    object_name = raw.get("object_name")
    if not isinstance(task_solution_id, str) or not task_solution_id.strip():
        return None
    if not isinstance(task_kind, str) or not task_kind.strip():
        return None
    if not isinstance(backend, str) or not backend.strip():
        return None
    if not isinstance(object_name, str) or not object_name.strip():
        return None

    robot_name = raw.get("robot_name", structured.get("robot", structured.get("robot_name")))
    scene_snapshot_id = raw.get("scene_snapshot_id")
    approval = raw.get("approval")
    approval_required = isinstance(approval, dict) and approval.get("required") is True
    return TaskSolutionResult(
        tool_name=tool_name,
        task_solution_id=task_solution_id,
        task_kind=task_kind,
        backend=backend,
        object_name=object_name,
        robot_name=robot_name if isinstance(robot_name, str) and robot_name else None,
        scene_snapshot_id=scene_snapshot_id if isinstance(scene_snapshot_id, str) and scene_snapshot_id else None,
        approval_required=approval_required,
        raw=dict(raw),
        feedback=dict(feedback),
        structured_content=dict(structured),
    )


def executable_plan_name_from_output(output: str) -> str | None:
    result = parse_executable_plan_result("", output)
    return result.plan_name if result is not None else None


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")


def _after_success_action(raw: dict[str, Any]) -> ManipulationFollowUpAction | None:
    next_action = raw.get("next_action")
    if not isinstance(next_action, dict):
        return None
    after_success = next_action.get("after_success")
    if not isinstance(after_success, dict):
        return None
    tool = after_success.get("tool", after_success.get("name"))
    arguments = after_success.get("arguments", after_success.get("args"))
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None
    return ManipulationFollowUpAction(tool=tool, arguments=dict(arguments))
