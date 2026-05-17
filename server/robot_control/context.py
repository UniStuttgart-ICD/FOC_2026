from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from robot_control.manipulation_plans import (
    parse_executable_plan_result,
    parse_task_solution_result,
)

TASK_SOLUTION_APPROVAL_MAX_AGE_S = 60.0


@dataclass
class PendingExecutablePlan:
    plan_name: str
    robot_name: str | None
    source_tool: str | None
    observed_at_s: float
    after_success_tool: str | None = None
    after_success_arguments: dict[str, Any] | None = None
    execute_via_mcp: bool = False

    @property
    def after_success(self) -> dict[str, Any] | None:
        if self.after_success_tool is None or self.after_success_arguments is None:
            return None
        return {
            "tool": self.after_success_tool,
            "arguments": dict(self.after_success_arguments),
        }


@dataclass
class RecentTaskSolution:
    task_solution_id: str
    task_kind: str
    object_name: str
    backend: str
    scene_snapshot_id: str | None
    approval_required: bool
    raw: dict[str, Any] | None = None


@dataclass
class RecentTaskFailure:
    task_solution_id: str
    task_kind: str
    object_name: str
    scene_snapshot_id: str | None
    failed_step: str
    failed_stage: str
    failed_tool_name: str | None
    failed_tool_arguments: dict[str, Any] | None
    failed_tool_result: Any
    completed_steps: list[dict[str, Any]]
    verified_plan_names: list[str]
    gripper_state: str | None
    attached_object_verified: bool
    released_object_verified: bool
    observed_at_s: float


@dataclass
class PendingTaskSolutionApproval:
    target_kind: str
    task_solution_id: str
    source_tool: str
    object_name: str
    expected_movement: str | None
    scene_snapshot_id: str | None
    approval_turn_id: str | None = None
    approved_at: float | None = None


@dataclass(frozen=True)
class TaskSolutionApprovalStatus:
    ok: bool
    reason: str | None = None


@dataclass
class RobotContextSnapshot:
    observed_at_s: float | None = None
    robot_name: str | None = None
    tcp_pose: dict[str, Any] | None = None
    gripper_state: str | None = None
    gripper_observed_at_s: float | None = None
    held_object_name: str | None = None
    last_execution_result: str | None = None
    executable_plan_observed_at_s: dict[str, float] = field(default_factory=dict)
    pending_executable_plans: dict[str, PendingExecutablePlan] = field(default_factory=dict)
    recent_task_solution: RecentTaskSolution | None = None
    recent_task_failure: RecentTaskFailure | None = None
    pending_task_solution_approval: PendingTaskSolutionApproval | None = None
    user_intent_revision: int = 0
    approval_intent_revision: int | None = None


class RobotContextStore:
    def __init__(self, *, time_fn: Callable[[], float] = time.monotonic) -> None:
        self._snapshot = RobotContextSnapshot()
        self._time_fn = time_fn

    @property
    def pending_plan(self) -> PendingExecutablePlan | None:
        return self.latest_pending_executable_plan(max_age_s=120.0)

    @property
    def recent_task_solution(self) -> RecentTaskSolution | None:
        return self._snapshot.recent_task_solution

    @property
    def recent_task_failure(self) -> RecentTaskFailure | None:
        return self._snapshot.recent_task_failure

    @property
    def pending_task_solution_approval(self) -> PendingTaskSolutionApproval | None:
        return self._snapshot.pending_task_solution_approval

    def has_recent_robot_observation(self, *, max_age_s: float) -> bool:
        observed_at_s = self._snapshot.observed_at_s
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s

    def held_object_name(self) -> str | None:
        return self._snapshot.held_object_name

    def remember_executable_plan(
        self,
        plan_name: str,
        *,
        robot_name: str | None = None,
        source_tool: str | None = None,
        after_success_tool: str | None = None,
        after_success_arguments: dict[str, Any] | None = None,
        execute_via_mcp: bool = False,
    ) -> None:
        if plan_name:
            observed_at_s = self._time_fn()
            self._snapshot.executable_plan_observed_at_s[plan_name] = observed_at_s
            self._snapshot.pending_executable_plans[plan_name] = PendingExecutablePlan(
                plan_name=plan_name,
                robot_name=robot_name,
                source_tool=source_tool,
                observed_at_s=observed_at_s,
                after_success_tool=after_success_tool,
                after_success_arguments=after_success_arguments,
                execute_via_mcp=execute_via_mcp,
            )

    def has_recent_executable_plan(self, plan_name: str, *, max_age_s: float) -> bool:
        return self.pending_executable_plan(plan_name, max_age_s=max_age_s) is not None

    def pending_executable_plan(
        self,
        plan_name: str,
        *,
        max_age_s: float,
    ) -> PendingExecutablePlan | None:
        pending = self._snapshot.pending_executable_plans.get(plan_name)
        if pending is None:
            return None
        if self._time_fn() - pending.observed_at_s > max_age_s:
            return None
        return pending

    def latest_pending_executable_plan(
        self,
        *,
        max_age_s: float,
    ) -> PendingExecutablePlan | None:
        pending = self._recent_pending_plans(max_age_s=max_age_s)
        if not pending:
            return None
        return max(pending, key=lambda plan: plan.observed_at_s)

    def consume_executable_plan(self, plan_name: str) -> bool:
        removed_pending = self._snapshot.pending_executable_plans.pop(plan_name, None)
        self._snapshot.executable_plan_observed_at_s.pop(plan_name, None)
        return removed_pending is not None

    def remember_task_solution(
        self,
        *,
        task_solution_id: str,
        task_kind: str,
        object_name: str,
        backend: str,
        scene_snapshot_id: str | None,
        approval_required: bool,
        raw: dict[str, Any] | None = None,
    ) -> None:
        self._snapshot.recent_task_solution = RecentTaskSolution(
            task_solution_id=task_solution_id,
            task_kind=task_kind,
            object_name=object_name,
            backend=backend,
            scene_snapshot_id=scene_snapshot_id,
            approval_required=approval_required,
            raw=dict(raw) if isinstance(raw, dict) else None,
        )

    def remember_task_failure(
        self,
        *,
        task_solution_id: str,
        task_kind: str,
        object_name: str,
        scene_snapshot_id: str | None,
        failed_step: str,
        failed_stage: str,
        failed_tool_name: str | None,
        failed_tool_arguments: dict[str, Any] | None,
        failed_tool_result: Any,
        completed_steps: list[dict[str, Any]],
        verified_plan_names: list[str],
        gripper_state: str | None,
        attached_object_verified: bool,
        released_object_verified: bool,
    ) -> None:
        self._snapshot.recent_task_failure = RecentTaskFailure(
            task_solution_id=task_solution_id,
            task_kind=task_kind,
            object_name=object_name,
            scene_snapshot_id=scene_snapshot_id,
            failed_step=failed_step,
            failed_stage=failed_stage,
            failed_tool_name=failed_tool_name,
            failed_tool_arguments=(
                dict(failed_tool_arguments) if isinstance(failed_tool_arguments, dict) else None
            ),
            failed_tool_result=failed_tool_result,
            completed_steps=[dict(step) for step in completed_steps],
            verified_plan_names=list(verified_plan_names),
            gripper_state=gripper_state,
            attached_object_verified=attached_object_verified,
            released_object_verified=released_object_verified,
            observed_at_s=self._time_fn(),
        )

    def remember_task_solution_approval_candidate(
        self,
        *,
        target_kind: str,
        task_solution_id: str,
        source_tool: str,
        object_name: str,
        expected_movement: str | None,
        scene_snapshot_id: str | None,
    ) -> None:
        self._snapshot.pending_task_solution_approval = PendingTaskSolutionApproval(
            target_kind=target_kind,
            task_solution_id=task_solution_id,
            source_tool=source_tool,
            object_name=object_name,
            expected_movement=expected_movement,
            scene_snapshot_id=scene_snapshot_id,
        )
        self._snapshot.approval_intent_revision = None

    def record_task_solution_approval(
        self,
        task_solution_id: str,
        *,
        approval_turn_id: str,
        approved_at: float | None = None,
    ) -> bool:
        approval = self._snapshot.pending_task_solution_approval
        if approval is None or approval.task_solution_id != task_solution_id:
            return False
        now_s = self._time_fn()
        approval_time = now_s
        if approved_at is not None and math.isfinite(approved_at) and approved_at <= now_s:
            approval_time = approved_at
        approval.approval_turn_id = approval_turn_id
        approval.approved_at = approval_time
        self._snapshot.approval_intent_revision = self._snapshot.user_intent_revision
        return True

    def mark_new_user_intent(self) -> None:
        self._snapshot.user_intent_revision += 1

    def task_solution_execution_approval_status(
        self,
        task_solution_id: str,
        *,
        scene_snapshot_id: str | None = None,
    ) -> TaskSolutionApprovalStatus:
        approval = self._snapshot.pending_task_solution_approval
        if approval is None or approval.approval_turn_id is None or approval.approved_at is None:
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="approval_missing",
            )
        if approval.task_solution_id != task_solution_id:
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="approval_for_different_task_solution",
            )
        if self._time_fn() - approval.approved_at > TASK_SOLUTION_APPROVAL_MAX_AGE_S:
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="approval_expired",
            )
        recent_task_solution = self._snapshot.recent_task_solution
        if recent_task_solution is not None and recent_task_solution.task_solution_id != task_solution_id:
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="approval_for_different_task_solution",
            )
        current_scene_snapshot_id = (
            recent_task_solution.scene_snapshot_id
            if recent_task_solution is not None
            else scene_snapshot_id
        )
        if (
            current_scene_snapshot_id is not None
            and approval.scene_snapshot_id is not None
            and current_scene_snapshot_id != approval.scene_snapshot_id
        ):
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="scene_snapshot_changed",
            )
        if (
            self._snapshot.approval_intent_revision is not None
            and self._snapshot.approval_intent_revision < self._snapshot.user_intent_revision
        ):
            return TaskSolutionApprovalStatus(
                ok=False,
                reason="approval_stale_after_new_user_intent",
            )
        return TaskSolutionApprovalStatus(ok=True)

    def gripper_state(self) -> str | None:
        return self._snapshot.gripper_state

    def has_recent_gripper_state(self, state: str, *, max_age_s: float) -> bool:
        if self._snapshot.gripper_state != state:
            return False
        observed_at_s = self._snapshot.gripper_observed_at_s
        if observed_at_s is None:
            return False
        return self._time_fn() - observed_at_s <= max_age_s

    def render_instruction_block(self) -> str:
        age = self._status_age_text()
        lines = [
            "Last-known robot context:",
            "- This context is advisory only.",
            "- For movement, relative commands, retries, or state-dependent actions, call moveit_get_current_pose first.",
            f"- status age: {age}",
        ]
        pending_lines = self._pending_plan_instruction_lines(max_age_s=120.0)
        if self._snapshot.robot_name is None:
            lines.append("- No robot status has been observed yet.")
            if self._snapshot.held_object_name:
                lines.append(f"- held object: {self._snapshot.held_object_name}")
            lines.extend(pending_lines)
            lines.extend(self._recent_task_failure_instruction_lines())
            return "\n".join(lines)

        lines.append(f"- robot: {self._snapshot.robot_name}")
        pose_text = self._tcp_pose_text()
        if pose_text:
            lines.append(f"- tcp pose: {pose_text}")
        if self._snapshot.gripper_state:
            lines.append(f"- gripper: {self._snapshot.gripper_state}")
        if self._snapshot.held_object_name:
            lines.append(f"- held object: {self._snapshot.held_object_name}")
        if self._snapshot.last_execution_result:
            lines.append(f"- last execution: {self._snapshot.last_execution_result}")
        lines.extend(pending_lines)
        lines.extend(self._recent_task_failure_instruction_lines())
        return "\n".join(lines)

    def latest_tcp_pose(self) -> dict[str, Any] | None:
        if self._snapshot.tcp_pose is None:
            return None
        return dict(self._snapshot.tcp_pose)

    def update_from_tool_result(self, tool_name: str, output: str) -> None:
        structured_content = _structured_content(output)
        if not isinstance(structured_content, dict) or structured_content.get("ok") is not True:
            return

        if tool_name == "moveit_close_gripper":
            self._snapshot.gripper_state = "closed"
            self._snapshot.gripper_observed_at_s = self._time_fn()
            return
        if tool_name == "moveit_open_gripper":
            self._snapshot.gripper_state = "open"
            self._snapshot.gripper_observed_at_s = self._time_fn()
            return
        if tool_name in {
            "moveit_attach_object",
            "moveit_verify_attached_object",
            "moveit_release_object",
            "moveit_verify_released_object",
        }:
            released_object = _released_object_name(structured_content)
            if released_object is not None and (
                self._snapshot.held_object_name is None
                or self._snapshot.held_object_name == released_object
            ):
                self._snapshot.held_object_name = None
                return
            held_object = _held_object_name(structured_content)
            if held_object is not None:
                self._snapshot.held_object_name = held_object
            return
        plan = parse_executable_plan_result(tool_name, output)
        task_solution = parse_task_solution_result(tool_name, output)
        if (
            tool_name
            in {"moveit_plan_pick_task", "moveit_plan_place_task", "moveit_plan_compound_task"}
            and task_solution is not None
        ):
            self.remember_task_solution(
                task_solution_id=task_solution.task_solution_id,
                task_kind=task_solution.task_kind,
                object_name=task_solution.object_name,
                backend=task_solution.backend,
                scene_snapshot_id=task_solution.scene_snapshot_id,
                approval_required=task_solution.approval_required,
                raw=task_solution.raw,
            )
            approval = task_solution.raw.get("approval")
            if isinstance(approval, dict):
                target_kind = approval.get("target_kind")
                approval_task_solution_id = approval.get("task_solution_id")
                source_tool = approval.get("source_tool")
                object_name = approval.get("object_name")
                expected_movement = approval.get("expected_movement")
                scene_snapshot_id = approval.get("scene_snapshot_id")
                if (
                    isinstance(target_kind, str)
                    and isinstance(approval_task_solution_id, str)
                    and isinstance(source_tool, str)
                    and isinstance(object_name, str)
                ):
                    self.remember_task_solution_approval_candidate(
                        target_kind=target_kind,
                        task_solution_id=approval_task_solution_id,
                        source_tool=source_tool,
                        object_name=object_name,
                        expected_movement=(
                            expected_movement if isinstance(expected_movement, str) else None
                        ),
                        scene_snapshot_id=(
                            scene_snapshot_id if isinstance(scene_snapshot_id, str) else None
                        ),
                    )
            return
        if tool_name in {
            "moveit_plan_free_motion",
            "moveit_plan_cartesian_motion",
            "moveit_plan_pick",
            "moveit_plan_place",
        } and plan is not None:
            self.remember_executable_plan(
                plan.plan_name,
                robot_name=plan.robot_name,
                source_tool=plan.tool_name,
                after_success_tool=plan.after_success.tool if plan.after_success is not None else None,
                after_success_arguments=(
                    plan.after_success.arguments if plan.after_success is not None else None
                ),
                execute_via_mcp=_requires_mcp_execution(plan.raw),
            )
            return
        if tool_name == "moveit_execute_plan":
            verification = structured_content.get("verification")
            if isinstance(verification, dict) and verification.get("result") == "pass":
                raw = structured_content.get("raw")
                feedback = structured_content.get("feedback")
                source = raw if isinstance(raw, dict) else feedback if isinstance(feedback, dict) else {}
                consumed_plan = source.get("plan_name") if isinstance(source, dict) else None
                if isinstance(consumed_plan, str):
                    self.consume_executable_plan(consumed_plan)
                self._snapshot.last_execution_result = "pass"
            return
        if tool_name not in {"moveit_get_current_pose", "moveit_get_robot_state"}:
            return

        self._snapshot.observed_at_s = self._time_fn()
        robot_name = structured_content.get("robot_name", structured_content.get("robot"))
        if isinstance(robot_name, str):
            self._snapshot.robot_name = robot_name
        tcp_pose = structured_content.get("tcp_pose")
        raw = structured_content.get("raw")
        if not isinstance(tcp_pose, dict) and isinstance(raw, dict):
            tcp_pose = raw.get("pose")
        if isinstance(tcp_pose, dict):
            self._snapshot.tcp_pose = tcp_pose
        gripper = structured_content.get("gripper")
        if isinstance(gripper, dict) and isinstance(gripper.get("state"), str):
            self._snapshot.gripper_state = gripper["state"]
            self._snapshot.gripper_observed_at_s = self._snapshot.observed_at_s
        last_execution = structured_content.get("last_execution")
        if isinstance(last_execution, dict) and isinstance(last_execution.get("result"), str):
            self._snapshot.last_execution_result = last_execution["result"]

    def _status_age_text(self) -> str:
        if self._snapshot.observed_at_s is None:
            return "unknown"
        return f"{self._time_fn() - self._snapshot.observed_at_s:.1f}s"

    def _tcp_pose_text(self) -> str | None:
        pose = self._snapshot.tcp_pose
        if not isinstance(pose, dict):
            return None
        position = pose.get("position")
        if not isinstance(position, dict):
            return None
        try:
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        except (KeyError, TypeError, ValueError):
            return None
        return f"x={x:.3f}, y={y:.3f}, z={z:.3f}"

    def _recent_pending_plans(self, *, max_age_s: float) -> list[PendingExecutablePlan]:
        return [
            pending
            for pending in self._snapshot.pending_executable_plans.values()
            if self._time_fn() - pending.observed_at_s <= max_age_s
        ]

    def _pending_plan_instruction_lines(self, *, max_age_s: float) -> list[str]:
        lines: list[str] = []
        for pending in self._recent_pending_plans(max_age_s=max_age_s):
            robot = f" for {pending.robot_name}" if pending.robot_name else ""
            age_s = self._time_fn() - pending.observed_at_s
            lines.append(
                f"- pending executable plan: {pending.plan_name}{robot} ({age_s:.1f}s old); "
                "execute only after explicit user request."
            )
        return lines

    def _recent_task_failure_instruction_lines(self) -> list[str]:
        failure = self._snapshot.recent_task_failure
        if failure is None:
            return []
        lines = [
            f"- recent task failure: {failure.task_solution_id}",
            f"- object: {failure.object_name}",
            f"- failed step: {failure.failed_step}",
            f"- failed stage: {failure.failed_stage}",
        ]
        if failure.failed_tool_name:
            lines.append(f"- failed tool: {failure.failed_tool_name}")
        if failure.completed_steps:
            completed = ", ".join(
                str(step.get("name") or step.get("handler") or "step")
                for step in failure.completed_steps
            )
            lines.append(f"- completed steps: {completed}")
        if failure.verified_plan_names:
            lines.append(f"- verified plans: {', '.join(failure.verified_plan_names)}")
        if failure.gripper_state:
            lines.append(f"- gripper at failure: {failure.gripper_state}")
        lines.append("- recovery requires explicit user/operator intent.")
        return lines


def _structured_content(output: str) -> Any:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload.get("structured_content")


def _held_object_name(structured_content: dict[str, Any]) -> str | None:
    raw = structured_content.get("raw")
    if not isinstance(raw, dict):
        return None
    holds_object = raw.get("mcp_gripper_holds_object")
    planning_state = raw.get("planning_scene_state")
    if holds_object is False or planning_state == "free":
        return None
    for key in ("attached_object", "mcp_attached_object"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if holds_object is not True and planning_state != "attached":
        return None
    value = raw.get("object_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _released_object_name(structured_content: dict[str, Any]) -> str | None:
    verification = structured_content.get("verification")
    if isinstance(verification, dict) and verification.get("result") not in {None, "pass"}:
        return None
    raw = structured_content.get("raw")
    if not isinstance(raw, dict):
        return None
    holds_object = raw.get("mcp_gripper_holds_object")
    planning_state = str(raw.get("planning_scene_state") or "").strip().lower()
    attached_object = raw.get("mcp_attached_object", raw.get("attached_object"))
    if isinstance(attached_object, str) and attached_object.strip():
        return None
    if planning_state == "attached":
        return None
    released_scene_states = {"free", "released", "detached", "world", "not_attached"}
    if holds_object is not False and planning_state not in released_scene_states:
        return None
    value = raw.get("object_name")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _requires_mcp_execution(raw: dict[str, Any]) -> bool:
    workflow_kind = raw.get("workflow_kind")
    if workflow_kind in {"pick", "place"}:
        return True
    next_action = raw.get("next_action")
    if isinstance(next_action, dict) and isinstance(next_action.get("after_success"), dict):
        return True
    return False
