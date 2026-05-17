from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

VerificationResult = Literal["pass", "fail", "unknown"]

SUCCESS_STATUSES = {"success", "success! "}


@dataclass(frozen=True)
class Evidence:
    kind: str
    summary: str
    topic: str | None = None
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "summary": self.summary}
        if self.topic is not None:
            data["topic"] = self.topic
        if self.path is not None:
            data["path"] = self.path
        return data


@dataclass(frozen=True)
class VerificationCheck:
    name: str
    passed: bool
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "details": self.details}


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    robot: str
    tool: str
    phase: str
    status: str
    message: str
    correction: str | None = None
    verification_result: VerificationResult = "unknown"
    checks: list[VerificationCheck] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    can_execute: bool = False

    @classmethod
    def pass_result(
        cls,
        *,
        robot: str,
        tool: str,
        phase: str,
        status: str,
        message: str,
        checks: list[VerificationCheck],
        evidence: list[Evidence],
        raw: dict[str, Any],
        can_execute: bool | None = None,
    ) -> "ToolResult":
        return cls(
            ok=True,
            robot=robot,
            tool=tool,
            phase=phase,
            status=status,
            message=message,
            correction=None,
            verification_result="pass",
            checks=checks,
            evidence=evidence,
            raw=raw,
            can_execute=(status in SUCCESS_STATUSES if can_execute is None else can_execute),
        )

    @classmethod
    def fail_result(
        cls,
        *,
        robot: str,
        tool: str,
        phase: str,
        status: str,
        message: str,
        checks: list[VerificationCheck],
        evidence: list[Evidence],
        raw: dict[str, Any],
        correction: str,
        verification_result: VerificationResult = "fail",
    ) -> "ToolResult":
        return cls(
            ok=False,
            robot=robot,
            tool=tool,
            phase=phase,
            status=status,
            message=message,
            correction=correction,
            verification_result=verification_result,
            checks=checks,
            evidence=evidence,
            raw=raw,
            can_execute=False,
        )

    def to_dict(self) -> dict[str, Any]:
        feedback: dict[str, Any] = {
            "phase": self.phase,
            "status": self.status,
            "message": self.message,
            "can_execute": self.can_execute,
        }
        if self.correction is not None:
            feedback["correction"] = self.correction

        return {
            "ok": self.ok,
            "robot": self.robot,
            "tool": self.tool,
            "feedback": feedback,
            "verification": {
                "result": self.verification_result,
                "checks": [check.to_dict() for check in self.checks],
            },
            "evidence": [item.to_dict() for item in self.evidence],
            "raw": self.raw,
        }


@dataclass(frozen=True)
class TaskStage:
    name: str
    stage_type: str
    status: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "stage_type": self.stage_type,
            "status": self.status,
            "evidence": self.evidence,
        }
        if self.raw:
            data["raw"] = self.raw
        return data


@dataclass(frozen=True)
class ExecutionApproval:
    required: bool
    target_kind: str
    task_solution_id: str
    source_tool: str
    object_name: str
    expected_movement: str
    scene_snapshot_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "target_kind": self.target_kind,
            "task_solution_id": self.task_solution_id,
            "source_tool": self.source_tool,
            "object_name": self.object_name,
            "expected_movement": self.expected_movement,
            "scene_snapshot_id": self.scene_snapshot_id,
        }


@dataclass(frozen=True)
class TaskSolution:
    task_solution_id: str
    task_kind: str
    backend: str
    stages: list[TaskStage]
    created_from_tool: str
    object_name: str
    robot_name: str
    scene_snapshot_id: str
    stage_report: dict[str, Any]
    approval: ExecutionApproval
    evidence: list[dict[str, Any]]
    planning_frame: str | None = None
    object_pose_age_s: float | None = None
    solver: str | None = None
    selected_cost: float | None = None
    clearance_m: float | None = None
    candidate_attempts: int | list[dict[str, Any]] | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_solution_id": self.task_solution_id,
            "task_kind": self.task_kind,
            "backend": self.backend,
            "object_name": self.object_name,
            "robot_name": self.robot_name,
            "created_from_tool": self.created_from_tool,
            "scene_snapshot_id": self.scene_snapshot_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_report": self.stage_report,
            "approval": self.approval.to_dict(),
            "evidence": self.evidence,
        }
        for key, value in {
            "planning_frame": self.planning_frame,
            "object_pose_age_s": self.object_pose_age_s,
            "solver": self.solver,
            "selected_cost": self.selected_cost,
            "clearance_m": self.clearance_m,
            "candidate_attempts": self.candidate_attempts,
        }.items():
            if value is not None:
                data[key] = value
        data.update(self.raw)
        return data


@dataclass(frozen=True)
class TaskSolutionResult:
    ok: bool
    task_solution: TaskSolution

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "raw": self.task_solution.to_dict()}


@dataclass(frozen=True)
class TaskExecutionResult:
    ok: bool
    task_solution_id: str
    task_kind: str
    backend: str
    stages: list[TaskStage]
    created_from_tool: str
    object_name: str
    robot_name: str
    scene_snapshot_id: str
    stage_report: dict[str, Any]
    approval: ExecutionApproval
    evidence: list[dict[str, Any]]
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "task_solution_id": self.task_solution_id,
            "task_kind": self.task_kind,
            "backend": self.backend,
            "object_name": self.object_name,
            "robot_name": self.robot_name,
            "created_from_tool": self.created_from_tool,
            "scene_snapshot_id": self.scene_snapshot_id,
            "stages": [stage.to_dict() for stage in self.stages],
            "stage_report": self.stage_report,
            "approval": self.approval.to_dict(),
            "evidence": self.evidence,
        }
        data.update(self.raw)
        return data
