from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Sequence, cast
from uuid import uuid4

from moveit_mcp.gripper import SimulatedGripperState
from moveit_mcp.models import (
    Evidence,
    ExecutionApproval,
    TaskExecutionResult,
    TaskSolution,
    TaskStage,
    ToolResult,
    VerificationCheck,
)
from moveit_mcp.pick import PickPlanInputError, build_oriented_pick_workflow, build_pick_candidates
from moveit_mcp.place import PlacePlanInputError, build_place_workflow
from moveit_mcp.vizor_client import (
    AttachSceneFeedback,
    CurrentPoseFeedback,
    DetachSceneFeedback,
    FakeRosbridgeTransport,
    PlanFeedback,
    Pose,
    RemoveSceneFeedback,
    RosbridgeTransport,
    VizorClient,
)

SUCCESS_STATUSES = {"success", "success! "}
PICK_PLANNING_STRATEGIES = {"auto", "cartesian", "sampled_approach"}
PICK_TASK_BACKENDS = {"emulated", "mtc"}
COMPOUND_TASK_GOALS = {"hold", "release", "move_and_release", "pick_place"}
COMPOUND_TASK_TARGET_GOALS = {"move_and_release", "pick_place"}
MANIPULATION_TASK_GOALS = {"hold", "place", "release", "move_and_release", "pick_place"}
MANIPULATION_TASK_TARGET_GOALS = {"place", "move_and_release", "pick_place"}
MANIPULATION_TASK_BACKENDS = {"staged_moveit"}
COMPOUND_STAGE_INTENTS = {
    "observe_current_state",
    "approach_object",
    "close_gripper",
    "verify_attached",
    "lift",
    "move_to_pose",
    "adjust_pose",
    "open_gripper",
    "release_object",
    "verify_released",
}
COMPOUND_REJECTED_INTENT_FRAGMENTS = ("slide", "push", "script", "code", "waypoint")
POSE_INPUT_CORRECTION = (
    "Retry with a target pose containing finite x, y, z coordinates and an orientation object "
    "with finite x, y, z, w values forming a normalized quaternion."
)
REUSED_PLAN_NAME_CORRECTION = "Omit plan_name or retry with a fresh unused plan_name."
UNVERIFIED_PLAN_CORRECTION = "Call a planning tool first, then execute only the returned raw.plan_name."
PLAN_NOT_EXECUTABLE_CORRECTION = (
    "Replan with a smaller or safer target, then execute only a successful returned raw.plan_name."
)
PHYSICAL_MODE_CORRECTION = (
    "Execution is blocked until /vizor_robot_control/physical is confirmed false; verify physical mode false before retrying."
)
EXECUTION_UNVERIFIED_CORRECTION = (
    "Check fake controller joint-state feedback, then replan or retry execution only after a verified plan is available."
)
CURRENT_POSE_CORRECTION = "Check the MoveIt current-pose service and retry after robot state feedback is available."
ROBOT_STATE_CORRECTION = (
    "Check rosbridge, /UR10/get_current_pose, /vizor_robot_control/physical, and "
    "/UR10/move_group/fake_controller_joint_states before retrying."
)
GRIPPER_NOT_CLOSED_CORRECTION = "Call close_gripper for this robot before retrying attach_object."
GRIPPER_COMMAND_CORRECTION = (
    "Check the Robotiq action server, /UR10/gripper_joint_states, and rosbridge before retrying."
)
REMOVE_ATTACHED_OBJECT_CORRECTION = "Release and verify the object before removing it from the planning scene."
PLANNING_SCENE_ATTACH_CORRECTION = (
    "Call moveit_list_scene_objects, then retry with an object that is still present as a free planning-scene object."
)
PLANNING_SCENE_CORRECTION = "Check rosbridge and /UR10/get_planning_scene before retrying scene-object tools."
OBJECT_NOT_FOUND_CORRECTION = "Call moveit_list_scene_objects, then retry with an object name from raw.objects."
PICK_OBJECT_CORRECTION = "Call moveit_list_scene_objects and moveit_get_object_context, then retry with one object_name."
ATTACHED_OBJECT_CORRECTION = (
    "Execute the pick plan, confirm the gripper is closed, attach the object in the planning scene, "
    "then retry moveit_verify_attached_object."
)
PLACE_OBJECT_CORRECTION = "Plan place only after the object is attached to the gripper."
PLACE_TARGET_CORRECTION = "Retry with a target_pose or target_position in base_link."
RELEASE_OBJECT_CORRECTION = (
    "Open the gripper through Verified Real Robot Execution, then retry with verified_gripper_open=true "
    "and an explicit object_pose for the released object."
)
COMPOUND_MTC_CORRECTION = 'Retry with backend="mtc" after the Vizor MTC compound task backend is available.'
MANIPULATION_BACKEND_CORRECTION = 'Retry with backend="staged_moveit"; no MTC fallback is available for this tool.'
MANIPULATION_REQUIREMENTS_CORRECTION = 'Retry with requirements {"goal": "hold", "object_name": "<planning-scene object>"} and backend="staged_moveit".'
COMPOUND_REQUIREMENTS_CORRECTION = (
    "Retry with requirements containing goal and object_name. For move_and_release, "
    "and pick_place include target_pose or target_position inside requirements."
)
COMPOUND_STAGE_INTENT_CORRECTION = (
    "Omit stage_intents or use only supported hint names: observe_current_state, approach_object, close_gripper, "
    "verify_attached, lift, move_to_pose, adjust_pose, open_gripper, release_object, verify_released."
)


class MoveItMcpTools:
    def __init__(self, *, client: VizorClient, pick_task_backend: str = "emulated") -> None:
        if pick_task_backend not in PICK_TASK_BACKENDS:
            raise ValueError(f"pick_task_backend must be one of {sorted(PICK_TASK_BACKENDS)}")
        self.client = client
        self.pick_task_backend = pick_task_backend
        self._planned: dict[tuple[str, str], dict[str, Any]] = {}
        self._used_plan_names: set[tuple[str, str]] = set()
        self._task_solutions: dict[str, TaskSolution] = {}
        self._task_solution_sequence = 0
        self.gripper = SimulatedGripperState.empty()

    @classmethod
    def with_fake_transport(cls, transport: FakeRosbridgeTransport, *, pick_task_backend: str = "emulated") -> "MoveItMcpTools":
        return cls(client=VizorClient(transport=transport), pick_task_backend=pick_task_backend)

    @classmethod
    def with_transport(cls, transport: RosbridgeTransport, *, pick_task_backend: str = "emulated") -> "MoveItMcpTools":
        return cls(client=VizorClient(transport=transport), pick_task_backend=pick_task_backend)

    def plan_free_motion(
        self,
        robot: str,
        name: str | dict[str, Any] | None = None,
        position: dict[str, Any] | None = None,
        timeout_s: float = 10.0,
        allow_existing_name: bool = False,
    ) -> dict[str, Any]:
        try:
            plan_name, pose_input = self._resolve_single_pose_args("plan_free_motion", robot, name, position)
            pose = Pose.from_input(pose_input)
            _validate_finite_pose(pose)
        except (KeyError, TypeError, ValueError) as exc:
            return self._invalid_input_result(
                robot=robot,
                tool="plan_free_motion",
                status="invalid pose",
                details=str(exc),
                plan_name=name if isinstance(name, str) else None,
            )

        name_error = self._reserve_plan_name(
            robot=robot,
            tool="plan_free_motion",
            name=plan_name,
            allow_existing_name=allow_existing_name,
        )
        if name_error is not None:
            return name_error

        self._planned.pop((robot, plan_name), None)
        feedback = self.client.plan_free_motion(
            robot=robot,
            name=plan_name,
            pose=pose,
            timeout_s=timeout_s,
        )
        return self._plan_result(tool="plan_free_motion", feedback=feedback)

    def plan_cartesian_motion(
        self,
        robot: str,
        name: str | list[dict[str, Any]] | None = None,
        positions: list[dict[str, Any]] | None = None,
        timeout_s: float = 10.0,
        allow_existing_name: bool = False,
    ) -> dict[str, Any]:
        try:
            plan_name, pose_inputs = self._resolve_pose_list_args("plan_cartesian_motion", robot, name, positions)
            poses = [Pose.from_input(value) for value in pose_inputs]
            for pose in poses:
                _validate_finite_pose(pose)
        except (KeyError, TypeError, ValueError) as exc:
            return self._invalid_input_result(
                robot=robot,
                tool="plan_cartesian_motion",
                status="invalid pose",
                details=str(exc),
                plan_name=name if isinstance(name, str) else None,
            )

        name_error = self._reserve_plan_name(
            robot=robot,
            tool="plan_cartesian_motion",
            name=plan_name,
            allow_existing_name=allow_existing_name,
        )
        if name_error is not None:
            return name_error

        self._planned.pop((robot, plan_name), None)
        feedback = self.client.plan_cartesian_motion(
            robot=robot,
            name=plan_name,
            poses=poses,
            timeout_s=timeout_s,
        )
        return self._plan_result(tool="plan_cartesian_motion", feedback=feedback)

    def get_current_pose(self, robot: str, timeout_s: float = 2.0) -> dict[str, Any]:
        feedback = self.client.get_current_pose(robot=robot, timeout_s=timeout_s)
        return self._current_pose_result(feedback)

    def get_robot_state(self, robot: str, timeout_s: float = 2.0) -> dict[str, Any]:
        feedback = self.client.get_robot_state(robot=robot, timeout_s=timeout_s)
        checks = [
            VerificationCheck("current_pose_observed", feedback.pose is not None, str(feedback.pose)),
            VerificationCheck("physical_mode_observed", feedback.physical_mode is not None, str(feedback.physical_mode)),
            VerificationCheck("joint_state_observed", feedback.joint_state is not None, str(feedback.joint_state)),
        ]
        evidence = [Evidence("ros_observation", feedback.source)]
        raw = {
            "planning_frame": feedback.planning_frame,
            "pose": feedback.pose.to_msg() if feedback.pose is not None else None,
            "physical_mode": feedback.physical_mode,
            "joint_state": feedback.joint_state,
            "source": feedback.source,
        }
        if feedback.ok:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_get_robot_state",
                phase="observed",
                status=feedback.status,
                message=feedback.message,
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_get_robot_state",
            phase="observed",
            status=feedback.status,
            message=feedback.message,
            correction=ROBOT_STATE_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def list_scene_objects(self, robot: str, timeout_s: float = 2.0) -> dict[str, Any]:
        feedback = self.client.list_scene_objects(robot=robot, timeout_s=timeout_s)
        objects_observed = feedback.ok
        checks = [VerificationCheck("planning_scene_observed", objects_observed, feedback.status)]
        evidence = [Evidence("ros_service", feedback.message, path=feedback.source)]
        raw = {
            "planning_frame": feedback.planning_frame,
            "object_count": len(feedback.objects),
            "objects": feedback.objects,
            "source": feedback.source,
        }
        if feedback.ok:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_list_scene_objects",
                phase="observed",
                status=feedback.status,
                message=feedback.message,
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_list_scene_objects",
            phase="observed",
            status=feedback.status,
            message=feedback.message,
            correction=PLANNING_SCENE_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def get_object_context(self, robot: str, object_name: str, timeout_s: float = 2.0) -> dict[str, Any]:
        feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        scene_observed = feedback.status != "planning scene unavailable"
        object_observed = feedback.ok and feedback.object_context is not None
        checks = [
            VerificationCheck("planning_scene_observed", scene_observed, feedback.status),
            VerificationCheck("object_observed", object_observed, object_name),
        ]
        evidence = [Evidence("ros_service", feedback.message, path=feedback.source)]
        raw = {
            "planning_frame": feedback.planning_frame,
            "object": feedback.object_context,
            "available_objects": feedback.available_objects,
            "source": feedback.source,
        }
        if feedback.ok:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_get_object_context",
                phase="observed",
                status=feedback.status,
                message=feedback.message,
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        correction = PLANNING_SCENE_CORRECTION if not scene_observed else OBJECT_NOT_FOUND_CORRECTION
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_get_object_context",
            phase="observed",
            status=feedback.status,
            message=feedback.message,
            correction=correction,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def plan_pick(
        self,
        robot: str,
        object_name: str,
        *,
        plan_name: str | None = None,
        grasp_face: str = "top",
        approach_distance_m: float = 0.08,
        grasp_standoff_m: float = 0.01,
        lift_distance_m: float = 0.1,
        planning_strategy: str = "auto",
        timeout_s: float = 10.0,
        allow_existing_name: bool = False,
    ) -> dict[str, Any]:
        if planning_strategy not in PICK_PLANNING_STRATEGIES:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_pick",
                phase="pre_plan",
                status="invalid planning strategy",
                message="Refusing to plan pick with an unsupported planning_strategy",
                correction='Use planning_strategy="auto", "cartesian", or "sampled_approach".',
                checks=[VerificationCheck("planning_strategy_valid", False, str(planning_strategy))],
                evidence=[Evidence("mcp_state", f"invalid planning_strategy: {planning_strategy}")],
                raw={
                    "planning_strategy": planning_strategy,
                    "available_planning_strategies": sorted(PICK_PLANNING_STRATEGIES),
                    "candidate_attempts": [],
                },
            ).to_dict()

        if not isinstance(object_name, str) or not object_name.strip():
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_pick",
                phase="pre_plan",
                status="invalid object name",
                message="Refusing to plan pick without a non-empty object_name",
                correction=PICK_OBJECT_CORRECTION,
                checks=[VerificationCheck("object_name_valid", False, str(object_name))],
                evidence=[Evidence("mcp_state", "missing object_name")],
                raw={
                    "plan_name": plan_name,
                    "object_name": object_name,
                    "planning_strategy": planning_strategy,
                    "candidate_attempts": [],
                },
            ).to_dict()

        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._pick_object_context_failed_result(
                robot=robot,
                object_name=object_name,
                plan_name=plan_name,
                feedback=object_feedback,
                planning_strategy=planning_strategy,
            )

        planner = "cartesian"
        planning_pipeline = None
        planner_id = None

        try:
            if planning_strategy == "cartesian":
                workflows = [
                    build_oriented_pick_workflow(
                        object_feedback.object_context,
                        requested_grasp_face=grasp_face,
                        approach_distance_m=approach_distance_m,
                        grasp_standoff_m=grasp_standoff_m,
                        lift_distance_m=lift_distance_m,
                    )
                ]
                resolved_strategy = "cartesian"
            elif planning_strategy == "sampled_approach":
                workflows = [
                    build_oriented_pick_workflow(
                        object_feedback.object_context,
                        requested_grasp_face=grasp_face,
                        approach_distance_m=approach_distance_m,
                        grasp_standoff_m=grasp_standoff_m,
                        lift_distance_m=lift_distance_m,
                    )
                ]
                resolved_strategy = "sampled_approach"
                planner = "sampled_approach"
                planning_pipeline = "ompl"
                planner_id = "RRTConnect"
            else:
                workflow = build_oriented_pick_workflow(
                    object_feedback.object_context,
                    requested_grasp_face=grasp_face,
                    approach_distance_m=approach_distance_m,
                    grasp_standoff_m=grasp_standoff_m,
                    lift_distance_m=lift_distance_m,
                )
                base_plan_name = plan_name or self._new_plan_name(robot, "plan_pick")
                preposition_plan_name = self._pick_preposition_plan_name(base_plan_name)
                local_pick_plan_name = self._pick_local_plan_name(base_plan_name)
                name_error = self._reserve_plan_name(
                    robot=robot,
                    tool="moveit_plan_pick",
                    name=preposition_plan_name,
                    allow_existing_name=allow_existing_name,
                )
                if name_error is not None:
                    return name_error

                self._planned.pop((robot, preposition_plan_name), None)
                preposition_pose = Pose.from_input(workflow["waypoints"][0])
                feedback = self.client.plan_free_motion(
                    robot=robot,
                    name=preposition_plan_name,
                    pose=preposition_pose,
                    timeout_s=timeout_s,
                )
                return self._pick_preposition_result(
                    feedback=feedback,
                    object_context=object_feedback.object_context,
                    workflow=workflow,
                    source=object_feedback.source,
                    local_pick_plan_name=local_pick_plan_name,
                    planning_strategy=planning_strategy,
                    planning_strategy_resolved="staged_preposition",
                )
        except (IndexError, KeyError, TypeError, ValueError, PickPlanInputError) as exc:
            return self._invalid_pick_workflow_result(
                robot=robot,
                object_name=object_name,
                plan_name=plan_name,
                object_context=object_feedback.object_context,
                exc=exc,
                planning_strategy=planning_strategy,
            )

        base_plan_name = plan_name or self._new_plan_name(robot, "plan_pick")
        multi_attempt = planning_strategy == "auto"
        candidate_attempts: list[dict[str, Any]] = []
        last_feedback: PlanFeedback | None = None
        last_workflow: dict[str, Any] | None = None

        for attempt_index, workflow in enumerate(workflows, start=1):
            attempt_plan_name = self._pick_attempt_plan_name(
                base_plan_name,
                attempt_index,
                multi_attempt=multi_attempt,
            )
            name_error = self._reserve_plan_name(
                robot=robot,
                tool="moveit_plan_pick",
                name=attempt_plan_name,
                allow_existing_name=allow_existing_name and attempt_index == 1,
            )
            if name_error is not None:
                return name_error

            self._planned.pop((robot, attempt_plan_name), None)
            if planning_strategy == "sampled_approach":
                poses = [Pose.from_input(waypoint) for waypoint in workflow["waypoints"]]
                feedback = self.client.plan_sampled_motion(
                    robot=robot,
                    name=attempt_plan_name,
                    poses=poses,
                    timeout_s=timeout_s,
                )
            else:
                first_segment = workflow["motion_segments"][0]
                segment_waypoints = [
                    workflow["waypoints"][index]
                    for index in first_segment["waypoint_indexes"]
                ]
                feedback = self.client.plan_cartesian_motion(
                    robot=robot,
                    name=attempt_plan_name,
                    poses=[Pose.from_input(waypoint) for waypoint in segment_waypoints],
                    timeout_s=timeout_s,
                )
            success = feedback.status in SUCCESS_STATUSES and feedback.trajectory_points > 0 and feedback.can_execute
            candidate_attempts.append(
                self._pick_candidate_attempt(
                    attempt_index=attempt_index,
                    plan_name=attempt_plan_name,
                    workflow=workflow,
                    feedback=feedback,
                    selected=success,
                    planner=planner,
                    planning_pipeline=planning_pipeline,
                    planner_id=planner_id,
                )
            )
            last_feedback = feedback
            last_workflow = workflow
            if success:
                return self._pick_plan_result(
                    feedback=feedback,
                    object_context=object_feedback.object_context,
                    workflow=workflow,
                    source=object_feedback.source,
                    planning_strategy=planning_strategy,
                    planning_strategy_resolved=resolved_strategy,
                    candidate_attempts=candidate_attempts,
                )

        if last_feedback is None or last_workflow is None:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_pick",
                phase="pre_plan",
                status="no pick candidates",
                message="Refusing to publish pick plan because no pick candidates were generated",
                correction=PICK_OBJECT_CORRECTION,
                checks=[VerificationCheck("pick_candidate_observed", False, object_name)],
                evidence=[Evidence("mcp_state", f"no pick candidates for object: {object_name}")],
                raw={
                    "plan_name": plan_name,
                    "object_name": object_name,
                    "planning_strategy": planning_strategy,
                    "planning_strategy_resolved": resolved_strategy,
                    "candidate_attempts": [],
                },
            ).to_dict()

        return self._pick_plan_result(
            feedback=last_feedback,
            object_context=object_feedback.object_context,
            workflow=last_workflow,
            source=object_feedback.source,
            planning_strategy=planning_strategy,
            planning_strategy_resolved=resolved_strategy,
            candidate_attempts=candidate_attempts,
        )

    def plan_place(
        self,
        robot: str,
        object_name: str,
        *,
        plan_name: str | None = None,
        target_pose: dict[str, Any] | None = None,
        target_position: dict[str, Any] | None = None,
        orientation_mode: str = "keep",
        place_face: str | None = None,
        support_face: str | None = None,
        approach_distance_m: float = 0.08,
        place_standoff_m: float = 0.01,
        retreat_distance_m: float = 0.1,
        timeout_s: float = 10.0,
        allow_existing_name: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(object_name, str) or not object_name.strip():
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_place",
                phase="pre_plan",
                status="invalid object name",
                message="Refusing to plan place without a non-empty object_name",
                correction=PICK_OBJECT_CORRECTION,
                checks=[VerificationCheck("object_name_valid", False, str(object_name))],
                evidence=[Evidence("mcp_state", "missing object_name")],
                raw={"plan_name": plan_name, "object_name": object_name},
            ).to_dict()

        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_place",
                phase="pre_plan",
                status=object_feedback.status,
                message=object_feedback.message,
                correction=OBJECT_NOT_FOUND_CORRECTION,
                checks=[VerificationCheck("object_context_observed", False, object_name)],
                evidence=[Evidence("ros_service", object_feedback.status, path=object_feedback.source)],
                raw={
                    "plan_name": plan_name,
                    "object_name": object_name,
                    "available_objects": object_feedback.available_objects,
                },
            ).to_dict()

        held_object = self.gripper.attached_object(robot)
        scene_attached = object_feedback.object_context.get("state") == "attached"
        if held_object not in {None, object_name} and not scene_attached:
            scene_attached = False
        if not scene_attached and held_object != object_name:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_place",
                phase="pre_plan",
                status="object not attached",
                message="Refusing to plan place because the object is not attached to the gripper",
                correction=PLACE_OBJECT_CORRECTION,
                checks=[
                    VerificationCheck("planning_scene_object_attached", scene_attached, str(object_feedback.object_context.get("state"))),
                    VerificationCheck("mcp_gripper_holds_object", held_object == object_name, str(held_object)),
                ],
                evidence=[Evidence("mcp_state", f"attached_object={held_object}")],
                raw={"plan_name": plan_name, "object_name": object_name, "mcp_attached_object": held_object},
            ).to_dict()

        current_feedback = self.client.get_current_pose(robot=robot, timeout_s=2.0)
        current_pose = current_feedback.pose.to_msg() if current_feedback.ok and current_feedback.pose is not None else None
        try:
            workflow = build_place_workflow(
                object_feedback.object_context,
                target_pose=target_pose,
                target_position=target_position,
                current_pose=current_pose,
                orientation_mode=orientation_mode,
                place_face=place_face,
                support_face=support_face,
                approach_distance_m=approach_distance_m,
                place_standoff_m=place_standoff_m,
                retreat_distance_m=retreat_distance_m,
            )
        except (KeyError, TypeError, ValueError, PlacePlanInputError) as exc:
            correction = getattr(exc, "correction", PLACE_TARGET_CORRECTION)
            raw = dict(getattr(exc, "raw", {}))
            raw.update({"plan_name": plan_name, "object_name": object_name})
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_place",
                phase="pre_plan",
                status=str(exc),
                message="Refusing to publish place plan because the place workflow could not be derived",
                correction=correction,
                checks=[VerificationCheck("place_workflow_derived", False, str(exc))],
                evidence=[Evidence("mcp_state", f"invalid place workflow for object: {object_name}")],
                raw=raw,
            ).to_dict()

        resolved_name = plan_name or self._new_plan_name(robot, "plan_place")
        name_error = self._reserve_plan_name(
            robot=robot,
            tool="moveit_plan_place",
            name=resolved_name,
            allow_existing_name=allow_existing_name,
        )
        if name_error is not None:
            return name_error

        self._planned.pop((robot, resolved_name), None)
        poses = [Pose.from_input(waypoint) for waypoint in workflow["waypoints"]]
        feedback = self.client.plan_cartesian_motion(
            robot=robot,
            name=resolved_name,
            poses=poses,
            timeout_s=timeout_s,
        )
        return self._place_plan_result(
            feedback=feedback,
            object_context=object_feedback.object_context,
            workflow=workflow,
            source=object_feedback.source,
        )

    def plan_pick_task(
        self,
        robot: str,
        object_name: str,
        *,
        grasp_face: str | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        if not isinstance(object_name, str) or not object_name.strip():
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_pick_task",
                phase="pre_plan",
                status="invalid object name",
                message="Refusing to create a pick task solution without a non-empty object_name",
                correction=PICK_OBJECT_CORRECTION,
                checks=[VerificationCheck("object_name_valid", False, str(object_name))],
                evidence=[Evidence("mcp_state", "missing object_name")],
                raw={"object_name": object_name},
            ).to_dict()

        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_pick_task",
                object_name=object_name,
                feedback=object_feedback,
            )

        if self.pick_task_backend == "mtc":
            return self._plan_mtc_pick_task_result(
                robot=robot,
                object_name=object_name,
                grasp_face=grasp_face,
                timeout_s=timeout_s,
                object_context=object_feedback.object_context,
                planning_frame=object_feedback.planning_frame,
                observed_at=observed_at,
            )

        try:
            candidate_workflows = build_pick_candidates(
                object_feedback.object_context,
                requested_grasp_face=grasp_face,
            )
            workflow = candidate_workflows[0]
        except (KeyError, TypeError, ValueError, PickPlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_pick_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PICK_OBJECT_CORRECTION),
            )
        candidate_attempts = _pick_task_candidate_attempts(candidate_workflows)

        solution = self._build_task_solution(
            robot=robot,
            object_name=object_name,
            task_kind="pick",
            created_from_tool="moveit_plan_pick_task",
            planning_frame=workflow.get("planning_frame"),
            object_context=object_feedback.object_context,
            observed_at=observed_at,
            stages=[
                TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
                TaskStage("connect_to_pre_grasp", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("approach_grasp", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("close_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
                TaskStage("attach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
                TaskStage("lift_object", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("verify_attached_object", "verification", "solved", [{"kind": "attachment_check"}]),
            ],
            expected_movement=f"pick {object_name}: approach grasp, attach, and lift object",
            raw={
                "selected_grasp_face": workflow["selected_grasp_face"],
                "waypoints": workflow["waypoints"],
                "workflow_steps": workflow["workflow_steps"],
                "parameters": workflow["parameters"],
                "object": object_feedback.object_context,
            },
            candidate_attempts=candidate_attempts,
        )
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def plan_place_task(
        self,
        robot: str,
        object_name: str,
        *,
        target_pose: dict[str, Any] | None = None,
        target_position: dict[str, Any] | None = None,
        orientation_mode: str = "keep",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_place_task",
                object_name=object_name,
                feedback=object_feedback,
            )
        held_object = self.gripper.attached_object(robot)
        scene_attached = object_feedback.object_context.get("state") == "attached"
        if not scene_attached and held_object != object_name:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_place_task",
                phase="pre_plan",
                status="object not attached",
                message="Refusing to create a place task solution because the object is not attached to the gripper",
                correction=PLACE_OBJECT_CORRECTION,
                checks=[
                    VerificationCheck("planning_scene_object_attached", scene_attached, str(object_feedback.object_context.get("state"))),
                    VerificationCheck("mcp_gripper_holds_object", held_object == object_name, str(held_object)),
                ],
                evidence=[Evidence("mcp_state", f"attached_object={held_object}")],
                raw={"object_name": object_name, "mcp_attached_object": held_object},
            ).to_dict()

        current_feedback = self.client.get_current_pose(robot=robot, timeout_s=2.0)
        current_pose = current_feedback.pose.to_msg() if current_feedback.ok and current_feedback.pose is not None else None
        try:
            workflow = build_place_workflow(
                object_feedback.object_context,
                target_pose=target_pose,
                target_position=target_position,
                current_pose=current_pose,
                orientation_mode=orientation_mode,
            )
        except (KeyError, TypeError, ValueError, PlacePlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_place_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PLACE_TARGET_CORRECTION),
            )

        solution = self._build_task_solution(
            robot=robot,
            object_name=object_name,
            task_kind="place",
            created_from_tool="moveit_plan_place_task",
            planning_frame=workflow.get("planning_frame"),
            object_context=object_feedback.object_context,
            observed_at=observed_at,
            stages=[
                TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
                TaskStage("connect_to_place", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("approach_place", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("open_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
                TaskStage("detach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
                TaskStage("retreat", "motion_plan", "solved", [{"kind": "emulated_motion_plan"}]),
                TaskStage("verify_released_object", "verification", "solved", [{"kind": "release_check"}]),
            ],
            expected_movement=f"place {object_name}: approach target, release, and retreat",
            raw={
                "target_object_pose": workflow["target_object_pose"],
                "release_tcp_pose": workflow["release_tcp_pose"],
                "waypoints": workflow["waypoints"],
                "workflow_steps": workflow["workflow_steps"],
                "parameters": workflow["parameters"],
                "release_after_execute": workflow["release_after_execute"],
                "object": object_feedback.object_context,
            },
        )
        solution.raw["execution_contract"] = _emulated_place_execution_contract(
            task_solution_id=solution.task_solution_id,
            object_name=object_name,
            scene_snapshot_id=solution.scene_snapshot_id,
            workflow=workflow,
        )
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def plan_manipulation_task(
        self,
        robot: str,
        *,
        requirements: dict[str, Any],
        backend: str,
        preferences: dict[str, Any] | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        if backend not in MANIPULATION_TASK_BACKENDS:
            result = ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                phase="pre_plan",
                status="staged_moveit backend required",
                message="Refusing to plan manipulation tasks without backend=\"staged_moveit\"",
                correction=MANIPULATION_BACKEND_CORRECTION,
                checks=[VerificationCheck("backend_is_staged_moveit", False, str(backend))],
                evidence=[Evidence("mcp_state", "manipulation task planning requires staged_moveit")],
                raw={"backend": backend, "requirements": requirements, "preferences": preferences},
            ).to_dict()
            result["retryable"] = True
            return result

        if not isinstance(requirements, dict):
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail=str(requirements),
                missing="requirements",
            )
        goal = requirements.get("goal")
        object_name = requirements.get("object_name")
        if goal not in MANIPULATION_TASK_GOALS:
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail=str(goal),
                missing="supported goal",
            )
        if goal == "release" and (not isinstance(object_name, str) or not object_name.strip()):
            object_name = self.gripper.attached_object(robot)
        if not isinstance(object_name, str) or not object_name.strip():
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail=str(object_name),
                missing="object_name",
            )

        normalized_preferences = dict(preferences) if isinstance(preferences, dict) else {}
        if preferences is not None and not isinstance(preferences, dict):
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail=str(preferences),
                missing="preferences object",
            )

        if goal == "release":
            return self._plan_staged_release_manipulation_task(
                robot=robot,
                object_name=object_name.strip(),
                requirements=requirements,
                preferences=normalized_preferences,
                timeout_s=timeout_s,
            )
        if goal in {"place", "move_and_release"}:
            return self._plan_staged_place_manipulation_task(
                robot=robot,
                object_name=object_name.strip(),
                task_kind=str(goal),
                requirements=requirements,
                preferences=normalized_preferences,
                timeout_s=timeout_s,
            )
        if goal == "pick_place":
            return self._plan_staged_pick_place_manipulation_task(
                robot=robot,
                object_name=object_name.strip(),
                requirements=requirements,
                preferences=normalized_preferences,
                timeout_s=timeout_s,
            )

        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                feedback=object_feedback,
            )

        try:
            workflow = build_oriented_pick_workflow(
                object_feedback.object_context,
                requested_grasp_face=normalized_preferences.get("grasp_face"),
                approach_distance_m=float(normalized_preferences.get("approach_distance_m", 0.08)),
                grasp_standoff_m=float(normalized_preferences.get("grasp_standoff_m", 0.01)),
                lift_distance_m=float(requirements.get("lift_distance_m", normalized_preferences.get("lift_distance_m", 0.1))),
            )
        except (KeyError, TypeError, ValueError, PickPlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PICK_OBJECT_CORRECTION),
            )

        params = _safe_dict(workflow.get("parameters"))
        selected_face = _safe_dict(workflow.get("selected_grasp_face"))
        contract_steps = _contract_hold_execution_steps(
            object_name=object_name,
            scene_snapshot_id="",
        )
        selected_candidate = {
            "attempt_index": 1,
            "status": "contract",
            "grasp_face": params.get("grasp_face") or selected_face.get("name"),
            "approach_distance_m": params.get("approach_distance_m"),
            "grasp_standoff_m": params.get("grasp_standoff_m"),
            "lift_distance_m": params.get("lift_distance_m"),
        }
        solution = self._build_task_solution(
            robot=robot,
            object_name=object_name,
            task_kind="hold",
            created_from_tool="moveit_plan_manipulation_task",
            planning_frame=workflow.get("planning_frame"),
            object_context=object_feedback.object_context,
            observed_at=observed_at,
            stages=_contract_hold_task_stages(),
            expected_movement=f"hold {object_name}: approach grasp, attach, and lift object",
            raw={
                "requirements": dict(requirements),
                "preferences": normalized_preferences,
                "selected_grasp_face": workflow["selected_grasp_face"],
                "selected_candidate": selected_candidate,
                "candidate_attempts": [],
                "waypoints": workflow["waypoints"],
                "workflow_steps": workflow["workflow_steps"],
                "parameters": workflow["parameters"],
                "object": object_feedback.object_context,
                "preview": _staged_waypoint_agent_path_preview(workflow["waypoints"], contract_steps),
            },
            candidate_attempts=[],
            backend="staged_moveit",
            solver="contract_moveit",
            selected_cost=1.0,
        )
        solution.raw["execution_contract"] = _contract_hold_execution_contract(
            task_solution_id=solution.task_solution_id,
            object_name=object_name,
            scene_snapshot_id=solution.scene_snapshot_id,
        )
        solution.raw["scene_snapshot"] = {
            "id": solution.scene_snapshot_id,
            "planning_frame": object_feedback.planning_frame,
            "object_count": len(object_feedback.available_objects),
        }
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def _plan_staged_release_manipulation_task(
        self,
        *,
        robot: str,
        object_name: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                feedback=object_feedback,
            )
        held_object = self.gripper.attached_object(robot)
        scene_attached = object_feedback.object_context.get("state") == "attached"
        if not scene_attached and held_object != object_name:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                phase="pre_plan",
                status="object not held",
                message="Refusing to plan release because the object is not attached or held.",
                correction="Observe the held object or execute a hold task before release.",
                checks=[
                    VerificationCheck("planning_scene_object_attached", scene_attached, str(object_feedback.object_context.get("state"))),
                    VerificationCheck("mcp_gripper_holds_object", held_object == object_name, str(held_object)),
                ],
                evidence=[Evidence("mcp_state", f"attached_object={held_object}")],
                raw={
                    "backend": "staged_moveit",
                    "requirements": requirements,
                    "preferences": preferences,
                    "failed_stage": "release_precondition",
                    "failure_code": "not_holding_object",
                    "object_name": object_name,
                },
            ).to_dict()
        object_pose = _object_pose_from_context(object_feedback.object_context)
        if object_pose is None:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                phase="pre_plan",
                status="release object pose missing",
                message="Refusing to plan release because the released object pose is not known.",
                correction="Refresh MoveIt object context, then retry release with full pose evidence.",
                checks=[VerificationCheck("released_object_pose_present", False, "missing")],
                evidence=[Evidence("mcp_state", f"object pose unavailable for {object_name}")],
                raw={
                    "backend": "staged_moveit",
                    "requirements": requirements,
                    "preferences": preferences,
                    "failed_stage": "release_pose",
                    "failure_code": "missing_release_pose",
                    "object_name": object_name,
                },
            ).to_dict()

        solution = self._build_task_solution(
            robot=robot,
            object_name=object_name,
            task_kind="release",
            created_from_tool="moveit_plan_manipulation_task",
            planning_frame=object_feedback.planning_frame,
            object_context=object_feedback.object_context,
            observed_at=observed_at,
            stages=[
                TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
                TaskStage("open_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
                TaskStage("detach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
                TaskStage("verify_released_object", "verification", "solved", [{"kind": "release_check"}]),
            ],
            expected_movement=f"release {object_name} in place",
            raw={
                "requirements": dict(requirements),
                "preferences": preferences,
                "release_after_execute": {"object_name": object_name, "object_pose": object_pose},
                "object": object_feedback.object_context,
                "preview": {"kind": "AgentPath", "name": "AgentPath", "motion_stages": [], "ar_preview_mode": "none_no_motion"},
            },
            backend="staged_moveit",
            solver="staged_moveit",
        )
        solution.raw["execution_contract"] = _staged_release_execution_contract(
            task_solution_id=solution.task_solution_id,
            object_name=object_name,
            scene_snapshot_id=solution.scene_snapshot_id,
            object_pose=object_pose,
        )
        solution.raw["scene_snapshot"] = {
            "id": solution.scene_snapshot_id,
            "planning_frame": object_feedback.planning_frame,
            "object_count": len(object_feedback.available_objects),
        }
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def _plan_staged_place_manipulation_task(
        self,
        *,
        robot: str,
        object_name: str,
        task_kind: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        target_pose = requirements.get("target_pose")
        target_position = requirements.get("target_position")
        if target_pose is None and target_position is None:
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail="missing target_pose or target_position",
                missing="target",
            )
        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                feedback=object_feedback,
            )
        held_object = self.gripper.attached_object(robot)
        scene_attached = object_feedback.object_context.get("state") == "attached"
        if not scene_attached and held_object != object_name:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                phase="pre_plan",
                status="object not attached",
                message="Refusing to plan place because the object is not attached to the gripper.",
                correction=PLACE_OBJECT_CORRECTION,
                checks=[
                    VerificationCheck("planning_scene_object_attached", scene_attached, str(object_feedback.object_context.get("state"))),
                    VerificationCheck("mcp_gripper_holds_object", held_object == object_name, str(held_object)),
                ],
                evidence=[Evidence("mcp_state", f"attached_object={held_object}")],
                raw={
                    "backend": "staged_moveit",
                    "requirements": requirements,
                    "preferences": preferences,
                    "failed_stage": "place_precondition",
                    "failure_code": "not_holding_object",
                    "object_name": object_name,
                },
            ).to_dict()

        current_feedback = self.client.get_current_pose(robot=robot, timeout_s=2.0)
        current_pose = current_feedback.pose.to_msg() if current_feedback.ok and current_feedback.pose is not None else None
        try:
            workflow = build_place_workflow(
                object_feedback.object_context,
                target_pose=target_pose if isinstance(target_pose, dict) else None,
                target_position=target_position if isinstance(target_position, dict) else None,
                current_pose=current_pose,
                orientation_mode=str(preferences.get("orientation_mode", "keep")),
                place_face=preferences.get("place_face") if isinstance(preferences.get("place_face"), str) else None,
                support_face=preferences.get("support_face") if isinstance(preferences.get("support_face"), str) else None,
                approach_distance_m=float(preferences.get("approach_distance_m", 0.08)),
                place_standoff_m=float(preferences.get("place_standoff_m", 0.01)),
                retreat_distance_m=float(preferences.get("retreat_distance_m", 0.1)),
            )
        except (KeyError, TypeError, ValueError, PlacePlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PLACE_TARGET_CORRECTION),
            )

        attempt = self._try_staged_place_candidate(
            robot=robot,
            object_name=object_name,
            task_kind=task_kind,
            attempt_index=1,
            workflow=workflow,
            timeout_s=timeout_s,
        )
        if attempt["status"] != "selected":
            return self._manipulation_task_failed_result(
                robot=robot,
                object_name=object_name,
                requirements=requirements,
                preferences=preferences,
                failed_stage=str(attempt.get("failed_stage") or "connect_to_place"),
                candidate_attempts=[attempt],
            )

        solution = self._build_task_solution(
            robot=robot,
            object_name=object_name,
            task_kind=task_kind,
            created_from_tool="moveit_plan_manipulation_task",
            planning_frame=workflow.get("planning_frame"),
            object_context=object_feedback.object_context,
            observed_at=observed_at,
            stages=_staged_place_task_stages(attempt["motion_stages"]),
            expected_movement=f"{task_kind} {object_name}: move to target, release, and retreat",
            raw={
                "requirements": dict(requirements),
                "preferences": preferences,
                "selected_candidate": _staged_candidate_public_summary(attempt),
                "candidate_attempts": [attempt],
                "target_object_pose": workflow["target_object_pose"],
                "release_tcp_pose": workflow["release_tcp_pose"],
                "waypoints": workflow["waypoints"],
                "workflow_steps": workflow["workflow_steps"],
                "parameters": workflow["parameters"],
                "release_after_execute": workflow["release_after_execute"],
                "object": object_feedback.object_context,
                "preview": _staged_agent_path_preview(attempt["motion_stages"]),
            },
            candidate_attempts=[attempt],
            backend="staged_moveit",
            solver="staged_moveit",
            selected_cost=round(1.0 + 0.1 * len(attempt["motion_stages"]), 3),
        )
        solution.raw["execution_contract"] = _staged_place_execution_contract(
            task_solution_id=solution.task_solution_id,
            object_name=object_name,
            scene_snapshot_id=solution.scene_snapshot_id,
            object_pose=workflow["release_after_execute"]["object_pose"],
            motion_stages=attempt["motion_stages"],
        )
        solution.raw["scene_snapshot"] = {
            "id": solution.scene_snapshot_id,
            "planning_frame": object_feedback.planning_frame,
            "object_count": len(object_feedback.available_objects),
        }
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def _plan_staged_pick_place_manipulation_task(
        self,
        *,
        robot: str,
        object_name: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        target_pose = requirements.get("target_pose")
        target_position = requirements.get("target_position")
        if target_pose is None and target_position is None:
            return self._invalid_manipulation_requirements_result(
                robot=robot,
                requirements=requirements,
                preferences=preferences,
                detail="missing target_pose or target_position",
                missing="target",
            )
        observed_at = datetime.now(timezone.utc)
        object_feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        if not object_feedback.ok or object_feedback.object_context is None:
            return self._task_object_context_failed_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                feedback=object_feedback,
            )
        try:
            candidate_workflows = build_pick_candidates(
                object_feedback.object_context,
                requested_grasp_face=preferences.get("grasp_face"),
                approach_distance_m=float(preferences.get("approach_distance_m", 0.08)),
                grasp_standoff_m=float(preferences.get("grasp_standoff_m", 0.01)),
                lift_distance_m=float(requirements.get("lift_distance_m", preferences.get("lift_distance_m", 0.1))),
                max_candidates=8,
            )
        except (KeyError, TypeError, ValueError, PickPlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PICK_OBJECT_CORRECTION),
            )
        current_feedback = self.client.get_current_pose(robot=robot, timeout_s=2.0)
        current_pose = current_feedback.pose.to_msg() if current_feedback.ok and current_feedback.pose is not None else None
        try:
            place_workflow = build_place_workflow(
                object_feedback.object_context,
                target_pose=target_pose if isinstance(target_pose, dict) else None,
                target_position=target_position if isinstance(target_position, dict) else None,
                current_pose=current_pose,
                orientation_mode=str(preferences.get("orientation_mode", "keep")),
                place_face=preferences.get("place_face") if isinstance(preferences.get("place_face"), str) else None,
                support_face=preferences.get("support_face") if isinstance(preferences.get("support_face"), str) else None,
                approach_distance_m=float(preferences.get("place_approach_distance_m", preferences.get("approach_distance_m", 0.08))),
                place_standoff_m=float(preferences.get("place_standoff_m", 0.01)),
                retreat_distance_m=float(preferences.get("retreat_distance_m", 0.1)),
            )
        except (KeyError, TypeError, ValueError, PlacePlanInputError) as exc:
            return self._invalid_task_workflow_result(
                robot=robot,
                tool="moveit_plan_manipulation_task",
                object_name=object_name,
                exc=exc,
                correction=getattr(exc, "correction", PLACE_TARGET_CORRECTION),
            )

        candidate_attempts: list[dict[str, Any]] = []
        best_failure: dict[str, Any] | None = None
        for attempt_index, pick_workflow in enumerate(candidate_workflows[:8], start=1):
            attempt = self._try_staged_hold_candidate(
                robot=robot,
                object_name=object_name,
                attempt_index=attempt_index,
                workflow=pick_workflow,
                timeout_s=timeout_s,
            )
            candidate_attempts.append(attempt)
            if attempt["status"] != "selected":
                if best_failure is None or len(attempt.get("motion_stages", [])) > len(best_failure.get("motion_stages", [])):
                    best_failure = attempt
                continue
            place_attempt = self._try_staged_place_candidate(
                robot=robot,
                object_name=object_name,
                task_kind="pick_place",
                attempt_index=attempt_index,
                workflow=place_workflow,
                timeout_s=timeout_s,
            )
            if place_attempt["status"] != "selected":
                combined_attempt = dict(attempt)
                combined_attempt["status"] = "failed"
                combined_attempt["selected"] = False
                combined_attempt["failed_stage"] = place_attempt.get("failed_stage")
                combined_attempt["failure_code"] = place_attempt.get("failure_code")
                combined_attempt["motion_stages"] = list(attempt.get("motion_stages", [])) + list(place_attempt.get("motion_stages", []))
                candidate_attempts[-1] = combined_attempt
                if best_failure is None or len(combined_attempt.get("motion_stages", [])) > len(best_failure.get("motion_stages", [])):
                    best_failure = combined_attempt
                continue
            combined_waypoints = list(pick_workflow["waypoints"]) + list(place_workflow["waypoints"])
            combined_motion_stages = list(attempt["motion_stages"]) + list(place_attempt["motion_stages"])
            solution = self._build_task_solution(
                robot=robot,
                object_name=object_name,
                task_kind="pick_place",
                created_from_tool="moveit_plan_manipulation_task",
                planning_frame=pick_workflow.get("planning_frame"),
                object_context=object_feedback.object_context,
                observed_at=observed_at,
                stages=[
                    *_staged_hold_task_stages(attempt["motion_stages"]),
                    *_staged_place_task_stages(place_attempt["motion_stages"])[1:],
                ],
                expected_movement=f"pick and place {object_name}: hold, move to target, release, and retreat",
                raw={
                    "requirements": dict(requirements),
                    "preferences": preferences,
                    "selected_grasp_face": pick_workflow["selected_grasp_face"],
                    "selected_candidate": _staged_candidate_public_summary(attempt),
                    "candidate_attempts": candidate_attempts,
                    "waypoints": combined_waypoints,
                    "workflow_steps": list(pick_workflow["workflow_steps"]) + list(place_workflow["workflow_steps"]),
                    "parameters": {
                        "pick": pick_workflow["parameters"],
                        "place": place_workflow["parameters"],
                    },
                    "target_object_pose": place_workflow["target_object_pose"],
                    "release_after_execute": place_workflow["release_after_execute"],
                    "object": object_feedback.object_context,
                    "preview": _staged_agent_path_preview(combined_motion_stages),
                },
                candidate_attempts=candidate_attempts,
                backend="staged_moveit",
                solver="staged_moveit",
                selected_cost=round(float(attempt_index) + 0.1 * len(combined_motion_stages), 3),
            )
            solution.raw["execution_contract"] = _staged_pick_place_execution_contract(
                task_solution_id=solution.task_solution_id,
                object_name=object_name,
                scene_snapshot_id=solution.scene_snapshot_id,
                release_object_pose=place_workflow["release_after_execute"]["object_pose"],
                motion_stages=combined_motion_stages,
            )
            solution.raw["scene_snapshot"] = {
                "id": solution.scene_snapshot_id,
                "planning_frame": object_feedback.planning_frame,
                "object_count": len(object_feedback.available_objects),
            }
            self._task_solutions[solution.task_solution_id] = solution
            return self._task_solution_planned_result(solution)

        failed_stage = str((best_failure or {}).get("failed_stage") or "pick_candidate_generation")
        return self._manipulation_task_failed_result(
            robot=robot,
            object_name=object_name,
            requirements=requirements,
            preferences=preferences,
            failed_stage=failed_stage,
            candidate_attempts=candidate_attempts,
        )

    def plan_compound_task(
        self,
        robot: str,
        object_name: str | None = None,
        *,
        task_goal: str | None = None,
        requirements: dict[str, Any] | None = None,
        preferences: dict[str, Any] | None = None,
        stage_intents: Sequence[str] | None = None,
        target_pose: dict[str, Any] | None = None,
        target_position: dict[str, Any] | None = None,
        backend: str | None = None,
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        requirements_error = self._compound_requirements_error(
            robot=robot,
            requirements=requirements,
            legacy_object_name=object_name,
            legacy_task_goal=task_goal,
        )
        if requirements_error is not None:
            return requirements_error
        assert isinstance(requirements, dict)
        normalized_requirements = dict(requirements)
        normalized_task_goal = str(normalized_requirements["goal"])
        normalized_object_name = str(normalized_requirements["object_name"])
        normalized_preferences = dict(preferences) if isinstance(preferences, dict) else None

        if preferences is not None and normalized_preferences is None:
            return self._invalid_compound_requirements_result(
                robot=robot,
                status="invalid compound preferences",
                message="Refusing to plan a compound task because preferences must be an object when provided",
                detail=str(preferences),
                requirements=normalized_requirements,
                preferences=None,
                stage_intents=stage_intents,
                object_name=normalized_object_name,
                task_goal=normalized_task_goal,
            )

        if backend != "mtc":
            result = ToolResult.fail_result(
                robot=robot,
                tool="moveit_plan_compound_task",
                phase="pre_plan",
                status="mtc backend required",
                message="Refusing to plan compound tasks without backend=\"mtc\"",
                correction=COMPOUND_MTC_CORRECTION,
                checks=[VerificationCheck("backend_is_mtc", False, str(backend))],
                evidence=[Evidence("mcp_state", "compound task planning requires MTC")],
                raw={
                    "backend": backend,
                    "requirements": normalized_requirements,
                    "preferences": normalized_preferences,
                    "stage_intents": list(stage_intents) if stage_intents is not None else None,
                    "object_name": normalized_object_name,
                    "task_goal": normalized_task_goal,
                },
            ).to_dict()
            result["retryable"] = True
            return result

        goal_error = self._compound_task_goal_error(
            robot=robot,
            object_name=normalized_object_name,
            task_goal=normalized_task_goal,
            requirements=normalized_requirements,
            preferences=normalized_preferences,
            stage_intents=stage_intents,
        )
        if goal_error is not None:
            return goal_error

        intent_error = self._compound_stage_intent_error(
            robot=robot,
            object_name=normalized_object_name,
            task_goal=normalized_task_goal,
            requirements=normalized_requirements,
            preferences=normalized_preferences,
            stage_intents=stage_intents,
        )
        if intent_error is not None:
            return intent_error

        planner_value = getattr(self.client, "plan_mtc_compound_task", None)
        if not callable(planner_value):
            return self._mtc_compound_task_failed_result(
                robot=robot,
                object_name=normalized_object_name,
                task_goal=normalized_task_goal,
                requirements=normalized_requirements,
                preferences=normalized_preferences,
                stage_intents=stage_intents,
                payload={
                    "failed_stage": "mtc_service_unavailable",
                    "blocker": "VizorClient.plan_mtc_compound_task is unavailable.",
                },
            )

        planner = cast(Callable[..., dict[str, Any] | None], planner_value)
        try:
            payload = planner(
                robot=robot,
                requirements=normalized_requirements,
                preferences=normalized_preferences,
                stage_intents=list(stage_intents) if stage_intents is not None else None,
                backend="mtc",
                timeout_s=timeout_s,
            )
        except TypeError as exc:
            payload = {
                "ok": False,
                "backend": "mtc",
                "requirements": normalized_requirements,
                "preferences": normalized_preferences,
                "stage_intents": list(stage_intents) if stage_intents is not None else None,
                "failed_stage": "mtc_client_contract_mismatch",
                "blocker": str(exc),
            }
        if not _mtc_compound_payload_has_solution(payload):
            return self._mtc_compound_task_failed_result(
                robot=robot,
                object_name=normalized_object_name,
                task_goal=normalized_task_goal,
                requirements=normalized_requirements,
                preferences=normalized_preferences,
                stage_intents=stage_intents,
                payload=payload,
            )
        assert isinstance(payload, dict)

        solution = self._build_mtc_compound_task_solution(
            robot=robot,
            object_name=normalized_object_name,
            task_goal=normalized_task_goal,
            requirements=normalized_requirements,
            preferences=normalized_preferences,
            stage_intents=stage_intents,
            payload=payload,
        )
        self._task_solutions[solution.task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def execute_task_solution(self, robot: str, task_solution_id: str, timeout_s: float = 60.0) -> dict[str, Any]:
        solution = self._task_solutions.get(task_solution_id)
        if solution is None:
            result = ToolResult.fail_result(
                robot=robot,
                tool="moveit_execute_task_solution",
                phase="pre_execute",
                status="unknown task solution id",
                message="Refusing to execute because this task solution was not planned by this MCP process",
                correction="Call moveit_plan_pick_task or moveit_plan_place_task, then retry with raw.task_solution_id.",
                checks=[VerificationCheck("task_solution_previously_planned", False, task_solution_id)],
                evidence=[Evidence("mcp_state", f"unknown task_solution_id: {task_solution_id}")],
                raw={"task_solution_id": task_solution_id},
            ).to_dict()
            result["error"] = "unknown_task_solution_id"
            result["retryable"] = False
            return result

        if solution.robot_name != robot:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_execute_task_solution",
                phase="pre_execute",
                status="robot mismatch",
                message="Refusing to execute a task solution planned for a different robot",
                correction=f"Retry with robot_name={solution.robot_name!r}.",
                checks=[VerificationCheck("task_solution_robot_matches", False, solution.robot_name)],
                evidence=[Evidence("mcp_state", f"task solution robot: {solution.robot_name}")],
                raw=solution.to_dict(),
            ).to_dict()

        self._task_solutions.pop(task_solution_id, None)
        if solution.created_from_tool in {"moveit_plan_compound_task", "moveit_plan_manipulation_task"}:
            contract_steps = _stored_task_solution_execution_contract(solution)
            if not contract_steps:
                return self._task_solution_executed_result(
                    solution=solution,
                    stages=[
                        TaskStage(
                            "execution_contract",
                            "execution_contract",
                            "failed",
                            [{"kind": "mcp_state", "summary": "missing supported execution contract"}],
                            {"execution_contract": solution.raw.get("execution_contract")},
                        )
                    ],
                    ok=False,
                    status="execution contract not executable",
                    message="Task solution has no supported typed execution contract.",
                )
            return self._execute_task_solution_contract(solution, contract_steps, timeout_s=timeout_s)

        executed_stages: list[TaskStage] = []
        for stage in solution.stages:
            ok, evidence, raw = self._execute_task_stage(
                solution=solution,
                stage=stage,
                timeout_s=timeout_s,
            )
            status = "executed" if ok else "failed"
            executed_stage = TaskStage(stage.name, stage.stage_type, status, evidence, raw)
            executed_stages.append(executed_stage)
            if not ok:
                return self._task_solution_executed_result(
                    solution=solution,
                    stages=executed_stages,
                    ok=False,
                    status=f"{stage.name} failed",
                    message=f"Task solution execution stopped at failed stage {stage.name}",
                )

        return self._task_solution_executed_result(
            solution=solution,
            stages=executed_stages,
            ok=True,
            status="task solution executed",
            message="Task solution stages executed in order with stage evidence",
        )

    def execute_plan(self, robot: str, name: str, timeout_s: float = 10.0) -> dict[str, Any]:
        planned = self._planned.get((robot, name))
        if planned is None:
            return ToolResult.fail_result(
                robot=robot,
                tool="execute_plan",
                phase="pre_execute",
                status="plan not verified",
                message="Refusing to execute because this plan was not verified by this MCP server",
                correction=UNVERIFIED_PLAN_CORRECTION,
                checks=[VerificationCheck("plan_previously_verified", False, name)],
                evidence=[Evidence("mcp_state", f"no verified plan record for: {name}")],
                raw={"plan_name": name},
            ).to_dict()

        if planned.get("can_execute") is not True:
            return ToolResult.fail_result(
                robot=robot,
                tool="execute_plan",
                phase="pre_execute",
                status="plan not executable",
                message="Refusing to execute because planning feedback did not mark this plan executable",
                correction=PLAN_NOT_EXECUTABLE_CORRECTION,
                checks=[
                    VerificationCheck("plan_previously_verified", True, name),
                    VerificationCheck("planned_can_execute", False, str(planned.get("status", "unknown"))),
                ],
                evidence=[Evidence("mcp_state", f"plan was observed but can_execute was false: {name}")],
                raw=dict(planned),
            ).to_dict()

        feedback = self.client.execute_plan(robot=robot, name=name, timeout_s=timeout_s)
        expected = planned.get("final_joint_positions")

        if feedback.physical_mode is not False:
            return ToolResult.fail_result(
                robot=robot,
                tool="execute_plan",
                phase="pre_execute",
                status=feedback.status,
                message="Refusing to execute because physical mode is enabled or could not be verified as false",
                correction=PHYSICAL_MODE_CORRECTION,
                checks=[
                    VerificationCheck("plan_previously_verified", True, name),
                    VerificationCheck("planned_can_execute", True, str(planned.get("status", "unknown"))),
                    VerificationCheck("physical_mode_safe", False, str(feedback.physical_mode)),
                    VerificationCheck("execute_command_published", feedback.command_published, str(feedback.command_published)),
                ],
                evidence=[Evidence("ros_param", str(feedback.physical_mode), path="/vizor_robot_control/physical")],
                raw={
                    "plan_name": name,
                    "physical_mode": feedback.physical_mode,
                    "command_published": feedback.command_published,
                    "expected_joint_state": expected,
                },
            ).to_dict()

        observed = feedback.observed_joint_state
        state_observed = observed is not None and len(observed) > 0
        tolerance = float(getattr(self.client, "joint_tolerance", 1e-3))
        positions_match = feedback.final_positions_match
        match_details = f"expected={expected}, observed={observed}, tolerance={tolerance}"
        if feedback.expected_joint_names is not None or feedback.observed_joint_names is not None:
            match_details += (
                f", expected_names={feedback.expected_joint_names}, "
                f"observed_names={feedback.observed_joint_names}"
            )
        checks = [
            VerificationCheck("plan_previously_verified", True, name),
            VerificationCheck("planned_can_execute", True, str(planned.get("status", "unknown"))),
            VerificationCheck("physical_mode_safe", True, str(feedback.physical_mode)),
            VerificationCheck("execute_command_published", feedback.command_published, str(feedback.command_published)),
            VerificationCheck("fake_controller_state_observed", state_observed, str(observed)),
            VerificationCheck("final_joint_positions_match", positions_match, match_details),
        ]
        evidence = [
            Evidence("ros_param", "false", path="/vizor_robot_control/physical"),
            Evidence("ros_topic", str(observed), topic=f"/{robot}/move_group/fake_controller_joint_states"),
        ]
        raw = {
            "plan_name": name,
            "physical_mode": feedback.physical_mode,
            "command_published": feedback.command_published,
            "expected_joint_state": expected,
            "observed_joint_state": observed,
            "joint_tolerance": tolerance,
        }
        if feedback.expected_joint_names is not None:
            raw["expected_joint_names"] = feedback.expected_joint_names
        if feedback.observed_joint_names is not None:
            raw["observed_joint_names"] = feedback.observed_joint_names

        if feedback.command_published and state_observed and positions_match:
            if planned.get("workflow_kind") == "pick":
                pick_ok, pick_checks, pick_evidence, pick_raw = self._complete_pick_object(
                    robot=robot,
                    planned=planned,
                    timeout_s=timeout_s,
                )
                checks.extend(pick_checks)
                evidence.extend(pick_evidence)
                raw["pick"] = pick_raw
                if not pick_ok:
                    return ToolResult.fail_result(
                        robot=robot,
                        tool="execute_plan",
                        phase="executed",
                        status="pick grasp or lift unverified",
                        message="Pick approach executed, but gripper close, object attach, or lift was not verified",
                        correction="Check gripper feedback and planning-scene attachment state, then retry pick before claiming the object was picked.",
                        checks=checks,
                        evidence=evidence,
                        raw=raw,
                    ).to_dict()
            if planned.get("workflow_kind") == "place":
                release_ok, release_checks, release_evidence, release_raw = self._release_place_object(
                    robot=robot,
                    planned=planned,
                    timeout_s=timeout_s,
                )
                checks.extend(release_checks)
                evidence.extend(release_evidence)
                raw["release"] = release_raw
                if not release_ok:
                    return ToolResult.fail_result(
                        robot=robot,
                        tool="execute_plan",
                        phase="executed",
                        status="place release unverified",
                        message="Place motion executed, but release or planning-scene detach was not verified",
                        correction="Check gripper feedback and planning-scene attachment state, then retry release before claiming the object was placed.",
                        checks=checks,
                        evidence=evidence,
                        raw=raw,
                    ).to_dict()

            status = "final joint state matched"
            message = "Execute command published and fake controller joint state matched the planned final positions"
            if planned.get("workflow_kind") == "pick":
                status = "pick motion, grasp, attach, and lift verified"
                message = "Pick approach executed, gripper closed, object attached, lift executed, and held object was verified"
            elif planned.get("workflow_kind") == "place":
                status = "place motion and release verified"
                message = "Execute command published, fake controller joint state matched, and place release was verified"
            return ToolResult.pass_result(
                robot=robot,
                tool="execute_plan",
                phase="executed",
                status=status,
                message=message,
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()

        return ToolResult.fail_result(
            robot=robot,
            tool="execute_plan",
            phase="executed",
            status="execution unverified",
            message="Execution could not be verified against fake controller joint state feedback",
            correction=EXECUTION_UNVERIFIED_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def explain_motion_failure(
        self,
        robot: str,
        failed_tool_name: str,
        failed_tool_result: dict[str, Any] | str,
        *,
        failed_tool_arguments: dict[str, Any] | None = None,
        user_intent: str | None = None,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        del timeout_s
        category, correction, retryable, suggested_next_tool = _diagnose_motion_failure(
            failed_tool_name=failed_tool_name,
            failed_tool_result=failed_tool_result,
            failed_tool_arguments=failed_tool_arguments,
            user_intent=user_intent,
        )
        result = ToolResult.pass_result(
            robot=robot,
            tool="moveit_explain_motion_failure",
            phase="diagnosed",
            status=category,
            message=f"Motion failure diagnosed as {category}.",
            checks=[
                VerificationCheck("failed_tool_name_provided", bool(failed_tool_name), failed_tool_name),
                VerificationCheck("failed_result_provided", True, _failure_text(failed_tool_result)),
            ],
            evidence=[Evidence("mcp_state", f"diagnosed failed tool: {failed_tool_name}")],
            raw={
                "category": category,
                "failed_tool_name": failed_tool_name,
                "failed_tool_arguments": failed_tool_arguments or {},
                "failed_tool_result": failed_tool_result,
                "user_intent": user_intent,
                "retryable": retryable,
                "suggested_next_tool": suggested_next_tool,
                "correction": correction,
            },
            can_execute=False,
        ).to_dict()
        result["correction"] = correction
        result["retryable"] = retryable
        result["suggested_next_tool"] = suggested_next_tool
        return result

    def verify_attached_object(self, robot: str, object_name: str, timeout_s: float = 2.0) -> dict[str, Any]:
        held_object = self.gripper.attached_object(robot)
        feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        object_context = feedback.object_context if feedback.ok else None
        scene_state = object_context.get("state") if isinstance(object_context, dict) else None
        attached_to = object_context.get("attached_to") if isinstance(object_context, dict) else None
        scene_attached = scene_state == "attached"
        attached_to_gripper = isinstance(attached_to, str) and attached_to in {"tool0", "ee_link", "wrist_3_link"}
        mcp_holds_object = held_object == object_name
        mcp_state_not_conflicting = held_object in {None, object_name}
        moves_with_gripper = scene_attached and attached_to_gripper and mcp_state_not_conflicting
        checks = [
            VerificationCheck("planning_scene_object_attached", scene_attached, str(scene_state)),
            VerificationCheck("attached_to_gripper_link", attached_to_gripper, str(attached_to)),
            VerificationCheck("mcp_gripper_state_not_conflicting", mcp_state_not_conflicting, str(held_object)),
        ]
        evidence = [
            Evidence("mcp_state", f"attached_object={held_object}"),
            Evidence("ros_service", feedback.message, path=feedback.source),
        ]
        raw = {
            "object_name": object_name,
            "mcp_attached_object": held_object,
            "mcp_gripper_holds_object": mcp_holds_object,
            "planning_scene_state": scene_state,
            "attached_to": attached_to,
            "moves_with_gripper": moves_with_gripper,
            "object": object_context,
            "available_objects": feedback.available_objects,
        }
        if moves_with_gripper:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_verify_attached_object",
                phase="verified",
                status="attached to gripper",
                message="Planning-scene object is attached to the gripper link and will move with the gripper.",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_verify_attached_object",
            phase="verified",
            status="object not attached to gripper",
            message="The object is not proven attached to the gripper.",
            correction=ATTACHED_OBJECT_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def release_object(
        self,
        robot: str,
        object_name: str,
        *,
        object_pose: dict[str, Any] | None,
        verified_gripper_open: bool = False,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        if not verified_gripper_open:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_release_object",
                phase="pre_execute",
                status="verified gripper open required",
                message="Refusing to detach the object before verified gripper open evidence is present",
                correction=RELEASE_OBJECT_CORRECTION,
                checks=[VerificationCheck("verified_gripper_open", False, str(verified_gripper_open))],
                evidence=[Evidence("mcp_state", f"attached_object={self.gripper.attached_object(robot)}")],
                raw={"object_name": object_name, "verified_gripper_open": verified_gripper_open},
            ).to_dict()
        if not isinstance(object_pose, dict):
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_release_object",
                phase="pre_execute",
                status="release object pose required",
                message="Refusing to detach the object without an explicit released object pose",
                correction=RELEASE_OBJECT_CORRECTION,
                checks=[VerificationCheck("released_object_pose_present", False, str(object_pose))],
                evidence=[Evidence("mcp_state", "missing released object pose")],
                raw={"object_name": object_name, "verified_gripper_open": verified_gripper_open},
            ).to_dict()
        try:
            released_pose = Pose.from_input(object_pose)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_release_object",
                phase="pre_execute",
                status="invalid release object pose",
                message="Refusing to detach the object because the released object pose is invalid",
                correction=POSE_INPUT_CORRECTION,
                checks=[VerificationCheck("released_object_pose_valid", False, str(exc))],
                evidence=[Evidence("mcp_state", "invalid released object pose")],
                raw={"object_name": object_name, "object_pose": object_pose},
            ).to_dict()

        feedback = self.client.detach_object(
            robot=robot,
            object_name=object_name,
            object_pose=released_pose,
            timeout_s=timeout_s,
        )
        if feedback.ok:
            self.gripper.set_state(robot, "open")
        checks = [
            VerificationCheck("verified_gripper_open", True, str(verified_gripper_open)),
            VerificationCheck("planning_scene_object_released", bool(feedback.ok), feedback.status),
        ]
        evidence = [
            Evidence("mcp_state", "gripper=open"),
            Evidence("ros_service", feedback.message, path=feedback.source),
        ]
        raw = {
            "object_name": object_name,
            "verified_gripper_open": verified_gripper_open,
            "gripper_opened": True,
            "planning_scene_released": bool(feedback.ok),
            "planning_scene_state": "free" if feedback.ok else "attached",
            "scene_update_published": feedback.scene_update_published,
            "released_object_pose": object_pose,
        }
        if feedback.ok:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_release_object",
                phase="executed",
                status="object released",
                message="Verified gripper open evidence was present and the object was detached into the planning scene.",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_release_object",
            phase="executed",
            status="release unverified",
            message="The object was not proven released into the planning scene.",
            correction="Check planning-scene attachment state, then retry release before claiming the object was placed.",
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def verify_released_object(self, robot: str, object_name: str, timeout_s: float = 2.0) -> dict[str, Any]:
        feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
        object_context = feedback.object_context if feedback.ok else None
        scene_state = object_context.get("state") if isinstance(object_context, dict) else None
        held_object = self.gripper.attached_object(robot)
        released = scene_state == "free" and held_object is None
        checks = [
            VerificationCheck("planning_scene_object_free", scene_state == "free", str(scene_state)),
            VerificationCheck("mcp_gripper_holds_no_object", held_object is None, str(held_object)),
        ]
        evidence = [
            Evidence("mcp_state", f"attached_object={held_object}"),
            Evidence("ros_service", feedback.message, path=feedback.source),
        ]
        raw = {
            "object_name": object_name,
            "mcp_attached_object": held_object,
            "mcp_gripper_holds_object": held_object == object_name,
            "planning_scene_state": scene_state,
            "released": released,
            "object": object_context,
            "available_objects": feedback.available_objects,
        }
        if released:
            return ToolResult.pass_result(
                robot=robot,
                tool="moveit_verify_released_object",
                phase="verified",
                status="object released",
                message="Planning-scene object is free and MCP gripper state holds no object.",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_verify_released_object",
            phase="verified",
            status="object release unverified",
            message="The object is not proven released from the gripper.",
            correction="Check gripper feedback and planning-scene attachment state, then retry release verification.",
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def open_gripper(self, robot: str, timeout_s: float = 5.0) -> dict[str, Any]:
        feedback = self.client.command_gripper(robot=robot, state="open", timeout_s=timeout_s)
        if not feedback.ok:
            return self._gripper_command_failed_result(robot=robot, tool="open_gripper", state="open", feedback=feedback)
        state = self.gripper.set_state(robot, "open")
        return ToolResult.pass_result(
            robot=robot,
            tool="open_gripper",
            phase="gripper",
            status="open",
            message="Gripper open command completed through the Robotiq action server",
            checks=_gripper_checks(feedback),
            evidence=[*_gripper_evidence(feedback), Evidence("mcp_state", state)],
            raw=_gripper_raw(feedback, state, None),
            can_execute=False,
        ).to_dict()

    def close_gripper(self, robot: str, timeout_s: float = 5.0) -> dict[str, Any]:
        feedback = self.client.command_gripper(robot=robot, state="closed", timeout_s=timeout_s)
        if not feedback.ok:
            return self._gripper_command_failed_result(robot=robot, tool="close_gripper", state="closed", feedback=feedback)
        state = self.gripper.set_state(robot, "closed")
        attached_object = self.gripper.attached_object(robot)
        return ToolResult.pass_result(
            robot=robot,
            tool="close_gripper",
            phase="gripper",
            status="closed",
            message="Gripper close command completed through the Robotiq action server",
            checks=_gripper_checks(feedback),
            evidence=[*_gripper_evidence(feedback), Evidence("mcp_state", state)],
            raw=_gripper_raw(feedback, state, attached_object),
            can_execute=False,
        ).to_dict()

    def _gripper_command_failed_result(
        self,
        *,
        robot: str,
        tool: str,
        state: str,
        feedback: Any,
    ) -> dict[str, Any]:
        return ToolResult.fail_result(
            robot=robot,
            tool=tool,
            phase="gripper",
            status=f"{state} unverified",
            message=f"Gripper {state} command was not verified through the Robotiq action server",
            correction=GRIPPER_COMMAND_CORRECTION,
            checks=_gripper_checks(feedback),
            evidence=[*_gripper_evidence(feedback), Evidence("mcp_state", self.gripper.get_state(robot))],
            raw=_gripper_raw(feedback, self.gripper.get_state(robot), self.gripper.attached_object(robot)),
        ).to_dict()

    def remove_scene_object(
        self,
        robot: str,
        object_name: str,
        *,
        timeout_s: float = 2.0,
    ) -> dict[str, Any]:
        feedback = self.client.remove_scene_object(
            robot=robot,
            object_name=object_name,
            timeout_s=timeout_s,
        )
        if not feedback.ok:
            correction = (
                REMOVE_ATTACHED_OBJECT_CORRECTION
                if feedback.status == "object attached"
                else OBJECT_NOT_FOUND_CORRECTION
                if feedback.status == "object not found"
                else PLANNING_SCENE_CORRECTION
            )
            return ToolResult.fail_result(
                robot=robot,
                tool="moveit_remove_scene_object",
                phase="scene_update",
                status=feedback.status,
                message=feedback.message,
                correction=correction,
                checks=_remove_scene_checks(feedback),
                evidence=[Evidence("ros_service", feedback.message, path=feedback.source)],
                raw=_remove_scene_raw(feedback),
            ).to_dict()

        return ToolResult.pass_result(
            robot=robot,
            tool="moveit_remove_scene_object",
            phase="scene_update",
            status=feedback.status,
            message=feedback.message,
            checks=_remove_scene_checks(feedback),
            evidence=[
                Evidence("ros_service", feedback.message, path=feedback.source),
                Evidence("mcp_state", f"removed:{object_name}"),
            ],
            raw=_remove_scene_raw(feedback),
            can_execute=False,
        ).to_dict()

    def attach_object(
        self,
        robot: str,
        object_name: str,
        *,
        verified_gripper_closed: bool = False,
    ) -> dict[str, Any]:
        state = self.gripper.get_state(robot)
        if verified_gripper_closed and state != "closed":
            state = self.gripper.set_state(robot, "closed")
        if state != "closed":
            state = self.gripper.get_state(robot)
            return ToolResult.fail_result(
                robot=robot,
                tool="attach_object",
                phase="gripper",
                status="gripper not closed",
                message="Refusing to attach object because simulated gripper is not closed",
                correction=GRIPPER_NOT_CLOSED_CORRECTION,
                checks=[VerificationCheck("gripper_closed", False, state)],
                evidence=[Evidence("mcp_state", state)],
                raw={"gripper_state": state, "attached_object": None},
            ).to_dict()

        feedback = self.client.attach_object(robot=robot, object_name=object_name)
        if not feedback.ok:
            return self._attach_scene_failed_result(robot=robot, object_name=object_name, state=state, feedback=feedback)

        self.gripper.attach(robot, object_name)
        state = self.gripper.get_state(robot)
        return ToolResult.pass_result(
            robot=robot,
            tool="attach_object",
            phase="gripper",
            status="attached",
            message="Object attached in MCP simulated gripper state and applied as a MoveIt attached collision object",
            checks=[
                VerificationCheck("gripper_closed", True, state),
                VerificationCheck("planning_scene_object_observed", True, object_name),
                VerificationCheck("planning_scene_diff_applied", feedback.scene_update_published, f"/{robot}/apply_planning_scene"),
                VerificationCheck("object_attached", True, object_name),
            ],
            evidence=[
                Evidence("ros_service", feedback.message, path=feedback.source),
                Evidence("ros_service", f"attached {object_name} to {feedback.link_name}", path=f"/{robot}/apply_planning_scene"),
                Evidence("mcp_state", f"attached:{object_name}"),
            ],
            raw={
                "gripper_state": state,
                "attached_object": object_name,
                "verified_gripper_closed": verified_gripper_closed,
                "scene_update_published": feedback.scene_update_published,
                "attached_to": feedback.link_name,
                "touch_links": feedback.touch_links,
                "planning_frame": feedback.planning_frame,
            },
            can_execute=False,
        ).to_dict()

    def _attach_scene_failed_result(
        self,
        *,
        robot: str,
        object_name: str,
        state: str,
        feedback: AttachSceneFeedback,
    ) -> dict[str, Any]:
        return ToolResult.fail_result(
            robot=robot,
            tool="attach_object",
            phase="gripper",
            status=feedback.status,
            message=feedback.message,
            correction=PLANNING_SCENE_ATTACH_CORRECTION,
            checks=[
                VerificationCheck("gripper_closed", True, state),
                VerificationCheck("planning_scene_object_observed", False, object_name),
                VerificationCheck("planning_scene_diff_applied", False, f"/{robot}/apply_planning_scene"),
            ],
            evidence=[Evidence("ros_service", feedback.message, path=feedback.source)],
            raw={
                "gripper_state": state,
                "attached_object": None,
                "scene_update_published": False,
                "attached_to": feedback.link_name,
                "planning_frame": feedback.planning_frame,
            },
        ).to_dict()

    def _pick_object_context_failed_result(
        self,
        *,
        robot: str,
        object_name: str,
        plan_name: str | None,
        feedback: Any,
        planning_strategy: str | None = None,
    ) -> dict[str, Any]:
        scene_observed = feedback.status != "planning scene unavailable"
        checks = [
            VerificationCheck("planning_scene_observed", scene_observed, feedback.status),
            VerificationCheck("object_observed", False, object_name),
        ]
        correction = PLANNING_SCENE_CORRECTION if not scene_observed else PICK_OBJECT_CORRECTION
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_pick",
            phase="pre_plan",
            status=feedback.status,
            message=feedback.message,
            correction=correction,
            checks=checks,
            evidence=[Evidence("ros_service", feedback.message, path=feedback.source)],
            raw={
                "plan_name": plan_name,
                "object_name": object_name,
                "available_objects": feedback.available_objects,
                "source": feedback.source,
                "planning_strategy": planning_strategy,
                "candidate_attempts": [],
            },
        ).to_dict()

    def _invalid_pick_workflow_result(
        self,
        *,
        robot: str,
        object_name: str,
        plan_name: str | None,
        object_context: dict[str, Any],
        exc: Exception,
        planning_strategy: str | None = None,
    ) -> dict[str, Any]:
        raw = {
            "plan_name": plan_name,
            "object_name": object_name,
            "planning_strategy": planning_strategy,
            "candidate_attempts": [],
        }
        correction = PICK_OBJECT_CORRECTION
        status = "invalid pick workflow"
        if isinstance(exc, PickPlanInputError):
            raw.update(exc.raw)
            correction = exc.correction
            status = exc.status
        elif object_context.get("grasp_faces"):
            raw["available_grasp_faces"] = [
                str(face.get("name"))
                for face in object_context["grasp_faces"]
                if isinstance(face, dict) and face.get("name")
            ]
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_pick",
            phase="pre_plan",
            status=status,
            message="Refusing to publish pick plan because pick workflow inputs are incomplete",
            correction=correction,
            checks=[
                VerificationCheck("object_context_observed", True, object_name),
                VerificationCheck("pick_workflow_valid", False, str(exc)),
            ],
            evidence=[Evidence("mcp_state", f"pick workflow invalid for object: {object_name}")],
            raw=raw,
        ).to_dict()

    def _resolve_single_pose_args(
        self,
        tool: str,
        robot: str,
        name: str | dict[str, Any] | None,
        position: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if position is None and isinstance(name, dict):
            return self._new_plan_name(robot, tool), name
        if position is None:
            raise ValueError("plan_free_motion requires a pose input")
        if name is None:
            return self._new_plan_name(robot, tool), position
        if not isinstance(name, str) or not name:
            raise ValueError("plan name must be a non-empty string")
        return name, position

    def _resolve_pose_list_args(
        self,
        tool: str,
        robot: str,
        name: str | list[dict[str, Any]] | None,
        positions: list[dict[str, Any]] | None,
    ) -> tuple[str, list[dict[str, Any]]]:
        if positions is None and isinstance(name, list):
            return self._new_plan_name(robot, tool), name
        if positions is None:
            raise ValueError("plan_cartesian_motion requires pose inputs")
        if name is None:
            return self._new_plan_name(robot, tool), positions
        if not isinstance(name, str) or not name:
            raise ValueError("plan name must be a non-empty string")
        return name, positions

    def _task_object_context_failed_result(
        self,
        *,
        robot: str,
        tool: str,
        object_name: str,
        feedback: Any,
    ) -> dict[str, Any]:
        scene_observed = feedback.status != "planning scene unavailable"
        correction = PLANNING_SCENE_CORRECTION if not scene_observed else PICK_OBJECT_CORRECTION
        return ToolResult.fail_result(
            robot=robot,
            tool=tool,
            phase="pre_plan",
            status=feedback.status,
            message=feedback.message,
            correction=correction,
            checks=[
                VerificationCheck("planning_scene_observed", scene_observed, feedback.status),
                VerificationCheck("object_observed", False, object_name),
            ],
            evidence=[Evidence("ros_service", feedback.message, path=feedback.source)],
            raw={"object_name": object_name, "available_objects": feedback.available_objects, "source": feedback.source},
        ).to_dict()

    def _plan_mtc_pick_task_result(
        self,
        *,
        robot: str,
        object_name: str,
        grasp_face: str | None,
        timeout_s: float,
        object_context: dict[str, Any],
        planning_frame: str | None,
        observed_at: datetime,
    ) -> dict[str, Any]:
        payload = self.client.plan_mtc_pick_task(
            robot=robot,
            object_name=object_name,
            grasp_face=grasp_face,
            timeout_s=timeout_s,
        )
        if not _mtc_payload_has_solution(payload):
            return self._mtc_pick_task_failed_result(
                robot=robot,
                object_name=object_name,
                grasp_face=grasp_face,
                payload=payload,
            )

        task_solution_id = str(payload["task_solution_id"])
        scene_snapshot = _safe_dict(payload.get("scene_snapshot"))
        scene_snapshot_id = str(scene_snapshot.get("id") or self._new_scene_snapshot_id())
        stages = [_mtc_task_stage(stage) for stage in _mtc_stage_summaries(payload)]
        stage_report = {
            "total": len(stages),
            "solved": sum(1 for stage in stages if stage.status == "solved"),
            "failed": sum(1 for stage in stages if stage.status == "failed"),
        }
        approval = ExecutionApproval(
            required=True,
            target_kind="task_solution",
            task_solution_id=task_solution_id,
            source_tool="moveit_plan_pick_task",
            object_name=object_name,
            expected_movement=f"pick {object_name}: MTC task solution execution",
            scene_snapshot_id=scene_snapshot_id,
        )
        evidence = [
            {"kind": "scene_snapshot", "id": scene_snapshot_id},
            {"kind": "stage_report", "count": len(stages)},
        ]
        evidence.extend(_safe_dict_list(payload.get("solution_evidence")))
        clearance = object_context.get("clearance")
        clearance_m = clearance.get("z_m") if isinstance(clearance, dict) else None
        selected_cost = _optional_float(payload.get("selected_cost"))
        object_pose_age_s = round((datetime.now(timezone.utc) - observed_at).total_seconds(), 3)
        solution = TaskSolution(
            task_solution_id=task_solution_id,
            task_kind="pick",
            backend="mtc",
            stages=stages,
            created_from_tool="moveit_plan_pick_task",
            object_name=object_name,
            robot_name=robot,
            scene_snapshot_id=scene_snapshot_id,
            stage_report=stage_report,
            approval=approval,
            evidence=evidence,
            planning_frame=planning_frame,
            object_pose_age_s=object_pose_age_s,
            solver=str(payload.get("solver") or "moveit_task_constructor"),
            selected_cost=selected_cost,
            clearance_m=clearance_m,
            candidate_attempts=_safe_dict_list(payload.get("candidate_attempts")),
            raw={
                "execution_target": "task_solution",
                "selected_grasp": _safe_dict(payload.get("selected_grasp")),
                "candidate_attempts": _safe_dict_list(payload.get("candidate_attempts")),
                "scene_snapshot": dict(scene_snapshot),
                "object": object_context,
            },
        )
        self._task_solutions[task_solution_id] = solution
        return self._task_solution_planned_result(solution)

    def _mtc_pick_task_failed_result(
        self,
        *,
        robot: str,
        object_name: str,
        grasp_face: str | None,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        failed_stage = str(payload.get("failed_stage") or "mtc_solution_incomplete")
        blocker = str(payload.get("blocker") or payload.get("message") or "MTC pick task did not return a solved task payload.")
        status = "mtc task planning failed"
        if failed_stage == "mtc_service_unavailable":
            status = "mtc task backend unavailable"
        raw = {
            "backend": "mtc",
            "robot_name": robot,
            "object_name": object_name,
            "grasp_face": grasp_face,
            "failed_stage": failed_stage,
            "blocker": blocker,
            "candidate_attempts": _safe_dict_list(payload.get("candidate_attempts")),
        }
        if "stage_summaries" in payload:
            raw["stage_summaries"] = [
                _mtc_stage_public_summary(stage)
                for stage in _safe_dict_list(payload.get("stage_summaries"))
            ]
        result = ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_pick_task",
            phase="planned",
            status=status,
            message=blocker,
            correction="Check the Vizor MTC task backend and retry after it returns a solved task payload.",
            checks=[VerificationCheck("mtc_task_solution_solved", False, failed_stage)],
            evidence=[Evidence("mcp_state", f"MTC pick task failed at {failed_stage}")],
            raw=raw,
        ).to_dict()
        result["failed_stage"] = failed_stage
        result["blocker"] = blocker
        result["retryable"] = True
        return result

    def _compound_requirements_error(
        self,
        *,
        robot: str,
        requirements: dict[str, Any] | None,
        legacy_object_name: str | None,
        legacy_task_goal: str | None,
    ) -> dict[str, Any] | None:
        if not isinstance(requirements, dict):
            return self._invalid_compound_requirements_result(
                robot=robot,
                status="invalid compound requirements",
                message="Refusing to plan a compound task without requirements",
                detail=str(requirements),
                requirements=None,
                preferences=None,
                stage_intents=None,
                object_name=legacy_object_name,
                task_goal=legacy_task_goal,
                missing="requirements",
            )

        goal = requirements.get("goal")
        object_name = requirements.get("object_name")
        if not isinstance(goal, str) or not goal.strip():
            return self._invalid_compound_requirements_result(
                robot=robot,
                status="invalid compound requirements",
                message="Refusing to plan a compound task without requirements.goal",
                detail=str(goal),
                requirements=requirements,
                preferences=None,
                stage_intents=None,
                object_name=object_name if isinstance(object_name, str) else legacy_object_name,
                task_goal=legacy_task_goal,
                missing="goal",
            )
        if not isinstance(object_name, str) or not object_name.strip():
            return self._invalid_compound_requirements_result(
                robot=robot,
                status="invalid compound requirements",
                message="Refusing to plan a compound task without requirements.object_name",
                detail=str(object_name),
                requirements=requirements,
                preferences=None,
                stage_intents=None,
                object_name=legacy_object_name,
                task_goal=goal,
                missing="object_name",
            )

        if goal in COMPOUND_TASK_TARGET_GOALS and not (
            isinstance(requirements.get("target_pose"), dict) or isinstance(requirements.get("target_position"), dict)
        ):
            return self._invalid_compound_requirements_result(
                robot=robot,
                status="invalid compound requirements",
                message="Refusing to plan this compound task without requirements.target_pose or requirements.target_position",
                detail=goal,
                requirements=requirements,
                preferences=None,
                stage_intents=None,
                object_name=object_name,
                task_goal=goal,
                missing="target_pose or target_position",
            )

        return None

    def _invalid_compound_requirements_result(
        self,
        *,
        robot: str,
        status: str,
        message: str,
        detail: str,
        requirements: dict[str, Any] | None,
        preferences: dict[str, Any] | None,
        stage_intents: Sequence[str] | None,
        object_name: Any,
        task_goal: Any,
        missing: str | None = None,
    ) -> dict[str, Any]:
        raw = {
            "backend": "mtc",
            "requirements": dict(requirements) if isinstance(requirements, dict) else requirements,
            "preferences": preferences,
            "stage_intents": list(stage_intents) if stage_intents is not None else None,
            "object_name": object_name,
            "task_goal": task_goal,
        }
        if missing is not None:
            raw["missing"] = missing
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_compound_task",
            phase="pre_plan",
            status=status,
            message=message,
            correction=COMPOUND_REQUIREMENTS_CORRECTION,
            checks=[VerificationCheck("compound_requirements_valid", False, detail)],
            evidence=[Evidence("mcp_state", message)],
            raw=raw,
        ).to_dict()

    def _compound_task_goal_error(
        self,
        *,
        robot: str,
        object_name: str,
        task_goal: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: Sequence[str] | None,
    ) -> dict[str, Any] | None:
        if task_goal in COMPOUND_TASK_GOALS:
            return None
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_compound_task",
            phase="pre_plan",
            status="unsupported task goal",
            message="Refusing to plan a compound task with an unsupported requirements.goal",
            correction=f"Use requirements.goal one of {sorted(COMPOUND_TASK_GOALS)}.",
            checks=[VerificationCheck("requirements_goal_supported", False, str(task_goal))],
            evidence=[Evidence("mcp_state", f"unsupported compound task goal: {task_goal}")],
            raw={
                "backend": "mtc",
                "requirements": requirements,
                "preferences": preferences,
                "stage_intents": list(stage_intents) if stage_intents is not None else None,
                "object_name": object_name,
                "task_goal": task_goal,
            },
        ).to_dict()

    def _compound_stage_intent_error(
        self,
        *,
        robot: str,
        object_name: str,
        task_goal: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: Sequence[str] | None,
    ) -> dict[str, Any] | None:
        invalid_intent = ""
        intents: list[Any] = []
        if stage_intents is not None:
            if isinstance(stage_intents, str):
                invalid_intent = stage_intents
            else:
                try:
                    intents = list(stage_intents)
                except TypeError:
                    intents = []
                    invalid_intent = str(stage_intents)
                if not all(isinstance(intent, str) for intent in intents):
                    invalid_intent = str(stage_intents)
            for intent in intents:
                if invalid_intent:
                    break
                normalized = intent.casefold()
                if intent not in COMPOUND_STAGE_INTENTS or any(fragment in normalized for fragment in COMPOUND_REJECTED_INTENT_FRAGMENTS):
                    invalid_intent = intent
                    break
        if not invalid_intent:
            return None

        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_compound_task",
            phase="pre_plan",
            status="unsupported stage intent",
            message="Refusing to plan compound task with unsupported or unsafe stage_intents",
            correction=COMPOUND_STAGE_INTENT_CORRECTION,
            checks=[VerificationCheck("stage_intents_supported", False, invalid_intent)],
            evidence=[Evidence("mcp_state", f"unsupported compound stage intent: {invalid_intent}")],
            raw={
                "backend": "mtc",
                "requirements": requirements,
                "preferences": preferences,
                "object_name": object_name,
                "task_goal": task_goal,
                "stage_intents": stage_intents if isinstance(stage_intents, str) else intents if stage_intents is not None else None,
                "unsupported_intent": invalid_intent,
            },
        ).to_dict()

    def _build_mtc_compound_task_solution(
        self,
        *,
        robot: str,
        object_name: str,
        task_goal: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: Sequence[str] | None,
        payload: dict[str, Any],
    ) -> TaskSolution:
        task_solution_id = str(payload["task_solution_id"])
        scene_snapshot = _safe_dict(payload.get("scene_snapshot"))
        scene_snapshot_id = str(scene_snapshot.get("id") or self._new_scene_snapshot_id())
        stages = [_mtc_task_stage(stage) for stage in _mtc_stage_summaries(payload)]
        execution_contract = _mtc_execution_contract(payload, object_name=object_name, scene_snapshot_id=scene_snapshot_id)
        stage_report = {
            "total": len(stages),
            "solved": sum(1 for stage in stages if stage.status == "solved"),
            "failed": sum(1 for stage in stages if stage.status == "failed"),
        }
        approval = ExecutionApproval(
            required=True,
            target_kind="task_solution",
            task_solution_id=task_solution_id,
            source_tool="moveit_plan_compound_task",
            object_name=object_name,
            expected_movement=f"{task_goal} {object_name}: MTC compound task solution execution",
            scene_snapshot_id=scene_snapshot_id,
        )
        evidence = [
            {"kind": "scene_snapshot", "id": scene_snapshot_id},
            {"kind": "stage_report", "count": len(stages)},
        ]
        evidence.extend(_safe_dict_list(payload.get("solution_evidence")))
        selected_cost = _optional_float(payload.get("selected_cost"))
        attempts = _safe_dict_list(payload.get("attempts"))
        candidate_count = payload.get("candidate_count")
        candidate_attempts = attempts or _safe_dict_list(payload.get("candidate_attempts"))
        preview = _safe_dict(payload.get("preview"))
        return TaskSolution(
            task_solution_id=task_solution_id,
            task_kind=task_goal,
            backend="mtc",
            stages=stages,
            created_from_tool="moveit_plan_compound_task",
            object_name=object_name,
            robot_name=robot,
            scene_snapshot_id=scene_snapshot_id,
            stage_report=stage_report,
            approval=approval,
            evidence=evidence,
            planning_frame=payload.get("planning_frame") if isinstance(payload.get("planning_frame"), str) else None,
            solver=str(payload.get("solver") or "moveit_task_constructor"),
            selected_cost=selected_cost,
            candidate_attempts=candidate_attempts if candidate_attempts else None,
            raw={
                "execution_target": "task_solution",
                "requirements": requirements,
                "preferences": preferences,
                "task_goal": task_goal,
                "object_name": object_name,
                "stage_intents": list(stage_intents) if stage_intents is not None else None,
                "candidate_count": candidate_count,
                "attempts": attempts,
                "scene_snapshot": scene_snapshot,
                "execution_contract": execution_contract,
                "preview": preview,
            },
        )

    def _mtc_compound_task_failed_result(
        self,
        *,
        robot: str,
        object_name: str,
        task_goal: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any] | None,
        stage_intents: Sequence[str] | None,
        payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        payload_dict = _safe_dict(payload)
        failed_stage = str(payload_dict.get("failed_stage") or "mtc_compound_solution_incomplete")
        blocker = str(payload_dict.get("blocker") or payload_dict.get("message") or "MTC compound task did not return a solved execution contract.")
        backend_message = payload_dict.get("message")
        message = backend_message if isinstance(backend_message, str) and backend_message else blocker
        backend_error = payload_dict.get("error")
        error = backend_error if isinstance(backend_error, str) and backend_error else None
        backend_correction = payload_dict.get("correction")
        correction = (
            backend_correction
            if isinstance(backend_correction, str) and backend_correction
            else "Check the Vizor MTC compound task backend and retry after it returns a solved execution_contract."
        )
        status = "mtc compound task planning failed"
        if failed_stage == "mtc_service_unavailable":
            status = "mtc compound task backend unavailable"
        raw = {
            "backend": "mtc",
            "robot_name": robot,
            "requirements": requirements,
            "preferences": preferences,
            "object_name": object_name,
            "task_goal": task_goal,
            "stage_intents": list(stage_intents) if stage_intents is not None else None,
            "failed_stage": failed_stage,
            "blocker": blocker,
        }
        if error is not None:
            raw["error"] = error
        if isinstance(backend_message, str) and backend_message:
            raw["message"] = backend_message
        if isinstance(backend_correction, str) and backend_correction:
            raw["correction"] = backend_correction
        if "preview" in payload_dict:
            raw["preview"] = _safe_dict(payload_dict.get("preview"))
        if "execution_contract" in payload_dict:
            raw["execution_contract"] = _mtc_non_executable_contract(payload_dict.get("execution_contract"))
        if "candidate_attempts" in payload_dict:
            raw["candidate_attempts"] = _safe_dict_list(payload_dict.get("candidate_attempts"))
        if "candidate_count" in payload_dict:
            raw["candidate_count"] = payload_dict.get("candidate_count")
        if "selected_cost" in payload_dict:
            raw["selected_cost"] = payload_dict.get("selected_cost")
        if "availability" in payload_dict:
            raw["availability"] = _safe_dict(payload_dict.get("availability"))
        if "stage_summaries" in payload_dict:
            raw["stage_summaries"] = [
                _mtc_stage_public_summary(stage)
                for stage in _safe_dict_list(payload_dict.get("stage_summaries"))
            ]
        result = ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_compound_task",
            phase="planned",
            status=status,
            message=message,
            correction=correction,
            checks=[VerificationCheck("mtc_compound_task_solution_solved", False, failed_stage)],
            evidence=[Evidence("mcp_state", f"MTC compound task failed at {failed_stage}")],
            raw=raw,
        ).to_dict()
        result["failed_stage"] = failed_stage
        result["blocker"] = blocker
        if error is not None:
            result["error"] = error
        result["retryable"] = True
        return result

    def _try_staged_hold_candidate(
        self,
        *,
        robot: str,
        object_name: str,
        attempt_index: int,
        workflow: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        params = _safe_dict(workflow.get("parameters"))
        selected_face = _safe_dict(workflow.get("selected_grasp_face"))
        return self._try_staged_motion_candidate(
            robot=robot,
            object_name=object_name,
            task_kind="hold",
            attempt_index=attempt_index,
            workflow=workflow,
            timeout_s=timeout_s,
            stage_specs=[
                {"name": "connect_to_pre_grasp", "planner": "free_motion", "waypoint_indexes": [0]},
                {"name": "approach_to_pre_grasp", "planner": "cartesian", "waypoint_indexes": [0, 1]},
                {"name": "post_grasp_lift", "planner": "cartesian", "waypoint_indexes": [1, 2]},
            ],
            attempt_metadata={
                "grasp_face": params.get("grasp_face") or selected_face.get("name"),
                "approach_distance_m": params.get("approach_distance_m"),
                "grasp_standoff_m": params.get("grasp_standoff_m"),
                "lift_distance_m": params.get("lift_distance_m"),
            },
        )

    def _try_staged_place_candidate(
        self,
        *,
        robot: str,
        object_name: str,
        task_kind: str,
        attempt_index: int,
        workflow: dict[str, Any],
        timeout_s: float,
    ) -> dict[str, Any]:
        params = _safe_dict(workflow.get("parameters"))
        return self._try_staged_motion_candidate(
            robot=robot,
            object_name=object_name,
            task_kind=task_kind,
            attempt_index=attempt_index,
            workflow=workflow,
            timeout_s=timeout_s,
            stage_specs=[
                {"name": "connect_to_place", "planner": "free_motion", "waypoint_indexes": [0]},
                {"name": "approach_place", "planner": "cartesian", "waypoint_indexes": [0, 1]},
                {"name": "retreat", "planner": "cartesian", "waypoint_indexes": [1, 2]},
            ],
            attempt_metadata={
                "approach_distance_m": params.get("approach_distance_m"),
                "place_standoff_m": params.get("place_standoff_m"),
                "retreat_distance_m": params.get("retreat_distance_m"),
            },
        )

    def _try_staged_motion_candidate(
        self,
        *,
        robot: str,
        object_name: str,
        task_kind: str,
        attempt_index: int,
        workflow: dict[str, Any],
        timeout_s: float,
        stage_specs: list[dict[str, Any]],
        attempt_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        attempt: dict[str, Any] = {
            "attempt_index": attempt_index,
            **attempt_metadata,
            "motion_stages": [],
            "selected": False,
        }
        waypoints_source = workflow.get("waypoints")
        if not isinstance(waypoints_source, list):
            attempt["status"] = "failed"
            attempt["failed_stage"] = "motion_stage_generation"
            attempt["failure_code"] = "invalid_motion_stage"
            return attempt

        for segment in stage_specs:
            segment_name = str(segment.get("name") or "motion_stage")
            planner = str(segment.get("planner") or "cartesian")
            indexes = segment.get("waypoint_indexes")
            if not isinstance(indexes, list):
                attempt["status"] = "failed"
                attempt["failed_stage"] = segment_name
                attempt["failure_code"] = "invalid_motion_stage"
                return attempt
            try:
                waypoints = [waypoints_source[index] for index in indexes]
                stage, planned = self._plan_staged_motion_stage(
                    robot=robot,
                    object_name=object_name,
                    task_kind=task_kind,
                    attempt_index=attempt_index,
                    segment_name=segment_name,
                    planner=planner,
                    waypoints=waypoints,
                    timeout_s=timeout_s,
                )
            except (IndexError, TypeError, ValueError):
                attempt["status"] = "failed"
                attempt["failed_stage"] = segment_name
                attempt["failure_code"] = "invalid_motion_stage"
                return attempt
            attempt["motion_stages"].append(stage)
            if planned:
                continue
            attempt["status"] = "failed"
            attempt["failed_stage"] = segment_name
            attempt["failure_code"] = "required_motion_stage_unplanned"
            return attempt

        attempt["status"] = "selected"
        attempt["selected"] = True
        return attempt

    def _plan_staged_motion_stage(
        self,
        *,
        robot: str,
        object_name: str,
        task_kind: str,
        attempt_index: int,
        segment_name: str,
        planner: str,
        waypoints: list[Any],
        timeout_s: float,
    ) -> tuple[dict[str, Any], bool]:
        if not waypoints:
            raise ValueError("staged motion requires at least one waypoint")
        plan_name = _staged_manipulation_plan_name(task_kind, object_name, attempt_index, segment_name)
        if planner == "free_motion":
            feedback = self.client.plan_free_motion(
                robot=robot,
                name=plan_name,
                pose=Pose.from_input(waypoints[-1]),
                timeout_s=timeout_s,
            )
        elif planner == "cartesian":
            feedback = self.client.plan_cartesian_motion(
                robot=robot,
                name=plan_name,
                poses=[Pose.from_input(waypoint) for waypoint in waypoints],
                timeout_s=timeout_s,
            )
        else:
            raise ValueError(f"unsupported staged planner: {planner}")

        planned = feedback.status in SUCCESS_STATUSES and feedback.trajectory_points > 0 and feedback.can_execute
        stage = {
            "name": segment_name,
            "plan_name": plan_name,
            "planner": planner,
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "preview_evidence": {
                "kind": "AgentPath",
                "stage": segment_name,
                "plan_name": plan_name,
                "planner": planner,
                "trajectory_points": feedback.trajectory_points,
                "topic": f"/{robot}/request/planned_path",
            },
        }
        if planned:
            self._planned[(robot, plan_name)] = {
                "plan_name": plan_name,
                "can_execute": True,
                "status": feedback.status,
                "planner": planner,
                "final_joint_positions": feedback.final_joint_positions,
            }
        else:
            self._planned.pop((robot, plan_name), None)
        return stage, planned

    @staticmethod
    def _invalid_manipulation_requirements_result(
        *,
        robot: str,
        requirements: Any,
        preferences: Any,
        detail: str,
        missing: str,
    ) -> dict[str, Any]:
        return ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_manipulation_task",
            phase="pre_plan",
            status="invalid manipulation requirements",
            message="Refusing to plan a manipulation task with incomplete requirements",
            correction=MANIPULATION_REQUIREMENTS_CORRECTION,
            checks=[VerificationCheck("manipulation_requirements_valid", False, detail)],
            evidence=[Evidence("mcp_state", f"invalid manipulation requirements: {missing}")],
            raw={
                "backend": "staged_moveit",
                "requirements": requirements,
                "preferences": preferences,
                "missing": missing,
            },
        ).to_dict()

    @staticmethod
    def _manipulation_task_failed_result(
        *,
        robot: str,
        object_name: str,
        requirements: dict[str, Any],
        preferences: dict[str, Any],
        failed_stage: str,
        candidate_attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        what_was_proven = [
            {
                "candidate": attempt.get("attempt_index"),
                "motion_stages": [
                    {
                        "name": stage.get("name"),
                        "planner": stage.get("planner"),
                        "plan_name": stage.get("plan_name"),
                        "trajectory_points": stage.get("trajectory_points"),
                        "can_execute": stage.get("can_execute"),
                    }
                    for stage in _safe_dict_list(attempt.get("motion_stages"))
                ],
            }
            for attempt in candidate_attempts
        ]
        what_is_uncertain = [
            "No complete hold task was proven because at least one required motion stage lacked executable trajectory preview.",
            "The object was not attached or lifted; planning stopped before any execution approval was issued.",
        ]
        suggested_next_action = (
            "Inspect the failed candidate stage, adjust the grasp face or object pose, then retry moveit_plan_manipulation_task."
        )
        raw = {
            "backend": "staged_moveit",
            "requirements": requirements,
            "preferences": preferences,
            "object_name": object_name,
            "failed_stage": failed_stage,
            "failure_code": "required_motion_stage_unplanned",
            "tried_candidates": len(candidate_attempts),
            "candidate_attempts": candidate_attempts,
            "what_was_proven": what_was_proven,
            "what_is_uncertain": what_is_uncertain,
            "suggested_next_action": suggested_next_action,
        }
        result = ToolResult.fail_result(
            robot=robot,
            tool="moveit_plan_manipulation_task",
            phase="planned",
            status="staged manipulation task planning failed",
            message=f"Required manipulation stage {failed_stage} could not be planned with preview evidence.",
            correction=suggested_next_action,
            checks=[VerificationCheck("required_motion_stages_planned", False, failed_stage)],
            evidence=[Evidence("mcp_state", f"staged manipulation failed at {failed_stage}")],
            raw=raw,
        ).to_dict()
        result["failed_stage"] = failed_stage
        result["failure_code"] = "required_motion_stage_unplanned"
        result["tried_candidates"] = len(candidate_attempts)
        result["what_was_proven"] = what_was_proven
        result["what_is_uncertain"] = what_is_uncertain
        result["suggested_next_action"] = suggested_next_action
        result["retryable"] = True
        return result

    @staticmethod
    def _invalid_task_workflow_result(
        *,
        robot: str,
        tool: str,
        object_name: str,
        exc: Exception,
        correction: str,
    ) -> dict[str, Any]:
        raw = dict(getattr(exc, "raw", {}))
        raw["object_name"] = object_name
        return ToolResult.fail_result(
            robot=robot,
            tool=tool,
            phase="pre_plan",
            status=str(exc),
            message="Refusing to create task solution because workflow inputs are incomplete",
            correction=correction,
            checks=[VerificationCheck("task_workflow_derived", False, str(exc))],
            evidence=[Evidence("mcp_state", f"invalid task workflow for object: {object_name}")],
            raw=raw,
        ).to_dict()

    def _build_task_solution(
        self,
        *,
        robot: str,
        object_name: str,
        task_kind: str,
        created_from_tool: str,
        planning_frame: str | None,
        object_context: dict[str, Any],
        observed_at: datetime,
        stages: list[TaskStage],
        expected_movement: str,
        raw: dict[str, Any],
        candidate_attempts: int | list[dict[str, Any]] | None = None,
        backend: str = "emulated",
        solver: str = "emulated_mtc_stages",
        selected_cost: float | None = None,
    ) -> TaskSolution:
        self._task_solution_sequence += 1
        task_solution_id = f"{task_kind}_task_{_slug(object_name)}_{self._task_solution_sequence:03d}"
        scene_snapshot_id = self._new_scene_snapshot_id()
        stage_report = {
            "total": len(stages),
            "solved": sum(1 for stage in stages if stage.status == "solved"),
            "failed": sum(1 for stage in stages if stage.status == "failed"),
        }
        approval = ExecutionApproval(
            required=True,
            target_kind="task_solution",
            task_solution_id=task_solution_id,
            source_tool=created_from_tool,
            object_name=object_name,
            expected_movement=expected_movement,
            scene_snapshot_id=scene_snapshot_id,
        )
        evidence = [
            {"kind": "scene_snapshot", "id": scene_snapshot_id},
            {"kind": "stage_report", "count": len(stages)},
        ]
        clearance = object_context.get("clearance")
        clearance_m = clearance.get("z_m") if isinstance(clearance, dict) else None
        object_pose_age_s = round((datetime.now(timezone.utc) - observed_at).total_seconds(), 3)
        return TaskSolution(
            task_solution_id=task_solution_id,
            task_kind=task_kind,
            backend=backend,
            stages=stages,
            created_from_tool=created_from_tool,
            object_name=object_name,
            robot_name=robot,
            scene_snapshot_id=scene_snapshot_id,
            stage_report=stage_report,
            approval=approval,
            evidence=evidence,
            planning_frame=planning_frame,
            object_pose_age_s=object_pose_age_s,
            solver=solver,
            selected_cost=round(1.0 + len(stages) * 0.06, 3) if selected_cost is None else selected_cost,
            clearance_m=clearance_m,
            candidate_attempts=1 if candidate_attempts is None else candidate_attempts,
            raw=raw,
        )

    def _task_solution_planned_result(self, solution: TaskSolution) -> dict[str, Any]:
        result = ToolResult.pass_result(
            robot=solution.robot_name,
            tool=solution.created_from_tool,
            phase="planned",
            status="task solution solved",
            message="Task solution solved with stage evidence; execution requires explicit approval",
            checks=[
                VerificationCheck("task_solution_solved", True, solution.task_solution_id),
                VerificationCheck("all_stages_solved", True, str(solution.stage_report)),
            ],
            evidence=[
                Evidence("mcp_state", f"task_solution_id={solution.task_solution_id}"),
                Evidence("mcp_state", f"scene_snapshot_id={solution.scene_snapshot_id}"),
            ],
            raw=solution.to_dict(),
            can_execute=True,
        ).to_dict()
        result["feedback"]["execution_target"] = "task_solution"
        return result

    def _execute_task_stage(
        self,
        *,
        solution: TaskSolution,
        stage: TaskStage,
        timeout_s: float,
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        robot = solution.robot_name
        object_name = solution.object_name
        if stage.stage_type == "observation":
            return True, [{"kind": "emulated_stage", "name": stage.name}], {"backend": solution.backend}
        if stage.stage_type == "motion_plan":
            return self._execute_task_motion_stage(
                solution=solution,
                stage=stage,
                timeout_s=timeout_s,
            )
        if stage.name == "close_gripper":
            result = self.close_gripper(robot, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_close_gripper"}], result
        if stage.name == "attach_object":
            result = self.attach_object(robot, object_name)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_attach_object"}], result
        if stage.name == "verify_attached_object":
            result = self.verify_attached_object(robot, object_name, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_verify_attached_object"}], result
        if stage.name == "open_gripper":
            result = self.open_gripper(robot, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_open_gripper"}], result
        if stage.name == "detach_object":
            release = solution.raw.get("release_after_execute")
            object_pose = release.get("object_pose") if isinstance(release, dict) else None
            if not isinstance(object_pose, dict):
                return False, [{"kind": "mcp_state", "summary": "missing release pose"}], {"release_after_execute": release}
            feedback = self.client.detach_object(
                robot=robot,
                object_name=object_name,
                object_pose=Pose.from_input(object_pose),
                timeout_s=timeout_s,
            )
            return (
                bool(feedback.ok),
                [{"kind": "ros_service", "path": feedback.source}],
                {"status": feedback.status, "scene_update_published": feedback.scene_update_published},
            )
        if stage.name == "verify_released_object":
            feedback = self.client.get_object_context(robot=robot, object_name=object_name, timeout_s=timeout_s)
            state = feedback.object_context.get("state") if feedback.ok and feedback.object_context else None
            released = state == "free" and self.gripper.attached_object(robot) is None
            return (
                released,
                [{"kind": "release_check", "object_name": object_name}],
                {"planning_scene_state": state, "mcp_attached_object": self.gripper.attached_object(robot)},
            )
        return False, [{"kind": "mcp_state", "summary": f"unknown stage {stage.name}"}], {"stage": stage.name}

    def _execute_task_solution_contract(
        self,
        solution: TaskSolution,
        steps: list[dict[str, Any]],
        *,
        timeout_s: float,
    ) -> dict[str, Any]:
        executed_stages: list[TaskStage] = []
        contract_state = {
            "verified_gripper_closed": False,
            "verified_gripper_open": False,
        }
        for step in steps:
            ok, evidence, raw = self._execute_task_contract_step(
                solution=solution,
                step=step,
                timeout_s=timeout_s,
                contract_state=contract_state,
            )
            status = "executed" if ok else "failed"
            stage = TaskStage(
                _mtc_contract_step_name(step),
                "execution_contract",
                status,
                evidence,
                raw,
            )
            executed_stages.append(stage)
            if not ok:
                return self._task_solution_executed_result(
                    solution=solution,
                    stages=executed_stages,
                    ok=False,
                    status=f"{stage.name} failed",
                    message=f"Task solution execution stopped at failed contract step {stage.name}",
                )

        return self._task_solution_executed_result(
            solution=solution,
            stages=executed_stages,
            ok=True,
            status="task solution executed",
            message="Task solution execution_contract steps executed in order with evidence",
        )

    def _execute_task_contract_step(
        self,
        *,
        solution: TaskSolution,
        step: dict[str, Any],
        timeout_s: float,
        contract_state: dict[str, bool],
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        robot = solution.robot_name
        arguments = _safe_dict(step.get("arguments"))
        object_name = _mtc_contract_step_object_name(step, solution.object_name)
        handler = _mtc_contract_canonical_handler(str(step.get("handler") or ""))
        if handler == "observe_current_state":
            return True, [{"kind": "execution_contract", "handler": "observe_current_state"}], {"step": step}
        if handler == "motion":
            plan_name = _mtc_contract_step_plan_name(step)
            if plan_name is None:
                return (
                    False,
                    [{"kind": "execution_contract", "handler": "motion"}],
                    {"step": step, "error": "missing executable plan_name"},
                )
            result = self.execute_plan(robot, plan_name, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_execute_plan"}], result
        if handler == "close_gripper":
            result = self.close_gripper(robot, timeout_s=timeout_s)
            if result.get("ok") is True:
                contract_state["verified_gripper_closed"] = True
                contract_state["verified_gripper_open"] = False
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_close_gripper"}], result
        if handler == "open_gripper":
            result = self.open_gripper(robot, timeout_s=timeout_s)
            if result.get("ok") is True:
                contract_state["verified_gripper_open"] = True
                contract_state["verified_gripper_closed"] = False
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_open_gripper"}], result
        if handler == "attach_object":
            verified_gripper_closed = bool(arguments.get("verified_gripper_closed") or contract_state["verified_gripper_closed"])
            result = self.attach_object(robot, object_name, verified_gripper_closed=verified_gripper_closed)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_attach_object"}], result
        if handler == "release_object":
            object_pose = arguments.get("object_pose") if isinstance(arguments.get("object_pose"), dict) else step.get("object_pose")
            verified_gripper_open = bool(arguments.get("verified_gripper_open") or contract_state["verified_gripper_open"])
            result = self.release_object(
                robot,
                object_name,
                object_pose=object_pose if isinstance(object_pose, dict) else None,
                verified_gripper_open=verified_gripper_open,
                timeout_s=timeout_s,
            )
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_release_object"}], result
        if handler == "verify_attached_object":
            result = self.verify_attached_object(robot, object_name, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_verify_attached_object"}], result
        if handler == "verify_released_object":
            result = self.verify_released_object(robot, object_name, timeout_s=timeout_s)
            return bool(result.get("ok")), [{"kind": "tool_result", "tool": "moveit_verify_released_object"}], result
        return False, [{"kind": "execution_contract", "handler": handler}], {"step": step, "error": "unsupported handler"}

    def _execute_task_motion_stage(
        self,
        *,
        solution: TaskSolution,
        stage: TaskStage,
        timeout_s: float,
    ) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
        plan_name = f"{solution.task_solution_id}__{stage.name}"
        try:
            planner, waypoints = self._task_motion_stage_waypoints(solution, stage)
        except (KeyError, TypeError, ValueError) as exc:
            return (
                False,
                [{"kind": "mcp_state", "summary": f"invalid motion stage {stage.name}"}],
                {"stage": stage.name, "error": str(exc)},
            )

        if planner == "free_motion":
            feedback = self.client.plan_free_motion(
                robot=solution.robot_name,
                name=plan_name,
                pose=Pose.from_input(waypoints[0]),
                timeout_s=timeout_s,
            )
        else:
            feedback = self.client.plan_cartesian_motion(
                robot=solution.robot_name,
                name=plan_name,
                poses=[Pose.from_input(waypoint) for waypoint in waypoints],
                timeout_s=timeout_s,
            )

        plan_ok = feedback.status in SUCCESS_STATUSES and feedback.trajectory_points > 0 and feedback.can_execute
        plan_raw = {
            "plan_name": plan_name,
            "can_execute": feedback.can_execute,
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "final_joint_positions": feedback.final_joint_positions,
            "task_solution_id": solution.task_solution_id,
            "task_stage": stage.name,
            "planner": planner,
        }
        if not plan_ok:
            return (
                False,
                [{"kind": "ros_topic", "topic": f"/{solution.robot_name}/request/planned_path"}],
                {"planning": plan_raw},
            )

        self._planned[(solution.robot_name, plan_name)] = plan_raw
        execute_result = self.execute_plan(solution.robot_name, plan_name, timeout_s=timeout_s)
        executed = bool(execute_result.get("ok"))
        return (
            executed,
            [
                {"kind": "tool_result", "tool": "moveit_plan_cartesian_motion" if planner == "cartesian" else "moveit_plan_free_motion"},
                {"kind": "tool_result", "tool": "moveit_execute_plan"},
            ],
            {
                "planning": plan_raw,
                "execution": execute_result,
            },
        )

    @staticmethod
    def _task_motion_stage_waypoints(
        solution: TaskSolution,
        stage: TaskStage,
    ) -> tuple[str, list[dict[str, Any]]]:
        waypoints = solution.raw.get("waypoints")
        if not isinstance(waypoints, list):
            raise ValueError("task solution raw.waypoints must be a list")

        if solution.task_kind in {"pick", "hold"}:
            indexes_by_stage = {
                "connect_to_pre_grasp": ("free_motion", [0]),
                "approach_grasp": ("cartesian", [0, 1]),
                "approach_to_pre_grasp": ("cartesian", [0, 1]),
                "lift_object": ("cartesian", [1, 2]),
                "post_grasp_lift": ("cartesian", [1, 2]),
            }
        elif solution.task_kind in {"place", "move_and_release"}:
            indexes_by_stage = {
                "connect_to_place": ("free_motion", [0]),
                "approach_place": ("cartesian", [0, 1]),
                "release_pose": ("cartesian", [0, 1]),
                "retreat": ("cartesian", [1, 2]),
            }
        elif solution.task_kind == "pick_place":
            indexes_by_stage = {
                "connect_to_pre_grasp": ("free_motion", [0]),
                "approach_to_pre_grasp": ("cartesian", [0, 1]),
                "post_grasp_lift": ("cartesian", [1, 2]),
                "connect_to_place": ("free_motion", [3]),
                "approach_place": ("cartesian", [3, 4]),
                "release_pose": ("cartesian", [3, 4]),
                "retreat": ("cartesian", [4, 5]),
            }
        else:
            indexes_by_stage = {}

        planner, indexes = indexes_by_stage[stage.name]
        return planner, [waypoints[index] for index in indexes]

    def _task_solution_executed_result(
        self,
        *,
        solution: TaskSolution,
        stages: list[TaskStage],
        ok: bool,
        status: str,
        message: str,
    ) -> dict[str, Any]:
        stage_report = {
            "total": len(stages),
            "executed": sum(1 for stage in stages if stage.status == "executed"),
            "failed": sum(1 for stage in stages if stage.status == "failed"),
        }
        evidence = [
            {"kind": "scene_snapshot", "id": solution.scene_snapshot_id},
            {"kind": "stage_report", "count": len(stages)},
        ]
        raw = TaskExecutionResult(
            ok=ok,
            task_solution_id=solution.task_solution_id,
            task_kind=solution.task_kind,
            backend=solution.backend,
            stages=stages,
            created_from_tool=solution.created_from_tool,
            object_name=solution.object_name,
            robot_name=solution.robot_name,
            scene_snapshot_id=solution.scene_snapshot_id,
            stage_report=stage_report,
            approval=solution.approval,
            evidence=evidence,
            raw={"planning_frame": solution.planning_frame},
        ).to_dict()
        if ok:
            result = ToolResult.pass_result(
                robot=solution.robot_name,
                tool="moveit_execute_task_solution",
                phase="executed",
                status=status,
                message=message,
                checks=[VerificationCheck("task_solution_stages_executed", True, str(stage_report))],
                evidence=[Evidence("mcp_state", f"executed {solution.task_solution_id}")],
                raw=raw,
                can_execute=False,
            ).to_dict()
        else:
            result = ToolResult.fail_result(
                robot=solution.robot_name,
                tool="moveit_execute_task_solution",
                phase="executed",
                status=status,
                message=message,
                correction="Inspect the failed stage evidence, then replan the task solution before retrying.",
                checks=[VerificationCheck("task_solution_stages_executed", False, str(stage_report))],
                evidence=[Evidence("mcp_state", f"failed {solution.task_solution_id}")],
                raw=raw,
            ).to_dict()
        result["feedback"]["execution_target"] = "task_solution"
        return result

    def _new_scene_snapshot_id(self) -> str:
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"scene_{date}_{self._task_solution_sequence:03d}"

    def _new_plan_name(self, robot: str, tool: str) -> str:
        for _ in range(10):
            name = _generate_plan_name(tool)
            if (robot, name) not in self._used_plan_names:
                return name
        return f"{_generate_plan_name(tool)}_{uuid4().hex[:8]}"

    @staticmethod
    def _pick_attempt_plan_name(base_name: str, attempt_index: int, *, multi_attempt: bool) -> str:
        if not multi_attempt:
            return base_name
        return f"{base_name}__a{attempt_index:02d}"

    @staticmethod
    def _pick_preposition_plan_name(base_name: str) -> str:
        return f"{base_name}__preposition"

    @staticmethod
    def _pick_local_plan_name(base_name: str) -> str:
        return f"{base_name}__local_pick"

    @staticmethod
    def _pick_candidate_attempt(
        *,
        attempt_index: int,
        plan_name: str,
        workflow: dict[str, Any],
        feedback: PlanFeedback,
        selected: bool,
        planner: str = "cartesian",
        planning_pipeline: str | None = None,
        planner_id: str | None = None,
    ) -> dict[str, Any]:
        parameters = workflow["parameters"]
        attempt = {
            "attempt_index": attempt_index,
            "plan_name": plan_name,
            "grasp_face": parameters["grasp_face"],
            "approach_distance_m": parameters["approach_distance_m"],
            "grasp_standoff_m": parameters["grasp_standoff_m"],
            "lift_distance_m": parameters["lift_distance_m"],
            "planner": planner,
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "selected": selected,
        }
        if planning_pipeline is not None:
            attempt["planning_pipeline"] = planning_pipeline
        if planner_id is not None:
            attempt["planner_id"] = planner_id
        return attempt

    def _reserve_plan_name(
        self,
        *,
        robot: str,
        tool: str,
        name: str,
        allow_existing_name: bool,
    ) -> dict[str, Any] | None:
        key = (robot, name)
        if key in self._used_plan_names and not allow_existing_name:
            return ToolResult.fail_result(
                robot=robot,
                tool=tool,
                phase="pre_plan",
                status="plan name already used",
                message="Refusing to reuse caller-provided plan name without allow_existing_name=True",
                correction=REUSED_PLAN_NAME_CORRECTION,
                checks=[VerificationCheck("plan_name_unique", False, name)],
                evidence=[Evidence("mcp_state", f"plan name already used: {name}")],
                raw={"plan_name": name},
            ).to_dict()
        self._used_plan_names.add(key)
        return None

    @staticmethod
    def _invalid_input_result(
        *,
        robot: str,
        tool: str,
        status: str,
        details: str,
        plan_name: str | None,
    ) -> dict[str, Any]:
        return ToolResult.fail_result(
            robot=robot,
            tool=tool,
            phase="pre_plan",
            status=status,
            message="Refusing to publish planning request because input validation failed",
            correction=POSE_INPUT_CORRECTION,
            checks=[VerificationCheck("pose_valid", False, details)],
            evidence=[Evidence("mcp_state", f"input validation failed for plan name: {plan_name or '<omitted>'}")],
            raw={"plan_name": plan_name},
        ).to_dict()

    def _current_pose_result(self, feedback: CurrentPoseFeedback) -> dict[str, Any]:
        pose_observed = feedback.ok and feedback.pose is not None
        checks = [VerificationCheck("current_pose_observed", pose_observed, feedback.status)]
        evidence = [Evidence("ros_service", feedback.message, path=feedback.source)]
        raw = {
            "planning_frame": feedback.planning_frame,
            "pose": feedback.pose.to_msg() if feedback.pose else None,
            "source": feedback.source,
        }
        if pose_observed:
            return ToolResult.pass_result(
                robot=feedback.robot,
                tool="get_current_pose",
                phase="observed",
                status=feedback.status,
                message="Current MoveIt pose observed",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=False,
            ).to_dict()
        return ToolResult.fail_result(
            robot=feedback.robot,
            tool="get_current_pose",
            phase="observed",
            status=feedback.status,
            message="Current MoveIt pose could not be observed",
            correction=CURRENT_POSE_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
            verification_result="unknown",
        ).to_dict()

    def _plan_result(self, *, tool: str, feedback: PlanFeedback) -> dict[str, Any]:
        status_success = feedback.status in SUCCESS_STATUSES
        trajectory_observed = feedback.trajectory_points > 0
        checks = [
            VerificationCheck("status_success", status_success, feedback.status),
            VerificationCheck("trajectory_observed", trajectory_observed, f"{feedback.trajectory_points} points"),
        ]
        evidence = [
            Evidence("ros_topic", feedback.status, topic=f"/{feedback.robot}/request/status"),
            Evidence(
                "ros_topic",
                f"plan {feedback.name}: {feedback.trajectory_points} trajectory points",
                topic=f"/{feedback.robot}/request/planned_path",
            ),
            Evidence("mcp_state", f"actual plan name: {feedback.name}"),
        ]
        raw = {
            "plan_name": feedback.name,
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "final_joint_positions": feedback.final_joint_positions,
            "planning_diagnostics": {
                "log_dir": "server/logs/moveit_planning",
                "join_key": feedback.name,
            },
        }

        if trajectory_observed:
            self._planned[(feedback.robot, feedback.name)] = raw

        if status_success and trajectory_observed:
            return ToolResult.pass_result(
                robot=feedback.robot,
                tool=tool,
                phase="planned",
                status=feedback.status,
                message="Plan succeeded and trajectory feedback was observed",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=feedback.can_execute,
            ).to_dict()

        return ToolResult.fail_result(
            robot=feedback.robot,
            tool=tool,
            phase="planned",
            status=feedback.status,
            message="Plan did not satisfy execution requirements",
            correction=PLAN_NOT_EXECUTABLE_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def _pick_plan_result(
        self,
        *,
        feedback: PlanFeedback,
        object_context: dict[str, Any],
        workflow: dict[str, Any],
        source: str,
        planning_strategy: str = "cartesian",
        planning_strategy_resolved: str = "cartesian",
        candidate_attempts: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        status_success = feedback.status in SUCCESS_STATUSES
        trajectory_observed = feedback.trajectory_points > 0
        checks = [
            VerificationCheck("object_context_observed", True, str(workflow["object_name"])),
            VerificationCheck("grasp_face_selected", True, str(workflow["selected_grasp_face"]["name"])),
            VerificationCheck("status_success", status_success, feedback.status),
            VerificationCheck("trajectory_observed", trajectory_observed, f"{feedback.trajectory_points} points"),
        ]
        evidence = [
            Evidence("ros_service", f"object context observed for {workflow['object_name']}", path=source),
            Evidence("mcp_state", f"selected grasp face: {workflow['selected_grasp_face']['name']}"),
            Evidence("ros_topic", feedback.status, topic=f"/{feedback.robot}/request/status"),
            Evidence(
                "ros_topic",
                f"pick plan {feedback.name}: {feedback.trajectory_points} trajectory points",
                topic=f"/{feedback.robot}/request/planned_path",
            ),
        ]
        raw = {
            "workflow_kind": "pick",
            "plan_name": feedback.name,
            "object_name": workflow["object_name"],
            "planning_frame": workflow.get("planning_frame"),
            "object": object_context,
            "selected_grasp_face": workflow["selected_grasp_face"],
            "waypoints": workflow["waypoints"],
            "motion_segments": _pick_motion_segments_with_plan_names(workflow, feedback.name),
            "post_grasp": {
                "object_name": workflow["object_name"],
                "lift_plan_name": f"{feedback.name}__lift",
                "lift_waypoints": [
                    workflow["waypoints"][index]
                    for index in workflow["motion_segments"][1]["waypoint_indexes"]
                ],
            },
            "workflow_steps": workflow["workflow_steps"],
            "parameters": workflow["parameters"],
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "final_joint_positions": feedback.final_joint_positions,
            "planning_strategy": planning_strategy,
            "planning_strategy_resolved": planning_strategy_resolved,
            "available_planning_strategies": sorted(PICK_PLANNING_STRATEGIES),
            "candidate_attempts": candidate_attempts or [],
        }

        if trajectory_observed:
            self._planned[(feedback.robot, feedback.name)] = raw

        if status_success and trajectory_observed:
            return ToolResult.pass_result(
                robot=feedback.robot,
                tool="moveit_plan_pick",
                phase="planned",
                status=feedback.status,
                message="Pick plan succeeded and trajectory feedback was observed",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=feedback.can_execute,
            ).to_dict()

        correction = PLAN_NOT_EXECUTABLE_CORRECTION
        if planning_strategy == "auto":
            correction = (
                "Auto pick planning tried multiple Cartesian candidates without an executable plan. "
                'Try planning_strategy="sampled_approach", specify a different grasp_face, or change the scene/object pose.'
            )

        return ToolResult.fail_result(
            robot=feedback.robot,
            tool="moveit_plan_pick",
            phase="planned",
            status=feedback.status,
            message="Pick plan did not satisfy execution requirements",
            correction=correction,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def _pick_preposition_result(
        self,
        *,
        feedback: PlanFeedback,
        object_context: dict[str, Any],
        workflow: dict[str, Any],
        source: str,
        local_pick_plan_name: str,
        planning_strategy: str,
        planning_strategy_resolved: str,
        ) -> dict[str, Any]:
        status_success = feedback.status in SUCCESS_STATUSES
        trajectory_observed = feedback.trajectory_points > 0
        preposition_can_execute = trajectory_observed and feedback.can_execute
        target_pose = workflow["waypoints"][0]
        staging_face = str(workflow["selected_grasp_face"]["name"])
        next_action = {
            "tool": "moveit_execute_plan",
            "plan_name": feedback.name,
            "after_success": {
                "tool": "moveit_plan_pick",
                "arguments": {
                    "object_name": workflow["object_name"],
                    "plan_name": local_pick_plan_name,
                    "grasp_face": staging_face,
                    "approach_distance_m": workflow["parameters"]["approach_distance_m"],
                    "grasp_standoff_m": workflow["parameters"]["grasp_standoff_m"],
                    "lift_distance_m": workflow["parameters"]["lift_distance_m"],
                    "planning_strategy": "cartesian",
                },
            },
        }
        checks = [
            VerificationCheck("object_context_observed", True, str(workflow["object_name"])),
            VerificationCheck("grasp_face_selected", True, staging_face),
            VerificationCheck("status_success", status_success, feedback.status),
            VerificationCheck("trajectory_observed", trajectory_observed, f"{feedback.trajectory_points} points"),
        ]
        evidence = [
            Evidence("ros_service", f"object context observed for {workflow['object_name']}", path=source),
            Evidence("mcp_state", f"selected staging face: {staging_face}"),
            Evidence("ros_topic", feedback.status, topic=f"/{feedback.robot}/request/status"),
            Evidence(
                "ros_topic",
                f"preposition plan {feedback.name}: {feedback.trajectory_points} trajectory points",
                topic=f"/{feedback.robot}/request/planned_path",
            ),
        ]
        raw = {
            "plan_name": feedback.name,
            "object_name": workflow["object_name"],
            "planning_frame": workflow.get("planning_frame"),
            "object": object_context,
            "selected_grasp_face": workflow["selected_grasp_face"],
            "preposition": {
                "plan_name": feedback.name,
                "planner": "free_motion",
                "staging_face": staging_face,
                "target_pose": target_pose,
            },
            "workflow_segments": [
                {
                    "name": "preposition",
                    "planner": "free_motion",
                    "plan_name": feedback.name,
                    "target_pose": target_pose,
                },
                {
                    "name": "local_cartesian_pick",
                    "planner": "cartesian",
                    "plan_name": local_pick_plan_name,
                    "waypoints": workflow["waypoints"],
                },
            ],
            "workflow_steps": workflow["workflow_steps"],
            "waypoints": workflow["waypoints"],
            "parameters": workflow["parameters"],
            "next_action": next_action,
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "final_joint_positions": feedback.final_joint_positions,
            "planning_strategy": planning_strategy,
            "planning_strategy_resolved": planning_strategy_resolved,
            "available_planning_strategies": sorted(PICK_PLANNING_STRATEGIES),
            "candidate_attempts": [],
        }

        if preposition_can_execute:
            self._planned[(feedback.robot, feedback.name)] = raw

        if status_success and preposition_can_execute:
            stage_report = [
                {
                    "name": "connect_to_pre_grasp",
                    "stage_type": "motion_plan",
                    "status": "solved",
                    "plan_name": feedback.name,
                },
                {
                    "name": "local_cartesian_pick",
                    "stage_type": "motion_plan",
                    "status": "failed",
                    "plan_name": local_pick_plan_name,
                },
            ]
            partial_raw = {
                "stage_report": stage_report,
                "candidate_attempts": 1,
                "blocker": "local cartesian approach failed after preposition",
                "scene_snapshot_id": object_context.get("scene_snapshot_id"),
                "partial_plan": {
                    "kind": "preposition",
                    "plan_name": feedback.name,
                },
            }
            self._planned.pop((feedback.robot, feedback.name), None)
            result = ToolResult.fail_result(
                robot=feedback.robot,
                tool="moveit_plan_pick",
                phase="planned",
                status="pick_segment_planning_failed",
                message="Pick preposition plan succeeded, but the local Cartesian pick segment failed",
                correction="Use the suggested diagnostic tool before retrying the pick workflow.",
                checks=checks,
                evidence=evidence,
                raw=partial_raw,
            ).to_dict()
            result["error"] = "pick_segment_planning_failed"
            result["failed_segment"] = "local_cartesian_pick"
            result["retryable"] = True
            result["suggested_next_tool"] = "moveit_explain_motion_failure"
            return result

        return ToolResult.fail_result(
            robot=feedback.robot,
            tool="moveit_plan_pick",
            phase="planned",
            status=feedback.status,
            message="Pick preposition plan did not satisfy execution requirements",
            correction=(
                "Replan the auto pick preposition with a smaller or safer staging pose, "
                "then inspect the returned diagnostic before retrying the pick workflow."
            ),
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def _place_plan_result(
        self,
        *,
        feedback: PlanFeedback,
        object_context: dict[str, Any],
        workflow: dict[str, Any],
        source: str,
    ) -> dict[str, Any]:
        status_success = feedback.status in SUCCESS_STATUSES
        trajectory_observed = feedback.trajectory_points > 0
        checks = [
            VerificationCheck("object_context_observed", True, str(workflow["object_name"])),
            VerificationCheck("object_attached_before_place", object_context.get("state") == "attached", str(object_context.get("state"))),
            VerificationCheck("status_success", status_success, feedback.status),
            VerificationCheck("trajectory_observed", trajectory_observed, f"{feedback.trajectory_points} points"),
        ]
        evidence = [
            Evidence("ros_service", f"object context observed for {workflow['object_name']}", path=source),
            Evidence("mcp_state", f"place release pose: {workflow['release_tcp_pose']['position']}"),
            Evidence("ros_topic", feedback.status, topic=f"/{feedback.robot}/request/status"),
            Evidence(
                "ros_topic",
                f"place plan {feedback.name}: {feedback.trajectory_points} trajectory points",
                topic=f"/{feedback.robot}/request/planned_path",
            ),
        ]
        raw = {
            "workflow_kind": "place",
            "plan_name": feedback.name,
            "object_name": workflow["object_name"],
            "planning_frame": workflow.get("planning_frame"),
            "object": object_context,
            "target_object_pose": workflow["target_object_pose"],
            "release_tcp_pose": workflow["release_tcp_pose"],
            "waypoints": workflow["waypoints"],
            "workflow_steps": workflow["workflow_steps"],
            "parameters": workflow["parameters"],
            "release_after_execute": workflow["release_after_execute"],
            "status": feedback.status,
            "trajectory_points": feedback.trajectory_points,
            "can_execute": feedback.can_execute,
            "final_joint_positions": feedback.final_joint_positions,
        }

        if trajectory_observed:
            self._planned[(feedback.robot, feedback.name)] = raw

        if status_success and trajectory_observed:
            return ToolResult.pass_result(
                robot=feedback.robot,
                tool="moveit_plan_place",
                phase="planned",
                status=feedback.status,
                message="Place plan succeeded and trajectory feedback was observed",
                checks=checks,
                evidence=evidence,
                raw=raw,
                can_execute=feedback.can_execute,
            ).to_dict()

        return ToolResult.fail_result(
            robot=feedback.robot,
            tool="moveit_plan_place",
            phase="planned",
            status=feedback.status,
            message="Place plan did not satisfy execution requirements",
            correction=PLAN_NOT_EXECUTABLE_CORRECTION,
            checks=checks,
            evidence=evidence,
            raw=raw,
        ).to_dict()

    def _complete_pick_object(
        self,
        *,
        robot: str,
        planned: dict[str, Any],
        timeout_s: float,
    ) -> tuple[bool, list[VerificationCheck], list[Evidence], dict[str, Any]]:
        post_grasp = planned.get("post_grasp")
        if not isinstance(post_grasp, dict):
            return (
                False,
                [VerificationCheck("pick_post_grasp_metadata_present", False, str(post_grasp))],
                [Evidence("mcp_state", "missing pick post-grasp metadata")],
                {},
            )

        object_name = str(post_grasp.get("object_name") or "")
        gripper = self.close_gripper(robot, timeout_s=timeout_s)
        gripper_closed = bool(gripper.get("ok"))
        attached = self.attach_object(robot, object_name) if gripper_closed else {"ok": False}
        scene_attached = bool(attached.get("ok"))
        lift_executed = False
        lift_plan_name = str(post_grasp.get("lift_plan_name") or f"{planned.get('plan_name')}__lift")
        lift_waypoints = post_grasp.get("lift_waypoints")

        if scene_attached and isinstance(lift_waypoints, list):
            feedback = self.client.plan_cartesian_motion(
                robot=robot,
                name=lift_plan_name,
                poses=[Pose.from_input(waypoint) for waypoint in lift_waypoints],
                timeout_s=timeout_s,
            )
            if feedback.can_execute and feedback.trajectory_points > 0:
                self._planned[(robot, lift_plan_name)] = {
                    "plan_name": lift_plan_name,
                    "can_execute": True,
                    "status": feedback.status,
                    "final_joint_positions": feedback.final_joint_positions,
                }
                lift_result = self.execute_plan(robot, lift_plan_name, timeout_s=timeout_s)
                lift_executed = bool(lift_result.get("ok"))

        held_object = self.gripper.attached_object(robot)
        checks = [
            VerificationCheck("pick_gripper_closed", gripper_closed, str(gripper.get("feedback"))),
            VerificationCheck("pick_scene_attached", scene_attached, str(attached.get("feedback"))),
            VerificationCheck("pick_lift_executed", lift_executed, lift_plan_name),
            VerificationCheck("pick_object_held", held_object == object_name, str(held_object)),
        ]
        evidence = [
            Evidence("mcp_state", f"pick object: {object_name}"),
            Evidence("mcp_state", f"pick lift plan: {lift_plan_name}"),
        ]
        raw = {
            "object_name": object_name,
            "gripper_closed": gripper_closed,
            "planning_scene_attached": scene_attached,
            "lift_executed": lift_executed,
            "held_object": held_object,
            "lift_plan_name": lift_plan_name,
        }
        return all(check.passed for check in checks), checks, evidence, raw

    def _release_place_object(
        self,
        *,
        robot: str,
        planned: dict[str, Any],
        timeout_s: float,
    ) -> tuple[bool, list[VerificationCheck], list[Evidence], dict[str, Any]]:
        release = planned.get("release_after_execute")
        object_name = release.get("object_name") if isinstance(release, dict) else planned.get("object_name")
        object_pose = release.get("object_pose") if isinstance(release, dict) else None
        if not isinstance(object_name, str) or not object_name or not isinstance(object_pose, dict):
            return (
                False,
                [VerificationCheck("place_release_metadata_present", False, str(release))],
                [Evidence("mcp_state", "missing place release metadata")],
                {"object_name": object_name, "gripper_opened": False, "planning_scene_released": False},
            )

        gripper_feedback = self.client.command_gripper(robot=robot, state="open", timeout_s=timeout_s)
        gripper_opened = bool(gripper_feedback.ok)
        if gripper_opened:
            self.gripper.set_state(robot, "open")
        detach_feedback: DetachSceneFeedback = self.client.detach_object(
            robot=robot,
            object_name=object_name,
            object_pose=Pose.from_input(object_pose),
            timeout_s=timeout_s,
        )
        planning_scene_released = bool(detach_feedback.ok)
        checks = [
            VerificationCheck("place_gripper_opened", gripper_opened, str(gripper_feedback.observed_joint_position)),
            VerificationCheck("planning_scene_object_released", planning_scene_released, detach_feedback.status),
        ]
        evidence = [
            *_gripper_evidence(gripper_feedback),
            Evidence("ros_service", detach_feedback.message, path=detach_feedback.source),
        ]
        raw = {
            "object_name": object_name,
            "gripper_opened": gripper_opened,
            "planning_scene_released": planning_scene_released,
            "gripper_state": self.gripper.get_state(robot),
            "detach_status": detach_feedback.status,
            "released_object_pose": object_pose,
        }
        return gripper_opened and planning_scene_released, checks, evidence, raw


def _pick_motion_segments_with_plan_names(workflow: dict[str, Any], plan_name: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for segment in workflow["motion_segments"]:
        item = dict(segment)
        item["plan_name"] = f"{plan_name}__lift" if item.get("name") == "post_grasp_lift" else plan_name
        segments.append(item)
    return segments


def _staged_manipulation_plan_name(task_kind: str, object_name: str, attempt_index: int, stage_name: str) -> str:
    return f"manipulation_{task_kind}_{_slug(object_name)}_c{attempt_index:02d}_{_slug(stage_name)}"


def _contract_hold_task_stages() -> list[TaskStage]:
    return [
        TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
        TaskStage(
            "connect_to_pre_grasp",
            "motion_plan",
            "solved",
            [{"kind": "task_contract_waypoint", "waypoint_index": 0}],
            {"waypoint_index": 0},
        ),
        TaskStage(
            "approach_to_pre_grasp",
            "motion_plan",
            "solved",
            [{"kind": "task_contract_waypoint", "waypoint_index": 1}],
            {"waypoint_index": 1},
        ),
        TaskStage("close_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
        TaskStage("attach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
        TaskStage(
            "post_grasp_lift",
            "motion_plan",
            "solved",
            [{"kind": "task_contract_waypoint", "waypoint_index": 2}],
            {"waypoint_index": 2},
        ),
        TaskStage("verify_attached_object", "verification", "solved", [{"kind": "attachment_check"}]),
    ]


def _staged_hold_task_stages(motion_stages: list[dict[str, Any]]) -> list[TaskStage]:
    motion_by_name = {str(stage.get("name")): stage for stage in motion_stages}
    return [
        TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
        TaskStage(
            "connect_to_pre_grasp",
            "motion_plan",
            "solved",
            [motion_by_name["connect_to_pre_grasp"]["preview_evidence"]],
            {"plan_name": motion_by_name["connect_to_pre_grasp"]["plan_name"]},
        ),
        TaskStage(
            "approach_to_pre_grasp",
            "motion_plan",
            "solved",
            [motion_by_name["approach_to_pre_grasp"]["preview_evidence"]],
            {"plan_name": motion_by_name["approach_to_pre_grasp"]["plan_name"]},
        ),
        TaskStage("close_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
        TaskStage("attach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
        TaskStage(
            "post_grasp_lift",
            "motion_plan",
            "solved",
            [motion_by_name["post_grasp_lift"]["preview_evidence"]],
            {"plan_name": motion_by_name["post_grasp_lift"]["plan_name"]},
        ),
        TaskStage("verify_attached_object", "verification", "solved", [{"kind": "attachment_check"}]),
    ]


def _staged_place_task_stages(motion_stages: list[dict[str, Any]]) -> list[TaskStage]:
    motion_by_name = {str(stage.get("name")): stage for stage in motion_stages}
    return [
        TaskStage("observe_current_state", "observation", "solved", [{"kind": "scene_snapshot"}]),
        TaskStage(
            "connect_to_place",
            "motion_plan",
            "solved",
            [motion_by_name["connect_to_place"]["preview_evidence"]],
            {"plan_name": motion_by_name["connect_to_place"]["plan_name"]},
        ),
        TaskStage(
            "approach_place",
            "motion_plan",
            "solved",
            [motion_by_name["approach_place"]["preview_evidence"]],
            {"plan_name": motion_by_name["approach_place"]["plan_name"]},
        ),
        TaskStage("open_gripper", "gripper", "solved", [{"kind": "gripper_command"}]),
        TaskStage("detach_object", "scene_update", "solved", [{"kind": "planning_scene_update"}]),
        TaskStage(
            "retreat",
            "motion_plan",
            "solved",
            [motion_by_name["retreat"]["preview_evidence"]],
            {"plan_name": motion_by_name["retreat"]["plan_name"]},
        ),
        TaskStage("verify_released_object", "verification", "solved", [{"kind": "release_check"}]),
    ]


def _staged_agent_path_preview(motion_stages: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "kind": "AgentPath",
        "name": "AgentPath",
        "motion_stages": [
            {
                "name": stage.get("name"),
                "plan_name": stage.get("plan_name"),
                "planner": stage.get("planner"),
                "trajectory_points": stage.get("trajectory_points"),
                "evidence": stage.get("preview_evidence"),
            }
            for stage in motion_stages
        ],
    }


def _staged_waypoint_agent_path_preview(
    waypoints: list[dict[str, Any]],
    stages: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "kind": "AgentPath",
        "name": "AgentPath",
        "motion_stages": [
            {
                "name": stage.get("name"),
                "waypoint_index": stage.get("waypoint_index"),
                "trajectory_points": 1,
                "evidence": {
                    "kind": "waypoint_preview",
                    "waypoint": waypoints[int(stage["waypoint_index"])],
                },
            }
            for stage in stages
            if isinstance(stage.get("waypoint_index"), int)
            and 0 <= int(stage["waypoint_index"]) < len(waypoints)
        ],
    }


def _staged_candidate_public_summary(attempt: dict[str, Any]) -> dict[str, Any]:
    return {
        "attempt_index": attempt.get("attempt_index"),
        "grasp_face": attempt.get("grasp_face"),
        "approach_distance_m": attempt.get("approach_distance_m"),
        "grasp_standoff_m": attempt.get("grasp_standoff_m"),
        "lift_distance_m": attempt.get("lift_distance_m"),
    }


def _contract_hold_execution_steps(
    *,
    object_name: str,
    scene_snapshot_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "step": 1,
            "handler": "motion",
            "name": "connect_to_pre_grasp",
            "waypoint_index": 0,
            "source_stage": "connect_to_pre_grasp",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "verified_motion_plan",
        },
        {
            "step": 2,
            "handler": "motion",
            "name": "approach_to_pre_grasp",
            "waypoint_index": 1,
            "source_stage": "approach_to_pre_grasp",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "verified_motion_plan",
        },
        {
            "step": 3,
            "handler": "close_gripper",
            "name": "close_gripper",
            "tool": "moveit_close_gripper",
            "source_stage": "close_gripper",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "verified_gripper_closed",
        },
        {
            "step": 4,
            "handler": "attach_object",
            "name": "attach_object",
            "tool": "moveit_attach_object",
            "source_stage": "attach_object",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "planning_scene_attached",
            "arguments": {"verified_gripper_closed": True},
        },
        {
            "step": 5,
            "handler": "motion",
            "name": "post_grasp_lift",
            "waypoint_index": 2,
            "source_stage": "post_grasp_lift",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "verified_motion_plan",
        },
        {
            "step": 6,
            "handler": "verify_attached_object",
            "name": "verify_attached_object",
            "tool": "moveit_verify_attached_object",
            "source_stage": "verify_attached_object",
            "object_name": object_name,
            "scene_snapshot_id": scene_snapshot_id,
            "required_proof": "attachment_check",
        },
    ]


def _contract_hold_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
) -> dict[str, Any]:
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": _contract_hold_execution_steps(
            object_name=object_name,
            scene_snapshot_id=scene_snapshot_id,
        ),
    }


def _staged_hold_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
    motion_stages: list[dict[str, Any]],
) -> dict[str, Any]:
    motion_by_name = {str(stage.get("name")): stage for stage in motion_stages}
    connect_plan_name = str(motion_by_name["connect_to_pre_grasp"]["plan_name"])
    approach_plan_name = str(motion_by_name["approach_to_pre_grasp"]["plan_name"])
    lift_plan_name = str(motion_by_name["post_grasp_lift"]["plan_name"])
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": [
            {
                "step": 1,
                "handler": "motion",
                "name": "connect_to_pre_grasp",
                "plan_handle": connect_plan_name,
                "waypoint_index": 0,
                "source_stage": "connect_to_pre_grasp",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 2,
                "handler": "motion",
                "name": "approach_to_pre_grasp",
                "plan_handle": approach_plan_name,
                "waypoint_index": 1,
                "source_stage": "approach_to_pre_grasp",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 3,
                "handler": "close_gripper",
                "name": "close_gripper",
                "tool": "moveit_close_gripper",
                "source_stage": "close_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_closed",
            },
            {
                "step": 4,
                "handler": "attach_object",
                "name": "attach_object",
                "tool": "moveit_attach_object",
                "source_stage": "attach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_attached",
                "arguments": {"verified_gripper_closed": True},
            },
            {
                "step": 5,
                "handler": "motion",
                "name": "post_grasp_lift",
                "plan_handle": lift_plan_name,
                "waypoint_index": 2,
                "source_stage": "post_grasp_lift",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 6,
                "handler": "verify_attached_object",
                "name": "verify_attached_object",
                "tool": "moveit_verify_attached_object",
                "source_stage": "verify_attached_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "attachment_check",
            },
        ],
    }


def _staged_release_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
    object_pose: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": [
            {
                "step": 1,
                "handler": "open_gripper",
                "name": "open_gripper",
                "tool": "moveit_open_gripper",
                "source_stage": "open_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_open",
            },
            {
                "step": 2,
                "handler": "release_object",
                "name": "release_object",
                "tool": "moveit_release_object",
                "source_stage": "detach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_update",
                "arguments": {"object_name": object_name, "object_pose": object_pose},
            },
            {
                "step": 3,
                "handler": "verify_released_object",
                "name": "verify_released_object",
                "tool": "moveit_verify_released_object",
                "source_stage": "verify_released_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "release_check",
                "arguments": {"object_name": object_name},
            },
        ],
    }


def _staged_place_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
    object_pose: dict[str, Any],
    motion_stages: list[dict[str, Any]],
) -> dict[str, Any]:
    motion_by_name = {str(stage.get("name")): stage for stage in motion_stages}
    connect_plan_name = str(motion_by_name["connect_to_place"]["plan_name"])
    approach_plan_name = str(motion_by_name["approach_place"]["plan_name"])
    retreat_plan_name = str(motion_by_name["retreat"]["plan_name"])
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": [
            {
                "step": 1,
                "handler": "motion",
                "name": "connect_to_place",
                "plan_handle": connect_plan_name,
                "waypoint_index": 0,
                "source_stage": "connect_to_place",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 2,
                "handler": "motion",
                "name": "approach_place",
                "plan_handle": approach_plan_name,
                "waypoint_index": 1,
                "source_stage": "approach_place",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 3,
                "handler": "open_gripper",
                "name": "open_gripper",
                "tool": "moveit_open_gripper",
                "source_stage": "open_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_open",
            },
            {
                "step": 4,
                "handler": "release_object",
                "name": "release_object",
                "tool": "moveit_release_object",
                "source_stage": "detach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_update",
                "arguments": {"object_name": object_name, "object_pose": object_pose},
            },
            {
                "step": 5,
                "handler": "motion",
                "name": "retreat",
                "plan_handle": retreat_plan_name,
                "waypoint_index": 2,
                "source_stage": "retreat",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 6,
                "handler": "verify_released_object",
                "name": "verify_released_object",
                "tool": "moveit_verify_released_object",
                "source_stage": "verify_released_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "release_check",
                "arguments": {"object_name": object_name},
            },
        ],
    }


def _staged_pick_place_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
    release_object_pose: dict[str, Any],
    motion_stages: list[dict[str, Any]],
) -> dict[str, Any]:
    motion_by_name = {str(stage.get("name")): stage for stage in motion_stages}
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": [
            {
                "step": 1,
                "handler": "motion",
                "name": "connect_to_pre_grasp",
                "plan_handle": str(motion_by_name["connect_to_pre_grasp"]["plan_name"]),
                "waypoint_index": 0,
                "source_stage": "connect_to_pre_grasp",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 2,
                "handler": "motion",
                "name": "approach_to_pre_grasp",
                "plan_handle": str(motion_by_name["approach_to_pre_grasp"]["plan_name"]),
                "waypoint_index": 1,
                "source_stage": "approach_to_pre_grasp",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 3,
                "handler": "close_gripper",
                "name": "close_gripper",
                "tool": "moveit_close_gripper",
                "source_stage": "close_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_closed",
            },
            {
                "step": 4,
                "handler": "attach_object",
                "name": "attach_object",
                "tool": "moveit_attach_object",
                "source_stage": "attach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_attached",
                "arguments": {"verified_gripper_closed": True},
            },
            {
                "step": 5,
                "handler": "motion",
                "name": "post_grasp_lift",
                "plan_handle": str(motion_by_name["post_grasp_lift"]["plan_name"]),
                "waypoint_index": 2,
                "source_stage": "post_grasp_lift",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 6,
                "handler": "motion",
                "name": "connect_to_place",
                "plan_handle": str(motion_by_name["connect_to_place"]["plan_name"]),
                "waypoint_index": 3,
                "source_stage": "connect_to_place",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 7,
                "handler": "motion",
                "name": "approach_place",
                "plan_handle": str(motion_by_name["approach_place"]["plan_name"]),
                "waypoint_index": 4,
                "source_stage": "approach_place",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 8,
                "handler": "open_gripper",
                "name": "open_gripper",
                "tool": "moveit_open_gripper",
                "source_stage": "open_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_open",
            },
            {
                "step": 9,
                "handler": "release_object",
                "name": "release_object",
                "tool": "moveit_release_object",
                "source_stage": "detach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_update",
                "arguments": {"object_name": object_name, "object_pose": release_object_pose},
            },
            {
                "step": 10,
                "handler": "motion",
                "name": "retreat",
                "plan_handle": str(motion_by_name["retreat"]["plan_name"]),
                "waypoint_index": 5,
                "source_stage": "retreat",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 11,
                "handler": "verify_released_object",
                "name": "verify_released_object",
                "tool": "moveit_verify_released_object",
                "source_stage": "verify_released_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "release_check",
                "arguments": {"object_name": object_name},
            },
        ],
    }


def _pick_task_candidate_attempts(candidate_workflows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for attempt_index, workflow in enumerate(candidate_workflows, start=1):
        params = _safe_dict(workflow.get("parameters"))
        selected_face = _safe_dict(workflow.get("selected_grasp_face"))
        attempts.append(
            {
                "attempt_index": attempt_index,
                "grasp_face": params.get("grasp_face") or selected_face.get("name"),
                "approach_distance_m": params.get("approach_distance_m"),
                "grasp_standoff_m": params.get("grasp_standoff_m"),
                "lift_distance_m": params.get("lift_distance_m"),
                "stage_evidence": [
                    {"stage_type": "GenerateGraspPose", "status": "candidate_generated"},
                    {"stage_type": "ComputeIK", "status": "emulated_not_run"},
                ],
                "status": "selected" if attempt_index == 1 else "generated",
                "selected": attempt_index == 1,
            }
        )
    return attempts


def _emulated_place_execution_contract(
    *,
    task_solution_id: str,
    object_name: str,
    scene_snapshot_id: str,
    workflow: dict[str, Any],
) -> dict[str, Any]:
    object_pose = workflow["release_after_execute"]["object_pose"]
    return {
        "target_kind": "task_solution",
        "task_solution_id": task_solution_id,
        "object_name": object_name,
        "scene_snapshot_id": scene_snapshot_id,
        "requires_explicit_approval": True,
        "can_execute": True,
        "steps": [
            {
                "step": 1,
                "handler": "motion",
                "name": "release_pose",
                "waypoint_index": 1,
                "source_stage": "approach_place",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "emulated_motion_plan",
            },
            {
                "step": 2,
                "handler": "open_gripper",
                "name": "open_gripper",
                "tool": "moveit_open_gripper",
                "source_stage": "open_gripper",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "verified_gripper_open",
            },
            {
                "step": 3,
                "handler": "release_object",
                "name": "release_object",
                "tool": "moveit_release_object",
                "source_stage": "detach_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "planning_scene_update",
                "arguments": {"object_name": object_name, "object_pose": object_pose},
            },
            {
                "step": 4,
                "handler": "motion",
                "name": "retreat",
                "waypoint_index": 2,
                "source_stage": "retreat",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "emulated_motion_plan",
            },
            {
                "step": 5,
                "handler": "verify_released_object",
                "name": "verify_released_object",
                "tool": "moveit_verify_released_object",
                "source_stage": "verify_released_object",
                "object_name": object_name,
                "scene_snapshot_id": scene_snapshot_id,
                "required_proof": "release_check",
                "arguments": {"object_name": object_name},
            },
        ],
    }


def _mtc_payload_has_solution(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    task_solution_id = payload.get("task_solution_id")
    stages = _mtc_stage_summaries(payload)
    return (
        payload.get("ok") is True
        and isinstance(task_solution_id, str)
        and bool(task_solution_id.strip())
        and bool(stages)
        and all(_mtc_stage_status(stage) == "solved" for stage in stages)
    )


def _mtc_compound_payload_has_solution(payload: dict[str, Any] | None) -> bool:
    if not isinstance(payload, dict):
        return False
    if not _mtc_payload_has_solution(payload):
        return False
    contract = _safe_dict(payload.get("execution_contract"))
    if contract.get("can_execute") is not True:
        return False
    scene_snapshot = _safe_dict(payload.get("scene_snapshot"))
    scene_snapshot_id = str(scene_snapshot.get("id") or "mtc_scene_snapshot")
    steps = _mtc_execution_contract(
        payload,
        object_name=str(payload.get("object_name") or "object"),
        scene_snapshot_id=scene_snapshot_id,
    )
    return bool(steps) and _mtc_compound_payload_has_preview_evidence(payload, steps)


def _mtc_compound_payload_has_preview_evidence(payload: dict[str, Any], steps: list[dict[str, Any]]) -> bool:
    preview = _safe_dict(payload.get("preview"))
    if not preview:
        return False
    has_solution_preview = (
        str(preview.get("solution_topic") or "") == "/solution"
        and _preview_value_is_available(preview.get("solution_preview"))
    )
    ar_preview_mode = str(preview.get("ar_preview_mode") or "")
    has_ar_preview = (
        str(preview.get("ar_preview_service") or "") == "/vizor_robot_control"
        and ar_preview_mode.casefold() != "none_no_motion"
        and _preview_value_is_available(ar_preview_mode)
    )
    if _mtc_contract_has_motion(steps):
        return has_solution_preview or has_ar_preview
    task_goal = str(payload.get("task_goal") or _safe_dict(payload.get("requirements")).get("goal") or "")
    if task_goal == "release" and ar_preview_mode.casefold() == "none_no_motion":
        return _mtc_contract_has_release_proof(steps)
    return has_solution_preview or has_ar_preview


def _mtc_contract_has_release_proof(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        handler = _mtc_contract_canonical_handler(str(step.get("handler") or ""))
        if handler in _MTC_RELEASE_PROOF_HANDLERS:
            return True
    return False


def _preview_value_is_available(value: Any) -> bool:
    normalized = str(value or "").strip().casefold()
    return normalized not in {"", "unavailable", "not_published", "not published", "failed", "none", "missing"}


def _mtc_contract_has_motion(steps: list[dict[str, Any]]) -> bool:
    for step in steps:
        handler = str(step.get("handler") or "")
        if handler in {"motion", "execute_plan"}:
            return True
        if any(key in step for key in ("plan_handle", "target_pose", "target_position", "waypoint", "waypoints", "waypoint_index")):
            return True
    return False


def _mtc_execution_contract(payload: dict[str, Any], *, object_name: str, scene_snapshot_id: str) -> list[dict[str, Any]]:
    raw_steps = payload.get("execution_contract")
    if isinstance(raw_steps, dict):
        steps_value = raw_steps.get("steps")
        if not isinstance(steps_value, list):
            steps_value = raw_steps.get("stages")
        raw_steps = steps_value
    if not isinstance(raw_steps, list):
        return []
    steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_steps, start=1):
        if not isinstance(item, dict):
            return []
        handler = item.get("handler")
        source_stage = item.get("source_stage")
        required_proof = item.get("required_proof")
        if not all(isinstance(value, str) and value for value in (handler, source_stage, required_proof)):
            return []
        if handler not in _MTC_CONTRACT_HANDLERS:
            return []
        step_object_name = item.get("object_name") or object_name
        step_scene_snapshot_id = item.get("scene_snapshot_id") or scene_snapshot_id
        if not isinstance(step_object_name, str) or not step_object_name:
            return []
        if not isinstance(step_scene_snapshot_id, str) or not step_scene_snapshot_id:
            return []
        step: dict[str, Any] = {
            "step": item.get("step") if isinstance(item.get("step"), int) else index,
            "handler": handler,
            "source_stage": source_stage,
            "object_name": step_object_name,
            "scene_snapshot_id": step_scene_snapshot_id,
            "required_proof": required_proof,
        }
        for key in ("name", "plan_handle"):
            value = item.get(key)
            if isinstance(value, str) and value:
                step[key] = value
        tool = item.get("tool")
        if isinstance(tool, str) and tool in _MTC_CONTRACT_TOOLS:
            step["tool"] = tool
        waypoint_index = item.get("waypoint_index")
        if isinstance(waypoint_index, int) and not isinstance(waypoint_index, bool) and waypoint_index >= 0:
            step["waypoint_index"] = waypoint_index
        for key in ("target_pose", "target_position", "waypoint", "waypoints"):
            if key in item:
                ok, value = _mtc_safe_contract_value(item[key])
                if ok:
                    step[key] = value
        arguments = _mtc_contract_arguments(item.get("arguments"))
        if arguments:
            step["arguments"] = arguments
        steps.append(step)
    return steps


_MTC_CONTRACT_HANDLERS = {
    "motion",
    "close_gripper",
    "open_gripper",
    "attach_object",
    "release_object",
    "verify_attached_object",
    "verify_released_object",
    "observe_current_state",
    "execute_plan",
    "verify_attached",
    "verify_released",
}
_MTC_CONTRACT_HANDLER_ALIASES = {
    "execute_plan": "motion",
    "verify_attached": "verify_attached_object",
    "verify_released": "verify_released_object",
}
_MTC_RELEASE_PROOF_HANDLERS = {
    "release_object",
    "verify_released_object",
    "verify_released",
}
_MTC_CONTRACT_TOOLS = {
    "moveit_execute_plan",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
    "moveit_release_object",
    "moveit_verify_attached_object",
    "moveit_verify_released_object",
}
_MTC_CONTRACT_ARGUMENT_KEYS = {
    "object_name",
    "object_pose",
    "plan_name",
    "robot_name",
    "verified_gripper_open",
    "link_name",
    "touch_links",
    "target_pose",
    "target_position",
    "timeout_s",
}
_MTC_UNSAFE_CONTRACT_KEY_FRAGMENTS = ("script", "code", "command", "callback", "eval", "lambda")


def _mtc_contract_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    arguments: dict[str, Any] = {}
    for key in _MTC_CONTRACT_ARGUMENT_KEYS:
        if key not in value:
            continue
        ok, safe_value = _mtc_safe_contract_value(value[key])
        if ok:
            arguments[key] = safe_value
    return arguments


def _mtc_safe_contract_value(value: Any) -> tuple[bool, Any]:
    if value is None or isinstance(value, (str, bool)):
        return True, value
    if isinstance(value, int) and not isinstance(value, bool):
        return True, value
    if isinstance(value, float):
        return (math.isfinite(value), value)
    if isinstance(value, list):
        items: list[Any] = []
        for item in value:
            ok, safe_item = _mtc_safe_contract_value(item)
            if not ok:
                return False, None
            items.append(safe_item)
        return True, items
    if isinstance(value, dict):
        data: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or _mtc_contract_key_is_unsafe(key):
                continue
            ok, safe_item = _mtc_safe_contract_value(item)
            if ok:
                data[key] = safe_item
        return True, data
    return False, None


def _mtc_contract_key_is_unsafe(key: str) -> bool:
    normalized = key.casefold()
    return any(fragment in normalized for fragment in _MTC_UNSAFE_CONTRACT_KEY_FRAGMENTS)


def _stored_task_solution_execution_contract(solution: TaskSolution) -> list[dict[str, Any]]:
    steps = solution.raw.get("execution_contract")
    if isinstance(steps, dict):
        steps = steps.get("steps")
    if not isinstance(steps, list):
        return []
    sanitized_steps: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict) or step.get("handler") not in _MTC_CONTRACT_HANDLERS:
            return []
        sanitized_steps.append(dict(step))
    return sanitized_steps


def _mtc_contract_canonical_handler(handler: str) -> str:
    return _MTC_CONTRACT_HANDLER_ALIASES.get(handler, handler)


def _mtc_contract_step_name(step: dict[str, Any]) -> str:
    name = step.get("name")
    if isinstance(name, str) and name:
        return name
    handler = step.get("handler")
    if isinstance(handler, str) and handler:
        return _mtc_contract_canonical_handler(handler)
    return "execution_contract_step"


def _mtc_contract_step_object_name(step: dict[str, Any], default_object_name: str) -> str:
    arguments = _safe_dict(step.get("arguments"))
    object_name = arguments.get("object_name") or step.get("object_name") or default_object_name
    return object_name if isinstance(object_name, str) and object_name else default_object_name


def _mtc_contract_step_plan_name(step: dict[str, Any]) -> str | None:
    arguments = _safe_dict(step.get("arguments"))
    for value in (arguments.get("plan_name"), step.get("plan_name"), step.get("plan_handle")):
        if isinstance(value, str) and value:
            return value
    return None


def _mtc_non_executable_contract(value: Any) -> dict[str, Any]:
    contract = _safe_dict(value)
    target_kind = contract.get("target_kind")
    requires_explicit_approval = contract.get("requires_explicit_approval")
    return {
        "target_kind": target_kind if isinstance(target_kind, str) and target_kind else "task_solution",
        "requires_explicit_approval": requires_explicit_approval if isinstance(requires_explicit_approval, bool) else True,
        "can_execute": False,
    }


def _mtc_stage_summaries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    stages = payload.get("stage_summaries")
    if stages is None:
        stages = payload.get("stages")
    return _safe_dict_list(stages)


def _mtc_task_stage(stage: dict[str, Any]) -> TaskStage:
    stage_type = _mtc_stage_type(stage)
    cost = _optional_float(stage.get("cost"))
    evidence = _safe_dict_list(stage.get("evidence"))
    stage_evidence: dict[str, Any] = {"kind": "mtc_stage", "stage_type": stage_type}
    if cost is not None:
        stage_evidence["cost"] = cost
    evidence.append(stage_evidence)
    raw: dict[str, Any] = {}
    for key in ("cost", "solution_count", "failure_count", "failed_reason", "planner"):
        if key in stage:
            raw[key] = stage[key]
    return TaskStage(
        name=str(stage.get("name") or stage_type),
        stage_type=stage_type,
        status=_mtc_stage_status(stage),
        evidence=evidence,
        raw=raw,
    )


def _mtc_stage_public_summary(stage: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "name": str(stage.get("name") or _mtc_stage_type(stage)),
        "stage_type": _mtc_stage_type(stage),
        "status": _mtc_stage_status(stage),
    }
    cost = _optional_float(stage.get("cost"))
    if cost is not None:
        summary["cost"] = cost
    failed_stage = stage.get("failed_stage")
    if isinstance(failed_stage, str) and failed_stage:
        summary["failed_stage"] = failed_stage
    return summary


def _mtc_stage_type(stage: dict[str, Any]) -> str:
    value = str(stage.get("stage_type") or stage.get("type") or stage.get("name") or "MTCStage")
    normalized = value.replace(" ", "").replace("_", "").casefold()
    aliases = {
        "currentstate": "CurrentState",
        "connect": "Connect",
        "generategrasppose": "GenerateGraspPose",
        "computeik": "ComputeIK",
        "moverelative": "MoveRelative",
        "modifyplanningscene": "ModifyPlanningScene",
    }
    return aliases.get(normalized, value)


def _mtc_stage_status(stage: dict[str, Any]) -> str:
    status = str(stage.get("status") or "")
    if status.casefold() in {"solved", "success", "succeeded"}:
        return "solved"
    if status.casefold() in {"failed", "failure", "error"}:
        return "failed"
    return status or "unknown"


def _safe_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _safe_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, dict)]


def _object_pose_from_context(object_context: dict[str, Any]) -> dict[str, Any] | None:
    pose = object_context.get("pose")
    if not isinstance(pose, dict):
        return None
    position = pose.get("position")
    orientation = pose.get("orientation")
    if not isinstance(position, dict) or not isinstance(orientation, dict):
        return None
    try:
        return {
            "position": {axis: float(position[axis]) for axis in ("x", "y", "z")},
            "orientation": {axis: float(orientation[axis]) for axis in ("x", "y", "z", "w")},
        }
    except (KeyError, TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    if not math.isfinite(float(value)):
        return None
    return float(value)


def _diagnose_motion_failure(
    *,
    failed_tool_name: str,
    failed_tool_result: dict[str, Any] | str,
    failed_tool_arguments: dict[str, Any] | None,
    user_intent: str | None,
) -> tuple[str, str, bool, str | None]:
    text = " ".join(
        value.casefold()
        for value in (
            failed_tool_name,
            _failure_text(failed_tool_result),
            str(failed_tool_arguments or {}),
            user_intent or "",
        )
    )
    if "physical" in text:
        return "physical_mode_blocked", PHYSICAL_MODE_CORRECTION, True, "moveit_get_robot_state"
    if "object not found" in text or "object_name" in text or "planning-scene" in text:
        return "object_grounding_failed", OBJECT_NOT_FOUND_CORRECTION, True, "moveit_list_scene_objects"
    if "gripper" in text or "attach" in text or "attached" in text:
        return "attachment_or_gripper_failed", ATTACHED_OBJECT_CORRECTION, True, "moveit_verify_attached_object"
    if "plan not verified" in text or "raw.plan_name" in text:
        return "missing_verified_plan", UNVERIFIED_PLAN_CORRECTION, True, "moveit_plan_free_motion"
    if "execution unverified" in text or "joint" in text or "fake_controller" in text:
        return "execution_unverified", EXECUTION_UNVERIFIED_CORRECTION, True, "moveit_get_robot_state"
    if "cartesian" in failed_tool_name or "waypoint" in text or "wave" in text or "trace" in text:
        return "cartesian_planning_failed", PLAN_NOT_EXECUTABLE_CORRECTION, True, "moveit_plan_cartesian_motion"
    if "incomplete path" in text or "not executable" in text or "trajectory" in text or "planning failed" in text:
        return "planning_failed", PLAN_NOT_EXECUTABLE_CORRECTION, True, "moveit_plan_free_motion"
    return "unknown_motion_failure", "Observe current robot state, inspect the failed result, then retry with a narrower plan.", True, "moveit_get_robot_state"


def _failure_text(value: dict[str, Any] | str) -> str:
    if isinstance(value, str):
        return value
    fragments: list[str] = []
    for key in ("error", "message", "status", "correction"):
        item = value.get(key)
        if isinstance(item, str):
            fragments.append(item)
    feedback = value.get("feedback")
    if isinstance(feedback, dict):
        for key in ("error", "message", "status", "correction"):
            item = feedback.get(key)
            if isinstance(item, str):
                fragments.append(item)
    return " ".join(fragments) or str(value)


def _remove_scene_checks(feedback: RemoveSceneFeedback) -> list[VerificationCheck]:
    return [
        VerificationCheck(
            "planning_scene_observed",
            feedback.status != "planning scene unavailable",
            feedback.status,
        ),
        VerificationCheck(
            "planning_scene_diff_applied",
            feedback.scene_update_published,
            feedback.source,
        ),
        VerificationCheck("object_removed", feedback.ok, feedback.object_name),
    ]


def _remove_scene_raw(feedback: RemoveSceneFeedback) -> dict[str, Any]:
    return {
        "object_name": feedback.object_name,
        "planning_frame": feedback.planning_frame,
        "scene_update_published": feedback.scene_update_published,
        "planning_scene_state": "removed" if feedback.ok else feedback.status,
    }


def _gripper_checks(feedback: Any) -> list[VerificationCheck]:
    return [
        VerificationCheck("robotiq_action_goal_sent", bool(feedback.command_sent), feedback.action_name),
        VerificationCheck(
            "robotiq_action_result_observed",
            feedback.action_result is not None,
            str(feedback.action_result),
        ),
        VerificationCheck(
            "robotiq_finger_joint_matched",
            _positions_match([feedback.expected_joint_position], [feedback.observed_joint_position], 1e-2),
            f"expected={_float_summary(feedback.expected_joint_position)}, observed={_float_summary(feedback.observed_joint_position)}",
        ),
    ]


def _gripper_evidence(feedback: Any) -> list[Evidence]:
    return [
        Evidence("ros_action", f"goal position {feedback.goal_position_m:.3f}m", path=feedback.action_name),
        Evidence("ros_topic", _float_summary(feedback.observed_joint_position), topic=feedback.joint_state_topic),
    ]


def _gripper_raw(feedback: Any, state: str, attached_object: str | None) -> dict[str, Any]:
    return {
        "gripper_state": state,
        "attached_object": attached_object,
        "action_name": feedback.action_name,
        "action_type": feedback.action_type,
        "joint_state_topic": feedback.joint_state_topic,
        "goal_position_m": feedback.goal_position_m,
        "speed_mps": feedback.speed_mps,
        "force": feedback.force,
        "expected_joint_position": feedback.expected_joint_position,
        "observed_joint_position": feedback.observed_joint_position,
        "action_result": feedback.action_result,
    }


def _float_summary(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{value:g}"


def _generate_plan_name(tool: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{tool}_{timestamp}_{uuid4().hex[:8]}"


def _slug(value: str) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")
    return slug or "object"


def _positions_match(expected: Any, observed: Any, tolerance: float) -> bool:
    if expected is None or observed is None or len(expected) != len(observed):
        return False
    if any(value is None for value in expected) or any(value is None for value in observed):
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(expected, observed))


def _validate_finite_pose(pose: Pose) -> None:
    values = [pose.position[axis] for axis in ("x", "y", "z")]
    values.extend(pose.orientation[axis] for axis in ("x", "y", "z", "w"))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Pose position and orientation values must be finite numbers")
