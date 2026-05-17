#!/usr/bin/env python3
"""Opt-in service boundary for a MoveIt Task Constructor pick backend."""

import importlib
import json
import traceback

import rospy
import rospkg
from std_srvs.srv import Trigger, TriggerResponse

SERVICE_NAME = "/vizor_mtc/plan_pick_task"
REQUEST_PARAM = f"{SERVICE_NAME}/request"
COMPOUND_SERVICE_NAME = "/vizor_mtc/plan_compound_task"
COMPOUND_REQUEST_PARAM = f"{COMPOUND_SERVICE_NAME}/request"
MTC_PACKAGES = (
    "moveit_task_constructor_core",
    "moveit_task_constructor_msgs",
)
MTC_PYTHON_MODULES = (
    "moveit.task_constructor",
    "pymoveit_mtc.core",
    "pymoveit_mtc.stages",
)
MTC_PICK_STAGE_TEMPLATE = (
    ("current_state", "CurrentState", "capture current planning scene and robot state"),
    ("connect_to_grasp", "Connect", "connect current state to the grasp branch"),
    ("generate_grasp_pose", "GenerateGraspPose", "sample object-relative grasp poses"),
    ("compute_grasp_ik", "ComputeIK", "wrap generated grasp poses with IK candidates"),
    ("approach_object", "MoveRelative", "approach along the configured gripper frame"),
    ("allow_gripper_object_collision", "ModifyPlanningScene", "allow gripper/object contact"),
    ("attach_object", "ModifyPlanningScene", "attach the object after grasp execution"),
    ("lift_object", "MoveRelative", "lift the attached object"),
)
COMPOUND_TASK_GOALS = ("hold", "release", "move_and_release", "pick_place")
COMPOUND_TASK_TARGET_GOALS = ("move_and_release", "pick_place")
COMPOUND_SUPPORT_TEXT = "Supported compound goals are hold, release, move_and_release, and pick_place."


def _compound_stage(intent):
    return {
        "intent": intent,
        "stage_type": "CompoundIntent",
        "status": "not_started",
        "message": "Real compound MTC task construction is not implemented in this proof service.",
    }


def _stage(name, stage_type, status, message, **extra):
    payload = {"name": name, "stage_type": stage_type, "status": status, "message": message}
    payload.update(extra)
    return payload


def _mtc_stage_summaries(status, message):
    return [
        _stage(name, stage_type, status, message, role=role)
        for name, stage_type, role in MTC_PICK_STAGE_TEMPLATE
    ]


def _gripper_responsibility():
    return {
        "open": "mtc_pregrasp_posture",
        "close": "execute_task_solution",
        "verification": "caller_after_execute",
    }


def _attach_responsibility():
    return {
        "attach": "mtc_modify_planning_scene",
        "detach": "separate_place_or_release_task",
        "verification": "caller_after_execute",
    }


def _blocked_candidate_attempt(robot_name, object_name, grasp_face, failed_stage, blocker):
    return {
        "candidate_index": 0,
        "backend": "mtc",
        "robot_name": robot_name,
        "object_name": object_name,
        "grasp_face": grasp_face,
        "ok": False,
        "failed_stage": failed_stage,
        "cost": None,
        "blocker": blocker,
    }


def _base_response(
    *,
    robot_name,
    object_name,
    grasp_face,
    failed_stage,
    message,
    blocker,
    correction,
    stage_summaries,
    candidate_attempts,
    availability=None,
):
    return {
        "ok": False,
        "backend": "mtc",
        "task_kind": "pick",
        "task_solution_id": "",
        "failed_stage": failed_stage,
        "message": message,
        "blocker": blocker,
        "correction": correction,
        "stage_summaries": stage_summaries,
        "candidate_attempts": candidate_attempts,
        "candidate_count": 0,
        "selected_cost": None,
        "selected_grasp_face": grasp_face,
        "robot_name": robot_name,
        "object_name": object_name,
        "grasp_face": grasp_face,
        "gripper_responsibility": _gripper_responsibility(),
        "attach_responsibility": _attach_responsibility(),
        "availability": availability or {},
    }


def _load_request(request_param=REQUEST_PARAM):
    value = rospy.get_param(request_param, {})
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return value if isinstance(value, dict) else {}


def _string_value(payload, key):
    value = payload.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _string_list_value(payload, key):
    value = payload.get(key)
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _dict_value(payload, key):
    value = payload.get(key)
    return value if isinstance(value, dict) else None


def _compound_execution_contract():
    return {
        "target_kind": "task_solution",
        "requires_explicit_approval": True,
        "can_execute": False,
        "execute_tool": "moveit_execute_task_solution",
    }


def _compound_preview_unavailable(failed_stage):
    return {
        "solution_topic": "/solution",
        "solution_preview": "not_published",
        "ar_preview_service": "/vizor_robot_control",
        "ar_preview_mode": "unavailable",
        "failed_stage": failed_stage,
    }


def _compound_failure_response(
    *,
    robot_name,
    object_name,
    task_goal,
    requirements,
    preferences,
    stage_intents,
    target_pose,
    target_position,
    failed_stage,
    error,
    message,
    blocker,
    correction,
    task_stages=None,
    candidate_attempts=None,
    availability=None,
):
    response = {
        "ok": False,
        "backend": "mtc",
        "task_kind": "compound",
        "failed_stage": failed_stage,
        "error": error,
        "message": message,
        "blocker": blocker,
        "correction": correction,
        "robot_name": robot_name,
        "object_name": object_name,
        "task_goal": task_goal,
        "requirements": requirements,
        "preferences": preferences,
        "stage_intents": stage_intents,
        "target_pose": target_pose,
        "target_position": target_position,
        "task_stages": list(task_stages or []),
        "candidate_attempts": list(candidate_attempts or []),
        "candidate_count": 0,
        "selected_cost": None,
        "scene_snapshot": {},
        "object_context": {},
        "selected_stage_evidence": [],
        "selected_grasp_evidence": {},
        "selected_place_evidence": {},
        "execution_contract": _compound_execution_contract(),
        "preview": _compound_preview_unavailable(failed_stage),
    }
    if availability is not None:
        response["availability"] = availability
    return response


def _check_mtc_availability():
    rospack = rospkg.RosPack()
    missing_packages = []
    package_paths = {}
    for package in MTC_PACKAGES:
        try:
            package_paths[package] = rospack.get_path(package)
        except rospkg.ResourceNotFound:
            missing_packages.append(package)

    imported_modules = []
    missing_modules = []
    module_errors = {}
    for module_name in MTC_PYTHON_MODULES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            missing_modules.append(module_name)
            module_errors[module_name] = str(exc)
        else:
            imported_modules.append(module_name)

    return {
        "package_paths": package_paths,
        "missing_packages": missing_packages,
        "imported_modules": imported_modules,
        "missing_modules": missing_modules,
        "module_errors": module_errors,
    }


def _has_mtc_python_api(availability):
    return all(module_name in availability["imported_modules"] for module_name in MTC_PYTHON_MODULES)


def _response_payload(robot_name, object_name, grasp_face):
    stages = [
        _stage("read_request", "ServiceBoundary", "solved", "Read request from ROS param because no custom srv package exists yet."),
    ]
    if not robot_name or not object_name:
        blocker = "robot_name and object_name are required before constructing an MTC task."
        stages.append(_stage("validate_request", "ServiceBoundary", "failed", blocker))
        return _base_response(
            robot_name=robot_name,
            object_name=object_name,
            grasp_face=grasp_face,
            failed_stage="validate_request",
            message="Missing robot_name or object_name.",
            blocker=blocker,
            correction="Call the service with robot_name and object_name.",
            stage_summaries=stages + _mtc_stage_summaries("not_started", "Request validation failed before MTC construction."),
            candidate_attempts=[],
        )

    availability = _check_mtc_availability()
    if availability["missing_packages"]:
        blocker = "Missing MTC ROS packages: " + ", ".join(availability["missing_packages"])
        stages.append(
            _stage(
                "check_mtc_packages",
                "ServiceBoundary",
                "failed",
                blocker,
            )
        )
        failed_stage = "check_mtc_packages"
        message = "MoveIt Task Constructor packages are not available in this image."
        correction = "Install the Noetic MTC packages in the image and rebuild before enabling this backend."
    elif not _has_mtc_python_api(availability):
        blocker = (
            "MTC ROS packages are installed, but the documented Python API modules are not importable: "
            + ", ".join(availability["missing_modules"])
        )
        stages.append(_stage("check_mtc_packages", "ServiceBoundary", "solved", "Required MTC ROS packages are installed."))
        stages.append(
            _stage(
                "check_mtc_python_api",
                "ServiceBoundary",
                "failed",
                blocker,
            )
        )
        failed_stage = "check_mtc_python_api"
        message = "MTC packages are installed, but no Python task API is available for this proof node."
        correction = "Expose moveit.task_constructor and pymoveit_mtc Python bindings, or provide a typed C++ MTC service."
    else:
        blocker = (
            "MTC Python API is importable, but this node has no typed pick service, UR10/Robotiq semantic binding, "
            "eef/group/ik_frame mapping, named gripper postures, or object-frame policy."
        )
        stages.append(_stage("check_mtc_packages", "ServiceBoundary", "solved", "Required MTC ROS packages are installed."))
        stages.append(_stage("check_mtc_python_api", "ServiceBoundary", "solved", "Documented MTC Python API modules are importable."))
        stages.append(
            _stage(
                "construct_pick_task",
                "ServiceBoundary",
                "failed",
                blocker,
            )
        )
        failed_stage = "construct_pick_task"
        message = "MTC backend boundary is reachable; real pick task construction is blocked by semantic configuration."
        correction = "Add a typed pick srv plus semantic config for group, eef, hand frame, gripper postures, and object frames."

    return _base_response(
        robot_name=robot_name,
        object_name=object_name,
        grasp_face=grasp_face,
        failed_stage=failed_stage,
        message=message,
        blocker=blocker,
        correction=correction,
        stage_summaries=stages + _mtc_stage_summaries("not_started", "MTC task was not constructed; no stage solution exists."),
        candidate_attempts=[
            _blocked_candidate_attempt(robot_name, object_name, grasp_face, failed_stage, blocker),
        ],
        availability=availability,
    )


def _compound_response_payload(payload):
    robot_name = _string_value(payload, "robot_name")
    backend = _string_value(payload, "backend")
    requirements = _dict_value(payload, "requirements") or {}
    preferences = _dict_value(payload, "preferences") or {}
    object_name = _string_value(requirements, "object_name") or _string_value(payload, "object_name")
    task_goal = _string_value(requirements, "goal") or _string_value(payload, "task_goal")
    stage_intents = _string_list_value(payload, "stage_intents")
    target_pose = _dict_value(requirements, "target_pose") or _dict_value(payload, "target_pose")
    target_position = _dict_value(requirements, "target_position") or _dict_value(payload, "target_position")
    if backend != "mtc":
        blocker = 'Compound MTC planning requires backend="mtc".'
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="validate_compound_backend",
            error="mtc_backend_required",
            message="Unsupported compound MTC backend.",
            blocker=blocker,
            correction='Retry with backend="mtc"; no fallback backend is available.',
        )

    if not robot_name or not object_name or not task_goal:
        blocker = "robot_name, requirements.goal, and requirements.object_name are required before constructing a compound MTC task."
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="validate_compound_request",
            error="invalid_compound_request",
            message="Missing compound MTC request fields.",
            blocker=blocker,
            correction="Call the service with robot_name and requirements containing goal and object_name.",
        )

    if task_goal not in COMPOUND_TASK_GOALS:
        blocker = f"Unsupported compound MTC goal: {task_goal}."
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="validate_compound_goal",
            error="unsupported_compound_goal",
            message="Unsupported compound MTC goal.",
            blocker=blocker,
            correction=COMPOUND_SUPPORT_TEXT,
        )

    if task_goal in COMPOUND_TASK_TARGET_GOALS and not (target_pose or target_position):
        blocker = f"Compound MTC goal {task_goal} requires requirements.target_pose or requirements.target_position."
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="validate_compound_target",
            error="missing_compound_target",
            message="Missing compound MTC transfer target.",
            blocker=blocker,
            correction="Retry with requirements.target_pose or requirements.target_position for move_and_release and pick_place.",
        )

    availability = _check_mtc_availability()
    if availability["missing_packages"]:
        blocker = "Missing MTC ROS packages: " + ", ".join(availability["missing_packages"])
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="check_mtc_packages",
            error="mtc_packages_unavailable",
            message="MoveIt Task Constructor packages are not available in this image.",
            blocker=blocker,
            correction="Install the Noetic MTC packages in the image and rebuild before enabling this backend.",
            availability=availability,
        )

    if not _has_mtc_python_api(availability):
        blocker = (
            "MTC ROS packages are installed, but the documented Python API modules are not importable: "
            + ", ".join(availability["missing_modules"])
        )
        return _compound_failure_response(
            robot_name=robot_name,
            object_name=object_name,
            task_goal=task_goal,
            requirements=requirements,
            preferences=preferences,
            stage_intents=stage_intents,
            target_pose=target_pose,
            target_position=target_position,
            failed_stage="check_mtc_python_api",
            error="mtc_python_api_unavailable",
            message="MTC packages are installed, but no Python compound task API is available for this proof node.",
            blocker=blocker,
            correction="Expose moveit.task_constructor and pymoveit_mtc Python bindings, or provide a typed C++ MTC service.",
            availability=availability,
        )

    blocker = (
        "Compound MTC planning is unsupported in this proof service: real task construction, typed request/response, "
        "stage composition, held-object proof, grasp/place semantics, /solution preview export, and AR preview evidence are not implemented."
    )
    if task_goal == "release":
        blocker = (
            "Plain release requires proof that the object is currently held plus real MTC release construction; "
            "this proof service has neither."
        )
    return _compound_failure_response(
        robot_name=robot_name,
        object_name=object_name,
        task_goal=task_goal,
        requirements=requirements,
        preferences=preferences,
        stage_intents=stage_intents,
        target_pose=target_pose,
        target_position=target_position,
        failed_stage="construct_compound_task",
        error="mtc_compound_not_implemented",
        message="MTC compound backend boundary is reachable but incomplete.",
        blocker=blocker,
        correction="Implement a real typed compound MTC service before exposing compound task execution.",
        task_stages=[_compound_stage(intent) for intent in stage_intents],
        availability=availability,
    )


def _handle(_request):
    try:
        payload = _load_request()
        robot_name = _string_value(payload, "robot_name")
        object_name = _string_value(payload, "object_name")
        grasp_face = _string_value(payload, "grasp_face")
        response = _response_payload(robot_name, object_name, grasp_face)
        return TriggerResponse(success=bool(response["ok"]), message=json.dumps(response, sort_keys=True))
    except Exception:
        rospy.logerr("MTC proof service failed:\n%s", traceback.format_exc())
        response = {
            "ok": False,
            "task_solution_id": "",
            "failed_stage": "proof_node_exception",
            "message": "MTC proof service raised an exception.",
            "blocker": "MTC service raised an exception before returning a backend response.",
            "correction": "Inspect /tmp/vizor_mtc_pick_server.log and fix the ROS-side exception before retrying.",
            "stage_summaries": [
                _stage("proof_node_exception", "ServiceBoundary", "failed", traceback.format_exc()),
            ],
            "backend": "mtc",
            "candidate_attempts": [],
            "candidate_count": 0,
            "selected_cost": None,
        }
        return TriggerResponse(success=False, message=json.dumps(response, sort_keys=True))


def _handle_compound(_request):
    try:
        payload = _load_request(COMPOUND_REQUEST_PARAM)
        response = _compound_response_payload(payload)
        return TriggerResponse(success=bool(response["ok"]), message=json.dumps(response, sort_keys=True))
    except Exception:
        rospy.logerr("MTC compound proof service failed:\n%s", traceback.format_exc())
        response = {
            "ok": False,
            "backend": "mtc",
            "task_kind": "compound",
            "failed_stage": "proof_node_exception",
            "error": "proof_node_exception",
            "message": "MTC compound proof service raised an exception.",
            "blocker": "MTC compound service raised an exception before returning a backend response.",
            "correction": "Inspect /tmp/vizor_mtc_pick_server.log and fix the ROS-side exception before retrying.",
            "requirements": {},
            "preferences": {},
            "stage_intents": [],
            "task_stages": [
                _stage("proof_node_exception", "ServiceBoundary", "failed", traceback.format_exc()),
            ],
            "candidate_attempts": [],
            "candidate_count": 0,
            "selected_cost": None,
            "scene_snapshot": {},
            "object_context": {},
            "selected_stage_evidence": [],
            "selected_grasp_evidence": {},
            "selected_place_evidence": {},
            "execution_contract": _compound_execution_contract(),
            "preview": _compound_preview_unavailable("proof_node_exception"),
        }
        return TriggerResponse(success=False, message=json.dumps(response, sort_keys=True))


def main():
    rospy.init_node("vizor_mtc_pick_server", anonymous=False)
    rospy.Service(SERVICE_NAME, Trigger, _handle)
    rospy.Service(COMPOUND_SERVICE_NAME, Trigger, _handle_compound)
    rospy.loginfo(
        "MTC proof service ready at %s. Request inputs are read from ROS param %s.",
        SERVICE_NAME,
        REQUEST_PARAM,
    )
    rospy.loginfo(
        "MTC compound proof service ready at %s. Request inputs are read from ROS param %s.",
        COMPOUND_SERVICE_NAME,
        COMPOUND_REQUEST_PARAM,
    )
    rospy.logwarn("Typed MTC pick service generation is deferred because no custom srv package exists yet.")
    rospy.spin()


if __name__ == "__main__":
    main()
