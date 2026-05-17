import json
import sys

from moveit_mcp.vizor_client import (
    FakeRosbridgeTransport,
    Pose,
    RoslibpyTransport,
    VizorClient,
    _parse_bool_param_value,
)

CURRENT_POSE = {
    "position": {"x": 0.57, "y": 0.39, "z": 0.62},
    "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
}

PLANNING_SCENE = {
    "scene": {
        "robot_model_name": "UR10",
        "world": {
            "collision_objects": [
                {
                    "id": "beam_001",
                    "header": {"frame_id": "base_link"},
                    "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                    "primitive_poses": [
                        {
                            "position": {"x": 0.4, "y": 0.2, "z": 0.12},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    ],
                    "meshes": [],
                    "mesh_poses": [],
                },
                {
                    "id": "ground_plane",
                    "header": {"frame_id": "base_link"},
                    "primitives": [{"type": 1, "dimensions": [5.0, 5.0, 0.01]}],
                    "primitive_poses": [
                        {
                            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    ],
                    "meshes": [],
                    "mesh_poses": [],
                },
            ]
        },
        "robot_state": {
            "attached_collision_objects": [
                {
                    "link_name": "tool0",
                    "touch_links": ["tool0", "robotiq_finger_tip"],
                    "object": {
                        "id": "held_part",
                        "header": {"frame_id": "tool0"},
                        "primitives": [{"type": 1, "dimensions": [0.05, 0.04, 0.03]}],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.0, "y": 0.0, "z": 0.04},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            }
                        ],
                        "meshes": [],
                        "mesh_poses": [],
                    },
                }
            ]
        },
        "object_colors": [{"id": "beam_001", "color": {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}}],
    }
}


class FakeRoslibpyModule:
    def __init__(self, responses):
        self.responses = responses
        self.service_calls = []

    class Ros:
        def __init__(self, *, host, port):
            self.host = host
            self.port = port

    def Service(self, client, name, service_type):
        module = self

        class FakeService:
            def call(self, request=None, *, timeout=None):
                module.service_calls.append((name, service_type, dict(request or {}), timeout))
                response = module.responses.get(name)
                if isinstance(response, Exception):
                    raise response
                if response is None:
                    raise RuntimeError(f"service unavailable: {name}")
                return response

        return FakeService()

    def ServiceRequest(self, payload=None):
        return dict(payload or {})


class FalseApplyResponseTransport(FakeRosbridgeTransport):
    def apply_planning_scene(self, robot, payload, timeout_s):
        super().apply_planning_scene(robot, payload, timeout_s)
        return False


class UnchangedApplyResponseTransport(FakeRosbridgeTransport):
    def apply_planning_scene(self, robot, payload, timeout_s):
        self.events.append(("apply_planning_scene", robot, timeout_s))
        self.applied_planning_scenes.append(payload)
        return True


def test_get_current_pose_reads_pose_service_without_publishing():
    transport = FakeRosbridgeTransport()
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    result = client.get_current_pose(robot="UR10", timeout_s=0.1)

    assert result.ok is True
    assert result.robot == "UR10"
    assert result.planning_frame == "base_link"
    assert result.pose.to_msg() == CURRENT_POSE
    assert result.status == "current pose observed"
    assert transport.published == []
    assert ("read_current_pose", "UR10", 0.1) in transport.events


def test_get_current_pose_reports_unavailable_pose_service():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)

    result = client.get_current_pose(robot="UR10", timeout_s=0.1)

    assert result.ok is False
    assert result.pose is None
    assert result.status == "current pose unavailable"


def test_get_robot_state_combines_pose_physical_mode_and_joint_state():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", [0, -1.57, 1.57, 0, 0, 0])
    client = VizorClient(transport=transport)

    result = client.get_robot_state(robot="UR10", timeout_s=0.1)

    assert result.ok is True
    assert result.robot == "UR10"
    assert result.pose is not None
    assert result.planning_frame == "base_link"
    assert result.physical_mode is False
    assert result.joint_state == [0, -1.57, 1.57, 0, 0, 0]
    assert transport.published == []
    assert ("read_joint_state", "/UR10/move_group/fake_controller_joint_states", 0.1) in transport.events


def test_get_robot_state_reports_incomplete_state_when_joint_state_is_missing():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    result = client.get_robot_state(robot="UR10", timeout_s=0.0)

    assert result.ok is False
    assert result.status == "robot state incomplete"
    assert result.pose is not None
    assert result.physical_mode is False
    assert result.joint_state is None


def test_list_scene_objects_reads_planning_scene_without_publishing():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    result = client.list_scene_objects(robot="UR10", timeout_s=0.1)

    assert result.ok is True
    assert result.planning_frame == "base_link"
    assert [obj["name"] for obj in result.objects] == ["beam_001", "ground_plane", "held_part"]
    beam = result.objects[0]
    assert beam["state"] == "free"
    assert beam["frame"] == "base_link"
    assert beam["pose"]["position"] == {"x": 0.4, "y": 0.2, "z": 0.12}
    assert beam["bounds"]["center"] == {"x": 0.4, "y": 0.2, "z": 0.12}
    assert beam["bounds"]["size"] == {"x": 0.3, "y": 0.04, "z": 0.04}
    assert beam["shapes"][0]["kind"] == "box"
    assert beam["color"] == {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
    assert result.objects[2]["state"] == "attached"
    assert result.objects[2]["attached_to"] == "tool0"
    assert transport.published == []
    assert ("read_planning_scene", "UR10", 0.1) in transport.events


def test_get_object_context_returns_bounds_faces_and_clearance_for_named_object():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    result = client.get_object_context(robot="UR10", object_name="beam_001", timeout_s=0.1)

    assert result.ok is True
    assert result.object_context["name"] == "beam_001"
    assert result.object_context["planning_frame"] == "base_link"
    assert result.object_context["bounds"]["min"] == {"x": 0.25, "y": 0.18, "z": 0.1}
    assert result.object_context["bounds"]["max"] == {"x": 0.55, "y": 0.22, "z": 0.14}
    assert result.object_context["clearance"]["reference"] == "ground_plane"
    assert result.object_context["clearance"]["z_m"] == 0.095
    face_names = {face["name"] for face in result.object_context["grasp_faces"]}
    assert {"top", "front", "right"}.issubset(face_names)
    top_face = next(face for face in result.object_context["grasp_faces"] if face["name"] == "top")
    assert top_face["alignment_axis"] == {"x": 1.0, "y": 0.0, "z": 0.0}


def test_get_object_context_reports_missing_object_with_available_names():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    result = client.get_object_context(robot="UR10", object_name="missing", timeout_s=0.1)

    assert result.ok is False
    assert result.status == "object not found"
    assert result.available_objects == ["beam_001", "ground_plane", "held_part"]


def test_plan_mtc_pick_task_preserves_structured_backend_response():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_mtc_pick_task_result(
        {
            "ok": True,
            "backend": "mtc",
            "task_solution_id": "mtc_pick_42",
            "stage_summaries": [
                {"name": "current_state", "stage_type": "CurrentState", "status": "solved"},
                {"name": "connect_to_grasp", "stage_type": "Connect", "status": "solved"},
                {"name": "generate_grasp_pose", "stage_type": "GenerateGraspPose", "status": "solved"},
                {"name": "compute_grasp_ik", "stage_type": "ComputeIK", "status": "solved"},
                {"name": "approach_object", "stage_type": "MoveRelative", "status": "solved"},
                {"name": "attach_object", "stage_type": "ModifyPlanningScene", "status": "solved"},
            ],
            "candidate_attempts": [
                {"grasp_face": "top", "ok": True, "cost": 3.25},
                {"grasp_face": "front", "ok": False, "failed_stage": "compute_grasp_ik"},
            ],
            "candidate_count": 2,
            "selected_cost": 3.25,
            "selected_grasp_face": "top",
            "gripper_responsibility": {"close": "execute_task_solution"},
            "attach_responsibility": {"attach": "mtc_modify_planning_scene"},
        }
    )

    result = client.plan_mtc_pick_task(
        robot="UR10",
        object_name="beam_001",
        grasp_face="top",
        timeout_s=0.1,
    )

    assert result["backend"] == "mtc"
    assert result["task_solution_id"] == "mtc_pick_42"
    assert [stage["stage_type"] for stage in result["stage_summaries"]] == [
        "CurrentState",
        "Connect",
        "GenerateGraspPose",
        "ComputeIK",
        "MoveRelative",
        "ModifyPlanningScene",
    ]
    assert result["candidate_attempts"][0]["cost"] == 3.25
    assert result["candidate_count"] == 2
    assert result["selected_cost"] == 3.25
    assert result["selected_grasp_face"] == "top"
    assert result["gripper_responsibility"] == {"close": "execute_task_solution"}
    assert result["attach_responsibility"] == {"attach": "mtc_modify_planning_scene"}
    assert result["robot_name"] == "UR10"
    assert result["object_name"] == "beam_001"


def test_plan_mtc_pick_task_unavailable_fallback_uses_structured_mtc_shape():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)

    result = client.plan_mtc_pick_task(
        robot="UR10",
        object_name="beam_001",
        grasp_face="front",
        timeout_s=0.0,
    )

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["failed_stage"] == "mtc_service_unavailable"
    assert result["candidate_attempts"] == []
    assert result["candidate_count"] == 0
    assert result["selected_cost"] is None
    assert result["selected_grasp_face"] == "front"
    assert "blocker" in result
    assert "correction" in result
    assert result["gripper_responsibility"]["close"] == "not_planned"
    assert result["attach_responsibility"]["attach"] == "not_planned"


def test_plan_mtc_compound_task_preserves_requirements_preferences_contract():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    requirements = {
        "goal": "pick_place",
        "object_name": "beam_001",
        "target_position": {"x": 0.5, "y": 0.1, "z": 0.2},
        "must_verify_release": True,
    }
    preferences = {
        "grasp_face": "top",
        "orientation_mode": "keep",
        "retreat_distance_m": 0.08,
    }
    transport.queue_mtc_compound_task_result(
        {
            "ok": True,
            "backend": "mtc",
            "task_kind": "compound",
            "task_solution_id": "mtc_compound_42",
            "task_stages": [
                {"intent": "pick", "stage_type": "Pick", "status": "solved"},
                {"intent": "place", "stage_type": "Place", "status": "solved"},
            ],
            "candidate_attempts": [{"candidate_index": 0, "ok": True, "cost": 6.5}],
            "candidate_count": 1,
            "selected_cost": 6.5,
            "scene_snapshot": {"id": "scene_42", "object_count": 3},
            "object_context": {"name": "beam_001", "state": "free"},
            "selected_stage_evidence": [{"intent": "pick", "solution_index": 0}],
            "selected_grasp_evidence": {"grasp_face": "top"},
            "selected_place_evidence": {"target_position": {"x": 0.5, "y": 0.1, "z": 0.2}},
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "execute_tool": "moveit_execute_task_solution",
            },
        }
    )

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements=requirements,
        preferences=preferences,
        stage_intents=["approach_object", "release_object"],
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["backend"] == "mtc"
    assert result["task_kind"] == "compound"
    assert result["task_solution_id"] == "mtc_compound_42"
    assert result["robot_name"] == "UR10"
    assert result["object_name"] == "beam_001"
    assert result["task_goal"] == "pick_place"
    assert result["requirements"] == requirements
    assert result["preferences"] == preferences
    assert result["stage_intents"] == ["approach_object", "release_object"]
    assert result["target_position"] == {"x": 0.5, "y": 0.1, "z": 0.2}
    assert result["task_stages"][0]["intent"] == "pick"
    assert result["candidate_count"] == 1
    assert result["selected_cost"] == 6.5
    assert result["failed_stage"] is None
    assert result["blocker"] is None
    assert result["scene_snapshot"]["id"] == "scene_42"
    assert result["object_context"]["name"] == "beam_001"
    assert result["selected_stage_evidence"] == [{"intent": "pick", "solution_index": 0}]
    assert result["selected_grasp_evidence"] == {"grasp_face": "top"}
    assert result["selected_place_evidence"]["target_position"] == {"x": 0.5, "y": 0.1, "z": 0.2}
    assert result["execution_contract"]["target_kind"] == "task_solution"
    assert (
        "plan_mtc_compound_task",
        "UR10",
        requirements,
        preferences,
        ("approach_object", "release_object"),
        "mtc",
        0.1,
    ) in transport.events


def test_plan_mtc_compound_task_adds_agent_path_preview_names():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_mtc_compound_task_result(
        {
            "ok": True,
            "task_solution_id": "mtc_compound_42",
            "task_stages": [
                {"name": "approach", "kind": "motion"},
                {"name": "close_gripper", "kind": "gripper"},
                {"name": "lift object", "kind": "motion"},
            ],
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
            },
        }
    )

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        timeout_s=0.1,
    )

    assert result["preview"]["public_name"] == "AgentPath"
    assert result["preview"]["stage_debug_names"] == [
        "AgentPath:01_approach",
        "AgentPath:02_close_gripper",
        "AgentPath:03_lift_object",
    ]
    assert result["execution_contract"]["agent_path_name"] == "AgentPath"
    assert result["execution_contract"]["approval_signal"] == {
        "topic": "/UR10/command/execute",
        "payload": "AgentPath",
    }


def test_plan_mtc_compound_task_preserves_long_ordered_agent_path_preview_names():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    stage_debug_names = [
        "AgentPath:01_approach_object",
        "AgentPath:02_descend_to_grasp",
        "AgentPath:03_close_gripper",
        "AgentPath:04_lift_object",
        "AgentPath:05_transfer_to_place",
        "AgentPath:06_release_object",
        "AgentPath:07_retreat",
    ]
    transport.queue_mtc_compound_task_result(
        {
            "ok": True,
            "task_solution_id": "mtc_compound_42",
            "task_stages": [
                {"name": "approach", "kind": "motion"},
                {"name": "lift", "kind": "motion"},
            ],
            "preview": {
                "public_name": "AgentPath",
                "stage_debug_names": list(stage_debug_names),
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
            },
        }
    )

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements={"goal": "pick_place", "object_name": "beam_001"},
        timeout_s=0.1,
    )

    assert result["preview"]["public_name"] == "AgentPath"
    assert result["preview"]["stage_debug_names"] == stage_debug_names


def test_plan_mtc_compound_task_unavailable_fails_without_solution_id():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    requirements = {"goal": "hold", "object_name": "beam_001"}
    preferences = {"grasp_face": "top"}

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements=requirements,
        preferences=preferences,
        timeout_s=0.0,
    )

    assert result["ok"] is False
    assert result["backend"] == "mtc"
    assert result["task_kind"] == "compound"
    assert result["failed_stage"] == "mtc_service_unavailable"
    assert result["candidate_count"] == 0
    assert result["selected_cost"] is None
    assert result["task_stages"] == []
    assert result["scene_snapshot"] == {}
    assert result["object_context"] == {}
    assert result["selected_stage_evidence"] == []
    assert result["selected_grasp_evidence"] == {}
    assert result["selected_place_evidence"] == {}
    assert result["requirements"] == requirements
    assert result["preferences"] == preferences
    assert result["stage_intents"] == []
    assert result["execution_contract"]["can_execute"] is False
    assert "blocker" in result
    assert "correction" in result
    assert "task_solution_id" not in result


def test_plan_mtc_compound_task_failure_strips_executable_contract_and_solution_id():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_mtc_compound_task_result(
        {
            "ok": False,
            "backend": "mtc",
            "failed_stage": "preview_solution",
            "error": "mtc_solution_preview_unavailable",
            "correction": "Fix preview generation before retrying.",
            "task_solution_id": "failed_solution_id",
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "motion",
                        "source_stage": "approach",
                        "required_proof": "mtc_stage_solution",
                    }
                ],
            },
        }
    )

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["execution_contract"]["can_execute"] is False
    assert "steps" not in result["execution_contract"]
    assert result["error"] == "mtc_solution_preview_unavailable"
    assert result["correction"] == "Fix preview generation before retrying."
    assert "task_solution_id" not in result


def test_plan_mtc_compound_task_failure_replaces_contract_list_with_non_executable_contract():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_mtc_compound_task_result(
        {
            "ok": False,
            "backend": "mtc",
            "failed_stage": "solve_task",
            "execution_contract": [
                {
                    "handler": "motion",
                    "source_stage": "approach",
                    "required_proof": "mtc_stage_solution",
                }
            ],
        }
    )

    result = client.plan_mtc_compound_task(
        robot="UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["execution_contract"] == {
        "target_kind": "task_solution",
        "requires_explicit_approval": True,
        "can_execute": False,
    }
    assert "task_solution_id" not in result


def test_roslibpy_compound_task_request_uses_requirements_preferences_contract(monkeypatch):
    response_payload = {
        "ok": False,
        "backend": "mtc",
        "failed_stage": "construct_compound_task",
        "blocker": "proof boundary only",
        "execution_contract": {"can_execute": False},
    }
    fake_roslibpy = FakeRoslibpyModule(
        {
            "/rosapi/set_param": {"success": True},
            "/vizor_mtc/plan_compound_task": {"success": False, "message": json.dumps(response_payload)},
        }
    )
    monkeypatch.setitem(sys.modules, "roslibpy", fake_roslibpy)
    transport = RoslibpyTransport()
    requirements = {
        "goal": "move_and_release",
        "object_name": "beam_001",
        "target_position": {"x": 0.5, "y": 0.1, "z": 0.2},
    }
    preferences = {"grasp_face": "top"}

    result = transport.plan_mtc_compound_task(
        "UR10",
        requirements,
        preferences,
        None,
        "mtc",
        0.1,
    )

    assert result == response_payload
    assert fake_roslibpy.service_calls[0][0] == "/rosapi/set_param"
    request_payload = fake_roslibpy.service_calls[0][2]
    assert request_payload["name"] == "/vizor_mtc/plan_compound_task/request"
    stored_request = json.loads(request_payload["value"])
    assert stored_request == {
        "backend": "mtc",
        "preferences": preferences,
        "requirements": requirements,
        "robot_name": "UR10",
    }
    assert fake_roslibpy.service_calls[1][0] == "/vizor_mtc/plan_compound_task"


def test_apply_planning_scene_uses_robot_namespaced_service_first(monkeypatch):
    fake_roslibpy = FakeRoslibpyModule({"/UR10/apply_planning_scene": {"success": True}})
    monkeypatch.setitem(sys.modules, "roslibpy", fake_roslibpy)
    transport = RoslibpyTransport()

    result = transport.apply_planning_scene("UR10", {"is_diff": True}, timeout_s=0.1)

    assert result is True
    assert fake_roslibpy.service_calls == [
        (
            "/UR10/apply_planning_scene",
            "moveit_msgs/ApplyPlanningScene",
            {"scene": {"is_diff": True}},
            0.1,
        )
    ]


def test_apply_planning_scene_falls_back_to_global_service(monkeypatch):
    fake_roslibpy = FakeRoslibpyModule(
        {
            "/UR10/apply_planning_scene": RuntimeError("missing service"),
            "/apply_planning_scene": {"success": True},
        }
    )
    monkeypatch.setitem(sys.modules, "roslibpy", fake_roslibpy)
    transport = RoslibpyTransport()

    result = transport.apply_planning_scene("UR10", {"is_diff": True}, timeout_s=0.1)

    assert result is True
    assert [call[0] for call in fake_roslibpy.service_calls] == [
        "/UR10/apply_planning_scene",
        "/apply_planning_scene",
    ]


def test_attach_object_updates_service_read_object_context():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    attached = client.attach_object(robot="UR10", object_name="beam_001", timeout_s=0.1)
    context = client.get_object_context(robot="UR10", object_name="beam_001", timeout_s=0.1)

    assert attached.ok is True
    assert context.ok is True
    assert context.object_context["state"] == "attached"
    assert context.object_context["attached_to"] == "tool0"
    beam_states = [
        item["state"]
        for item in client.list_scene_objects(robot="UR10", timeout_s=0.1).objects
        if item["name"] == "beam_001"
    ]
    assert beam_states == ["attached"]


def test_attach_object_trusts_verified_scene_when_apply_response_is_false():
    transport = FalseApplyResponseTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    attached = client.attach_object(robot="UR10", object_name="beam_001", timeout_s=0.1)
    context = client.get_object_context(robot="UR10", object_name="beam_001", timeout_s=0.1)

    assert attached.ok is True
    assert attached.status == "attached collision object verified"
    assert context.object_context["state"] == "attached"
    assert context.object_context["attached_to"] == "tool0"


def test_attach_object_fails_when_apply_response_does_not_change_service_scene():
    transport = UnchangedApplyResponseTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    attached = client.attach_object(robot="UR10", object_name="beam_001", timeout_s=0.1)
    context = client.get_object_context(robot="UR10", object_name="beam_001", timeout_s=0.1)

    assert attached.ok is False
    assert attached.status == "attached collision object unverified"
    assert context.object_context["state"] == "free"


def test_detach_object_translates_mesh_center_to_target_pose():
    mesh_object = {
        "id": "mesh_post",
        "header": {"frame_id": "tool0"},
        "primitives": [],
        "primitive_poses": [],
        "meshes": [
            {
                "vertices": [
                    {"x": 0.24, "y": -0.67, "z": 0.0},
                    {"x": 0.28, "y": -0.63, "z": 0.2},
                ],
                "triangles": [],
            }
        ],
        "mesh_poses": [],
    }
    scene = {
        "scene": {
            "world": {"collision_objects": []},
            "robot_state": {
                "attached_collision_objects": [
                    {"link_name": "tool0", "object": mesh_object, "touch_links": ["tool0"]}
                ]
            },
            "object_colors": [],
        }
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", scene, planning_frame="base_link")
    client = VizorClient(transport=transport)

    detached = client.detach_object(
        robot="UR10",
        object_name="mesh_post",
        object_pose=Pose.from_input(
            {
                "position": {"x": 0.42, "y": -0.55, "z": 0.16},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        ),
        timeout_s=0.1,
    )
    context = client.get_object_context(robot="UR10", object_name="mesh_post", timeout_s=0.1)

    assert detached.ok is True
    assert context.object_context["state"] == "free"
    assert context.object_context["bounds"]["center"] == {"x": 0.42, "y": -0.55, "z": 0.16}


def test_remove_scene_object_removes_free_world_object_with_readback():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    removed = client.remove_scene_object(robot="UR10", object_name="beam_001", timeout_s=0.1)
    objects = client.list_scene_objects(robot="UR10", timeout_s=0.1).objects

    assert removed.ok is True
    assert removed.status == "scene object removed"
    assert "beam_001" not in {item["name"] for item in objects}
    assert transport.applied_planning_scenes[-1]["world"]["collision_objects"] == [
        {"id": "beam_001", "header": {"frame_id": "base_link"}, "operation": 1}
    ]


def test_remove_scene_object_refuses_attached_object():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    client = VizorClient(transport=transport)

    removed = client.remove_scene_object(robot="UR10", object_name="held_part", timeout_s=0.1)

    assert removed.ok is False
    assert removed.status == "object attached"
    assert "release" in removed.message
    assert transport.applied_planning_scenes == []



def test_plan_free_prepares_before_publish_and_stores_final_positions():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_status("/UR10/request/status", "stale status")
    transport.queue_planned_path(
        "/UR10/request/planned_path",
        name="stale_plan",
        points=1,
        final_positions=[9, 9, 9],
    )
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="plan_a",
        points=3,
        final_positions=[0, -1.57, 1.57, 0, 0, 0],
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")

    result = client.plan_free_motion(
        robot="UR10",
        name="plan_a",
        pose=Pose.position_only(0.5, 0.0, 0.3),
        timeout_s=0.1,
    )

    assert result.status == "success! "
    assert result.trajectory_points == 3
    assert result.final_joint_positions == [0, -1.57, 1.57, 0, 0, 0]
    assert result.can_execute is True
    assert transport.published[-1][0] == "/UR10/request/free"
    assert transport.published[-1][1]["name"] == "plan_a"
    assert transport.events.index(("prepare_for_plan", "/UR10/request/status", "/UR10/request/planned_path")) < transport.events.index(("publish", "/UR10/request/free"))


def test_wait_for_planned_path_loops_past_non_matching_names():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="other_plan",
        points=1,
        final_positions=[1, 1, 1],
    )
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="plan_b",
        points=2,
        final_positions=[2, 2, 2],
    )

    result = client.plan_cartesian_motion(
        robot="UR10",
        name="plan_b",
        poses=[Pose.position_only(0.4, 0.0, 0.3), Pose.position_only(0.5, 0.0, 0.3)],
        timeout_s=0.1,
    )

    assert result.status == "success! "
    assert result.trajectory_points == 2
    assert result.final_joint_positions == [2, 2, 2]
    assert result.can_execute is True


def test_plan_sampled_motion_publishes_sampled_request_and_stores_final_positions():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    poses = [Pose.position_only(0.4, 0.0, 0.3), Pose.position_only(0.5, 0.0, 0.3)]
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="sampled_a",
        points=4,
        final_positions=[0, -1.57, 1.57, 0, 0, 0],
    )

    result = client.plan_sampled_motion(
        robot="UR10",
        name="sampled_a",
        poses=poses,
        timeout_s=0.1,
    )

    assert result.status == "success! "
    assert result.trajectory_points == 4
    assert result.final_joint_positions == [0, -1.57, 1.57, 0, 0, 0]
    assert result.can_execute is True
    assert transport.published[-1] == (
        "/UR10/request/sampled",
        {"name": "sampled_a", "poses": [pose.to_msg() for pose in poses]},
    )


def test_plan_cartesian_incomplete_path_is_not_executable():
    transport = FakeRosbridgeTransport()
    client = VizorClient(transport=transport)
    transport.queue_status_after_publish("/UR10/request/status", "incomplete path")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="plan_c",
        points=2,
        final_positions=[0.1, 0.2],
    )

    result = client.plan_cartesian_motion(
        robot="UR10",
        name="plan_c",
        poses=[Pose.position_only(0.4, 0.0, 0.3), Pose.position_only(0.5, 0.0, 0.3)],
        timeout_s=0.1,
    )

    assert result.status == "incomplete path"
    assert result.trajectory_points == 2
    assert result.can_execute is False


def test_execute_reads_physical_mode_drains_stale_state_and_verifies_final_positions():
    transport = FakeRosbridgeTransport(physical_mode=False)
    client = VizorClient(transport=transport)
    final_positions = [0, -1.57, 1.57, 0, 0, 0]
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="plan_d",
        points=3,
        final_positions=final_positions,
    )
    client.plan_free_motion(
        robot="UR10",
        name="plan_d",
        pose=Pose.position_only(0.5, 0.0, 0.3),
        timeout_s=0.1,
    )
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", [9, 9, 9, 9, 9, 9])
    transport.queue_joint_state_after_publish("/UR10/move_group/fake_controller_joint_states", final_positions)

    result = client.execute_plan(robot="UR10", name="plan_d", timeout_s=0.1)

    assert result.physical_mode is False
    assert result.command_published is True
    assert result.observed_joint_state == final_positions
    assert result.expected_joint_state == final_positions
    assert result.final_positions_match is True
    assert transport.published[-1] == ("/UR10/command/execute", {"data": "plan_d"})
    assert ("read_physical_mode", "/vizor_robot_control/physical") in transport.events
    assert transport.events.index(("prepare_for_execute", "/UR10/move_group/fake_controller_joint_states")) < transport.events.index(("publish", "/UR10/command/execute"))


def test_execute_agent_path_publishes_public_name_for_cached_task():
    transport = FakeRosbridgeTransport(physical_mode=False)
    client = VizorClient(transport=transport)
    final_positions = [0, -1.57, 1.57, 0, 0, 0]
    stage_debug_names = [
        "AgentPath:01_approach_object",
        "AgentPath:02_descend_to_grasp",
        "AgentPath:03_close_gripper",
        "AgentPath:04_lift_object",
        "AgentPath:05_retreat",
    ]
    client.register_agent_path(
        robot="UR10",
        task_solution_id="mtc_compound_42",
        stage_debug_names=stage_debug_names,
        final_joint_positions=final_positions,
    )
    transport.queue_joint_state_after_publish("/UR10/move_group/fake_controller_joint_states", final_positions)

    result = client.execute_plan(robot="UR10", name="AgentPath", timeout_s=0.1)

    assert result.command_published is True
    assert result.name == "AgentPath"
    assert result.expected_joint_state == final_positions
    assert result.final_positions_match is True
    assert client._active_agent_paths["UR10"]["stage_debug_names"] == stage_debug_names
    assert transport.published == [("/UR10/command/execute", {"data": "AgentPath"})]
    assert not any(topic == "/UR10/request/sampled" for topic, _ in transport.published)


def test_stop_agent_path_invalidates_cached_task_until_replanned():
    transport = FakeRosbridgeTransport(physical_mode=False)
    client = VizorClient(transport=transport)
    final_positions = [0, -1.57, 1.57, 0, 0, 0]
    client.register_agent_path(
        robot="UR10",
        task_solution_id="mtc_compound_42",
        stage_debug_names=["AgentPath:01_approach"],
        final_joint_positions=final_positions,
    )

    stopped = client.stop_agent_path(robot="UR10")
    result = client.execute_plan(robot="UR10", name="AgentPath", timeout_s=0.1)

    assert stopped == {
        "ok": True,
        "robot": "UR10",
        "name": "AgentPath",
        "status": "agent path invalidated",
        "requires_reobserve_replan": True,
    }
    assert transport.published[-1] == ("/UR10/command/stop", {"data": "AgentPath"})
    assert not any(topic == "/UR10/request/sampled" for topic, _ in transport.published)
    assert result.command_published is False
    assert result.status == "AgentPath requires re-observe/replan"
    assert [payload for topic, payload in transport.published if topic == "/UR10/command/execute"] == []


def test_execute_refuses_safe_physical_mode_when_plan_final_state_is_missing():
    transport = FakeRosbridgeTransport(physical_mode=False)
    client = VizorClient(transport=transport)

    result = client.execute_plan(robot="UR10", name="missing_final_state", timeout_s=0.1)

    assert result.command_published is False
    assert result.status == "plan final state unavailable"
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_physical_mode_parser_accepts_rosapi_string_values():
    assert _parse_bool_param_value("false") is False
    assert _parse_bool_param_value("true") is True
    assert _parse_bool_param_value("__unknown__") is None


def test_execute_blocks_when_physical_mode_is_unknown_or_true():
    transport = FakeRosbridgeTransport(physical_mode=None)
    client = VizorClient(transport=transport)

    unknown = client.execute_plan(robot="UR10", name="plan_x", timeout_s=0.1)

    assert unknown.command_published is False
    assert unknown.status == "physical mode unknown"
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)

    transport.set_physical_mode(True)
    enabled = client.execute_plan(robot="UR10", name="plan_x", timeout_s=0.1)

    assert enabled.command_published is False
    assert enabled.status == "physical mode enabled"
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)
