import pytest

from moveit_mcp.tools import MoveItMcpTools
from moveit_mcp.vizor_client import FakeRosbridgeTransport

FINAL_POSITIONS = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
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
        "robot_state": {"attached_collision_objects": []},
        "object_colors": [],
    }
}


VERTICAL_BEAM_SCENE = {
    "scene": {
        "robot_model_name": "UR10",
        "world": {
            "collision_objects": [
                {
                    "id": "vertical_beam",
                    "header": {"frame_id": "base_link"},
                    "operation": 0,
                    "primitives": [{"type": 1, "dimensions": [0.04, 0.04, 0.30]}],
                    "primitive_poses": [
                        {
                            "position": {"x": 0.4, "y": 0.2, "z": 0.16},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        }
                    ],
                    "meshes": [],
                    "mesh_poses": [],
                }
            ]
        },
        "robot_state": {"attached_collision_objects": []},
        "object_colors": [],
    }
}


ATTACHED_PLANNING_SCENE = {
    "scene": {
        "robot_model_name": "UR10",
        "world": {
            "collision_objects": [
                PLANNING_SCENE["scene"]["world"]["collision_objects"][1],
            ]
        },
        "robot_state": {
            "attached_collision_objects": [
                {
                    "link_name": "tool0",
                    "object": PLANNING_SCENE["scene"]["world"]["collision_objects"][0],
                    "touch_links": ["tool0", "wrist_3_link"],
                }
            ]
        },
        "object_colors": [],
    }
}


def _vertical_beam_object_context() -> dict:
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", VERTICAL_BEAM_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    object_context = tools.client.get_object_context(robot="UR10", object_name="vertical_beam").object_context
    assert object_context is not None
    return object_context


def _queue_successful_hold_preview(transport: FakeRosbridgeTransport, object_name: str = "vertical_beam") -> None:
    for plan_name in [
        f"manipulation_hold_{object_name}_c01_connect_to_pre_grasp",
        f"manipulation_hold_{object_name}_c01_approach_to_pre_grasp",
        f"manipulation_hold_{object_name}_c01_post_grasp_lift",
    ]:
        transport.queue_status_after_publish("/UR10/request/status", "success! ")
        transport.queue_planned_path_after_publish(
            "/UR10/request/planned_path",
            name=plan_name,
            points=5,
            final_positions=FINAL_POSITIONS,
        )


def test_get_current_pose_returns_read_only_pose_feedback():
    transport = FakeRosbridgeTransport()
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_current_pose("UR10")

    assert result["ok"] is True
    assert result["tool"] == "get_current_pose"
    assert result["feedback"]["can_execute"] is False
    assert result["verification"]["result"] == "pass"
    assert result["raw"] == {
        "planning_frame": "base_link",
        "pose": CURRENT_POSE,
        "source": "/UR10/get_current_pose",
    }


def test_get_robot_state_returns_read_only_observation_feedback():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", FINAL_POSITIONS)
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_robot_state("UR10", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_get_robot_state"
    assert result["feedback"]["can_execute"] is False
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["planning_frame"] == "base_link"
    assert result["raw"]["pose"] == CURRENT_POSE
    assert result["raw"]["physical_mode"] is False
    assert result["raw"]["joint_state"] == FINAL_POSITIONS
    assert transport.published == []


def test_get_robot_state_fails_when_joint_state_feedback_is_missing():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_robot_state("UR10", timeout_s=0.0)

    assert result["ok"] is False
    assert result["tool"] == "moveit_get_robot_state"
    assert result["verification"]["result"] == "fail"
    assert result["raw"]["pose"] == CURRENT_POSE
    assert result["raw"]["physical_mode"] is False
    assert result["raw"]["joint_state"] is None
    checks = {check["name"]: check for check in result["verification"]["checks"]}
    assert checks["current_pose_observed"]["passed"] is True
    assert checks["physical_mode_observed"]["passed"] is True
    assert checks["joint_state_observed"]["passed"] is False
    assert "fake_controller_joint_states" in result["feedback"]["correction"]


def test_list_scene_objects_returns_read_only_scene_summary():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.list_scene_objects("UR10", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_list_scene_objects"
    assert result["feedback"]["can_execute"] is False
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["planning_frame"] == "base_link"
    assert [obj["name"] for obj in result["raw"]["objects"]] == ["beam_001", "ground_plane"]
    assert transport.published == []


def test_get_object_context_returns_grasp_relevant_object_context():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_object_context("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_get_object_context"
    assert result["feedback"]["can_execute"] is False
    assert result["raw"]["object"]["name"] == "beam_001"
    assert result["raw"]["object"]["bounds"]["center"] == {"x": 0.4, "y": 0.2, "z": 0.12}
    assert result["raw"]["object"]["clearance"]["reference"] == "ground_plane"
    assert len(result["raw"]["object"]["grasp_faces"]) == 6


def test_get_object_context_failure_suggests_scene_listing():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.get_object_context("UR10", "missing", timeout_s=0.1)

    assert result["ok"] is False
    assert result["tool"] == "moveit_get_object_context"
    assert result["feedback"]["status"] == "object not found"
    assert result["feedback"]["correction"] == "Call moveit_list_scene_objects, then retry with an object name from raw.objects."
    assert result["raw"]["available_objects"] == ["beam_001", "ground_plane"]


def test_explain_motion_failure_returns_retry_guidance():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.explain_motion_failure(
        "UR10",
        "moveit_plan_cartesian_motion",
        {"ok": False, "feedback": {"status": "incomplete path"}},
        failed_tool_arguments={"waypoints": [{"x": 0.4, "y": 0.0, "z": 0.3}]},
        user_intent="wave to me",
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_explain_motion_failure"
    assert result["retryable"] is True
    assert result["suggested_next_tool"] == "moveit_plan_cartesian_motion"
    assert result["raw"]["category"] == "cartesian_planning_failed"


def test_verify_attached_object_proves_scene_attachment_to_gripper():
    scene = {
        "scene": {
            "robot_state": {
                "attached_collision_objects": [
                    {
                        "link_name": "tool0",
                        "touch_links": ["tool0"],
                        "object": {
                            "id": "beam_001",
                            "header": {"frame_id": "tool0"},
                            "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                            "primitive_poses": [
                                {
                                    "position": {"x": 0.0, "y": 0.0, "z": 0.04},
                                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                                }
                            ],
                        },
                    }
                ]
            }
        }
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", scene, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    result = tools.verify_attached_object("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_verify_attached_object"
    assert result["raw"]["attached_to"] == "tool0"
    assert result["raw"]["moves_with_gripper"] is True


def test_verify_attached_object_fails_without_scene_attachment():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    result = tools.verify_attached_object("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is False
    assert result["tool"] == "moveit_verify_attached_object"
    assert "moveit_verify_attached_object" in result["feedback"]["correction"]


def test_auto_pick_candidates_use_top_only_by_default_for_horizontal_beams():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    object_context = tools.client.get_object_context(robot="UR10", object_name="beam_001").object_context

    from moveit_mcp.pick import build_pick_candidates

    candidates = build_pick_candidates(
        object_context,
        requested_grasp_face=None,
        approach_distance_m=0.08,
        grasp_standoff_m=0.01,
        lift_distance_m=0.1,
        max_candidates=8,
    )

    grasp_faces = [candidate["parameters"]["grasp_face"] for candidate in candidates]
    assert grasp_faces == ["top", "top", "top"]
    assert all(candidate["waypoints"] for candidate in candidates)


def test_auto_pick_candidates_allow_explicit_horizontal_side_preference():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    object_context = tools.client.get_object_context(robot="UR10", object_name="beam_001").object_context

    from moveit_mcp.pick import build_pick_candidates

    candidates = build_pick_candidates(
        object_context,
        requested_grasp_face="front",
        approach_distance_m=0.08,
        grasp_standoff_m=0.01,
        lift_distance_m=0.1,
        max_candidates=8,
    )

    grasp_faces = [candidate["parameters"]["grasp_face"] for candidate in candidates]
    assert grasp_faces == ["front", "front", "front"]
    assert all(candidate["waypoints"] for candidate in candidates)


def test_auto_pick_candidates_use_side_faces_for_vertical_beams_without_required_face():
    object_context = _vertical_beam_object_context()

    from moveit_mcp.pick import build_pick_candidates

    candidates = build_pick_candidates(
        object_context,
        requested_grasp_face="top",
        required_grasp_face=False,
        approach_distance_m=0.08,
        grasp_standoff_m=0.01,
        lift_distance_m=0.1,
        max_candidates=8,
    )

    candidate_faces = [candidate["parameters"]["grasp_face"] for candidate in candidates]
    assert candidate_faces
    assert "top" not in candidate_faces
    assert set(candidate_faces) <= {"front", "back", "left", "right"}
    assert all(candidate["waypoints"] for candidate in candidates)


def test_pick_candidates_respect_required_vertical_top_face():
    object_context = _vertical_beam_object_context()

    from moveit_mcp.pick import build_pick_candidates

    candidates = build_pick_candidates(
        object_context,
        requested_grasp_face="top",
        required_grasp_face=True,
        approach_distance_m=0.08,
        grasp_standoff_m=0.01,
        lift_distance_m=0.1,
        max_candidates=8,
    )

    assert [candidate["parameters"]["grasp_face"] for candidate in candidates] == ["top", "top", "top"]


def test_auto_pick_candidates_skip_vertical_inner_side_facing_neighbor():
    vertical_scene = {
        "scene": {
            "world": {
                "collision_objects": [
                    {
                        "id": "vertical_beam",
                        "operation": 0,
                        "primitives": [
                            {"type": 1, "dimensions": [0.04, 0.04, 0.30]},
                        ],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.2, "z": 0.16},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        ],
                    },
                    {
                        "id": "neighbor_beam",
                        "operation": 0,
                        "primitives": [
                            {"type": 1, "dimensions": [0.04, 0.04, 0.30]},
                        ],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.32, "z": 0.16},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        ],
                    },
                ],
            },
        },
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", vertical_scene, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    object_context = tools.client.get_object_context(robot="UR10", object_name="vertical_beam").object_context

    from moveit_mcp.pick import build_pick_candidates

    candidates = build_pick_candidates(
        object_context,
        requested_grasp_face=None,
        approach_distance_m=0.08,
        grasp_standoff_m=0.01,
        lift_distance_m=0.1,
        max_candidates=8,
    )

    candidate_faces = [candidate["parameters"]["grasp_face"] for candidate in candidates]
    assert candidate_faces
    assert candidate_faces[0] == "back"
    assert "front" not in candidate_faces
    assert set(candidate_faces) <= {"back", "left", "right"}


def test_plan_pick_derives_grasp_waypoints_and_registers_pending_plan():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_beam",
        points=6,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_beam",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_plan_pick"
    assert result["feedback"]["can_execute"] is True
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["plan_name"] == "pick_beam"
    assert result["raw"]["object_name"] == "beam_001"
    assert result["raw"]["selected_grasp_face"]["name"] == "top"
    assert [step["name"] for step in result["raw"]["workflow_steps"]] == [
        "approach",
        "pre_grasp",
        "close_gripper",
        "attach_object",
        "lift",
    ]
    assert result["raw"]["workflow_kind"] == "pick"
    assert result["raw"]["motion_segments"][0]["name"] == "approach_to_pre_grasp"
    assert result["raw"]["motion_segments"][0]["plan_name"] == "pick_beam"
    assert result["raw"]["motion_segments"][0]["waypoint_indexes"] == [0, 1]
    assert result["raw"]["motion_segments"][1]["name"] == "post_grasp_lift"
    assert result["raw"]["motion_segments"][1]["waypoint_indexes"] == [1, 2]
    assert [point["position"] for point in result["raw"]["waypoints"]] == [
        {"x": 0.4, "y": 0.2, "z": 0.22},
        {"x": 0.4, "y": 0.2, "z": 0.15},
        {"x": 0.4, "y": 0.2, "z": 0.25},
    ]
    assert [point["orientation"] for point in result["raw"]["waypoints"]] == [
        {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
        {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
        {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    ]
    assert transport.published[-1][0] == "/UR10/request/cartesian"
    assert transport.published[-1][1]["name"] == "pick_beam"
    assert transport.published[-1][1]["poses"] == result["raw"]["waypoints"][0:2]
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)
    assert transport.action_goals == []


def test_plan_pick_task_returns_emulated_task_solution_with_stage_evidence():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick_task("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_plan_pick_task"
    assert result["feedback"]["can_execute"] is True
    assert result["feedback"]["execution_target"] == "task_solution"
    raw = result["raw"]
    assert raw["task_kind"] == "pick"
    assert raw["backend"] == "emulated"
    assert raw["object_name"] == "beam_001"
    assert raw["robot_name"] == "UR10"
    assert raw["created_from_tool"] == "moveit_plan_pick_task"
    assert raw["planning_frame"] == "base_link"
    assert raw["object_pose_age_s"] >= 0.0
    assert raw["solver"] == "emulated_mtc_stages"
    assert len(raw["candidate_attempts"]) > 1
    assert raw["candidate_attempts"][0]["selected"] is True
    assert raw["candidate_attempts"][0]["grasp_face"] == "top"
    assert raw["selected_cost"] > 0.0
    assert raw["clearance_m"] == 0.095
    assert [stage["name"] for stage in raw["stages"]] == [
        "observe_current_state",
        "connect_to_pre_grasp",
        "approach_grasp",
        "close_gripper",
        "attach_object",
        "lift_object",
        "verify_attached_object",
    ]
    assert all(stage["status"] == "solved" for stage in raw["stages"])
    assert raw["stage_report"]["solved"] == 7
    assert raw["approval"]["required"] is True
    assert raw["approval"]["target_kind"] == "task_solution"
    assert raw["approval"]["task_solution_id"] == raw["task_solution_id"]
    assert raw["evidence"] == [
        {"kind": "scene_snapshot", "id": raw["scene_snapshot_id"]},
        {"kind": "stage_report", "count": 7},
    ]
    assert raw["task_solution_id"] in tools._task_solutions
    assert not any(event[0] == "plan_mtc_pick_task" for event in transport.events)
    assert transport.published == []


def test_plan_manipulation_task_hold_uses_required_grasp_face() -> None:
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", VERTICAL_BEAM_SCENE, planning_frame="base_link")
    _queue_successful_hold_preview(transport)
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={
            "goal": "hold",
            "object_name": "vertical_beam",
            "grasp_face": "top",
            "lift_distance_m": 0.1,
        },
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["requirements"]["grasp_face"] == "top"
    assert raw["selected_grasp_face"]["name"] == "top"
    assert raw["selected_candidate"]["grasp_face"] == "top"


def test_plan_manipulation_task_hold_returns_contract_without_candidate_preview_search():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_plan_manipulation_task"
    raw = result["raw"]
    assert raw["backend"] == "staged_moveit"
    assert raw["task_kind"] == "hold"
    assert raw["task_solution_id"] in tools._task_solutions
    assert raw["solver"] == "contract_moveit"
    assert raw["selected_candidate"]["attempt_index"] == 1
    assert raw["selected_candidate"]["status"] == "contract"
    assert raw["selected_candidate"]["grasp_face"] == "top"
    assert raw["candidate_attempts"] == []
    assert raw["preview"]["kind"] == "AgentPath"
    assert [stage["name"] for stage in raw["preview"]["motion_stages"]] == [
        "connect_to_pre_grasp",
        "approach_to_pre_grasp",
        "post_grasp_lift",
    ]
    assert [stage["waypoint_index"] for stage in raw["preview"]["motion_stages"]] == [0, 1, 2]
    assert raw["execution_contract"]["can_execute"] is True
    assert [step["handler"] for step in raw["execution_contract"]["steps"]] == [
        "motion",
        "motion",
        "close_gripper",
        "attach_object",
        "motion",
        "verify_attached_object",
    ]
    assert transport.published == []
    assert not any(topic == "/UR10/request/sampled" for topic, _ in transport.published)


def test_plan_manipulation_task_hold_accepts_zero_lift_distance():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001", "lift_distance_m": 0.0},
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["parameters"]["lift_distance_m"] == 0.0
    assert raw["selected_candidate"]["lift_distance_m"] == 0.0
    assert raw["waypoints"][2]["position"]["z"] == raw["waypoints"][1]["position"]["z"]
    assert raw["execution_contract"]["can_execute"] is True


def test_plan_manipulation_task_hold_returns_contract_without_preview_planning():
    dynamic_2_scene = {
        "scene": {
            "world": {
                "collision_objects": [
                    {
                        "id": "dynamic_2",
                        "header": {"frame_id": "base_link"},
                        "primitives": [{"type": 1, "dimensions": [0.3, 0.04, 0.04]}],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.15, "y": -0.7, "z": 0.12},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            }
                        ],
                    },
                    PLANNING_SCENE["scene"]["world"]["collision_objects"][1],
                ],
            },
            "robot_state": {"attached_collision_objects": []},
            "object_colors": [],
        }
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", dynamic_2_scene, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "dynamic_2", "lift_distance_m": 0.1},
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["task_kind"] == "hold"
    assert raw["object_name"] == "dynamic_2"
    assert raw["execution_contract"]["can_execute"] is True
    assert [step["handler"] for step in raw["execution_contract"]["steps"]] == [
        "motion",
        "motion",
        "close_gripper",
        "attach_object",
        "motion",
        "verify_attached_object",
    ]
    assert raw["selected_grasp_face"]["name"] == "top"
    assert raw["candidate_attempts"] == []
    assert transport.published == []


def test_plan_manipulation_task_hold_does_not_fail_without_required_stage_preview():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_plan_manipulation_task"
    raw = result["raw"]
    assert raw["task_solution_id"] in tools._task_solutions
    assert raw["candidate_attempts"] == []
    assert [stage["name"] for stage in raw["stages"]] == [
        "observe_current_state",
        "connect_to_pre_grasp",
        "approach_to_pre_grasp",
        "close_gripper",
        "attach_object",
        "post_grasp_lift",
        "verify_attached_object",
    ]
    assert transport.published == []


def test_plan_place_task_returns_execution_contract_for_verified_release():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_PLANNING_SCENE, planning_frame="base_link")
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    result = tools.plan_place_task(
        "UR10",
        "beam_001",
        target_position={"x": 0.55, "y": 0.2, "z": 0.12},
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    contract = raw["execution_contract"]
    assert contract["task_solution_id"] == raw["task_solution_id"]
    assert contract["object_name"] == "beam_001"
    assert contract["scene_snapshot_id"] == raw["scene_snapshot_id"]
    steps = contract["steps"]
    assert [step["handler"] for step in steps] == [
        "motion",
        "open_gripper",
        "release_object",
        "motion",
        "verify_released_object",
    ]
    assert all(step["source_stage"] and step["required_proof"] for step in steps)
    release_step = next(step for step in steps if step["handler"] == "release_object")
    assert release_step["object_name"] == "beam_001"
    assert release_step["scene_snapshot_id"] == raw["scene_snapshot_id"]
    assert release_step["arguments"]["object_pose"] == raw["release_after_execute"]["object_pose"]
    assert transport.published == []
    assert transport.action_goals == []


def test_plan_manipulation_task_release_uses_current_held_object_without_motion():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.set_state("UR10", "closed")
    tools.gripper.attach("UR10", "beam_001")

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "release"},
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["task_kind"] == "release"
    assert raw["object_name"] == "beam_001"
    assert raw["preview"]["ar_preview_mode"] == "none_no_motion"
    assert [step["handler"] for step in raw["execution_contract"]["steps"]] == [
        "open_gripper",
        "release_object",
        "verify_released_object",
    ]
    assert raw["execution_contract"]["steps"][1]["arguments"]["object_pose"]


def test_plan_manipulation_task_place_reuses_staged_place_contract():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_PLANNING_SCENE, planning_frame="base_link")
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    for plan_name in [
        "manipulation_place_beam_001_c01_connect_to_place",
        "manipulation_place_beam_001_c01_approach_place",
        "manipulation_place_beam_001_c01_retreat",
    ]:
        transport.queue_status_after_publish("/UR10/request/status", "success! ")
        transport.queue_planned_path_after_publish(
            "/UR10/request/planned_path",
            name=plan_name,
            points=5,
            final_positions=FINAL_POSITIONS,
        )
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={
            "goal": "place",
            "object_name": "beam_001",
            "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
        },
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["task_kind"] == "place"
    assert raw["created_from_tool"] == "moveit_plan_manipulation_task"
    assert raw["preview"]["name"] == "AgentPath"
    assert [stage["name"] for stage in raw["preview"]["motion_stages"]] == [
        "connect_to_place",
        "approach_place",
        "retreat",
    ]
    assert [stage["planner"] for stage in raw["preview"]["motion_stages"]] == [
        "free_motion",
        "cartesian",
        "cartesian",
    ]
    assert [step["handler"] for step in raw["execution_contract"]["steps"]] == [
        "motion",
        "motion",
        "open_gripper",
        "release_object",
        "motion",
        "verify_released_object",
    ]
    assert raw["execution_contract"]["steps"][0]["source_stage"] == "connect_to_place"
    assert raw["execution_contract"]["steps"][1]["source_stage"] == "approach_place"
    assert raw["execution_contract"]["steps"][4]["source_stage"] == "retreat"
    assert [topic for topic, _ in transport.published] == [
        "/UR10/request/free",
        "/UR10/request/cartesian",
        "/UR10/request/cartesian",
    ]


def test_plan_manipulation_task_pick_place_combines_hold_and_release_contract():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    for plan_name in [
        "manipulation_hold_beam_001_c01_connect_to_pre_grasp",
        "manipulation_hold_beam_001_c01_approach_to_pre_grasp",
        "manipulation_hold_beam_001_c01_post_grasp_lift",
        "manipulation_pick_place_beam_001_c01_connect_to_place",
        "manipulation_pick_place_beam_001_c01_approach_place",
        "manipulation_pick_place_beam_001_c01_retreat",
    ]:
        transport.queue_status_after_publish("/UR10/request/status", "success! ")
        transport.queue_planned_path_after_publish(
            "/UR10/request/planned_path",
            name=plan_name,
            points=5,
            final_positions=FINAL_POSITIONS,
        )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_manipulation_task(
        "UR10",
        requirements={
            "goal": "pick_place",
            "object_name": "beam_001",
            "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
        },
        backend="staged_moveit",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    raw = result["raw"]
    assert raw["task_kind"] == "pick_place"
    assert raw["selected_candidate"]["attempt_index"] == 1
    assert len(raw["waypoints"]) == 6
    assert [stage["name"] for stage in raw["preview"]["motion_stages"]] == [
        "connect_to_pre_grasp",
        "approach_to_pre_grasp",
        "post_grasp_lift",
        "connect_to_place",
        "approach_place",
        "retreat",
    ]
    assert [stage["planner"] for stage in raw["preview"]["motion_stages"]] == [
        "free_motion",
        "cartesian",
        "cartesian",
        "free_motion",
        "cartesian",
        "cartesian",
    ]
    assert [step["handler"] for step in raw["execution_contract"]["steps"]] == [
        "motion",
        "motion",
        "close_gripper",
        "attach_object",
        "motion",
        "motion",
        "motion",
        "open_gripper",
        "release_object",
        "motion",
        "verify_released_object",
    ]
    assert raw["execution_contract"]["steps"][5]["source_stage"] == "connect_to_place"
    assert raw["execution_contract"]["steps"][6]["source_stage"] == "approach_place"
    assert raw["execution_contract"]["steps"][9]["source_stage"] == "retreat"
    assert [topic for topic, _ in transport.published] == [
        "/UR10/request/free",
        "/UR10/request/cartesian",
        "/UR10/request/cartesian",
        "/UR10/request/free",
        "/UR10/request/cartesian",
        "/UR10/request/cartesian",
    ]


def test_plan_pick_task_mtc_enabled_unavailable_fails_without_emulated_fallback():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport, pick_task_backend="mtc")

    result = tools.plan_pick_task("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_pick_task"
    assert result["feedback"]["status"] == "mtc task backend unavailable"
    assert result["feedback"]["can_execute"] is False
    assert result["raw"]["backend"] == "mtc"
    assert result["raw"]["failed_stage"] == "mtc_service_unavailable"
    assert result["raw"]["blocker"]
    assert "task_solution_id" not in result["raw"]
    assert ("plan_mtc_pick_task", "UR10", "beam_001", None, 0.1) in transport.events
    assert tools._task_solutions == {}
    assert transport.published == []


def test_plan_pick_task_mtc_enabled_solved_payload_converts_to_task_solution_contract():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_mtc_pick_task_result(
        {
            "ok": True,
            "task_solution_id": "mtc_pick_dynamic_1_001",
            "selected_cost": 3.42,
            "selected_grasp": {
                "grasp_face": "front",
                "approach_distance_m": 0.08,
                "score": 0.91,
            },
            "candidate_attempts": [
                {
                    "attempt_index": 1,
                    "grasp_face": "front",
                    "approach_distance_m": 0.08,
                    "status": "solved",
                    "cost": 3.42,
                    "selected": True,
                },
                {
                    "attempt_index": 2,
                    "grasp_face": "left",
                    "approach_distance_m": 0.12,
                    "status": "failed",
                    "failed_stage": "ComputeIK",
                    "selected": False,
                },
            ],
            "stage_summaries": [
                {"name": "current state", "stage_type": "CurrentState", "status": "solved", "cost": 0.0},
                {"name": "connect", "stage_type": "Connect", "status": "solved", "cost": 0.8},
                {
                    "name": "generate grasp pose",
                    "stage_type": "GenerateGraspPose",
                    "status": "solved",
                    "cost": 0.2,
                    "raw_authoring": {"cpp": "do not expose"},
                },
                {"name": "grasp pose ik", "stage_type": "ComputeIK", "status": "solved", "cost": 0.3},
                {"name": "approach", "stage_type": "MoveRelative", "status": "solved", "cost": 0.9},
                {"name": "attach object", "stage_type": "ModifyPlanningScene", "status": "solved", "cost": 0.0},
                {"name": "lift", "stage_type": "MoveRelative", "status": "solved", "cost": 1.22},
            ],
            "solution_evidence": [{"kind": "mtc_solution", "solution_index": 0}],
            "scene_snapshot": {"id": "mtc_scene_001", "object_count": 2},
        }
    )
    tools = MoveItMcpTools.with_fake_transport(transport, pick_task_backend="mtc")

    result = tools.plan_pick_task("UR10", "beam_001", grasp_face="front", timeout_s=0.1)

    assert result["ok"] is True
    assert result["feedback"]["execution_target"] == "task_solution"
    raw = result["raw"]
    assert raw["backend"] == "mtc"
    assert raw["task_solution_id"] == "mtc_pick_dynamic_1_001"
    assert raw["execution_target"] == "task_solution"
    assert raw["approval"]["task_solution_id"] == "mtc_pick_dynamic_1_001"
    assert raw["selected_cost"] == 3.42
    assert raw["selected_grasp"]["grasp_face"] == "front"
    assert raw["candidate_attempts"][1]["failed_stage"] == "ComputeIK"
    assert raw["scene_snapshot_id"] == "mtc_scene_001"
    assert raw["object"]["name"] == "beam_001"
    assert raw["stage_report"] == {"total": 7, "solved": 7, "failed": 0}
    assert [stage["stage_type"] for stage in raw["stages"]] == [
        "CurrentState",
        "Connect",
        "GenerateGraspPose",
        "ComputeIK",
        "MoveRelative",
        "ModifyPlanningScene",
        "MoveRelative",
    ]
    assert all("raw_authoring" not in stage.get("raw", {}) for stage in raw["stages"])
    assert raw["evidence"][-1] == {"kind": "mtc_solution", "solution_index": 0}
    assert raw["task_solution_id"] in tools._task_solutions
    assert ("plan_mtc_pick_task", "UR10", "beam_001", "front", 0.1) in transport.events
    assert transport.published == []


def test_plan_compound_task_requires_mtc_backend_without_storing_solution():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_compound_task"
    assert result["feedback"]["status"] == "mtc backend required"
    assert result["retryable"] is True
    assert result["raw"]["backend"] is None
    assert result["raw"]["requirements"] == {"goal": "hold", "object_name": "beam_001"}
    assert result["raw"]["object_name"] == "beam_001"
    assert result["raw"]["task_goal"] == "hold"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_missing_requirements_instead_of_upgrading_legacy_stage_intents():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        "beam_001",
        task_goal="hold",
        stage_intents=["observe_current_state", "approach_object", "close_gripper", "verify_attached"],
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_compound_task"
    assert result["feedback"]["status"] == "invalid compound requirements"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


@pytest.mark.parametrize("task_goal", ["move_and_release", "pick_place"])
def test_plan_compound_task_requires_target_for_transfer_goals(task_goal):
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": task_goal, "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["status"] == "invalid compound requirements"
    assert result["raw"]["missing"] == "target_pose or target_position"
    assert "task_solution_id" not in result["raw"]


def test_plan_compound_task_allows_plain_release_but_fails_closed_without_mtc_solution():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "release", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["status"] == "mtc compound task backend unavailable"
    assert result["raw"]["failed_stage"] == "mtc_service_unavailable"
    assert result["raw"]["task_goal"] == "release"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_retired_approach_hold_adjust_release_goal():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        requirements={
            "goal": "approach_hold_adjust_release",
            "object_name": "beam_001",
            "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
        },
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["status"] == "unsupported task goal"
    assert "release" in result["feedback"]["correction"]
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


@pytest.mark.parametrize(
    "stage_intents",
    [
        ["observe_current_state", "slide", "verify_released"],
        ["observe_current_state", "push", "verify_released"],
        ["observe_current_state", "run_python_script", "verify_released"],
        ["observe_current_state", "approach_object", "unknown_stage"],
    ],
)
def test_plan_compound_task_rejects_unsafe_or_unknown_stage_intents(stage_intents):
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.plan_compound_task(
        "UR10",
        requirements={
            "goal": "move_and_release",
            "object_name": "beam_001",
            "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
        },
        stage_intents=stage_intents,
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_compound_task"
    assert result["feedback"]["status"] == "unsupported stage intent"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_mtc_unavailable_fails_without_storing_solution():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": False,
            "backend": "mtc",
            "failed_stage": "mtc_service_unavailable",
            "blocker": "MTC compound task service did not respond.",
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={
            "goal": "pick_place",
            "object_name": "beam_001",
            "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
        },
        preferences={"grasp_face": "top", "orientation_mode": "keep"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_compound_task"
    assert result["feedback"]["status"] == "mtc compound task backend unavailable"
    assert result["raw"]["failed_stage"] == "mtc_service_unavailable"
    assert result["raw"]["blocker"] == "MTC compound task service did not respond."
    assert result["raw"]["requirements"] == {
        "goal": "pick_place",
        "object_name": "beam_001",
        "target_position": {"x": 0.55, "y": 0.2, "z": 0.12},
    }
    assert result["raw"]["preferences"] == {"grasp_face": "top", "orientation_mode": "keep"}
    assert result["raw"]["stage_intents"] is None
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_preserves_backend_failure_metadata_without_solution_id():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": False,
            "backend": "mtc",
            "failed_stage": "preview_solution",
            "error": "mtc_solution_preview_unavailable",
            "message": "Solved MTC task could not be previewed.",
            "blocker": "The /solution publisher did not produce preview evidence.",
            "correction": "Fix the /solution preview publisher before retrying.",
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "not_published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "unavailable",
            },
            "execution_contract": {"can_execute": False},
            "task_solution_id": "backend_must_not_leak",
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["correction"] == "Fix the /solution preview publisher before retrying."
    assert result["raw"]["error"] == "mtc_solution_preview_unavailable"
    assert result["raw"]["message"] == "Solved MTC task could not be previewed."
    assert result["raw"]["correction"] == "Fix the /solution preview publisher before retrying."
    assert result["raw"]["preview"]["solution_preview"] == "not_published"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_solved_payload_without_preview_evidence():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": True,
            "task_solution_id": "mtc_compound_no_preview",
            "task_goal": "hold",
            "backend": "mtc",
            "stage_summaries": [{"name": "hold", "stage_type": "MoveRelative", "status": "solved"}],
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "motion",
                        "source_stage": "hold",
                        "required_proof": "mtc_stage_solution",
                        "plan_handle": "mtc_compound_no_preview/hold",
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["failed_stage"] == "mtc_compound_solution_incomplete"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_solved_payload_when_contract_cannot_execute():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": True,
            "task_solution_id": "mtc_compound_not_executable",
            "task_goal": "hold",
            "backend": "mtc",
            "stage_summaries": [{"name": "hold", "stage_type": "MoveRelative", "status": "solved"}],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "previewed",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": False,
                "steps": [
                    {
                        "handler": "motion",
                        "source_stage": "hold",
                        "required_proof": "mtc_stage_solution",
                        "plan_handle": "mtc_compound_not_executable/hold",
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["failed_stage"] == "mtc_compound_solution_incomplete"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_solved_backend_payload_missing_can_execute():
    transport = FakeRosbridgeTransport()
    tools = MoveItMcpTools.with_fake_transport(transport)
    transport.queue_mtc_compound_task_result(
        {
            "ok": True,
            "task_solution_id": "mtc_compound_missing_can_execute",
            "task_goal": "hold",
            "backend": "mtc",
            "stage_summaries": [{"name": "hold", "stage_type": "MoveRelative", "status": "solved"}],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "previewed",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "steps": [
                    {
                        "handler": "motion",
                        "source_stage": "hold",
                        "required_proof": "mtc_stage_solution",
                        "plan_handle": "mtc_compound_missing_can_execute/hold",
                    }
                ],
            },
        }
    )

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["failed_stage"] == "mtc_compound_solution_incomplete"
    assert result["raw"]["execution_contract"]["can_execute"] is False
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_rejects_unsupported_execution_contract_handler():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": True,
            "task_solution_id": "mtc_compound_unsafe_handler",
            "task_goal": "hold",
            "backend": "mtc",
            "stage_summaries": [{"name": "script", "stage_type": "Opaque", "status": "solved"}],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "previewed",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "run_python_script",
                        "source_stage": "script",
                        "required_proof": "script_returned",
                        "arguments": {"script": "move_robot()"},
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["failed_stage"] == "mtc_compound_solution_incomplete"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_plain_release_none_no_motion_rejects_metadata_only_release_proof():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": True,
            "task_solution_id": "mtc_release_without_proof",
            "task_goal": "release",
            "backend": "mtc",
            "stage_summaries": [
                {"name": "open gripper", "stage_type": "MoveGripper", "status": "solved"},
                {"name": "release_object", "stage_type": "ModifyPlanningScene", "status": "solved"},
            ],
            "task_stages": [{"handler": "verify_released_object", "status": "solved"}],
            "solution_evidence": [{"kind": "release_object", "source": "metadata_only"}],
            "selected_stage_evidence": [{"kind": "verify_released_object", "source": "metadata_only"}],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "not_published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "none_no_motion",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "open_gripper",
                        "source_stage": "open gripper",
                        "required_proof": "verified_gripper_open",
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "release", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["failed_stage"] == "mtc_compound_solution_incomplete"
    assert "task_solution_id" not in result["raw"]
    assert tools._task_solutions == {}


def test_plan_compound_task_mtc_solved_payload_stores_execution_contract():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    def plan_mtc_compound_task(**kwargs):
        assert kwargs["robot"] == "UR10"
        assert "object_name" not in kwargs
        assert "task_goal" not in kwargs
        assert "target_pose" not in kwargs
        assert "target_position" not in kwargs
        assert kwargs["requirements"] == {
            "goal": "pick_place",
            "object_name": "beam_001",
            "target_pose": {
                "position": {"x": 0.55, "y": 0.2, "z": 0.18},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }
        assert kwargs["preferences"] == {"grasp_face": "top", "retreat_distance_m": 0.08}
        assert kwargs["stage_intents"] == [
            "observe_current_state",
            "approach_object",
            "close_gripper",
            "verify_attached",
            "lift",
            "adjust_pose",
            "open_gripper",
            "release_object",
            "verify_released",
        ]
        return {
            "ok": True,
            "task_solution_id": "mtc_compound_001",
            "task_goal": "pick_place",
            "backend": "mtc",
            "selected_cost": 4.2,
            "candidate_count": 3,
            "attempts": [{"attempt_index": 1, "status": "solved", "selected": True}],
            "scene_snapshot": {"id": "compound_scene_001", "object_count": 2},
            "stage_summaries": [
                {"name": "current state", "stage_type": "CurrentState", "status": "solved", "cost": 0.0},
                {"name": "approach", "stage_type": "MoveRelative", "status": "solved", "cost": 1.0},
                {"name": "release", "stage_type": "ModifyPlanningScene", "status": "solved", "cost": 0.0},
            ],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "previewed",
            },
            "solution_evidence": [{"kind": "mtc_solution", "solution_index": 0}],
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "step": 1,
                        "handler": "observe_current_state",
                        "source_stage": "current state",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "compound_scene_001",
                        "required_proof": "current_state_observed",
                    },
                    {
                        "step": 2,
                        "handler": "execute_plan",
                        "source_stage": "approach",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "compound_scene_001",
                        "plan_handle": "mtc_compound_001/approach",
                        "required_proof": "plan_execution_verified",
                    },
                    {
                        "step": 3,
                        "handler": "verify_released",
                        "source_stage": "release",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "compound_scene_001",
                        "required_proof": "object_released",
                    },
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={
            "goal": "pick_place",
            "object_name": "beam_001",
            "target_pose": {
                "position": {"x": 0.55, "y": 0.2, "z": 0.18},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
        preferences={"grasp_face": "top", "retreat_distance_m": 0.08},
        stage_intents=[
            "observe_current_state",
            "approach_object",
            "close_gripper",
            "verify_attached",
            "lift",
            "adjust_pose",
            "open_gripper",
            "release_object",
            "verify_released",
        ],
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["feedback"]["execution_target"] == "task_solution"
    raw = result["raw"]
    assert raw["backend"] == "mtc"
    assert raw["task_kind"] == "pick_place"
    assert raw["created_from_tool"] == "moveit_plan_compound_task"
    assert raw["task_solution_id"] == "mtc_compound_001"
    assert raw["scene_snapshot_id"] == "compound_scene_001"
    assert raw["selected_cost"] == 4.2
    assert raw["candidate_count"] == 3
    assert raw["attempts"][0]["status"] == "solved"
    assert raw["requirements"]["goal"] == "pick_place"
    assert raw["requirements"]["object_name"] == "beam_001"
    assert raw["preferences"] == {"grasp_face": "top", "retreat_distance_m": 0.08}
    assert raw["stage_intents"][-1] == "verify_released"
    assert raw["execution_contract"][1]["plan_handle"] == "mtc_compound_001/approach"
    assert raw["execution_contract"][1]["required_proof"] == "plan_execution_verified"
    assert raw["evidence"][-1] == {"kind": "mtc_solution", "solution_index": 0}
    assert raw["task_solution_id"] in tools._task_solutions


def test_plan_compound_move_and_release_uses_current_vizor_client_contract():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())
    seen_kwargs = {}

    def plan_mtc_compound_task(**kwargs):
        seen_kwargs.update(kwargs)
        return {
            "ok": True,
            "task_solution_id": "mtc_move_release_001",
            "task_goal": "move_and_release",
            "backend": "mtc",
            "object_name": "dynamic_5",
            "stage_summaries": [{"name": "release", "stage_type": "MoveRelative", "status": "solved"}],
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "previewed",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "motion",
                        "name": "release_pose",
                        "waypoint_index": 0,
                        "tool": "moveit_execute_plan",
                        "source_stage": "release",
                        "required_proof": "mtc_stage_solution",
                        "arguments": {"plan_name": "mtc_move_release_001/release"},
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task

    result = tools.plan_compound_task(
        "UR10",
        requirements={
            "goal": "move_and_release",
            "object_name": "dynamic_5",
            "target_position": {"x": 0.2604, "y": -0.8001, "z": 0.4807},
        },
        backend="mtc",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    step = result["raw"]["execution_contract"][0]
    assert step["name"] == "release_pose"
    assert step["tool"] == "moveit_execute_plan"
    assert step["waypoint_index"] == 0
    assert step["arguments"] == {"plan_name": "mtc_move_release_001/release"}
    assert seen_kwargs == {
        "robot": "UR10",
        "requirements": {
            "goal": "move_and_release",
            "object_name": "dynamic_5",
            "target_position": {"x": 0.2604, "y": -0.8001, "z": 0.4807},
        },
        "preferences": None,
        "stage_intents": None,
        "backend": "mtc",
        "timeout_s": 0.1,
    }


def test_execute_compound_task_solution_runs_stored_execution_contract_steps():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene("UR10", ATTACHED_PLANNING_SCENE, planning_frame="base_link")
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.085, "requested_position": 0.085})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.0])
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    def plan_mtc_compound_task(**_kwargs):
        return {
            "ok": True,
            "task_solution_id": "mtc_release_contract_001",
            "task_goal": "release",
            "backend": "mtc",
            "object_name": "beam_001",
            "stage_summaries": [{"name": "legacy mtc release stage", "stage_type": "ModifyPlanningScene", "status": "solved"}],
            "scene_snapshot": {"id": "release_scene_001"},
            "preview": {
                "solution_topic": "/solution",
                "solution_preview": "not_published",
                "ar_preview_service": "/vizor_robot_control",
                "ar_preview_mode": "none_no_motion",
            },
            "execution_contract": {
                "target_kind": "task_solution",
                "requires_explicit_approval": True,
                "can_execute": True,
                "steps": [
                    {
                        "handler": "open_gripper",
                        "source_stage": "open gripper",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "release_scene_001",
                        "required_proof": "verified_gripper_open",
                    },
                    {
                        "handler": "release_object",
                        "source_stage": "release",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "release_scene_001",
                        "required_proof": "planning_scene_update",
                        "arguments": {
                            "object_name": "beam_001",
                            "object_pose": {
                                "position": {"x": 0.55, "y": 0.2, "z": 0.12},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            },
                        },
                    },
                    {
                        "handler": "verify_released_object",
                        "source_stage": "verify release",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "release_scene_001",
                        "required_proof": "release_check",
                    },
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task
    planned = tools.plan_compound_task(
        "UR10",
        requirements={"goal": "release", "object_name": "beam_001"},
        backend="mtc",
        timeout_s=0.1,
    )
    assert planned["ok"] is True

    result = tools.execute_task_solution("UR10", planned["raw"]["task_solution_id"], timeout_s=0.1)

    assert result["ok"] is True
    assert [stage["name"] for stage in result["raw"]["stages"]] == [
        "open_gripper",
        "release_object",
        "verify_released_object",
    ]
    assert all(stage["status"] == "executed" for stage in result["raw"]["stages"])
    assert result["raw"]["stage_report"]["executed"] == 3
    assert tools.gripper.attached_object("UR10") is None
    assert len(transport.applied_planning_scenes) == 1


def test_plan_pick_task_emulated_dynamic_vertical_object_reports_multiple_candidate_attempts():
    dynamic_scene = {
        "scene": {
            "world": {
                "collision_objects": [
                    {
                        "id": "dynamic_1",
                        "operation": 0,
                        "primitives": [{"type": 1, "dimensions": [0.04, 0.04, 0.30]}],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.2, "z": 0.16},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        ],
                    },
                ],
            },
        },
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", dynamic_scene, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick_task("UR10", "dynamic_1", timeout_s=0.1)

    assert result["ok"] is True
    attempts = result["raw"]["candidate_attempts"]
    assert len(attempts) > 1
    assert {attempt["grasp_face"] for attempt in attempts} <= {"front", "back", "left", "right"}
    assert "top" not in {attempt["grasp_face"] for attempt in attempts}
    assert all("approach_distance_m" in attempt for attempt in attempts)
    assert attempts[0]["selected"] is True
    assert result["raw"]["selected_grasp_face"]["name"] in {"front", "back", "left", "right"}


def test_plan_pick_aligns_top_grasp_with_rotated_beam_axis():
    scene = {
        **PLANNING_SCENE,
        "scene": {
            **PLANNING_SCENE["scene"],
            "world": {
                "collision_objects": [
                    {
                        **PLANNING_SCENE["scene"]["world"]["collision_objects"][0],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.2, "z": 0.12},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.707106781187, "w": 0.707106781187},
                            }
                        ],
                    },
                    PLANNING_SCENE["scene"]["world"]["collision_objects"][1],
                ]
            },
        },
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", scene, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_rotated",
        points=6,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_rotated",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    top_face = result["raw"]["selected_grasp_face"]
    assert top_face["name"] == "top"
    assert top_face["alignment_axis"] == pytest.approx({"x": 0.0, "y": 1.0, "z": 0.0})
    orientation = result["raw"]["waypoints"][0]["orientation"]
    assert orientation == pytest.approx({"x": 0.707106781187, "y": 0.707106781187, "z": 0.0, "w": 0.0})
    assert all(point["orientation"] == orientation for point in result["raw"]["waypoints"])


def test_plan_pick_cartesian_uses_explicit_side_for_horizontal_beam_when_requested():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_front",
        points=6,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_front",
        grasp_face="front",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["raw"]["selected_grasp_face"]["name"] == "front"
    assert [point["position"] for point in result["raw"]["waypoints"]] == [
        {"x": 0.4, "y": 0.3, "z": 0.12},
        {"x": 0.4, "y": 0.23, "z": 0.12},
        {"x": 0.4, "y": 0.23, "z": 0.22},
    ]
    orientation = result["raw"]["waypoints"][0]["orientation"]
    assert orientation == pytest.approx({"x": 0.707106781187, "y": 0.0, "z": 0.0, "w": 0.707106781187})
    assert all(point["orientation"] == orientation for point in result["raw"]["waypoints"])


def test_plan_pick_cartesian_uses_side_face_for_vertical_beam():
    vertical_scene = {
        "scene": {
            "world": {
                "collision_objects": [
                    {
                        "id": "vertical_beam",
                        "operation": 0,
                        "primitives": [
                            {"type": 1, "dimensions": [0.04, 0.04, 0.30]},
                        ],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.2, "z": 0.16},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        ],
                    },
                ],
            },
        },
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", vertical_scene, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_vertical",
        points=6,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "vertical_beam",
        plan_name="pick_vertical",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["raw"]["selected_grasp_face"]["name"] in {"front", "back", "left", "right"}
    assert result["raw"]["selected_grasp_face"]["name"] != "top"
    assert [topic for topic, _ in transport.published].count("/UR10/request/cartesian") == 1


def test_plan_pick_missing_object_suggests_scene_listing_without_publishing():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick("UR10", "missing", plan_name="pick_missing", timeout_s=0.1)

    assert result["ok"] is False
    assert result["tool"] == "moveit_plan_pick"
    assert result["feedback"]["status"] == "object not found"
    assert "moveit_list_scene_objects" in result["feedback"]["correction"]
    assert result["raw"]["available_objects"] == ["beam_001", "ground_plane"]
    assert result["raw"]["candidate_attempts"] == []
    assert transport.published == []


def test_plan_pick_invalid_grasp_face_reports_available_faces():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick("UR10", "beam_001", plan_name="pick_bad_face", grasp_face="diagonal")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "grasp face not available"
    assert "raw.object.grasp_faces" in result["feedback"]["correction"]
    assert result["raw"]["available_grasp_faces"] == ["right", "left", "front", "back", "top", "bottom"]
    assert result["raw"]["candidate_attempts"] == []
    assert transport.published == []


def test_plan_pick_cartesian_strategy_remains_one_shot():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "incomplete path")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_cartesian",
        points=4,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_cartesian",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["raw"]["planning_strategy"] == "cartesian"
    assert result["raw"]["planning_strategy_resolved"] == "cartesian"
    assert len(result["raw"]["candidate_attempts"]) == 1
    assert [topic for topic, _ in transport.published].count("/UR10/request/cartesian") == 1


def test_plan_pick_auto_reports_partial_when_preposition_succeeds_but_local_pick_fails():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_auto__preposition",
        points=5,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_auto",
        planning_strategy="auto",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["error"] == "pick_segment_planning_failed"
    assert result["failed_segment"] == "local_cartesian_pick"
    assert result["retryable"] is True
    assert result["feedback"]["can_execute"] is False
    assert result["suggested_next_tool"] != "moveit_execute_plan"
    assert "plan_name" not in result["raw"]
    assert result["raw"]["partial_plan"]["kind"] == "preposition"
    assert result["raw"]["partial_plan"]["plan_name"] == "pick_auto__preposition"
    assert result["raw"]["stage_report"][-1]["name"] == "local_cartesian_pick"
    assert result["raw"]["stage_report"][-1]["status"] == "failed"
    assert isinstance(result["raw"]["candidate_attempts"], int)
    assert transport.published[-1][0] == "/UR10/request/free"
    assert transport.published[-1][1]["name"] == "pick_auto__preposition"
    assert transport.published[-1][1]["target_pose"] == {
        "position": {"x": 0.4, "y": 0.2, "z": 0.22},
        "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
    }
    assert not any(topic == "/UR10/request/cartesian" for topic, _ in transport.published)


def test_plan_pick_auto_rejects_non_executable_preposition_without_partial_diagnostic():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_auto_missing_final__preposition",
        points=5,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_auto_missing_final",
        planning_strategy="auto",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["can_execute"] is False
    assert result["feedback"]["message"] == "Pick preposition plan did not satisfy execution requirements"
    assert "partial_plan" not in result["raw"]
    assert "failed_segment" not in result
    assert not any(
        stage.get("name") == "local_cartesian_pick" and stage.get("status") == "failed"
        for stage in result["raw"].get("stage_report", [])
    )
    assert ("UR10", "pick_auto_missing_final__preposition") not in tools._planned


def test_plan_pick_auto_uses_side_preposition_for_vertical_beam():
    vertical_scene = {
        "scene": {
            "world": {
                "collision_objects": [
                    {
                        "id": "vertical_beam",
                        "operation": 0,
                        "primitives": [
                            {"type": 1, "dimensions": [0.04, 0.04, 0.30]},
                        ],
                        "primitive_poses": [
                            {
                                "position": {"x": 0.4, "y": 0.2, "z": 0.16},
                                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
                            },
                        ],
                    },
                ],
            },
        },
    }
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", vertical_scene, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_vertical_auto__preposition",
        points=5,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "vertical_beam",
        plan_name="pick_vertical_auto",
        planning_strategy="auto",
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["failed_segment"] == "local_cartesian_pick"
    assert result["raw"]["partial_plan"]["plan_name"] == "pick_vertical_auto__preposition"
    assert result["raw"]["stage_report"][-1]["status"] == "failed"
    assert not any(topic == "/UR10/request/cartesian" for topic, _ in transport.published)


def test_plan_pick_sampled_approach_routes_to_sampled_rrtconnect_backend():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_sampled",
        points=6,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_sampled",
        planning_strategy="sampled_approach",
    )

    assert result["ok"] is True
    assert result["raw"]["planning_strategy"] == "sampled_approach"
    assert result["raw"]["planning_strategy_resolved"] == "sampled_approach"
    assert result["raw"]["candidate_attempts"] == [
        {
            "attempt_index": 1,
            "plan_name": "pick_sampled",
            "grasp_face": "top",
            "approach_distance_m": 0.08,
            "grasp_standoff_m": 0.01,
            "lift_distance_m": 0.1,
            "planner": "sampled_approach",
            "planning_pipeline": "ompl",
            "planner_id": "RRTConnect",
            "status": "success! ",
            "trajectory_points": 6,
            "can_execute": True,
            "selected": True,
        }
    ]
    assert transport.published[-1][0] == "/UR10/request/sampled"
    assert transport.published[-1][1]["name"] == "pick_sampled"
    assert transport.published[-1][1]["poses"] == result["raw"]["waypoints"]
    assert not any(topic == "/UR10/request/cartesian" for topic, _ in transport.published)
    assert not any(topic == "/UR10/request/free" for topic, _ in transport.published)


def test_plan_place_derives_release_waypoints_for_attached_object():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_PLANNING_SCENE, planning_frame="base_link")
    transport.set_current_pose("UR10", CURRENT_POSE, planning_frame="base_link")
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="place_beam",
        points=5,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")

    result = tools.plan_place(
        "UR10",
        "beam_001",
        plan_name="place_beam",
        target_position={"x": 0.55, "y": 0.2, "z": 0.12},
        orientation_mode="keep",
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_plan_place"
    assert result["feedback"]["can_execute"] is True
    assert result["raw"]["workflow_kind"] == "place"
    assert result["raw"]["object_name"] == "beam_001"
    assert result["raw"]["plan_name"] == "place_beam"
    assert result["raw"]["release_tcp_pose"]["position"] == {"x": 0.55, "y": 0.2, "z": 0.13}
    assert [step["name"] for step in result["raw"]["workflow_steps"]] == [
        "carry_approach",
        "release_pose",
        "open_gripper",
        "detach_object",
        "retreat",
    ]
    assert transport.published[-1][0] == "/UR10/request/cartesian"
    assert transport.published[-1][1]["name"] == "place_beam"
    assert transport.published[-1][1]["poses"] == result["raw"]["waypoints"]
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_plan_free_motion_returns_pass_feedback_when_status_and_path_observed():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_a",
        points=5,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_free_motion("UR10", "pick_a", {"x": 0.5, "y": 0.0, "z": 0.3})

    assert result["ok"] is True
    assert result["feedback"]["can_execute"] is True
    assert result["raw"]["plan_name"] == "pick_a"
    assert result["raw"]["planning_diagnostics"] == {
        "log_dir": "server/logs/moveit_planning",
        "join_key": "pick_a",
    }
    assert result["verification"]["result"] == "pass"
    assert {c["name"] for c in result["verification"]["checks"]} == {"status_success", "trajectory_observed"}
    assert any("pick_a" in item["summary"] for item in result["evidence"])


def test_plan_free_motion_fails_when_status_success_but_no_path_observed():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_free_motion("UR10", "missing_path", {"x": 0.5, "y": 0.0, "z": 0.3}, timeout_s=0.1)

    assert result["ok"] is False
    assert result["feedback"]["can_execute"] is False
    assert "smaller or safer target" in result["feedback"]["correction"]
    assert result["raw"]["plan_name"] == "missing_path"
    assert any(c["name"] == "trajectory_observed" and not c["passed"] for c in result["verification"]["checks"])


def test_plan_cartesian_motion_rejects_incomplete_path_by_default():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "incomplete path")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="lin_a",
        points=4,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_cartesian_motion(
        "UR10",
        "lin_a",
        [{"x": 0.4, "y": 0.0, "z": 0.3}, {"x": 0.5, "y": 0.0, "z": 0.3}],
    )

    assert result["ok"] is False
    assert result["feedback"]["status"] == "incomplete path"
    assert result["feedback"]["can_execute"] is False
    assert "smaller or safer target" in result["feedback"]["correction"]


def test_plan_name_is_generated_when_omitted(monkeypatch):
    monkeypatch.setattr("moveit_mcp.tools._generate_plan_name", lambda tool: f"{tool}_generated")
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="plan_free_motion_generated",
        points=2,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.plan_free_motion("UR10", {"x": 0.5, "y": 0.0, "z": 0.3})

    assert result["ok"] is True
    assert result["raw"]["plan_name"] == "plan_free_motion_generated"
    assert transport.published[-1][1]["name"] == "plan_free_motion_generated"
    assert any("plan_free_motion_generated" in item["summary"] for item in result["evidence"])


def test_rejects_reused_caller_provided_plan_name_unless_allowed():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="reuse_me",
        points=2,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    first = tools.plan_free_motion("UR10", "reuse_me", {"x": 0.5, "y": 0.0, "z": 0.3})
    rejected = tools.plan_free_motion("UR10", "reuse_me", {"x": 0.6, "y": 0.0, "z": 0.3})

    assert first["ok"] is True
    assert rejected["ok"] is False
    assert rejected["feedback"]["status"] == "plan name already used"
    assert "Omit plan_name" in rejected["feedback"]["correction"]
    assert len(transport.published) == 1

    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="reuse_me",
        points=3,
        final_positions=FINAL_POSITIONS,
    )

    allowed = tools.plan_free_motion(
        "UR10",
        "reuse_me",
        {"x": 0.7, "y": 0.0, "z": 0.3},
        allow_existing_name=True,
    )

    assert allowed["ok"] is True
    assert len(transport.published) == 2


def test_accepts_full_pose_input_and_validates_quaternion():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pose_plan",
        points=2,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)
    pose = {
        "position": {"x": 0.5, "y": 0.0, "z": 0.3},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.7071068, "w": 0.7071068},
    }

    result = tools.plan_free_motion("UR10", "pose_plan", pose)

    assert result["ok"] is True
    assert transport.published[-1][1]["target_pose"] == pose

    invalid = tools.plan_free_motion(
        "UR10",
        "invalid_pose",
        {
            "position": {"x": 0.5, "y": 0.0, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 2.0},
        },
    )

    assert invalid["ok"] is False
    assert invalid["feedback"]["status"] == "invalid pose"
    assert "finite x, y, z" in invalid["feedback"]["correction"]
    assert "normalized quaternion" in invalid["feedback"]["correction"]

    non_finite = tools.plan_free_motion("UR10", "non_finite_pose", {"x": float("nan"), "y": 0.0, "z": 0.3})

    assert non_finite["ok"] is False
    assert non_finite["feedback"]["status"] == "invalid pose"
    assert "finite x, y, z" in non_finite["feedback"]["correction"]
    assert len(transport.published) == 1
