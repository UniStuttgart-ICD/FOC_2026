import inspect

from moveit_mcp.tools import MoveItMcpTools
from moveit_mcp.vizor_client import FakeRosbridgeTransport

FINAL_POSITIONS = [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]
JOINT_TOPIC = "/UR10/move_group/fake_controller_joint_states"
BEAM_OBJECT = {
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
}
ATTACHED_SCENE = {
    "scene": {
        "world": {"collision_objects": []},
        "robot_state": {
            "attached_collision_objects": [
                {
                    "link_name": "tool0",
                    "object": BEAM_OBJECT,
                    "touch_links": ["tool0", "wrist_3_link"],
                }
            ]
        },
        "object_colors": [],
    }
}


def _plan_success(tools: MoveItMcpTools, transport: FakeRosbridgeTransport, name: str = "plan_a") -> dict:
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name=name,
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    return tools.plan_free_motion("UR10", name, {"x": 0.5, "y": 0.0, "z": 0.3})


def _queue_pick_task_motion_execution(transport: FakeRosbridgeTransport, task_solution_id: str) -> None:
    for stage in ("connect_to_pre_grasp", "approach_grasp", "lift_object"):
        plan_name = f"{task_solution_id}__{stage}"
        transport.queue_status_after_publish("/UR10/request/status", "success! ")
        transport.queue_planned_path_after_publish(
            "/UR10/request/planned_path",
            name=plan_name,
            points=3,
            final_positions=FINAL_POSITIONS,
        )
        transport.queue_joint_state_after_publish(JOINT_TOPIC, FINAL_POSITIONS)


def _queue_staged_hold_preview(transport: FakeRosbridgeTransport) -> None:
    for plan_name in [
        "manipulation_hold_beam_001_c01_connect_to_pre_grasp",
        "manipulation_hold_beam_001_c01_approach_to_pre_grasp",
        "manipulation_hold_beam_001_c01_post_grasp_lift",
    ]:
        transport.queue_status_after_publish("/UR10/request/status", "success! ")
        transport.queue_planned_path_after_publish(
            "/UR10/request/planned_path",
            name=plan_name,
            points=3,
            final_positions=FINAL_POSITIONS,
        )


def _acm_allows(acm: dict, first: str, second: str) -> bool:
    names = acm.get("entry_names", [])
    if first not in names or second not in names:
        return False
    first_index = names.index(first)
    second_index = names.index(second)
    values = acm.get("entry_values", [])
    if first_index >= len(values):
        return False
    enabled = values[first_index].get("enabled", [])
    return second_index < len(enabled) and bool(enabled[second_index])


def test_execute_rejects_unplanned_plan_without_publishing():
    transport = FakeRosbridgeTransport()
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.execute_plan("UR10", "never_planned")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "plan not verified"
    assert "Call a planning tool first" in result["feedback"]["correction"]
    assert "raw.plan_name" in result["feedback"]["correction"]
    assert any(c["name"] == "plan_previously_verified" and not c["passed"] for c in result["verification"]["checks"])
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_task_solution_rejects_unknown_id_without_publishing():
    transport = FakeRosbridgeTransport()
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.execute_task_solution("UR10", "missing_solution", timeout_s=0.1)

    assert result["ok"] is False
    assert result["error"] == "unknown_task_solution_id"
    assert result["retryable"] is False
    assert result["raw"]["task_solution_id"] == "missing_solution"
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_task_solution_default_timeout_is_sixty_seconds():
    timeout_param = inspect.signature(MoveItMcpTools.execute_task_solution).parameters["timeout_s"]

    assert timeout_param.default == 60.0


def test_execute_rejects_planned_plan_when_can_execute_was_false():
    transport = FakeRosbridgeTransport()
    transport.queue_status_after_publish("/UR10/request/status", "incomplete path")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="partial_plan",
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = tools.plan_cartesian_motion(
        "UR10",
        "partial_plan",
        [{"x": 0.4, "y": 0.0, "z": 0.3}, {"x": 0.5, "y": 0.0, "z": 0.3}],
    )
    assert plan["ok"] is False
    assert plan["feedback"]["can_execute"] is False

    result = tools.execute_plan("UR10", "partial_plan")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "plan not executable"
    assert "smaller or safer target" in result["feedback"]["correction"]
    assert any(c["name"] == "planned_can_execute" and not c["passed"] for c in result["verification"]["checks"])
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_blocks_unknown_physical_mode_before_publishing():
    transport = FakeRosbridgeTransport(physical_mode=None)
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = _plan_success(tools, transport, "unknown_physical")
    assert plan["ok"] is True

    result = tools.execute_plan("UR10", "unknown_physical")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "physical mode unknown"
    assert "confirmed false" in result["feedback"]["correction"]
    assert any(c["name"] == "physical_mode_safe" and not c["passed"] for c in result["verification"]["checks"])
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_blocks_true_physical_mode_before_publishing():
    transport = FakeRosbridgeTransport(physical_mode=True)
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = _plan_success(tools, transport, "physical_enabled")
    assert plan["ok"] is True

    result = tools.execute_plan("UR10", "physical_enabled")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "physical mode enabled"
    assert "confirmed false" in result["feedback"]["correction"]
    assert any(c["name"] == "physical_mode_safe" and not c["passed"] for c in result["verification"]["checks"])
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_verifies_fake_controller_joint_state_matches_final_positions_within_tolerance():
    transport = FakeRosbridgeTransport(physical_mode=False)
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = _plan_success(tools, transport, "plan_a")
    assert plan["ok"] is True
    intermediate = [0.0, -1.0, 1.0, 0.0, 0.0, 0.0]
    observed = [0.0, -1.5705, 1.5705, 0.0, 0.0, 0.0]
    transport.queue_joint_state_after_publish(JOINT_TOPIC, intermediate)
    transport.queue_joint_state_after_publish(JOINT_TOPIC, observed)

    result = tools.execute_plan("UR10", "plan_a")

    assert result["ok"] is True
    assert result["feedback"]["phase"] == "executed"
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["expected_joint_state"] == FINAL_POSITIONS
    assert result["raw"]["observed_joint_state"] == observed
    assert any(c["name"] == "final_joint_positions_match" and c["passed"] for c in result["verification"]["checks"])
    assert transport.published[-1] == ("/UR10/command/execute", {"data": "plan_a"})


def test_execute_verifies_fake_controller_joint_state_by_joint_name():
    planned_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    final_positions = [-1.4529, -0.9237, 0.9681, 1.5264, 1.5708, -1.4529]
    observed_names = [
        "elbow_joint",
        "shoulder_lift_joint",
        "shoulder_pan_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    observed_positions = [
        final_positions[2],
        final_positions[1],
        final_positions[0],
        final_positions[3],
        final_positions[4],
        final_positions[5],
    ]
    transport = FakeRosbridgeTransport(physical_mode=False)
    tools = MoveItMcpTools.with_fake_transport(transport)
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="joint_name_order_plan",
        points=3,
        final_positions=final_positions,
        joint_names=planned_names,
    )
    plan = tools.plan_free_motion("UR10", "joint_name_order_plan", {"x": 0.5, "y": 0.0, "z": 0.3})
    assert plan["ok"] is True
    transport.queue_joint_state_after_publish(
        JOINT_TOPIC,
        observed_positions,
        names=observed_names,
    )

    result = tools.execute_plan("UR10", "joint_name_order_plan")

    assert result["ok"] is True
    assert result["verification"]["result"] == "pass"
    assert result["raw"]["expected_joint_state"] == final_positions
    assert result["raw"]["observed_joint_state"] == observed_positions
    assert any(c["name"] == "final_joint_positions_match" and c["passed"] for c in result["verification"]["checks"])


def test_execute_rejects_after_failed_replan_reuses_successful_name():
    transport = FakeRosbridgeTransport(physical_mode=False)
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = _plan_success(tools, transport, "reuse_after_failure")
    assert plan["ok"] is True

    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    failed_replan = tools.plan_free_motion(
        "UR10",
        "reuse_after_failure",
        {"x": 0.6, "y": 0.0, "z": 0.3},
        timeout_s=0.1,
        allow_existing_name=True,
    )
    assert failed_replan["ok"] is False

    result = tools.execute_plan("UR10", "reuse_after_failure")

    assert result["ok"] is False
    assert result["feedback"]["status"] == "plan not verified"
    assert not any(topic == "/UR10/command/execute" for topic, _ in transport.published)


def test_execute_fails_when_fake_controller_joint_state_does_not_match_final_positions():
    transport = FakeRosbridgeTransport(physical_mode=False)
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = _plan_success(tools, transport, "mismatch_plan")
    assert plan["ok"] is True
    observed = [9.0, 9.0, 9.0, 9.0, 9.0, 9.0]
    transport.queue_joint_state_after_publish(JOINT_TOPIC, observed)

    result = tools.execute_plan("UR10", "mismatch_plan", timeout_s=0.1)

    assert result["ok"] is False
    assert result["feedback"]["status"] == "execution unverified"
    assert "retry execution only after a verified plan" in result["feedback"]["correction"]
    assert result["verification"]["result"] == "fail"
    assert any(c["name"] == "final_joint_positions_match" and not c["passed"] for c in result["verification"]["checks"])
    assert transport.published[-1] == ("/UR10/command/execute", {"data": "mismatch_plan"})


def test_execute_pick_plan_closes_attaches_lifts_and_verifies_object() -> None:
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_beam",
        points=4,
        final_positions=FINAL_POSITIONS,
    )
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools = MoveItMcpTools.with_fake_transport(transport)
    plan = tools.plan_pick(
        "UR10",
        "beam_001",
        plan_name="pick_beam",
        planning_strategy="cartesian",
        timeout_s=0.1,
    )
    assert plan["ok"] is True

    transport.queue_joint_state_after_publish(JOINT_TOPIC, FINAL_POSITIONS)
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="pick_beam__lift",
        points=3,
        final_positions=FINAL_POSITIONS,
    )
    transport.queue_joint_state_after_publish(JOINT_TOPIC, FINAL_POSITIONS)

    result = tools.execute_plan("UR10", "pick_beam", timeout_s=0.1)

    assert result["ok"] is True
    assert result["feedback"]["status"] == "pick motion, grasp, attach, and lift verified"
    assert result["raw"]["pick"]["object_name"] == "beam_001"
    assert result["raw"]["pick"]["gripper_closed"] is True
    assert result["raw"]["pick"]["planning_scene_attached"] is True
    assert result["raw"]["pick"]["lift_executed"] is True
    assert tools.gripper.attached_object("UR10") == "beam_001"


def test_execute_pick_task_solution_runs_stored_stages_in_order() -> None:
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools = MoveItMcpTools.with_fake_transport(transport)
    planned = tools.plan_pick_task("UR10", "beam_001", timeout_s=0.1)
    assert planned["ok"] is True
    _queue_pick_task_motion_execution(transport, planned["raw"]["task_solution_id"])

    result = tools.execute_task_solution(
        "UR10",
        planned["raw"]["task_solution_id"],
        timeout_s=0.1,
    )

    assert result["ok"] is True
    assert result["tool"] == "moveit_execute_task_solution"
    assert result["feedback"]["execution_target"] == "task_solution"
    assert result["raw"]["task_solution_id"] == planned["raw"]["task_solution_id"]
    assert [stage["name"] for stage in result["raw"]["stages"]] == [
        "observe_current_state",
        "connect_to_pre_grasp",
        "approach_grasp",
        "close_gripper",
        "attach_object",
        "lift_object",
        "verify_attached_object",
    ]
    assert all(stage["status"] == "executed" for stage in result["raw"]["stages"])
    assert result["raw"]["stage_report"]["executed"] == 7
    assert result["raw"]["evidence"][-1] == {"kind": "stage_report", "count": 7}
    assert tools.gripper.attached_object("UR10") == "beam_001"
    assert [payload for topic, payload in transport.published if topic == "/UR10/command/execute"] == [
        {"data": f"{planned['raw']['task_solution_id']}__connect_to_pre_grasp"},
        {"data": f"{planned['raw']['task_solution_id']}__approach_grasp"},
        {"data": f"{planned['raw']['task_solution_id']}__lift_object"},
    ]
    assert [topic for topic, _ in transport.published].count("/UR10/request/free") == 1
    assert [topic for topic, _ in transport.published].count("/UR10/request/cartesian") == 2


def test_execute_manipulation_task_reapplies_pick_contact_allowance_for_approach() -> None:
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    _queue_staged_hold_preview(transport)
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools = MoveItMcpTools.with_fake_transport(transport)
    planned = tools.plan_manipulation_task(
        "UR10",
        requirements={"goal": "hold", "object_name": "beam_001"},
        backend="staged_moveit",
        timeout_s=0.1,
    )
    assert planned["ok"] is True
    initial_enabled_acms = [
        payload["allowed_collision_matrix"]
        for payload in transport.applied_planning_scenes
        if _acm_allows(payload.get("allowed_collision_matrix", {}), "beam_001", "tool0")
    ]

    for _ in range(3):
        transport.queue_joint_state_after_publish(JOINT_TOPIC, FINAL_POSITIONS)
    result = tools.execute_task_solution("UR10", planned["raw"]["task_solution_id"], timeout_s=0.1)

    assert result["ok"] is False
    assert result["feedback"]["status"] == "approach_to_pre_grasp failed"
    enabled_acms = [
        payload["allowed_collision_matrix"]
        for payload in transport.applied_planning_scenes
        if _acm_allows(payload.get("allowed_collision_matrix", {}), "beam_001", "tool0")
    ]
    assert len(initial_enabled_acms) == 2
    assert len(enabled_acms) == 3


def test_attach_object_accepts_verified_external_gripper_close_without_action_goal() -> None:
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.attach_object(
        "UR10",
        "beam_001",
        verified_gripper_closed=True,
    )

    assert result["ok"] is True
    assert result["raw"]["gripper_state"] == "closed"
    assert result["raw"]["verified_gripper_closed"] is True
    assert tools.gripper.attached_object("UR10") == "beam_001"
    assert transport.action_goals == []


def test_execute_task_solution_consumes_id_before_stage_side_effects() -> None:
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools = MoveItMcpTools.with_fake_transport(transport)
    planned = tools.plan_pick_task("UR10", "beam_001", timeout_s=0.1)
    assert planned["ok"] is True
    task_solution_id = planned["raw"]["task_solution_id"]
    _queue_pick_task_motion_execution(transport, task_solution_id)
    first = tools.execute_task_solution("UR10", task_solution_id, timeout_s=0.1)
    assert first["ok"] is True
    published_count = len(transport.published)
    action_goal_count = len(transport.action_goals)
    scene_update_count = len(transport.applied_planning_scenes)

    second = tools.execute_task_solution("UR10", task_solution_id, timeout_s=0.1)

    assert second["ok"] is False
    assert second["feedback"]["phase"] == "pre_execute"
    assert second["error"] == "unknown_task_solution_id"
    assert second["retryable"] is False
    assert second["raw"]["task_solution_id"] == task_solution_id
    assert len(transport.published) == published_count
    assert len(transport.action_goals) == action_goal_count
    assert len(transport.applied_planning_scenes) == scene_update_count


def test_execute_place_plan_releases_attached_object_after_verified_motion():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_planning_scene("UR10", ATTACHED_SCENE, planning_frame="base_link")
    transport.set_current_pose(
        "UR10",
        {
            "position": {"x": 0.45, "y": 0.2, "z": 0.35},
            "orientation": {"x": 1.0, "y": 0.0, "z": 0.0, "w": 0.0},
        },
        planning_frame="base_link",
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="place_release",
        points=4,
        final_positions=FINAL_POSITIONS,
    )
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.085, "requested_position": 0.085})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.0])
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.attach("UR10", "beam_001")
    plan = tools.plan_place(
        "UR10",
        "beam_001",
        plan_name="place_release",
        target_position={"x": 0.55, "y": 0.2, "z": 0.12},
        timeout_s=0.1,
    )
    assert plan["ok"] is True
    transport.queue_joint_state_after_publish(JOINT_TOPIC, FINAL_POSITIONS)

    result = tools.execute_plan("UR10", "place_release", timeout_s=0.1)

    assert result["ok"] is True
    assert result["raw"]["release"]["object_name"] == "beam_001"
    assert result["raw"]["release"]["gripper_opened"] is True
    assert result["raw"]["release"]["planning_scene_released"] is True
    assert tools.gripper.attached_object("UR10") is None
    scene_objects = tools.list_scene_objects("UR10", timeout_s=0.1)
    beam = next(obj for obj in scene_objects["raw"]["objects"] if obj["name"] == "beam_001")
    assert beam["state"] == "free"
    assert beam["bounds"]["center"] == {"x": 0.55, "y": 0.2, "z": 0.12}


def test_release_object_requires_verified_gripper_open():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.set_state("UR10", "closed")
    tools.gripper.attach("UR10", "beam_001")

    result = tools.release_object(
        "UR10",
        "beam_001",
        object_pose={
            "position": {"x": 0.55, "y": 0.2, "z": 0.12},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        verified_gripper_open=False,
        timeout_s=0.1,
    )

    assert result["ok"] is False
    assert result["feedback"]["status"] == "verified gripper open required"
    assert tools.gripper.attached_object("UR10") == "beam_001"


def test_release_and_verify_released_object_after_verified_open():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)
    tools.gripper.set_state("UR10", "closed")
    tools.gripper.attach("UR10", "beam_001")

    release = tools.release_object(
        "UR10",
        "beam_001",
        object_pose={
            "position": {"x": 0.55, "y": 0.2, "z": 0.12},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        verified_gripper_open=True,
        timeout_s=0.1,
    )
    verify = tools.verify_released_object("UR10", "beam_001", timeout_s=0.1)

    assert release["ok"] is True
    assert release["tool"] == "moveit_release_object"
    assert release["raw"]["planning_scene_released"] is True
    assert tools.gripper.attached_object("UR10") is None
    assert verify["ok"] is True
    assert verify["tool"] == "moveit_verify_released_object"
    assert verify["raw"]["planning_scene_state"] == "free"


def test_remove_scene_object_requires_free_world_object_and_verifies_readback():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene(
        "UR10",
        {
            "scene": {
                "world": {"collision_objects": [BEAM_OBJECT]},
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.remove_scene_object("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is True
    assert result["tool"] == "moveit_remove_scene_object"
    assert result["raw"]["object_name"] == "beam_001"
    assert result["raw"]["planning_scene_state"] == "removed"
    assert not tools.list_scene_objects("UR10", timeout_s=0.1)["raw"]["objects"]


def test_remove_scene_object_refuses_attached_object():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", ATTACHED_SCENE, planning_frame="base_link")
    tools = MoveItMcpTools.with_fake_transport(transport)

    result = tools.remove_scene_object("UR10", "beam_001", timeout_s=0.1)

    assert result["ok"] is False
    assert result["feedback"]["status"] == "object attached"
    assert "Release and verify" in result["feedback"]["correction"]
