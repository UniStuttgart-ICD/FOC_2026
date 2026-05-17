import json

from robot_control.context import RobotContextStore


def test_empty_robot_context_renders_advisory_block() -> None:
    store = RobotContextStore()

    text = store.render_instruction_block()

    assert "Last-known robot context" in text
    assert "No robot status has been observed yet" in text
    assert "advisory only" in text
    assert "moveit_get_current_pose" in text


def test_robot_context_updates_from_current_pose_tool_output() -> None:
    store = RobotContextStore()
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
            }
        }
    )

    store.update_from_tool_result("moveit_get_current_pose", output)

    text = store.render_instruction_block()
    assert "UR10" in text
    assert "x=0.100" in text
    assert "y=0.200" in text
    assert "z=0.300" in text


def test_robot_context_ignores_legacy_status_tool_output() -> None:
    store = RobotContextStore()
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot_name": "UR10",
                "tcp_pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
                "gripper": {"state": "open"},
                "last_execution": {"result": "pass"},
            }
        }
    )

    store.update_from_tool_result("moveit_get_robot_status", output)

    text = store.render_instruction_block()
    assert "gripper: open" not in text
    assert "last execution: pass" not in text


def test_robot_context_updates_from_robot_state_observation() -> None:
    store = RobotContextStore(time_fn=lambda: 10.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "raw": {
                    "pose": {
                        "position": {"x": 0.57, "y": 0.39, "z": 0.62},
                        "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
                    },
                    "physical_mode": False,
                    "joint_state": [0, -1.57, 1.57, 0, 0, 0],
                },
            }
        }
    )

    store.update_from_tool_result("moveit_get_robot_state", output)

    assert store.has_recent_robot_observation(max_age_s=1.0)
    latest_pose = store.latest_tcp_pose()
    assert latest_pose is not None
    assert latest_pose["position"]["z"] == 0.62


def test_robot_context_reports_recent_and_stale_pose_observations() -> None:
    now = 100.0
    store = RobotContextStore(time_fn=lambda: now)

    assert store.has_recent_robot_observation(max_age_s=15.0) is False

    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
            }
        }
    )
    store.update_from_tool_result("moveit_get_current_pose", output)

    assert store.has_recent_robot_observation(max_age_s=15.0) is True

    now = 116.0
    assert store.has_recent_robot_observation(max_age_s=15.0) is False


def test_robot_context_renders_recent_task_failure_for_recovery() -> None:
    store = RobotContextStore(time_fn=lambda: 500.0)

    store.remember_task_failure(
        task_solution_id="place_task_dynamic_5_002",
        task_kind="place",
        object_name="dynamic_5",
        scene_snapshot_id="scene_20260515_001",
        failed_step="retreat",
        failed_stage="planning",
        failed_tool_name="moveit_plan_cartesian_motion",
        failed_tool_arguments={"plan_name": "place_task_dynamic_5_002_retreat_try2"},
        failed_tool_result={"ok": False, "feedback": {"status": "incomplete path"}},
        completed_steps=[
            {"name": "release_pose", "handler": "motion"},
            {"name": "open_gripper", "handler": "open_gripper"},
            {"name": "release_object", "handler": "release_object"},
        ],
        verified_plan_names=["place_task_dynamic_5_002_release_pose_try1"],
        gripper_state="open",
        attached_object_verified=False,
        released_object_verified=False,
    )

    failure = store.recent_task_failure
    assert failure is not None
    assert failure.task_solution_id == "place_task_dynamic_5_002"
    assert failure.failed_step == "retreat"
    assert failure.failed_tool_result["feedback"]["status"] == "incomplete path"

    text = store.render_instruction_block()
    assert "recent task failure: place_task_dynamic_5_002" in text
    assert "object: dynamic_5" in text
    assert "failed step: retreat" in text
    assert "completed steps: release_pose, open_gripper, release_object" in text
    assert "recovery requires explicit user/operator intent" in text


def test_robot_context_remembers_recent_executable_plan_names() -> None:
    now = 200.0
    store = RobotContextStore(time_fn=lambda: now)

    store.remember_executable_plan(
        "plan-1",
        robot_name="UR10",
        source_tool="moveit_plan_free_motion",
    )

    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is True
    assert store.has_recent_executable_plan("missing", max_age_s=60.0) is False
    pending = store.pending_executable_plan("plan-1", max_age_s=60.0)
    assert pending is not None
    assert pending.robot_name == "UR10"
    assert pending.source_tool == "moveit_plan_free_motion"
    assert pending.observed_at_s == 200.0
    assert "pending executable plan: plan-1" in store.render_instruction_block()

    now = 261.0
    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is False
    assert store.pending_executable_plan("plan-1", max_age_s=60.0) is None


def test_robot_context_returns_latest_recent_pending_plan() -> None:
    now = 200.0
    store = RobotContextStore(time_fn=lambda: now)

    store.remember_executable_plan("plan-1", robot_name="UR10")
    now = 205.0
    store.remember_executable_plan("plan-2", robot_name="UR10")

    latest = store.latest_pending_executable_plan(max_age_s=60.0)

    assert latest is not None
    assert latest.plan_name == "plan-2"

    now = 266.0
    assert store.latest_pending_executable_plan(max_age_s=60.0) is None


def test_robot_context_updates_pending_plan_from_planning_tool_output() -> None:
    store = RobotContextStore(time_fn=lambda: 250.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "plan-2"},
            }
        }
    )

    store.update_from_tool_result("moveit_plan_free_motion", output)

    pending = store.pending_executable_plan("plan-2", max_age_s=10.0)
    assert pending is not None
    assert pending.robot_name == "UR10"
    assert pending.source_tool == "moveit_plan_free_motion"


def test_robot_context_updates_pending_plan_from_pick_planning_tool_output() -> None:
    store = RobotContextStore(time_fn=lambda: 255.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "pick-plan-1"},
            }
        }
    )

    store.update_from_tool_result("moveit_plan_pick", output)

    pending = store.pending_executable_plan("pick-plan-1", max_age_s=10.0)
    assert pending is not None
    assert pending.robot_name == "UR10"
    assert pending.source_tool == "moveit_plan_pick"


def test_robot_context_ignores_partial_legacy_pick_diagnostic() -> None:
    store = RobotContextStore(time_fn=lambda: 255.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "error": "pick_segment_planning_failed",
                "failed_segment": "local_cartesian_pick",
                "feedback": {"can_execute": False},
                "raw": {
                    "partial_plan": {
                        "kind": "preposition",
                        "plan_name": "pick_dynamic_5_preposition",
                    }
                },
            }
        }
    )

    store.update_from_tool_result("moveit_plan_pick", output)

    assert store.pending_plan is None


def test_robot_context_tracks_task_solution_without_pending_plan() -> None:
    store = RobotContextStore(time_fn=lambda: 260.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True, "execution_target": "task_solution"},
                "raw": {
                    "task_solution_id": "pick_task_dynamic_5_001",
                    "task_kind": "pick",
                    "backend": "emulated",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": "scene_20260515_001",
                    "waypoints": [
                        {"position": {"x": 0.4, "y": 0.1, "z": 0.3}},
                        {"position": {"x": 0.5, "y": 0.1, "z": 0.3}},
                    ],
                    "workflow_steps": [
                        {"kind": "motion", "name": "approach", "waypoint_index": 0},
                        {"kind": "motion", "name": "pre_grasp", "waypoint_index": 1},
                    ],
                    "approval": {
                        "required": True,
                        "target_kind": "task_solution",
                        "task_solution_id": "pick_task_dynamic_5_001",
                        "source_tool": "moveit_plan_pick_task",
                        "object_name": "dynamic_5",
                        "expected_movement": "approach grasp, close gripper, attach object, lift object",
                        "scene_snapshot_id": "scene_20260515_001",
                    },
                },
            }
        }
    )

    store.update_from_tool_result("moveit_plan_pick_task", output)

    assert store.pending_plan is None
    solution = store.recent_task_solution
    assert solution is not None
    assert solution.task_solution_id == "pick_task_dynamic_5_001"
    assert solution.task_kind == "pick"
    assert solution.object_name == "dynamic_5"
    assert solution.backend == "emulated"
    assert solution.scene_snapshot_id == "scene_20260515_001"
    assert solution.approval_required is True
    assert solution.raw is not None
    assert solution.raw["waypoints"][1]["position"]["x"] == 0.5
    assert solution.raw["workflow_steps"][0]["name"] == "approach"
    approval = store.pending_task_solution_approval
    assert approval is not None
    assert approval.target_kind == "task_solution"
    assert approval.task_solution_id == "pick_task_dynamic_5_001"
    assert approval.source_tool == "moveit_plan_pick_task"
    assert approval.object_name == "dynamic_5"
    assert approval.expected_movement == "approach grasp, close gripper, attach object, lift object"
    assert approval.scene_snapshot_id == "scene_20260515_001"
    assert approval.approval_turn_id is None
    assert approval.approved_at is None


def test_robot_context_records_task_solution_approval() -> None:
    store = RobotContextStore(time_fn=lambda: 300.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="approach grasp, close gripper, attach object, lift object",
        scene_snapshot_id="scene_20260515_001",
    )

    assert store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-7",
        approved_at=299.0,
    ) is True

    approval = store.pending_task_solution_approval
    assert approval is not None
    assert approval.approval_turn_id == "turn-7"
    assert approval.approved_at == 299.0


def test_robot_context_clamps_future_task_solution_approval_time_to_store_clock() -> None:
    now = 100.0
    store = RobotContextStore(time_fn=lambda: now)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )

    assert store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-7",
        approved_at=9999.0,
    ) is True

    approval = store.pending_task_solution_approval
    assert approval is not None
    assert approval.approved_at == 100.0

    now = 161.1
    status = store.task_solution_execution_approval_status(
        "pick_task_dynamic_5_001",
        scene_snapshot_id="scene_20260515_001",
    )
    assert status.ok is False
    assert status.reason == "approval_expired"


def test_robot_context_rejects_stale_task_solution_approval_after_new_user_intent() -> None:
    store = RobotContextStore(time_fn=lambda: 400.0)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="approach grasp, close gripper, attach object, lift object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-7",
        approved_at=401.0,
    )

    assert store.task_solution_execution_approval_status(
        "pick_task_dynamic_5_001",
        scene_snapshot_id="scene_20260515_001",
    ).ok is True

    store.mark_new_user_intent()

    status = store.task_solution_execution_approval_status(
        "pick_task_dynamic_5_001",
        scene_snapshot_id="scene_20260515_001",
    )
    assert status.ok is False
    assert status.reason == "approval_stale_after_new_user_intent"


def test_robot_context_rejects_task_solution_approval_after_60_seconds() -> None:
    now = 400.0
    store = RobotContextStore(time_fn=lambda: now)
    store.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_compound_task",
        object_name="dynamic_5",
        expected_movement="hold object",
        scene_snapshot_id="scene_20260515_001",
    )
    store.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-7",
        approved_at=400.0,
    )

    now = 460.0
    assert store.task_solution_execution_approval_status(
        "pick_task_dynamic_5_001",
        scene_snapshot_id="scene_20260515_001",
    ).ok is True

    now = 460.1
    status = store.task_solution_execution_approval_status(
        "pick_task_dynamic_5_001",
        scene_snapshot_id="scene_20260515_001",
    )

    assert status.ok is False
    assert status.reason == "approval_expired"


def test_robot_context_preserves_pick_follow_up_after_success() -> None:
    store = RobotContextStore(time_fn=lambda: 256.0)
    follow_up_args = {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "plan_name": "local-pick-plan",
        "planning_strategy": "cartesian",
    }
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {
                    "plan_name": "preposition-plan",
                    "next_action": {
                        "after_success": {
                            "tool": "moveit_plan_pick",
                            "arguments": follow_up_args,
                        }
                    },
                },
            }
        }
    )

    store.update_from_tool_result("moveit_plan_pick", output)

    pending = store.pending_executable_plan("preposition-plan", max_age_s=10.0)
    assert pending is not None
    assert pending.after_success == {
        "tool": "moveit_plan_pick",
        "arguments": follow_up_args,
    }
    assert pending.execute_via_mcp is True


def test_robot_context_preserves_place_follow_up_after_success() -> None:
    store = RobotContextStore(time_fn=lambda: 257.0)
    follow_up_args = {
        "robot_name": "UR10",
        "object_name": "beam_001",
        "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
        "orientation_mode": "horizontal",
    }
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {
                    "plan_name": "place-preposition-plan",
                    "next_action": {
                        "after_success": {
                            "tool": "moveit_plan_place",
                            "arguments": follow_up_args,
                        }
                    },
                },
            }
        }
    )

    store.update_from_tool_result("moveit_plan_place", output)

    pending = store.pending_executable_plan("place-preposition-plan", max_age_s=10.0)
    assert pending is not None
    assert pending.after_success == {
        "tool": "moveit_plan_place",
        "arguments": follow_up_args,
    }


def test_robot_context_marks_pick_workflow_plan_for_mcp_execution() -> None:
    store = RobotContextStore(time_fn=lambda: 258.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "local-pick-plan", "workflow_kind": "pick"},
            }
        }
    )

    store.update_from_tool_result("moveit_plan_pick", output)

    pending = store.pending_executable_plan("local-pick-plan", max_age_s=10.0)
    assert pending is not None
    assert pending.execute_via_mcp is True


def test_robot_context_updates_pending_plan_from_place_planning_tool_output() -> None:
    store = RobotContextStore(time_fn=lambda: 260.0)
    output = json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True},
                "raw": {"plan_name": "place-plan-1"},
            }
        }
    )

    store.update_from_tool_result("moveit_plan_place", output)

    pending = store.pending_executable_plan("place-plan-1", max_age_s=10.0)
    assert pending is not None
    assert pending.robot_name == "UR10"
    assert pending.source_tool == "moveit_plan_place"


def test_robot_context_consumes_pending_plan_after_execution() -> None:
    store = RobotContextStore(time_fn=lambda: 300.0)
    store.remember_executable_plan("plan-1", robot_name="UR10")

    assert store.consume_executable_plan("plan-1") is True
    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is False
    assert store.consume_executable_plan("plan-1") is False


def test_robot_context_tracks_recent_gripper_state_from_gripper_tools() -> None:
    now = 300.0
    store = RobotContextStore(time_fn=lambda: now)
    ok_output = json.dumps({"structured_content": {"ok": True}})

    assert store.gripper_state() is None
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is False

    store.update_from_tool_result("moveit_close_gripper", ok_output)
    assert store.gripper_state() == "closed"
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is True

    now = 331.0
    assert store.has_recent_gripper_state("closed", max_age_s=30.0) is False

    store.update_from_tool_result("moveit_open_gripper", ok_output)
    assert store.gripper_state() == "open"
    assert store.has_recent_gripper_state("open", max_age_s=30.0) is True


def test_robot_context_tracks_held_object_until_verified_release_proof() -> None:
    store = RobotContextStore()

    store.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": "dynamic_5",
                        "mcp_gripper_holds_object": True,
                        "planning_scene_state": "attached",
                    },
                }
            }
        ),
    )

    assert "held object: dynamic_5" in store.render_instruction_block()
    assert store.held_object_name() == "dynamic_5"

    store.update_from_tool_result(
        "moveit_open_gripper",
        json.dumps({"structured_content": {"ok": True}}),
    )

    assert "held object: dynamic_5" in store.render_instruction_block()

    store.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": None,
                        "mcp_gripper_holds_object": False,
                        "planning_scene_state": "free",
                    },
                }
            }
        ),
    )

    assert "held object: dynamic_5" not in store.render_instruction_block()
    assert store.held_object_name() is None


def test_robot_context_requires_recent_held_object_evidence() -> None:
    now = 100.0
    store = RobotContextStore(time_fn=lambda: now)

    store.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "planning_scene_state": "attached",
                        "mcp_gripper_holds_object": True,
                    },
                }
            }
        ),
    )

    assert store.has_recent_held_object("dynamic_5", max_age_s=30.0) is True

    now = 131.0

    assert store.has_recent_held_object("dynamic_5", max_age_s=30.0) is False


def test_robot_context_keeps_held_object_when_release_proof_still_names_attached_object() -> None:
    store = RobotContextStore()
    _mark_dynamic_5_held(store)

    store.update_from_tool_result(
        "moveit_verify_released_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": "dynamic_5",
                        "mcp_gripper_holds_object": False,
                        "planning_scene_state": "free",
                    },
                }
            }
        ),
    )

    assert "held object: dynamic_5" in store.render_instruction_block()
    assert store.held_object_name() == "dynamic_5"


def test_robot_context_keeps_held_object_when_release_proof_scene_state_attached() -> None:
    store = RobotContextStore()
    _mark_dynamic_5_held(store)

    store.update_from_tool_result(
        "moveit_verify_released_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": None,
                        "mcp_gripper_holds_object": False,
                        "planning_scene_state": "attached",
                    },
                }
            }
        ),
    )

    assert "held object: dynamic_5" in store.render_instruction_block()
    assert store.held_object_name() == "dynamic_5"


def test_robot_context_does_not_track_held_object_without_attachment_evidence() -> None:
    store = RobotContextStore()

    store.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                    },
                }
            }
        ),
    )

    assert "held object: dynamic_5" not in store.render_instruction_block()


def _mark_dynamic_5_held(store: RobotContextStore) -> None:
    store.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": "dynamic_5",
                        "mcp_gripper_holds_object": True,
                        "planning_scene_state": "attached",
                    },
                }
            }
        ),
    )
