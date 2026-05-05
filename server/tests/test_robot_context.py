import json

from voice_runtime.robot_context import RobotContextStore


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


def test_robot_context_still_accepts_legacy_status_tool_output() -> None:
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
    assert "gripper: open" in text
    assert "last execution: pass" in text


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


def test_robot_context_remembers_recent_executable_plan_names() -> None:
    now = 200.0
    store = RobotContextStore(time_fn=lambda: now)

    store.remember_executable_plan("plan-1")

    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is True
    assert store.has_recent_executable_plan("missing", max_age_s=60.0) is False

    now = 261.0
    assert store.has_recent_executable_plan("plan-1", max_age_s=60.0) is False


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
