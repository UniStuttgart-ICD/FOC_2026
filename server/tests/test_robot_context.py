import json

from voice_runtime.robot_context import RobotContextStore


def test_empty_robot_context_renders_advisory_block() -> None:
    store = RobotContextStore()

    text = store.render_instruction_block()

    assert "Last-known robot context" in text
    assert "No robot status has been observed yet" in text
    assert "advisory only" in text
    assert "moveit_get_robot_status" in text


def test_robot_context_updates_from_status_tool_output() -> None:
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
    assert "UR10" in text
    assert "x=0.100" in text
    assert "y=0.200" in text
    assert "z=0.300" in text
    assert "gripper: open" in text
    assert "last execution: pass" in text
