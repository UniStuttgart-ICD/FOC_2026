from moveit_mcp.tools import MoveItMcpTools
from moveit_mcp.vizor_client import FakeRosbridgeTransport, VizorClient

ENVELOPE_KEYS = {"ok", "robot", "tool", "feedback", "verification", "evidence", "raw"}

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
                    "operation": 0,
                }
            ]
        },
        "robot_state": {"attached_collision_objects": []},
        "object_colors": [],
    }
}


def test_open_and_close_gripper_return_verified_state():
    transport = FakeRosbridgeTransport()
    tools = MoveItMcpTools(client=VizorClient(transport=transport))

    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.085, "requested_position": 0.085})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.0])
    opened = tools.open_gripper("UR10")
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    closed = tools.close_gripper("UR10")

    assert ENVELOPE_KEYS.issubset(opened)
    assert opened["ok"] is True
    assert opened["tool"] == "open_gripper"
    assert opened["verification"]["result"] == "pass"
    assert opened["raw"]["gripper_state"] == "open"
    assert closed["ok"] is True
    assert closed["tool"] == "close_gripper"
    assert closed["verification"]["result"] == "pass"
    assert closed["raw"]["gripper_state"] == "closed"


def test_close_gripper_sends_robotiq_action_goal_and_verifies_joint_state():
    transport = FakeRosbridgeTransport()
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools = MoveItMcpTools(client=VizorClient(transport=transport))

    result = tools.close_gripper("UR10")

    assert result["ok"] is True
    assert result["feedback"]["message"] == "Gripper close command completed through the Robotiq action server"
    assert result["raw"]["gripper_state"] == "closed"
    assert result["raw"]["action_name"] == "/UR10/command_robotiq_action"
    assert result["raw"]["goal_position_m"] == 0.0
    assert result["raw"]["expected_joint_position"] == 0.8
    assert result["raw"]["observed_joint_position"] == 0.8
    assert result["evidence"] == [
        {
            "kind": "ros_action",
            "summary": "goal position 0.000m",
            "path": "/UR10/command_robotiq_action",
        },
        {"kind": "ros_topic", "summary": "0.8", "topic": "/UR10/gripper_joint_states"},
        {"kind": "mcp_state", "summary": "closed"},
    ]

    assert transport.published == []
    assert transport.action_goals == [
        (
            "/UR10/command_robotiq_action",
            "robotiq_2f_gripper_msgs/CommandRobotiqGripperAction",
            {
                "emergency_release": False,
                "emergency_release_dir": 0,
                "stop": False,
                "position": 0.0,
                "speed": 0.05,
                "force": 50.0,
            },
        )
    ]


def test_attach_object_requires_closed_gripper():
    tools = MoveItMcpTools.with_fake_transport(FakeRosbridgeTransport())

    result = tools.attach_object("UR10", "beam_001")

    assert ENVELOPE_KEYS.issubset(result)
    assert result["ok"] is False
    assert result["feedback"]["status"] == "gripper not closed"
    assert "close_gripper" in result["feedback"]["correction"]
    assert result["raw"]["gripper_state"] == "open"
    assert result["raw"]["attached_object"] is None


def test_attach_object_applies_moveit_attached_collision_object():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE)
    tools = MoveItMcpTools(client=VizorClient(transport=transport, task_id_factory=lambda: 1))
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools.close_gripper("UR10")

    result = tools.attach_object("UR10", "beam_001")

    assert result["ok"] is True
    assert result["raw"]["attached_object"] == "beam_001"
    assert result["raw"]["scene_update_published"] is True
    assert result["raw"]["attached_to"] == "tool0"
    assert result["verification"]["result"] == "pass"
    assert transport.published == []
    payload = transport.applied_planning_scenes[-1]
    attached = payload["robot_state"]["attached_collision_objects"][0]
    assert payload["is_diff"] is True
    assert payload["robot_state"]["is_diff"] is True
    assert attached["link_name"] == "tool0"
    assert attached["object"]["id"] == "beam_001"
    assert attached["object"]["operation"] == 0
    assert payload["world"]["collision_objects"][0]["id"] == "beam_001"
    assert payload["world"]["collision_objects"][0]["operation"] == 1


def test_attach_object_allows_verifier_to_confirm_service_scene_attachment():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene("UR10", PLANNING_SCENE)
    tools = MoveItMcpTools(client=VizorClient(transport=transport))
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools.close_gripper("UR10")

    attached = tools.attach_object("UR10", "beam_001")
    verified = tools.verify_attached_object("UR10", "beam_001", timeout_s=0.1)

    assert attached["ok"] is True
    assert verified["ok"] is True
    assert verified["raw"]["planning_scene_state"] == "attached"
    assert verified["raw"]["attached_to"] == "tool0"


def test_opening_gripper_clears_attached_object():
    transport = FakeRosbridgeTransport()
    tools = MoveItMcpTools(client=VizorClient(transport=transport))
    transport.set_planning_scene("UR10", PLANNING_SCENE)
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.0, "requested_position": 0.0})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.8])
    tools.close_gripper("UR10")
    attached = tools.attach_object("UR10", "beam_001")
    assert attached["ok"] is True

    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.085, "requested_position": 0.085})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.0])
    opened = tools.open_gripper("UR10")

    assert opened["ok"] is True
    assert opened["raw"]["gripper_state"] == "open"
    assert opened["raw"]["attached_object"] is None
