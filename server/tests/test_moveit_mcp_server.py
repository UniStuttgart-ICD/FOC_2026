import sys

import pytest

import moveit_mcp.server as server
from moveit_mcp.server import build_mcp, build_tools
from moveit_mcp.tools import MoveItMcpTools
from moveit_mcp.vizor_client import FakeRosbridgeTransport, VizorClient

CANONICAL_TOOLS = {
    "moveit_get_current_pose",
    "moveit_get_robot_state",
    "moveit_list_scene_objects",
    "moveit_get_object_context",
    "moveit_plan_pick",
    "moveit_plan_place",
    "moveit_plan_compound_task",
    "moveit_plan_free_motion",
    "moveit_plan_cartesian_motion",
    "moveit_execute_plan",
    "moveit_explain_motion_failure",
    "moveit_verify_attached_object",
    "moveit_remove_scene_object",
    "moveit_open_gripper",
    "moveit_close_gripper",
    "moveit_attach_object",
}

LEGACY_TOOL_NAMES = {
    "plan_free_motion",
    "plan_cartesian_motion",
    "execute_plan",
    "get_current_pose",
    "open_gripper",
    "close_gripper",
    "attach_object",
}


def test_build_tools_uses_fake_transport_for_tests():
    tools = build_tools(transport=FakeRosbridgeTransport())

    assert hasattr(tools, "plan_free_motion")
    assert hasattr(tools, "execute_plan")
    assert hasattr(tools, "get_current_pose")
    assert hasattr(tools, "open_gripper")


def test_build_tools_can_enable_mtc_pick_task_backend():
    tools = build_tools(transport=FakeRosbridgeTransport(), pick_task_backend="mtc")

    assert tools.pick_task_backend == "mtc"


@pytest.mark.asyncio
async def test_build_mcp_registers_moveit_tools():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name for tool in await mcp.list_tools()}

    assert CANONICAL_TOOLS.issubset(registered)
    assert "moveit_plan_and_execute_free_motion" not in registered
    assert "moveit_plan_and_execute_cartesian_motion" not in registered


@pytest.mark.asyncio
async def test_legacy_tool_names_are_not_registered():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name for tool in await mcp.list_tools()}

    assert LEGACY_TOOL_NAMES.isdisjoint(registered)


@pytest.mark.asyncio
async def test_canonical_gripper_tool_calls_underlying_tools():
    transport = FakeRosbridgeTransport()
    transport.queue_action_result("/UR10/command_robotiq_action", {"position": 0.085, "requested_position": 0.085})
    transport.queue_joint_state_after_action("/UR10/gripper_joint_states", [0.0])
    tools = MoveItMcpTools(client=VizorClient(transport=transport, task_id_factory=lambda: 1))
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_open_gripper", {})

    assert payload["ok"] is True
    assert payload["tool"] == "open_gripper"
    assert payload["raw"]["gripper_state"] == "open"


@pytest.mark.asyncio
async def test_canonical_current_pose_tool_uses_ur10_by_default():
    transport = FakeRosbridgeTransport()
    transport.set_current_pose(
        "UR10",
        {
            "position": {"x": 0.57, "y": 0.39, "z": 0.62},
            "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
        },
        planning_frame="base_link",
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_get_current_pose", {})

    assert payload["ok"] is True
    assert payload["robot"] == "UR10"
    assert payload["tool"] == "get_current_pose"


@pytest.mark.asyncio
async def test_canonical_robot_state_tool_uses_ur10_by_default():
    transport = FakeRosbridgeTransport(physical_mode=False)
    transport.set_current_pose(
        "UR10",
        {
            "position": {"x": 0.57, "y": 0.39, "z": 0.62},
            "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
        },
        planning_frame="base_link",
    )
    transport.queue_joint_state("/UR10/move_group/fake_controller_joint_states", [0.0, -1.57, 1.57, 0.0, 0.0, 0.0])
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool("moveit_get_robot_state", {})

    assert payload["ok"] is True
    assert payload["robot"] == "UR10"
    assert payload["tool"] == "moveit_get_robot_state"
    assert payload["raw"]["physical_mode"] is False
    assert payload["raw"]["joint_state"] == [0.0, -1.57, 1.57, 0.0, 0.0, 0.0]


@pytest.mark.asyncio
async def test_canonical_scene_object_tools_use_ur10_by_default():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene(
        "UR10",
        {
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
                        }
                    ]
                },
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, listed = await mcp.call_tool("moveit_list_scene_objects", {})
    _, context = await mcp.call_tool("moveit_get_object_context", {"object_name": "beam_001"})

    assert listed["ok"] is True
    assert listed["robot"] == "UR10"
    assert listed["raw"]["objects"][0]["name"] == "beam_001"
    assert context["ok"] is True
    assert context["robot"] == "UR10"
    assert context["raw"]["object"]["name"] == "beam_001"


@pytest.mark.asyncio
async def test_canonical_plan_pick_tool_uses_ur10_by_default():
    transport = FakeRosbridgeTransport()
    transport.set_planning_scene(
        "UR10",
        {
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
                        }
                    ]
                },
                "robot_state": {"attached_collision_objects": []},
                "object_colors": [],
            }
        },
        planning_frame="base_link",
    )
    transport.queue_status_after_publish("/UR10/request/status", "success! ")
    transport.queue_planned_path_after_publish(
        "/UR10/request/planned_path",
        name="server_pick",
        points=4,
        final_positions=[0.0, -1.57, 1.57, 0.0, 0.0, 0.0],
    )
    tools = build_tools(transport=transport)
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(
        "moveit_plan_pick",
        {"object_name": "beam_001", "plan_name": "server_pick", "planning_strategy": "cartesian"},
    )

    assert payload["ok"] is True
    assert payload["robot"] == "UR10"
    assert payload["tool"] == "moveit_plan_pick"
    assert payload["raw"]["object_name"] == "beam_001"
    assert payload["raw"]["plan_name"] == "server_pick"


@pytest.mark.asyncio
async def test_canonical_planning_tools_have_agent_facing_descriptions():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name: tool for tool in await mcp.list_tools()}

    for name in ("moveit_plan_free_motion", "moveit_plan_cartesian_motion"):
        description = registered[name].description
        assert "raw.plan_name" in description
        assert "feedback.can_execute" in description
        assert "base_link" in description

    pick_description = registered["moveit_plan_pick"].description
    assert "moveit_list_scene_objects" in pick_description
    assert "moveit_get_object_context" in pick_description
    assert "raw.plan_name" in pick_description
    assert "raw.preposition" in pick_description
    assert "raw.workflow_segments" in pick_description
    assert "does not move" in pick_description.lower()

    place_description = registered["moveit_plan_place"].description
    assert "target object pose" in place_description
    assert "release TCP pose" in place_description
    assert "does not move" in place_description.lower()

    failure_description = registered["moveit_explain_motion_failure"].description
    assert "failed planner or executor result" in failure_description
    assert "retry guidance" in failure_description

    attached_description = registered["moveit_verify_attached_object"].description
    assert "moved with the gripper" in attached_description
    assert "does not execute" in attached_description


@pytest.mark.asyncio
async def test_canonical_planning_tool_schemas_use_agent_facing_names():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name: tool for tool in await mcp.list_tools()}
    free_schema = registered["moveit_plan_free_motion"].inputSchema
    cartesian_schema = registered["moveit_plan_cartesian_motion"].inputSchema
    pick_schema = registered["moveit_plan_pick"].inputSchema
    place_schema = registered["moveit_plan_place"].inputSchema

    assert set(free_schema["required"]) == {"target_pose"}
    assert "target_pose" in free_schema["properties"]
    assert "position" not in free_schema["properties"]
    assert "allow_existing_name" not in free_schema["properties"]
    assert free_schema["properties"]["robot_name"]["const"] == "UR10"
    assert free_schema["properties"]["robot_name"]["default"] == "UR10"

    assert set(cartesian_schema["required"]) == {"waypoints"}
    assert "waypoints" in cartesian_schema["properties"]
    assert "positions" not in cartesian_schema["properties"]
    assert "allow_existing_name" not in cartesian_schema["properties"]
    assert cartesian_schema["properties"]["robot_name"]["const"] == "UR10"
    assert cartesian_schema["properties"]["robot_name"]["default"] == "UR10"

    assert set(pick_schema["required"]) == {"object_name"}
    assert "object_name" in pick_schema["properties"]
    assert "grasp_face" in pick_schema["properties"]
    assert set(pick_schema["properties"]["planning_strategy"]["enum"]) == {"auto", "cartesian", "sampled_approach"}
    assert pick_schema["properties"]["planning_strategy"]["default"] == "auto"
    assert "waypoints" not in pick_schema["properties"]
    assert "allow_existing_name" not in pick_schema["properties"]
    assert pick_schema["properties"]["robot_name"]["const"] == "UR10"
    assert pick_schema["properties"]["robot_name"]["default"] == "UR10"

    assert set(place_schema["required"]) == {"object_name"}
    assert "target_pose" in place_schema["properties"]
    assert "target_position" in place_schema["properties"]
    assert "orientation_mode" in place_schema["properties"]
    assert place_schema["properties"]["orientation_mode"]["default"] == "keep"
    assert "allow_existing_name" not in place_schema["properties"]
    assert place_schema["properties"]["robot_name"]["const"] == "UR10"
    assert place_schema["properties"]["robot_name"]["default"] == "UR10"

    failure_schema = registered["moveit_explain_motion_failure"].inputSchema
    assert {"failed_tool_name", "failed_tool_result"}.issubset(failure_schema["required"])
    assert "failed_tool_arguments" in failure_schema["properties"]

    attached_schema = registered["moveit_verify_attached_object"].inputSchema
    assert set(attached_schema["required"]) == {"object_name"}
    assert "object_name" in attached_schema["properties"]


@pytest.mark.asyncio
async def test_canonical_scene_object_tool_schema_uses_object_name():
    tools = build_tools(transport=FakeRosbridgeTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name: tool for tool in await mcp.list_tools()}
    schema = registered["moveit_get_object_context"].inputSchema

    assert set(schema["required"]) == {"object_name"}
    assert "object_name" in schema["properties"]
    assert schema["properties"]["robot_name"]["const"] == "UR10"


@pytest.mark.asyncio
async def test_canonical_compound_task_tool_uses_ur10_by_default():
    tools = build_tools(transport=FakeRosbridgeTransport())

    def plan_mtc_compound_task(**kwargs):
        assert kwargs["robot"] == "UR10"
        assert kwargs["requirements"] == {"goal": "hold", "object_name": "beam_001"}
        assert kwargs["preferences"] == {"grasp_face": "top"}
        assert kwargs["stage_intents"] is None
        return {
            "ok": True,
            "task_solution_id": "server_compound_001",
            "task_goal": "hold",
            "backend": "mtc",
            "stage_summaries": [
                {"name": "current state", "stage_type": "CurrentState", "status": "solved"},
                {"name": "hold", "stage_type": "MoveRelative", "status": "solved"},
            ],
            "scene_snapshot": {"id": "server_scene_001"},
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
                        "step": 1,
                        "handler": "observe_current_state",
                        "source_stage": "current state",
                        "object_name": "beam_001",
                        "scene_snapshot_id": "server_scene_001",
                        "required_proof": "current_state_observed",
                    }
                ],
            },
        }

    tools.client.plan_mtc_compound_task = plan_mtc_compound_task
    mcp = build_mcp(tools=tools)

    _, payload = await mcp.call_tool(
        "moveit_plan_compound_task",
        {
            "requirements": {"goal": "hold", "object_name": "beam_001"},
            "preferences": {"grasp_face": "top"},
            "backend": "mtc",
        },
    )

    assert payload["ok"] is True
    assert payload["robot"] == "UR10"
    assert payload["tool"] == "moveit_plan_compound_task"
    assert payload["raw"]["task_solution_id"] == "server_compound_001"
    assert payload["raw"]["execution_contract"][0]["handler"] == "observe_current_state"


def test_main_parses_rosbridge_cli_args(monkeypatch):
    captured = {}
    fake_tools = object()

    class DummyMcp:
        def run(self, *, transport: str) -> None:
            captured["transport"] = transport

    def fake_build_tools(*, host: str, port: int):
        captured["rosbridge_host"] = host
        captured["rosbridge_port"] = port
        return fake_tools

    def fake_build_mcp(*, tools, host: str, port: int):
        captured["tools"] = tools
        captured["http_host"] = host
        captured["http_port"] = port
        return DummyMcp()

    monkeypatch.setattr(
        sys,
        "argv",
        ["moveit_mcp", "--rosbridge-host", "vizor", "--rosbridge-port", "9091"],
    )
    monkeypatch.setattr(server, "build_tools", fake_build_tools)
    monkeypatch.setattr(server, "build_mcp", fake_build_mcp)

    server.main()

    assert captured == {
        "rosbridge_host": "vizor",
        "rosbridge_port": 9091,
        "tools": fake_tools,
        "http_host": "127.0.0.1",
        "http_port": 8000,
        "transport": "stdio",
    }


def test_main_parses_streamable_http_cli_args(monkeypatch):
    captured = {}
    fake_tools = object()

    class DummyMcp:
        def run(self, *, transport: str) -> None:
            captured["transport"] = transport

    def fake_build_tools(*, host: str, port: int):
        captured["rosbridge_host"] = host
        captured["rosbridge_port"] = port
        return fake_tools

    def fake_build_mcp(*, tools, host: str, port: int):
        captured["tools"] = tools
        captured["http_host"] = host
        captured["http_port"] = port
        return DummyMcp()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "moveit_mcp",
            "--rosbridge-host",
            "localhost",
            "--rosbridge-port",
            "9090",
            "--transport",
            "streamable-http",
            "--http-host",
            "127.0.0.1",
            "--http-port",
            "8765",
        ],
    )
    monkeypatch.setattr(server, "build_tools", fake_build_tools)
    monkeypatch.setattr(server, "build_mcp", fake_build_mcp)

    server.main()

    assert captured == {
        "rosbridge_host": "localhost",
        "rosbridge_port": 9090,
        "tools": fake_tools,
        "http_host": "127.0.0.1",
        "http_port": 8765,
        "transport": "streamable-http",
    }
