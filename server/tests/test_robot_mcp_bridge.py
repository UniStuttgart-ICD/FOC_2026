import json

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from robot_control.call_validation import agent_tool_description
from robot_control.mcp_bridge import RobotMCPBridge, RobotMCPError


class FakeServer:
    def __init__(self):
        self.connected = False
        self.cleaned = False
        self.called = []

    async def connect(self):
        self.connected = True

    async def cleanup(self):
        self.cleaned = True

    async def list_tools(self):
        return [
            Tool(name="get_current_pose", description="Pose", inputSchema={"type": "object"}),
            Tool(name="plan_free_motion", description="Plan free", inputSchema={"type": "object"}),
            Tool(name="robot_control", description="Too broad", inputSchema={"type": "object"}),
            Tool(name="dangerous_tool", description="Nope", inputSchema={"type": "object"}),
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"ok": True},
        )


class FakeCanonicalServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_get_current_pose",
                description="Canonical pose",
                inputSchema={"type": "object"},
            )
        ]


class FakeLiveShapeServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(name="moveit_get_current_pose", description="Canonical pose", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_free_motion", description="Canonical plan", inputSchema={"type": "object"}),
            Tool(name="moveit_execute_plan", description="Canonical execute", inputSchema={"type": "object"}),
            Tool(name="get_current_pose", description="Legacy pose", inputSchema={"type": "object"}),
            Tool(name="plan_free_motion", description="Legacy plan", inputSchema={"type": "object"}),
            Tool(name="execute_plan", description="Legacy execute", inputSchema={"type": "object"}),
        ]


class FakeLegacyWorkflowServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="plan_and_execute_free_motion",
                description="Legacy workflow",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="plan_and_execute_cartesian_motion",
                description="Legacy Cartesian workflow",
                inputSchema={"type": "object"},
            ),
        ]


@pytest.mark.asyncio
async def test_lists_only_allowed_tools_as_codex_function_tools_with_canonical_names():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer())

    await bridge.connect()

    assert bridge.function_tools() == [
        {
            "type": "function",
            "name": "moveit_get_current_pose",
            "description": agent_tool_description("moveit_get_current_pose"),
            "parameters": {"type": "object"},
            "strict": None,
        },
        {
            "type": "function",
            "name": "moveit_plan_free_motion",
            "description": agent_tool_description("moveit_plan_free_motion"),
            "parameters": {"type": "object"},
            "strict": None,
        },
    ]


@pytest.mark.asyncio
async def test_deduplicates_legacy_aliases_and_prefers_canonical_mcp_tools():
    server = FakeLiveShapeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)

    await bridge.connect()

    assert [tool["name"] for tool in bridge.function_tools()] == [
        "moveit_get_current_pose",
        "moveit_plan_free_motion",
        "moveit_execute_plan",
    ]

    await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert server.called == [("moveit_get_current_pose", {"robot_name": "UR10"})]


@pytest.mark.asyncio
async def test_calls_allowed_tool_and_serializes_result():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert server.called == [("get_current_pose", {"robot_name": "UR10"})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"ok": True}, "is_error": False}


@pytest.mark.asyncio
async def test_calls_canonical_listed_tool_by_advertised_name():
    server = FakeCanonicalServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    assert bridge.function_tools() == [
        {
            "type": "function",
            "name": "moveit_get_current_pose",
            "description": agent_tool_description("moveit_get_current_pose"),
            "parameters": {"type": "object"},
            "strict": None,
        }
    ]

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert server.called == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"ok": True}, "is_error": False}


@pytest.mark.asyncio
async def test_maps_legacy_plan_and_execute_workflows_to_canonical_agent_tools():
    server = FakeLegacyWorkflowServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    assert [tool["name"] for tool in bridge.function_tools()] == [
        "moveit_plan_and_execute_free_motion",
        "moveit_plan_and_execute_cartesian_motion",
    ]

    free_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    await bridge.call_tool("moveit_plan_and_execute_free_motion", free_args)

    assert server.called == [("plan_and_execute_free_motion", free_args)]


@pytest.mark.asyncio
async def test_normalizes_cartesian_points_alias_before_mcp_call():
    server = FakeLegacyWorkflowServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()
    points = [
        {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.1, "y": 0.3, "z": 0.38},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    ]

    await bridge.call_tool(
        "moveit_plan_and_execute_cartesian_motion",
        {"robot_name": "UR10", "points": points, "timeout_s": 10.0},
    )

    assert server.called == [
        (
            "plan_and_execute_cartesian_motion",
            {"robot_name": "UR10", "waypoints": points, "timeout_s": 10.0},
        )
    ]


@pytest.mark.asyncio
async def test_strips_cartesian_alias_when_waypoints_are_already_canonical():
    server = FakeLegacyWorkflowServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()
    points = [
        {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    ]
    waypoints = [
        {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }
    ]

    await bridge.call_tool(
        "moveit_plan_and_execute_cartesian_motion",
        {"robot_name": "UR10", "points": points, "waypoints": waypoints, "timeout_s": 10.0},
    )

    assert server.called == [
        (
            "plan_and_execute_cartesian_motion",
            {"robot_name": "UR10", "waypoints": waypoints, "timeout_s": 10.0},
        )
    ]


@pytest.mark.asyncio
async def test_does_not_advertise_broad_robot_control_tool():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer())

    await bridge.connect()

    names = {tool["name"] for tool in bridge.function_tools()}
    assert "robot_control" not in names


@pytest.mark.asyncio
async def test_validation_failure_returns_compatible_error_json_without_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR5"})

    assert json.loads(output) == {
        "ok": False,
        "error": "Only Vizor robot UR10 is allowed",
        "correction": 'Retry with robot_name="UR10".',
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }
    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_unknown_tool_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("dangerous_tool", {})

    assert json.loads(output)["error"] == "Tool is not allowed: dangerous_tool"
    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_out_of_bounds_motion_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool(
        "moveit_plan_free_motion",
        {
            "robot_name": "UR10",
            "target_pose": {
                "position": {"x": 99, "y": 0, "z": 0},
                "orientation": {"x": 0, "y": 0, "z": 0, "w": 1},
            },
        },
    )

    assert json.loads(output)["error"] == "Target is outside simulation workspace"
    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_extra_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10", "unexpected": True})

    assert json.loads(output)["error"].startswith("Unexpected argument")
    assert server.called == []


@pytest.mark.asyncio
async def test_unadvertised_safe_tool_raises_bridge_error():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    with pytest.raises(RobotMCPError, match="Tool is not allowed"):
        await bridge.call_tool("moveit_open_gripper", {"robot_name": "UR10"})

    assert server.called == []


@pytest.mark.asyncio
async def test_disconnect_cleans_up_server():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()
    await bridge.disconnect()

    assert server.cleaned is True


@pytest.mark.asyncio
async def test_bridge_advertises_agent_friendly_descriptions():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeLiveShapeServer())

    await bridge.connect()

    tools = {tool["name"]: tool for tool in bridge.function_tools()}
    assert "current end-effector pose" in tools["moveit_get_current_pose"]["description"]
    assert "target pose" in tools["moveit_plan_free_motion"]["description"]
    assert "returned plan" in tools["moveit_execute_plan"]["description"]
