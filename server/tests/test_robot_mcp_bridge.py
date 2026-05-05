import json

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from robot_mcp_bridge import RobotMCPBridge, RobotMCPError


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
            Tool(name="get_robot_status", description="Status", inputSchema={"type": "object"}),
            Tool(name="plan_free_motion", description="Plan free", inputSchema={"type": "object"}),
            Tool(name="robot_control", description="Too broad", inputSchema={"type": "object"}),
            Tool(name="dangerous_tool", description="Nope", inputSchema={"type": "object"}),
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"success": True},
        )


class FakeCanonicalServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_get_robot_status",
                description="Canonical status",
                inputSchema={"type": "object"},
            )
        ]


@pytest.mark.asyncio
async def test_lists_only_allowed_tools_as_codex_function_tools_with_canonical_names():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer())

    await bridge.connect()

    assert bridge.function_tools() == [
        {
            "type": "function",
            "name": "moveit_get_robot_status",
            "description": "Status",
            "parameters": {"type": "object"},
            "strict": None,
        },
        {
            "type": "function",
            "name": "moveit_plan_free_motion",
            "description": "Plan free",
            "parameters": {"type": "object"},
            "strict": None,
        },
    ]


@pytest.mark.asyncio
async def test_calls_allowed_tool_and_serializes_result():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_robot_status", {"robot_name": "UR10"})

    assert server.called == [("get_robot_status", {"robot_name": "UR10"})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"success": True}, "is_error": False}


@pytest.mark.asyncio
async def test_calls_canonical_listed_tool_by_advertised_name():
    server = FakeCanonicalServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    assert bridge.function_tools() == [
        {
            "type": "function",
            "name": "moveit_get_robot_status",
            "description": "Canonical status",
            "parameters": {"type": "object"},
            "strict": None,
        }
    ]

    output = await bridge.call_tool("moveit_get_robot_status", {"robot_name": "UR10"})

    assert server.called == [("moveit_get_robot_status", {"robot_name": "UR10"})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"success": True}, "is_error": False}


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

    output = await bridge.call_tool("moveit_get_robot_status", {"robot_name": "UR5"})

    assert json.loads(output) == {
        "error": "Only Vizor robot UR10 is allowed",
        "correction": 'Retry with robot_name="UR10".',
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
async def test_rejects_unsafe_motion_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool(
        "moveit_plan_free_motion",
        {
            "robot_name": "UR10",
            "position": {
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

    output = await bridge.call_tool("moveit_get_robot_status", {"robot_name": "UR10", "unexpected": True})

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
