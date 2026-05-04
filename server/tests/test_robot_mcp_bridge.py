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
            Tool(name="connect_robot", description="Connect", inputSchema={"type": "object"}),
            Tool(name="robot_control", description="Too broad", inputSchema={"type": "object"}),
            Tool(name="dangerous_tool", description="Nope", inputSchema={"type": "object"}),
        ]

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"success": True},
        )


@pytest.mark.asyncio
async def test_lists_only_allowed_tools_as_codex_function_tools():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer())

    await bridge.connect()

    assert bridge.function_tools() == [
        {
            "type": "function",
            "name": "get_robot_status",
            "description": "Status",
            "parameters": {"type": "object"},
            "strict": None,
        },
        {
            "type": "function",
            "name": "connect_robot",
            "description": "Connect",
            "parameters": {"type": "object"},
            "strict": None,
        },
    ]


@pytest.mark.asyncio
async def test_calls_allowed_tool_and_serializes_result():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool("get_robot_status", {})

    assert server.called == [("get_robot_status", {})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"success": True}, "is_error": False}


@pytest.mark.asyncio
async def test_does_not_advertise_broad_robot_control_tool():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer())

    await bridge.connect()

    names = {tool["name"] for tool in bridge.function_tools()}
    assert "robot_control" not in names


@pytest.mark.asyncio
async def test_rejects_non_loopback_connect_robot_ip():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    with pytest.raises(RobotMCPError, match="Only simulation robot IP"):
        await bridge.call_tool("connect_robot", {"robot_ip": "192.168.1.10"})

    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_unknown_tool_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    with pytest.raises(RobotMCPError, match="Tool is not allowed"):
        await bridge.call_tool("dangerous_tool", {})

    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_unsafe_motion_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    bridge._tools.append(Tool(name="move_to_position", description="Move", inputSchema={"type": "object"}))

    with pytest.raises(RobotMCPError, match="outside simulation workspace"):
        await bridge.call_tool("move_to_position", {"positions": [[99, 0, 0]]})

    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_unsafe_joint_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    bridge._tools.append(Tool(name="move_joints", description="Move", inputSchema={"type": "object"}))

    with pytest.raises(RobotMCPError, match="outside joint limit"):
        await bridge.call_tool("move_joints", {"positions": [[7, 0, 0, 0, 0, 0]]})

    assert server.called == []


@pytest.mark.asyncio
async def test_rejects_extra_arguments_before_mcp_call():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    with pytest.raises(RobotMCPError, match="Unexpected argument"):
        await bridge.call_tool("get_robot_status", {"unexpected": True})

    assert server.called == []


@pytest.mark.asyncio
async def test_disconnect_cleans_up_server():
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()
    await bridge.disconnect()

    assert server.cleaned is True
