import json
from datetime import timedelta
from typing import Any, cast

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from process_trace import MemoryTraceWriter, ProcessTracer, TraceOptions
from robot_control.call_validation import WORKSPACE_ABS_LIMIT_M, agent_tool_description
from robot_control.mcp_bridge import RobotMCPBridge, RobotMCPError, StreamableHttpMCPServer


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


class FakeFailedResultServer(FakeServer):
    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        return CallToolResult(
            content=[TextContent(type="text", text="plan failed")],
            structuredContent={
                "ok": False,
                "error": "Target is outside simulation workspace",
                "retryable": False,
            },
            isError=True,
        )


class FakeExceptionServer(FakeServer):
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    async def call_tool(self, tool_name, arguments):
        self.called.append((tool_name, arguments))
        raise self.exc


class FakeClientSession:
    def __init__(self):
        self.called = []

    async def call_tool(
        self,
        tool_name,
        arguments,
        read_timeout_seconds=None,
    ):
        self.called.append((tool_name, arguments, read_timeout_seconds))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structuredContent={"ok": True},
        )


def _trace_record_names(writer):
    return [record["name"] for record in writer.records]


def _trace_record(writer, name):
    return next(record for record in writer.records if record["name"] == name)


class FakeCanonicalServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_get_current_pose",
                description="Canonical pose",
                inputSchema={"type": "object"},
            ),
            Tool(name="moveit_get_robot_state", description="State", inputSchema={"type": "object"}),
            Tool(name="moveit_list_scene_objects", description="Scene objects", inputSchema={"type": "object"}),
            Tool(name="moveit_get_object_context", description="Object context", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_pick", description="Plan pick", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_place", description="Plan place", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_pick_task", description="Plan pick task", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_place_task", description="Plan place task", inputSchema={"type": "object"}),
            Tool(
                name="moveit_plan_manipulation_task",
                description="Plan manipulation task",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="moveit_plan_compound_task",
                description="Plan compound task",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="moveit_execute_task_solution",
                description="Execute task solution",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "robot_name": {"type": "string"},
                        "task_solution_id": {"type": "string"},
                        "scene_snapshot_id": {"type": "string"},
                        "timeout_s": {"type": "number"},
                    },
                    "required": ["robot_name", "task_solution_id", "scene_snapshot_id"],
                },
            ),
            Tool(
                name="moveit_explain_motion_failure",
                description="Explain failure",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="moveit_verify_attached_object",
                description="Verify attached object",
                inputSchema={"type": "object"},
            ),
        ]


class FakeContractInternalServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_release_object",
                description="Release object",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="moveit_verify_released_object",
                description="Verify released object",
                inputSchema={"type": "object"},
            ),
            Tool(
                name="moveit_remove_scene_object",
                description="Remove scene object",
                inputSchema={"type": "object"},
            ),
        ]


class FakeHostileManipulationSchemaServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_plan_manipulation_task",
                description="Plan manipulation task",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "requirements": {
                            "type": "object",
                            "properties": {
                                "goal": {"type": "string", "enum": ["slide", "push"]},
                                "object_name": {"type": "string"},
                            },
                        }
                    },
                },
            ),
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


class FakeCartesianAliasServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(
                name="moveit_plan_cartesian_motion",
                description="Canonical Cartesian plan",
                inputSchema={"type": "object"},
            ),
        ]


class FakeOnlyLegacyTaskPlanningServer(FakeServer):
    async def list_tools(self):
        return [
            Tool(name="moveit_plan_pick_task", description="Plan pick task", inputSchema={"type": "object"}),
            Tool(name="moveit_plan_place_task", description="Plan place task", inputSchema={"type": "object"}),
        ]


TASK_SOLUTION_EXECUTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "task_solution_id": {"type": "string"},
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "task_solution_id"],
    "additionalProperties": False,
}
TASK_PLAN_EXECUTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "task_solution_id": {"type": "string"},
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "task_solution_id"],
    "additionalProperties": False,
}
UNIFIED_TASK_EXECUTION_PARAMETERS = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "task_solution_id": {"type": "string"},
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "task_solution_id"],
    "additionalProperties": False,
}
COORDINATE_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": -WORKSPACE_ABS_LIMIT_M, "maximum": WORKSPACE_ABS_LIMIT_M},
        "y": {"type": "number", "minimum": -WORKSPACE_ABS_LIMIT_M, "maximum": WORKSPACE_ABS_LIMIT_M},
        "z": {"type": "number", "minimum": -WORKSPACE_ABS_LIMIT_M, "maximum": WORKSPACE_ABS_LIMIT_M},
    },
    "required": ["x", "y", "z"],
    "additionalProperties": False,
}
QUATERNION_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number"},
        "y": {"type": "number"},
        "z": {"type": "number"},
        "w": {"type": "number"},
    },
    "required": ["x", "y", "z", "w"],
    "additionalProperties": False,
}
TARGET_POSE_SCHEMA = {
    "type": "object",
    "properties": {
        "position": COORDINATE_SCHEMA,
        "orientation": QUATERNION_SCHEMA,
    },
    "required": ["position"],
    "additionalProperties": False,
}
MANIPULATION_PREFERENCES_SCHEMA = {
    "type": "object",
    "properties": {
        "grasp_face": {
            "type": "string",
            "description": "Optional grasp face hint for task planning.",
        }
    },
    "additionalProperties": True,
}
LIFT_DISTANCE_DESCRIPTION = (
    "Post-grasp lift distance. Use requirements.lift_distance_m=0.0 for bare hold, "
    "support, or hold-in-place requests. Use the default 0.10 m or an explicit positive "
    "value only when the user asks to pick up, lift, raise, grab and lift, carry, or move "
    "after grasping."
)
MANIPULATION_TASK_PLANNING_PARAMETERS = {
    "type": "object",
    "properties": {
        "robot_name": {"type": "string"},
        "requirements": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string"},
                "target_pose": TARGET_POSE_SCHEMA,
                "target_position": COORDINATE_SCHEMA,
                "grasp_face": {
                    "type": "string",
                    "description": (
                        "Hard grasp-face requirement copied from explicit user wording, such as "
                        "'from the top'. Use preferences.grasp_face only for agent-chosen planner hints."
                    ),
                },
                "goal": {
                    "type": "string",
                    "enum": ["hold", "place", "release", "move_and_release", "pick_place"],
                },
                "lift_distance_m": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 0.2,
                    "description": LIFT_DISTANCE_DESCRIPTION,
                },
            },
            "required": ["goal"],
            "additionalProperties": False,
        },
        "preferences": MANIPULATION_PREFERENCES_SCHEMA,
        "timeout_s": {"type": "number"},
    },
    "required": ["robot_name", "requirements"],
    "additionalProperties": False,
}


@pytest.mark.asyncio
async def test_connect_emits_mcp_connect_and_list_tools_spans():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer(), tracer=tracer)

    await bridge.connect()

    names = _trace_record_names(writer)
    assert "robot.mcp.connect" in names
    assert "robot.mcp.list_tools" in names


@pytest.mark.asyncio
async def test_valid_call_tool_emits_validation_and_mcp_call_spans():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server, tracer=tracer)
    await bridge.connect()

    await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    validation = _trace_record(writer, "robot.call_validation")
    mcp_call = _trace_record(writer, "robot.mcp.call_tool")
    assert validation["attributes"] == {
        "tool.name": "moveit_get_current_pose",
        "tool.arguments": {"robot_name": "UR10"},
    }
    assert mcp_call["attributes"]["tool.name"] == "moveit_get_current_pose"
    assert mcp_call["attributes"]["mcp.tool.name"] == "get_current_pose"
    assert mcp_call["attributes"]["tool.arguments"] == {"robot_name": "UR10"}
    assert mcp_call["attributes"]["mcp.client.read_timeout_s"] == 30.0


@pytest.mark.asyncio
async def test_call_tool_trace_records_extended_mcp_client_timeout():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server, tracer=tracer)
    await bridge.connect()

    await bridge.call_tool(
        "moveit_get_current_pose",
        {"robot_name": "UR10", "timeout_s": 60.0},
    )

    mcp_call = _trace_record(writer, "robot.mcp.call_tool")
    assert mcp_call["attributes"]["mcp.client.read_timeout_s"] == 65.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("arguments", "expected_timeout"),
    [
        ({}, timedelta(seconds=30)),
        ({"timeout_s": 10.0}, timedelta(seconds=30)),
        ({"timeout_s": 60.0}, timedelta(seconds=65)),
    ],
)
async def test_streamable_http_mcp_server_passes_per_call_read_timeout(
    arguments,
    expected_timeout,
):
    session = FakeClientSession()
    server = StreamableHttpMCPServer("http://127.0.0.1:8765/mcp")
    server._session = cast(Any, session)

    await server.call_tool("moveit_plan_free_motion", arguments)

    assert session.called == [
        ("moveit_plan_free_motion", arguments, expected_timeout),
    ]


@pytest.mark.asyncio
async def test_failed_structured_tool_result_marks_mcp_call_span_failed_result():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    server = FakeFailedResultServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server, tracer=tracer)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert json.loads(output) == {
        "content": ["plan failed"],
        "structured_content": {
            "ok": False,
            "error": "Target is outside simulation workspace",
            "retryable": False,
        },
        "is_error": True,
    }
    mcp_call = _trace_record(writer, "robot.mcp.call_tool")
    assert mcp_call["status"] == "failed-result"
    assert mcp_call["attributes"]["mcp.transport.ok"] is True
    assert mcp_call["attributes"]["tool.result.ok"] is False
    assert mcp_call["attributes"]["tool.result.is_error"] is True
    assert mcp_call["attributes"]["tool.result.error"] == "Target is outside simulation workspace"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError("read timed out"), "timed out"),
        (RuntimeError("server exploded"), "failed"),
    ],
)
async def test_transport_exceptions_raise_robot_mcp_error(exc, expected):
    server = FakeExceptionServer(exc)
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    with pytest.raises(RobotMCPError) as error:
        await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert expected in str(error.value)
    assert error.value.__cause__ is exc
    assert server.called == [("get_current_pose", {"robot_name": "UR10"})]


@pytest.mark.asyncio
async def test_include_tool_payloads_false_omits_tool_arguments():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer, TraceOptions(include_tool_payloads=False))
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeServer(), tracer=tracer)
    await bridge.connect()

    await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    for record in writer.records:
        assert "tool.arguments" not in record["attributes"]


@pytest.mark.asyncio
async def test_validation_failure_emits_blocked_event_and_skips_mcp_call():
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    server = FakeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server, tracer=tracer)
    await bridge.connect()

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR5"})

    assert json.loads(output) == {
        "ok": False,
        "error": "Only Vizor robot UR10 is allowed",
        "correction": 'Retry with robot_name="UR10".',
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }
    validation = _trace_record(writer, "robot.call_validation")
    blocked = _trace_record(writer, "robot.call_validation.blocked")
    assert validation["status"] == "ok"
    assert blocked["record_type"] == "event"
    assert blocked["attributes"] == {
        "tool.name": "moveit_get_current_pose",
        "reason": "Only Vizor robot UR10 is allowed",
        "correction": 'Retry with robot_name="UR10".',
        "tool.arguments": {"robot_name": "UR5"},
    }
    assert "robot.mcp.call_tool" not in _trace_record_names(writer)
    assert server.called == []


@pytest.mark.asyncio
async def test_lists_only_allowed_tools_as_langchain_function_tools_with_canonical_names():
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
    ]


@pytest.mark.asyncio
async def test_deduplicates_legacy_aliases_and_prefers_canonical_mcp_tools():
    server = FakeLiveShapeServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)

    await bridge.connect()

    assert [tool["name"] for tool in bridge.function_tools()] == [
        "moveit_get_current_pose",
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

    tools = {tool["name"]: tool for tool in bridge.function_tools()}
    assert list(tools) == [
        "moveit_get_current_pose",
        "moveit_get_robot_state",
        "moveit_list_scene_objects",
        "moveit_get_object_context",
        "moveit_plan_manipulation_task",
        "moveit_execute_task",
        "moveit_explain_motion_failure",
    ]
    assert tools["moveit_plan_manipulation_task"] == {
        "type": "function",
        "name": "moveit_plan_manipulation_task",
        "description": agent_tool_description("moveit_plan_manipulation_task"),
        "parameters": MANIPULATION_TASK_PLANNING_PARAMETERS,
        "strict": None,
    }

    output = await bridge.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})

    assert server.called == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert json.loads(output) == {"content": ["ok"], "structured_content": {"ok": True}, "is_error": False}

    await bridge.call_tool("moveit_get_robot_state", {"robot_name": "UR10"})
    assert ("moveit_get_robot_state", {"robot_name": "UR10"}) in server.called

    await bridge.call_tool("moveit_get_object_context", {"robot_name": "UR10", "object_name": "beam_001"})
    assert ("moveit_get_object_context", {"robot_name": "UR10", "object_name": "beam_001"}) in server.called

    pick_args = {"robot_name": "UR10", "object_name": "beam_001", "planning_strategy": "cartesian"}
    await bridge.call_tool("moveit_plan_pick", pick_args)
    assert ("moveit_plan_pick", pick_args) in server.called

    place_args = {
        "robot_name": "UR10",
        "object_name": "beam_001",
        "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
        "orientation_mode": "horizontal",
    }
    await bridge.call_tool("moveit_plan_place", place_args)
    assert ("moveit_plan_place", place_args) in server.called

    compound_args = {
        "robot_name": "UR10",
        "requirements": {"goal": "hold", "object_name": "beam_001"},
        "backend": "mtc",
    }
    await bridge.call_tool("moveit_plan_compound_task", compound_args)
    assert ("moveit_plan_compound_task", compound_args) in server.called

    pick_task_args = {"robot_name": "UR10", "object_name": "beam_001"}
    await bridge.call_tool("moveit_plan_pick_task", pick_task_args)
    assert ("moveit_plan_pick_task", pick_task_args) in server.called

    place_task_args = {
        "robot_name": "UR10",
        "object_name": "beam_001",
        "target_position": {"x": 0.75, "y": 0.2, "z": 0.28},
        "orientation_mode": "keep",
    }
    await bridge.call_tool("moveit_plan_place_task", place_task_args)
    assert ("moveit_plan_place_task", place_task_args) in server.called

    execute_task_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_beam_001_001",
    }
    await bridge.call_tool("moveit_execute_task_solution", execute_task_args)
    assert ("moveit_execute_task_solution", execute_task_args) in server.called

    failure_args = {
        "robot_name": "UR10",
        "failed_tool_name": "moveit_plan_pick",
        "failed_tool_result": {"ok": False, "feedback": {"status": "incomplete path"}},
    }
    await bridge.call_tool("moveit_explain_motion_failure", failure_args)
    assert ("moveit_explain_motion_failure", failure_args) in server.called

    verify_args = {"robot_name": "UR10", "object_name": "beam_001"}
    await bridge.call_tool("moveit_verify_attached_object", verify_args)
    assert ("moveit_verify_attached_object", verify_args) in server.called


@pytest.mark.asyncio
async def test_contract_internal_tools_are_hidden_and_require_contract_call_path():
    server = FakeContractInternalServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    assert bridge.function_tools() == []
    assert bridge.contract_tool_names() == {
        "moveit_release_object",
        "moveit_verify_released_object",
        "moveit_remove_scene_object",
    }

    direct_output = json.loads(
        await bridge.call_tool(
            "moveit_verify_released_object",
            {"robot_name": "UR10", "object_name": "dynamic_5"},
        )
    )
    assert direct_output["ok"] is False
    assert direct_output["code"] == "contract_internal_tool"
    assert server.called == []

    release_args = {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "object_pose": {
            "position": {"x": 0.57, "y": 0.39, "z": 0.62},
            "orientation": {"x": 0.0, "y": -0.70710678, "z": -0.70710678, "w": 0.0},
        },
        "verified_gripper_open": True,
    }
    contract_output = json.loads(
        await bridge.call_contract_tool("moveit_release_object", release_args)
    )

    assert contract_output == {"content": ["ok"], "structured_content": {"ok": True}, "is_error": False}
    assert server.called == [("moveit_release_object", release_args)]


@pytest.mark.asyncio
async def test_task_solution_execution_schema_hides_upstream_scene_snapshot_id():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeCanonicalServer())
    await bridge.connect()

    tools = {tool["name"]: tool for tool in bridge.function_tools()}

    assert "moveit_execute_task" in tools
    assert "moveit_execute_task_solution" not in tools
    assert "moveit_execute_task_plan" not in tools
    assert "moveit_execute_plan" not in tools


@pytest.mark.asyncio
async def test_manipulation_task_schema_overrides_goal_enum_from_upstream_schema():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeHostileManipulationSchemaServer())
    await bridge.connect()

    tools = {tool["name"]: tool for tool in bridge.function_tools()}
    tool = tools["moveit_plan_manipulation_task"]
    parameters = tool["parameters"]
    requirements = parameters["properties"]["requirements"]
    requirement_properties = requirements["properties"]
    goal = requirement_properties["goal"]

    assert tool["strict"] is None
    assert parameters["additionalProperties"] is False
    assert parameters["required"] == ["robot_name", "requirements"]
    assert "backend" not in parameters["properties"]
    assert requirements["additionalProperties"] is False
    assert requirements["required"] == ["goal"]
    assert requirement_properties["target_pose"] == TARGET_POSE_SCHEMA
    assert "orientation" not in TARGET_POSE_SCHEMA["required"]
    assert requirement_properties["target_position"] == COORDINATE_SCHEMA
    assert requirement_properties["grasp_face"]["type"] == "string"
    assert "hard" in requirement_properties["grasp_face"]["description"].lower()
    assert goal["enum"] == ["hold", "place", "release", "move_and_release", "pick_place"]
    assert "slide" not in goal["enum"]
    assert requirement_properties["lift_distance_m"] == {
        "type": "number",
        "minimum": 0.0,
        "maximum": 0.2,
        "description": LIFT_DISTANCE_DESCRIPTION,
    }
    preferences = parameters["properties"]["preferences"]
    assert preferences["additionalProperties"] is True
    assert preferences["properties"]["grasp_face"]["type"] == "string"
    assert "grasp face hint" in preferences["properties"]["grasp_face"]["description"]


@pytest.mark.asyncio
async def test_bridge_advertises_synthetic_unified_task_execution_tool():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeCanonicalServer())
    await bridge.connect()

    tools = {tool["name"]: tool for tool in bridge.function_tools()}

    assert tools["moveit_execute_task"] == {
        "type": "function",
        "name": "moveit_execute_task",
        "description": agent_tool_description("moveit_execute_task"),
        "parameters": UNIFIED_TASK_EXECUTION_PARAMETERS,
        "strict": None,
    }


@pytest.mark.asyncio
async def test_bridge_returns_structured_error_for_direct_synthetic_task_execution():
    server = FakeCanonicalServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    output = await bridge.call_tool(
        "moveit_execute_task",
        {"robot_name": "UR10", "task_solution_id": "compound_task_dynamic_5_001"},
    )

    assert json.loads(output) == {
        "ok": False,
        "error": "moveit_execute_task is executed by Agent Control, not the MCP bridge",
        "correction": "Route this task_solution_id through Agent Control's unified task executor.",
        "retryable": False,
        "code": "agent_control_execution_required",
    }
    assert server.called == []


@pytest.mark.asyncio
async def test_model_visible_tools_hide_internal_pick_and_place_task_planners():
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=FakeCanonicalServer())
    await bridge.connect()

    names = {tool["name"] for tool in bridge.function_tools()}

    assert "moveit_plan_manipulation_task" in names
    assert "moveit_plan_compound_task" not in names
    assert "moveit_plan_pick" not in names
    assert "moveit_plan_place" not in names
    assert "moveit_plan_free_motion" not in names
    assert "moveit_plan_cartesian_motion" not in names
    assert "moveit_plan_pick_task" not in names
    assert "moveit_plan_place_task" not in names


@pytest.mark.asyncio
async def test_internal_pick_and_place_task_planners_do_not_expose_task_plan_execution():
    bridge = RobotMCPBridge(
        "http://127.0.0.1:8765/mcp",
        server=FakeOnlyLegacyTaskPlanningServer(),
    )
    await bridge.connect()

    assert bridge.function_tools() == []


@pytest.mark.asyncio
async def test_legacy_plan_and_execute_workflows_are_not_advertised():
    server = FakeLegacyWorkflowServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    assert bridge.function_tools() == []


@pytest.mark.asyncio
async def test_normalizes_cartesian_points_alias_before_mcp_call():
    server = FakeCartesianAliasServer()
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
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "points": points, "timeout_s": 10.0},
    )

    assert server.called == [
        (
            "moveit_plan_cartesian_motion",
            {"robot_name": "UR10", "waypoints": points, "timeout_s": 10.0},
        )
    ]


@pytest.mark.asyncio
async def test_adds_staged_moveit_backend_for_manipulation_mcp_call():
    server = FakeCanonicalServer()
    bridge = RobotMCPBridge("http://127.0.0.1:8765/mcp", server=server)
    await bridge.connect()

    await bridge.call_tool(
        "moveit_plan_manipulation_task",
        {
            "robot_name": "UR10",
            "requirements": {
                "goal": "hold",
                "object_name": "dynamic_2",
                "lift_distance_m": 0.1,
            },
            "preferences": {},
            "timeout_s": 10.0,
        },
    )

    assert server.called == [
        (
            "moveit_plan_manipulation_task",
            {
                "robot_name": "UR10",
                "requirements": {
                    "goal": "hold",
                    "object_name": "dynamic_2",
                    "lift_distance_m": 0.1,
                },
                "preferences": {},
                "timeout_s": 10.0,
                "backend": "staged_moveit",
            },
        )
    ]


@pytest.mark.asyncio
async def test_strips_cartesian_alias_when_waypoints_are_already_canonical():
    server = FakeCartesianAliasServer()
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
        "moveit_plan_cartesian_motion",
        {"robot_name": "UR10", "points": points, "waypoints": waypoints, "timeout_s": 10.0},
    )

    assert server.called == [
        (
            "moveit_plan_cartesian_motion",
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
    assert "moveit_plan_free_motion" not in tools
    assert "moveit_execute_plan" not in tools


def test_manipulation_task_description_explains_hold_lift_semantics():
    description = agent_tool_description("moveit_plan_manipulation_task")

    assert "post-grasp lift" in description
    assert "requirements.lift_distance_m=0.0" in description
    assert "bare hold" in description
    assert "support" in description
    assert "hold-in-place" in description
    assert "default 0.10 m" in description
    assert "pick up" in description
    assert "raise" in description
    assert "grab and lift" in description
    assert "carry" in description
