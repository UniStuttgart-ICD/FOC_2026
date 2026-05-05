import json

import pytest

from codex_auth import CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from openai_codex_agent_processor import OpenAICodexAgentProcessor
from voice_runtime.agent_turn import AgentTurnInput


class Store:
    def get_credentials(self):
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class ScriptedBackend:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def create_response(self, credentials, *, model, instructions, input_items, tools):
        self.requests.append(
            {
                "model": model,
                "instructions": instructions,
                "input_items": list(input_items),
                "tools": list(tools),
            }
        )
        return self.results.pop(0)

    async def close(self):
        pass


class BehaviorBridge:
    def __init__(self):
        self.calls = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    def function_tools(self):
        return [
            {
                "type": "function",
                "name": "moveit_get_current_pose",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_free_motion",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_execute_plan",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_and_execute_free_motion",
                "parameters": {"type": "object"},
                "strict": None,
            },
        ]

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if name == "moveit_get_current_pose":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "raw": {
                            "pose": {
                                "position": {"x": 0.1, "y": 0.2, "z": 0.3},
                                "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
                            }
                        },
                    }
                }
            )
        if name == "moveit_plan_free_motion":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": "plan-1"},
                    }
                }
            )
        if name == "moveit_execute_plan":
            return json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
        if name == "moveit_plan_and_execute_free_motion":
            return json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
        return json.dumps({"structured_content": {"ok": True}})


async def run_processor(processor, text):
    turn = AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])
    return [chunk async for chunk in processor.run_turn(turn)]


def tool_call(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return CodexToolCall(
        call_id=call_id,
        item_id=item_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def output_item(name, call_id="call-1", item_id="item-1", arguments=None):
    arguments = arguments or {"robot_name": "UR10"}
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


@pytest.mark.asyncio
async def test_robot_action_preflight_gets_current_pose_before_codex_request():
    backend = ScriptedBackend([CodexResponseResult(text="I can wave from the current pose.")])
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "wave to me")

    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert "robot: UR10" in backend.requests[0]["instructions"]
    assert "x=0.100" in backend.requests[0]["instructions"]
    assert chunks == ["I can wave from the current pose."]


@pytest.mark.asyncio
async def test_non_robot_action_still_gets_current_pose_in_instructions():
    backend = ScriptedBackend([CodexResponseResult(text="I can help with robot commands.")])
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "what can you do?")

    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert "robot: UR10" in backend.requests[0]["instructions"]
    assert chunks == ["I can help with robot commands."]


@pytest.mark.asyncio
async def test_relative_movement_behavior_observes_before_answering():
    pose = tool_call("moveit_get_current_pose")
    backend = ScriptedBackend(
        [
            CodexResponseResult(tool_calls=[pose], output_items=[output_item("moveit_get_current_pose")]),
            CodexResponseResult(text="I checked the robot and can plan the relative move."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls[0] == ("moveit_get_current_pose", {"robot_name": "UR10"})
    assert chunks == ["I checked the robot and can plan the relative move."]


@pytest.mark.asyncio
async def test_repairs_missing_relative_target_pose_from_current_pose_context():
    tool = tool_call(
        "moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10},
    )
    backend = ScriptedBackend(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls[1] == (
        "moveit_plan_and_execute_free_motion",
        {
            "robot_name": "UR10",
            "plan_name": "move_up_50mm",
            "timeout_s": 10,
            "target_pose": {
                "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
            },
        },
    )
    assert chunks == ["Moved up 50 mm."]


@pytest.mark.asyncio
async def test_repairs_missing_cartesian_waypoints_from_current_pose_context():
    tool = tool_call(
        "moveit_plan_and_execute_cartesian_motion",
        arguments={"robot_name": "UR10", "plan_name": "move_up_cartesian_50mm", "timeout_s": 10},
    )
    backend = ScriptedBackend(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[output_item("moveit_plan_and_execute_cartesian_motion", arguments=tool.arguments)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls[1] == (
        "moveit_plan_and_execute_cartesian_motion",
        {
            "robot_name": "UR10",
            "plan_name": "move_up_cartesian_50mm",
            "timeout_s": 10,
            "waypoints": [
                {
                    "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                    "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
                }
            ],
        },
    )
    assert chunks == ["Moved up 50 mm."]


@pytest.mark.asyncio
async def test_repairs_back_up_as_negative_x_not_positive_z():
    tool = tool_call(
        "moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "plan_name": "back_up_50mm", "timeout_s": 10},
    )
    backend = ScriptedBackend(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)],
            ),
            CodexResponseResult(text="Moved back 50 mm."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    await run_processor(processor, "back up a bit")

    assert bridge.calls[1][1]["target_pose"] == {
        "position": {"x": 0.05, "y": 0.2, "z": 0.3},
        "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
    }


@pytest.mark.asyncio
async def test_plan_tool_is_auto_executed_once_plan_is_executable():
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    plan = tool_call("moveit_plan_free_motion", arguments=plan_args)
    backend = ScriptedBackend(
        [
            CodexResponseResult(
                tool_calls=[plan],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )
    bridge = BehaviorBridge()
    processor = OpenAICodexAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        model="gpt-5.4-mini",
        credential_store=Store(),
        backend_client=backend,
        tool_bridge=bridge,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "plan-1"}),
    ]
    assert chunks == ["Moved up 50 mm."]
