import json
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage

from agent_control.langchain_agent_processor import LangChainAgentProcessor
from robot_control.job_board import RobotJobBoard
from voice_runtime.agent_turn import AgentTurnInput


class ScriptedChatModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.requests: list[list[BaseMessage]] = []
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        clone = BoundScriptedChatModel(self.responses, self.requests, list(tools))
        self.bound_tools = clone.bound_tools
        return clone


class BoundScriptedChatModel:
    def __init__(
        self,
        responses: list[AIMessage],
        requests: list[list[BaseMessage]],
        bound_tools: list[dict[str, Any]],
    ):
        self.responses = responses
        self.requests = requests
        self.bound_tools = bound_tools

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.requests.append(list(messages))
        return self.responses.pop(0)


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
                "name": "moveit_plan_cartesian_motion",
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
            {
                "type": "function",
                "name": "moveit_plan_and_execute_cartesian_motion",
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
        if name == "moveit_plan_and_execute_cartesian_motion":
            return json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
        return json.dumps({"structured_content": {"ok": True}})


class NoopRobotJobWorker:
    async def start(self):
        pass

    async def stop(self):
        pass


async def run_processor(processor, text):
    turn = AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])
    try:
        return [chunk async for chunk in processor.run_turn(turn)]
    finally:
        await processor.disconnect()


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def tool_call(name: str, call_id: str = "call-1", arguments: dict[str, Any] | None = None) -> AIMessage:
    arguments = arguments or {"robot_name": "UR10"}
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": arguments, "id": call_id, "type": "tool_call"}],
    )


def make_processor(
    responses: list[AIMessage],
    bridge: BehaviorBridge | None = None,
    *,
    robot_job_board: RobotJobBoard | None = None,
):
    selected_bridge = bridge or BehaviorBridge()
    chat_model = ScriptedChatModel(responses)
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=chat_model,
        model_label="gpt-5.5",
        tool_bridge=selected_bridge,
        robot_job_board=robot_job_board,
        robot_job_worker=NoopRobotJobWorker(),
    )
    return processor, chat_model, selected_bridge


def system_content(chat_model: ScriptedChatModel, request_index: int = 0) -> str:
    system = chat_model.requests[request_index][0]
    assert isinstance(system, SystemMessage)
    return str(system.content)


@pytest.mark.asyncio
async def test_robot_action_preflight_gets_current_pose_before_model_request():
    processor, chat_model, bridge = make_processor([ai_text("I can wave from the current pose.")])

    chunks = await run_processor(processor, "wave to me")

    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert "robot: UR10" in system_content(chat_model)
    assert "x=0.100" in system_content(chat_model)
    assert chunks == ["I can wave from the current pose."]


@pytest.mark.asyncio
async def test_non_robot_action_still_gets_current_pose_in_instructions():
    processor, chat_model, bridge = make_processor([ai_text("I can help with robot commands.")])

    chunks = await run_processor(processor, "what can you do?")

    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert "robot: UR10" in system_content(chat_model)
    assert chunks == ["I can help with robot commands."]


@pytest.mark.asyncio
async def test_relative_movement_behavior_observes_before_answering():
    processor, _, bridge = make_processor(
        [
            tool_call("moveit_get_current_pose"),
            ai_text("I checked the robot and can plan the relative move."),
        ]
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert chunks == ["I checked the robot and can plan the relative move."]


@pytest.mark.asyncio
async def test_missing_motion_arguments_are_not_repaired_from_user_text():
    board = RobotJobBoard()
    incomplete_args = {"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10}
    processor, _, bridge = make_processor(
        [
            tool_call("moveit_plan_and_execute_free_motion", arguments=incomplete_args),
            ai_text("I need complete motion arguments."),
        ],
        robot_job_board=board,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    job = await board.claim_next()
    assert job is not None
    assert job.tool_name == "moveit_plan_and_execute_free_motion"
    assert job.arguments == incomplete_args
    assert chunks == ["I need complete motion arguments."]


@pytest.mark.asyncio
async def test_plan_tool_is_not_auto_executed_once_plan_is_executable():
    board = RobotJobBoard()
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    processor, _, bridge = make_processor(
        [tool_call("moveit_plan_free_motion", arguments=plan_args), ai_text("Moved up 50 mm.")],
        robot_job_board=board,
    )

    chunks = await run_processor(processor, "move up a bit")

    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    job = await board.claim_next()
    assert job is not None
    assert job.tool_name == "moveit_plan_free_motion"
    assert job.arguments == plan_args
    assert chunks == ["Moved up 50 mm."]
