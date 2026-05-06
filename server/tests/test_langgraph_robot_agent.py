import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from robot_control.context import RobotContextStore
from voice_runtime.agent_turn import AgentTurnInput


def test_langgraph_dependency_is_available() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    assert InMemorySaver is not None
    assert StateGraph is not None
    assert START != END


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.requests: list[list[BaseMessage]] = []
        self.bound_tools: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        clone = FakeBoundChatModel(self.responses, self.requests, list(tools))
        self.bound_tools = clone.bound_tools
        return clone


class FakeBoundChatModel:
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


class RaisingChatModel:
    def __init__(self, exc: BaseException):
        self.exc = exc

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        return RaisingBoundChatModel(self.exc)


class RaisingBoundChatModel:
    def __init__(self, exc: BaseException):
        self.exc = exc

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        raise self.exc


class CapturingLogger:
    def __init__(self) -> None:
        self.info_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.exception_messages: list[str] = []

    def info(self, message: str, *args: Any) -> None:
        self.info_messages.append(message.format(*args))

    def warning(self, message: str, *args: Any) -> None:
        self.warning_messages.append(message.format(*args))

    def exception(self, message: str, *args: Any) -> None:
        self.exception_messages.append(message.format(*args))


class FakeBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def function_tools(self) -> list[dict[str, Any]]:
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

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
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
                                "orientation": {
                                    "x": 0.0,
                                    "y": -0.7071,
                                    "z": -0.7071,
                                    "w": 0.0,
                                },
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
            return json.dumps(
                {"structured_content": {"ok": True, "verification": {"result": "pass"}}}
            )
        return json.dumps({"structured_content": {"ok": True}})


@dataclass(frozen=True)
class GraphFixture:
    graph: Any
    model: FakeChatModel
    bridge: FakeBridge


def make_graph(responses: list[AIMessage], *, bridge: FakeBridge | None = None) -> GraphFixture:
    from langgraph_robot_agent import LangGraphRobotAgent

    model = FakeChatModel(responses)
    selected_bridge = bridge or FakeBridge()
    graph = LangGraphRobotAgent(
        model=model,
        tool_bridge=selected_bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )
    return GraphFixture(graph=graph, model=model, bridge=selected_bridge)


def turn(text: str) -> AgentTurnInput:
    return AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])


def model_state() -> Any:
    return {
        "user_text": "hello",
        "messages": [HumanMessage(content="hello")],
        "tools": [],
        "tool_turns": 0,
        "observed_this_turn": False,
        "final_text": "",
        "error_text": None,
    }


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def ai_tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def last_tool_message(model: FakeChatModel) -> ToolMessage:
    message = model.requests[-1][-1]
    assert isinstance(message, ToolMessage)
    return message


def last_tool_content(model: FakeChatModel) -> str:
    content = last_tool_message(model).content
    assert isinstance(content, str)
    return content


@pytest.mark.asyncio
async def test_graph_observes_current_pose_before_simple_model_response() -> None:
    fixture = make_graph([ai_text("oauth-ok")])

    text = await fixture.graph.run_turn(turn("hello"))

    assert text == "oauth-ok"
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    assert "Last-known robot context" in str(first_request[0].content)
    assert "robot: UR10" in str(first_request[0].content)
    assert fixture.model.bound_tools == fixture.bridge.function_tools()


@pytest.mark.asyncio
async def test_model_request_logs_failure_before_reraising(monkeypatch: pytest.MonkeyPatch) -> None:
    from langgraph_robot_agent import LangGraphRobotAgent

    fake_logger = CapturingLogger()
    monkeypatch.setattr("langgraph_robot_agent.logger", fake_logger)
    graph = LangGraphRobotAgent(
        model=RaisingChatModel(RuntimeError("boom")),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    with pytest.raises(RuntimeError, match="boom"):
        await graph._call_model(model_state())

    assert any("Codex LangChain request start" in message for message in fake_logger.info_messages)
    assert any(
        "Codex LangChain request failed" in message and "boom" in message
        for message in fake_logger.exception_messages
    )


@pytest.mark.asyncio
async def test_model_request_logs_cancellation_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from langgraph_robot_agent import LangGraphRobotAgent

    fake_logger = CapturingLogger()
    monkeypatch.setattr("langgraph_robot_agent.logger", fake_logger)
    graph = LangGraphRobotAgent(
        model=RaisingChatModel(asyncio.CancelledError()),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    with pytest.raises(asyncio.CancelledError):
        await graph._call_model(model_state())

    assert any("Codex LangChain request start" in message for message in fake_logger.info_messages)
    assert any("Codex LangChain request cancelled" in message for message in fake_logger.warning_messages)


@pytest.mark.asyncio
async def test_graph_runs_full_langchain_tool_loop_and_returns_final_model_text() -> None:
    fixture = make_graph(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("The pose is ready."),
        ]
    )

    text = await fixture.graph.run_turn(turn("where is the pose?"))

    assert text == "The pose is ready."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert isinstance(fixture.model.requests[1][-1], ToolMessage)
    assert fixture.model.requests[1][-1].tool_call_id == "call-1"


@pytest.mark.asyncio
async def test_graph_ignores_legacy_status_as_active_observation_tool() -> None:
    class LegacyStatusBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_get_robot_status",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ]

    fixture = make_graph([ai_text("ok")], bridge=LegacyStatusBridge())

    text = await fixture.graph.run_turn(turn("hello"))

    assert text == "ok"
    assert fixture.bridge.calls == []


@pytest.mark.asyncio
async def test_graph_stops_after_max_tool_turns() -> None:
    fixture = make_graph(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}, call_id=f"call-{i}")
            for i in range(4)
        ]
    )

    text = await fixture.graph.run_turn(turn("pose"))

    assert text == "I could not confirm that the action completed."
    assert len(fixture.model.requests) == 4


@pytest.mark.asyncio
async def test_graph_does_not_auto_execute_executable_plan() -> None:
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    fixture = make_graph([ai_tool_call("moveit_plan_free_motion", plan_args), ai_text("Plan ready.")])

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]


@pytest.mark.asyncio
async def test_graph_does_not_repair_missing_motion_arguments_from_user_text() -> None:
    incomplete_args = {"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10}
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_and_execute_free_motion", incomplete_args),
            ai_text("I need complete motion arguments."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls[1] == ("moveit_plan_and_execute_free_motion", incomplete_args)


@pytest.mark.asyncio
async def test_graph_rejects_additional_tool_calls_after_first_without_executing_them() -> None:
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    fixture = make_graph(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "moveit_plan_free_motion",
                        "args": plan_args,
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "moveit_execute_plan",
                        "args": {"robot_name": "UR10", "plan_name": "plan-1"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            ai_text("Plan is ready; I did not execute twice."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    tool_messages = fixture.model.requests[1][-2:]
    assert [message.tool_call_id for message in tool_messages if isinstance(message, ToolMessage)] == [
        "call-1",
        "call-2",
    ]
    rejected = tool_messages[1]
    assert isinstance(rejected, ToolMessage)
    assert isinstance(rejected.content, str)
    rejected_output = json.loads(rejected.content)
    assert rejected_output["ok"] is False
    assert rejected_output["suggested_next_tool"] == "moveit_execute_plan"


@pytest.mark.asyncio
async def test_graph_sends_policy_failure_as_tool_message_when_motion_lacks_fresh_observation() -> None:
    class NoObservationBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_plan_free_motion",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ]

    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    fixture = make_graph(
        [ai_tool_call("moveit_plan_free_motion", plan_args), ai_text("I need a fresh pose.")],
        bridge=NoObservationBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a fresh pose."
    assert fixture.bridge.calls == []
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_blocks_blind_execute_plan_even_after_fresh_pose() -> None:
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "invented-plan"},
            ),
            ai_text("I need to plan before executing."),
        ]
    )

    text = await fixture.graph.run_turn(turn("execute the last plan"))

    assert text == "I need to plan before executing."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Cannot execute an unknown or stale plan.",
        "correction": "Plan first, then execute the returned plan_name.",
        "retryable": True,
        "suggested_next_tool": "moveit_plan_free_motion",
    }


@pytest.mark.asyncio
async def test_graph_converts_robot_mcp_error_to_structured_tool_output() -> None:
    from robot_control.mcp_bridge import RobotMCPError

    class ErrorBridge(FakeBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            raise RobotMCPError("robot server unavailable")

    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_plan_free_motion",
                {
                    "robot_name": "UR10",
                    "target_pose": {
                        "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            ),
            ai_text("The robot server is unavailable."),
        ],
        bridge=ErrorBridge(),
    )

    await fixture.graph.run_turn(turn("move up"))

    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "robot server unavailable",
        "correction": "Check the robot control server, then retry the robot action.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_preserves_structured_robot_tool_failure_as_tool_output() -> None:
    class FailureBridge(FakeBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            return json.dumps(
                {
                    "ok": False,
                    "error": "Planning failed",
                    "correction": "Check the target and plan again.",
                    "retryable": True,
                    "suggested_next_tool": "moveit_get_current_pose",
                }
            )

    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_plan_free_motion",
                {
                    "robot_name": "UR10",
                    "target_pose": {
                        "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            ),
            ai_text("I need a valid plan before executing."),
        ],
        bridge=FailureBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a valid plan before executing."
    output = json.loads(last_tool_content(fixture.model))
    assert output["ok"] is False
    assert output["retryable"] is True
    assert output["suggested_next_tool"] == "moveit_get_current_pose"


@pytest.mark.asyncio
async def test_graph_persists_context_between_turns_with_same_instance() -> None:
    fixture = make_graph([ai_text("first"), ai_text("second")])

    await fixture.graph.run_turn(turn("first"))
    await fixture.graph.run_turn(turn("second"))

    system = fixture.model.requests[-1][0]
    assert isinstance(system, SystemMessage)
    assert "robot: UR10" in str(system.content)
    assert fixture.graph.latest_state()["tool_turns"] == 0


@pytest.mark.asyncio
async def test_graph_blocks_attach_before_gripper_is_closed() -> None:
    class AttachBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_get_current_pose",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_attach_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_attach_object", {"robot_name": "UR10", "object_name": "cube"}
            ),
            ai_text("I need to close the gripper before attaching."),
        ],
        bridge=AttachBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "I need to close the gripper before attaching."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Cannot attach object before the gripper is known closed.",
        "correction": "Close the gripper or observe gripper state before attaching.",
        "retryable": True,
        "suggested_next_tool": "moveit_close_gripper",
    }


@pytest.mark.asyncio
async def test_graph_allows_attach_after_close_gripper_tool_result() -> None:
    class GripperBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_get_current_pose",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_close_gripper",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_attach_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            self.calls.append((name, arguments))
            return json.dumps({"structured_content": {"ok": True}})

    fixture = make_graph(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "moveit_close_gripper",
                        "args": {"robot_name": "UR10"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": "moveit_attach_object",
                        "args": {"robot_name": "UR10", "object_name": "cube"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            ),
            ai_text("Attached the cube."),
        ],
        bridge=GripperBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "Attached the cube."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_close_gripper", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
