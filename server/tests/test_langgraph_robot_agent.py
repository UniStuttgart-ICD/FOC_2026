import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from process_trace import MemoryTraceWriter, ProcessTracer
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
        self.bind_kwargs: list[dict[str, Any]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any):
        self.bind_kwargs.append(dict(kwargs))
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
                "name": "moveit_explain_motion_failure",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_verify_attached_object",
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


class TaskExecutionBridge(FakeBridge):
    def function_tools(self) -> list[dict[str, Any]]:
        return [
            *super().function_tools(),
            {
                "type": "function",
                "name": "moveit_execute_task_plan",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_execute_task_solution",
                "parameters": {"type": "object"},
                "strict": None,
            },
        ]


class FakeUserSensingBridge:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def read_context(self, *, max_age_s: float) -> str:
        self.calls.append(max_age_s)
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "gaze": {
                        "available": True,
                        "target": "beam_001",
                        "age_s": 0.1,
                        "stale": False,
                    },
                    "attention": {
                        "available": True,
                        "fresh": True,
                        "dominant_target": "beam_001",
                        "last_stable_target": "beam_001",
                        "ranked_targets": [
                            {
                                "target": "beam_001",
                                "confidence": "high",
                                "dwell_s": 3.4,
                                "last_seen_age_s": 0.1,
                            }
                        ],
                    },
                    "user": {
                        "available": True,
                        "position": {"x": 0.34, "y": -0.72, "z": 1.25},
                        "age_s": 0.2,
                        "stale": False,
                    },
                    "manual_target": {
                        "available": False,
                        "position": None,
                        "age_s": None,
                        "stale": True,
                    },
                }
            }
        )


class FakeVerifiedExecutionClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, float]] = []
        self.gripper_calls: list[tuple[str, str, float]] = []

    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        self.calls.append((robot_name, plan_name, timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "execute_plan",
                    "phase": "executed",
                    "status": "executed",
                    "feedback": {"plan_name": plan_name, "trajectory_points": 2},
                    "verification": {"result": "pass"},
                },
                "is_error": False,
            }
        )

    async def close_gripper(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        self.gripper_calls.append((robot_name, "close", timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "moveit_close_gripper",
                    "phase": "gripper",
                    "status": "gripper_closed",
                    "verification": {"result": "pass"},
                },
                "is_error": False,
            }
        )


class FakeFailingVerifiedExecutionClient(FakeVerifiedExecutionClient):
    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        self.calls.append((robot_name, plan_name, timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": False,
                    "robot": robot_name,
                    "tool": "execute_plan",
                    "phase": "pre_execute",
                    "status": "failed",
                    "feedback": {"plan_name": plan_name, "trajectory_points": 0},
                    "verification": {"result": "fail"},
                    "error": "Verified execution failed.",
                    "correction": "Inspect the real robot execution server.",
                },
                "is_error": True,
            }
        )


class FakeVerifiedExecutionClientWithoutPlanFeedback(FakeVerifiedExecutionClient):
    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        self.calls.append((robot_name, plan_name, timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "execute_plan",
                    "phase": "executed",
                    "status": "executed",
                    "verification": {"result": "pass"},
                },
                "is_error": False,
            }
        )


@dataclass(frozen=True)
class GraphFixture:
    graph: Any
    model: FakeChatModel
    bridge: FakeBridge
    user_sensing_bridge: FakeUserSensingBridge | None = None
    verified_execution_client: FakeVerifiedExecutionClient | None = None


def make_graph(
    responses: list[AIMessage],
    *,
    bridge: FakeBridge | None = None,
    robot_context: RobotContextStore | None = None,
    job_submitter: Any | None = None,
    user_sensing_bridge: FakeUserSensingBridge | None = None,
    verified_execution_client: FakeVerifiedExecutionClient | None = None,
    tracer: ProcessTracer | None = None,
) -> GraphFixture:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent
    from user_sensing.context import UserSensingContextStore

    model = FakeChatModel(responses)
    selected_bridge = bridge or FakeBridge()
    graph = LangGraphRobotAgent(
        model=model,
        tool_bridge=selected_bridge,
        robot_context=robot_context or RobotContextStore(),
        user_sensing_bridge=user_sensing_bridge,
        user_sensing_context=UserSensingContextStore(),
        thread_id="test-session",
        job_submitter=job_submitter,
        verified_execution_client=verified_execution_client,
        tracer=tracer,
    )
    return GraphFixture(
        graph=graph,
        model=model,
        bridge=selected_bridge,
        user_sensing_bridge=user_sensing_bridge,
        verified_execution_client=verified_execution_client,
    )


def turn(text: str) -> AgentTurnInput:
    return AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])


def model_state() -> Any:
    return {
        "user_text": "hello",
        "messages": [HumanMessage(content="hello")],
        "tools": [],
        "tool_turns": 0,
        "observed_this_turn": False,
        "allow_pending_plan_execution": True,
        "needs_action_tool": False,
        "action_tool_ran": False,
        "queued_robot_job": False,
        "missing_action_repairs": 0,
        "final_text": "",
        "error_text": None,
    }


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def ai_content_parts(parts: list[dict[str, Any]]) -> AIMessage:
    return AIMessage(content=cast(Any, parts))


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


def records_named(writer: MemoryTraceWriter, name: str) -> list[dict[str, Any]]:
    return [record for record in writer.records if record["name"] == name]


def approved_pick_task_context() -> RobotContextStore:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.update_from_tool_result("moveit_plan_pick_task", approved_pick_task_output())
    assert context.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    return context


def approved_pick_task_output() -> str:
    return json.dumps(
        {
            "structured_content": {
                "ok": True,
                "robot": "UR10",
                "feedback": {"can_execute": True, "execution_target": "task_solution"},
                "raw": {
                    "task_solution_id": "pick_task_dynamic_5_001",
                    "task_kind": "pick",
                    "backend": "emulated",
                    "object_name": "dynamic_5",
                    "robot_name": "UR10",
                    "created_from_tool": "moveit_plan_pick_task",
                    "scene_snapshot_id": "scene_20260515_001",
                    "waypoints": [
                        {
                            "position": {"x": 0.40, "y": 0.10, "z": 0.32},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                        {
                            "position": {"x": 0.46, "y": 0.10, "z": 0.32},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                        {
                            "position": {"x": 0.46, "y": 0.10, "z": 0.42},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    ],
                    "workflow_steps": [
                        {"kind": "motion", "name": "approach", "waypoint_index": 0},
                        {"kind": "motion", "name": "pre_grasp", "waypoint_index": 1},
                        {"kind": "gripper", "name": "close"},
                        {"kind": "scene", "name": "attach_object"},
                        {"kind": "motion", "name": "lift", "waypoint_index": 2},
                    ],
                    "approval": {
                        "required": True,
                        "target_kind": "task_solution",
                        "task_solution_id": "pick_task_dynamic_5_001",
                        "source_tool": "moveit_plan_pick_task",
                        "object_name": "dynamic_5",
                        "expected_movement": "approach grasp, close gripper, attach object, lift object",
                        "scene_snapshot_id": "scene_20260515_001",
                    },
                },
            }
        }
    )


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
    assert fixture.model.bound_tools[0] == {
        "type": "function",
        "function": {
            "name": "moveit_get_current_pose",
            "parameters": {"type": "object"},
        },
    }


@pytest.mark.asyncio
async def test_graph_loads_user_sensing_context_before_model_response() -> None:
    user_sensing = FakeUserSensingBridge()
    fixture = make_graph([ai_text("ready")], user_sensing_bridge=user_sensing)

    text = await fixture.graph.run_turn(turn("what am I looking at?"))

    assert text == "ready"
    assert user_sensing.calls == [2.0]
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    system_text = str(first_request[0].content)
    assert "User sensing context" in system_text
    assert "gaze target: beam_001" in system_text
    assert "user position: x=0.340, y=-0.720, z=1.250" in system_text
    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "vizor_get_sensor_context" not in tool_names


@pytest.mark.asyncio
async def test_graph_hides_sim_task_solution_tool_in_verified_execution_mode() -> None:
    fixture = make_graph(
        [ai_text("ready")],
        bridge=TaskExecutionBridge(),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("pick up dynamic_5"))

    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "moveit_execute_task_plan" in tool_names
    assert "moveit_execute_task_solution" not in tool_names
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    assert (
        "Use moveit_execute_task_plan for returned pick task_solution_id values; "
        "moveit_execute_task_solution is not available in real-robot mode."
    ) in str(first_request[0].content)


@pytest.mark.asyncio
async def test_graph_hides_verified_task_plan_tool_in_simulation_mode() -> None:
    fixture = make_graph([ai_text("ready")], bridge=TaskExecutionBridge())

    await fixture.graph.run_turn(turn("pick up dynamic_5"))

    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "moveit_execute_task_solution" in tool_names
    assert "moveit_execute_task_plan" not in tool_names
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    assert (
        "Use moveit_execute_task_solution for task_solution_id values; "
        "moveit_execute_task_plan is not available without Verified Real Robot Execution."
    ) in str(first_request[0].content)


@pytest.mark.asyncio
async def test_graph_traces_user_sensing_summary_and_payload() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    user_sensing = FakeUserSensingBridge()
    fixture = make_graph(
        [ai_text("ready")],
        user_sensing_bridge=user_sensing,
        tracer=tracer,
    )

    await fixture.graph.run_turn(turn("what am I looking at?"))

    context_update = records_named(writer, "user_sensing.context_update")[-1]
    assert context_update["attributes"]["attention.dominant_target"] == "beam_001"
    assert context_update["attributes"]["attention.fresh"] is True
    assert context_update["attributes"]["gaze.stale"] is False
    assert context_update["attributes"]["user.available"] is True
    tool_result = records_named(writer, "user_sensing.mcp.tool_result")[-1]
    assert "beam_001" in tool_result["attributes"]["tool.result"]


@pytest.mark.asyncio
async def test_graph_refreshes_user_sensing_before_each_model_call() -> None:
    user_sensing = FakeUserSensingBridge()
    action_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
        },
    }
    fixture = make_graph(
        [ai_tool_call("moveit_plan_free_motion", action_args), ai_text("planned")],
        user_sensing_bridge=user_sensing,
    )

    text = await fixture.graph.run_turn(turn("move there"))

    assert text == "planned"
    assert user_sensing.calls == [2.0, 2.0]


@pytest.mark.asyncio
async def test_repair_path_refreshes_user_sensing_before_retry_model_call() -> None:
    user_sensing = FakeUserSensingBridge()
    fixture = make_graph(
        [ai_text("I will move there now."), ai_text("")],
        user_sensing_bridge=user_sensing,
    )

    text = await fixture.graph.run_turn(turn("move there"))

    assert text == "Where would you like me to move?"
    assert user_sensing.calls == [2.0, 2.0]


@pytest.mark.asyncio
async def test_graph_turn_emits_graph_and_node_spans() -> None:
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    fixture = make_graph([ai_text("oauth-ok")], tracer=tracer)

    text = await fixture.graph.run_turn(turn("hello"))

    assert text == "oauth-ok"
    span_names = [record["name"] for record in writer.records if record["record_type"] == "span"]
    assert "agent.graph_turn" in span_names
    assert "agent.langgraph_node.observe_current_pose" in span_names
    assert "agent.langgraph_node.call_model" in span_names
    assert "agent.langgraph_node.final_response" in span_names
    graph_span = records_named(writer, "agent.graph_turn")[-1]
    assert graph_span["module"] == "agent_control"
    assert graph_span["attributes"] == {
        "thread_id": "test-session",
        "message_count": 1,
        "user_text": "hello",
    }
    node_span = records_named(writer, "agent.langgraph_node.call_model")[-1]
    assert node_span["attributes"]["node.name"] == "call_model"
    assert node_span["attributes"]["message_count"] >= 1


@pytest.mark.asyncio
async def test_call_model_emits_model_call_span() -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    writer = MemoryTraceWriter()
    graph = LangGraphRobotAgent(
        model=FakeChatModel([ai_text("response")]),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
        tracer=ProcessTracer(writer),
    )

    result = await graph._call_model(model_state())

    assert result["messages"] == [ai_text("response")]
    model_span = records_named(writer, "agent.model_call")[-1]
    assert model_span["module"] == "agent_control"
    assert model_span["status"] == "ok"
    assert model_span["attributes"] == {
        "tool_turns": 0,
        "message_count": 2,
        "tool_count": 6,
        "tool_call_count": 0,
        "tool_call_names": [],
        "text_length": len("response"),
    }


@pytest.mark.asyncio
async def test_policy_blocked_call_emits_task_policy_and_does_not_call_mcp() -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    writer = MemoryTraceWriter()
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model=FakeChatModel([]),
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
        tracer=ProcessTracer(writer),
    )
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }

    output = await graph._call_policy_checked_tool("moveit_plan_free_motion", plan_args)

    assert bridge.calls == []
    assert json.loads(output)["ok"] is False
    policy_span = records_named(writer, "robot.task_policy")[-1]
    assert policy_span["module"] == "robot_control"
    assert policy_span["status"] == "ok"
    assert policy_span["attributes"]["tool.name"] == "moveit_plan_free_motion"
    assert policy_span["attributes"]["decision_ok"] is False
    assert policy_span["attributes"]["suggested_next_tool"] == "moveit_get_current_pose"


@pytest.mark.asyncio
async def test_successful_tool_call_emits_context_update() -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    writer = MemoryTraceWriter()
    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model=FakeChatModel([]),
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
        tracer=ProcessTracer(writer),
    )

    output = await graph._call_policy_checked_tool(
        "moveit_get_current_pose", {"robot_name": "UR10"}
    )

    assert json.loads(output)["structured_content"]["ok"] is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    event = records_named(writer, "robot.context_update")[-1]
    assert event["record_type"] == "event"
    assert event["module"] == "robot_control"
    assert event["attributes"] == {"tool.name": "moveit_get_current_pose"}


@pytest.mark.asyncio
async def test_graph_speaks_text_content_part_without_reasoning_metadata() -> None:
    fixture = make_graph(
        [
            ai_content_parts(
                [
                    {"type": "reasoning", "summary": []},
                    {"type": "text", "text": "Moved up 100 mm."},
                ]
            )
        ]
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "Moved up 100 mm."


@pytest.mark.asyncio
async def test_model_request_logs_failure_before_reraising(monkeypatch: pytest.MonkeyPatch) -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    fake_logger = CapturingLogger()
    monkeypatch.setattr("agent_control.langgraph_robot_agent.logger", fake_logger)
    graph = LangGraphRobotAgent(
        model=RaisingChatModel(RuntimeError("boom")),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    with pytest.raises(RuntimeError, match="boom"):
        await graph._call_model(model_state())

    assert any("LangChain request start" in message for message in fake_logger.info_messages)
    assert any(
        "LangChain request failed" in message and "boom" in message
        for message in fake_logger.exception_messages
    )


@pytest.mark.asyncio
async def test_model_request_logs_cancellation_before_reraising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    fake_logger = CapturingLogger()
    monkeypatch.setattr("agent_control.langgraph_robot_agent.logger", fake_logger)
    graph = LangGraphRobotAgent(
        model=RaisingChatModel(asyncio.CancelledError()),
        tool_bridge=FakeBridge(),
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    with pytest.raises(asyncio.CancelledError):
        await graph._call_model(model_state())

    assert any("LangChain request start" in message for message in fake_logger.info_messages)
    assert any("LangChain request cancelled" in message for message in fake_logger.warning_messages)


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
            for i in range(7)
        ]
    )

    text = await fixture.graph.run_turn(turn("pose"))

    assert text == "I could not confirm that the action completed."
    assert len(fixture.model.requests) == 7


@pytest.mark.asyncio
async def test_graph_preserves_valid_history_after_max_tool_turns() -> None:
    fixture = make_graph(
        [
            *[
                ai_tool_call(
                    "moveit_get_current_pose",
                    {"robot_name": "UR10"},
                    call_id=f"call-{i}",
                )
                for i in range(7)
            ],
            ai_text("Recovered."),
        ]
    )

    await fixture.graph.run_turn(turn("pose"))
    text = await fixture.graph.run_turn(turn("hello again"))

    assert text == "Recovered."
    second_turn_messages = fixture.model.requests[-1][1:]
    dangling_tool_call_ids: list[str] = []
    for index, message in enumerate(second_turn_messages[:-1]):
        if not isinstance(message, AIMessage) or not message.tool_calls:
            continue
        next_message = second_turn_messages[index + 1]
        if not isinstance(next_message, ToolMessage):
            dangling_tool_call_ids.extend(str(call["id"]) for call in message.tool_calls)
            continue
        if next_message.tool_call_id != message.tool_calls[-1]["id"]:
            dangling_tool_call_ids.extend(str(call["id"]) for call in message.tool_calls)

    assert dangling_tool_call_ids == []


@pytest.mark.asyncio
async def test_graph_allows_observe_action_verify_action_sequence() -> None:
    wave_args = {
        "robot_name": "UR10",
        "waypoints": [
            {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
            {"position": {"x": 0.1, "y": 0.3, "z": 0.3}},
        ],
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}, call_id="observe-1"),
            ai_tool_call("moveit_plan_cartesian_motion", wave_args, call_id="wave-1"),
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}, call_id="observe-2"),
            ai_tool_call("moveit_plan_cartesian_motion", wave_args, call_id="wave-2"),
            ai_text("Waved."),
        ]
    )

    text = await fixture.graph.run_turn(turn("wave"))

    assert text == "Waved."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_cartesian_motion", wave_args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_cartesian_motion", wave_args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]


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
async def test_explicit_execute_request_queues_latest_pending_plan_with_worker() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    board = RobotJobBoard()
    robot_context = RobotContextStore()
    robot_context.remember_executable_plan(
        "plan-1",
        robot_name="UR10",
        source_tool="moveit_plan_free_motion",
        execute_via_mcp=True,
        after_success_tool="moveit_plan_pick",
        after_success_arguments={
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "planning_strategy": "cartesian",
        },
    )
    fixture = make_graph(
        [ai_text("model fallback")],
        robot_context=robot_context,
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("execute it"))

    assert text == "Execution queued."
    assert fixture.model.requests == []
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    job = await board.claim_next()
    assert job is not None
    assert job.tool_name == "moveit_execute_plan"
    assert job.arguments == {
        "robot_name": "UR10",
        "plan_name": "plan-1",
        "timeout_s": 30.0,
    }
    assert job.after_success_tool == "moveit_plan_pick"
    assert job.after_success_arguments == {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "planning_strategy": "cartesian",
    }
    assert job.execute_via_mcp is True


@pytest.mark.asyncio
async def test_graph_does_not_auto_execute_pending_plan_when_turn_disables_it() -> None:
    class RecordingSubmitter:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, Any], str | None]] = []

        async def submit_tool(
            self,
            tool_name: str,
            arguments: dict[str, Any],
            *,
            requested_by_turn_id: str | None = None,
            user_text: str | None = None,
        ) -> str:
            self.calls.append((tool_name, arguments, user_text))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "status": "queued",
                        "tool_name": tool_name,
                    },
                    "is_error": False,
                }
            )

    robot_context = RobotContextStore()
    robot_context.update_from_tool_result(
        "moveit_plan_free_motion",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": "UR10",
                    "feedback": {"can_execute": True},
                    "raw": {"plan_name": "plan-1"},
                }
            }
        ),
    )
    submitter = RecordingSubmitter()
    model = FakeChatModel([ai_text("I will explain the previous failure.")])
    bridge = FakeBridge()
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    agent = LangGraphRobotAgent(
        model=model,
        tool_bridge=bridge,
        robot_context=robot_context,
        job_submitter=cast(Any, submitter),
        thread_id="test-thread",
    )

    text = await agent.run_turn(
        AgentTurnInput(
            user_text="execute the plan",
            messages=[{"role": "user", "content": "execute the plan"}],
            allow_pending_plan_execution=False,
        )
    )

    assert text == "I will explain the previous failure."
    assert submitter.calls == []
    assert model.requests


@pytest.mark.asyncio
async def test_graph_blocks_direct_execute_when_turn_disables_execution() -> None:
    context = RobotContextStore()
    context.remember_executable_plan("plan-1", robot_name="UR10")
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
            ),
            ai_text("I need explicit confirmation first."),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(
        AgentTurnInput(
            user_text="recovery prompt says execute the returned plan",
            messages=[{"role": "user", "content": "recovery prompt says execute the returned plan"}],
            allow_pending_plan_execution=False,
        )
    )

    assert text == "I need explicit confirmation first."
    assert verified_client.calls == []
    output = json.loads(last_tool_content(fixture.model))
    assert output["error"] == "Execution requires an explicit user request."


@pytest.mark.asyncio
async def test_pending_plan_does_not_treat_look_as_ok_confirmation() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    board = RobotJobBoard()
    robot_context = RobotContextStore()
    robot_context.remember_executable_plan("plan-1", robot_name="UR10")
    fixture = make_graph(
        [ai_text("Plan details are available.")],
        robot_context=robot_context,
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("look at the plan"))

    assert text == "Plan details are available."
    assert fixture.model.requests
    assert await board.claim_next() is None


@pytest.mark.asyncio
async def test_pending_plan_does_not_auto_execute_task_stage_attempt() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    board = RobotJobBoard()
    robot_context = RobotContextStore()
    robot_context.remember_executable_plan(
        "pick_task_dynamic_5_001_approach_3d80d6ba_try2",
        robot_name="UR10",
        source_tool="moveit_plan_free_motion",
    )
    fixture = make_graph(
        [ai_text("I need the full task plan before executing.")],
        robot_context=robot_context,
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("execute it"))

    assert text == "I need the full task plan before executing."
    assert fixture.model.requests
    assert await board.claim_next() is None


@pytest.mark.asyncio
async def test_graph_queues_long_running_action_tool_when_submitter_is_present() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    board = RobotJobBoard()
    action_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "timeout_s": 10,
    }
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_plan_free_motion",
                action_args,
                call_id="action-1",
            ),
            ai_text("Queued the motion."),
        ],
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("move up a bit"))

    assert text == "Planning now. I will report when a plan is ready."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    job = await board.claim_next()
    assert job is not None
    assert job.tool_name == "moveit_plan_free_motion"
    assert job.arguments == action_args
    assert len(fixture.model.requests) == 1
    assert fixture.graph.latest_state()["action_tool_ran"] is True
    assert fixture.graph.latest_state()["queued_robot_job"] is True
    assert fixture.graph.latest_state()["observed_this_turn"] is False


@pytest.mark.asyncio
async def test_graph_stops_after_queuing_first_plan_job_with_worker() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    board = RobotJobBoard()
    first_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    second_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.45},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_free_motion", first_args, call_id="plan-1"),
            ai_tool_call("moveit_plan_free_motion", second_args, call_id="plan-2"),
            ai_text("Queued both plans."),
        ],
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("move up a bit"))

    assert text == "Planning now. I will report when a plan is ready."
    assert len(fixture.model.requests) == 1
    first_job = await board.claim_next()
    assert first_job is not None
    assert first_job.tool_name == "moveit_plan_free_motion"
    assert first_job.arguments == first_args
    assert await board.claim_next() is None


@pytest.mark.asyncio
async def test_graph_applies_task_policy_before_queued_action_tool() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

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

    board = RobotJobBoard()
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
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a fresh pose."
    assert fixture.bridge.calls == []
    assert await board.claim_next() is None
    assert board.events_since(0) == []
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_motion_request_retries_when_model_only_promises_action() -> None:
    cartesian_args = {
        "robot_name": "UR10",
        "waypoints": [
            {"position": {"x": 0.1, "y": 0.2, "z": 0.35}},
            {"position": {"x": 0.1, "y": 0.2, "z": 0.25}},
            {"position": {"x": 0.1, "y": 0.2, "z": 0.3}},
        ],
    }
    fixture = make_graph(
        [
            ai_text("I’ll get a fresh pose, then do a simple up-down gesture."),
            ai_tool_call("moveit_plan_cartesian_motion", cartesian_args),
            ai_text("Moved up and down."),
        ]
    )

    text = await fixture.graph.run_turn(turn("Have the robot move up and down"))

    assert text == "Moved up and down."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_cartesian_motion", cartesian_args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert len(fixture.model.requests) == 3
    corrective_request = fixture.model.requests[1]
    assert any(
        "did not call a MoveIt action tool" in str(message.content)
        for message in corrective_request
        if isinstance(message, HumanMessage)
    )
    assert fixture.model.bind_kwargs == [
        {"tool_choice": "auto"},
        {"tool_choice": "required"},
        {"tool_choice": "auto"},
    ]


@pytest.mark.asyncio
async def test_motion_request_does_not_synthesize_action_when_required_tool_retry_fails() -> None:
    fixture = make_graph(
        [
            ai_text("I’ll use a MoveIt action tool now."),
            ai_text(""),
        ]
    )

    text = await fixture.graph.run_turn(turn("move up a bit"))

    assert text == "Where would you like me to move?"
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert fixture.model.bind_kwargs == [
        {"tool_choice": "auto"},
        {"tool_choice": "required"},
    ]


@pytest.mark.asyncio
async def test_ambiguous_motion_request_clarifies_after_required_tool_retry_fails() -> None:
    fixture = make_graph(
        [
            ai_text("I’ll move there now."),
            ai_text(""),
        ]
    )

    text = await fixture.graph.run_turn(turn("move there"))

    assert text == "Where would you like me to move?"
    assert [name for name, _ in fixture.bridge.calls] == ["moveit_get_current_pose"]
    assert fixture.model.bind_kwargs == [
        {"tool_choice": "auto"},
        {"tool_choice": "required"},
    ]


@pytest.mark.asyncio
async def test_graph_does_not_repair_missing_motion_arguments_from_user_text() -> None:
    incomplete_args = {"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10}
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_free_motion", incomplete_args),
            ai_text("I need complete motion arguments."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls[1] == ("moveit_plan_free_motion", incomplete_args)


@pytest.mark.asyncio
async def test_graph_rejects_additional_tool_calls_after_first_without_executing_them() -> None:
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
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
        ],
        tracer=tracer,
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
    event = records_named(writer, "agent.extra_tool_call_rejected")[-1]
    assert event["record_type"] == "event"
    assert event["module"] == "agent_control"
    assert event["attributes"] == {
        "tool.name": "moveit_execute_plan",
        "tool_call_id": "call-2",
        "tool_call_index": 1,
    }


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
async def test_graph_blocks_execute_plan_without_explicit_user_request() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_executable_plan("plan-1", robot_name="UR10")
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "plan-1"},
            ),
            ai_text("I need explicit confirmation first."),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("move up a bit"))

    assert text == "I need explicit confirmation first."
    assert verified_client.calls == []
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Execution requires an explicit user request.",
        "correction": "Ask the user to explicitly confirm execution, then retry moveit_execute_plan.",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_graph_blocks_execute_task_solution_without_recorded_approval() -> None:
    class TaskSolutionBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_solution",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_solution",
                {
                    "robot_name": "UR10",
                    "task_solution_id": "pick_task_dynamic_5_001",
                    "timeout_s": 10.0,
                },
            ),
            ai_text("I need explicit task approval first."),
        ],
        bridge=TaskSolutionBridge(),
    )

    text = await fixture.graph.run_turn(turn("execute the pick task"))

    assert text == "I need explicit task approval first."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Task solution execution requires explicit approval",
        "correction": "Ask for explicit approval for the returned task_solution_id before executing.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_records_explicit_task_solution_approval_before_execution() -> None:
    class TaskSolutionBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_solution",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id="pick_task_dynamic_5_001",
        task_kind="pick",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="pick dynamic_5",
        scene_snapshot_id="scene_20260515_001",
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 10.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_solution", execute_args),
            ai_text("Executed the task solution."),
        ],
        bridge=TaskSolutionBridge(),
        robot_context=context,
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text == "Executed the task solution."
    assert ("moveit_execute_task_solution", execute_args) in fixture.bridge.calls
    approval = context.pending_task_solution_approval
    assert approval is not None
    assert approval.approval_turn_id is not None
    assert approval.approved_at == 100.0
    output = json.loads(last_tool_content(fixture.model))
    assert output == {"structured_content": {"ok": True}}


@pytest.mark.asyncio
async def test_graph_blocks_emulated_task_solution_in_verified_execution_mode() -> None:
    class TaskSolutionBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_solution",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id="pick_task_dynamic_5_001",
        task_kind="pick",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="pick dynamic_5",
        scene_snapshot_id="scene_20260515_001",
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 10.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_solution", execute_args),
            ai_text("I cannot verify physical task execution for that task solution."),
        ],
        bridge=TaskSolutionBridge(),
        robot_context=context,
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text == "I cannot verify physical task execution for that task solution."
    assert ("moveit_execute_task_solution", execute_args) not in fixture.bridge.calls
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Wrong task execution tool for real-robot mode",
        "correction": "Use moveit_execute_task_plan with the same task_solution_id.",
        "retryable": True,
        "suggested_next_tool": "moveit_execute_task_plan",
    }


@pytest.mark.asyncio
async def test_graph_executes_approved_pick_task_plan_through_verified_execution() -> None:
    class PickTaskPlanBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
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
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                plan_name = str(arguments["plan_name"])
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name == "moveit_verify_attached_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified pick task executed."),
        ],
        bridge=PickTaskPlanBridge(),
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert text == "Verified pick task executed."
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert [call[0] for call in verified_client.calls] == ["UR10", "UR10", "UR10"]
    assert [call[2] for call in verified_client.calls] == [9.0, 9.0, 9.0]
    assert verified_client.gripper_calls == [("UR10", "close", 9.0)]
    assert verified_plan_names[0].startswith("pick_task_dynamic_5_001_approach_")
    assert verified_plan_names[0].endswith("_try1")
    assert verified_plan_names[1].startswith("pick_task_dynamic_5_001_pre_grasp_")
    assert verified_plan_names[1].endswith("_try1")
    assert verified_plan_names[2].startswith("pick_task_dynamic_5_001_lift_")
    assert verified_plan_names[2].endswith("_try1")
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        (
            "moveit_plan_free_motion",
            {
                "robot_name": "UR10",
                "plan_name": verified_plan_names[0],
                "target_pose": {
                    "position": {"x": 0.40, "y": 0.10, "z": 0.32},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
                "timeout_s": 9.0,
            },
        ),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        (
            "moveit_plan_cartesian_motion",
            {
                "robot_name": "UR10",
                "plan_name": verified_plan_names[1],
                "waypoints": [
                    {
                        "position": {"x": 0.46, "y": 0.10, "z": 0.32},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    }
                ],
                "timeout_s": 9.0,
            },
        ),
        (
            "moveit_attach_object",
            {
                "robot_name": "UR10",
                "object_name": "dynamic_5",
                "verified_gripper_closed": True,
            },
        ),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        (
            "moveit_plan_cartesian_motion",
            {
                "robot_name": "UR10",
                "plan_name": verified_plan_names[2],
                "waypoints": [
                    {
                        "position": {"x": 0.46, "y": 0.10, "z": 0.42},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    }
                ],
                "timeout_s": 9.0,
            },
        ),
        (
            "moveit_verify_attached_object",
            {"robot_name": "UR10", "object_name": "dynamic_5", "timeout_s": 9.0},
        ),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(last_tool_content(fixture.model))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["task_solution_id"] == "pick_task_dynamic_5_001"
    assert output["structured_content"]["verified_plan_names"] == verified_plan_names


@pytest.mark.asyncio
async def test_graph_does_not_leave_task_stage_attempt_as_pending_plan() -> None:
    class PickTaskPlanBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
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
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                plan_name = str(arguments["plan_name"])
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name == "moveit_verify_attached_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    robot_context = approved_pick_task_context()
    verified_client = FakeVerifiedExecutionClientWithoutPlanFeedback()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified pick task executed."),
        ],
        bridge=PickTaskPlanBridge(),
        robot_context=robot_context,
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert robot_context.latest_pending_executable_plan(max_age_s=60.0) is None


@pytest.mark.asyncio
async def test_graph_retries_pick_task_pre_grasp_with_unique_free_motion_plan() -> None:
    class RetryingPickTaskPlanBridge(FakeBridge):
        def __init__(self) -> None:
            super().__init__()
            self.failed_pre_grasp_once = False

        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
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
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            plan_name = str(arguments.get("plan_name") or "")
            if (
                name == "moveit_plan_cartesian_motion"
                and "_pre_grasp_" in plan_name
                and not self.failed_pre_grasp_once
            ):
                self.failed_pre_grasp_once = True
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": False,
                            "robot": "UR10",
                            "feedback": {"status": "incomplete path", "can_execute": False},
                            "verification": {"result": "fail"},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name == "moveit_verify_attached_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified pick task executed."),
        ],
        bridge=RetryingPickTaskPlanBridge(),
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    planning_calls = [call for call in fixture.bridge.calls if call[0].startswith("moveit_plan_")]
    pre_grasp_calls = [
        call for call in planning_calls if "_pre_grasp_" in str(call[1].get("plan_name"))
    ]
    assert [call[0] for call in pre_grasp_calls] == [
        "moveit_plan_cartesian_motion",
        "moveit_plan_free_motion",
    ]
    assert str(pre_grasp_calls[0][1]["plan_name"]).endswith("_try1")
    assert str(pre_grasp_calls[1][1]["plan_name"]).endswith("_try2")
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert len(verified_plan_names) == 3
    assert verified_plan_names[1] == pre_grasp_calls[1][1]["plan_name"]


@pytest.mark.asyncio
async def test_graph_retries_task_plan_pose_observation_after_attach() -> None:
    class FlakyPoseAfterAttachBridge(FakeBridge):
        def __init__(self) -> None:
            super().__init__()
            self.fail_next_pose = False
            self.failed_pose_once = False

        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
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
                if self.fail_next_pose:
                    self.fail_next_pose = False
                    self.failed_pose_once = True
                    self.calls.append((name, arguments))
                    return json.dumps(
                        {
                            "structured_content": {
                                "ok": False,
                                "robot": "UR10",
                                "feedback": {"status": "current pose unavailable"},
                                "verification": {"result": "unknown"},
                            }
                        }
                    )
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                plan_name = str(arguments["plan_name"])
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name == "moveit_attach_object":
                self.fail_next_pose = True
            if name == "moveit_verify_attached_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    bridge = FlakyPoseAfterAttachBridge()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified pick task executed."),
        ],
        bridge=bridge,
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert bridge.failed_pose_once is True
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert len(verified_plan_names) == 3
    assert verified_plan_names[2].startswith("pick_task_dynamic_5_001_lift_")
    output = json.loads(last_tool_content(fixture.model))
    assert output["structured_content"]["ok"] is True


@pytest.mark.asyncio
async def test_graph_executes_returned_task_stage_plan_names() -> None:
    class RenamingPickTaskPlanBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_execute_task_plan",
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
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                plan_name = f"cached__{arguments['plan_name']}"
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": plan_name},
                        }
                    }
                )
            if name == "moveit_verify_attached_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified pick task executed."),
        ],
        bridge=RenamingPickTaskPlanBridge(),
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    verified_plan_names = [call[1] for call in verified_client.calls]
    assert [call[0] for call in verified_client.calls] == ["UR10", "UR10", "UR10"]
    assert [call[2] for call in verified_client.calls] == [9.0, 9.0, 9.0]
    assert verified_plan_names[0].startswith("cached__pick_task_dynamic_5_001_approach_")
    assert verified_plan_names[0].endswith("_try1")
    assert verified_plan_names[1].startswith("cached__pick_task_dynamic_5_001_pre_grasp_")
    assert verified_plan_names[1].endswith("_try1")
    assert verified_plan_names[2].startswith("cached__pick_task_dynamic_5_001_lift_")
    assert verified_plan_names[2].endswith("_try1")
    output = json.loads(last_tool_content(fixture.model))
    assert output["structured_content"]["verified_plan_names"] == verified_plan_names


@pytest.mark.asyncio
async def test_graph_rejects_task_plan_when_recent_solution_raw_is_missing() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id="pick_task_dynamic_5_001",
        task_kind="pick",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="pick_task_dynamic_5_001",
        source_tool="moveit_plan_pick_task",
        object_name="dynamic_5",
        expected_movement="approach grasp, close gripper, attach object, lift object",
        scene_snapshot_id="scene_20260515_001",
    )
    assert context.record_task_solution_approval(
        "pick_task_dynamic_5_001",
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("I need a fresh task plan."),
        ],
        robot_context=context,
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Task plan execution requires the recent raw task solution.",
        "correction": "Plan the pick task again, then retry moveit_execute_task_plan with that task_solution_id.",
        "retryable": True,
        "suggested_next_tool": "moveit_plan_pick_task",
    }


@pytest.mark.asyncio
async def test_graph_rejects_place_task_plan_execution_until_supported() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id="place_task_dynamic_5_001",
        task_kind="place",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="place_task_dynamic_5_001",
        source_tool="moveit_plan_place_task",
        object_name="dynamic_5",
        expected_movement="place dynamic_5",
        scene_snapshot_id="scene_20260515_001",
    )
    assert context.record_task_solution_approval(
        "place_task_dynamic_5_001",
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "place_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Place task execution is not supported yet."),
        ],
        robot_context=context,
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("yes, execute the place task"))

    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Task plan execution currently supports pick task solutions only.",
        "correction": "Use a pick task solution, or execute place workflows through supported verified plan steps.",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_graph_uses_verified_execution_client_for_explicit_execute_plan() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_executable_plan("plan-1", robot_name="UR10")
    verified_client = FakeVerifiedExecutionClient()
    writer = MemoryTraceWriter()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
            ),
            ai_text("Executed plan-1."),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
        tracer=ProcessTracer(writer),
    )

    text = await fixture.graph.run_turn(turn("please execute plan-1 now"))

    assert text == "Execution complete."
    assert verified_client.calls == [("UR10", "plan-1", 5.0)]
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert len(fixture.model.requests) == 1
    assert context.has_recent_executable_plan("plan-1", max_age_s=60.0) is False
    span = records_named(writer, "robot.verified_execution.execute_plan")[-1]
    assert span["module"] == "robot_control"
    assert span["status"] == "ok"
    assert span["duration_ms"] >= 0
    assert span["attributes"] == {
        "plan_name": "plan-1",
        "robot_name": "UR10",
        "timeout_s": 5.0,
        "execute.status": "executed",
        "execute.ok": True,
    }


@pytest.mark.asyncio
async def test_verified_preposition_execute_continues_with_after_success_pick_plan() -> None:
    class PickBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_plan_pick",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            if name == "moveit_get_current_pose":
                return await FakeBridge.call_tool(self, name, arguments)
            if name == "moveit_plan_pick":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": "cartesian-pick-plan"},
                        }
                    }
                )
            return await FakeBridge.call_tool(self, name, arguments)

    context = RobotContextStore(time_fn=lambda: 100.0)
    context.update_from_tool_result(
        "moveit_plan_free_motion",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": "UR10",
                    "feedback": {"can_execute": True},
                    "raw": {
                        "plan_name": "pick-preposition-plan",
                        "next_action": {
                            "after_success": {
                                "tool": "moveit_plan_pick",
                                "arguments": {
                                    "robot_name": "UR10",
                                    "object_name": "beam_001",
                                    "planning_strategy": "cartesian",
                                },
                            }
                        },
                    },
                }
            }
        ),
    )
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {
                    "robot_name": "UR10",
                    "plan_name": "pick-preposition-plan",
                    "timeout_s": 5.0,
                },
            ),
            ai_text("Pick plan ready."),
        ],
        bridge=PickBridge(),
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("please execute the preposition plan now"))

    assert text == "Execution complete."
    assert verified_client.calls == [("UR10", "pick-preposition-plan", 5.0)]
    assert len(fixture.model.requests) == 1
    assert ("moveit_plan_pick", {
        "robot_name": "UR10",
        "object_name": "beam_001",
        "planning_strategy": "cartesian",
    }) in fixture.bridge.calls
    assert context.has_recent_executable_plan("cartesian-pick-plan", max_age_s=60.0) is True


@pytest.mark.asyncio
async def test_graph_treats_proceed_as_explicit_execute_confirmation() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_executable_plan("plan-1", robot_name="UR10")
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 5.0},
            ),
            ai_text("Executed plan-1."),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes proceed"))

    assert text == "Execution complete."
    assert verified_client.calls == [("UR10", "plan-1", 5.0)]
    assert len(fixture.model.requests) == 1


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
