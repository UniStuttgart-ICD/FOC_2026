import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage

from agent_control.langchain_agent_processor import LangChainAgentProcessor
from process_trace import MemoryTraceWriter, ProcessTracer
from voice_runtime.agent_turn import AgentTurnInput


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


class FakeBridge:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.calls = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    def function_tools(self):
        return [
            {
                "type": "function",
                "name": "moveit_get_current_pose",
                "parameters": {"type": "object"},
                "strict": None,
            }
        ]

    async def call_tool(self, name, arguments) -> str:
        self.calls.append((name, arguments))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": "UR10",
                    "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
                }
            }
        )


class FakeUserSensingBridge:
    def __init__(self):
        self.connected = False
        self.disconnected = False
        self.calls: list[float] = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

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
                    "user": {"available": False, "position": None, "stale": True},
                    "manual_target": {"available": False, "position": None, "stale": True},
                }
            }
        )


@dataclass(frozen=True)
class TurnResult:
    chunks: list[str]
    processor: LangChainAgentProcessor


async def _run_turn(processor: LangChainAgentProcessor, text: str) -> TurnResult:
    turn = AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])
    try:
        chunks = [chunk async for chunk in processor.run_turn(turn)]
    finally:
        await processor.disconnect()
    return TurnResult(chunks=chunks, processor=processor)


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def ai_tool_call(name: str, args: dict[str, Any], call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def records_named(writer: MemoryTraceWriter, name: str) -> list[dict[str, Any]]:
    return [record for record in writer.records if record["name"] == name]


@pytest.mark.asyncio
async def test_generic_processor_runs_langgraph_without_oauth_credentials():
    model = FakeChatModel([ai_text("ready")])
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gpt-5.4-mini",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["ready"]
    assert bridge.connected is True
    assert bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert model.requests


@pytest.mark.asyncio
async def test_generic_processor_executes_model_tool_call():
    model = FakeChatModel(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}),
            ai_text("pose observed"),
        ]
    )
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gemini-2.5-flash",
        tool_bridge=bridge,
    )

    result = await _run_turn(processor, "where are you?")

    assert result.chunks == ["pose observed"]
    assert bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]


@pytest.mark.asyncio
async def test_processor_connects_and_disconnects_user_sensing_bridge() -> None:
    model = FakeChatModel([ai_text("ready")])
    robot_bridge = FakeBridge()
    user_sensing_bridge = FakeUserSensingBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="gpt-5.4-mini",
        tool_bridge=robot_bridge,
        user_sensing_bridge=user_sensing_bridge,
        user_sensing_max_age_s=3.5,
    )

    result = await _run_turn(processor, "hello")
    await processor.disconnect()

    assert result.chunks == ["ready"]
    assert user_sensing_bridge.connected is True
    assert user_sensing_bridge.calls == [3.5]
    assert user_sensing_bridge.disconnected is True


@pytest.mark.asyncio
async def test_processor_emits_backend_turn_and_passes_tracer_to_created_bridge_and_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_bridges: list[Any] = []
    created_graphs: list[Any] = []

    class CreatedBridge(FakeBridge):
        def __init__(self, url: str, *, tracer: ProcessTracer):
            super().__init__()
            self.url = url
            self.tracer = tracer
            created_bridges.append(self)

    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            user_sensing_bridge: Any | None = None,
            user_sensing_context: Any | None = None,
            user_sensing_max_age_s: float = 2.0,
            thread_id: str,
            job_submitter: Any | None = None,
            robot_job_blackboard_summary: Any | None = None,
            verified_execution_client: Any | None = None,
            tracer: ProcessTracer,
        ):
            self.model = model
            self.tool_bridge = tool_bridge
            self.robot_context = robot_context
            self.user_sensing_bridge = user_sensing_bridge
            self.user_sensing_context = user_sensing_context
            self.user_sensing_max_age_s = user_sensing_max_age_s
            self.thread_id = thread_id
            self.job_submitter = job_submitter
            self.robot_job_blackboard_summary = robot_job_blackboard_summary
            self.verified_execution_client = verified_execution_client
            self.tracer = tracer
            created_graphs.append(self)

        async def run_turn(self, turn: AgentTurnInput) -> str:
            return f"fake graph: {turn.user_text}"

    monkeypatch.setattr("agent_control.langchain_agent_processor.RobotMCPBridge", CreatedBridge)
    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    writer = MemoryTraceWriter()
    tracer = ProcessTracer(writer)
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="gpt-5.4-mini",
        tracer=tracer,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["fake graph: hello"]
    assert created_bridges[0].tracer is tracer
    assert created_graphs[0].tracer is tracer
    assert created_graphs[0].job_submitter is processor._robot_job_submitter
    assert created_graphs[0].robot_job_blackboard_summary() is None
    assert created_graphs[0].verified_execution_client is None
    assert created_graphs[0].user_sensing_bridge is None
    backend_span = records_named(writer, "agent.backend_turn")[-1]
    assert backend_span["record_type"] == "span"
    assert backend_span["module"] == "agent_control"
    assert backend_span["status"] == "ok"
    assert backend_span["attributes"] == {
        "model_label": "gpt-5.4-mini",
        "message_count": 1,
    }


@pytest.mark.asyncio
async def test_backend_turn_span_is_recorded_before_yielded_chunk_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            thread_id: str,
            job_submitter: Any | None = None,
            robot_job_blackboard_summary: Any | None = None,
            verified_execution_client: Any | None = None,
            tracer: ProcessTracer,
        ):
            pass

        async def run_turn(self, turn: AgentTurnInput) -> str:
            return f"fake graph: {turn.user_text}"

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    writer = MemoryTraceWriter()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="gpt-5.4-mini",
        tool_bridge=FakeBridge(),
        tracer=ProcessTracer(writer),
    )
    turn = AgentTurnInput(user_text="hello", messages=[{"role": "user", "content": "hello"}])
    chunks = processor.run_turn(turn)

    first_chunk = await chunks.__anext__()

    assert first_chunk == "fake graph: hello"
    backend_spans = records_named(writer, "agent.backend_turn")
    assert len(backend_spans) == 1
    assert backend_spans[0]["status"] == "ok"
    assert "error_type" not in backend_spans[0]["attributes"]

    await chunks.aclose()
    await processor.disconnect()

    assert records_named(writer, "agent.backend_turn") == backend_spans


@pytest.mark.asyncio
async def test_langchain_processor_starts_and_stops_robot_job_worker() -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.started = False
            self.stopped = False

        async def start(self) -> None:
            self.started = True

        async def stop(self) -> None:
            self.stopped = True

    worker = FakeWorker()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_worker=worker,
    )

    await processor.connect()
    await processor.disconnect()

    assert worker.started is True
    assert worker.stopped is True


@pytest.mark.asyncio
async def test_langchain_processor_passes_verified_client_to_owned_job_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[dict[str, Any]] = []

    class FakeWorker:
        def __init__(
            self,
            *,
            board: Any,
            tool_bridge: Any,
            verified_execution_client: Any | None = None,
        ) -> None:
            created.append(
                {
                    "board": board,
                    "tool_bridge": tool_bridge,
                    "verified_execution_client": verified_execution_client,
                }
            )

        async def start(self) -> None:
            pass

        async def stop(self) -> None:
            pass

    monkeypatch.setattr("agent_control.langchain_agent_processor.RobotJobWorker", FakeWorker)
    verified_client: Any = object()
    bridge = FakeBridge()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=bridge,
        verified_execution_client=verified_client,
    )

    await processor.connect()

    assert created == [
        {
            "board": processor._robot_job_board,
            "tool_bridge": bridge,
            "verified_execution_client": verified_client,
        }
    ]


@pytest.mark.asyncio
async def test_langchain_processor_notifications_report_terminal_job_events() -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )
    await board.claim_next()
    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "Action complete."
    assert job.job_id not in text


@pytest.mark.asyncio
async def test_langchain_processor_notifications_report_execute_completion_without_plan_name() -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1"},
            "turn-1",
        )
    )
    await board.claim_next()
    await board.complete(job.job_id, '{"structured_content": {"ok": true}}')

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "Execution complete."
    assert "plan-1" not in text


@pytest.mark.asyncio
async def test_langchain_processor_notifications_record_pending_plan_from_job_result() -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10"},
            "turn-1",
        )
    )
    await board.claim_next()
    await board.complete(
        job.job_id,
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

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "Plan ready."
    pending = processor._robot_context.pending_executable_plan("plan-1", max_age_s=60.0)
    assert pending is not None
    assert pending.robot_name == "UR10"
    assert pending.source_tool == "moveit_plan_free_motion"


@pytest.mark.asyncio
async def test_processor_records_terminal_job_results_before_next_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    class FakeGraphAgent:
        def __init__(
            self,
            *,
            robot_context: Any,
            **kwargs: Any,
        ):
            self.robot_context = robot_context

        async def run_turn(self, turn: AgentTurnInput) -> str:
            pending = self.robot_context.pending_executable_plan("plan-1", max_age_s=60.0)
            return "plan-context-ready" if pending is not None else "missing-plan-context"

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10"},
            "turn-1",
        )
    )
    await board.claim_next()
    await board.complete(
        job.job_id,
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

    result = await _run_turn(processor, "execute it")

    assert result.chunks == ["plan-context-ready"]


@pytest.mark.asyncio
async def test_processor_adds_job_blackboard_summary_to_model_context() -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    board = RobotJobBoard()
    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1"},
            "turn-1",
        )
    )
    await board.claim_next()
    await board.complete(
        job.job_id,
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "verification": {"result": "pass"},
                    "raw": {"plan_name": "plan-1"},
                }
            }
        ),
    )
    model = FakeChatModel([ai_text("ready")])
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )

    result = await _run_turn(processor, "status")

    assert result.chunks == ["ready"]
    system = model.requests[0][0]
    assert isinstance(system.content, str)
    assert "Robot Job Blackboard:" in system.content
    assert "moveit_execute_plan: completed" in system.content
    assert "plan_name=plan-1" in system.content
    assert "result recorded in Robot Context" in system.content


@pytest.mark.asyncio
async def test_langchain_processor_passes_verified_execution_client_to_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created_graphs: list[Any] = []
    verified_client: Any = object()

    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            thread_id: str,
            job_submitter: Any | None = None,
            robot_job_blackboard_summary: Any | None = None,
            verified_execution_client: Any | None = None,
            tracer: ProcessTracer,
        ):
            self.verified_execution_client = verified_execution_client
            created_graphs.append(self)

        async def run_turn(self, turn: AgentTurnInput) -> str:
            return "ready"

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        verified_execution_client=verified_client,
    )

    result = await _run_turn(processor, "hello")

    assert result.chunks == ["ready"]
    assert created_graphs[0].verified_execution_client is verified_client


@pytest.mark.asyncio
async def test_failed_job_notification_sends_planner_data_to_recovery_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    turns: list[AgentTurnInput] = []

    class FakeGraphAgent:
        def __init__(
            self,
            *,
            model: Any,
            tool_bridge: Any,
            robot_context: Any,
            thread_id: str,
            job_submitter: Any | None = None,
            robot_job_blackboard_summary: Any | None = None,
            verified_execution_client: Any | None = None,
            tracer: ProcessTracer,
        ):
            pass

        async def run_turn(self, turn: AgentTurnInput) -> str:
            turns.append(turn)
            return "I cannot touch z=0 because planning failed; I can try a little higher."

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    planner_result = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "feedback": {
                    "message": "Planning failed; execution was not attempted",
                    "correction": "Choose a nearby reachable pose.",
                },
                "plan": {"trajectory_points": 0},
            }
        }
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {
                "robot_name": "UR10",
                "target_pose": {
                    "position": {"x": 0.57, "y": 0.39, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "turn-1",
            user_text="touch z equals zero",
        )
    )
    await board.claim_next()
    await board.fail(
        job.job_id,
        "Planning failed; execution was not attempted Choose a nearby reachable pose.",
        result=planner_result,
    )

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "I cannot touch z=0 because planning failed; I can try a little higher."
    assert len(turns) == 1
    assert "Original user request: touch z equals zero" in turns[0].user_text
    assert "Tool: moveit_plan_free_motion" in turns[0].user_text
    assert '"z": 0.0' in turns[0].user_text
    assert "Planning failed; execution was not attempted" in turns[0].user_text


@pytest.mark.asyncio
async def test_failed_execute_job_notification_reports_blocker_without_recovery_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    class FakeGraphAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run_turn(self, turn: AgentTurnInput) -> str:
            raise AssertionError("failed execute notifications must not run recovery turns")

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    result = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "robot": "UR10",
                "tool": "execute_plan",
                "feedback": {
                    "message": "Execution could not be verified against fake controller joint state feedback",
                    "correction": "Check fake controller joint-state feedback.",
                },
                "verification": {"result": "fail"},
            },
            "is_error": False,
        }
    )
    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 10.0},
            "turn-1",
            user_text="proceed with the execution",
        )
    )
    await board.claim_next()
    await board.fail(
        job.job_id,
        (
            "Execution could not be verified against fake controller joint state feedback "
            "Check fake controller joint-state feedback."
        ),
        result=result,
    )

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == (
        "Execution did not verify: "
        "Execution could not be verified against fake controller joint state feedback "
        "Check fake controller joint-state feedback."
    )


@pytest.mark.asyncio
async def test_failed_job_recovery_turn_disables_pending_auto_execute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from robot_control.job_board import RobotJobBoard, SubmitRobotJob

    turns: list[AgentTurnInput] = []

    class FakeGraphAgent:
        def __init__(self, **kwargs: Any) -> None:
            pass

        async def run_turn(self, turn: AgentTurnInput) -> str:
            turns.append(turn)
            return "I could not complete that robot action."

    monkeypatch.setattr("agent_control.langchain_agent_processor.LangGraphRobotAgent", FakeGraphAgent)
    board = RobotJobBoard()
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=FakeChatModel([]),
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    stream = processor.notifications()
    planner_result = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "feedback": {
                    "message": "Planning failed; execution was not attempted",
                    "correction": "Choose a nearby reachable pose.",
                },
            }
        }
    )
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10"},
            "turn-1",
            user_text="execute the plan",
        )
    )
    await board.claim_next()
    await board.fail(
        job.job_id,
        "Planning failed; execution was not attempted Choose a nearby reachable pose.",
        result=planner_result,
    )

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "I could not complete that robot action."
    assert len(turns) == 1
    assert turns[0].allow_pending_plan_execution is False


@pytest.mark.asyncio
async def test_failed_job_notification_repairs_retry_claim_into_queued_action() -> None:
    from robot_control.job_board import RobotJobBoard, RobotJobStatus, SubmitRobotJob

    board = RobotJobBoard()
    retry_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.57, "y": 0.39, "z": 0.1},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    model = FakeChatModel(
        [
            ai_text("I can try a little higher."),
            ai_tool_call("moveit_plan_free_motion", retry_args),
            ai_text("I am retrying a little higher and will report the result."),
        ]
    )
    processor = LangChainAgentProcessor(
        "http://127.0.0.1:8765/mcp",
        chat_model=model,
        model_label="fake",
        tool_bridge=FakeBridge(),
        robot_job_board=board,
    )
    planner_result = json.dumps(
        {
            "structured_content": {
                "ok": False,
                "feedback": {
                    "message": "Planning failed; execution was not attempted",
                    "correction": "Choose a nearby reachable pose.",
                },
                "plan": {"trajectory_points": 0},
            }
        }
    )
    stream = processor.notifications()
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {
                "robot_name": "UR10",
                "target_pose": {
                    "position": {"x": 0.57, "y": 0.39, "z": 0.0},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            },
            "turn-1",
            user_text="touch z equals zero",
        )
    )
    await board.claim_next()
    await board.fail(
        job.job_id,
        "Planning failed; execution was not attempted Choose a nearby reachable pose.",
        result=planner_result,
    )

    text = await asyncio.wait_for(stream.__anext__(), timeout=1)

    assert text == "Planning now. I will report when a plan is ready."
    queued_retry = [
        candidate
        for event in board.events_since(0)
        if (candidate := board.get(event.job_id)) is not None
        and candidate.job_id != job.job_id
        and candidate.status is RobotJobStatus.QUEUED
    ]
    assert len(queued_retry) == 1
    assert queued_retry[0].tool_name == "moveit_plan_free_motion"
    assert queued_retry[0].arguments == retry_args
