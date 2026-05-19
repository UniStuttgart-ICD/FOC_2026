import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from agent_control.status_replies import (
    EXECUTION_COMPLETE_REPLIES,
    PHYSICAL_EXECUTION_FAILED_REPLIES,
    PHYSICAL_STATUS_UNAVAILABLE_REPLIES,
    PLAN_READY_REPLIES,
)
from process_trace import MemoryTraceWriter, ProcessTracer
from robot_control.context import RobotContextStore
from voice_runtime.agent_turn import AgentTurnInput


def test_langgraph_dependency_is_available() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    assert InMemorySaver is not None
    assert StateGraph is not None
    assert START != END


def test_supported_task_solution_kinds_do_not_include_unadvertised_approach_workflow() -> None:
    from agent_control.langgraph_robot_agent import SUPPORTED_TASK_SOLUTION_KINDS

    assert "approach_hold_adjust_release" not in SUPPORTED_TASK_SOLUTION_KINDS


def test_task_plan_failure_result_text_uses_plain_language_for_closed_resource() -> None:
    from agent_control.langgraph_robot_agent import _task_plan_failure_result_text

    result = {
        "task_solution_id": "pick_place_task_dynamic_0_001",
        "failed_step": "approach_to_pre_grasp",
        "failed_stage": "observe_current_pose",
        "failed_tool_result": {
            "ok": False,
            "error": "Robot MCP tool moveit_get_current_pose failed: ClosedResourceError",
        },
        "completed_steps": [{"name": "connect_to_pre_grasp"}],
    }

    text = _task_plan_failure_result_text(result)

    assert text == (
        "I could not finish the task because the robot connection closed while I was "
        "checking the current pose. Completed before the failure: connect to pre-grasp. "
        "Please approve the next action before I retry or replan."
    )
    assert "pick_place_task_dynamic_0_001" not in text
    assert "approach_to_pre_grasp" not in text
    assert "observe_current_pose" not in text
    assert "moveit_get_current_pose" not in text
    assert "ClosedResourceError" not in text
    assert "MoveIt/tool failure" not in text


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

    def contract_tool_names(self) -> set[str]:
        return {
            str(tool["name"])
            for tool in self.function_tools()
            if isinstance(tool.get("name"), str)
        }

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
                "name": "moveit_execute_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
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


class TaskPlannerSurfaceBridge(TaskExecutionBridge):
    def function_tools(self) -> list[dict[str, Any]]:
        return [
            *super().function_tools(),
            {
                "type": "function",
                "name": "moveit_plan_pick_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_place_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_compound_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_manipulation_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_pick",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_plan_place",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_release_object",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_open_gripper",
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
            {
                "type": "function",
                "name": "moveit_verify_released_object",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_remove_scene_object",
                "parameters": {"type": "object"},
                "strict": None,
            },
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_plan_manipulation_task":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "feedback": {
                            "can_execute": True,
                            "execution_target": "task_solution",
                        },
                        "task_solution_id": "manipulation_hold_dynamic_5_001",
                        "task_kind": "hold",
                        "object_name": "dynamic_5",
                        "raw": {
                            "task_solution_id": "manipulation_hold_dynamic_5_001",
                            "task_kind": "hold",
                            "backend": "staged_moveit",
                            "object_name": "dynamic_5",
                            "robot_name": "UR10",
                            "created_from_tool": "moveit_plan_manipulation_task",
                            "scene_snapshot_id": "scene_20260515_001",
                        },
                    }
                }
            )
        return await super().call_tool(name, arguments)


class TimeoutTaskPlannerBridge(TaskPlannerSurfaceBridge):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_plan_manipulation_task":
            from robot_control.mcp_bridge import RobotMCPError

            self.calls.append((name, arguments))
            raise RobotMCPError(
                "Robot MCP tool moveit_plan_manipulation_task timed out: TimeoutError: read timed out"
            )
        return await super().call_tool(name, arguments)


class SchemaRejectingTaskPlannerBridge(TaskPlannerSurfaceBridge):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_plan_manipulation_task":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "ok": False,
                    "error": "Unexpected argument for moveit_plan_manipulation_task: backend",
                    "correction": "Remove backend; Robot Control selects the planner backend.",
                    "retryable": True,
                    "suggested_next_tool": "moveit_plan_manipulation_task",
                }
            )
        return await super().call_tool(name, arguments)


class FailingManipulationPlannerBridge(TaskPlannerSurfaceBridge):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_plan_manipulation_task":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": False,
                        "robot": "UR10",
                        "tool": "moveit_plan_manipulation_task",
                        "feedback": {
                            "phase": "planned",
                            "status": "staged manipulation task planning failed",
                            "message": (
                                "Required manipulation stage approach_to_pre_grasp could not be "
                                "planned with preview evidence."
                            ),
                            "can_execute": False,
                            "correction": (
                                "Inspect the failed candidate stage, adjust the grasp face or "
                                "object pose, then retry moveit_plan_manipulation_task."
                            ),
                        },
                        "verification": {
                            "result": "fail",
                            "checks": [
                                {
                                    "name": "required_motion_stages_planned",
                                    "passed": False,
                                    "details": "approach_to_pre_grasp",
                                }
                            ],
                        },
                        "failed_stage": "approach_to_pre_grasp",
                        "failure_code": "required_motion_stage_unplanned",
                        "suggested_next_action": (
                            "Inspect the failed candidate stage, adjust the grasp face or object "
                            "pose, then retry moveit_plan_manipulation_task."
                        ),
                        "retryable": True,
                    },
                    "is_error": False,
                }
            )
        return await super().call_tool(name, arguments)


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


class MissingUserPositionBridge(FakeUserSensingBridge):
    async def read_context(self, *, max_age_s: float) -> str:
        self.calls.append(max_age_s)
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "user": {
                        "available": False,
                        "position": None,
                        "age_s": None,
                        "stale": True,
                    },
                }
            }
        )


class FakeGeometryWorldContext:
    def render_instruction_block(self) -> str:
        return "\n".join(
            [
                "Geometry World Context:",
                "- physical model: physical_frame",
                "- hologram model: hologram_frame",
                '- {"object_name": "dynamic_1", "hologram_model_name": "hologram_frame", '
                '"target_pose": {"position": {"x": 0.0, "y": -0.8, "z": 0.1}, '
                '"orientation": {"x": 0.5, "y": 0.5, "z": -0.5, "w": 0.5}}}',
            ]
        )


class FakeVerifiedExecutionClient:
    def __init__(self, readiness: dict[str, Any] | None = None) -> None:
        self.readiness = readiness or {
            "server_available": True,
            "robot_connected": True,
            "gripper_connected": True,
            "error": None,
        }
        self.readiness_calls: list[float] = []
        self.calls: list[tuple[str, str, float]] = []
        self.gripper_calls: list[tuple[str, str, float]] = []
        self.home_calls: list[tuple[str, float]] = []
        self.sync_calls: list[tuple[str, float]] = []

    async def get_readiness(self, timeout_s: float) -> dict[str, Any]:
        self.readiness_calls.append(timeout_s)
        return dict(self.readiness)

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

    async def open_gripper(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        self.gripper_calls.append((robot_name, "open", timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "moveit_open_gripper",
                    "phase": "gripper",
                    "status": "gripper_open",
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

    async def go_home(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        self.home_calls.append((robot_name, timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "moveit_go_home",
                    "phase": "recovery",
                    "status": "homed",
                    "verification": {"result": "pass"},
                    "feedback": {"state_sync_published": True},
                },
                "is_error": False,
            }
        )

    async def sync_real_robot_state(
        self,
        *,
        robot_name: str,
        timeout_s: float,
    ) -> str:
        self.sync_calls.append((robot_name, timeout_s))
        return json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "robot": robot_name,
                    "tool": "moveit_sync_real_robot_state",
                    "phase": "observation",
                    "status": "state_synced",
                    "verification": {"result": "pass"},
                    "feedback": {"state_sync_published": True},
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


class EventRecordingVerifiedExecutionClient(FakeVerifiedExecutionClient):
    def __init__(
        self,
        events: list[tuple[str, str]],
        physical_started: asyncio.Event,
        *,
        motion_delay_s: float = 0.01,
    ) -> None:
        super().__init__()
        self.events = events
        self.physical_started = physical_started
        self.motion_delay_s = motion_delay_s

    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        self.calls.append((robot_name, plan_name, timeout_s))
        self.events.append(("physical_start", plan_name))
        self.physical_started.set()
        await asyncio.sleep(self.motion_delay_s)
        self.events.append(("physical_finish", plan_name))
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
        self.events.append(("physical_close_gripper", robot_name))
        return await super().close_gripper(robot_name=robot_name, timeout_s=timeout_s)


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
    geometry_world_context: Any | None = None,
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
        geometry_world_context=geometry_world_context,
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
        "manipulation_planner_repairs": 0,
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


def latest_state_tool_content(fixture: GraphFixture) -> str:
    tool_messages = [
        message
        for message in fixture.graph.latest_state()["messages"]
        if isinstance(message, ToolMessage)
    ]
    assert tool_messages
    content = tool_messages[-1].content
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


def approved_contract_task_context(
    *,
    task_solution_id: str,
    task_kind: str,
    object_name: str,
    raw: dict[str, Any],
) -> RobotContextStore:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id=task_solution_id,
        task_kind=task_kind,
        object_name=object_name,
        backend="mcp",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw=raw,
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id=task_solution_id,
        source_tool=f"moveit_plan_{task_kind}_task",
        object_name=object_name,
        expected_movement=f"{task_kind} {object_name}",
        scene_snapshot_id="scene_20260515_001",
    )
    assert context.record_task_solution_approval(
        task_solution_id,
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    return context


def compound_task_raw(
    *,
    task_solution_id: str,
    task_kind: str,
    object_name: str = "dynamic_5",
    workflow_steps: list[dict[str, Any]] | None = None,
    execution_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "task_solution_id": task_solution_id,
        "task_kind": task_kind,
        "backend": "emulated",
        "object_name": object_name,
        "robot_name": "UR10",
        "created_from_tool": (
            "moveit_plan_place_task" if task_kind in {"place", "move_and_release"} else "moveit_plan_pick_task"
        ),
        "scene_snapshot_id": "scene_20260515_001",
        "waypoints": [
            {
                "position": {"x": 0.50, "y": 0.10, "z": 0.40},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            {
                "position": {"x": 0.58, "y": 0.10, "z": 0.32},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            {
                "position": {"x": 0.58, "y": 0.10, "z": 0.42},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        ],
        "workflow_steps": workflow_steps or [],
    }
    if execution_contract is not None:
        raw["execution_contract"] = contract_with_proof(execution_contract)
    return raw


def contract_with_proof(execution_contract: dict[str, Any]) -> dict[str, Any]:
    contract = dict(execution_contract)
    key = "stages" if isinstance(contract.get("stages"), list) else "steps"
    steps = contract.get(key)
    if not isinstance(steps, list):
        return contract
    proven_steps: list[Any] = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            proven_steps.append(step)
            continue
        proven_step = dict(step)
        proven_step.setdefault("source_stage", f"mtc_stage_{index}")
        proven_step.setdefault(
            "required_proof",
            str(
                proven_step.get("handler")
                or proven_step.get("intent")
                or proven_step.get("name")
                or "stage"
            ),
        )
        proven_steps.append(proven_step)
    contract[key] = proven_steps
    return contract


def place_execution_contract() -> dict[str, Any]:
    return {
        "goal": "place",
        "stages": [
            {"kind": "motion", "intent": "place_motion", "name": "place", "waypoint_index": 1},
            {"kind": "gripper", "intent": "verified_open", "name": "open_gripper"},
            {
                "kind": "scene",
                "intent": "release_detach",
                "name": "release_object",
                "arguments": {
                    "object_pose": {
                        "position": {"x": 0.58, "y": 0.10, "z": 0.32},
                        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                    },
                },
            },
            {"kind": "motion", "intent": "retreat", "name": "retreat", "waypoint_index": 2},
            {"kind": "verify", "intent": "release_proof", "name": "verify_release"},
        ],
    }


def hold_execution_contract() -> dict[str, Any]:
    return {
        "goal": "pick_and_hold",
        "stages": [
            {"kind": "motion", "intent": "approach", "name": "approach", "waypoint_index": 0},
            {"kind": "motion", "intent": "grasp", "name": "grasp", "waypoint_index": 1},
            {"kind": "gripper", "intent": "verified_close", "name": "close_gripper"},
            {"kind": "scene", "intent": "attach", "name": "attach_object"},
            {"kind": "motion", "intent": "hold_retreat", "name": "hold", "waypoint_index": 2},
            {"kind": "verify", "intent": "attachment_proof", "name": "verify_attachment"},
        ],
    }


def bare_hold_execution_contract() -> dict[str, Any]:
    return {
        "goal": "hold",
        "steps": [
            {
                "step": 1,
                "handler": "motion",
                "name": "connect_to_pre_grasp",
                "plan_handle": "hold_preview_connect",
                "waypoint_index": 0,
                "source_stage": "connect_to_pre_grasp",
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 2,
                "handler": "motion",
                "name": "approach_to_pre_grasp",
                "plan_handle": "hold_preview_approach",
                "waypoint_index": 1,
                "source_stage": "approach_to_pre_grasp",
                "required_proof": "verified_motion_plan",
            },
            {
                "step": 3,
                "handler": "close_gripper",
                "name": "close_gripper",
                "source_stage": "close_gripper",
                "required_proof": "verified_gripper_closed",
            },
            {
                "step": 4,
                "handler": "attach_object",
                "name": "attach_object",
                "source_stage": "attach_object",
                "required_proof": "planning_scene_attached",
            },
            {
                "step": 5,
                "handler": "verify_attached_object",
                "name": "verify_attached_object",
                "source_stage": "verify_attached_object",
                "required_proof": "attachment_check",
            },
        ],
    }


class CompoundTaskPlanBridge(FakeBridge):
    def __init__(self, *, release_proof: bool = False) -> None:
        super().__init__()
        self.release_proof = release_proof

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
                "name": "moveit_open_gripper",
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
            {
                "type": "function",
                "name": "moveit_release_object",
                "parameters": {"type": "object"},
                "strict": None,
            },
            {
                "type": "function",
                "name": "moveit_verify_released_object",
                "parameters": {"type": "object"},
                "strict": None,
            },
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_get_current_pose":
            return await FakeBridge.call_tool(self, name, arguments)
        self.calls.append((name, arguments))
        if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": arguments["plan_name"]},
                    }
                }
            )
        if name == "moveit_open_gripper":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "tool": "moveit_open_gripper",
                        "verification": {"result": "pass"},
                    }
                }
            )
        if name in {"moveit_verify_attached_object", "moveit_verify_released_object"}:
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "object_name": arguments["object_name"],
                        "verification": {"result": "pass"},
                        "raw": {
                            "object_name": arguments["object_name"],
                            "mcp_attached_object": (
                                None if self.release_proof else arguments["object_name"]
                            ),
                            "mcp_gripper_holds_object": not self.release_proof,
                            "planning_scene_state": (
                                "free" if self.release_proof else "attached"
                            ),
                        },
                    }
                }
            )
        return json.dumps({"structured_content": {"ok": True}})


class HiddenContractToolBridge(CompoundTaskPlanBridge):
    def __init__(self, *, release_proof: bool = False) -> None:
        super().__init__(release_proof=release_proof)
        self.contract_calls: list[tuple[str, dict[str, Any]]] = []

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            tool
            for tool in super().function_tools()
            if tool["name"]
            not in {"moveit_release_object", "moveit_verify_released_object"}
        ]

    def contract_tool_names(self) -> set[str]:
        return {
            str(tool["name"])
            for tool in super().function_tools()
            if isinstance(tool.get("name"), str)
        }

    async def call_contract_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.contract_calls.append((name, arguments))
        return await super().call_tool(name, arguments)


class StageSynchronizedTaskBridge(CompoundTaskPlanBridge):
    def function_tools(self) -> list[dict[str, Any]]:
        return [
            *super().function_tools(),
            {
                "type": "function",
                "name": "moveit_execute_task",
                "parameters": {"type": "object"},
                "strict": None,
            },
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_execute_plan":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": arguments["robot_name"],
                        "tool": "moveit_execute_plan",
                        "status": "executed",
                        "feedback": {"plan_name": arguments["plan_name"]},
                        "verification": {"result": "pass"},
                    }
                }
            )
        if name == "moveit_close_gripper":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": arguments["robot_name"],
                        "tool": "moveit_close_gripper",
                        "status": "gripper_closed",
                        "verification": {"result": "pass"},
                    }
                }
            )
        if name == "moveit_attach_object":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": arguments["robot_name"],
                        "tool": "moveit_attach_object",
                        "object_name": arguments["object_name"],
                        "verification": {"result": "pass"},
                        "raw": {
                            "mcp_attached_object": arguments["object_name"],
                            "mcp_gripper_holds_object": True,
                            "planning_scene_state": "attached",
                        },
                    }
                }
            )
        return await super().call_tool(name, arguments)


class BlockingParallelExecutionBridge(StageSynchronizedTaskBridge):
    def __init__(
        self,
        events: list[tuple[str, str]],
        physical_started: asyncio.Event,
    ) -> None:
        super().__init__()
        self.events = events
        self.physical_started = physical_started
        self._waited_for_first_motion = False

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_execute_plan":
            self.calls.append((name, arguments))
            plan_name = str(arguments["plan_name"])
            self.events.append(("ar_start", plan_name))
            if not self._waited_for_first_motion:
                self._waited_for_first_motion = True
                try:
                    await asyncio.wait_for(self.physical_started.wait(), timeout=0.05)
                    self.events.append(("ar_saw_physical_start", plan_name))
                except TimeoutError:
                    self.events.append(("ar_no_physical_before_finish", plan_name))
            self.events.append(("ar_finish", plan_name))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": arguments["robot_name"],
                        "tool": "moveit_execute_plan",
                        "status": "executed",
                        "feedback": {"plan_name": arguments["plan_name"]},
                        "verification": {"result": "pass"},
                    }
                }
            )
        if name == "moveit_close_gripper":
            self.events.append(("ar_close_gripper", str(arguments["robot_name"])))
        if name == "moveit_attach_object":
            self.events.append(("ar_attach_object", str(arguments["object_name"])))
        return await super().call_tool(name, arguments)


class FailingParallelExecutionBridge(BlockingParallelExecutionBridge):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_execute_plan":
            self.calls.append((name, arguments))
            plan_name = str(arguments["plan_name"])
            self.events.append(("ar_start", plan_name))
            try:
                await asyncio.wait_for(self.physical_started.wait(), timeout=0.05)
                self.events.append(("ar_saw_physical_start", plan_name))
            except TimeoutError:
                self.events.append(("ar_no_physical_before_finish", plan_name))
            self.events.append(("ar_fail", plan_name))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": False,
                        "robot": arguments["robot_name"],
                        "tool": "moveit_execute_plan",
                        "status": "execution_failed",
                        "feedback": {"plan_name": arguments["plan_name"]},
                        "verification": {"result": "fail"},
                        "error": "AR/RViz execution failed.",
                    },
                    "is_error": True,
                }
            )
        return await super().call_tool(name, arguments)


class FailingTaskStageBridge(StageSynchronizedTaskBridge):
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "moveit_plan_free_motion":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": False,
                        "robot": arguments["robot_name"],
                        "tool": "plan_free_motion",
                        "feedback": {
                            "phase": "planned",
                            "status": "planning result invalid",
                            "message": "Plan did not satisfy execution requirements",
                            "can_execute": False,
                            "correction": (
                                "Replan with a smaller or safer target, then execute only "
                                "a successful returned raw.plan_name."
                            ),
                        },
                        "verification": {"result": "fail"},
                        "raw": {
                            "plan_name": arguments["plan_name"],
                            "trajectory_points": 0,
                            "can_execute": False,
                        },
                    },
                    "is_error": False,
                }
            )
        if name == "moveit_explain_motion_failure":
            self.calls.append((name, arguments))
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "tool": "moveit_explain_motion_failure",
                        "correction": (
                            "Observe current robot state, inspect the failed result, "
                            "then retry with a narrower plan."
                        ),
                        "retryable": True,
                        "suggested_next_tool": "moveit_get_robot_state",
                    },
                    "is_error": False,
                }
            )
        return await super().call_tool(name, arguments)


def approved_hold_contract_context(task_solution_id: str = "hold_task_dynamic_5_001") -> RobotContextStore:
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="hold",
        execution_contract=hold_execution_contract(),
    )
    return approved_contract_task_context(
        task_solution_id=task_solution_id,
        task_kind="hold",
        object_name="dynamic_5",
        raw=raw,
    )


def bridge_tool_names(bridge: FakeBridge) -> list[str]:
    return [name for name, _arguments in bridge.calls]


def executed_ar_rviz_plan_names(bridge: FakeBridge) -> list[str]:
    return [
        str(arguments["plan_name"])
        for name, arguments in bridge.calls
        if name == "moveit_execute_plan"
    ]


def assert_subsequence(values: list[str], expected: list[str]) -> None:
    position = 0
    for value in values:
        if position < len(expected) and value == expected[position]:
            position += 1
    if position != len(expected):
        pytest.fail(
            f"Missing stage call sequence {expected[position:]}; observed {values}"
        )


def assert_stage_synchronized_ar_rviz_calls(bridge: FakeBridge) -> None:
    names = bridge_tool_names(bridge)
    assert "moveit_execute_task_solution" not in names
    assert_subsequence(
        names,
        [
            "moveit_plan_free_motion",
            "moveit_execute_plan",
            "moveit_plan_cartesian_motion",
            "moveit_execute_plan",
            "moveit_close_gripper",
            "moveit_attach_object",
            "moveit_plan_cartesian_motion",
            "moveit_execute_plan",
            "moveit_verify_attached_object",
        ],
    )


def event_index(events: list[tuple[str, str]], event: tuple[str, str]) -> int:
    try:
        return events.index(event)
    except ValueError:
        pytest.fail(f"Missing event {event}; observed {events}")


def write_dynamic_role_model(tmp_path: Any) -> Any:
    model_path = tmp_path / "physical_model.json"
    model_path.write_text(
        json.dumps(
            {
                "bodies": [
                    {"id": "dynamic_1", "state": {"role": {"type": "unassigned"}}},
                    {"id": "dynamic_2", "state": {"role": {"type": "unassigned"}}},
                ],
                "operation_history": [],
            }
        ),
        encoding="utf-8",
    )
    return model_path


def patch_dynamic_role_update(monkeypatch: pytest.MonkeyPatch, model_path: Any) -> None:
    import agent_control.langgraph_robot_agent as agent_module
    from robot_control.shared_geometry.role_update import update_dynamic_role as real_update

    def update_dynamic_role_for_test(
        object_name: str,
        role: dict[str, object],
        reason: str,
    ) -> dict[str, object]:
        return real_update(object_name, role, reason, model_path=model_path)

    monkeypatch.setattr(
        agent_module,
        "update_dynamic_role",
        update_dynamic_role_for_test,
        raising=False,
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
async def test_graph_injects_geometry_world_context_without_extra_tool_call() -> None:
    fixture = make_graph(
        [ai_text("ready")],
        geometry_world_context=FakeGeometryWorldContext(),
    )

    text = await fixture.graph.run_turn(turn("bring element 1 here"))

    assert text == "ready"
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    system_text = str(first_request[0].content)
    assert "Geometry World Context" in system_text
    assert '"object_name": "dynamic_1"' in system_text
    assert '"hologram_model_name": "hologram_frame"' in system_text
    assert '"position": {"x": 0.0, "y": -0.8, "z": 0.1}' in system_text
    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "geometry_get_hologram_target_pose" not in tool_names


@pytest.mark.asyncio
async def test_graph_exposes_local_dynamic_role_update_tool_to_model() -> None:
    fixture = make_graph([ai_text("ready")])

    await fixture.graph.run_turn(turn("dynamic_1 is a structural support"))

    tools = {
        tool["function"]["name"]: tool["function"]
        for tool in fixture.model.bound_tools
    }
    assert "geometry_update_dynamic_role" in tools
    role_tool = tools["geometry_update_dynamic_role"]
    assert "ask the human" in role_tool["description"].lower()
    assert role_tool["parameters"]["required"] == ["object_name", "role", "reason"]
    assert set(role_tool["parameters"]["properties"]) == {"object_name", "role", "reason"}
    assert not any(name.startswith("geometry_get_") for name in tools)


@pytest.mark.asyncio
async def test_graph_dynamic_role_tool_updates_physical_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    model_path = write_dynamic_role_model(tmp_path)
    patch_dynamic_role_update(monkeypatch, model_path)
    fixture = make_graph(
        [
            ai_tool_call(
                "geometry_update_dynamic_role",
                {
                    "object_name": "dynamic_1",
                    "role": {"type": "supporting_column", "supports": ["dynamic_2"]},
                    "reason": "operator confirmed structural support",
                },
            ),
            ai_text("Noted."),
        ]
    )

    text = await fixture.graph.run_turn(turn("dynamic_1 supports dynamic_2"))

    assert text == "Noted."
    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": True,
        "object_name": "dynamic_1",
        "role": {"type": "supporting_column", "supports": ["dynamic_2"]},
        "physical_model_updated": True,
    }
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["bodies"][0]["state"]["role"] == {
        "type": "supporting_column",
        "supports": ["dynamic_2"],
    }
    assert ("geometry_update_dynamic_role", {}) not in fixture.bridge.calls
    assert all(name != "geometry_update_dynamic_role" for name, _ in fixture.bridge.calls)


@pytest.mark.asyncio
async def test_graph_dynamic_role_tool_returns_structured_failure_for_invalid_role(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    model_path = write_dynamic_role_model(tmp_path)
    patch_dynamic_role_update(monkeypatch, model_path)
    fixture = make_graph(
        [
            ai_tool_call(
                "geometry_update_dynamic_role",
                {
                    "object_name": "dynamic_1",
                    "role": {"type": "left_support"},
                    "reason": "operator used view-dependent wording",
                },
            ),
            ai_text("I need a structural role."),
        ]
    )

    text = await fixture.graph.run_turn(turn("dynamic_1 is the left support"))

    assert text == "I need a structural role."
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert "view-dependent" in output["error"]
    assert output["retryable"] is True
    model = json.loads(model_path.read_text(encoding="utf-8"))
    assert model["bodies"][0]["state"]["role"] == {"type": "unassigned"}
    assert all(name != "geometry_update_dynamic_role" for name, _ in fixture.bridge.calls)


@pytest.mark.asyncio
async def test_graph_uses_unified_task_execution_tool_in_verified_execution_mode() -> None:
    fixture = make_graph(
        [ai_text("ready")],
        bridge=TaskPlannerSurfaceBridge(),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("pick up dynamic_5"))

    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "moveit_execute_task" in tool_names
    assert "moveit_execute_task_plan" not in tool_names
    assert "moveit_execute_task_solution" not in tool_names
    assert "moveit_plan_manipulation_task" in tool_names
    assert "moveit_plan_compound_task" not in tool_names
    assert "moveit_plan_free_motion" not in tool_names
    assert "moveit_plan_cartesian_motion" not in tool_names
    assert "moveit_plan_pick" not in tool_names
    assert "moveit_plan_place" not in tool_names
    assert "moveit_plan_pick_task" not in tool_names
    assert "moveit_plan_place_task" not in tool_names
    assert "moveit_open_gripper" not in tool_names
    assert "moveit_close_gripper" not in tool_names
    assert "moveit_attach_object" not in tool_names
    assert "moveit_verify_attached_object" not in tool_names
    assert "moveit_release_object" not in tool_names
    assert "moveit_verify_released_object" not in tool_names
    assert "moveit_remove_scene_object" not in tool_names
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    assert (
        "Use moveit_execute_task for returned task_solution_id values. It executes AR/RViz and "
        "real-robot motion stages in parallel when Verified Real Robot Execution is ready"
    ) in str(first_request[0].content)


@pytest.mark.asyncio
async def test_graph_routes_model_visible_manipulation_planner_to_native_backend() -> None:
    args = {
        "robot_name": "UR10",
        "requirements": {
            "goal": "hold",
            "object_name": "dynamic_5",
            "lift_distance_m": 0.10,
        },
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_manipulation_task", args),
            ai_text("Task planned."),
        ],
        bridge=TaskPlannerSurfaceBridge(),
    )

    text = await fixture.graph.run_turn(turn("pick up dynamic_5"))

    assert text in PLAN_READY_REPLIES
    assert ("moveit_plan_manipulation_task", args) in fixture.bridge.calls
    assert len(fixture.model.requests) == 1
    assert all(name != "moveit_plan_compound_task" for name, _ in fixture.bridge.calls)
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["task_solution_id"] == "manipulation_hold_dynamic_5_001"


@pytest.mark.asyncio
async def test_graph_resolves_human_relative_move_before_calling_manipulation_planner() -> None:
    args = {
        "robot_name": "UR10",
        "requirements": {
            "goal": "move",
            "motion": {
                "type": "human_relative",
                "relation": "toward_user",
                "distance_m": 0.20,
            },
        },
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [ai_tool_call("moveit_plan_manipulation_task", args), ai_text("Task planned.")],
        bridge=TaskPlannerSurfaceBridge(),
        user_sensing_bridge=FakeUserSensingBridge(),
    )

    text = await fixture.graph.run_turn(turn("come closer to me"))

    assert text in PLAN_READY_REPLIES
    manipulation_calls = [
        call_args for name, call_args in fixture.bridge.calls if name == "moveit_plan_manipulation_task"
    ]
    assert len(manipulation_calls) == 1
    motion = manipulation_calls[0]["requirements"]["motion"]
    assert motion["type"] == "relative_tcp"
    assert motion["resolved_from"]["type"] == "human_relative"
    assert motion["resolved_from"]["relation"] == "toward_user"
    assert motion["delta_m"]["x"] == pytest.approx(0.0505, abs=0.0001)
    assert motion["delta_m"]["y"] == pytest.approx(-0.1935, abs=0.0001)
    assert motion["delta_m"]["z"] == 0.0


@pytest.mark.asyncio
async def test_graph_rejects_human_relative_move_without_fresh_user_position() -> None:
    args = {
        "robot_name": "UR10",
        "requirements": {
            "goal": "move",
            "motion": {
                "type": "human_relative",
                "relation": "away_from_user",
                "distance_m": 0.20,
            },
        },
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [ai_tool_call("moveit_plan_manipulation_task", args), ai_text("Should not replan.")],
        bridge=TaskPlannerSurfaceBridge(),
        user_sensing_bridge=MissingUserPositionBridge(),
    )

    text = await fixture.graph.run_turn(turn("go away from me"))

    assert "fresh vizor user position" in text.lower()
    assert all(name != "moveit_plan_manipulation_task" for name, _ in fixture.bridge.calls)


@pytest.mark.asyncio
async def test_graph_sends_failed_manipulation_planner_result_back_to_model() -> None:
    args = {
        "robot_name": "UR10",
        "requirements": {
            "goal": "hold",
            "object_name": "dynamic_5",
            "lift_distance_m": 0.10,
        },
        "timeout_s": 9.0,
    }
    planner_correction = (
        "Inspect the failed candidate stage, adjust the grasp face or object pose, "
        "then retry moveit_plan_manipulation_task."
    )
    reflected_text = (
        "I could not plan a safe grasp for dynamic_5. I need a different grasp face or "
        "object pose before retrying."
    )
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_manipulation_task", args),
            ai_text(reflected_text),
        ],
        bridge=FailingManipulationPlannerBridge(),
    )

    text = await fixture.graph.run_turn(turn("pick up dynamic_5"))

    assert text == reflected_text
    assert len(fixture.model.requests) == 2
    output = json.loads(last_tool_content(fixture.model))
    assert output["structured_content"]["ok"] is False
    assert output["structured_content"]["feedback"]["correction"] == planner_correction
    assert text != planner_correction


@pytest.mark.asyncio
async def test_graph_stops_after_manipulation_planner_timeout() -> None:
    args = {
        "robot_name": "UR10",
        "requirements": {
            "goal": "hold",
            "object_name": "dynamic_5",
            "lift_distance_m": 0.10,
        },
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_manipulation_task", args),
            ai_text("I should not be asked to replan."),
        ],
        bridge=TimeoutTaskPlannerBridge(),
    )

    text = await fixture.graph.run_turn(turn("pick up dynamic_5"))

    assert text == "Planning timed out before a complete task solution was returned."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_manipulation_task", args),
    ]
    assert len(fixture.model.requests) == 1


@pytest.mark.asyncio
async def test_graph_stops_after_repeated_manipulation_schema_error() -> None:
    args = {
        "robot_name": "UR10",
        "backend": "staged_moveit",
        "requirements": {
            "goal": "hold",
            "object_name": "dynamic_5",
            "lift_distance_m": 0.10,
        },
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_plan_manipulation_task", args, call_id="plan-1"),
            ai_tool_call("moveit_plan_manipulation_task", args, call_id="plan-2"),
            ai_text("I should not be asked again."),
        ],
        bridge=SchemaRejectingTaskPlannerBridge(),
    )

    text = await fixture.graph.run_turn(turn("pick up dynamic_5"))

    assert text == "Remove backend; Robot Control selects the planner backend."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_manipulation_task", args),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_manipulation_task", args),
    ]
    assert len(fixture.model.requests) == 2


@pytest.mark.asyncio
async def test_graph_routes_go_home_to_verified_execution_client() -> None:
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [ai_tool_call("moveit_go_home", {"robot_name": "UR10", "timeout_s": 12.0})],
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("go home"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert verified_client.home_calls == [("UR10", 12.0)]
    assert fixture.bridge.calls == []


@pytest.mark.asyncio
async def test_graph_blocks_go_home_without_explicit_user_wording() -> None:
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [ai_tool_call("moveit_go_home", {"robot_name": "UR10", "timeout_s": 12.0})],
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("what is the robot state?"))

    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert "explicit user/operator intent" in output["error"]
    assert verified_client.home_calls == []
    assert all(name != "moveit_go_home" for name, _ in fixture.bridge.calls)


@pytest.mark.asyncio
async def test_graph_routes_sync_real_robot_state_to_verified_execution_client() -> None:
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_sync_real_robot_state",
                {"robot_name": "UR10", "timeout_s": 6.0},
            )
        ],
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("sync the real robot state"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert verified_client.sync_calls == [("UR10", 6.0)]
    assert fixture.bridge.calls == []


@pytest.mark.asyncio
async def test_graph_uses_unified_task_execution_tool_in_simulation_mode() -> None:
    fixture = make_graph([ai_text("ready")], bridge=TaskPlannerSurfaceBridge())

    await fixture.graph.run_turn(turn("pick up dynamic_5"))

    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "moveit_execute_task" in tool_names
    assert "moveit_execute_task_solution" not in tool_names
    assert "moveit_execute_task_plan" not in tool_names
    assert "moveit_plan_manipulation_task" in tool_names
    assert "moveit_plan_compound_task" not in tool_names
    assert "moveit_plan_pick_task" not in tool_names
    assert "moveit_plan_place_task" not in tool_names
    first_request = fixture.model.requests[0]
    assert isinstance(first_request[0], SystemMessage)
    assert (
        "Use moveit_execute_task for returned task_solution_id values. It executes AR/RViz and "
        "real-robot motion stages in parallel when Verified Real Robot Execution is ready"
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
            "tool_count": 3,
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
    execute_args = {"robot_name": "UR10", "plan_name": "invented-plan"}

    output = await graph._call_policy_checked_tool(
        "moveit_execute_plan",
        execute_args,
        allow_execution=True,
    )

    assert bridge.calls == []
    assert json.loads(output)["ok"] is False
    policy_span = records_named(writer, "robot.task_policy")[-1]
    assert policy_span["module"] == "robot_control"
    assert policy_span["status"] == "ok"
    assert policy_span["attributes"]["tool.name"] == "moveit_execute_plan"
    assert policy_span["attributes"]["decision_ok"] is False
    assert policy_span["attributes"]["suggested_next_tool"] == "moveit_plan_free_motion"


@pytest.mark.asyncio
async def test_graph_rejects_cartesian_for_compound_release_request() -> None:
    cartesian_args = {
        "robot_name": "UR10",
        "waypoints": [
            {"position": {"x": 0.1, "y": 0.5, "z": 0.3}},
        ],
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_get_current_pose", {"robot_name": "UR10"}, call_id="pose-1"),
            ai_tool_call("moveit_plan_cartesian_motion", cartesian_args, call_id="plan-1"),
            ai_text("I need to use the compound task for that action."),
        ]
    )

    text = await fixture.graph.run_turn(
        turn("move it 30cm in robot left side and then release the gripper")
    )

    assert text == "I need to use the compound task for that action."
    assert ("moveit_plan_cartesian_motion", cartesian_args) not in fixture.bridge.calls
    assert all(name == "moveit_get_current_pose" for name, _args in fixture.bridge.calls)
    output = json.loads(last_tool_content(fixture.model))
    assert output["ok"] is False
    assert output["error"] == "Compound manipulation tasks must use task planning tools."
    assert output["retryable"] is True
    assert output["suggested_next_tool"] == "moveit_plan_manipulation_task"
    assert "moveit_plan_manipulation_task" in output["correction"]


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
        "timeout_s": 120.0,
    }
    assert job.after_success_tool == "moveit_plan_pick"
    assert job.after_success_arguments == {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "planning_strategy": "cartesian",
    }
    assert job.execute_via_mcp is True


@pytest.mark.asyncio
async def test_verified_execute_runs_gripper_release_after_success() -> None:
    context = RobotContextStore()
    context.remember_executable_plan(
        "place-plan-1",
        robot_name="UR10",
        source_tool="moveit_plan_place",
        after_success_tool="moveit_open_gripper",
        after_success_arguments={"robot_name": "UR10", "timeout_s": 5.0},
    )
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_plan",
                {"robot_name": "UR10", "plan_name": "place-plan-1", "timeout_s": 30.0},
            ),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("execute plan"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert fixture.model.requests
    assert verified_client.calls == [("UR10", "place-plan-1", 30.0)]
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_open_gripper", {"robot_name": "UR10", "timeout_s": 5.0}),
    ]


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
async def test_graph_applies_task_policy_before_queued_execute_tool() -> None:
    from agent_control.robot_job_submission import RobotJobSubmitter
    from robot_control.job_board import RobotJobBoard

    class NoObservationBridge(FakeBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                {
                    "type": "function",
                    "name": "moveit_execute_plan",
                    "parameters": {"type": "object"},
                    "strict": None,
                }
            ]

    board = RobotJobBoard()
    execute_args = {"robot_name": "UR10", "plan_name": "invented-plan"}
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_plan", execute_args),
            ai_text("I need to plan before executing."),
        ],
        bridge=NoObservationBridge(),
        job_submitter=RobotJobSubmitter(board),
    )

    text = await fixture.graph.run_turn(turn("execute the plan"))

    assert text == "I need to plan before executing."
    assert fixture.bridge.calls == []
    assert await board.claim_next() is None
    assert board.events_since(0) == []
    output = json.loads(last_tool_content(fixture.model))
    assert output == {
        "ok": False,
        "error": "Cannot execute an unknown or stale plan.",
        "correction": "Plan first, then execute the returned plan_name.",
        "retryable": True,
        "suggested_next_tool": "moveit_plan_free_motion",
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
async def test_graph_sends_motion_plan_to_bridge_without_fresh_pose_policy_failure() -> None:
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
        [ai_tool_call("moveit_plan_free_motion", plan_args), ai_text("Plan ready.")],
        bridge=NoObservationBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text in PLAN_READY_REPLIES
    assert fixture.bridge.calls == [("moveit_plan_free_motion", plan_args)]


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
        "code": "approval_missing",
        "retryable": True,
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
async def test_graph_execute_task_runs_staged_ar_rviz_when_verified_client_missing() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 10.0,
    }
    bridge = StageSynchronizedTaskBridge()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_pick_task_context(),
    )

    text = await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert text in PHYSICAL_STATUS_UNAVAILABLE_REPLIES
    assert_stage_synchronized_ar_rviz_calls(bridge)
    output = json.loads(latest_state_tool_content(fixture))
    structured = output["structured_content"]
    assert structured["ok"] is True
    assert structured["tool"] == "moveit_execute_task"
    assert structured["task_solution_id"] == "pick_task_dynamic_5_001"
    assert structured["simulation"]["ok"] is True
    assert structured["real_robot"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_graph_execute_task_runs_release_only_contract() -> None:
    class ReleaseOnlyTaskBridge(TaskPlannerSurfaceBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            if name == "moveit_verify_released_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                            "raw": {
                                "object_name": arguments["object_name"],
                                "planning_scene_state": "released",
                                "mcp_attached_object": None,
                                "mcp_gripper_holds_object": False,
                                "attached": False,
                                "released_object_pose": {
                                    "position": {"x": 0.047, "y": -0.703, "z": 0.189},
                                    "orientation": {
                                        "x": 0.5,
                                        "y": 0.5,
                                        "z": 0.5,
                                        "w": 0.5,
                                    },
                                },
                            },
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    task_solution_id = "release_task_dynamic_0_003"
    release_object_pose = {
        "position": {"x": 0.047, "y": -0.703, "z": 0.189},
        "orientation": {"x": 0.5, "y": 0.5, "z": 0.5, "w": 0.5},
    }
    raw = {
        "task_solution_id": task_solution_id,
        "task_kind": "release",
        "object_name": "dynamic_0",
        "scene_snapshot_id": "scene_20260519_003",
        "execution_contract": {
            "steps": [
                {
                    "handler": "open_gripper",
                    "name": "open_gripper",
                    "tool": "moveit_open_gripper",
                    "source_stage": "open_gripper",
                    "required_proof": "verified_gripper_open",
                },
                {
                    "handler": "release_object",
                    "name": "release_object",
                    "tool": "moveit_release_object",
                    "source_stage": "detach_object",
                    "required_proof": "planning_scene_update",
                    "arguments": {
                        "object_name": "dynamic_0",
                        "object_pose": release_object_pose,
                    },
                },
                {
                    "handler": "verify_released_object",
                    "name": "verify_released_object",
                    "tool": "moveit_verify_released_object",
                    "source_stage": "verify_released_object",
                    "required_proof": "release_check",
                    "arguments": {"object_name": "dynamic_0"},
                },
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=ReleaseOnlyTaskBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="release",
            object_name="dynamic_0",
            raw=raw,
        ),
    )

    text = await fixture.graph.run_turn(turn("execute"))

    assert text in PHYSICAL_STATUS_UNAVAILABLE_REPLIES
    assert ("moveit_open_gripper", {"robot_name": "UR10", "timeout_s": 9.0}) in fixture.bridge.calls
    assert (
        "moveit_release_object",
        {
            "robot_name": "UR10",
            "object_name": "dynamic_0",
            "object_pose": release_object_pose,
            "verified_gripper_open": True,
        },
    ) in fixture.bridge.calls
    assert (
        "moveit_verify_released_object",
        {"robot_name": "UR10", "object_name": "dynamic_0", "timeout_s": 9.0},
    ) in fixture.bridge.calls


@pytest.mark.asyncio
async def test_graph_execute_task_stops_physical_after_readiness_then_failure() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    bridge = StageSynchronizedTaskBridge()
    verified_client = FakeFailingVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert text in PHYSICAL_EXECUTION_FAILED_REPLIES
    assert_stage_synchronized_ar_rviz_calls(bridge)
    assert verified_client.readiness_calls == [9.0]
    assert len(verified_client.calls) == 1
    assert verified_client.gripper_calls == []
    output = json.loads(latest_state_tool_content(fixture))
    structured = output["structured_content"]
    assert structured["ok"] is True
    assert structured["tool"] == "moveit_execute_task"
    assert structured["simulation"]["ok"] is True
    assert structured["real_robot"]["ok"] is False
    assert structured["real_robot"]["status"] == "failed"


@pytest.mark.asyncio
async def test_graph_execute_task_stage_failure_reports_failure_without_model_replan() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model replan"),
        ],
        bridge=FailingTaskStageBridge(),
        robot_context=approved_hold_contract_context(),
    )

    text = await fixture.graph.run_turn(turn("execute"))

    assert text.startswith("I could not finish the task because the planner reported:")
    assert "Plan did not satisfy execution requirements" in text
    assert "hold_task_dynamic_5_001" not in text
    assert "No task steps completed before the failure." in text
    assert "No new plan was executed." not in text
    assert "Please approve the next action" in text
    assert "unexpected model replan" not in text
    assert len(fixture.model.requests) == 1
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["task_solution_id"] == "hold_task_dynamic_5_001"
    assert output["failed_step"] == "approach"
    assert output["failed_stage"] == "planning"
    assert output["correction"] == (
        "Inspect the failed tool result, then replan before retrying task execution."
    )
    assert output["suggested_next_tool"] == "moveit_explain_motion_failure"
    assert "moveit_explain_motion_failure" in bridge_tool_names(fixture.bridge)
    planning_call_indexes = [
        index
        for index, call in enumerate(fixture.bridge.calls)
        if call[0] == "moveit_plan_free_motion"
    ]
    pose_call_indexes = [
        index
        for index, call in enumerate(fixture.bridge.calls)
        if call[0] == "moveit_get_current_pose"
    ]
    assert len(planning_call_indexes) == 2
    assert any(
        planning_call_indexes[0] < index < planning_call_indexes[1]
        for index in pose_call_indexes
    )
    task_retry_pose_calls = [
        fixture.bridge.calls[index]
        for index in pose_call_indexes
        if planning_call_indexes[0] < index < planning_call_indexes[1]
    ]
    assert all(
        call[1] == {"robot_name": "UR10", "timeout_s": 2.0}
        for call in task_retry_pose_calls
    )
    explain_calls = [
        call for call in fixture.bridge.calls if call[0] == "moveit_explain_motion_failure"
    ]
    assert explain_calls[0][1]["timeout_s"] == 9.0


@pytest.mark.asyncio
async def test_graph_execute_task_runs_ar_rviz_and_physical_when_verified_ready() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    bridge = StageSynchronizedTaskBridge()
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_hold_contract_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert_stage_synchronized_ar_rviz_calls(bridge)
    ar_rviz_plan_names = executed_ar_rviz_plan_names(bridge)
    assert len(ar_rviz_plan_names) == 3
    assert verified_client.readiness_calls == [9.0]
    assert [call[1] for call in verified_client.calls] == ar_rviz_plan_names
    assert verified_client.gripper_calls == [("UR10", "close", 9.0)]
    output = json.loads(latest_state_tool_content(fixture))
    structured = output["structured_content"]
    assert structured["ok"] is True
    assert structured["simulation"]["ok"] is True
    assert structured["real_robot"]["status"] == "executed"


@pytest.mark.asyncio
async def test_graph_execute_task_dispatches_physical_before_ar_rviz_motion_finishes() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    events: list[tuple[str, str]] = []
    physical_started = asyncio.Event()
    bridge = BlockingParallelExecutionBridge(events, physical_started)
    verified_client = EventRecordingVerifiedExecutionClient(events, physical_started)
    writer = MemoryTraceWriter()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_hold_contract_context(),
        verified_execution_client=verified_client,
        tracer=ProcessTracer(writer),
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    ar_rviz_plan_names = executed_ar_rviz_plan_names(bridge)
    first_plan = ar_rviz_plan_names[0]
    assert ("ar_saw_physical_start", first_plan) in events
    assert event_index(events, ("physical_start", first_plan)) < event_index(
        events,
        ("ar_finish", first_plan),
    )
    first_gripper_index = event_index(events, ("ar_close_gripper", "UR10"))
    for event in events[:first_gripper_index]:
        if event[0] == "physical_start":
            assert event_index(events, ("physical_finish", event[1])) < first_gripper_index
    output = json.loads(latest_state_tool_content(fixture))
    structured = output["structured_content"]
    assert structured["ok"] is True
    assert structured["real_robot"]["synchronization"]["mode"] == "parallel_dispatch"
    dispatch_events = records_named(writer, "robot.task_motion.parallel_dispatch")
    assert dispatch_events
    first_dispatch = dispatch_events[0]["attributes"]
    assert first_dispatch["plan_name"] == first_plan
    assert isinstance(first_dispatch["dispatch_skew_ms"], float)


@pytest.mark.asyncio
async def test_graph_execute_task_ar_rviz_failure_keeps_physical_dispatch_evidence() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    events: list[tuple[str, str]] = []
    physical_started = asyncio.Event()
    bridge = FailingParallelExecutionBridge(events, physical_started)
    verified_client = EventRecordingVerifiedExecutionClient(events, physical_started)
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_hold_contract_context(),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert text.startswith("I could not finish the task because the planner reported:")
    assert "AR/RViz execution failed" in text
    assert "verified_execution" not in text
    assert bridge_tool_names(bridge).count("moveit_execute_plan") == 1
    assert "moveit_close_gripper" not in bridge_tool_names(bridge)
    assert len(verified_client.calls) == 1
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["failed_stage"] == "verified_execution"
    real_robot = output["recovery"]["real_robot"]
    assert real_robot["status"] == "executed"
    assert real_robot["verified_plan_names"] == [verified_client.calls[0][1]]
    assert real_robot["synchronization"]["mode"] == "parallel_dispatch"


@pytest.mark.asyncio
async def test_graph_execute_task_skips_physical_when_verified_readiness_unavailable() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    bridge = StageSynchronizedTaskBridge()
    verified_client = FakeVerifiedExecutionClient(
        readiness={
            "server_available": True,
            "robot_connected": False,
            "gripper_connected": True,
            "error": "robot disconnected",
        }
    )
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_hold_contract_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert_stage_synchronized_ar_rviz_calls(bridge)
    assert verified_client.readiness_calls == [9.0]
    assert verified_client.calls == []
    assert verified_client.gripper_calls == []
    output = json.loads(latest_state_tool_content(fixture))
    structured = output["structured_content"]
    assert structured["ok"] is True
    assert structured["simulation"]["ok"] is True
    assert structured["real_robot"]["status"] == "unavailable"


@pytest.mark.asyncio
async def test_graph_execute_task_records_stage_synchronized_process_trace_evidence() -> None:
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    bridge = StageSynchronizedTaskBridge()
    writer = MemoryTraceWriter()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_hold_contract_context(),
        tracer=ProcessTracer(writer),
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert_stage_synchronized_ar_rviz_calls(bridge)
    trace_tool_names = [
        record["attributes"].get("tool.name")
        for record in writer.records
        if isinstance(record.get("attributes"), dict)
    ]
    assert "moveit_execute_task_solution" not in trace_tool_names
    for tool_name in [
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_execute_plan",
        "moveit_close_gripper",
        "moveit_attach_object",
        "moveit_verify_attached_object",
    ]:
        assert tool_name in trace_tool_names


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
                            "raw": {
                                "object_name": arguments["object_name"],
                                "mcp_attached_object": arguments["object_name"],
                                "mcp_gripper_holds_object": True,
                                "planning_scene_state": "attached",
                            },
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

    assert text in EXECUTION_COMPLETE_REPLIES
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
        ("moveit_get_current_pose", {"robot_name": "UR10", "timeout_s": 2.0}),
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
        ("moveit_get_current_pose", {"robot_name": "UR10", "timeout_s": 2.0}),
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
        ("moveit_get_current_pose", {"robot_name": "UR10", "timeout_s": 2.0}),
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
    ]
    output = json.loads(latest_state_tool_content(fixture))
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
                            "raw": {
                                "object_name": arguments["object_name"],
                                "mcp_attached_object": arguments["object_name"],
                                "mcp_gripper_holds_object": True,
                                "planning_scene_state": "attached",
                            },
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
                            "raw": {
                                "object_name": arguments["object_name"],
                                "mcp_attached_object": arguments["object_name"],
                                "mcp_gripper_holds_object": True,
                                "planning_scene_state": "attached",
                            },
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
    pose_call_indexes = [
        index
        for index, call in enumerate(fixture.bridge.calls)
        if call[0] == "moveit_get_current_pose"
    ]
    pre_grasp_call_indexes = [
        index
        for index, call in enumerate(fixture.bridge.calls)
        if call in pre_grasp_calls
    ]
    assert [call[0] for call in pre_grasp_calls] == [
        "moveit_plan_cartesian_motion",
        "moveit_plan_free_motion",
    ]
    assert any(
        pre_grasp_call_indexes[0] < index < pre_grasp_call_indexes[1]
        for index in pose_call_indexes
    )
    assert str(pre_grasp_calls[0][1]["plan_name"]).endswith("_try1")
    assert str(pre_grasp_calls[1][1]["plan_name"]).endswith("_try2")
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert len(verified_plan_names) == 3
    assert verified_plan_names[1] == pre_grasp_calls[1][1]["plan_name"]


@pytest.mark.asyncio
async def test_graph_returns_bounded_failure_for_dynamic_1_first_stage_timeout() -> None:
    from robot_control.mcp_bridge import RobotMCPError

    class TimeoutFirstStageBridge(FakeBridge):
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
                raise RobotMCPError(
                    f"Robot MCP tool {name} timed out: TimeoutError: read timed out"
                )
            return json.dumps({"structured_content": {"ok": True}})

    task_solution_id = "pick_task_dynamic_1_001"
    context = RobotContextStore(time_fn=lambda: 100.0)
    payload = json.loads(approved_pick_task_output())
    raw = payload["structured_content"]["raw"]
    raw["task_solution_id"] = task_solution_id
    raw["object_name"] = "dynamic_1"
    raw["scene_snapshot_id"] = "scene_dynamic_1_loaded"
    approval = raw["approval"]
    approval["task_solution_id"] = task_solution_id
    approval["object_name"] = "dynamic_1"
    approval["scene_snapshot_id"] = "scene_dynamic_1_loaded"
    context.update_from_tool_result("moveit_plan_pick_task", json.dumps(payload))
    assert context.record_task_solution_approval(
        task_solution_id,
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("unexpected model replan"),
        ],
        bridge=TimeoutFirstStageBridge(),
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the dynamic_1 pick task"))

    assert "a robot tool timed out while I was planning" in text
    assert "pick_task_dynamic_1_001" not in text
    assert "read timed out" not in text
    assert "No task steps completed before the failure." in text
    assert "No new plan was executed." not in text
    assert "Please approve the next action" in text
    assert "unexpected model replan" not in text
    assert len(fixture.model.requests) == 1
    assert verified_client.calls == []
    planning_calls = [
        call for call in fixture.bridge.calls if call[0].startswith("moveit_plan_")
    ]
    assert len(planning_calls) == 2
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["error"] == "Task plan planning failed at approach."
    assert output["task_solution_id"] == task_solution_id
    assert output["failed_step"] == "approach"
    assert output["retryable"] is True
    assert output["correction"] == (
        "Inspect the failed tool result, then replan before retrying task execution."
    )
    assert output["suggested_next_tool"] == "moveit_explain_motion_failure"
    assert output["failed_tool_name"] == "moveit_plan_free_motion"
    assert output["failed_tool_arguments"]["robot_name"] == "UR10"
    assert output["failed_tool_arguments"]["plan_name"].startswith(
        "pick_task_dynamic_1_001_approach_"
    )
    assert output["failed_tool_result"]["ok"] is False
    assert "timed out" in output["failed_tool_result"]["error"]
    assert "held object: dynamic_1" not in context.render_instruction_block()


@pytest.mark.asyncio
async def test_graph_stops_task_plan_retry_when_pose_refresh_after_planning_failure_fails() -> None:
    class PoseRefreshFailingRetryBridge(FakeBridge):
        def __init__(self) -> None:
            super().__init__()
            self.failed_pre_grasp_once = False
            self.pose_refresh_failures_remaining = 0

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
                if self.pose_refresh_failures_remaining > 0:
                    self.pose_refresh_failures_remaining -= 1
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
            plan_name = str(arguments.get("plan_name") or "")
            if (
                name == "moveit_plan_cartesian_motion"
                and "_pre_grasp_" in plan_name
                and not self.failed_pre_grasp_once
            ):
                self.failed_pre_grasp_once = True
                self.pose_refresh_failures_remaining = 3
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
            ai_text("I could not refresh the robot pose after planning failed."),
        ],
        bridge=PoseRefreshFailingRetryBridge(),
        robot_context=approved_pick_task_context(),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    pre_grasp_calls = [
        call
        for call in fixture.bridge.calls
        if call[0] in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
        and "_pre_grasp_" in str(call[1].get("plan_name"))
    ]
    assert [call[0] for call in pre_grasp_calls] == ["moveit_plan_cartesian_motion"]
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["failed_stage"] == "observe_current_pose"
    assert output["failed_tool_name"] == "moveit_get_current_pose"
    assert output["failed_tool_arguments"] == {"robot_name": "UR10", "timeout_s": 2.0}


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
    output = json.loads(latest_state_tool_content(fixture))
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
    output = json.loads(latest_state_tool_content(fixture))
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

    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan execution requires the recent raw task solution.",
        "correction": (
            "Plan the compound task again, then retry moveit_execute_task_plan with "
            "that task_solution_id."
        ),
        "retryable": True,
            "suggested_next_tool": "moveit_plan_manipulation_task",
    }


@pytest.mark.asyncio
async def test_graph_executes_typed_place_release_contract() -> None:
    class PlaceTaskContractBridge(FakeBridge):
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
                    "name": "moveit_release_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
                {
                    "type": "function",
                    "name": "moveit_verify_released_object",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            if name == "moveit_get_current_pose":
                return await FakeBridge.call_tool(self, name, arguments)
            self.calls.append((name, arguments))
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "robot": "UR10",
                            "feedback": {"can_execute": True},
                            "raw": {"plan_name": arguments["plan_name"]},
                        }
                    }
                )
            if name == "moveit_release_object":
                return json.dumps({"structured_content": {"ok": True}})
            if name == "moveit_verify_released_object":
                return json.dumps(
                    {
                        "structured_content": {
                            "ok": True,
                            "object_name": arguments["object_name"],
                            "verification": {"result": "pass"},
                            "raw": {
                                "object_name": arguments["object_name"],
                                "planning_scene_state": "released",
                                "mcp_attached_object": None,
                            },
                        }
                    }
                )
            return json.dumps({"structured_content": {"ok": True}})

    raw = {
        "task_solution_id": "place_task_dynamic_5_001",
        "task_kind": "place",
        "object_name": "dynamic_5",
        "scene_snapshot_id": "scene_20260515_001",
        "waypoints": [
            {
                "position": {"x": 0.50, "y": 0.10, "z": 0.32},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        ],
        "execution_contract": contract_with_proof({
            "steps": [
                {
                    "handler": "motion",
                    "name": "place",
                    "waypoint_index": 0,
                    "source_stage": "place",
                    "required_proof": "emulated_motion_plan",
                },
                {
                    "handler": "open_gripper",
                    "name": "open_gripper",
                    "source_stage": "open_gripper",
                    "required_proof": "verified_gripper_open",
                },
                {
                    "handler": "release_object",
                    "name": "release_object",
                    "tool": "moveit_release_object",
                    "source_stage": "release_object",
                    "required_proof": "planning_scene_update",
                    "arguments": {
                        "object_name": "dynamic_5",
                        "object_pose": {
                            "position": {"x": 0.58, "y": 0.10, "z": 0.32},
                            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                        },
                    },
                },
                {
                    "handler": "verify_released_object",
                    "name": "verify_released_object",
                    "tool": "moveit_verify_released_object",
                    "source_stage": "verify_released_object",
                    "required_proof": "release_check",
                    "arguments": {"object_name": "dynamic_5"},
                },
            ],
        }),
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "place_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified place task executed."),
        ],
        bridge=PlaceTaskContractBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id="place_task_dynamic_5_001",
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the place task"))

    assert text in EXECUTION_COMPLETE_REPLIES
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert len(verified_plan_names) == 1
    assert verified_plan_names[0].startswith("place_task_dynamic_5_001_place_")
    assert verified_client.gripper_calls == [("UR10", "open", 9.0)]
    assert ("moveit_release_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "object_pose": {
            "position": {"x": 0.58, "y": 0.10, "z": 0.32},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "verified_gripper_open": True,
    }) in fixture.bridge.calls
    assert ("moveit_verify_released_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "timeout_s": 9.0,
    }) in fixture.bridge.calls
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["task_solution_id"] == "place_task_dynamic_5_001"
    assert output["structured_content"]["verified_plan_names"] == verified_plan_names


@pytest.mark.asyncio
async def test_graph_execute_task_plan_executes_logged_place_contract_shape() -> None:
    task_solution_id = "place_task_dynamic_5_002"
    scene_snapshot_id = "scene_20260515_001"
    release_object_pose = {
        "position": {"x": 0.58, "y": 0.10, "z": 0.32},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    raw = {
        "task_solution_id": task_solution_id,
        "task_kind": "place",
        "backend": "emulated",
        "object_name": "dynamic_5",
        "robot_name": "UR10",
        "created_from_tool": "moveit_plan_place_task",
        "scene_snapshot_id": scene_snapshot_id,
        "target_object_pose": release_object_pose,
        "release_tcp_pose": {
            "position": {"x": 0.58, "y": 0.10, "z": 0.33},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        "waypoints": [
            {
                "position": {"x": 0.58, "y": 0.10, "z": 0.40},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            {
                "position": {"x": 0.58, "y": 0.10, "z": 0.33},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
            {
                "position": {"x": 0.58, "y": 0.10, "z": 0.43},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        ],
        "workflow_steps": [
            {"name": "carry_approach", "kind": "motion", "waypoint_index": 0},
            {"name": "release_pose", "kind": "motion", "waypoint_index": 1},
            {"name": "open_gripper", "kind": "gripper", "tool": "moveit_open_gripper"},
            {"name": "detach_object", "kind": "scene", "object_name": "dynamic_5"},
            {"name": "retreat", "kind": "motion", "waypoint_index": 2},
        ],
        "release_after_execute": {
            "object_name": "dynamic_5",
            "object_pose": release_object_pose,
        },
        "execution_contract": {
            "target_kind": "task_solution",
            "task_solution_id": task_solution_id,
            "object_name": "dynamic_5",
            "scene_snapshot_id": scene_snapshot_id,
            "requires_explicit_approval": True,
            "can_execute": True,
            "steps": [
                {
                    "step": 1,
                    "handler": "motion",
                    "name": "release_pose",
                    "waypoint_index": 1,
                    "source_stage": "approach_place",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                },
                {
                    "step": 2,
                    "handler": "open_gripper",
                    "name": "open_gripper",
                    "tool": "moveit_open_gripper",
                    "source_stage": "open_gripper",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "verified_gripper_open",
                },
                {
                    "step": 3,
                    "handler": "release_object",
                    "name": "release_object",
                    "tool": "moveit_release_object",
                    "source_stage": "detach_object",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "planning_scene_update",
                    "arguments": {"object_name": "dynamic_5", "object_pose": release_object_pose},
                },
                {
                    "step": 4,
                    "handler": "motion",
                    "name": "retreat",
                    "waypoint_index": 2,
                    "source_stage": "retreat",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                },
                {
                    "step": 5,
                    "handler": "verify_released_object",
                    "name": "verify_released_object",
                    "tool": "moveit_verify_released_object",
                    "source_stage": "verify_released_object",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "release_check",
                    "arguments": {"object_name": "dynamic_5"},
                },
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified place task executed."),
        ],
        bridge=CompoundTaskPlanBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert [call[0] for call in verified_client.calls] == ["UR10", "UR10"]
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert verified_plan_names[0].startswith("place_task_dynamic_5_002_release_pose_")
    assert verified_plan_names[1].startswith("place_task_dynamic_5_002_retreat_")
    assert verified_client.gripper_calls == [("UR10", "open", 9.0)]
    assert ("moveit_release_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "object_pose": release_object_pose,
        "verified_gripper_open": True,
    }) in fixture.bridge.calls
    assert ("moveit_verify_released_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "timeout_s": 9.0,
    }) in fixture.bridge.calls
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["tool"] == "moveit_execute_task_plan"
    assert output["structured_content"]["task_solution_id"] == task_solution_id
    assert output["structured_content"]["verified_plan_names"] == verified_plan_names
    assert output["structured_content"]["release_verification"] == {"result": "pass"}


@pytest.mark.asyncio
async def test_graph_executes_long_hybrid_contract_from_cached_task_plan() -> None:
    class HybridTaskPlanBridge(CompoundTaskPlanBridge):
        def function_tools(self) -> list[dict[str, Any]]:
            return [
                *super().function_tools(),
                {
                    "type": "function",
                    "name": "moveit_plan_manipulation_task",
                    "parameters": {"type": "object"},
                    "strict": None,
                },
            ]

    task_solution_id = "hybrid_pick_place_dynamic_5_001"
    scene_snapshot_id = "scene_20260515_001"
    release_object_pose = {
        "position": {"x": 0.68, "y": 0.18, "z": 0.30},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    pick_contact_allowance = {
        "category": "gripper_touch_links_to_target",
        "object_name": "dynamic_5",
        "pairs": [["tool0", "dynamic_5"]],
    }
    place_contact_allowance = {
        "category": "held_object_to_world",
        "object_name": "dynamic_5",
        "pairs": [["dynamic_5", "ground_plane"]],
    }
    waypoints = [
        {
            "position": {"x": 0.40, "y": 0.10, "z": 0.36},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.46, "y": 0.10, "z": 0.32},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.46, "y": 0.10, "z": 0.44},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.62, "y": 0.18, "z": 0.44},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.68, "y": 0.18, "z": 0.30},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        {
            "position": {"x": 0.62, "y": 0.18, "z": 0.46},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    ]
    raw = {
        "task_solution_id": task_solution_id,
        "task_kind": "pick_place",
        "backend": "staged_moveit",
        "object_name": "dynamic_5",
        "robot_name": "UR10",
        "created_from_tool": "moveit_plan_manipulation_task",
        "scene_snapshot_id": scene_snapshot_id,
        "waypoints": waypoints,
        "execution_contract": {
            "steps": [
                {
                    "handler": "motion",
                    "name": "connect_to_pre_grasp",
                    "waypoint_index": 0,
                    "source_stage": "connect_to_pre_grasp",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "free_motion",
                },
                {
                    "handler": "motion",
                    "name": "approach_to_pre_grasp",
                    "waypoint_index": 1,
                    "source_stage": "approach_to_pre_grasp",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "cartesian",
                    "contact_allowance": pick_contact_allowance,
                },
                {
                    "handler": "close_gripper",
                    "name": "close_gripper",
                    "source_stage": "close_gripper",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "verified_gripper_closed",
                },
                {
                    "handler": "attach_object",
                    "name": "attach_object",
                    "source_stage": "attach_object",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "planning_scene_attached",
                },
                {
                    "handler": "motion",
                    "name": "post_grasp_lift",
                    "waypoint_index": 2,
                    "source_stage": "post_grasp_lift",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "cartesian",
                },
                {
                    "handler": "motion",
                    "name": "connect_to_place",
                    "waypoint_index": 3,
                    "source_stage": "connect_to_place",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "free_motion",
                },
                {
                    "handler": "motion",
                    "name": "approach_place",
                    "waypoint_index": 4,
                    "source_stage": "approach_place",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "cartesian",
                    "contact_allowance": place_contact_allowance,
                },
                {
                    "handler": "open_gripper",
                    "name": "open_gripper",
                    "source_stage": "open_gripper",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "verified_gripper_open",
                },
                {
                    "handler": "release_object",
                    "name": "release_object",
                    "tool": "moveit_release_object",
                    "source_stage": "release_object",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "planning_scene_update",
                    "arguments": {"object_name": "dynamic_5", "object_pose": release_object_pose},
                },
                {
                    "handler": "motion",
                    "name": "retreat",
                    "waypoint_index": 5,
                    "source_stage": "retreat",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "emulated_motion_plan",
                    "planner": "cartesian",
                },
                {
                    "handler": "verify_released_object",
                    "name": "verify_released_object",
                    "tool": "moveit_verify_released_object",
                    "source_stage": "verify_released_object",
                    "object_name": "dynamic_5",
                    "scene_snapshot_id": scene_snapshot_id,
                    "required_proof": "release_check",
                    "arguments": {"object_name": "dynamic_5"},
                },
            ]
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    bridge = HybridTaskPlanBridge(release_proof=True)
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified hybrid task executed."),
        ],
        bridge=bridge,
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="pick_place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the cached hybrid task"))

    assert text in EXECUTION_COMPLETE_REPLIES
    tool_names = {tool["function"]["name"] for tool in fixture.model.bound_tools}
    assert "moveit_plan_manipulation_task" in tool_names
    assert "moveit_plan_free_motion" not in tool_names
    assert "moveit_plan_cartesian_motion" not in tool_names
    assert "moveit_plan_pick" not in tool_names
    assert "moveit_plan_place" not in tool_names
    assert all(name != "moveit_plan_manipulation_task" for name, _ in bridge.calls)
    planning_calls = [
        call
        for call in bridge.calls
        if call[0] in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}
    ]
    assert [call[0] for call in planning_calls] == [
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_free_motion",
        "moveit_plan_cartesian_motion",
        "moveit_plan_cartesian_motion",
    ]
    expected_motion_stages = [
        "connect_to_pre_grasp",
        "approach_to_pre_grasp",
        "post_grasp_lift",
        "connect_to_place",
        "approach_place",
        "retreat",
    ]
    assert [
        str(arguments["plan_name"]).startswith(f"{task_solution_id}_{stage}_")
        for (_tool_name, arguments), stage in zip(planning_calls, expected_motion_stages)
    ] == [True, True, True, True, True, True]
    approach_call = next(
        call
        for call in planning_calls
        if str(call[1]["plan_name"]).startswith(
            f"{task_solution_id}_approach_to_pre_grasp_"
        )
    )
    assert len(approach_call[1]["waypoints"]) == 1
    assert approach_call[1]["waypoints"][0] == waypoints[1]
    assert approach_call[1]["contact_allowance"] == pick_contact_allowance
    place_approach_call = next(
        call
        for call in planning_calls
        if str(call[1]["plan_name"]).startswith(f"{task_solution_id}_approach_place_")
    )
    assert place_approach_call[1]["contact_allowance"] == place_contact_allowance
    assert [call[1] for call in verified_client.calls] == [
        arguments["plan_name"] for _tool_name, arguments in planning_calls
    ]
    assert verified_client.gripper_calls == [("UR10", "close", 9.0), ("UR10", "open", 9.0)]
    assert ("moveit_release_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "object_pose": release_object_pose,
        "verified_gripper_open": True,
    }) in bridge.calls
    assert ("moveit_verify_released_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "timeout_s": 9.0,
    }) in bridge.calls
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["task_solution_id"] == task_solution_id
    assert output["structured_content"]["release_verification"] == {"result": "pass"}


@pytest.mark.asyncio
async def test_graph_executes_cached_release_contract_with_hidden_internal_tools() -> None:
    task_solution_id = "place_task_dynamic_5_hidden_internal"
    release_object_pose = {
        "position": {"x": 0.58, "y": 0.10, "z": 0.32},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract={
            "goal": "place",
            "stages": [
                {"kind": "motion", "intent": "place_motion", "name": "place", "waypoint_index": 1},
                {"kind": "gripper", "intent": "verified_open", "name": "open_gripper"},
                {
                    "kind": "scene",
                    "intent": "release_detach",
                    "name": "release_object",
                    "tool": "moveit_release_object",
                    "arguments": {"object_pose": release_object_pose},
                },
                {"kind": "verify", "intent": "release_proof", "name": "verify_release"},
            ],
        },
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    bridge = HiddenContractToolBridge(release_proof=True)
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified place task executed."),
        ],
        bridge=bridge,
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    assert "moveit_release_object" not in {
        str(tool["name"]) for tool in bridge.function_tools()
    }

    text = await fixture.graph.run_turn(turn("yes, execute the place task"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert ("moveit_release_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "object_pose": release_object_pose,
        "verified_gripper_open": True,
    }) in bridge.contract_calls
    assert ("moveit_verify_released_object", {
        "robot_name": "UR10",
        "object_name": "dynamic_5",
        "timeout_s": 9.0,
    }) in bridge.contract_calls
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["release_verification"] == {"result": "pass"}


@pytest.mark.asyncio
async def test_graph_rejects_cached_contract_tool_missing_from_contract_capabilities() -> None:
    task_solution_id = "place_task_dynamic_5_unsupported_internal"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract={
            "goal": "place",
            "stages": [
                {"kind": "gripper", "intent": "verified_open", "name": "open_gripper"},
                {
                    "handler": "release_object",
                    "kind": "scene",
                    "intent": "release_detach",
                    "name": "release_object",
                    "tool": "moveit_release_object_v2",
                },
            ],
        },
    )
    bridge = HiddenContractToolBridge(release_proof=True)
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {
                    "robot_name": "UR10",
                    "task_solution_id": task_solution_id,
                    "timeout_s": 9.0,
                },
            ),
            ai_text("I need a supported release contract."),
        ],
        bridge=bridge,
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("yes, execute the place task"))

    assert bridge.contract_calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan release_object requires unavailable tool: moveit_release_object_v2.",
        "correction": "Expose the release/detach MCP tool in the bridge, then replan.",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_graph_execute_task_plan_updates_physical_pose_after_release_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_control.langgraph_robot_agent as agent_module

    release_pose = {
        "position": {"x": 0.58, "y": 0.10, "z": 0.32},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    calls: list[dict[str, Any]] = []

    def update_physical_model_pose_for_test(
        object_name: str,
        reason: str,
        pose_evidence: dict[str, object],
    ) -> dict[str, object]:
        calls.append(
            {
                "object_name": object_name,
                "reason": reason,
                "pose_evidence": pose_evidence,
            }
        )
        return {"ok": True, "object_name": object_name, "reason": reason}

    monkeypatch.setattr(
        agent_module,
        "update_physical_model_pose",
        update_physical_model_pose_for_test,
        raising=False,
    )

    class ReleaseProofPoseBridge(CompoundTaskPlanBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            output = await super().call_tool(name, arguments)
            if name != "moveit_verify_released_object":
                return output
            payload = json.loads(output)
            payload["structured_content"]["raw"]["object_pose"] = release_pose
            return json.dumps(payload)

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {"robot_name": "UR10", "task_solution_id": task_solution_id, "timeout_s": 9.0},
            ),
            ai_text("Verified place task executed."),
        ],
        bridge=ReleaseProofPoseBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert calls == [
        {
            "object_name": "dynamic_5",
            "reason": "verified_place_release",
            "pose_evidence": {
                "object_name": "dynamic_5",
                "source": "moveit_verify_released_object",
                "pose": release_pose,
            },
        }
    ]
    assert all(name != "moveit_get_object_context" for name, _ in fixture.bridge.calls)
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["physical_model_update"] == {
        "ok": True,
        "object_name": "dynamic_5",
        "reason": "verified_place_release",
    }


@pytest.mark.asyncio
async def test_graph_execute_task_plan_keeps_execution_success_when_pose_update_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_control.langgraph_robot_agent as agent_module

    release_pose = {
        "position": {"x": 0.58, "y": 0.10, "z": 0.32},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }

    def failing_pose_update(
        object_name: str,
        reason: str,
        pose_evidence: dict[str, object],
    ) -> dict[str, object]:
        return {
            "ok": False,
            "object_name": object_name,
            "reason": reason,
            "error": "physical model write failed",
            "retryable": True,
        }

    monkeypatch.setattr(
        agent_module,
        "update_physical_model_pose",
        failing_pose_update,
        raising=False,
    )

    class ReleaseProofPoseBridge(CompoundTaskPlanBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            output = await super().call_tool(name, arguments)
            if name != "moveit_verify_released_object":
                return output
            payload = json.loads(output)
            payload["structured_content"]["raw"]["pose"] = release_pose
            return json.dumps(payload)

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {"robot_name": "UR10", "task_solution_id": task_solution_id, "timeout_s": 9.0},
            ),
            ai_text("Verified place task executed."),
        ],
        bridge=ReleaseProofPoseBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text in EXECUTION_COMPLETE_REPLIES
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["verification"] == {"result": "pass"}
    assert output["structured_content"]["physical_model_update"] == {
        "ok": False,
        "object_name": "dynamic_5",
        "reason": "verified_place_release",
        "error": "physical model write failed",
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_graph_execute_task_plan_does_not_sync_pose_without_release_pose_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_control.langgraph_robot_agent as agent_module

    calls: list[dict[str, Any]] = []

    def update_physical_model_pose_for_test(
        object_name: str,
        reason: str,
        pose_evidence: dict[str, object],
    ) -> dict[str, object]:
        calls.append(
            {
                "object_name": object_name,
                "reason": reason,
                "pose_evidence": pose_evidence,
            }
        )
        return {"ok": True}

    monkeypatch.setattr(
        agent_module,
        "update_physical_model_pose",
        update_physical_model_pose_for_test,
        raising=False,
    )

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {"robot_name": "UR10", "task_solution_id": task_solution_id, "timeout_s": 9.0},
            ),
            ai_text("Verified place task executed."),
        ],
        bridge=CompoundTaskPlanBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    text = await fixture.graph.run_turn(turn("yes, execute"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert calls == []
    assert all(name != "moveit_get_object_context" for name, _ in fixture.bridge.calls)
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["physical_model_update"] == {
        "ok": False,
        "error": "Full object pose evidence was not found in verified release proof.",
        "correction": (
            "Use release proof with object position and orientation, "
            "or run an explicit operator sync."
        ),
        "retryable": True,
    }


@pytest.mark.asyncio
async def test_graph_execute_task_plan_rejects_release_proof_when_object_still_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_control.langgraph_robot_agent as agent_module

    calls: list[dict[str, object]] = []

    def update_physical_model_pose_for_test(
        object_name: str,
        reason: str,
        pose_evidence: dict[str, object],
    ) -> dict[str, object]:
        calls.append({"object_name": object_name, "reason": reason, "pose_evidence": pose_evidence})
        return {"ok": True}

    monkeypatch.setattr(
        agent_module,
        "update_physical_model_pose",
        update_physical_model_pose_for_test,
        raising=False,
    )

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {"robot_name": "UR10", "task_solution_id": task_solution_id, "timeout_s": 9.0},
            ),
            ai_text("Release proof failed."),
        ],
        bridge=CompoundTaskPlanBridge(release_proof=False),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("yes, execute"))

    assert calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["error"] == "Task plan verify_released_object failed at verify_release."
    assert output["failed_tool_name"] == "moveit_verify_released_object"
    failed_result = output["failed_tool_result"]
    assert failed_result["structured_content"]["raw"]["mcp_attached_object"] == "dynamic_5"
    assert "physical_model_update" not in output


@pytest.mark.asyncio
async def test_graph_execute_task_plan_rejects_release_proof_when_gripper_free_but_scene_attached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import agent_control.langgraph_robot_agent as agent_module

    calls: list[dict[str, object]] = []

    def update_physical_model_pose_for_test(
        object_name: str,
        reason: str,
        pose_evidence: dict[str, object],
    ) -> dict[str, object]:
        calls.append({"object_name": object_name, "reason": reason, "pose_evidence": pose_evidence})
        return {"ok": True}

    monkeypatch.setattr(
        agent_module,
        "update_physical_model_pose",
        update_physical_model_pose_for_test,
        raising=False,
    )

    class InconsistentReleaseProofBridge(CompoundTaskPlanBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            output = await super().call_tool(name, arguments)
            if name != "moveit_verify_released_object":
                return output
            payload = json.loads(output)
            payload["structured_content"]["raw"].update(
                {
                    "mcp_attached_object": arguments["object_name"],
                    "mcp_gripper_holds_object": False,
                    "planning_scene_state": "attached",
                }
            )
            return json.dumps(payload)

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    fixture = make_graph(
        [
            ai_tool_call(
                "moveit_execute_task_plan",
                {"robot_name": "UR10", "task_solution_id": task_solution_id, "timeout_s": 9.0},
            ),
            ai_text("Release proof failed."),
        ],
        bridge=InconsistentReleaseProofBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    await fixture.graph.run_turn(turn("yes, execute"))

    assert calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output["ok"] is False
    assert output["error"] == "Task plan verify_released_object failed at verify_release."
    assert output["failed_tool_name"] == "moveit_verify_released_object"
    failed_result = output["failed_tool_result"]
    assert failed_result["structured_content"]["raw"]["mcp_attached_object"] == "dynamic_5"
    assert failed_result["structured_content"]["raw"]["mcp_gripper_holds_object"] is False
    assert failed_result["structured_content"]["raw"]["planning_scene_state"] == "attached"


@pytest.mark.asyncio
async def test_graph_feeds_place_retreat_failure_back_for_explanation() -> None:
    class RetreatFailureBridge(CompoundTaskPlanBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            if name in {"moveit_plan_free_motion", "moveit_plan_cartesian_motion"}:
                self.calls.append((name, arguments))
                plan_name = str(arguments.get("plan_name") or "")
                if "_retreat_" in plan_name:
                    return json.dumps(
                        {
                            "structured_content": {
                                "ok": False,
                                "robot": "UR10",
                                "tool": "plan_cartesian_motion",
                                "feedback": {
                                    "status": "incomplete path",
                                    "can_execute": False,
                                    "correction": (
                                        "Replan with a smaller or safer target, then execute "
                                        "only a successful returned raw.plan_name."
                                    ),
                                },
                                "verification": {"result": "fail"},
                                "raw": {"plan_name": plan_name, "trajectory_points": 1},
                            }
                        }
                    )
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
            return await super().call_tool(name, arguments)

    task_solution_id = "place_task_dynamic_5_002"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="place",
        execution_contract=place_execution_contract(),
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("unexpected model explanation"),
        ],
        bridge=RetreatFailureBridge(release_proof=True),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="place",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=FakeVerifiedExecutionClient(),
    )

    text = await fixture.graph.run_turn(turn("execute"))

    assert (
        text
        == "I could not finish the task because MoveIt could not find a complete path "
        "for retreat. Completed before the failure: place, open gripper, release object. "
        "Please approve the next action before I retry or replan."
    )
    assert len(fixture.model.requests) == 1
    task_failure = json.loads(latest_state_tool_content(fixture))
    assert task_failure["suggested_next_tool"] == "moveit_explain_motion_failure"
    assert task_failure["failed_tool_name"] == "moveit_plan_cartesian_motion"
    assert "_retreat_" in task_failure["failed_tool_arguments"]["plan_name"]
    assert task_failure["failed_tool_result"]["structured_content"]["ok"] is False
    recovery = task_failure["recovery"]
    assert recovery["task_solution_id"] == task_solution_id
    assert recovery["object_name"] == "dynamic_5"
    assert recovery["failed_step"] == "retreat"
    assert recovery["gripper_state"] == "open"
    assert recovery["completed_steps"][-1]["handler"] == "release_object"
    assert recovery["verified_plan_names"][0].startswith("place_task_dynamic_5_002_")
    assert "recent task failure: place_task_dynamic_5_002" in fixture.graph._robot_context.render_instruction_block()
    assert "unexpected model explanation" not in text
    assert "moveit_explain_motion_failure" in bridge_tool_names(fixture.bridge)


@pytest.mark.asyncio
async def test_graph_executes_hold_contract_and_keeps_held_object_context() -> None:
    task_solution_id = "hold_task_dynamic_5_001"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="hold",
        execution_contract=hold_execution_contract(),
    )
    context = approved_contract_task_context(
        task_solution_id=task_solution_id,
        task_kind="hold",
        object_name="dynamic_5",
        raw=raw,
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified hold task executed."),
        ],
        bridge=CompoundTaskPlanBridge(),
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, pick it up and hold it"))

    assert text in EXECUTION_COMPLETE_REPLIES
    verified_plan_names = [call[1] for call in verified_client.calls]
    assert len(verified_plan_names) == 3
    assert verified_client.gripper_calls == [("UR10", "close", 9.0)]
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["verification"] == {"result": "pass"}
    assert "held object: dynamic_5" in context.render_instruction_block()


@pytest.mark.asyncio
async def test_graph_executes_zero_lift_hold_without_post_grasp_lift() -> None:
    task_solution_id = "hold_task_dynamic_5_zero_lift"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="hold",
        execution_contract=bare_hold_execution_contract(),
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    bridge = StageSynchronizedTaskBridge()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=bridge,
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="hold",
            object_name="dynamic_5",
            raw=raw,
        ),
    )

    text = await fixture.graph.run_turn(turn("yes, hold it"))

    assert text in PHYSICAL_STATUS_UNAVAILABLE_REPLIES
    names = bridge_tool_names(bridge)
    assert_subsequence(
        names,
        [
            "moveit_plan_free_motion",
            "moveit_execute_plan",
            "moveit_plan_cartesian_motion",
            "moveit_execute_plan",
            "moveit_close_gripper",
            "moveit_attach_object",
            "moveit_verify_attached_object",
        ],
    )
    assert names.count("moveit_plan_cartesian_motion") == 1
    assert not any("_post_grasp_lift_" in str(arguments.get("plan_name", "")) for _name, arguments in bridge.calls)
    output = json.loads(latest_state_tool_content(fixture))
    completed = output["structured_content"]["simulation"]["completed_steps"]
    assert [step["name"] for step in completed] == [
        "connect_to_pre_grasp",
        "approach_to_pre_grasp",
        "close_gripper",
        "attach_object",
        "verify_attached_object",
    ]


@pytest.mark.asyncio
async def test_graph_executes_move_contract_from_raw_target_pose() -> None:
    task_solution_id = "move_task_tcp_001"
    target_pose = {
        "position": {"x": 0.50, "y": 0.10, "z": 0.52},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }
    raw = {
        "task_solution_id": task_solution_id,
        "task_kind": "move",
        "object_name": "tcp",
        "scene_snapshot_id": "scene_20260515_001",
        "target_pose": target_pose,
        "execution_contract": {
            "steps": [
                {
                    "step": 1,
                    "handler": "motion",
                    "name": "move_tcp",
                    "plan_handle": "move_preview_tcp",
                    "source_stage": "move_tcp",
                    "required_proof": "verified_motion_plan",
                },
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("unexpected model fallback"),
        ],
        bridge=CompoundTaskPlanBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id=task_solution_id,
            task_kind="move",
            object_name="tcp",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, lift up by 10cm"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert len(verified_client.calls) == 1
    assert verified_client.calls[0][0] == "UR10"
    assert verified_client.calls[0][1].startswith("move_task_tcp_001_move_tcp_")
    assert verified_client.calls[0][1].endswith("_try1")
    assert verified_client.calls[0][2] == 9.0
    planning_calls = [
        arguments
        for name, arguments in fixture.bridge.calls
        if name == "moveit_plan_cartesian_motion"
    ]
    assert planning_calls[0]["waypoints"] == [target_pose]
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True


@pytest.mark.asyncio
async def test_graph_executes_move_and_release_contract_and_clears_held_object() -> None:
    task_solution_id = "move_release_task_dynamic_5_001"
    raw = compound_task_raw(
        task_solution_id=task_solution_id,
        task_kind="move_and_release",
        execution_contract={
            "goal": "move_and_release",
                "stages": [
                    {
                        "kind": "motion",
                        "intent": "move_held_object",
                        "name": "move",
                        "waypoint_index": 0,
                    },
                    {"kind": "gripper", "intent": "verified_open", "name": "open_gripper"},
                    {
                        "kind": "scene",
                        "intent": "release_detach",
                        "name": "release_object",
                        "arguments": {
                            "object_pose": {
                                "position": {"x": 0.58, "y": 0.10, "z": 0.32},
                                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                            },
                        },
                    },
                    {"kind": "verify", "intent": "release_proof", "name": "verify_release"},
                ],
            },
    )
    context = approved_contract_task_context(
        task_solution_id=task_solution_id,
        task_kind="move_and_release",
        object_name="dynamic_5",
        raw=raw,
    )
    context.update_from_tool_result(
        "moveit_verify_attached_object",
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "raw": {
                        "object_name": "dynamic_5",
                        "mcp_attached_object": "dynamic_5",
                        "mcp_gripper_holds_object": True,
                        "planning_scene_state": "attached",
                    },
                }
            }
        ),
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": task_solution_id,
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Verified move-and-release task executed."),
        ],
        bridge=CompoundTaskPlanBridge(release_proof=True),
        robot_context=context,
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, move it left and release it"))

    assert text in EXECUTION_COMPLETE_REPLIES
    assert len(verified_client.calls) == 1
    assert verified_client.calls[0][1].startswith("move_release_task_dynamic_5_001_move_")
    assert verified_client.gripper_calls == [("UR10", "open", 9.0)]
    output = json.loads(latest_state_tool_content(fixture))
    assert output["structured_content"]["ok"] is True
    assert output["structured_content"]["release_verification"] == {"result": "pass"}
    assert "held object: dynamic_5" not in context.render_instruction_block()


@pytest.mark.asyncio
async def test_graph_rejects_unknown_contract_handler_without_verified_calls() -> None:
    raw = {
        "task_solution_id": "pick_task_dynamic_5_001",
        "task_kind": "pick",
            "object_name": "dynamic_5",
            "scene_snapshot_id": "scene_20260515_001",
            "execution_contract": {
                "steps": [
                {
                    "handler": "push_object",
                    "name": "push_object",
                    "source_stage": "mtc_push",
                    "required_proof": "push_object",
                },
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("I need a supported task contract."),
        ],
        bridge=TaskExecutionBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id="pick_task_dynamic_5_001",
            task_kind="pick",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    text = await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert text == "I need a supported task contract."
    assert len(fixture.model.requests) == 2
    assert "moveit_execute_task_plan" not in text
    assert verified_client.calls == []
    assert verified_client.gripper_calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan workflow contains an unsupported step handler: push_object.",
        "correction": "Plan a supported pick/place task again, then retry moveit_execute_task_plan.",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_graph_rejects_contract_step_missing_proof_fields() -> None:
    raw = {
        "task_solution_id": "pick_task_dynamic_5_001",
        "task_kind": "pick",
        "object_name": "dynamic_5",
        "scene_snapshot_id": "scene_20260515_001",
        "waypoints": [
            {
                "position": {"x": 0.50, "y": 0.10, "z": 0.40},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
        ],
        "execution_contract": {
            "steps": [
                {"handler": "motion", "name": "approach", "waypoint_index": 0},
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "pick_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("I need a proof-backed task contract."),
        ],
        bridge=TaskExecutionBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id="pick_task_dynamic_5_001",
            task_kind="pick",
            object_name="dynamic_5",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the pick task"))

    assert verified_client.calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan execution_contract step is missing source_stage.",
        "correction": "Replan with a backend task solution that includes source stage and proof metadata.",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_graph_rejects_verified_motion_contract_without_plan_handle() -> None:
    raw = {
        "task_solution_id": "hold_task_dynamic_1_003",
        "task_kind": "hold",
        "object_name": "dynamic_1",
        "scene_snapshot_id": "scene_20260518_003",
        "execution_contract": {
            "steps": [
                {
                    "handler": "motion",
                    "name": "connect_to_pre_grasp",
                    "source_stage": "connect_to_pre_grasp",
                    "required_proof": "verified_motion_plan",
                    "waypoint_index": 0,
                },
            ],
        },
    }
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "hold_task_dynamic_1_003",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("I need a proof-backed task contract."),
        ],
        bridge=TaskExecutionBridge(),
        robot_context=approved_contract_task_context(
            task_solution_id="hold_task_dynamic_1_003",
            task_kind="hold",
            object_name="dynamic_1",
            raw=raw,
        ),
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the hold task"))

    assert verified_client.calls == []
    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan execution_contract motion step is missing plan_handle.",
        "correction": "Plan the task again so verified motion steps include a plan_handle.",
        "retryable": False,
    }


@pytest.mark.asyncio
async def test_graph_rejects_unknown_task_kind_execution() -> None:
    context = RobotContextStore(time_fn=lambda: 100.0)
    context.remember_task_solution(
        task_solution_id="slide_task_dynamic_5_001",
        task_kind="slide",
        object_name="dynamic_5",
        backend="emulated",
        scene_snapshot_id="scene_20260515_001",
        approval_required=True,
        raw={
            "task_solution_id": "slide_task_dynamic_5_001",
            "task_kind": "slide",
            "object_name": "dynamic_5",
            "execution_contract": {"steps": []},
        },
    )
    context.remember_task_solution_approval_candidate(
        target_kind="task_solution",
        task_solution_id="slide_task_dynamic_5_001",
        source_tool="moveit_plan_slide_task",
        object_name="dynamic_5",
        expected_movement="slide dynamic_5",
        scene_snapshot_id="scene_20260515_001",
    )
    assert context.record_task_solution_approval(
        "slide_task_dynamic_5_001",
        approval_turn_id="turn-approved",
        approved_at=100.0,
    )
    execute_args = {
        "robot_name": "UR10",
        "task_solution_id": "slide_task_dynamic_5_001",
        "timeout_s": 9.0,
    }
    verified_client = FakeVerifiedExecutionClient()
    fixture = make_graph(
        [
            ai_tool_call("moveit_execute_task_plan", execute_args),
            ai_text("Slide task execution is not supported."),
        ],
        robot_context=context,
        verified_execution_client=verified_client,
    )

    await fixture.graph.run_turn(turn("yes, execute the slide task"))

    output = json.loads(latest_state_tool_content(fixture))
    assert output == {
        "ok": False,
        "error": "Task plan execution does not support task kind: slide.",
        "correction": "Plan a supported pick/place task, then retry moveit_execute_task_plan.",
        "retryable": False,
    }
    assert verified_client.calls == []
    assert verified_client.gripper_calls == []


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

    assert text in EXECUTION_COMPLETE_REPLIES
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

    assert text in EXECUTION_COMPLETE_REPLIES
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

    assert text in EXECUTION_COMPLETE_REPLIES
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
