import json
import os
from dataclasses import dataclass
from typing import Any, cast

import pytest
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from robot_control.call_validation import agent_tool_description
from robot_control.context import RobotContextStore
from voice_runtime.agent_turn import AgentTurnInput
from voice_runtime.profiles import AgentProfile, ReasoningEffort

RUN_LIVE_DYNAMIC_5_PICK_DUMMY_E2E = "RUN_LIVE_DYNAMIC_5_PICK_DUMMY_E2E"
_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}


class FakeChatModel:
    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.requests: list[list[BaseMessage]] = []
        self.bound_tool_batches: list[list[dict[str, Any]]] = []

    def bind_tools(self, tools: list[dict[str, Any]], **kwargs: Any) -> "FakeBoundChatModel":
        self.bound_tool_batches.append(list(tools))
        return FakeBoundChatModel(self.responses, self.requests)


class FakeBoundChatModel:
    def __init__(self, responses: list[AIMessage], requests: list[list[BaseMessage]]):
        self.responses = responses
        self.requests = requests

    async def ainvoke(self, messages: list[BaseMessage]) -> AIMessage:
        self.requests.append(list(messages))
        try:
            return self.responses.pop(0)
        except IndexError as exc:
            raise AssertionError("fake model received an unexpected request") from exc


class PickTaskE2EBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def function_tools(self) -> list[dict[str, Any]]:
        return [
            _function_tool("moveit_get_current_pose"),
            _function_tool("moveit_plan_pick_task"),
            _function_tool("moveit_execute_task_plan"),
            _function_tool("moveit_execute_task_solution"),
            _function_tool("moveit_plan_free_motion"),
            _function_tool("moveit_plan_cartesian_motion"),
            _function_tool("moveit_close_gripper"),
            _function_tool("moveit_attach_object"),
            _function_tool("moveit_verify_attached_object"),
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
                                "position": {"x": 0.10, "y": 0.20, "z": 0.30},
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
        if name == "moveit_plan_pick_task":
            assert arguments["object_name"] == "dynamic_5"
            return _pick_task_solution_output()
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


@dataclass(frozen=True)
class GraphFixture:
    graph: Any
    model: FakeChatModel
    bridge: PickTaskE2EBridge
    robot_context: RobotContextStore
    verified_execution_client: FakeVerifiedExecutionClient


def make_graph(responses: list[AIMessage]) -> GraphFixture:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent

    model = FakeChatModel(responses)
    bridge = PickTaskE2EBridge()
    robot_context = RobotContextStore(time_fn=lambda: 100.0)
    verified_client = FakeVerifiedExecutionClient()
    graph = LangGraphRobotAgent(
        model=model,
        tool_bridge=bridge,
        robot_context=robot_context,
        thread_id="test-dynamic-5-pick",
        verified_execution_client=verified_client,
    )
    return GraphFixture(
        graph=graph,
        model=model,
        bridge=bridge,
        robot_context=robot_context,
        verified_execution_client=verified_client,
    )


def _function_tool(name: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": agent_tool_description(name),
        "parameters": {"type": "object"},
        "strict": None,
    }


def _turn(text: str) -> AgentTurnInput:
    return AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])


def _ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)


def _ai_tool_call(name: str, args: dict[str, Any], call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


def _last_tool_output(model: FakeChatModel) -> dict[str, Any]:
    message = model.requests[-1][-1]
    assert isinstance(message, ToolMessage)
    content = message.content
    assert isinstance(content, str)
    return json.loads(content)


def _pick_task_solution_output() -> str:
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
                    "scene_snapshot_id": "scene_dynamic_5_loaded",
                    "plan_name": "internal_approach_only_plan",
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
                        "expected_movement": (
                            "approach grasp, close gripper, attach object, lift object"
                        ),
                        "scene_snapshot_id": "scene_dynamic_5_loaded",
                    },
                },
            }
        }
    )


def _live_reasoning_effort_from_env() -> ReasoningEffort:
    raw = os.getenv("LIVE_DYNAMIC_5_PICK_REASONING_EFFORT", "high")
    if raw not in _REASONING_EFFORTS:
        raise ValueError(f"Unsupported LIVE_DYNAMIC_5_PICK_REASONING_EFFORT: {raw}")
    return cast(ReasoningEffort, raw)


def _live_agent_profile_from_env() -> AgentProfile:
    provider = os.getenv("LIVE_DYNAMIC_5_PICK_AGENT_PROVIDER", "gemini_api")
    if provider == "openai_api":
        return AgentProfile(
            provider="openai_api",
            model=os.getenv("LIVE_DYNAMIC_5_PICK_OPENAI_MODEL", "gpt-5.4-mini"),
            reasoning_effort=_live_reasoning_effort_from_env(),
            api_key_env=os.getenv("LIVE_DYNAMIC_5_PICK_OPENAI_KEY_ENV", "OPENAI_API_KEY"),
        )
    if provider == "anthropic_api":
        return AgentProfile(
            provider="anthropic_api",
            model=os.getenv("LIVE_DYNAMIC_5_PICK_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
            reasoning_effort=_live_reasoning_effort_from_env(),
            api_key_env=os.getenv("LIVE_DYNAMIC_5_PICK_ANTHROPIC_KEY_ENV", "ANTHROPIC_API_KEY"),
        )
    return AgentProfile(
        provider="gemini_api",
        model=os.getenv("LIVE_DYNAMIC_5_PICK_GEMINI_MODEL", "gemini-3.1-flash-lite-preview"),
        reasoning_effort=_live_reasoning_effort_from_env(),
        api_key_env=os.getenv("LIVE_DYNAMIC_5_PICK_GEMINI_KEY_ENV", "GOOGLE_API_KEY"),
    )


@pytest.mark.asyncio
async def test_dynamic_5_pick_task_plans_then_executes_verified_task_plan() -> None:
    fixture = make_graph(
        [
            _ai_tool_call(
                "moveit_plan_pick_task",
                {"robot_name": "UR10", "object_name": "dynamic_5", "timeout_s": 9.0},
                "plan-pick-task",
            ),
            _ai_text("Pick task planned for dynamic_5."),
            _ai_tool_call(
                "moveit_execute_task_plan",
                {
                    "robot_name": "UR10",
                    "task_solution_id": "pick_task_dynamic_5_001",
                    "timeout_s": 9.0,
                },
                "execute-task-plan",
            ),
            _ai_text("Verified pick task executed."),
        ]
    )

    planned_text = await fixture.graph.run_turn(_turn("pick up dynamic_5"))

    assert planned_text == "Pick task planned for dynamic_5."
    assert fixture.verified_execution_client.calls == []
    assert fixture.robot_context.pending_plan is None

    executed_text = await fixture.graph.run_turn(
        _turn("yes, execute the dynamic_5 pick task")
    )

    assert executed_text == "Verified pick task executed."
    tool_names = [name for name, _ in fixture.bridge.calls]
    assert "moveit_plan_pick_task" in tool_names
    assert "moveit_execute_task_plan" not in tool_names
    assert "moveit_execute_task_solution" not in tool_names
    assert "moveit_execute_plan" not in tool_names
    bound_tool_name_batches = [
        {tool["function"]["name"] for tool in batch}
        for batch in fixture.model.bound_tool_batches
    ]
    assert bound_tool_name_batches
    assert any(
        "moveit_execute_task_plan" in names
        for names in bound_tool_name_batches
    )
    assert all(
        "moveit_execute_task_solution" not in names
        for names in bound_tool_name_batches
    )
    assert [name for name in tool_names if name == "moveit_attach_object"] == [
        "moveit_attach_object"
    ]
    assert [name for name in tool_names if name == "moveit_verify_attached_object"] == [
        "moveit_verify_attached_object"
    ]

    verified_plan_names = [plan_name for _, plan_name, _ in fixture.verified_execution_client.calls]
    assert [robot_name for robot_name, _, _ in fixture.verified_execution_client.calls] == [
        "UR10",
        "UR10",
        "UR10",
    ]
    assert [timeout_s for _, _, timeout_s in fixture.verified_execution_client.calls] == [
        9.0,
        9.0,
        9.0,
    ]
    assert fixture.verified_execution_client.gripper_calls == [("UR10", "close", 9.0)]
    attach_calls = [
        args for name, args in fixture.bridge.calls if name == "moveit_attach_object"
    ]
    assert attach_calls == [
        {
            "robot_name": "UR10",
            "object_name": "dynamic_5",
            "verified_gripper_closed": True,
        }
    ]
    assert verified_plan_names[0].startswith("pick_task_dynamic_5_001_approach_")
    assert verified_plan_names[1].startswith("pick_task_dynamic_5_001_pre_grasp_")
    assert verified_plan_names[2].startswith("pick_task_dynamic_5_001_lift_")
    assert "internal_approach_only_plan" not in verified_plan_names
    assert fixture.robot_context.pending_plan is None

    output = _last_tool_output(fixture.model)
    assert output["structured_content"]["tool"] == "moveit_execute_task_plan"
    assert output["structured_content"]["object_name"] == "dynamic_5"
    assert output["structured_content"]["verified_plan_names"] == verified_plan_names


@pytest.mark.asyncio
@pytest.mark.live
@pytest.mark.llm
@pytest.mark.native_llm
@pytest.mark.skipif(
    os.getenv(RUN_LIVE_DYNAMIC_5_PICK_DUMMY_E2E) != "1",
    reason=f"set {RUN_LIVE_DYNAMIC_5_PICK_DUMMY_E2E}=1",
)
async def test_live_llm_dynamic_5_pick_task_uses_dummy_verified_execution() -> None:
    from agent_control.langgraph_robot_agent import LangGraphRobotAgent
    from agent_control.model_factory import build_agent_chat_model

    bridge = PickTaskE2EBridge()
    robot_context = RobotContextStore(time_fn=lambda: 100.0)
    verified_client = FakeVerifiedExecutionClient()
    graph = LangGraphRobotAgent(
        model=build_agent_chat_model(_live_agent_profile_from_env()),
        tool_bridge=bridge,
        robot_context=robot_context,
        thread_id="live-test-dynamic-5-pick",
        verified_execution_client=verified_client,
    )

    await graph.run_turn(_turn("pick up dynamic_5"))
    await graph.run_turn(_turn("yes, execute the planned dynamic_5 pick task"))

    tool_names = [name for name, _ in bridge.calls]
    pick_calls = [args for name, args in bridge.calls if name == "moveit_plan_pick_task"]
    assert pick_calls
    assert pick_calls[-1]["object_name"] == "dynamic_5"
    assert "moveit_execute_task_solution" not in tool_names
    assert "moveit_execute_plan" not in tool_names
    assert "moveit_close_gripper" not in tool_names
    assert verified_client.gripper_calls

    verified_plan_names = [plan_name for _, plan_name, _ in verified_client.calls]
    assert len(verified_plan_names) == 3
    assert verified_plan_names[0].startswith("pick_task_dynamic_5_001_approach_")
    assert verified_plan_names[1].startswith("pick_task_dynamic_5_001_pre_grasp_")
    assert verified_plan_names[2].startswith("pick_task_dynamic_5_001_lift_")
    assert "internal_approach_only_plan" not in verified_plan_names
