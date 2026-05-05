import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from codex_auth import CodexAuthError, CodexCredentials
from codex_backend_client import CodexResponseResult, CodexToolCall
from robot_control.context import RobotContextStore
from voice_runtime.agent_turn import AgentTurnInput


def test_langgraph_dependency_is_available() -> None:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph

    assert InMemorySaver is not None
    assert StateGraph is not None
    assert START != END


class FakeStore:
    def get_credentials(self) -> CodexCredentials:
        return CodexCredentials(access="access", refresh="refresh", account_id="acct")


class AuthErrorStore:
    def get_credentials(self) -> CodexCredentials:
        raise CodexAuthError("login required")


class FakeBackend:
    def __init__(self, results: list[CodexResponseResult]):
        self.results = list(results)
        self.requests: list[dict[str, Any]] = []

    async def create_response(
        self,
        credentials: CodexCredentials,
        *,
        model: str,
        instructions: str,
        input_items: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> CodexResponseResult:
        self.requests.append(
            {
                "credentials": credentials,
                "model": model,
                "instructions": instructions,
                "input_items": list(input_items),
                "tools": list(tools),
            }
        )
        return self.results.pop(0)


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
    backend: FakeBackend
    bridge: FakeBridge


def make_graph(results: list[CodexResponseResult], *, bridge: FakeBridge | None = None) -> GraphFixture:
    from langgraph_robot_agent import LangGraphRobotAgent

    backend = FakeBackend(results)
    selected_bridge = bridge or FakeBridge()
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=FakeStore(),
        backend_client=backend,
        tool_bridge=selected_bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )
    return GraphFixture(graph=graph, backend=backend, bridge=selected_bridge)


def turn(text: str) -> AgentTurnInput:
    return AgentTurnInput(user_text=text, messages=[{"role": "user", "content": text}])


def tool_call(
    name: str,
    call_id: str = "call-1",
    item_id: str = "item-1",
    arguments: dict[str, Any] | None = None,
) -> CodexToolCall:
    arguments = arguments or {"robot_name": "UR10"}
    return CodexToolCall(
        call_id=call_id,
        item_id=item_id,
        name=name,
        arguments=arguments,
        raw_arguments=json.dumps(arguments),
    )


def output_item(
    name: str,
    call_id: str = "call-1",
    item_id: str = "item-1",
    arguments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    arguments = arguments or {"robot_name": "UR10"}
    return {
        "type": "function_call",
        "id": item_id,
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments),
    }


@pytest.mark.asyncio
async def test_graph_auth_error_does_not_observe_robot() -> None:
    from langgraph_robot_agent import LangGraphRobotAgent

    bridge = FakeBridge()
    graph = LangGraphRobotAgent(
        model="gpt-5.4-mini",
        credential_store=AuthErrorStore(),
        backend_client=FakeBackend([CodexResponseResult(text="should-not-run")]),
        tool_bridge=bridge,
        robot_context=RobotContextStore(),
        thread_id="test-session",
    )

    with pytest.raises(CodexAuthError):
        await graph.run_turn(turn("hello"))

    assert bridge.calls == []


@pytest.mark.asyncio
async def test_graph_observes_current_pose_before_simple_codex_response() -> None:
    fixture = make_graph([CodexResponseResult(text="oauth-ok")])

    text = await fixture.graph.run_turn(turn("hello"))

    assert text == "oauth-ok"
    assert fixture.bridge.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert fixture.backend.requests[0]["model"] == "gpt-5.4-mini"
    assert fixture.backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hello"}]}
    ]
    assert "Last-known robot context" in fixture.backend.requests[0]["instructions"]
    assert "robot: UR10" in fixture.backend.requests[0]["instructions"]


@pytest.mark.asyncio
async def test_graph_uses_easy_assistant_history_items_without_synthetic_output_ids() -> None:
    fixture = make_graph([CodexResponseResult(text="ok")])
    messages: list[Mapping[str, Any]] = [
        {"role": "user", "content": "move up"},
        {"role": "assistant", "content": "Moved up."},
        {"role": "user", "content": "again"},
    ]

    await fixture.graph.run_turn(AgentTurnInput(user_text="again", messages=messages))

    assert fixture.backend.requests[0]["input_items"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "move up"}]},
        {"role": "assistant", "content": "Moved up."},
        {"role": "user", "content": [{"type": "input_text", "text": "again"}]},
    ]
    assistant_item = fixture.backend.requests[0]["input_items"][1]
    assert "id" not in assistant_item
    assert "status" not in assistant_item


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

    fixture = make_graph([CodexResponseResult(text="ok")], bridge=LegacyStatusBridge())

    text = await fixture.graph.run_turn(turn("hello"))

    assert text == "ok"
    assert fixture.bridge.calls == []


@pytest.mark.asyncio
async def test_graph_sends_tool_output_back_to_codex() -> None:
    pose = tool_call("moveit_get_current_pose")
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[pose],
                output_items=[output_item("moveit_get_current_pose")],
            ),
            CodexResponseResult(text="Robot pose is ready."),
        ]
    )

    text = await fixture.graph.run_turn(turn("where is the pose?"))

    assert text == "Robot pose is ready."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    assert fixture.backend.requests[1]["input_items"][-1]["type"] == "function_call_output"
    assert fixture.backend.requests[1]["input_items"][-1]["call_id"] == "call-1"


@pytest.mark.asyncio
async def test_graph_stops_after_max_tool_turns() -> None:
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool_call("moveit_get_current_pose", call_id=f"call-{i}")],
                output_items=[output_item("moveit_get_current_pose", call_id=f"call-{i}")],
            )
            for i in range(4)
        ]
    )

    text = await fixture.graph.run_turn(turn("pose"))

    assert text == "I completed the action but have nothing to report."
    assert len(fixture.backend.requests) == 4


@pytest.mark.asyncio
async def test_graph_repairs_missing_relative_target_pose_and_preserves_orientation() -> None:
    tool = tool_call(
        "moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "plan_name": "move_up_50mm", "timeout_s": 10},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[
                    output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)
                ],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls[1][1]["target_pose"] == {
        "position": {"x": 0.1, "y": 0.2, "z": 0.35},
        "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
    }


@pytest.mark.asyncio
async def test_graph_repairs_back_up_as_negative_x() -> None:
    tool = tool_call(
        "moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "plan_name": "back_up_50mm", "timeout_s": 10},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[
                    output_item("moveit_plan_and_execute_free_motion", arguments=tool.arguments)
                ],
            ),
            CodexResponseResult(text="Moved back 50 mm."),
        ]
    )

    await fixture.graph.run_turn(turn("back up a bit"))

    assert fixture.bridge.calls[1][1]["target_pose"]["position"] == {
        "x": 0.05,
        "y": 0.2,
        "z": 0.3,
    }


@pytest.mark.asyncio
async def test_graph_repairs_cartesian_waypoints_from_current_pose() -> None:
    tool = tool_call(
        "moveit_plan_and_execute_cartesian_motion",
        arguments={"robot_name": "UR10", "plan_name": "move_up_cartesian_50mm", "timeout_s": 10},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[
                    output_item(
                        "moveit_plan_and_execute_cartesian_motion",
                        arguments=tool.arguments,
                    )
                ],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls[1][1]["waypoints"] == [
        {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": -0.7071, "z": -0.7071, "w": 0.0},
        }
    ]


@pytest.mark.asyncio
async def test_graph_auto_executes_executable_plan() -> None:
    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    plan = tool_call("moveit_plan_free_motion", arguments=plan_args)
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[plan],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="Moved up 50 mm."),
        ]
    )

    await fixture.graph.run_turn(turn("move up a bit"))

    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_plan_free_motion", plan_args),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "plan-1"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]


@pytest.mark.asyncio
async def test_graph_sends_policy_failure_as_tool_output_when_motion_lacks_fresh_observation() -> None:
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

        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            self.calls.append((name, arguments))
            return json.dumps({"structured_content": {"ok": True}})

    plan_args = {
        "robot_name": "UR10",
        "target_pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.35},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    tool = tool_call("moveit_plan_free_motion", arguments=plan_args)
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[tool],
                output_items=[output_item("moveit_plan_free_motion", arguments=plan_args)],
            ),
            CodexResponseResult(text="I need a fresh pose before moving."),
        ],
        bridge=NoObservationBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a fresh pose before moving."
    assert fixture.bridge.calls == []
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output == {
        "ok": False,
        "error": "Fresh robot pose is required before motion.",
        "correction": "Call moveit_get_current_pose, then retry the motion.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_blocks_blind_execute_plan_even_after_fresh_pose() -> None:
    execute = tool_call(
        "moveit_execute_plan",
        arguments={"robot_name": "UR10", "plan_name": "invented-plan"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[execute],
                output_items=[output_item("moveit_execute_plan", arguments=execute.arguments)],
            ),
            CodexResponseResult(text="I need to plan before executing."),
        ]
    )

    text = await fixture.graph.run_turn(turn("execute the last plan"))

    assert text == "I need to plan before executing."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
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

    failing_tool = tool_call(
        "moveit_plan_free_motion",
        arguments={
            "robot_name": "UR10",
            "target_pose": {
                "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
    )

    class ErrorBridge(FakeBridge):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
            if name == "moveit_get_current_pose":
                return await super().call_tool(name, arguments)
            raise RobotMCPError("robot server unavailable")

    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[failing_tool],
                output_items=[output_item("moveit_plan_free_motion", arguments=failing_tool.arguments)],
            ),
            CodexResponseResult(text="The robot server is unavailable."),
        ],
        bridge=ErrorBridge(),
    )

    await fixture.graph.run_turn(turn("move up"))

    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output == {
        "ok": False,
        "error": "robot server unavailable",
        "correction": "Check the robot control server, then retry the robot action.",
        "retryable": True,
        "suggested_next_tool": "moveit_get_current_pose",
    }


@pytest.mark.asyncio
async def test_graph_preserves_structured_robot_tool_failure_as_tool_output() -> None:
    bad_tool = tool_call(
        "moveit_plan_free_motion",
        arguments={
            "robot_name": "UR10",
            "target_pose": {
                "position": {"x": 0.1, "y": 0.2, "z": 0.35},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        },
    )

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
            CodexResponseResult(
                tool_calls=[bad_tool],
                output_items=[output_item("moveit_plan_free_motion", arguments=bad_tool.arguments)],
            ),
            CodexResponseResult(text="I need a valid plan before executing."),
        ],
        bridge=FailureBridge(),
    )

    text = await fixture.graph.run_turn(turn("move up"))

    assert text == "I need a valid plan before executing."
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
    assert output["ok"] is False
    assert output["retryable"] is True
    assert output["suggested_next_tool"] == "moveit_get_current_pose"


@pytest.mark.asyncio
async def test_graph_persists_context_between_turns_with_same_instance() -> None:
    fixture = make_graph([CodexResponseResult(text="first"), CodexResponseResult(text="second")])

    await fixture.graph.run_turn(turn("first"))
    await fixture.graph.run_turn(turn("second"))

    latest_state = fixture.graph.latest_state()
    assert "robot: UR10" in latest_state["instructions"]
    assert latest_state["tool_turns"] == 0


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

    attach = tool_call(
        "moveit_attach_object",
        arguments={"robot_name": "UR10", "object_name": "cube"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[attach],
                output_items=[output_item("moveit_attach_object", arguments=attach.arguments)],
            ),
            CodexResponseResult(text="I need to close the gripper before attaching."),
        ],
        bridge=AttachBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "I need to close the gripper before attaching."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
    output = json.loads(fixture.backend.requests[1]["input_items"][-1]["output"])
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

    close = tool_call("moveit_close_gripper", call_id="call-1", arguments={"robot_name": "UR10"})
    attach = tool_call(
        "moveit_attach_object",
        call_id="call-2",
        arguments={"robot_name": "UR10", "object_name": "cube"},
    )
    fixture = make_graph(
        [
            CodexResponseResult(
                tool_calls=[close, attach],
                output_items=[
                    output_item(
                        "moveit_close_gripper",
                        call_id="call-1",
                        arguments=close.arguments,
                    ),
                    output_item(
                        "moveit_attach_object",
                        call_id="call-2",
                        arguments=attach.arguments,
                    ),
                ],
            ),
            CodexResponseResult(text="Attached the cube."),
        ],
        bridge=GripperBridge(),
    )

    text = await fixture.graph.run_turn(turn("attach the cube"))

    assert text == "Attached the cube."
    assert fixture.bridge.calls == [
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
        ("moveit_close_gripper", {"robot_name": "UR10"}),
        ("moveit_attach_object", {"robot_name": "UR10", "object_name": "cube"}),
        ("moveit_get_current_pose", {"robot_name": "UR10"}),
    ]
