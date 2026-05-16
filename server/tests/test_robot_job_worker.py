import json
from typing import Any

import pytest

from robot_control.job_board import RobotJobBoard, RobotJobEventType, RobotJobStatus, SubmitRobotJob
from robot_control.job_worker import RobotJobWorker


class RecordingToolBridge:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        return self.result


class MappingToolBridge:
    def __init__(self, results: dict[str, str]) -> None:
        self.results = results
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        try:
            return self.results[name]
        except KeyError as exc:
            raise AssertionError(f"unexpected call: {name} {arguments}") from exc


class FailingToolBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        raise RuntimeError("planning failed")


class RecordingVerifiedExecutionClient:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[tuple[str, str, float]] = []

    async def execute_plan(
        self,
        *,
        robot_name: str,
        plan_name: str,
        timeout_s: float,
    ) -> str:
        self.calls.append((robot_name, plan_name, timeout_s))
        return self.result


class RoutingToolBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "moveit_plan_free_motion":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": "first-plan"},
                    }
                }
            )
        if name == "moveit_execute_plan":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "verification": {"result": "pass"},
                        "raw": {"plan_name": arguments["plan_name"]},
                    }
                }
            )
        return json.dumps({"structured_content": {"ok": True}})


class PickContinuationBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((name, arguments))
        if name == "moveit_plan_pick" and arguments.get("planning_strategy") == "auto":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "feedback": {"can_execute": True},
                        "raw": {
                            "plan_name": "pick_preposition",
                            "next_action": {
                                "tool": "moveit_execute_plan",
                                "plan_name": "pick_preposition",
                                "after_success": {
                                    "tool": "moveit_plan_pick",
                                    "arguments": {
                                        "object_name": "dynamic_5",
                                        "plan_name": "pick_local",
                                        "planning_strategy": "cartesian",
                                    },
                                },
                            },
                        },
                    }
                }
            )
        if name == "moveit_execute_plan":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "verification": {"result": "pass"},
                        "raw": {"plan_name": arguments["plan_name"]},
                    }
                }
            )
        if name == "moveit_plan_pick" and arguments.get("planning_strategy") == "cartesian":
            return json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": "pick_local", "workflow_kind": "pick"},
                    }
                }
            )
        raise AssertionError(f"unexpected call: {name} {arguments}")


@pytest.mark.asyncio
async def test_worker_executes_exact_queued_call_and_marks_job_completed() -> None:
    board = RobotJobBoard()
    bridge = RecordingToolBridge('{"structured_content": {"ok": true}}')
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "timeout_s": 10},
            "turn-1",
        )
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert stored.result == '{"structured_content": {"ok": true}}'
    assert bridge.calls[0][1] is stored.arguments
    assert bridge.calls == [
        (
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "timeout_s": 10},
        )
    ]


@pytest.mark.asyncio
async def test_worker_routes_queued_execute_plan_to_verified_execution_client() -> None:
    board = RobotJobBoard()
    bridge = RecordingToolBridge('{"structured_content": {"ok": true}}')
    verified_client = RecordingVerifiedExecutionClient(
        json.dumps(
            {
                "structured_content": {
                    "ok": True,
                    "verification": {"result": "pass"},
                    "feedback": {"plan_name": "plan-1"},
                }
            }
        )
    )
    worker = RobotJobWorker(
        board=board,
        tool_bridge=bridge,
        verified_execution_client=verified_client,
    )
    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "plan-1", "timeout_s": 7.0},
            "turn-1",
        )
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert stored.result == verified_client.result
    assert verified_client.calls == [("UR10", "plan-1", 7.0)]
    assert bridge.calls == []


@pytest.mark.asyncio
async def test_worker_routes_mcp_required_execute_plan_to_tool_bridge() -> None:
    board = RobotJobBoard()
    bridge = RecordingToolBridge('{"structured_content": {"ok": true}}')
    verified_client = RecordingVerifiedExecutionClient(
        json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
    )
    worker = RobotJobWorker(
        board=board,
        tool_bridge=bridge,
        verified_execution_client=verified_client,
    )
    job = await board.submit(
        SubmitRobotJob(
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "pick-local", "timeout_s": 7.0},
            "turn-1",
            execute_via_mcp=True,
        )
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.COMPLETED
    assert bridge.calls == [
        (
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "pick-local", "timeout_s": 7.0},
        )
    ]
    assert verified_client.calls == []


@pytest.mark.asyncio
async def test_worker_records_tool_failure_without_retrying_or_rewriting_args() -> None:
    board = RobotJobBoard()
    bridge = FailingToolBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob("moveit_open_gripper", {"robot_name": "UR10"}, "turn-1")
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert bridge.calls[0][1] is stored.arguments
    assert bridge.calls == [("moveit_open_gripper", {"robot_name": "UR10"})]
    assert stored.status is RobotJobStatus.FAILED
    assert stored.error == "planning failed"


@pytest.mark.asyncio
async def test_worker_marks_structured_failed_tool_result_as_failed() -> None:
    board = RobotJobBoard()
    bridge = RecordingToolBridge(
        json.dumps(
            {
                "structured_content": {
                    "ok": False,
                    "feedback": {
                        "message": "Planning failed; execution was not attempted",
                        "correction": "Replan with a smaller or safer target.",
                    },
                }
            }
        )
    )
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    job = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "timeout_s": 10},
            "turn-1",
        )
    )

    ran = await worker.run_once()

    assert ran is True
    stored = board.get(job.job_id)
    assert stored is not None
    assert stored.status is RobotJobStatus.FAILED
    assert stored.error == (
        "Planning failed; execution was not attempted "
        "Replan with a smaller or safer target."
    )
    assert stored.result == bridge.result
    event = board.events_since(0)[-1]
    assert event.event_type is RobotJobEventType.FAILED
    assert event.payload["result"] == bridge.result


@pytest.mark.asyncio
async def test_worker_auto_executes_first_successful_plan_before_remaining_candidates() -> None:
    board = RobotJobBoard()
    bridge = RoutingToolBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    first = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": {"position": {"x": 0.1}}},
            "turn-1",
            user_text="go ahead and move up a bit",
        )
    )
    second = await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": {"position": {"x": 0.2}}},
            "turn-1",
            user_text="go ahead and move up a bit",
        )
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is True
    assert await worker.run_once() is False

    stored_first = board.get(first.job_id)
    stored_second = board.get(second.job_id)
    assert stored_first is not None
    assert stored_second is not None
    assert stored_first.status is RobotJobStatus.COMPLETED
    assert stored_second.status is RobotJobStatus.CANCELLED
    assert bridge.calls == [
        (
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": {"position": {"x": 0.1}}},
        ),
        (
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "first-plan", "timeout_s": 10.0},
        ),
    ]


@pytest.mark.asyncio
async def test_worker_auto_executes_successful_pick_plan_with_typed_contract() -> None:
    board = RobotJobBoard()
    bridge = MappingToolBridge(
        {
            "moveit_plan_pick": json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "robot": "UR10",
                        "feedback": {"can_execute": True},
                        "raw": {"plan_name": "pick-plan-1", "selected_grasp_face": "top"},
                    }
                }
            ),
            "moveit_execute_plan": json.dumps(
                {
                    "structured_content": {
                        "ok": True,
                        "verification": {"result": "pass"},
                        "raw": {"plan_name": "pick-plan-1"},
                    }
                }
            ),
        }
    )
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    await board.submit(
        SubmitRobotJob(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "beam_001"},
            "turn-1",
            user_text="pick it up and execute it",
        )
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is True

    assert bridge.calls == [
        ("moveit_plan_pick", {"robot_name": "UR10", "object_name": "beam_001"}),
        ("moveit_execute_plan", {"robot_name": "UR10", "plan_name": "pick-plan-1", "timeout_s": 10.0}),
    ]


@pytest.mark.asyncio
async def test_worker_does_not_auto_execute_planning_only_request() -> None:
    board = RobotJobBoard()
    bridge = RoutingToolBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    await board.submit(
        SubmitRobotJob(
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": {"position": {"x": 0.1}}},
            "turn-1",
            user_text="plan a move up but do not execute",
        )
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is False

    assert bridge.calls == [
        (
            "moveit_plan_free_motion",
            {"robot_name": "UR10", "target_pose": {"position": {"x": 0.1}}},
        )
    ]


@pytest.mark.asyncio
async def test_worker_does_not_auto_execute_plain_pick_request() -> None:
    board = RobotJobBoard()
    bridge = PickContinuationBridge()
    worker = RobotJobWorker(board=board, tool_bridge=bridge)
    await board.submit(
        SubmitRobotJob(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "dynamic_5", "planning_strategy": "auto"},
            "turn-1",
            user_text="pick up dynamic 5",
        )
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is False

    assert bridge.calls == [
        (
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "dynamic_5", "planning_strategy": "auto"},
        )
    ]


@pytest.mark.asyncio
async def test_worker_queues_pick_after_success_continuation_after_preposition_execute() -> None:
    board = RobotJobBoard()
    bridge = PickContinuationBridge()
    verified_client = RecordingVerifiedExecutionClient(
        json.dumps({"structured_content": {"ok": True, "verification": {"result": "pass"}}})
    )
    worker = RobotJobWorker(
        board=board,
        tool_bridge=bridge,
        verified_execution_client=verified_client,
    )
    await board.submit(
        SubmitRobotJob(
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "dynamic_5", "planning_strategy": "auto"},
            "turn-1",
            user_text="go ahead and pick up dynamic 5",
        )
    )

    assert await worker.run_once() is True
    assert await worker.run_once() is True
    assert await worker.run_once() is True
    assert await worker.run_once() is True
    assert await worker.run_once() is False

    assert bridge.calls == [
        (
            "moveit_plan_pick",
            {"robot_name": "UR10", "object_name": "dynamic_5", "planning_strategy": "auto"},
        ),
        (
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "pick_preposition", "timeout_s": 10.0},
        ),
        (
            "moveit_plan_pick",
            {"object_name": "dynamic_5", "plan_name": "pick_local", "planning_strategy": "cartesian"},
        ),
        (
            "moveit_execute_plan",
            {"robot_name": "UR10", "plan_name": "pick_local", "timeout_s": 10.0},
        ),
    ]
    assert verified_client.calls == []
