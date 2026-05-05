import json

import pytest

from test_support.live_robot_smoke import (
    LiveSmokeRun,
    RecordedToolCall,
    RecordingRobotToolAdapter,
    validate_ambiguous_clarification,
    validate_bit_movement,
    validate_position_query,
    validate_wave_motion,
)


class FakeRobotToolAdapter:
    def __init__(self) -> None:
        self.connected = False
        self.disconnected = False
        self.calls: list[tuple[str, dict]] = []

    async def connect(self) -> None:
        self.connected = True

    async def disconnect(self) -> None:
        self.disconnected = True

    def function_tools(self) -> list[dict]:
        return [
            {"type": "function", "name": "moveit_get_current_pose", "parameters": {"type": "object"}},
            {
                "type": "function",
                "name": "moveit_plan_and_execute_free_motion",
                "parameters": {"type": "object"},
            },
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        self.calls.append((name, arguments))
        return json.dumps({"structured_content": {"ok": True, "tool": name}})


@pytest.mark.asyncio
async def test_recording_adapter_delegates_and_records_json_output() -> None:
    delegate = FakeRobotToolAdapter()
    recorder = RecordingRobotToolAdapter(delegate)

    await recorder.connect()
    output = await recorder.call_tool("moveit_get_current_pose", {"robot_name": "UR10"})
    await recorder.disconnect()

    assert delegate.connected is True
    assert delegate.disconnected is True
    assert output == json.dumps({"structured_content": {"ok": True, "tool": "moveit_get_current_pose"}})
    assert delegate.calls == [("moveit_get_current_pose", {"robot_name": "UR10"})]
    assert [call.name for call in recorder.calls] == ["moveit_get_current_pose"]
    assert recorder.calls[0].arguments == {"robot_name": "UR10"}
    assert recorder.calls[0].output_json == {
        "structured_content": {"ok": True, "tool": "moveit_get_current_pose"}
    }


def test_position_query_requires_pose_observation_and_no_motion() -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="The current pose is x=0.1, y=0.2, z=0.3.",
        tool_calls=[pose_call(z=0.3)],
    )

    result = validate_position_query(run)

    assert result.passed is True
    assert result.reason == "position query observed current pose without movement"


def test_position_query_rejects_motion_tools() -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="Moved.",
        tool_calls=[pose_call(z=0.3), verified_execution_call()],
    )

    result = validate_position_query(run)

    assert result.passed is False
    assert "unexpected robot tools" in result.reason


@pytest.mark.parametrize("tool_name", ["moveit_close_gripper", "moveit_attach_object"])
def test_position_query_rejects_gripper_attach_side_effect_tool_call(tool_name: str) -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="The current pose is x=0.1, y=0.2, z=0.3.",
        tool_calls=[pose_call(z=0.3), robot_tool_call(tool_name)],
    )

    result = validate_position_query(run)

    assert result.passed is False
    assert "unexpected robot tools" in result.reason
    assert result.details == {"robot_tools": [tool_name]}


@pytest.mark.parametrize(
    ("output_json", "output_text"),
    [
        ({"structured_content": {"ok": False, "error": "pose unavailable"}}, "failed"),
        (
            {
                "structured_content": {
                    "ok": False,
                    "raw": {"pose": {"position": {"x": 0.1, "y": 0.2, "z": 0.3}}},
                }
            },
            "failed with pose payload",
        ),
        ({"structured_content": {"ok": True, "raw": {}}}, "missing pose"),
    ],
)
def test_position_query_rejects_pose_call_without_successful_parseable_pose(
    output_json: dict,
    output_text: str,
) -> None:
    run = LiveSmokeRun(
        prompt="what is the current position?",
        reply="I could not get the current pose.",
        tool_calls=[
            RecordedToolCall(
                name="moveit_get_current_pose",
                arguments={"robot_name": "UR10"},
                output_text=output_text,
                output_json=output_json,
            )
        ],
    )

    result = validate_position_query(run)

    assert result.passed is False
    assert result.reason == "position query did not observe a successful parseable current pose"


def test_move_up_bit_accepts_verified_plus_z_motion() -> None:
    run = LiveSmokeRun(
        prompt="move up a bit",
        reply="Moved up 50 mm.",
        tool_calls=[pose_call(z=0.30), verified_execution_call(), pose_call(z=0.35)],
    )

    result = validate_bit_movement(run, direction="up")

    assert result.passed is True
    assert result.details["delta_z_m"] == pytest.approx(0.05)


def test_move_up_bit_rejects_wrong_direction() -> None:
    run = LiveSmokeRun(
        prompt="move up a bit",
        reply="Moved up 50 mm.",
        tool_calls=[pose_call(z=0.30), verified_execution_call(), pose_call(z=0.25)],
    )

    result = validate_bit_movement(run, direction="up")

    assert result.passed is False
    assert "expected +Z movement" in result.reason


def test_move_down_bit_accepts_verified_minus_z_motion() -> None:
    run = LiveSmokeRun(
        prompt="move down a bit",
        reply="Moved down 50 mm.",
        tool_calls=[pose_call(z=0.35), verified_execution_call(), pose_call(z=0.30)],
    )

    result = validate_bit_movement(run, direction="down")

    assert result.passed is True
    assert result.details["delta_z_m"] == pytest.approx(-0.05)


def test_wave_motion_accepts_visible_verified_cartesian_sweep() -> None:
    run = LiveSmokeRun(
        prompt="wave to me",
        reply="I waved.",
        tool_calls=[
            pose_call(z=0.62, y=0.39),
            verified_cartesian_execution_call(
                [
                    waypoint(y=0.49, z=0.70),
                    waypoint(y=0.29, z=0.70),
                    waypoint(y=0.49, z=0.70),
                    waypoint(y=0.29, z=0.70),
                    waypoint(y=0.39, z=0.62),
                ]
            ),
            pose_call(z=0.62, y=0.39),
        ],
    )

    result = validate_wave_motion(run)

    assert result.passed is True
    assert result.details["lateral_span_m"] == pytest.approx(0.20)
    assert result.details["vertical_lift_m"] == pytest.approx(0.08)


def test_wave_motion_rejects_timid_sweep() -> None:
    run = LiveSmokeRun(
        prompt="wave to me",
        reply="I waved.",
        tool_calls=[
            pose_call(z=0.62, y=0.39),
            verified_cartesian_execution_call(
                [
                    waypoint(y=0.43, z=0.65),
                    waypoint(y=0.35, z=0.65),
                    waypoint(y=0.43, z=0.65),
                    waypoint(y=0.35, z=0.65),
                ]
            ),
            pose_call(z=0.62, y=0.39),
        ],
    )

    result = validate_wave_motion(run)

    assert result.passed is False
    assert "expected at least 0.18 m lateral wave span" in result.reason


def test_ambiguous_command_accepts_clarification_without_motion() -> None:
    run = LiveSmokeRun(
        prompt="move there",
        reply="Where would you like me to move?",
        tool_calls=[pose_call(z=0.30)],
    )

    result = validate_ambiguous_clarification(run)

    assert result.passed is True
    assert result.reason == "ambiguous command asked for clarification without movement"


def test_ambiguous_command_rejects_motion_execution() -> None:
    run = LiveSmokeRun(
        prompt="move there",
        reply="I moved there.",
        tool_calls=[pose_call(z=0.30), verified_execution_call()],
    )

    result = validate_ambiguous_clarification(run)

    assert result.passed is False
    assert "unexpected robot tools" in result.reason


@pytest.mark.parametrize("tool_name", ["moveit_open_gripper", "moveit_attach_object"])
def test_ambiguous_command_rejects_gripper_attach_side_effect_tool_call(tool_name: str) -> None:
    run = LiveSmokeRun(
        prompt="do it there",
        reply="Where would you like me to do that?",
        tool_calls=[robot_tool_call(tool_name)],
    )

    result = validate_ambiguous_clarification(run)

    assert result.passed is False
    assert "unexpected robot tools" in result.reason
    assert result.details == {"robot_tools": [tool_name]}


def pose_call(*, z: float, x: float = 0.10, y: float = 0.20) -> RecordedToolCall:
    output_json = {
        "structured_content": {
            "ok": True,
            "robot": "UR10",
            "raw": {
                "pose": {
                    "position": {"x": x, "y": y, "z": z},
                    "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                }
            },
        }
    }
    return RecordedToolCall(
        name="moveit_get_current_pose",
        arguments={"robot_name": "UR10"},
        output_text=json.dumps(output_json),
        output_json=output_json,
    )


def verified_execution_call() -> RecordedToolCall:
    output_json = {
        "structured_content": {
            "ok": True,
            "verification": {"result": "pass"},
        }
    }
    return RecordedToolCall(
        name="moveit_plan_and_execute_free_motion",
        arguments={"robot_name": "UR10", "target_pose": {"x": 0.1, "y": 0.2, "z": 0.35}},
        output_text=json.dumps(output_json),
        output_json=output_json,
    )


def waypoint(*, y: float, z: float, x: float = 0.10) -> dict[str, object]:
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
    }


def verified_cartesian_execution_call(waypoints: list[dict[str, object]]) -> RecordedToolCall:
    output_json = {
        "structured_content": {
            "ok": True,
            "verification": {"result": "pass"},
        }
    }
    return RecordedToolCall(
        name="moveit_plan_and_execute_cartesian_motion",
        arguments={"robot_name": "UR10", "waypoints": waypoints},
        output_text=json.dumps(output_json),
        output_json=output_json,
    )


def robot_tool_call(name: str) -> RecordedToolCall:
    output_json = {"structured_content": {"ok": True}}
    arguments = {"robot_name": "UR10"}
    if name == "moveit_attach_object":
        arguments["object_name"] = "cube"
    return RecordedToolCall(
        name=name,
        arguments=arguments,
        output_text=json.dumps(output_json),
        output_json=output_json,
    )
