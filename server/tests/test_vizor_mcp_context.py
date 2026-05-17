import sys
import time
from types import SimpleNamespace

import pytest

from vizor_mcp.ros_client import FakeVizorRosTransport, RoslibpyVizorRosTransport
from vizor_mcp.server import build_mcp, build_tools
from vizor_mcp.tools import VizorMcpTools
from vizor_mcp.transforms import (
    DF2025_ARCHIVED_OFFSET_M,
    unity_position_to_archived_robot_base,
    unity_position_to_robot,
)


def test_unity_position_to_robot_swaps_axes_without_default_offset() -> None:
    position = unity_position_to_robot({"x": 1.147, "y": -0.513, "z": 0.771})

    assert position == pytest.approx({"x": -0.513, "y": 0.771, "z": 1.147})


def test_archived_robot_base_transform_keeps_df2025_calibration_available() -> None:
    position = unity_position_to_archived_robot_base(
        {"x": 1.147, "y": -0.513, "z": 0.771},
        offset_m=DF2025_ARCHIVED_OFFSET_M,
    )

    assert position == pytest.approx({"x": 0.340, "y": -0.720, "z": 1.250})


def test_sensor_context_aggregates_gaze_user_pose_and_manual_target() -> None:
    transport = FakeVizorRosTransport(now_s=100.0)
    transport.record_message(
        "/HOLO1_GazePoint",
        "std_msgs/String",
        {"data": "dynamic_beam_001"},
        received_at_s=99.8,
    )
    transport.record_message(
        "/HOLO1_Transform",
        "geometry_msgs/Pose",
        {
            "position": {"x": 1.147, "y": -0.513, "z": 0.771},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        received_at_s=99.7,
    )
    transport.record_message(
        "/Robot/target_manual",
        "geometry_msgs/Pose",
        {
            "position": {"x": 0.42, "y": 0.11, "z": 0.73},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
        received_at_s=99.6,
    )
    tools = VizorMcpTools.with_transport(transport)

    context = tools.get_sensor_context(max_age_s=1.0)

    assert context["ok"] is True
    assert context["tool"] == "vizor_get_sensor_context"
    assert context["source"] == "rosbridge"
    assert context["freshness"]["stale"] is False
    assert context["gaze"] == {
        "available": True,
        "target": "beam_001",
        "raw_target": "dynamic_beam_001",
        "age_s": pytest.approx(0.2),
        "stale": False,
        "source_topic": "/HOLO1_GazePoint",
        "message_type": "std_msgs/String",
    }
    assert context["user"]["available"] is True
    assert context["user"]["position"] == pytest.approx({"x": -0.513, "y": 0.771, "z": 1.147})
    assert context["user"]["frame"] == "base_link"
    assert context["user"]["raw_unity_pose"]["position"] == {"x": 1.147, "y": -0.513, "z": 0.771}
    assert context["manual_target"]["available"] is True
    assert context["manual_target"]["position"] == pytest.approx({"x": 0.11, "y": 0.73, "z": 0.42})
    assert context["manual_target"]["frame"] == "base_link"
    assert context["calibration"]["offset_enabled"] is False
    assert context["calibration"]["active_offset_m"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert context["calibration"]["archived_offset_m"] == DF2025_ARCHIVED_OFFSET_M
    assert context["attention"]["dominant_target"] == "beam_001"
    assert context["attention"]["ranked_targets"][0]["target"] == "beam_001"


def test_missing_and_old_topics_are_reported_per_field() -> None:
    transport = FakeVizorRosTransport(now_s=100.0)
    transport.record_message(
        "/HOLO1_Transform",
        "geometry_msgs/Pose",
        {"position": {"x": 1.0, "y": 2.0, "z": 3.0}},
        received_at_s=94.0,
    )
    tools = VizorMcpTools.with_transport(transport)

    context = tools.get_sensor_context(max_age_s=2.0)

    assert context["ok"] is True
    assert context["freshness"]["stale"] is True
    assert context["gaze"]["available"] is False
    assert context["gaze"]["stale"] is True
    assert context["gaze"]["age_s"] is None
    assert context["user"]["available"] is True
    assert context["user"]["stale"] is True
    assert context["user"]["age_s"] == pytest.approx(6.0)
    assert context["manual_target"]["available"] is False
    assert context["manual_target"]["stale"] is True


def test_disconnected_rosbridge_reports_retryable_context_with_cache() -> None:
    transport = FakeVizorRosTransport(now_s=100.0, connected=False)
    transport.record_message(
        "/HOLO1_GazePoint",
        "std_msgs/String",
        {"data": "dynamic_column_a"},
        received_at_s=99.9,
    )
    tools = VizorMcpTools.with_transport(transport)

    context = tools.get_sensor_context(max_age_s=2.0)

    assert context["ok"] is False
    assert context["retryable"] is True
    assert context["rosbridge"]["connected"] is False
    assert context["gaze"]["target"] == "column_a"
    assert context["gaze"]["stale"] is False
    assert context["attention"]["dominant_target"] == "column_a"


def test_ros_transport_holo1_tracking_keepalive_republishes(monkeypatch: pytest.MonkeyPatch) -> None:
    published: list[tuple[str, dict[str, str]]] = []

    class FakeRos:
        is_connected = True

        def __init__(self, *, host: str, port: int) -> None:
            self.host = host
            self.port = port

        def run(self) -> None:
            pass

        def terminate(self) -> None:
            self.is_connected = False

    class FakeTopic:
        def __init__(self, _client: FakeRos, topic: str, _message_type: str) -> None:
            self.topic = topic

        def subscribe(self, _callback: object) -> None:
            pass

        def unsubscribe(self) -> None:
            pass

        def publish(self, message: dict[str, str]) -> None:
            published.append((self.topic, message))

    fake_roslibpy = SimpleNamespace(Ros=FakeRos, Topic=FakeTopic, Message=lambda data: data)
    monkeypatch.setitem(sys.modules, "roslibpy", fake_roslibpy)

    transport = RoslibpyVizorRosTransport()
    transport.connect()
    transport.start_holo1_tracking_keepalive(interval_s=0.01)
    deadline = time.monotonic() + 0.5
    while len(published) < 2 and time.monotonic() < deadline:
        time.sleep(0.01)
    transport.close()

    assert len(published) >= 2
    assert published[0] == ("/WorkerPool/control", {"data": "HOLO1_position_on"})


def test_ros_transport_startup_tolerates_rosbridge_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    subscribed: list[str] = []

    class RosTimeoutError(Exception):
        pass

    class FakeRos:
        is_connected = False

        def __init__(self, *, host: str, port: int) -> None:
            self.host = host
            self.port = port

        def run(self) -> None:
            raise RosTimeoutError("not ready")

        def terminate(self) -> None:
            pass

    class FakeTopic:
        def __init__(self, _client: FakeRos, topic: str, _message_type: str) -> None:
            self.topic = topic

        def subscribe(self, _callback: object) -> None:
            subscribed.append(self.topic)

        def unsubscribe(self) -> None:
            pass

    fake_roslibpy = SimpleNamespace(
        Ros=FakeRos,
        Topic=FakeTopic,
        Message=lambda data: data,
    )
    monkeypatch.setitem(sys.modules, "roslibpy", fake_roslibpy)

    transport = RoslibpyVizorRosTransport()
    transport.connect()

    assert transport.is_connected() is False
    assert subscribed == [
        "/HOLO1_GazePoint",
        "/HOLO1_Transform",
        "/Robot/target_manual",
    ]


@pytest.mark.asyncio
async def test_mcp_registers_read_only_sensor_context_tool() -> None:
    tools = build_tools(transport=FakeVizorRosTransport())
    mcp = build_mcp(tools=tools)

    registered = {tool.name: tool for tool in await mcp.list_tools()}

    assert set(registered) == {"vizor_get_sensor_context", "vizor_get_status"}
    annotations = registered["vizor_get_sensor_context"].annotations
    assert annotations is not None
    assert annotations.readOnlyHint is True
    assert annotations.destructiveHint is False
    assert annotations.idempotentHint is True


@pytest.mark.asyncio
async def test_mcp_sensor_context_tool_returns_structured_payload() -> None:
    transport = FakeVizorRosTransport(now_s=100.0)
    transport.record_message(
        "/HOLO1_GazePoint",
        "std_msgs/String",
        {"data": "dynamic_beam_001"},
        received_at_s=99.9,
    )
    mcp = build_mcp(tools=build_tools(transport=transport))

    _, payload = await mcp.call_tool("vizor_get_sensor_context", {"max_age_s": 1.0})

    assert payload["ok"] is True
    assert payload["gaze"]["target"] == "beam_001"
    assert payload["attention"]["dominant_target"] == "beam_001"
