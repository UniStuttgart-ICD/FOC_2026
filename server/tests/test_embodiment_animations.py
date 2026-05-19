from __future__ import annotations

import asyncio
import json

import pytest

from embodiment.animations import (
    EMBODIMENT_SET_ANIMATION_TOOL,
    EmbodimentAnimationController,
    FakeAnimationRosTransport,
)
from voice_runtime.profiles import (
    EmbodimentMotionProfile,
    EmbodimentProfile,
    EmbodimentTouchTriggerProfile,
)


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_lifecycle_starts_and_stops_blink() -> None:
    transport = FakeAnimationRosTransport(connected=False)
    controller = EmbodimentAnimationController(
        EmbodimentProfile(enabled=True),
        transport=transport,
    )

    await controller.start()
    await controller.stop()

    assert transport.published == [
        ("/HOLO1_AnimSignal", "std_msgs/String", "start_blink"),
        ("/HOLO1_AnimSignal", "std_msgs/String", "stop_blink"),
    ]
    assert transport.connect_calls == 1
    assert transport.close_calls == 1


@pytest.mark.asyncio
async def test_wave_and_move_publish_explicit_start_and_stop_signals() -> None:
    transport = FakeAnimationRosTransport()
    controller = EmbodimentAnimationController(
        EmbodimentProfile(enabled=True, start_blink_on_connect=False),
        transport=transport,
        sleep_fn=_no_sleep,
    )

    await controller.start_animation("wave", side="left")
    await controller.stop_animation("wave", side="left")
    await controller.start_animation("move")
    await controller.stop_animation("move")

    assert [item[2] for item in transport.published] == [
        "start_waveL",
        "end_waveL",
        "start_move",
        "stop_move",
    ]


@pytest.mark.asyncio
async def test_agent_tool_call_controls_animation() -> None:
    transport = FakeAnimationRosTransport()
    controller = EmbodimentAnimationController(
        EmbodimentProfile(enabled=True),
        transport=transport,
        sleep_fn=_no_sleep,
    )

    output = await controller.handle_tool_call(
        EMBODIMENT_SET_ANIMATION_TOOL,
        {"motion": "blink", "action": "stop", "reason": "quiet"},
    )

    assert json.loads(output)["ok"] is True
    assert transport.published == [
        ("/HOLO1_AnimSignal", "std_msgs/String", "stop_blink"),
    ]


@pytest.mark.asyncio
async def test_custom_motion_uses_profile_registered_signals() -> None:
    transport = FakeAnimationRosTransport()
    controller = EmbodimentAnimationController(
        EmbodimentProfile(
            enabled=True,
            motions={
                "nod": EmbodimentMotionProfile(
                    start_signal="start_nod",
                    stop_signal="stop_nod",
                )
            },
        ),
        transport=transport,
    )

    await controller.start_animation("nod")
    await controller.stop_animation("nod")

    assert [item[2] for item in transport.published] == ["start_nod", "stop_nod"]


@pytest.mark.asyncio
async def test_touch_trigger_starts_configured_motion_with_cooldown() -> None:
    now_s = 10.0

    def time_fn() -> float:
        return now_s

    transport = FakeAnimationRosTransport()
    controller = EmbodimentAnimationController(
        EmbodimentProfile(
            enabled=True,
            start_blink_on_connect=False,
            touch_trigger=EmbodimentTouchTriggerProfile(
                enabled=True,
                topic="/touch",
                link_name="arm_link",
                motion="move",
                cooldown_s=1.0,
            ),
        ),
        transport=transport,
        time_fn=time_fn,
        sleep_fn=_no_sleep,
    )

    await controller.start()
    transport.record_string("/touch", json.dumps({"link_name": "arm_link"}))
    await asyncio.sleep(0)
    transport.record_string("/touch", json.dumps({"link_name": "arm_link"}))
    await asyncio.sleep(0)

    assert [item[2] for item in transport.published] == ["start_move"]


@pytest.mark.asyncio
async def test_agent_tool_call_rejects_play_action() -> None:
    transport = FakeAnimationRosTransport()
    controller = EmbodimentAnimationController(
        EmbodimentProfile(enabled=True),
        transport=transport,
    )

    output = await controller.handle_tool_call(
        EMBODIMENT_SET_ANIMATION_TOOL,
        {"motion": "wave", "action": "play", "reason": "old action"},
    )

    body = json.loads(output)
    assert body["ok"] is False
    assert body["error"] == "action must be start or stop"
    assert transport.published == []
