from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import RLock
from typing import Any, Protocol

from loguru import logger

from voice_runtime.profiles import EmbodimentProfile

EMBODIMENT_SET_ANIMATION_TOOL = "embodiment_set_animation"
EMBODIMENT_FAKE_DEATH_TOOL = "embodiment_fake_death"
EMBODIMENT_TOOL_NAMES = frozenset(
    {EMBODIMENT_SET_ANIMATION_TOOL, EMBODIMENT_FAKE_DEATH_TOOL}
)


class AnimationRosTransport(Protocol):
    host: str
    port: int

    def connect(self) -> None: ...
    def close(self) -> None: ...
    def is_connected(self) -> bool: ...
    def publish_string(self, topic: str, message_type: str, data: str) -> None: ...
    def subscribe_string(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[str], None],
    ) -> None: ...


class FakeAnimationRosTransport:
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 9090,
        connected: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.connected = connected
        self.connect_calls = 0
        self.close_calls = 0
        self.published: list[tuple[str, str, str]] = []
        self.subscriptions: dict[str, Callable[[str], None]] = {}

    def connect(self) -> None:
        self.connect_calls += 1
        self.connected = True

    def close(self) -> None:
        self.close_calls += 1
        self.connected = False

    def is_connected(self) -> bool:
        return self.connected

    def publish_string(self, topic: str, message_type: str, data: str) -> None:
        self.published.append((topic, message_type, data))

    def subscribe_string(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[str], None],
    ) -> None:
        self.subscriptions[topic] = callback

    def record_string(self, topic: str, data: str) -> None:
        callback = self.subscriptions.get(topic)
        if callback is not None:
            callback(data)


class RoslibpyAnimationRosTransport:
    def __init__(self, *, host: str = "localhost", port: int = 9090) -> None:
        import roslibpy

        self.roslibpy = roslibpy
        self.host = host
        self.port = port
        self.client = roslibpy.Ros(host=host, port=port)
        self._lock = RLock()
        self._publishers: dict[tuple[str, str], Any] = {}
        self._subscribers: dict[str, Any] = {}
        self._connected = False

    def connect(self) -> None:
        if self.is_connected():
            return
        try:
            self.client.run()
        except Exception as exc:
            if exc.__class__.__name__ != "RosTimeoutError":
                raise
        self._connected = True

    def close(self) -> None:
        with self._lock:
            subscribers = list(self._subscribers.values())
            self._subscribers.clear()
            self._publishers.clear()
        for subscriber in subscribers:
            try:
                subscriber.unsubscribe()
            except Exception:
                pass
        self.client.terminate()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected and self.client.is_connected)

    def publish_string(self, topic: str, message_type: str, data: str) -> None:
        publisher = self._publisher(topic, message_type)
        publisher.publish(self.roslibpy.Message({"data": data}))

    def subscribe_string(
        self,
        topic: str,
        message_type: str,
        callback: Callable[[str], None],
    ) -> None:
        with self._lock:
            if topic in self._subscribers:
                return
            subscriber = self.roslibpy.Topic(self.client, topic, message_type)
            subscriber.subscribe(lambda msg: callback(str(msg.get("data") or "")))
            self._subscribers[topic] = subscriber

    def _publisher(self, topic: str, message_type: str) -> Any:
        key = (topic, message_type)
        with self._lock:
            publisher = self._publishers.get(key)
            if publisher is None:
                publisher = self.roslibpy.Topic(self.client, topic, message_type)
                self._publishers[key] = publisher
            return publisher


@dataclass(frozen=True)
class _MotionSignal:
    start: str
    stop: str


class EmbodimentAnimationController:
    def __init__(
        self,
        config: EmbodimentProfile,
        *,
        transport: AnimationRosTransport | None = None,
        time_fn: Callable[[], float] = time.monotonic,
        sleep_fn: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.config = config
        self._transport = transport or RoslibpyAnimationRosTransport(
            host=config.rosbridge_host,
            port=config.rosbridge_port,
        )
        self._time_fn = time_fn
        self._sleep_fn = sleep_fn
        self._started = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_touch_trigger_s: float | None = None
        self._connect_warning_logged = False

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def transport(self) -> AnimationRosTransport:
        return self._transport

    async def start(self) -> None:
        if not self.enabled:
            return
        self._loop = asyncio.get_running_loop()
        if not self._ensure_connected():
            return
        self._subscribe_touch_trigger()
        self._started = True
        if self.config.start_blink_on_connect:
            await self.start_animation("blink")

    async def stop(self) -> None:
        if not self.enabled:
            return
        if self.config.stop_blink_on_disconnect and self._started:
            await self.stop_animation("blink")
        try:
            self._transport.close()
        except Exception as exc:
            logger.warning("Embodiment animation ROSBridge close failed: {}", exc)
        self._started = False
        self._loop = None

    async def fake_death(self) -> dict[str, Any]:
        await self.stop_animation("blink")
        return _tool_result(ok=True, action="fake_death", signal="stop_blink")

    async def start_animation(self, motion: str, *, side: str | None = None) -> dict[str, Any]:
        signal = self._signal(motion, side=side).start
        return await self._publish_signal(signal, action="start", motion=motion, side=side)

    async def stop_animation(self, motion: str, *, side: str | None = None) -> dict[str, Any]:
        signal = self._signal(motion, side=side).stop
        return await self._publish_signal(signal, action="stop", motion=motion, side=side)

    async def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> str:
        if name == EMBODIMENT_FAKE_DEATH_TOOL:
            return json.dumps(await self.fake_death(), ensure_ascii=False)
        if name != EMBODIMENT_SET_ANIMATION_TOOL:
            return json.dumps(
                _tool_result(ok=False, error=f"Unknown embodiment tool: {name}"),
                ensure_ascii=False,
            )

        action = _clean(arguments.get("action")) or "start"
        motion = _clean(arguments.get("motion")) or "move"
        side = _normalize_side(arguments.get("side"))
        try:
            if action == "start":
                result = await self.start_animation(motion, side=side)
            elif action == "stop":
                result = await self.stop_animation(motion, side=side)
            else:
                result = _tool_result(
                    ok=False,
                    error="action must be start or stop",
                    action=action,
                    motion=motion,
                    side=side,
                )
        except ValueError as exc:
            result = _tool_result(ok=False, error=str(exc), action=action, motion=motion, side=side)
        return json.dumps(result, ensure_ascii=False)

    async def _publish_signal(
        self,
        signal: str,
        *,
        action: str,
        motion: str,
        side: str | None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return _tool_result(
                ok=False,
                error="Embodiment animations are disabled for this profile.",
                action=action,
                motion=motion,
                side=side,
            )
        if not self._ensure_connected():
            return _tool_result(
                ok=False,
                error="Embodiment animation ROSBridge is unavailable.",
                action=action,
                motion=motion,
                side=side,
                host=self.config.rosbridge_host,
                port=self.config.rosbridge_port,
                topic=self.config.animation_topic,
                retryable=True,
            )
        try:
            self._transport.publish_string(
                self.config.animation_topic,
                self.config.animation_topic_type,
                signal,
            )
        except Exception as exc:
            logger.warning("Embodiment animation publish failed signal={}: {}", signal, exc)
            return _tool_result(
                ok=False,
                error="Embodiment animation publish failed.",
                action=action,
                motion=motion,
                side=side,
                signal=signal,
                host=self.config.rosbridge_host,
                port=self.config.rosbridge_port,
                topic=self.config.animation_topic,
                retryable=True,
            )
        return _tool_result(
            ok=True,
            action=action,
            motion=motion,
            side=side,
            signal=signal,
            topic=self.config.animation_topic,
            message_type=self.config.animation_topic_type,
            host=self.config.rosbridge_host,
            port=self.config.rosbridge_port,
            connected=self._transport.is_connected(),
        )

    def _ensure_connected(self) -> bool:
        try:
            if not self._transport.is_connected():
                self._transport.connect()
            return self._transport.is_connected()
        except Exception as exc:
            if not self._connect_warning_logged:
                logger.warning("Embodiment animation ROSBridge connection failed: {}", exc)
                self._connect_warning_logged = True
            return False

    def _subscribe_touch_trigger(self) -> None:
        trigger = self.config.touch_trigger
        if not trigger.enabled or not trigger.topic:
            return
        try:
            self._transport.subscribe_string(
                trigger.topic,
                trigger.topic_type,
                self._handle_touch_message,
            )
        except Exception as exc:
            logger.warning("Embodiment touch trigger subscription failed: {}", exc)

    def _handle_touch_message(self, data: str) -> None:
        trigger = self.config.touch_trigger
        if not trigger.enabled:
            return
        touched_link = _touch_link_name(data)
        configured_link = _clean(trigger.link_name)
        if configured_link and touched_link != configured_link:
            return
        now_s = self._time_fn()
        if (
            self._last_touch_trigger_s is not None
            and now_s - self._last_touch_trigger_s < trigger.cooldown_s
        ):
            return
        self._last_touch_trigger_s = now_s
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self.start_animation(trigger.motion), loop)

    def _signal(self, motion: str, *, side: str | None) -> _MotionSignal:
        motion = motion.strip().casefold()
        configured = self.config.motions.get(motion)
        if configured is not None:
            return _MotionSignal(
                start=configured.start_signal,
                stop=configured.stop_signal,
            )
        if motion == "blink":
            return _MotionSignal(start="start_blink", stop="stop_blink")
        if motion == "move":
            return _MotionSignal(start="start_move", stop="stop_move")
        if motion == "wave":
            suffix = "L" if side == "left" else "R"
            return _MotionSignal(start=f"start_wave{suffix}", stop=f"end_wave{suffix}")
        raise ValueError("motion must be one of blink, move, or wave")


def create_embodiment_animation_controller(
    config: EmbodimentProfile,
) -> EmbodimentAnimationController | None:
    if not config.enabled:
        return None
    return EmbodimentAnimationController(config)


def embodiment_tool_definitions() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": EMBODIMENT_SET_ANIMATION_TOOL,
            "description": (
                "Control the agent's AR embodiment animation. Use blink for the looping "
                "alive/awake state, wave for a greeting or acknowledgment, and move for "
                "an embodied movement cue. Animations support only explicit start and stop "
                "requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "motion": {
                        "type": "string",
                        "description": (
                            "Registered motion name, such as blink, wave, move, or a "
                            "profile-specific custom motion."
                        ),
                    },
                    "action": {
                        "type": "string",
                        "enum": ["start", "stop"],
                        "description": "Start or stop the selected animation.",
                    },
                    "side": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Wave hand side. Defaults to right for wave.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Short reason this embodiment change matches the turn.",
                    },
                },
                "required": ["motion", "action", "reason"],
                "additionalProperties": False,
            },
            "strict": None,
        },
        {
            "type": "function",
            "name": EMBODIMENT_FAKE_DEATH_TOOL,
            "description": (
                "Make the agent embodiment appear inactive by stopping the looping blink "
                "animation without ending the conversation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "Short reason the persona is faking death or becoming inactive."
                        ),
                    },
                },
                "required": ["reason"],
                "additionalProperties": False,
            },
            "strict": None,
        },
    ]


def _tool_result(**values: Any) -> dict[str, Any]:
    payload = {"tool": "embodiment_animation"}
    payload.update({key: value for key, value in values.items() if value is not None})
    return payload


def _normalize_side(value: Any) -> str | None:
    side = _clean(value)
    if side is None:
        return None
    if side in {"left", "l"}:
        return "left"
    if side in {"right", "r"}:
        return "right"
    raise ValueError("side must be left or right")


def _touch_link_name(data: str) -> str | None:
    text = data.strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    for key in ("link_name", "link", "name", "target"):
        value = _clean(payload.get(key))
        if value:
            return value
    return text


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text.casefold() if text else None
