from __future__ import annotations

import time
from dataclasses import dataclass
from queue import Queue
from threading import Event, RLock, Thread
from typing import Any, Callable, Protocol

GAZE_TOPIC = "/HOLO1_GazePoint"
USER_TRANSFORM_TOPIC = "/HOLO1_Transform"
MANUAL_TARGET_TOPIC = "/Robot/target_manual"
HOLO1_CONTROL_TOPIC = "/WorkerPool/control"

DEFAULT_SENSOR_TOPICS: dict[str, tuple[str, str]] = {
    "gaze": (GAZE_TOPIC, "std_msgs/String"),
    "user_transform": (USER_TRANSFORM_TOPIC, "geometry_msgs/Pose"),
    "manual_target": (MANUAL_TARGET_TOPIC, "geometry_msgs/Pose"),
}


@dataclass(frozen=True)
class TopicReading:
    topic: str
    message_type: str
    payload: dict[str, Any]
    received_at_s: float


class VizorSensorTransport(Protocol):
    host: str
    port: int

    def is_connected(self) -> bool: ...
    def latest_message(self, topic: str) -> TopicReading | None: ...
    def add_listener(self, listener: Callable[[TopicReading], None]) -> None: ...


class FakeVizorRosTransport:
    def __init__(
        self,
        *,
        now_s: float = 0.0,
        connected: bool = True,
        host: str = "localhost",
        port: int = 9090,
    ) -> None:
        self.host = host
        self.port = port
        self.now_s = now_s
        self._connected = connected
        self._messages: dict[str, TopicReading] = {}
        self._listeners: list[Callable[[TopicReading], None]] = []
        self.published: list[tuple[str, str, dict[str, Any]]] = []

    def is_connected(self) -> bool:
        return self._connected

    def set_connected(self, connected: bool) -> None:
        self._connected = connected

    def latest_message(self, topic: str) -> TopicReading | None:
        return self._messages.get(topic)

    def add_listener(self, listener: Callable[[TopicReading], None]) -> None:
        self._listeners.append(listener)

    def record_message(
        self,
        topic: str,
        message_type: str,
        payload: dict[str, Any],
        *,
        received_at_s: float | None = None,
    ) -> None:
        reading = TopicReading(
            topic=topic,
            message_type=message_type,
            payload=payload,
            received_at_s=self.now_s if received_at_s is None else received_at_s,
        )
        self._messages[topic] = reading
        for listener in list(self._listeners):
            listener(reading)

    def publish(self, topic: str, message_type: str, payload: dict[str, Any]) -> None:
        self.published.append((topic, message_type, payload))


class RoslibpyVizorRosTransport:
    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 9090,
        topics: dict[str, tuple[str, str]] | None = None,
        time_fn: Any = time.monotonic,
    ) -> None:
        import roslibpy

        self.roslibpy = roslibpy
        self.host = host
        self.port = port
        self._topics = topics or DEFAULT_SENSOR_TOPICS
        self._time_fn = time_fn
        self._lock = RLock()
        self._messages: dict[str, TopicReading] = {}
        self._listeners: list[Callable[[TopicReading], None]] = []
        self._subscribers: dict[str, Any] = {}
        self._tracking_keepalive_stop: Event | None = None
        self._tracking_keepalive_thread: Thread | None = None
        self._connected = False
        self.client = roslibpy.Ros(host=host, port=port)

    def connect(self) -> None:
        self.subscribe_defaults()
        try:
            self.client.run()
        except Exception as exc:
            if exc.__class__.__name__ != "RosTimeoutError":
                raise
        self._connected = True

    def close(self) -> None:
        self.stop_holo1_tracking_keepalive()
        for subscriber in list(self._subscribers.values()):
            try:
                subscriber.unsubscribe()
            except Exception:
                pass
        self._subscribers.clear()
        self.client.terminate()
        self._connected = False

    def is_connected(self) -> bool:
        return bool(self._connected and self.client.is_connected)

    def latest_message(self, topic: str) -> TopicReading | None:
        with self._lock:
            return self._messages.get(topic)

    def add_listener(self, listener: Callable[[TopicReading], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def subscribe_defaults(self) -> None:
        for topic, message_type in self._topics.values():
            self.subscribe(topic, message_type)

    def subscribe(self, topic: str, message_type: str) -> None:
        if topic in self._subscribers:
            return
        subscriber = self.roslibpy.Topic(self.client, topic, message_type)
        subscriber.subscribe(lambda msg, t=topic, mt=message_type: self._record(t, mt, msg))
        self._subscribers[topic] = subscriber

    def enable_holo1_tracking(self) -> None:
        publisher = self.roslibpy.Topic(self.client, HOLO1_CONTROL_TOPIC, "std_msgs/String")
        publisher.publish(self.roslibpy.Message({"data": "HOLO1_position_on"}))

    def start_holo1_tracking_keepalive(self, *, interval_s: float = 10.0) -> None:
        self.enable_holo1_tracking()
        if interval_s <= 0:
            return
        with self._lock:
            if self._tracking_keepalive_thread is not None:
                return
            stop_event = Event()
            thread = Thread(
                target=self._tracking_keepalive_loop,
                args=(stop_event, float(interval_s)),
                daemon=True,
            )
            self._tracking_keepalive_stop = stop_event
            self._tracking_keepalive_thread = thread
            thread.start()

    def stop_holo1_tracking_keepalive(self) -> None:
        with self._lock:
            stop_event = self._tracking_keepalive_stop
            thread = self._tracking_keepalive_thread
            self._tracking_keepalive_stop = None
            self._tracking_keepalive_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=1.0)

    def _tracking_keepalive_loop(self, stop_event: Event, interval_s: float) -> None:
        while not stop_event.wait(interval_s):
            if not self.is_connected():
                continue
            try:
                self.enable_holo1_tracking()
            except Exception:
                pass

    def _record(self, topic: str, message_type: str, payload: dict[str, Any]) -> None:
        reading = TopicReading(
            topic=topic,
            message_type=message_type,
            payload=payload,
            received_at_s=float(self._time_fn()),
        )
        with self._lock:
            self._messages[topic] = reading
            listeners = list(self._listeners)
        for listener in listeners:
            listener(reading)


class QueueingVizorRosTransport(FakeVizorRosTransport):
    """Small queue helper for future wait-based tests without ROSBridge."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.queues: dict[str, Queue[TopicReading]] = {}
