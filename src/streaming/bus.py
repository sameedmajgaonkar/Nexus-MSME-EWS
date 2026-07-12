"""Event-bus abstraction for the streaming demo (plan.md §14).

Two interchangeable implementations behind one publish/subscribe API:

  InProcessBus — the local default: a thread-safe queue drained by a daemon
      consumer thread. No Docker daemon is available in this environment, so
      this is the path actually exercised; it preserves the exact event flow
      (publish -> bus -> consumer -> feature refresh -> re-score -> alert).
  KafkaBus — activates when KAFKA_BOOTSTRAP_SERVERS is set: kafka-python
      against Redpanda (Kafka-wire-compatible, the docker-compose 'redpanda'
      service) or any Kafka cluster. Coded but not exercised locally.

Every published event is stamped with an event_id + ts; publish returns the
event_id so callers (POST /api/events/simulate) can hand it back to the UI.
"""

import json
import logging
import os
import queue
import threading
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Callable

logger = logging.getLogger("event_bus")

POLL_TIMEOUT_S = 0.2


def _stamp(event: dict) -> dict:
    event = dict(event)
    event.setdefault("event_id", uuid.uuid4().hex)
    event.setdefault("ts", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    return event


class InProcessBus:
    """Thread-safe in-process bus: one daemon consumer thread drains the queue
    and dispatches each event to every handler subscribed to its topic."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._handlers: dict[str, list[Callable[[dict], None]]] = defaultdict(list)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def publish(self, topic: str, event: dict) -> str:
        event = _stamp(event)
        self._queue.put((topic, event))
        logger.info("published topic=%s event_id=%s", topic, event["event_id"])
        return event["event_id"]

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._handlers[topic].append(handler)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="bus-consumer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                topic, event = self._queue.get(timeout=POLL_TIMEOUT_S)
            except queue.Empty:
                continue
            for handler in self._handlers.get(topic, []):
                try:
                    handler(event)
                except Exception:  # a bad event must never kill the consumer
                    logger.exception(
                        "handler failed topic=%s event_id=%s", topic, event.get("event_id")
                    )
            self._queue.task_done()


class KafkaBus:
    """kafka-python bus for Redpanda/Kafka (KAFKA_BOOTSTRAP_SERVERS set).

    Wire-compatible with the InProcessBus API. Not exercised locally (no
    Docker daemon to run the docker-compose 'redpanda' service); the import
    is lazy so environments without a broker never touch kafka-python."""

    def __init__(self, bootstrap_servers: str) -> None:
        from kafka import KafkaProducer  # lazy: only when a broker is configured

        self._bootstrap = bootstrap_servers
        self._producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        self._consumer_threads: list[threading.Thread] = []
        self._stop = threading.Event()
        self._subscriptions: list[tuple[str, Callable[[dict], None]]] = []

    def publish(self, topic: str, event: dict) -> str:
        event = _stamp(event)
        self._producer.send(topic, event)
        self._producer.flush()
        return event["event_id"]

    def subscribe(self, topic: str, handler: Callable[[dict], None]) -> None:
        self._subscriptions.append((topic, handler))

    def start(self) -> None:
        from kafka import KafkaConsumer

        for topic, handler in self._subscriptions:
            def _run(topic=topic, handler=handler) -> None:
                consumer = KafkaConsumer(
                    topic,
                    bootstrap_servers=self._bootstrap,
                    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                    auto_offset_reset="latest",
                    consumer_timeout_ms=int(POLL_TIMEOUT_S * 1000),
                )
                while not self._stop.is_set():
                    for message in consumer:
                        try:
                            handler(message.value)
                        except Exception:
                            logger.exception("kafka handler failed topic=%s", topic)
                consumer.close()

            t = threading.Thread(target=_run, name=f"kafka-consumer-{topic}", daemon=True)
            t.start()
            self._consumer_threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        for t in self._consumer_threads:
            t.join(timeout=2.0)
        self._consumer_threads.clear()
        self._producer.close()


def get_bus():
    """KafkaBus when KAFKA_BOOTSTRAP_SERVERS is set, InProcessBus otherwise."""
    servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS")
    if servers:
        return KafkaBus(servers)
    return InProcessBus()
