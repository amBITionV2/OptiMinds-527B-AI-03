# batch_consumer.py
# Simple Kafka consumer wrapper that triggers pull_images.fetch_and_process for messages.

import json
import signal
import sys
from typing import Optional

from confluent_kafka import Consumer, KafkaError, KafkaException

from .pull_images import fetch_and_process

conf = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "my-group",
    "enable.auto.commit": False,
    # Make the allowed processing gap very large
    "max.poll.interval.ms": 86400000,
    # Usual session settings
    "session.timeout.ms": 45000,         # 45s (heartbeats handle liveness)
    "auto.offset.reset": "earliest",
}

KAFKA_CONF = {
    "bootstrap.servers": "localhost:9092",
    "group.id": "batchexec-consumer",
    "auto.offset.reset": "earliest",
    "enable.auto.commit": False,  # we commit manually AFTER successful processing
}
KAFKA_TOPIC = "batchexec"


_running = True


def _graceful_shutdown(signum, frame):
    global _running
    _running = False


def parse_payload(raw: str) -> Optional[tuple[int, int]]:
    """
    Accepts either:
      - JSON: {"userId": 1, "testId": 42}
      - Colon format: "1:42" (i.e., "<userId>:<testId>")

    Returns (user_id, test_id) or None if invalid.
    """
    # 1) Try JSON first (backward compatible)
    try:
        data = json.loads(raw)
        user_id = int(data["userId"])
        test_id = int(data["testId"])
        return user_id, test_id
    except Exception:
        pass  # fall through to colon format

    # 2) Try "number:number"
    try:
        parts = raw.strip().split(":")
        if len(parts) != 2:
            raise ValueError("Expected two parts split by ':'")
        user_id = int(parts[0].strip())
        test_id = int(parts[1].strip())
        return user_id, test_id
    except Exception as e:
        print(f"[WARN] Invalid message '{raw}': {e}")
        return None


def handle_message(payload: str, consumer: Consumer, message) -> None:
    """
    Synchronously process one message: pull images for userId/testId.
    Commit offset only after success.
    """
    parsed = parse_payload(payload)
    if not parsed:
        # skip invalid message, commit so we don't loop forever on bad payloads
        consumer.commit(message=message, asynchronous=False)
        return

    user_id, test_id = parsed
    print(f"→ Start batch: userId={user_id}, testId={test_id}")

    try:
        stats = fetch_and_process(user_id=user_id, task_id=test_id)
        print(f"  Pulled {stats['fetched']} images from DB")
        print("  ✔ Batch done & offset committed\n")
        # On success commit offset
        consumer.commit(message=message, asynchronous=False)

    except Exception as e:
        # Do NOT commit; message will be retried depending on your policy
        print(f"  ✖ Error during batch: {e}\n")
        # Optionally move the message to a DLQ


def main():
    # graceful shutdown
    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    consumer = Consumer(KAFKA_CONF)
    consumer.subscribe([KAFKA_TOPIC])
    print(f"Listening for messages on topic '{KAFKA_TOPIC}'… Ctrl+C to stop.")

    try:
        while _running:
            msg = consumer.poll(1.0)
            if msg is None:
                continue

            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                raise KafkaException(msg.error())

            payload = msg.value().decode("utf-8").strip()
            handle_message(payload, consumer, msg)

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping consumer…")
        consumer.close()


if __name__ == "__main__":
    main()
