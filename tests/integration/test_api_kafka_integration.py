"""Integration tests for API -> Kafka flow.

Tests that events sent to the API are properly published to Kafka.
"""

import json
import time
from typing import Any, Dict

import pytest
from kafka.admin import KafkaAdminClient
from kafka.admin.new_topic import NewTopic


@pytest.mark.integration
class TestAPIToKafkaIntegration:
    """Test API to Kafka event publishing."""

    @pytest.mark.asyncio
    async def test_view_event_published_to_kafka(
        self,
        api_client,
        kafka_consumer,
        test_event_data: Dict[str, Any],
        wait_for_services,
        kafka_bootstrap_servers,
    ):
        """Test that view event is published to Kafka."""
        topic = "movie-events"

        # Ensure topic exists by creating it if needed
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=kafka_bootstrap_servers.split(","), client_id="test-admin"
            )
            topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            admin_client.close()
            time.sleep(1)  # Wait for topic to be fully created
        except Exception:
            # Topic might already exist, which is fine
            pass

        # Subscribe to topic first (before sending)
        kafka_consumer.subscribe([topic])

        # Wait for consumer to be ready and get partition assignment
        # Poll until we get partition assignment
        assignment_ready = False
        for _ in range(10):
            kafka_consumer.poll(timeout_ms=1000)
            if kafka_consumer.assignment():
                assignment_ready = True
                break
            time.sleep(0.5)

        assert assignment_ready, "Consumer partition assignment not ready"
        time.sleep(1)  # Additional wait to ensure consumer is fully ready

        # Send event to API
        response = await api_client.post("/events/view", json=test_event_data)
        assert response.status_code == 201

        # Wait for Kafka to process and make message available
        time.sleep(3)

        # Consume message from Kafka using poll
        messages = []
        max_attempts = 20

        for _ in range(max_attempts):
            msg_pack = kafka_consumer.poll(timeout_ms=2000)
            if msg_pack:
                for topic_partition, msgs in msg_pack.items():
                    for msg in msgs:
                        messages.append(msg)
                if len(messages) >= 1:
                    break
            time.sleep(0.3)

        assert len(messages) > 0, "No messages received from Kafka"

        # Verify message content
        message = messages[0]
        event_data = json.loads(message.value)

        assert event_data["event_type"] == "view"
        assert event_data["user_id"] == test_event_data["user_id"]
        assert event_data["movie_id"] == test_event_data["movie_id"]
        assert "event_id" in event_data
        assert "timestamp" in event_data

    @pytest.mark.asyncio
    async def test_rating_event_published_to_kafka(
        self, api_client, kafka_consumer, wait_for_services, kafka_bootstrap_servers
    ):
        """Test that rating event is published to Kafka."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        rating_event = {
            "user_id": "test_user_rating",
            "movie_id": "test_movie_rating",
            "rating": 5,
            "timestamp": timestamp,
        }

        topic = "movie-events"

        # Ensure topic exists
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=kafka_bootstrap_servers.split(","), client_id="test-admin"
            )
            topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            admin_client.close()
            time.sleep(1)
        except Exception:
            pass

        # Subscribe to topic first (before sending)
        kafka_consumer.subscribe([topic])

        # Wait for consumer to be ready and get partition assignment
        assignment_ready = False
        for _ in range(10):
            kafka_consumer.poll(timeout_ms=1000)
            if kafka_consumer.assignment():
                assignment_ready = True
                break
            time.sleep(0.5)

        assert assignment_ready, "Consumer partition assignment not ready"
        time.sleep(1)  # Additional wait to ensure consumer is fully ready

        # Send event to API
        response = await api_client.post("/events/rating", json=rating_event)
        assert response.status_code == 201

        # Wait for Kafka to process
        time.sleep(3)

        # Consume message from Kafka using poll
        messages = []
        max_attempts = 20

        for _ in range(max_attempts):
            msg_pack = kafka_consumer.poll(timeout_ms=2000)
            if msg_pack:
                for topic_partition, msgs in msg_pack.items():
                    for msg in msgs:
                        messages.append(msg)
                if len(messages) >= 1:
                    break
            time.sleep(0.3)

        assert len(messages) > 0, "No messages received from Kafka"

        # Verify message content
        message = messages[0]
        event_data = json.loads(message.value)

        assert event_data["event_type"] == "rating"
        assert event_data["user_id"] == rating_event["user_id"]
        assert event_data["movie_id"] == rating_event["movie_id"]
        assert event_data["rating"] == rating_event["rating"]

    @pytest.mark.asyncio
    async def test_multiple_events_published_to_kafka(
        self,
        api_client,
        kafka_consumer,
        test_event_data: Dict[str, Any],
        wait_for_services,
        kafka_bootstrap_servers,
    ):
        """Test that multiple events are published to Kafka."""
        from datetime import datetime, timezone

        topic = "movie-events"

        # Ensure topic exists
        try:
            admin_client = KafkaAdminClient(
                bootstrap_servers=kafka_bootstrap_servers.split(","), client_id="test-admin"
            )
            topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
            admin_client.create_topics(new_topics=topic_list, validate_only=False)
            admin_client.close()
            time.sleep(1)
        except Exception:
            pass

        # Subscribe to topic first (before sending)
        kafka_consumer.subscribe([topic])

        # Wait for consumer to be ready and get partition assignment
        assignment_ready = False
        for _ in range(10):
            kafka_consumer.poll(timeout_ms=1000)
            if kafka_consumer.assignment():
                assignment_ready = True
                break
            time.sleep(0.5)

        assert assignment_ready, "Consumer partition assignment not ready"
        time.sleep(1)  # Additional wait to ensure consumer is fully ready

        # Send multiple events
        num_events = 5
        for i in range(num_events):
            event = test_event_data.copy()
            event["user_id"] = f"user_{i}"
            event["movie_id"] = f"movie_{i}"
            # Update timestamp for each event
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            event["timestamp"] = timestamp

            response = await api_client.post("/events/view", json=event)
            assert response.status_code == 201

        # Wait for Kafka to process all events
        time.sleep(4)

        # Consume messages from Kafka using poll
        messages = []
        max_attempts = 30

        for _ in range(max_attempts):
            msg_pack = kafka_consumer.poll(timeout_ms=2000)
            if msg_pack:
                for topic_partition, msgs in msg_pack.items():
                    for msg in msgs:
                        messages.append(msg)
                if len(messages) >= num_events:
                    break
            time.sleep(0.3)

        assert (
            len(messages) >= num_events
        ), f"Expected at least {num_events} messages, got {len(messages)}"

        # Verify all messages are valid
        user_ids = set()
        for message in messages:
            event_data = json.loads(message.value)
            assert event_data["event_type"] == "view"
            user_ids.add(event_data["user_id"])

        assert len(user_ids) == num_events, "All events should have unique user IDs"
