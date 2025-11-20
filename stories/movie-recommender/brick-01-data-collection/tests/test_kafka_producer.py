"""Tests for Kafka event producer."""

import pytest
import pytest_asyncio
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.kafka_producer import (
    KafkaEventProducer,
    create_producer_from_env,
    EVENT_TOPIC_MAP
)


class TestKafkaEventProducer:
    """Tests for KafkaEventProducer class."""

    @pytest.mark.asyncio
    async def test_producer_start_initializes_successfully(self, mock_aiokafka_producer):
        """Test that producer starts successfully."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        await producer.start()

        assert producer.producer is not None
        mock_aiokafka_producer.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_producer_stop_gracefully_shuts_down(self, mock_aiokafka_producer):
        """Test that producer stops gracefully."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        await producer.start()
        await producer.stop()

        mock_aiokafka_producer.stop.assert_called_once()
        assert producer.producer is None

    @pytest.mark.asyncio
    async def test_producer_send_event_with_valid_data_succeeds(self, mock_aiokafka_producer):
        """Test that sending event with valid data succeeds."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        await producer.start()

        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": 3600
        }

        await producer.send_event("view", event_data)

        mock_aiokafka_producer.send.assert_called_once()
        call_args = mock_aiokafka_producer.send.call_args
        assert call_args.kwargs["topic"] == "movie-events"
        assert call_args.kwargs["key"] == "user_123"
        assert call_args.kwargs["value"] == event_data

    @pytest.mark.asyncio
    async def test_producer_send_event_with_topic_prefix_succeeds(self, mock_aiokafka_producer):
        """Test that topic prefix is applied correctly."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix="dev-"
        )

        await producer.start()

        event_data = {"user_id": "user_123", "movie_id": "movie_456"}

        await producer.send_event("view", event_data)

        call_args = mock_aiokafka_producer.send.call_args
        assert call_args.kwargs["topic"] == "dev-movie-events"

    @pytest.mark.asyncio
    async def test_producer_send_event_without_user_id_raises_error(self, mock_aiokafka_producer):
        """Test that event without user_id raises error."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        await producer.start()

        event_data = {"movie_id": "movie_456"}  # Missing user_id

        with pytest.raises(ValueError, match="user_id"):
            await producer.send_event("view", event_data)

    @pytest.mark.asyncio
    async def test_producer_send_event_with_invalid_event_type_raises_error(self, mock_aiokafka_producer):
        """Test that invalid event type raises error."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        await producer.start()

        event_data = {"user_id": "user_123", "movie_id": "movie_456"}

        with pytest.raises(ValueError, match="Unknown event_type"):
            await producer.send_event("invalid_type", event_data)

    @pytest.mark.asyncio
    async def test_producer_send_event_without_start_raises_error(self, mock_aiokafka_producer):
        """Test that sending event without starting raises error."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        event_data = {"user_id": "user_123", "movie_id": "movie_456"}

        with pytest.raises(RuntimeError, match="not started"):
            await producer.send_event("view", event_data)

    @pytest.mark.asyncio
    async def test_producer_send_event_retries_on_failure(self, mock_aiokafka_producer):
        """Test that producer retries on Kafka error."""
        # Import KafkaError from the mocked module
        from src.kafka_producer import KafkaError

        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )
        producer._retry_max_attempts = 3

        await producer.start()

        # Fail first 2 attempts, succeed on 3rd
        mock_aiokafka_producer.send.side_effect = [
            KafkaError("Connection error"),
            KafkaError("Connection error"),
            None  # Success on 3rd attempt
        ]

        event_data = {"user_id": "user_123", "movie_id": "movie_456"}

        await producer.send_event("view", event_data)

        # Should have been called 3 times
        assert mock_aiokafka_producer.send.call_count == 3

    @pytest.mark.asyncio
    async def test_producer_send_event_fails_after_max_retries(self, mock_aiokafka_producer):
        """Test that producer raises error after max retries."""
        # Import KafkaError from the mocked module
        from src.kafka_producer import KafkaError

        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )
        producer._retry_max_attempts = 3

        await producer.start()

        # Always fail
        mock_aiokafka_producer.send.side_effect = KafkaError("Connection error")

        event_data = {"user_id": "user_123", "movie_id": "movie_456"}

        with pytest.raises(KafkaError):
            await producer.send_event("view", event_data)

        # Should have been called 3 times (max retries)
        assert mock_aiokafka_producer.send.call_count == 3

    @pytest.mark.asyncio
    async def test_producer_context_manager_starts_and_stops(self, mock_aiokafka_producer):
        """Test that async context manager properly starts and stops."""
        async with KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        ) as producer:
            assert producer.producer is not None
            mock_aiokafka_producer.start.assert_called_once()

        mock_aiokafka_producer.stop.assert_called_once()

    @pytest.mark.parametrize("event_type", ["view", "rating", "search", "skip"])
    def test_get_topic_name_with_valid_event_types_returns_correct_topic(self, event_type, mock_aiokafka_producer):
        """Test that topic name generation works for all event types."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        topic = producer._get_topic_name(event_type)
        assert topic == EVENT_TOPIC_MAP[event_type]

    def test_get_topic_name_with_topic_prefix_applies_prefix(self, mock_aiokafka_producer):
        """Test that topic prefix is applied to topic name."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix="prod-"
        )

        topic = producer._get_topic_name("view")
        assert topic == "prod-movie-events"

    def test_get_topic_name_with_invalid_event_type_raises_error(self, mock_aiokafka_producer):
        """Test that invalid event type raises error."""
        producer = KafkaEventProducer(
            bootstrap_servers="localhost:9092",
            topic_prefix=""
        )

        with pytest.raises(ValueError, match="Unknown event_type"):
            producer._get_topic_name("invalid_type")

    def test_producer_init_with_empty_bootstrap_servers_raises_error(self, mock_aiokafka_producer):
        """Test that empty bootstrap servers raises error."""
        with pytest.raises(ValueError, match="cannot be empty"):
            KafkaEventProducer(
                bootstrap_servers="",
                topic_prefix=""
            )


class TestCreateProducerFromEnv:
    """Tests for create_producer_from_env function."""

    @patch.dict("os.environ", {
        "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
        "KAFKA_TOPIC_PREFIX": "test-"
    })
    def test_create_producer_from_env_loads_configuration(self, mock_aiokafka_producer):
        """Test that producer is created with environment variables."""
        producer = create_producer_from_env()

        assert producer.bootstrap_servers == "test-kafka:9092"
        assert producer.topic_prefix == "test-"

    @patch.dict("os.environ", {}, clear=True)
    def test_create_producer_from_env_uses_defaults(self, mock_aiokafka_producer):
        """Test that producer uses defaults when env vars not set."""
        producer = create_producer_from_env()

        assert producer.bootstrap_servers == "localhost:9092"
        assert producer.topic_prefix == ""

    @patch.dict("os.environ", {
        "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
        "KAFKA_RETRY_MAX_ATTEMPTS": "5"
    })
    def test_create_producer_from_env_loads_retry_attempts(self, mock_aiokafka_producer):
        """Test that retry attempts are loaded from environment."""
        producer = create_producer_from_env()

        assert producer._retry_max_attempts == 5

    @patch.dict("os.environ", {
        "KAFKA_BOOTSTRAP_SERVERS": "test-kafka:9092",
        "KAFKA_RETRY_MAX_ATTEMPTS": "invalid"
    })
    def test_create_producer_from_env_handles_invalid_retry_attempts(self, mock_aiokafka_producer):
        """Test that invalid retry attempts are handled gracefully."""
        producer = create_producer_from_env()

        # Should use default (3)
        assert producer._retry_max_attempts == 3
