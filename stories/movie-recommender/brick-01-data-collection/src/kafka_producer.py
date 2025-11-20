"""Kafka event producer for publishing user interaction events.

This module provides an async Kafka producer for publishing validated events
to Kafka topics. Includes retry logic, error handling, and automatic topic
configuration.
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional

try:
    from aiokafka import AIOKafkaProducer
    from aiokafka.errors import KafkaError
    AIOKAFKA_AVAILABLE = True
except ImportError:
    AIOKAFKA_AVAILABLE = False
    # Create stub classes for type hints
    class AIOKafkaProducer:
        pass
    class KafkaError(Exception):
        pass

logger = logging.getLogger(__name__)

# Event type to topic name mapping
EVENT_TOPIC_MAP = {
    "view": "movie-events",
    "rating": "movie-events",
    "search": "movie-events",
    "skip": "movie-events",
}


class KafkaEventProducer:
    """
    Async Kafka event producer with retry logic and error handling.

    Publishes user interaction events to Kafka topics with automatic
    retry on failures, compression, and structured logging.

    Attributes:
        bootstrap_servers: Kafka broker addresses (comma-separated)
        topic_prefix: Prefix for topic names (e.g., "prod-", "dev-")
        producer: AIOKafkaProducer instance (created in start())

    Example:
        >>> import asyncio
        >>> from src.kafka_producer import KafkaEventProducer
        >>>
        >>> async def main():
        ...     producer = KafkaEventProducer(
        ...         bootstrap_servers="localhost:9092",
        ...         topic_prefix="dev-"
        ...     )
        ...
        ...     async with producer:
        ...         await producer.send_event(
        ...             event_type="view",
        ...             event_data={
        ...                 "user_id": "user_123",
        ...                 "movie_id": "movie_456",
        ...                 "timestamp": "2024-01-15T20:30:00Z"
        ...             }
        ...         )
        >>>
        >>> asyncio.run(main())
    """

    def __init__(
        self,
        bootstrap_servers: str,
        topic_prefix: str = ""
    ) -> None:
        """
        Initialize Kafka event producer.

        Args:
            bootstrap_servers: Kafka broker addresses (comma-separated string)
                             e.g., "localhost:9092" or "kafka1:9092,kafka2:9092"
            topic_prefix: Prefix for topic names (default: empty string)
                         e.g., "prod-" or "dev-"

        Raises:
            ValueError: If bootstrap_servers is empty
            ImportError: If aiokafka is not installed

        Example:
            >>> producer = KafkaEventProducer(
            ...     bootstrap_servers="localhost:9092",
            ...     topic_prefix="dev-"
            ... )
        """
        if not bootstrap_servers:
            raise ValueError("bootstrap_servers cannot be empty")

        if not AIOKAFKA_AVAILABLE:
            raise ImportError(
                "aiokafka is not installed. Install it with: pip install aiokafka"
            )

        self.bootstrap_servers = bootstrap_servers
        self.topic_prefix = topic_prefix
        self.producer: Optional[AIOKafkaProducer] = None
        self._retry_max_attempts = 3
        self._retry_base_delay = 0.1  # Base delay in seconds

        logger.debug(
            f"KafkaEventProducer initialized",
            extra={
                "bootstrap_servers": bootstrap_servers,
                "topic_prefix": topic_prefix,
            }
        )

    async def start(self) -> None:
        """
        Initialize and start the Kafka producer.

        Creates AIOKafkaProducer instance with compression and other
        production settings.

        Raises:
            KafkaError: If producer initialization fails
            ConnectionError: If cannot connect to Kafka brokers

        Example:
            >>> producer = KafkaEventProducer("localhost:9092")
            >>> await producer.start()
            >>> # Producer is ready to send events
        """
        if self.producer is not None:
            logger.warning("Producer already started, skipping start()")
            return

        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
                compression_type="gzip",
                acks="all",  # Wait for all replicas to acknowledge
                retries=0,  # We handle retries manually
                enable_idempotence=True,  # Prevent duplicates
                max_in_flight_requests_per_connection=5,
                request_timeout_ms=30000,  # 30 seconds
            )

            await self.producer.start()

            logger.info(
                "Kafka producer started",
                extra={
                    "bootstrap_servers": self.bootstrap_servers,
                    "topic_prefix": self.topic_prefix,
                }
            )

        except Exception as e:
            logger.error(
                "Failed to start Kafka producer",
                extra={
                    "bootstrap_servers": self.bootstrap_servers,
                    "error": str(e),
                },
                exc_info=True
            )
            raise

    async def stop(self) -> None:
        """
        Stop and clean up the Kafka producer.

        Performs graceful shutdown, ensuring all pending messages
        are sent before closing connections.

        Example:
            >>> await producer.start()
            >>> # ... send events ...
            >>> await producer.stop()
        """
        if self.producer is None:
            logger.warning("Producer not started, skipping stop()")
            return

        try:
            await self.producer.stop()
            self.producer = None

            logger.info("Kafka producer stopped")

        except Exception as e:
            logger.error(
                "Error stopping Kafka producer",
                extra={"error": str(e)},
                exc_info=True
            )
            # Set to None even if stop fails
            self.producer = None

    def _get_topic_name(self, event_type: str) -> str:
        """
        Get topic name for event type.

        Maps event type to topic name with optional prefix.

        Args:
            event_type: Event type (e.g., "view", "rating", "search", "skip")

        Returns:
            Topic name with prefix if configured

        Raises:
            ValueError: If event_type is not recognized

        Example:
            >>> producer = KafkaEventProducer("localhost:9092", topic_prefix="dev-")
            >>> topic = producer._get_topic_name("view")
            >>> print(topic)
            'dev-movie-events'
        """
        if event_type not in EVENT_TOPIC_MAP:
            raise ValueError(
                f"Unknown event_type: {event_type}. "
                f"Valid types: {list(EVENT_TOPIC_MAP.keys())}"
            )

        base_topic = EVENT_TOPIC_MAP[event_type]
        topic_name = f"{self.topic_prefix}{base_topic}" if self.topic_prefix else base_topic

        logger.debug(
            f"Topic name resolved",
            extra={
                "event_type": event_type,
                "topic_name": topic_name,
            }
        )

        return topic_name

    async def send_event(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ) -> None:
        """
        Send event to Kafka topic with retry logic.

        Publishes event to appropriate Kafka topic based on event type.
        Includes automatic retry with exponential backoff on failures.

        Args:
            event_type: Type of event (e.g., "view", "rating", "search", "skip")
            event_data: Event data dictionary (must include "user_id" as key)

        Raises:
            ValueError: If event_type is invalid or user_id is missing
            KafkaError: If all retry attempts fail
            RuntimeError: If producer is not started

        Example:
            >>> await producer.start()
            >>> await producer.send_event(
            ...     event_type="view",
            ...     event_data={
            ...         "user_id": "user_123",
            ...         "movie_id": "movie_456",
            ...         "timestamp": "2024-01-15T20:30:00Z",
            ...         "watch_time_seconds": 3600
            ...     }
            ... )
        """
        if self.producer is None:
            raise RuntimeError("Producer not started. Call start() first.")

        # Validate event_type
        if event_type not in EVENT_TOPIC_MAP:
            raise ValueError(
                f"Unknown event_type: {event_type}. "
                f"Valid types: {list(EVENT_TOPIC_MAP.keys())}"
            )

        # Validate user_id exists for partitioning
        if "user_id" not in event_data:
            raise ValueError("event_data must contain 'user_id' key for partitioning")

        user_id = event_data["user_id"]
        topic = self._get_topic_name(event_type)

        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(1, self._retry_max_attempts + 1):
            try:
                # Send message to Kafka
                await self.producer.send(
                    topic=topic,
                    key=user_id,
                    value=event_data
                )

                logger.debug(
                    "Event sent to Kafka",
                    extra={
                        "event_type": event_type,
                        "topic": topic,
                        "user_id": user_id,
                        "attempt": attempt,
                    }
                )

                # Success - return
                return

            except KafkaError as e:
                last_exception = e

                if attempt < self._retry_max_attempts:
                    # Calculate exponential backoff delay
                    delay = self._retry_base_delay * (2 ** (attempt - 1))

                    logger.warning(
                        "Kafka send failed, retrying",
                        extra={
                            "event_type": event_type,
                            "topic": topic,
                            "user_id": user_id,
                            "attempt": attempt,
                            "max_attempts": self._retry_max_attempts,
                            "delay_seconds": delay,
                            "error": str(e),
                        }
                    )

                    # Wait before retry
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error(
                        "Kafka send failed after all retries",
                        extra={
                            "event_type": event_type,
                            "topic": topic,
                            "user_id": user_id,
                            "attempts": attempt,
                            "error": str(e),
                        },
                        exc_info=True
                    )
                    raise

            except Exception as e:
                # Non-Kafka errors - don't retry
                logger.error(
                    "Unexpected error sending event to Kafka",
                    extra={
                        "event_type": event_type,
                        "topic": topic,
                        "user_id": user_id,
                        "error": str(e),
                    },
                    exc_info=True
                )
                raise

        # Should not reach here, but just in case
        if last_exception:
            raise last_exception

    async def __aenter__(self) -> "KafkaEventProducer":
        """
        Async context manager entry.

        Returns:
            KafkaEventProducer instance (started)
        """
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """
        Async context manager exit.

        Args:
            exc_type: Exception type (if any)
            exc_val: Exception value (if any)
            exc_tb: Exception traceback (if any)
        """
        await self.stop()


def create_producer_from_env() -> KafkaEventProducer:
    """
    Create KafkaEventProducer from environment variables.

    Reads configuration from environment:
        - KAFKA_BOOTSTRAP_SERVERS: Kafka broker addresses (required)
        - KAFKA_TOPIC_PREFIX: Topic prefix (optional, default: empty)
        - KAFKA_RETRY_MAX_ATTEMPTS: Max retry attempts (optional, default: 3)

    Returns:
        Configured KafkaEventProducer instance

    Raises:
        ValueError: If required environment variables are missing

    Example:
        >>> import os
        >>> os.environ["KAFKA_BOOTSTRAP_SERVERS"] = "localhost:9092"
        >>> os.environ["KAFKA_TOPIC_PREFIX"] = "dev-"
        >>> producer = create_producer_from_env()
    """
    bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    topic_prefix = os.getenv("KAFKA_TOPIC_PREFIX", "")

    # Validate bootstrap_servers
    if not bootstrap_servers:
        raise ValueError(
            "KAFKA_BOOTSTRAP_SERVERS environment variable is required"
        )

    producer = KafkaEventProducer(
        bootstrap_servers=bootstrap_servers,
        topic_prefix=topic_prefix
    )

    # Override retry attempts if provided
    retry_attempts = os.getenv("KAFKA_RETRY_MAX_ATTEMPTS")
    if retry_attempts:
        try:
            producer._retry_max_attempts = int(retry_attempts)
            if producer._retry_max_attempts < 1:
                raise ValueError("KAFKA_RETRY_MAX_ATTEMPTS must be >= 1")
        except ValueError as e:
            logger.warning(
                f"Invalid KAFKA_RETRY_MAX_ATTEMPTS: {retry_attempts}, "
                f"using default: {producer._retry_max_attempts}",
                extra={"error": str(e)}
            )

    logger.info(
        "Kafka producer created from environment",
        extra={
            "bootstrap_servers": bootstrap_servers,
            "topic_prefix": topic_prefix,
            "retry_max_attempts": producer._retry_max_attempts,
        }
    )

    return producer


# Default configuration values
DEFAULT_BOOTSTRAP_SERVERS = "localhost:9092"
DEFAULT_TOPIC_PREFIX = ""
DEFAULT_RETRY_MAX_ATTEMPTS = 3
