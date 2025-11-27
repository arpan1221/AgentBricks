"""Shared pytest fixtures for integration tests."""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

import httpx
import pytest
import pytest_asyncio
from kafka import KafkaConsumer
from pymongo import MongoClient

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def kafka_bootstrap_servers() -> str:
    """Kafka bootstrap servers for integration tests."""
    return os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")


@pytest.fixture(scope="session")
def mongo_uri() -> str:
    """MongoDB URI for integration tests."""
    return os.getenv("MONGO_URI", "mongodb://admin:password@localhost:27017")


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """API base URL for integration tests."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def wait_for_services(kafka_bootstrap_servers: str, mongo_uri: str, api_base_url: str) -> None:
    """Wait for all services to be ready before running tests."""
    max_wait = 120  # 2 minutes
    wait_interval = 5
    waited = 0

    # Wait for Kafka
    kafka_ready = False
    while waited < max_wait and not kafka_ready:
        try:
            consumer = KafkaConsumer(
                bootstrap_servers=kafka_bootstrap_servers.split(","), consumer_timeout_ms=1000
            )
            consumer.close()
            kafka_ready = True
        except Exception:
            time.sleep(wait_interval)
            waited += wait_interval

    if not kafka_ready:
        pytest.skip("Kafka is not available")

    # Wait for MongoDB
    mongo_ready = False
    while waited < max_wait and not mongo_ready:
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            client.admin.command("ping")
            client.close()
            mongo_ready = True
        except Exception:
            time.sleep(wait_interval)
            waited += wait_interval

    if not mongo_ready:
        pytest.skip("MongoDB is not available")

    # Wait for API
    api_ready = False
    while waited < max_wait and not api_ready:
        try:
            response = httpx.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                api_ready = True
        except Exception:
            time.sleep(wait_interval)
            waited += wait_interval

    if not api_ready:
        pytest.skip("API is not available")


@pytest.fixture
def kafka_consumer(kafka_bootstrap_servers: str) -> KafkaConsumer:
    """Kafka consumer for reading events."""
    import uuid

    consumer = KafkaConsumer(
        bootstrap_servers=kafka_bootstrap_servers.split(","),
        group_id=f"test-consumer-{uuid.uuid4().hex[:8]}",
        auto_offset_reset="latest",  # Only read new messages after subscription
        enable_auto_commit=True,
        consumer_timeout_ms=20000,
        value_deserializer=lambda m: m.decode("utf-8") if m else None,
    )
    yield consumer
    consumer.close()


@pytest.fixture
def mongo_client(mongo_uri: str) -> MongoClient:
    """MongoDB client for integration tests."""
    client = MongoClient(mongo_uri)
    yield client
    client.close()


@pytest_asyncio.fixture
async def api_client(api_base_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for API integration tests."""
    async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def test_event_data() -> Dict[str, Any]:
    """Test event data for integration tests."""
    # Use Z suffix for UTC timezone (ISO 8601 format)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return {
        "user_id": "test_user_integration",
        "movie_id": "test_movie_integration",
        "timestamp": timestamp,
        "watch_time_seconds": 3600,
        "session_id": "test_session_integration",
        "device": "desktop",
    }
