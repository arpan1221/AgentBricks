"""Shared pytest fixtures for brick-01-data-collection tests."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch
import sys
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from fastapi import FastAPI
from fastapi.testclient import TestClient
import httpx

# Test data fixtures
@pytest.fixture
def view_event_data() -> Dict[str, Any]:
    """Valid view event data."""
    return {
        "user_id": "user_123",
        "movie_id": "movie_456",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "watch_time_seconds": 3600,
        "session_id": "session_789",
        "device": "desktop"
    }


@pytest.fixture
def rating_event_data() -> Dict[str, Any]:
    """Valid rating event data."""
    return {
        "user_id": "user_123",
        "movie_id": "movie_456",
        "rating": 5,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def search_event_data() -> Dict[str, Any]:
    """Valid search event data."""
    return {
        "user_id": "user_123",
        "query": "action movies",
        "results_count": 25,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def skip_event_data() -> Dict[str, Any]:
    """Valid skip event data."""
    return {
        "user_id": "user_123",
        "movie_id": "movie_456",
        "watch_duration_seconds": 600,
        "movie_duration_seconds": 5400,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@pytest.fixture
def future_timestamp() -> datetime:
    """Future timestamp for testing validation."""
    return datetime.now(timezone.utc) + timedelta(days=1)


@pytest.fixture
def past_timestamp() -> datetime:
    """Past timestamp for testing."""
    return datetime.now(timezone.utc) - timedelta(days=1)


# FastAPI test client fixtures
@pytest.fixture
def test_app() -> FastAPI:
    """FastAPI test application."""
    # Import here to avoid circular imports
    from src.api import app
    return app


@pytest.fixture
def test_client(test_app: FastAPI) -> TestClient:
    """Synchronous test client."""
    return TestClient(test_app)


@pytest_asyncio.fixture
async def async_client(test_app: FastAPI) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async test client."""
    async with httpx.AsyncClient(app=test_app, base_url="http://test") as client:
        yield client


# Kafka producer mock fixtures
@pytest.fixture
def kafka_producer_mock() -> AsyncMock:
    """Mock Kafka producer."""
    mock = AsyncMock()
    mock.publish = AsyncMock(return_value=None)  # API uses publish method
    mock.send = AsyncMock(return_value=None)
    mock.start = AsyncMock(return_value=None)
    mock.stop = AsyncMock(return_value=None)
    return mock


@pytest.fixture(autouse=True)
def mock_aiokafka_producer(monkeypatch):
    """Mock AIOKafkaProducer class - auto-use for all tests."""
    mock_producer_class = MagicMock()
    mock_producer_instance = AsyncMock()
    mock_producer_class.return_value = mock_producer_instance
    mock_producer_instance.start = AsyncMock()
    mock_producer_instance.stop = AsyncMock()
    mock_producer_instance.send = AsyncMock()

    # Create a mock KafkaError
    class MockKafkaError(Exception):
        pass

    # Patch the import check to make AIOKAFKA_AVAILABLE = True
    monkeypatch.setattr('src.kafka_producer.AIOKAFKA_AVAILABLE', True)

    # Patch the AIOKafkaProducer class
    monkeypatch.setattr(
        'src.kafka_producer.AIOKafkaProducer',
        mock_producer_class
    )

    # Patch KafkaError
    monkeypatch.setattr('src.kafka_producer.KafkaError', MockKafkaError)

    # Mock aiokafka module if not already imported
    if 'aiokafka' not in sys.modules:
        mock_aiokafka_module = MagicMock()
        mock_errors_module = MagicMock()
        mock_errors_module.KafkaError = MockKafkaError
        mock_aiokafka_module.errors = mock_errors_module
        sys.modules['aiokafka'] = mock_aiokafka_module
        sys.modules['aiokafka.errors'] = mock_errors_module

    return mock_producer_instance
