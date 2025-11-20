"""Tests for FastAPI endpoints."""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from src.api import app


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for /health endpoint."""

    async def test_health_endpoint_returns_200_with_healthy_status(self, async_client):
        """Test that health endpoint returns 200 with healthy status."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    async def test_health_endpoint_includes_request_id_header(self, async_client):
        """Test that health endpoint includes request ID header."""
        response = await async_client.get("/health")

        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
class TestMetricsEndpoint:
    """Tests for /metrics endpoint."""

    async def test_metrics_endpoint_returns_200(self, async_client):
        """Test that metrics endpoint returns 200."""
        response = await async_client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

    async def test_metrics_endpoint_returns_prometheus_format(self, async_client):
        """Test that metrics endpoint returns Prometheus format."""
        response = await async_client.get("/metrics")

        content = response.text
        assert "# HELP" in content or "# TYPE" in content


@pytest.mark.asyncio
class TestViewEventEndpoint:
    """Tests for POST /events/view endpoint."""

    async def test_view_event_with_valid_data_returns_201(self, async_client, kafka_producer_mock):
        """Test that valid view event returns 201."""
        # Mock the Kafka producer dependency
        async def mock_get_kafka_producer():
            yield kafka_producer_mock

        with patch("src.api.get_kafka_producer", side_effect=mock_get_kafka_producer):
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600,
                "device": "desktop"
            }

            response = await async_client.post("/events/view", json=event_data)

            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "success"
            assert data["event_type"] == "view"
            assert "event_id" in data
            assert data["user_id"] == "user_123"
            assert data["movie_id"] == "movie_456"

    async def test_view_event_with_invalid_data_returns_422(self, async_client):
        """Test that invalid view event returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": -1  # Invalid: negative
        }

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        data = response.json()
        # FastAPI default format: {"detail": [...]}
        # Custom handler format: {"status": "error", "error_type": "validation_error", ...}
        # Accept either format
        assert "detail" in data or ("status" in data and data.get("status") == "error")
        # Request ID should be in headers (always added by middleware)
        assert "X-Request-ID" in response.headers

    async def test_view_event_error_response_includes_request_id(self, async_client):
        """Test that error response includes request ID."""
        invalid_data = {"invalid": "data"}

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        # Request ID should be in headers
        assert "X-Request-ID" in response.headers
        # May also be in response body if custom handler is used
        data = response.json()
        # Either in body (custom handler) or just in headers (default handler)
        assert "X-Request-ID" in response.headers or "request_id" in data
        assert "X-Request-ID" in response.headers


@pytest.mark.asyncio
class TestRatingEventEndpoint:
    """Tests for POST /events/rating endpoint."""

    async def test_rating_event_with_valid_data_returns_201(self, async_client, kafka_producer_mock):
        """Test that valid rating event returns 201."""
        async def mock_get_kafka_producer():
            yield kafka_producer_mock

        with patch("src.api.get_kafka_producer", side_effect=mock_get_kafka_producer):
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "rating": 5,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            response = await async_client.post("/events/rating", json=event_data)

            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "success"
            assert data["event_type"] == "rating"
            assert data["rating"] == 5

    async def test_rating_event_with_invalid_rating_returns_422(self, async_client):
        """Test that invalid rating returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 6,  # Invalid: > 5
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        response = await async_client.post("/events/rating", json=invalid_data)

        assert response.status_code == 422


@pytest.mark.asyncio
class TestSearchEventEndpoint:
    """Tests for POST /events/search endpoint."""

    async def test_search_event_with_valid_data_returns_201(self, async_client, kafka_producer_mock):
        """Test that valid search event returns 201."""
        async def mock_get_kafka_producer():
            yield kafka_producer_mock

        with patch("src.api.get_kafka_producer", side_effect=mock_get_kafka_producer):
            event_data = {
                "user_id": "user_123",
                "query": "action movies",
                "results_count": 25,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            response = await async_client.post("/events/search", json=event_data)

            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "success"
            assert data["event_type"] == "search"
            assert data["query"] == "action movies"
            assert data["results_count"] == 25

    async def test_search_event_with_empty_query_returns_422(self, async_client):
        """Test that empty query returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "query": "",  # Invalid: empty
            "results_count": 25,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        response = await async_client.post("/events/search", json=invalid_data)

        assert response.status_code == 422


@pytest.mark.asyncio
class TestSkipEventEndpoint:
    """Tests for POST /events/skip endpoint."""

    async def test_skip_event_with_valid_data_returns_201(self, async_client, kafka_producer_mock):
        """Test that valid skip event returns 201."""
        async def mock_get_kafka_producer():
            yield kafka_producer_mock

        with patch("src.api.get_kafka_producer", side_effect=mock_get_kafka_producer):
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "watch_duration_seconds": 600,
                "movie_duration_seconds": 5400,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            response = await async_client.post("/events/skip", json=event_data)

            assert response.status_code == 201
            data = response.json()
            assert data["status"] == "success"
            assert data["event_type"] == "skip"

    async def test_skip_event_with_invalid_durations_returns_422(self, async_client):
        """Test that invalid durations return 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "watch_duration_seconds": 6000,  # Invalid: > movie_duration
            "movie_duration_seconds": 5400,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

        response = await async_client.post("/events/skip", json=invalid_data)

        assert response.status_code == 422


@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for error handling."""

    async def test_kafka_error_returns_500(self, async_client, test_app):
        """Test that Kafka errors return 500."""
        # Mock Kafka producer to raise error
        mock_producer = AsyncMock()
        mock_producer.publish = AsyncMock(side_effect=Exception("Kafka error"))

        # Create mock dependency function
        async def mock_get_producer():
            yield mock_producer

        # Use FastAPI's dependency override system
        from src.api import get_kafka_producer

        # Override the dependency
        original_overrides = dict(test_app.dependency_overrides)
        test_app.dependency_overrides[get_kafka_producer] = mock_get_producer

        try:
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600
            }

            response = await async_client.post("/events/view", json=event_data)

            # Should get 500 error when publish fails
            # Endpoint catches exception and raises HTTPException
            # HTTPException returns {"detail": "..."} format
            assert response.status_code == 500, (
                f"Expected 500, got {response.status_code}: {response.json()}"
            )
            data = response.json()
            # HTTPException format: {"detail": "..."}
            assert "detail" in data
            # Request ID should be in headers (always added by middleware)
            assert "X-Request-ID" in response.headers

            # Verify publish was called (should be called before exception)
            mock_producer.publish.assert_called_once()
        finally:
            # Restore original overrides
            test_app.dependency_overrides.clear()
            test_app.dependency_overrides.update(original_overrides)

    async def test_validation_error_returns_422_with_details(self, async_client):
        """Test that validation errors return 422 with error details."""
        invalid_data = {"invalid": "data"}

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        data = response.json()
        # FastAPI default format: {"detail": [{"loc": [...], "msg": "...", ...}]}
        # Custom handler format: {"status": "error", "error_type": "validation_error", "errors": [...]}
        # Accept either format
        assert "detail" in data or ("error_type" in data and "errors" in data)
        # If it's the default format, detail should be a list
        # If it's custom format, errors should be present
        if "detail" in data:
            assert isinstance(data["detail"], list)
        if "error_type" in data:
            assert data["error_type"] == "validation_error"
            assert "errors" in data


@pytest.mark.asyncio
class TestMiddleware:
    """Tests for middleware functionality."""

    async def test_request_id_middleware_adds_header(self, async_client):
        """Test that request ID middleware adds header to response."""
        response = await async_client.get("/health")

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0  # Should be UUID

    async def test_timing_middleware_adds_header(self, async_client):
        """Test that timing middleware adds process time header."""
        response = await async_client.get("/health")

        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0
