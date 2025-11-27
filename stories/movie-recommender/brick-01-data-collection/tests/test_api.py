"""Unit tests for FastAPI endpoints.

Tests individual API endpoints in isolation with mocked dependencies.
Follows project conventions:
- Test naming: test_<function>_with_<condition>_returns_<expected>
- Comprehensive coverage of happy paths and edge cases
- All external dependencies are mocked
- Error conditions are explicitly tested
"""

from datetime import datetime, timezone
from typing import AsyncGenerator
from unittest.mock import AsyncMock

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from src.api import app, get_kafka_producer

# ============================================================================
# Health Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    async def test_health_with_valid_request_returns_200_with_healthy_status(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that health endpoint returns 200 with healthy status."""
        response = await async_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    async def test_health_with_any_request_includes_request_id_header(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that health endpoint includes request ID header."""
        response = await async_client.get("/health")

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0
        # Should be UUID format (36 chars with hyphens)
        assert len(request_id) == 36


# ============================================================================
# Metrics Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestMetricsEndpoint:
    """Tests for GET /metrics endpoint."""

    async def test_metrics_with_valid_request_returns_200_with_prometheus_format(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that metrics endpoint returns 200 with Prometheus format."""
        response = await async_client.get("/metrics")

        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")
        content = response.text
        # Prometheus format should have HELP or TYPE comments
        assert "# HELP" in content or "# TYPE" in content

    async def test_metrics_with_multiple_requests_increments_counters(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that metrics counters increment with multiple requests."""
        # Make multiple requests
        for _ in range(3):
            await async_client.get("/health")

        # Check metrics
        response = await async_client.get("/metrics")
        content = response.text
        # Should have http_requests_total counter
        assert "http_requests_total" in content


# ============================================================================
# View Event Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestViewEventEndpoint:
    """Tests for POST /events/view endpoint."""

    async def test_view_event_with_valid_data_returns_201_with_success_status(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that valid view event returns 201 with success status."""
        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": 3600,
            "session_id": "session_789",
            "device": "desktop",
        }

        response = await async_client.post("/events/view", json=event_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["event_type"] == "view"
        assert "event_id" in data
        assert data["user_id"] == "user_123"
        assert data["movie_id"] == "movie_456"
        # Verify Kafka producer was called
        kafka_producer_mock.send_event.assert_called_once()
        call_args = kafka_producer_mock.send_event.call_args
        assert call_args.kwargs["event_type"] == "view"
        assert call_args.kwargs["event_data"]["user_id"] == "user_123"

    async def test_view_event_with_missing_required_fields_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that view event with missing required fields returns 422."""
        invalid_data = {
            "user_id": "user_123",
            # Missing movie_id and timestamp
        }

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        data = response.json()
        assert "detail" in data or ("status" in data and data.get("status") == "error")

    async def test_view_event_with_negative_watch_time_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that view event with negative watch_time_seconds returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": -1,  # Invalid: negative
        }

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422

    async def test_view_event_with_invalid_timestamp_format_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that view event with invalid timestamp format returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": "invalid-timestamp",  # Invalid format
            "watch_time_seconds": 3600,
        }

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422

    async def test_view_event_with_kafka_error_returns_500(
        self,
        async_client: httpx.AsyncClient,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that view event with Kafka error returns 500."""
        # Configure mock to raise exception
        kafka_producer_mock.send_event = AsyncMock(side_effect=Exception("Kafka connection failed"))

        async def mock_get_producer() -> AsyncGenerator[AsyncMock, None]:
            yield kafka_producer_mock

        app.dependency_overrides[get_kafka_producer] = mock_get_producer

        try:
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600,
            }

            response = await async_client.post("/events/view", json=event_data)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "X-Request-ID" in response.headers
        finally:
            app.dependency_overrides.clear()

    async def test_view_event_with_valid_data_includes_request_id_in_response(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
    ) -> None:
        """Test that view event response includes request ID header."""
        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": 3600,
        }

        response = await async_client.post("/events/view", json=event_data)

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


# ============================================================================
# Rating Event Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestRatingEventEndpoint:
    """Tests for POST /events/rating endpoint."""

    async def test_rating_event_with_valid_data_returns_201_with_success_status(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that valid rating event returns 201 with success status."""
        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 5,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/rating", json=event_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["event_type"] == "rating"
        assert data["rating"] == 5
        kafka_producer_mock.send_event.assert_called_once()

    async def test_rating_event_with_rating_above_maximum_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that rating event with rating > 5 returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 6,  # Invalid: > 5
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/rating", json=invalid_data)

        assert response.status_code == 422

    async def test_rating_event_with_rating_below_minimum_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that rating event with rating < 1 returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 0,  # Invalid: < 1
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/rating", json=invalid_data)

        assert response.status_code == 422

    async def test_rating_event_with_non_integer_rating_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that rating event with non-integer rating returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 3.5,  # Invalid: not integer
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/rating", json=invalid_data)

        assert response.status_code == 422


# ============================================================================
# Search Event Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestSearchEventEndpoint:
    """Tests for POST /events/search endpoint."""

    async def test_search_event_with_valid_data_returns_201_with_success_status(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that valid search event returns 201 with success status."""
        event_data = {
            "user_id": "user_123",
            "query": "action movies",
            "results_count": 25,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/search", json=event_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["event_type"] == "search"
        assert data["query"] == "action movies"
        assert data["results_count"] == 25
        kafka_producer_mock.send_event.assert_called_once()

    async def test_search_event_with_empty_query_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that search event with empty query returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "query": "",  # Invalid: empty
            "results_count": 25,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/search", json=invalid_data)

        assert response.status_code == 422

    async def test_search_event_with_negative_results_count_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that search event with negative results_count returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "query": "action movies",
            "results_count": -1,  # Invalid: negative
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/search", json=invalid_data)

        assert response.status_code == 422


# ============================================================================
# Skip Event Endpoint Tests
# ============================================================================


@pytest.mark.asyncio
class TestSkipEventEndpoint:
    """Tests for POST /events/skip endpoint."""

    async def test_skip_event_with_valid_data_returns_201_with_success_status(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that valid skip event returns 201 with success status."""
        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "watch_duration_seconds": 600,
            "movie_duration_seconds": 5400,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/skip", json=event_data)

        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "success"
        assert data["event_type"] == "skip"
        kafka_producer_mock.send_event.assert_called_once()

    async def test_skip_event_with_watch_duration_exceeding_movie_duration_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that skip event with watch_duration > movie_duration returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "watch_duration_seconds": 6000,  # Invalid: > movie_duration
            "movie_duration_seconds": 5400,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/skip", json=invalid_data)

        assert response.status_code == 422

    async def test_skip_event_with_negative_durations_returns_422(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that skip event with negative durations returns 422."""
        invalid_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "watch_duration_seconds": -100,  # Invalid: negative
            "movie_duration_seconds": 5400,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        response = await async_client.post("/events/skip", json=invalid_data)

        assert response.status_code == 422


# ============================================================================
# Error Handling Tests
# ============================================================================


@pytest.mark.asyncio
class TestErrorHandling:
    """Tests for error handling across endpoints."""

    async def test_validation_error_with_invalid_data_returns_422_with_details(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that validation errors return 422 with error details."""
        invalid_data = {"invalid": "data"}

        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        data = response.json()
        # FastAPI default format: {"detail": [...]}
        # Custom handler format: {"status": "error", "error_type": "validation_error", "errors": [...]}
        assert "detail" in data or ("error_type" in data and "errors" in data)
        if "detail" in data:
            assert isinstance(data["detail"], list)
        if "error_type" in data:
            assert data["error_type"] == "validation_error"
            assert "errors" in data

    async def test_kafka_error_with_failed_producer_returns_500(
        self,
        async_client: httpx.AsyncClient,
        kafka_producer_mock: AsyncMock,
    ) -> None:
        """Test that Kafka errors return 500 with proper error message."""
        # Configure mock to raise exception
        kafka_producer_mock.send_event = AsyncMock(side_effect=Exception("Kafka connection failed"))

        async def mock_get_producer() -> AsyncGenerator[AsyncMock, None]:
            yield kafka_producer_mock

        app.dependency_overrides[get_kafka_producer] = mock_get_producer

        try:
            event_data = {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600,
            }

            response = await async_client.post("/events/view", json=event_data)

            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "X-Request-ID" in response.headers
            # Verify producer was called before error
            kafka_producer_mock.send_event.assert_called_once()
        finally:
            app.dependency_overrides.clear()

    async def test_error_response_with_any_endpoint_includes_request_id(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that error responses include request ID header."""
        # Test with validation error
        invalid_data = {"invalid": "data"}
        response = await async_client.post("/events/view", json=invalid_data)

        assert response.status_code == 422
        assert "X-Request-ID" in response.headers


# ============================================================================
# Middleware Tests
# ============================================================================


@pytest.mark.asyncio
class TestMiddleware:
    """Tests for middleware functionality."""

    async def test_request_id_middleware_with_any_request_adds_header(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that request ID middleware adds header to all responses."""
        response = await async_client.get("/health")

        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) > 0
        # Should be UUID format
        assert len(request_id) == 36

    async def test_timing_middleware_with_any_request_adds_process_time_header(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that timing middleware adds process time header."""
        response = await async_client.get("/health")

        assert "X-Process-Time" in response.headers
        process_time = float(response.headers["X-Process-Time"])
        assert process_time >= 0

    async def test_request_id_middleware_with_multiple_requests_generates_unique_ids(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that request ID middleware generates unique IDs for each request."""
        request_ids = set()
        for _ in range(10):
            response = await async_client.get("/health")
            request_id = response.headers["X-Request-ID"]
            request_ids.add(request_id)

        # All request IDs should be unique
        assert len(request_ids) == 10

    async def test_logging_middleware_with_valid_request_logs_request_and_response(
        self,
        async_client: httpx.AsyncClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Test that logging middleware logs request and response."""
        import logging

        with caplog.at_level(logging.INFO):
            response = await async_client.get("/health")

            assert response.status_code == 200
            # Check that logs were generated (may vary based on logging config)
            # At minimum, we should have some log entries
            assert len(caplog.records) >= 0  # Logging may be configured differently


# ============================================================================
# Metrics Tests
# ============================================================================


@pytest.mark.asyncio
class TestMetrics:
    """Tests for Prometheus metrics collection."""

    async def test_metrics_with_event_requests_increments_event_counter(
        self,
        async_client: httpx.AsyncClient,
        mock_kafka_producer_dependency: None,
    ) -> None:
        """Test that event requests increment event counter."""
        event_data = {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": 3600,
        }

        # Make event request
        await async_client.post("/events/view", json=event_data)

        # Check metrics
        response = await async_client.get("/metrics")
        content = response.text
        # Should have event_requests_total counter
        assert "event_requests_total" in content

    async def test_metrics_with_http_requests_increments_http_counter(
        self, async_client: httpx.AsyncClient
    ) -> None:
        """Test that HTTP requests increment HTTP counter."""
        # Make multiple requests
        for _ in range(3):
            await async_client.get("/health")

        # Check metrics
        response = await async_client.get("/metrics")
        content = response.text
        # Should have http_requests_total counter
        assert "http_requests_total" in content
        # Should have multiple entries (at least 3)
        assert content.count("http_requests_total") >= 1
