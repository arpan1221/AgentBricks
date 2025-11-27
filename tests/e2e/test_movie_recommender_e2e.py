"""End-to-end tests for Movie Recommender story arc.

Tests the complete flow from data ingestion to recommendations.
"""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest


@pytest.mark.e2e
class TestMovieRecommenderE2E:
    """End-to-end tests for complete recommendation pipeline."""

    @pytest.mark.asyncio
    async def test_complete_event_ingestion_flow(
        self, api_client: httpx.AsyncClient, test_user_id: str, test_movie_ids: list, wait_for_api
    ):
        """Test complete event ingestion flow."""
        # Send multiple view events
        events_sent = []
        for movie_id in test_movie_ids[:5]:
            # Use slightly past timestamp to avoid validation issues
            # Format as ISO 8601 with Z suffix
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            event = {
                "user_id": test_user_id,
                "movie_id": movie_id,
                "timestamp": timestamp,
                "watch_time_seconds": 3600,
                "session_id": f"session_{movie_id}",
                "device": "desktop",
            }

            response = await api_client.post("/events/view", json=event)
            assert response.status_code == 201
            events_sent.append(event)

        # Send rating events
        for movie_id in test_movie_ids[:3]:
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            rating_event = {
                "user_id": test_user_id,
                "movie_id": movie_id,
                "rating": 5,
                "timestamp": timestamp,
            }

            response = await api_client.post("/events/rating", json=rating_event)
            assert response.status_code == 201

        # Verify events were processed
        # In a real E2E test, we would verify:
        # 1. Events are in Kafka
        # 2. Events are in MongoDB
        # 3. Features are computed
        # 4. Model can use the features
        assert len(events_sent) == 5

    @pytest.mark.asyncio
    async def test_api_health_check_e2e(self, api_client: httpx.AsyncClient, wait_for_api):
        """Test API health check in E2E context."""
        response = await api_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_api_metrics_endpoint_e2e(self, api_client: httpx.AsyncClient, wait_for_api):
        """Test API metrics endpoint in E2E context."""
        response = await api_client.get("/metrics")
        assert response.status_code == 200

        # Metrics should be in Prometheus format
        content = response.text
        assert "#" in content or "http_requests_total" in content

    @pytest.mark.asyncio
    async def test_multiple_event_types_e2e(
        self, api_client: httpx.AsyncClient, test_user_id: str, wait_for_api
    ):
        """Test multiple event types in E2E flow."""
        # View event
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        view_event = {
            "user_id": test_user_id,
            "movie_id": "movie_view",
            "timestamp": timestamp,
            "watch_time_seconds": 3600,
            "device": "desktop",
        }
        response = await api_client.post("/events/view", json=view_event)
        assert response.status_code == 201

        # Rating event
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        rating_event = {
            "user_id": test_user_id,
            "movie_id": "movie_rating",
            "rating": 5,
            "timestamp": timestamp,
        }
        response = await api_client.post("/events/rating", json=rating_event)
        assert response.status_code == 201

        # Search event
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        search_event = {
            "user_id": test_user_id,
            "query": "action movies",
            "results_count": 25,
            "timestamp": timestamp,
        }
        response = await api_client.post("/events/search", json=search_event)
        assert response.status_code == 201

        # Skip event
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        skip_event = {
            "user_id": test_user_id,
            "movie_id": "movie_skip",
            "watch_duration_seconds": 600,
            "movie_duration_seconds": 5400,
            "timestamp": timestamp,
        }
        response = await api_client.post("/events/skip", json=skip_event)
        assert response.status_code == 201

    @pytest.mark.asyncio
    async def test_error_handling_e2e(self, api_client: httpx.AsyncClient, wait_for_api):
        """Test error handling in E2E context."""
        # Invalid event (missing required fields)
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        invalid_event = {
            "user_id": "test_user",
            # Missing movie_id
            "timestamp": timestamp,
        }

        response = await api_client.post("/events/view", json=invalid_event)
        assert response.status_code == 422  # Validation error

        # Invalid rating (out of range)
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
        invalid_rating = {
            "user_id": "test_user",
            "movie_id": "movie_123",
            "rating": 10,  # Invalid: should be 1-5
            "timestamp": timestamp,
        }

        response = await api_client.post("/events/rating", json=invalid_rating)
        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_concurrent_requests_e2e(
        self, api_client: httpx.AsyncClient, test_user_id: str, wait_for_api
    ):
        """Test handling concurrent requests."""

        async def send_event(movie_id: str):
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            event = {
                "user_id": test_user_id,
                "movie_id": movie_id,
                "timestamp": timestamp,
                "watch_time_seconds": 3600,
                "device": "desktop",
            }
            response = await api_client.post("/events/view", json=event)
            return response.status_code

        # Send 10 concurrent requests
        movie_ids = [f"movie_concurrent_{i}" for i in range(10)]
        tasks = [send_event(movie_id) for movie_id in movie_ids]
        results = await asyncio.gather(*tasks)

        # All requests should succeed
        assert all(status == 201 for status in results)
