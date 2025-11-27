"""Shared pytest fixtures for E2E tests."""

import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import httpx
import pytest
import pytest_asyncio

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def api_base_url() -> str:
    """API base URL for E2E tests."""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture(scope="session")
def recommendation_api_url() -> str:
    """Recommendation API URL for E2E tests."""
    return os.getenv("RECOMMENDATION_API_URL", "http://localhost:8001")


@pytest_asyncio.fixture
async def api_client(api_base_url: str) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for API E2E tests."""
    async with httpx.AsyncClient(base_url=api_base_url, timeout=30.0) as client:
        yield client


@pytest_asyncio.fixture
async def recommendation_client(
    recommendation_api_url: str,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    """Async HTTP client for recommendation API E2E tests."""
    async with httpx.AsyncClient(base_url=recommendation_api_url, timeout=30.0) as client:
        yield client


@pytest.fixture
def wait_for_api(api_base_url: str) -> None:
    """Wait for API to be ready."""
    max_wait = 120
    wait_interval = 5
    waited = 0

    while waited < max_wait:
        try:
            response = httpx.get(f"{api_base_url}/health", timeout=5)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(wait_interval)
        waited += wait_interval

    pytest.skip("API is not available")


@pytest.fixture
def test_user_id() -> str:
    """Test user ID for E2E tests."""
    return "e2e_test_user"


@pytest.fixture
def test_movie_ids() -> list:
    """Test movie IDs for E2E tests."""
    return [f"movie_{i}" for i in range(10)]
