"""Brick data and information management.

This module provides access to brick stories, objectives, and task information.
"""

from pathlib import Path
from typing import Any, Dict

# Brick names mapping
BRICK_NAMES = {
    1: "data-collection",
    2: "feature-engineering",
    3: "model-training",
    4: "recommendation-service",
    5: "monitoring",
    6: "orchestration",
}

# Brick information database
BRICK_INFO: Dict[int, Dict[str, Any]] = {
    1: {
        "title": "Data Collection Service",
        "story_intro": """
Welcome to MovieWorld! You've been hired as a Platform Engineer at MovieWorld,
a fast-growing streaming platform with millions of users. Your first task is to
build the event collection service that will capture all user interactions.

Every view, rating, search, and skip needs to be captured in real-time and
sent to our data pipeline. This is the foundation of our recommendation system.
        """,
        "objectives": [
            "Understand event-driven architecture",
            "Design RESTful APIs with FastAPI",
            "Integrate with Kafka for event streaming",
            "Implement proper schema validation",
            "Handle errors gracefully",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Setup development environment and project structure",
                "acceptance_criteria": [
                    "Project structure created",
                    "Dependencies installed",
                    "Docker Compose setup working",
                ],
            },
            {
                "number": 2,
                "description": "Define event schemas using Pydantic",
                "acceptance_criteria": [
                    "ViewEvent schema defined",
                    "RatingEvent schema defined",
                    "SearchEvent schema defined",
                    "SkipEvent schema defined",
                ],
            },
            {
                "number": 3,
                "description": "Implement API endpoints for all event types",
                "acceptance_criteria": [
                    "POST /events/view endpoint",
                    "POST /events/rating endpoint",
                    "POST /events/search endpoint",
                    "POST /events/skip endpoint",
                    "All endpoints return 200 on success",
                ],
            },
        ],
    },
    2: {
        "title": "Feature Engineering",
        "story_intro": """
Now that we're collecting events, we need to transform raw events into
meaningful features for our ML models. This brick focuses on feature
engineering with point-in-time correctness - a critical concept in ML systems.

You'll build a feature store using DuckDB and compute features that respect
temporal boundaries to prevent data leakage.
        """,
        "objectives": [
            "Understand feature engineering concepts",
            "Implement point-in-time correctness",
            "Build a feature store",
            "Compute user and item features",
            "Handle temporal data correctly",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Setup feature store with DuckDB",
                "acceptance_criteria": [
                    "DuckDB database initialized",
                    "Schema defined for features",
                    "Connection handling implemented",
                ],
            },
            {
                "number": 2,
                "description": "Implement user feature computation",
                "acceptance_criteria": [
                    "Total watch time feature",
                    "Average watch time feature",
                    "Favorite genre feature",
                    "Days since last active",
                ],
            },
        ],
    },
    3: {
        "title": "Model Training",
        "story_intro": """
Time to train our recommendation model! You'll implement a Neural
Collaborative Filtering (NCF) model using PyTorch. This brick teaches
you how to build production-grade training pipelines with proper
experiment tracking and model versioning.
        """,
        "objectives": [
            "Understand Neural Collaborative Filtering",
            "Build training pipelines",
            "Integrate with MLflow for experiment tracking",
            "Implement proper model versioning",
            "Handle negative sampling",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Implement NCF model architecture",
                "acceptance_criteria": [
                    "Model class defined",
                    "GMF and MLP paths implemented",
                    "Forward pass working",
                ],
            },
            {
                "number": 2,
                "description": "Build training loop",
                "acceptance_criteria": [
                    "Training loop implemented",
                    "Validation loop implemented",
                    "Early stopping implemented",
                ],
            },
        ],
    },
    4: {
        "title": "Recommendation Service",
        "story_intro": """
Now we need to serve recommendations in real-time! This brick focuses on
low-latency serving with a two-stage retrieval and ranking architecture.
You'll learn about caching, fallback strategies, and optimization techniques.
        """,
        "objectives": [
            "Build low-latency serving systems",
            "Implement two-stage retrieval+ranking",
            "Use FAISS for vector search",
            "Implement caching strategies",
            "Handle cold-start scenarios",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Implement retrieval stage with FAISS",
                "acceptance_criteria": [
                    "FAISS index built",
                    "Retrieval function implemented",
                    "Returns top-K candidates",
                ],
            },
            {
                "number": 2,
                "description": "Implement ranking stage",
                "acceptance_criteria": [
                    "Ranking model loaded",
                    "Scoring function implemented",
                    "Returns ranked recommendations",
                ],
            },
        ],
    },
    5: {
        "title": "Monitoring",
        "story_intro": """
Production systems need observability! In this brick, you'll implement
comprehensive monitoring with Prometheus metrics, Grafana dashboards,
and alerting rules. Learn how to monitor ML systems in production.
        """,
        "objectives": [
            "Understand observability concepts",
            "Implement Prometheus metrics",
            "Create Grafana dashboards",
            "Set up alerting rules",
            "Monitor ML-specific metrics",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Add Prometheus metrics to services",
                "acceptance_criteria": [
                    "Request counters implemented",
                    "Latency histograms implemented",
                    "Error rate metrics implemented",
                ],
            },
            {
                "number": 2,
                "description": "Create Grafana dashboards",
                "acceptance_criteria": [
                    "API metrics dashboard",
                    "Model performance dashboard",
                    "System health dashboard",
                ],
            },
        ],
    },
    6: {
        "title": "Orchestration",
        "story_intro": """
Finally, let's orchestrate the entire pipeline! You'll use Apache Airflow
to coordinate all components - from data generation to model deployment.
This brick teaches you MLOps and workflow orchestration.
        """,
        "objectives": [
            "Understand workflow orchestration",
            "Build Airflow DAGs",
            "Handle task dependencies",
            "Implement retry logic",
            "Set up SLA monitoring",
        ],
        "tasks": [
            {
                "number": 1,
                "description": "Create main pipeline DAG",
                "acceptance_criteria": [
                    "DAG defined with all tasks",
                    "Task dependencies set",
                    "Retry logic implemented",
                ],
            },
            {
                "number": 2,
                "description": "Add conditional deployment logic",
                "acceptance_criteria": [
                    "Model evaluation check",
                    "Conditional deployment",
                    "Notifications on success/failure",
                ],
            },
        ],
    },
}


def get_brick_info(brick_number: int) -> Dict[str, Any]:
    """Get information about a specific brick.

    Args:
        brick_number: The brick number (1-6)

    Returns:
        dict: Brick information including title, story, objectives, and tasks

    Raises:
        ValueError: If brick_number is invalid
    """
    if brick_number not in BRICK_INFO:
        raise ValueError(f"Invalid brick number: {brick_number}")

    info = BRICK_INFO[brick_number].copy()

    # Try to load additional info from README if it exists
    brick_name = BRICK_NAMES.get(brick_number)
    if brick_name:
        readme_path = Path(
            f"stories/movie-recommender/brick-{brick_number:02d}-{brick_name}/README.md"
        )
        if readme_path.exists():
            # Could parse README for additional info
            pass

    return info


def get_brick_path(brick_number: int) -> Path:
    """Get the path to a brick directory.

    Args:
        brick_number: The brick number (1-6)

    Returns:
        Path: Path to the brick directory
    """
    brick_name = BRICK_NAMES.get(brick_number)
    if not brick_name:
        raise ValueError(f"Invalid brick number: {brick_number}")

    return Path(f"stories/movie-recommender/brick-{brick_number:02d}-{brick_name}")
