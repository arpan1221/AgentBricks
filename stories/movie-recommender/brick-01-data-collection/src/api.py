"""MovieWorld Event Collection API.

A FastAPI application for collecting user interaction events (views, ratings,
searches, skips) for the movie recommendation system. Events are validated
and published to Kafka for downstream processing.

Features:
    - Event validation using Pydantic schemas
    - Kafka event publishing
    - Request/response logging
    - Prometheus metrics
    - Health checks
    - OpenAPI documentation
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator, Dict, Any, Optional
import sys

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.middleware.base import BaseHTTPMiddleware

from src.schemas import (
    ViewEvent,
    RatingEvent,
    SearchEvent,
    SkipEvent,
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
EVENT_COUNTER = Counter(
    "event_requests_total",
    "Total number of event requests",
    ["event_type", "status"]
)

REQUEST_DURATION = Histogram(
    "request_duration_seconds",
    "Request duration in seconds",
    ["method", "endpoint", "status_code"]
)

REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"]
)


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to generate and add request ID to all requests."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Add request ID to request and response."""
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for structured request/response logging."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Log request and response details."""
        start_time = time.time()
        request_id = getattr(request.state, "request_id", "unknown")

        # Log request
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query_params": dict(request.query_params),
                "client": request.client.host if request.client else None,
            }
        )

        # Process request
        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "duration_seconds": round(duration, 3),
                }
            )

            # Record metrics
            REQUEST_COUNTER.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()

            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).observe(duration)

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "duration_seconds": round(duration, 3),
                },
                exc_info=True
            )
            raise


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to add timing headers to responses."""

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """Add timing headers to response."""
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        response.headers["X-Process-Time"] = str(round(duration, 3))
        return response


# Kafka producer dependency (stub implementation)
class KafkaProducer:
    """Kafka producer for publishing events."""

    def __init__(self, bootstrap_servers: str = "localhost:9092"):
        """
        Initialize Kafka producer.

        Args:
            bootstrap_servers: Kafka broker addresses
        """
        self.bootstrap_servers = bootstrap_servers
        logger.info(f"Kafka producer initialized with brokers: {bootstrap_servers}")

    async def publish(
        self,
        topic: str,
        key: str,
        value: Dict[str, Any]
    ) -> None:
        """
        Publish event to Kafka topic.

        Args:
            topic: Kafka topic name
            key: Message key
            value: Message value (serialized event)

        Raises:
            Exception: If publishing fails
        """
        # TODO: Implement actual Kafka publishing
        # For now, just log the event
        logger.info(
            "Publishing event to Kafka",
            extra={
                "topic": topic,
                "key": key,
                "event_type": value.get("event_type", "unknown"),
            }
        )

        # Simulate async Kafka publishing
        # In production, use aiokafka or similar
        # await self._producer.send(topic, key=key, value=json.dumps(value))


async def get_kafka_producer() -> AsyncGenerator[KafkaProducer, None]:
    """
    Dependency to get Kafka producer instance.

    Yields:
        KafkaProducer instance
    """
    # In production, this would be a singleton or connection pool
    producer = KafkaProducer()
    try:
        yield producer
    finally:
        # Cleanup if needed
        pass


def get_logger() -> logging.Logger:
    """
    Dependency to get logger instance.

    Returns:
        Logger instance
    """
    return logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan context manager.

    Handles startup and shutdown logic.
    """
    # Startup
    logger.info("Starting MovieWorld Event Collection API...")
    logger.info("API is ready to accept requests")

    yield

    # Shutdown
    logger.info("Shutting down MovieWorld Event Collection API...")


# Create FastAPI app
app = FastAPI(
    title="MovieWorld Event Collection API",
    version="1.0.0",
    description=__doc__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on environment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware (order matters - first added is last executed)
app.add_middleware(TimingMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(RequestIDMiddleware)


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(
    request: Request,
    exc: ValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: ValidationError exception

    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.warning(
        "Validation error",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "errors": exc.errors(),
        }
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "error_type": "validation_error",
            "message": "Validation failed",
            "errors": exc.errors(),
            "request_id": request_id,
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Exception instance

    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", "unknown")

    logger.error(
        "Unhandled exception",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "error": str(exc),
        },
        exc_info=True
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "error_type": "internal_server_error",
            "message": "An unexpected error occurred",
            "request_id": request_id,
        }
    )


# Health check endpoint
@app.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Returns API health status",
    tags=["System"]
)
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.

    Returns:
        Dictionary with status and timestamp

    Example:
        >>> GET /health
        {
            "status": "healthy",
            "timestamp": "2024-01-15T20:30:00Z"
        }
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Metrics endpoint
@app.get(
    "/metrics",
    status_code=status.HTTP_200_OK,
    summary="Prometheus metrics",
    description="Returns Prometheus metrics in text format",
    tags=["System"]
)
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Prometheus metrics in text format

    Example:
        >>> GET /metrics
        # HELP http_requests_total Total HTTP requests
        # TYPE http_requests_total counter
        http_requests_total{method="GET",endpoint="/health",status_code="200"} 10
        ...
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


# View event endpoint
@app.post(
    "/events/view",
    status_code=status.HTTP_201_CREATED,
    summary="Record view event",
    description="Record a user viewing a movie",
    tags=["Events"],
    response_model=Dict[str, Any]
)
async def record_view_event(
    event: ViewEvent,
    request: Request,
    producer: KafkaProducer = Depends(get_kafka_producer),
    logger: logging.Logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Record a user view event.

    Args:
        event: ViewEvent instance (validated)
        request: FastAPI request object
        producer: Kafka producer dependency
        logger: Logger dependency

    Returns:
        Dictionary with status and event details

    Raises:
        HTTPException: If event processing fails

    Example Request:
        >>> POST /events/view
        {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": "2024-01-15T20:30:00Z",
            "watch_time_seconds": 3600,
            "session_id": "session_789",
            "device": "desktop"
        }

    Example Response:
        {
            "status": "success",
            "event_type": "view",
            "event_id": "550e8400-e29b-41d4-a716-446655440000",
            "user_id": "user_123",
            "movie_id": "movie_456",
            "timestamp": "2024-01-15T20:30:00Z"
        }
    """
    request_id = getattr(request.state, "request_id", "unknown")
    event_id = str(uuid.uuid4())

    try:
        # Convert event to dict
        event_dict = event.model_dump()
        event_dict["event_type"] = "view"
        event_dict["event_id"] = event_id
        event_dict["request_id"] = request_id

        # Publish to Kafka
        await producer.publish(
            topic="movie-events",
            key=event.user_id,
            value=event_dict
        )

        # Record metric
        EVENT_COUNTER.labels(event_type="view", status="success").inc()

        logger.info(
            "View event recorded",
            extra={
                "request_id": request_id,
                "event_id": event_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
            }
        )

        return {
            "status": "success",
            "event_type": "view",
            "event_id": event_id,
            "user_id": event.user_id,
            "movie_id": event.movie_id,
            "timestamp": event.timestamp.isoformat(),
        }

    except Exception as e:
        EVENT_COUNTER.labels(event_type="view", status="error").inc()
        logger.error(
            "Failed to record view event",
            extra={
                "request_id": request_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "error": str(e),
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record view event: {str(e)}"
        )


# Rating event endpoint
@app.post(
    "/events/rating",
    status_code=status.HTTP_201_CREATED,
    summary="Record rating event",
    description="Record a user rating a movie",
    tags=["Events"],
    response_model=Dict[str, Any]
)
async def record_rating_event(
    event: RatingEvent,
    request: Request,
    producer: KafkaProducer = Depends(get_kafka_producer),
    logger: logging.Logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Record a user rating event.

    Args:
        event: RatingEvent instance (validated)
        request: FastAPI request object
        producer: Kafka producer dependency
        logger: Logger dependency

    Returns:
        Dictionary with status and event details

    Example Request:
        >>> POST /events/rating
        {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 5,
            "timestamp": "2024-01-15T20:45:00Z"
        }

    Example Response:
        {
            "status": "success",
            "event_type": "rating",
            "event_id": "550e8400-e29b-41d4-a716-446655440001",
            "user_id": "user_123",
            "movie_id": "movie_456",
            "rating": 5
        }
    """
    request_id = getattr(request.state, "request_id", "unknown")
    event_id = str(uuid.uuid4())

    try:
        # Convert event to dict
        event_dict = event.model_dump()
        event_dict["event_type"] = "rating"
        event_dict["event_id"] = event_id
        event_dict["request_id"] = request_id

        # Publish to Kafka
        await producer.publish(
            topic="movie-events",
            key=event.user_id,
            value=event_dict
        )

        # Record metric
        EVENT_COUNTER.labels(event_type="rating", status="success").inc()

        logger.info(
            "Rating event recorded",
            extra={
                "request_id": request_id,
                "event_id": event_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "rating": event.rating,
            }
        )

        return {
            "status": "success",
            "event_type": "rating",
            "event_id": event_id,
            "user_id": event.user_id,
            "movie_id": event.movie_id,
            "rating": event.rating,
            "timestamp": event.timestamp.isoformat(),
        }

    except Exception as e:
        EVENT_COUNTER.labels(event_type="rating", status="error").inc()
        logger.error(
            "Failed to record rating event",
            extra={
                "request_id": request_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "error": str(e),
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record rating event: {str(e)}"
        )


# Search event endpoint
@app.post(
    "/events/search",
    status_code=status.HTTP_201_CREATED,
    summary="Record search event",
    description="Record a user searching for movies",
    tags=["Events"],
    response_model=Dict[str, Any]
)
async def record_search_event(
    event: SearchEvent,
    request: Request,
    producer: KafkaProducer = Depends(get_kafka_producer),
    logger: logging.Logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Record a user search event.

    Args:
        event: SearchEvent instance (validated)
        request: FastAPI request object
        producer: Kafka producer dependency
        logger: Logger dependency

    Returns:
        Dictionary with status and event details

    Example Request:
        >>> POST /events/search
        {
            "user_id": "user_123",
            "query": "sci-fi thriller",
            "results_count": 42,
            "timestamp": "2024-01-15T19:00:00Z"
        }

    Example Response:
        {
            "status": "success",
            "event_type": "search",
            "event_id": "550e8400-e29b-41d4-a716-446655440002",
            "user_id": "user_123",
            "query": "sci-fi thriller",
            "results_count": 42
        }
    """
    request_id = getattr(request.state, "request_id", "unknown")
    event_id = str(uuid.uuid4())

    try:
        # Convert event to dict
        event_dict = event.model_dump()
        event_dict["event_type"] = "search"
        event_dict["event_id"] = event_id
        event_dict["request_id"] = request_id

        # Publish to Kafka
        await producer.publish(
            topic="movie-events",
            key=event.user_id,
            value=event_dict
        )

        # Record metric
        EVENT_COUNTER.labels(event_type="search", status="success").inc()

        logger.info(
            "Search event recorded",
            extra={
                "request_id": request_id,
                "event_id": event_id,
                "user_id": event.user_id,
                "query": event.query,
                "results_count": event.results_count,
            }
        )

        return {
            "status": "success",
            "event_type": "search",
            "event_id": event_id,
            "user_id": event.user_id,
            "query": event.query,
            "results_count": event.results_count,
            "timestamp": event.timestamp.isoformat(),
        }

    except Exception as e:
        EVENT_COUNTER.labels(event_type="search", status="error").inc()
        logger.error(
            "Failed to record search event",
            extra={
                "request_id": request_id,
                "user_id": event.user_id,
                "query": event.query,
                "error": str(e),
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record search event: {str(e)}"
        )


# Skip event endpoint
@app.post(
    "/events/skip",
    status_code=status.HTTP_201_CREATED,
    summary="Record skip event",
    description="Record a user skipping a movie",
    tags=["Events"],
    response_model=Dict[str, Any]
)
async def record_skip_event(
    event: SkipEvent,
    request: Request,
    producer: KafkaProducer = Depends(get_kafka_producer),
    logger: logging.Logger = Depends(get_logger)
) -> Dict[str, Any]:
    """
    Record a user skip event.

    Args:
        event: SkipEvent instance (validated)
        request: FastAPI request object
        producer: Kafka producer dependency
        logger: Logger dependency

    Returns:
        Dictionary with status and event details

    Example Request:
        >>> POST /events/skip
        {
            "user_id": "user_123",
            "movie_id": "movie_456",
            "watch_duration_seconds": 600,
            "movie_duration_seconds": 5400,
            "timestamp": "2024-01-15T20:10:00Z"
        }

    Example Response:
        {
            "status": "success",
            "event_type": "skip",
            "event_id": "550e8400-e29b-41d4-a716-446655440003",
            "user_id": "user_123",
            "movie_id": "movie_456"
        }
    """
    request_id = getattr(request.state, "request_id", "unknown")
    event_id = str(uuid.uuid4())

    try:
        # Convert event to dict
        event_dict = event.model_dump()
        event_dict["event_type"] = "skip"
        event_dict["event_id"] = event_id
        event_dict["request_id"] = request_id

        # Publish to Kafka
        await producer.publish(
            topic="movie-events",
            key=event.user_id,
            value=event_dict
        )

        # Record metric
        EVENT_COUNTER.labels(event_type="skip", status="success").inc()

        logger.info(
            "Skip event recorded",
            extra={
                "request_id": request_id,
                "event_id": event_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "watch_duration_seconds": event.watch_duration_seconds,
            }
        )

        return {
            "status": "success",
            "event_type": "skip",
            "event_id": event_id,
            "user_id": event.user_id,
            "movie_id": event.movie_id,
            "timestamp": event.timestamp.isoformat(),
        }

    except Exception as e:
        EVENT_COUNTER.labels(event_type="skip", status="error").inc()
        logger.error(
            "Failed to record skip event",
            extra={
                "request_id": request_id,
                "user_id": event.user_id,
                "movie_id": event.movie_id,
                "error": str(e),
            },
            exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record skip event: {str(e)}"
        )
