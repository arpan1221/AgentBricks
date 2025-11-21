"""Prometheus metrics for recommendation service monitoring.

This module provides comprehensive metrics collection for the recommendation
service, including HTTP request metrics, model performance metrics, cache
statistics, and business metrics.

Metrics follow Prometheus naming conventions:
- Counters: _total suffix
- Histograms: _seconds, _bytes, or descriptive units
- Gauges: Current state values
- Labels: Lowercase, snake_case
"""

import logging
import time
from typing import Optional, Dict, Any
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    REGISTRY
)
from prometheus_client.core import CollectorRegistry
from fastapi import Request, Response
from fastapi.responses import Response as FastAPIResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

logger = logging.getLogger(__name__)

# ============================================================================
# Prometheus Metrics
# ============================================================================

# HTTP Request Metrics
HTTP_REQUESTS_TOTAL = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['endpoint', 'method', 'status'],
    registry=REGISTRY
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['endpoint', 'method'],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
    registry=REGISTRY
)

# Connection Metrics
ACTIVE_CONNECTIONS = Gauge(
    'active_connections',
    'Number of active connections',
    registry=REGISTRY
)

# Model Performance Metrics
MODEL_PREDICTION_SCORE = Histogram(
    'model_prediction_score',
    'Distribution of model prediction scores',
    ['model_name', 'user_id'],
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    registry=REGISTRY
)

MODEL_INFERENCE_DURATION_SECONDS = Histogram(
    'model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_name'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
    registry=REGISTRY
)

# Cache Metrics
CACHE_HITS_TOTAL = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type', 'key_pattern'],
    registry=REGISTRY
)

CACHE_MISSES_TOTAL = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type', 'key_pattern'],
    registry=REGISTRY
)

CACHE_OPERATION_DURATION_SECONDS = Histogram(
    'cache_operation_duration_seconds',
    'Cache operation duration in seconds',
    ['operation', 'cache_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=REGISTRY
)

# Fallback Metrics
FALLBACK_USED_TOTAL = Counter(
    'fallback_used_total',
    'Total number of times fallback strategies were used',
    ['fallback_type', 'endpoint'],
    registry=REGISTRY
)

# Recommendation Metrics
RECOMMENDATION_LATENCY_SECONDS = Histogram(
    'recommendation_latency_seconds',
    'End-to-end recommendation generation latency',
    ['user_id', 'k'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    registry=REGISTRY
)

RECOMMENDATIONS_GENERATED_TOTAL = Counter(
    'recommendations_generated_total',
    'Total number of recommendations generated',
    ['endpoint', 'cached'],
    registry=REGISTRY
)

# Retrieval Metrics
RETRIEVAL_DURATION_SECONDS = Histogram(
    'retrieval_duration_seconds',
    'FAISS retrieval duration in seconds',
    ['k', 'index_type'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1],
    registry=REGISTRY
)

RANKING_DURATION_SECONDS = Histogram(
    'ranking_duration_seconds',
    'Model ranking duration in seconds',
    ['num_candidates'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    registry=REGISTRY
)


# ============================================================================
# FastAPI Middleware
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    Prometheus middleware for FastAPI request instrumentation.

    Automatically tracks HTTP request metrics including:
    - Request count by endpoint, method, and status
    - Request duration by endpoint and method

    Args:
        app: FastAPI application instance
        group_paths: Whether to group path parameters (default: True)
                     e.g., /recommend/user_123 -> /recommend/{user_id}

    Example:
        >>> from fastapi import FastAPI
        >>> from src.metrics import PrometheusMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(PrometheusMiddleware)
    """

    def __init__(
        self,
        app: Any,
        group_paths: bool = True
    ) -> None:
        """Initialize Prometheus middleware."""
        super().__init__(app)
        self.group_paths = group_paths
        logger.info("PrometheusMiddleware initialized")

    async def dispatch(self, request: Request, call_next: Any) -> Response:
        """
        Process request and collect metrics.

        Args:
            request: FastAPI request object
            call_next: Next middleware/handler in chain

        Returns:
            Response object
        """
        # Increment active connections
        ACTIVE_CONNECTIONS.inc()

        # Get endpoint path (grouped if enabled)
        endpoint = self._get_endpoint_path(request)
        method = request.method

        # Record start time
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Get status code
            status_code = response.status_code
            status_class = f"{status_code // 100}xx"  # 2xx, 4xx, 5xx

            # Record metrics
            HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint,
                method=method,
                status=status_class
            ).inc()

            duration = time.time() - start_time
            HTTP_REQUEST_DURATION_SECONDS.labels(
                endpoint=endpoint,
                method=method
            ).observe(duration)

            return response

        except Exception as e:
            # Record error
            HTTP_REQUESTS_TOTAL.labels(
                endpoint=endpoint,
                method=method,
                status="5xx"
            ).inc()

            duration = time.time() - start_time
            HTTP_REQUEST_DURATION_SECONDS.labels(
                endpoint=endpoint,
                method=method
            ).observe(duration)

            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

            # Re-raise exception
            raise

        finally:
            # Decrement active connections
            ACTIVE_CONNECTIONS.dec()

    def _get_endpoint_path(self, request: Request) -> str:
        """
        Get endpoint path, optionally grouping path parameters.

        Args:
            request: FastAPI request object

        Returns:
            Endpoint path string
        """
        # Try to get route path from FastAPI
        if hasattr(request, 'scope') and 'route' in request.scope:
            route = request.scope.get('route')
            if route and hasattr(route, 'path'):
                path = route.path

                if self.group_paths:
                    # Group path parameters
                    # /recommend/user_123 -> /recommend/{user_id}
                    # /recommend/user_123?k=10 -> /recommend/{user_id}
                    if '{' in path:
                        return path
                    else:
                        # Try to detect path parameters
                        parts = path.split('/')
                        grouped_parts = []
                        for part in parts:
                            if part and not part.startswith('?'):
                                # Check if it looks like an ID (UUID, numeric, etc.)
                                if self._looks_like_id(part):
                                    # Get the previous part as parameter name
                                    if grouped_parts:
                                        param_name = grouped_parts[-1].rstrip('s')  # plural -> singular
                                        grouped_parts[-1] = f"{{{param_name}_id}}"
                                else:
                                    grouped_parts.append(part)
                            else:
                                grouped_parts.append(part)
                        return '/'.join(grouped_parts) if grouped_parts else path

                return path

        # Fallback to raw path
        return request.url.path

    def _looks_like_id(self, value: str) -> bool:
        """
        Check if a path segment looks like an ID.

        Args:
            value: Path segment to check

        Returns:
            True if looks like an ID
        """
        # Check for UUID format
        if len(value) == 36 and value.count('-') == 4:
            return True

        # Check for numeric ID
        if value.isdigit():
            return True

        # Check for common ID patterns (user_123, movie_456)
        if '_' in value and value.split('_')[-1].isdigit():
            return True

        return False


def metrics_endpoint() -> FastAPIResponse:
    """
    Metrics endpoint for Prometheus scraping.

    Returns:
        Response with Prometheus metrics in text format

    Example:
        >>> from fastapi import FastAPI
        >>> from src.metrics import metrics_endpoint
        >>>
        >>> app = FastAPI()
        >>> app.get("/metrics")(metrics_endpoint)
    """
    return FastAPIResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


# ============================================================================
# Custom Metrics Functions
# ============================================================================

def record_recommendation_latency(
    duration: float,
    user_id: Optional[str] = None,
    k: Optional[int] = None
) -> None:
    """
    Record recommendation generation latency.

    Args:
        duration: Latency in seconds
        user_id: Optional user identifier for labeling
        k: Optional number of recommendations

    Example:
        >>> start = time.time()
        >>> recommendations = generate_recommendations(user_id)
        >>> duration = time.time() - start
        >>> record_recommendation_latency(duration, user_id, k=10)
    """
    labels = {}
    if user_id:
        labels['user_id'] = str(user_id)
    if k:
        labels['k'] = str(k)

    # Use default labels if not provided
    user_label = str(user_id) if user_id else "unknown"
    k_label = str(k) if k else "unknown"

    RECOMMENDATION_LATENCY_SECONDS.labels(
        user_id=user_label,
        k=k_label
    ).observe(duration)

    logger.debug(
        f"Recorded recommendation latency",
        extra={
            "duration_seconds": duration,
            "user_id": user_id,
            "k": k
        }
    )


def record_model_score(
    score: float,
    model_name: str = "ncf",
    user_id: Optional[str] = None
) -> None:
    """
    Record model prediction score distribution.

    Args:
        score: Prediction score (0-1)
        model_name: Name of the model (default: "ncf")
        user_id: Optional user identifier for labeling

    Example:
        >>> score = model.predict(user_id, item_id)
        >>> record_model_score(score.item(), model_name="ncf", user_id=user_id)
    """
    if not (0.0 <= score <= 1.0):
        logger.warning(
            f"Model score out of range [0, 1]: {score}",
            extra={"model_name": model_name, "user_id": user_id}
        )
        score = max(0.0, min(1.0, score))

    user_label = str(user_id) if user_id else "unknown"

    MODEL_PREDICTION_SCORE.labels(
        model_name=model_name,
        user_id=user_label
    ).observe(score)

    logger.debug(
        f"Recorded model score",
        extra={
            "score": score,
            "model_name": model_name,
            "user_id": user_id
        }
    )


def record_cache_operation(
    hit: bool,
    cache_type: str = "redis",
    key_pattern: str = "unknown"
) -> None:
    """
    Record cache hit or miss.

    Args:
        hit: True for cache hit, False for cache miss
        cache_type: Type of cache (default: "redis")
        key_pattern: Pattern of cache key (e.g., "recommendations", "user_features")

    Example:
        >>> cached = cache.get("recommendations:user_123:10")
        >>> record_cache_operation(cached is not None, key_pattern="recommendations")
    """
    if hit:
        CACHE_HITS_TOTAL.labels(
            cache_type=cache_type,
            key_pattern=key_pattern
        ).inc()
    else:
        CACHE_MISSES_TOTAL.labels(
            cache_type=cache_type,
            key_pattern=key_pattern
        ).inc()

    logger.debug(
        f"Recorded cache operation",
        extra={
            "hit": hit,
            "cache_type": cache_type,
            "key_pattern": key_pattern
        }
    )


def record_fallback_usage(
    fallback_type: str,
    endpoint: str = "recommend"
) -> None:
    """
    Record when fallback recommendation strategy is used.

    Args:
        fallback_type: Type of fallback used:
                      - "popular_region": Popular in user's region
                      - "trending_global": Globally trending
                      - "random_popular": Random popular movies
        endpoint: Endpoint name where fallback was used (default: "recommend")

    Example:
        >>> if not recommendations:
        ...     recommendations = get_popular_movies()
        ...     record_fallback_usage("popular_region", endpoint="recommend")
    """
    FALLBACK_USED_TOTAL.labels(
        fallback_type=fallback_type,
        endpoint=endpoint
    ).inc()

    logger.info(
        f"Fallback strategy used",
        extra={
            "fallback_type": fallback_type,
            "endpoint": endpoint
        }
    )


def record_model_inference_duration(
    duration: float,
    model_name: str = "ncf"
) -> None:
    """
    Record model inference duration.

    Args:
        duration: Inference duration in seconds
        model_name: Name of the model (default: "ncf")

    Example:
        >>> start = time.time()
        >>> predictions = model.batch_predict(user_ids, item_ids)
        >>> duration = time.time() - start
        >>> record_model_inference_duration(duration, model_name="ncf")
    """
    MODEL_INFERENCE_DURATION_SECONDS.labels(
        model_name=model_name
    ).observe(duration)

    logger.debug(
        f"Recorded model inference duration",
        extra={
            "duration_seconds": duration,
            "model_name": model_name
        }
    )


def record_retrieval_duration(
    duration: float,
    k: int,
    index_type: str = "hnsw"
) -> None:
    """
    Record FAISS retrieval duration.

    Args:
        duration: Retrieval duration in seconds
        k: Number of candidates retrieved
        index_type: Type of FAISS index (default: "hnsw")

    Example:
        >>> start = time.time()
        >>> candidates = retriever.retrieve(user_emb, k=100)
        >>> duration = time.time() - start
        >>> record_retrieval_duration(duration, k=100, index_type="hnsw")
    """
    RETRIEVAL_DURATION_SECONDS.labels(
        k=str(k),
        index_type=index_type
    ).observe(duration)

    logger.debug(
        f"Recorded retrieval duration",
        extra={
            "duration_seconds": duration,
            "k": k,
            "index_type": index_type
        }
    )


def record_ranking_duration(
    duration: float,
    num_candidates: int
) -> None:
    """
    Record model ranking duration.

    Args:
        duration: Ranking duration in seconds
        num_candidates: Number of candidates being ranked

    Example:
        >>> start = time.time()
        >>> ranked = rank_candidates(user_id, candidates)
        >>> duration = time.time() - start
        >>> record_ranking_duration(duration, num_candidates=len(candidates))
    """
    RANKING_DURATION_SECONDS.labels(
        num_candidates=str(num_candidates)
    ).observe(duration)

    logger.debug(
        f"Recorded ranking duration",
        extra={
            "duration_seconds": duration,
            "num_candidates": num_candidates
        }
    )


def record_cache_operation_duration(
    duration: float,
    operation: str,
    cache_type: str = "redis"
) -> None:
    """
    Record cache operation duration (get/set/delete).

    Args:
        duration: Operation duration in seconds
        operation: Operation type ("get", "set", "delete")
        cache_type: Type of cache (default: "redis")

    Example:
        >>> start = time.time()
        >>> cache.set(key, value)
        >>> duration = time.time() - start
        >>> record_cache_operation_duration(duration, operation="set")
    """
    CACHE_OPERATION_DURATION_SECONDS.labels(
        operation=operation,
        cache_type=cache_type
    ).observe(duration)

    logger.debug(
        f"Recorded cache operation duration",
        extra={
            "duration_seconds": duration,
            "operation": operation,
            "cache_type": cache_type
        }
    )


def record_recommendations_generated(
    endpoint: str,
    cached: bool
) -> None:
    """
    Record that recommendations were generated.

    Args:
        endpoint: Endpoint name (e.g., "recommend", "batch_recommend")
        cached: Whether results came from cache

    Example:
        >>> recommendations = get_recommendations(user_id)
        >>> record_recommendations_generated("recommend", cached=False)
    """
    RECOMMENDATIONS_GENERATED_TOTAL.labels(
        endpoint=endpoint,
        cached=str(cached).lower()
    ).inc()

    logger.debug(
        f"Recorded recommendations generated",
        extra={
            "endpoint": endpoint,
            "cached": cached
        }
    )


def get_metrics_summary() -> Dict[str, Any]:
    """
    Get summary of current metrics values.

    Useful for health checks or status endpoints.

    Returns:
        Dictionary with metric summaries

    Example:
        >>> summary = get_metrics_summary()
        >>> print(f"Total requests: {summary['total_requests']}")
    """
    # This is a simplified summary - in production, you'd query actual metric values
    # For now, return structure that can be expanded
    return {
        "metrics_available": True,
        "metric_types": [
            "http_requests_total",
            "http_request_duration_seconds",
            "active_connections",
            "model_prediction_score",
            "cache_hits_total",
            "cache_misses_total",
            "fallback_used_total",
            "recommendation_latency_seconds"
        ],
        "note": "Use /metrics endpoint for Prometheus format"
    }
