"""FastAPI recommendation service for movie recommendations.

This module provides a production-ready recommendation API with:
- Two-stage retrieval (FAISS) + ranking (NCF model)
- Redis caching for performance
- Fallback strategies for cold-start users
- Prometheus metrics and structured logging
- Batch recommendation support
"""

import json
import logging

# Import model and feature store
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import redis
import torch
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Gauge, Histogram
from pydantic import BaseModel, Field

# Add brick-03 path for model import
brick03_path = Path(__file__).parent.parent.parent / "brick-03-model-training" / "src"
if str(brick03_path) not in sys.path:
    sys.path.insert(0, str(brick03_path))

# Initialize logger early for import error handling
logger = logging.getLogger(__name__)

try:
    from model import NCF
except ImportError:
    logger.warning("Could not import NCF model. Model loading will be disabled.")
    NCF = None

# Add brick-02 path for feature store import
brick02_path = Path(__file__).parent.parent.parent / "brick-02-feature-engineering" / "src"
if str(brick02_path) not in sys.path:
    sys.path.insert(0, str(brick02_path))

try:
    from feature_store import FeatureStore
except ImportError:
    logger.warning("Could not import FeatureStore. Feature store will be disabled.")
    FeatureStore = None

# Prometheus metrics
REQUEST_COUNTER = Counter(
    "recommendation_requests_total",
    "Total number of recommendation requests",
    ["endpoint", "status"],
)

REQUEST_DURATION = Histogram(
    "recommendation_request_duration_seconds", "Request duration in seconds", ["endpoint"]
)

CACHE_HITS = Counter("recommendation_cache_hits_total", "Total number of cache hits", ["type"])

CACHE_MISSES = Counter(
    "recommendation_cache_misses_total", "Total number of cache misses", ["type"]
)

ACTIVE_REQUESTS = Gauge(
    "recommendation_active_requests", "Number of active recommendation requests"
)


# Pydantic models
class RecommendationItem(BaseModel):
    """Single recommendation item."""

    movie_id: str = Field(..., description="Movie identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Recommendation score (0-1)")

    class Config:
        json_schema_extra = {"example": {"movie_id": "movie_123", "score": 0.85}}


class RecommendationResponse(BaseModel):
    """Recommendation response."""

    user_id: str = Field(..., description="User identifier")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    cached: bool = Field(False, description="Whether results came from cache")
    retrieval_time_ms: float = Field(..., description="Time taken for retrieval in milliseconds")
    ranking_time_ms: float = Field(..., description="Time taken for ranking in milliseconds")
    total_time_ms: float = Field(..., description="Total request time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_456",
                "recommendations": [
                    {"movie_id": "movie_123", "score": 0.85},
                    {"movie_id": "movie_789", "score": 0.82},
                ],
                "cached": False,
                "retrieval_time_ms": 15.2,
                "ranking_time_ms": 45.3,
                "total_time_ms": 62.5,
            }
        }


class BatchRecommendationRequest(BaseModel):
    """Batch recommendation request."""

    user_ids: List[str] = Field(..., min_items=1, max_items=100, description="List of user IDs")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations per user")
    force_refresh: bool = Field(False, description="Force refresh cache")

    class Config:
        json_schema_extra = {
            "example": {"user_ids": ["user_1", "user_2"], "k": 10, "force_refresh": False}
        }


class BatchRecommendationResponse(BaseModel):
    """Batch recommendation response."""

    recommendations: Dict[str, List[RecommendationItem]] = Field(
        ..., description="Dictionary mapping user_id to recommendations"
    )
    total_time_ms: float = Field(..., description="Total batch processing time in milliseconds")

    class Config:
        json_schema_extra = {
            "example": {
                "recommendations": {
                    "user_1": [{"movie_id": "movie_123", "score": 0.85}],
                    "user_2": [{"movie_id": "movie_456", "score": 0.82}],
                },
                "total_time_ms": 125.3,
            }
        }


# Global state
class RecommenderState:
    """Global state for recommendation service."""

    def __init__(self):
        self.model: Optional[torch.nn.Module] = None
        self.redis_client: Optional[redis.Redis] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.movie_id_to_index: Dict[str, int] = {}
        self.index_to_movie_id: Dict[int, str] = {}
        self.user_id_to_index: Dict[str, int] = {}
        self.index_to_user_id: Dict[int, str] = {}
        self.popular_movies: List[str] = []
        self.trending_movies: List[str] = []
        self.feature_store = None
        self.device: torch.device = torch.device("cpu")


state = RecommenderState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown.

    Loads model, initializes Redis, and sets up FAISS index on startup.
    Cleans up resources on shutdown.
    """
    # Startup
    logger.info("Starting recommendation service...")

    # Load configuration from environment if not already set
    import os

    if not hasattr(app.state, "config") or not app.state.config:
        app.state.config = {
            "model_path": os.getenv("MODEL_PATH", "./models/ncf_best.pth"),
            "redis_host": os.getenv("REDIS_HOST", "localhost"),
            "redis_port": int(os.getenv("REDIS_PORT", "6379")),
            "redis_db": int(os.getenv("REDIS_DB", "0")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "300")),
            "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "./data/movie_embeddings.faiss"),
            "faiss_mapping_path": os.getenv("FAISS_MAPPING_PATH", "./data/movie_mappings.json"),
            "embedding_dim": int(os.getenv("EMBEDDING_DIM", "64")),
            "popular_movies_path": os.getenv("POPULAR_MOVIES_PATH", "./data/popular_movies.json"),
            "trending_movies_path": os.getenv(
                "TRENDING_MOVIES_PATH", "./data/trending_movies.json"
            ),
            "use_gpu": os.getenv("USE_GPU", "false").lower() == "true",
        }

    try:
        # Load model
        model_path = Path(app.state.config.get("model_path", "./models/ncf_best.pth"))
        if model_path.exists() and NCF is not None:
            try:
                state.model = NCF.load(str(model_path), device=state.device)
                state.model.eval()  # Set to evaluation mode
                logger.info(f"Model loaded from {model_path}")

                # Load ID to index mappings if they exist
                model_dir = model_path.parent
                user_mapping_path = model_dir / "user_id_mappings.json"
                movie_mapping_path = model_dir / "movie_id_mappings.json"

                # Load user ID mappings
                if user_mapping_path.exists():
                    with open(user_mapping_path) as f:
                        state.user_id_to_index = json.load(f)
                        state.index_to_user_id = {v: k for k, v in state.user_id_to_index.items()}
                    logger.info(f"Loaded {len(state.user_id_to_index)} user ID mappings")
                else:
                    logger.warning(f"User ID mappings not found at {user_mapping_path}")

                # Load movie ID mappings (also check FAISS mapping path)
                if movie_mapping_path.exists():
                    with open(movie_mapping_path) as f:
                        movie_mappings = json.load(f)
                        # Convert to integer indices if needed
                        state.movie_id_to_index = {
                            k: int(v) if isinstance(v, (int, str)) else v
                            for k, v in movie_mappings.items()
                        }
                        state.index_to_movie_id = {v: k for k, v in state.movie_id_to_index.items()}
                    logger.info(
                        "Loaded %d movie ID mappings from model directory",
                        len(state.movie_id_to_index),
                    )
                else:
                    logger.warning(f"Movie ID mappings not found at {movie_mapping_path}")

            except Exception as e:
                logger.error(f"Failed to load model: {e}", exc_info=True)
                state.model = None
        else:
            if not model_path.exists():
                logger.warning(f"Model not found at {model_path}, using placeholder")
            if NCF is None:
                logger.warning("NCF model class not available, using placeholder")

        # Initialize Redis
        redis_host = app.state.config.get("redis_host", "localhost")
        redis_port = app.state.config.get("redis_port", 6379)
        redis_db = app.state.config.get("redis_db", 0)

        try:
            state.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Test connection
            state.redis_client.ping()
            logger.info(f"Redis connected to {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            state.redis_client = None

        # Initialize FAISS index
        embedding_dim = app.state.config.get("embedding_dim", 64)
        index_path = Path(app.state.config.get("faiss_index_path", "./data/movie_embeddings.faiss"))

        try:
            if index_path.exists():
                state.faiss_index = faiss.read_index(str(index_path))

                # Load movie ID mappings
                mapping_path = Path(
                    app.state.config.get("faiss_mapping_path", "./data/movie_mappings.json")
                )
                if mapping_path.exists():
                    with open(mapping_path) as f:
                        mappings = json.load(f)
                        state.movie_id_to_index = mappings
                        state.index_to_movie_id = {v: k for k, v in mappings.items()}

                logger.info(f"FAISS index loaded with {state.faiss_index.ntotal} vectors")
            else:
                logger.warning(f"FAISS index not found at {index_path}, creating empty index")
                state.faiss_index = faiss.IndexFlatL2(embedding_dim)
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            state.faiss_index = faiss.IndexFlatL2(embedding_dim)

        # Load popular/trending movies
        try:
            popular_path = Path(
                app.state.config.get("popular_movies_path", "./data/popular_movies.json")
            )
            if popular_path.exists():
                with open(popular_path) as f:
                    state.popular_movies = json.load(f)

            trending_path = Path(
                app.state.config.get("trending_movies_path", "./data/trending_movies.json")
            )
            if trending_path.exists():
                with open(trending_path) as f:
                    state.trending_movies = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load popular/trending movies: {e}")

        # Set device
        if torch.cuda.is_available() and app.state.config.get("use_gpu", False):
            state.device = torch.device("cuda")
            if state.model:
                state.model = state.model.to(state.device)
            logger.info("Using GPU for inference")
        else:
            logger.info("Using CPU for inference")

        logger.info("Recommendation service started successfully")

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down recommendation service...")
    if state.redis_client:
        state.redis_client.close()
    logger.info("Shutdown complete")


# FastAPI app
app = FastAPI(
    title="Movie Recommendation Service",
    version="1.0.0",
    description="""
    Production-ready movie recommendation API with two-stage retrieval and ranking.

    Features:
    - FAISS-based approximate nearest neighbor retrieval
    - NCF model-based ranking
    - Redis caching for performance
    - Fallback strategies for cold-start users
    - Batch recommendation support
    """,
    lifespan=lifespan,
)

# Store config in app state
app.state.config = {}


def get_config() -> Dict[str, Any]:
    """Get application configuration."""
    return app.state.config


@app.on_event("startup")
async def startup_event():
    """Additional startup tasks."""
    # Load configuration from environment or config file
    import os

    app.state.config = {
        "model_path": os.getenv("MODEL_PATH", "./models/ncf_best.pth"),
        "redis_host": os.getenv("REDIS_HOST", "localhost"),
        "redis_port": int(os.getenv("REDIS_PORT", "6379")),
        "redis_db": int(os.getenv("REDIS_DB", "0")),
        "cache_ttl": int(os.getenv("CACHE_TTL", "300")),  # 5 minutes
        "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "./data/movie_embeddings.faiss"),
        "faiss_mapping_path": os.getenv("FAISS_MAPPING_PATH", "./data/movie_mappings.json"),
        "embedding_dim": int(os.getenv("EMBEDDING_DIM", "64")),
        "popular_movies_path": os.getenv("POPULAR_MOVIES_PATH", "./data/popular_movies.json"),
        "trending_movies_path": os.getenv("TRENDING_MOVIES_PATH", "./data/trending_movies.json"),
        "use_gpu": os.getenv("USE_GPU", "false").lower() == "true",
    }


def get_user_embedding(user_id: str) -> np.ndarray:
    """
    Get user embedding from model or feature store.

    Extracts user embedding from the NCF model's embedding layers.
    Falls back to feature store if model is not available.

    Args:
        user_id: User identifier

    Returns:
        User embedding vector

    Raises:
        ValueError: If user_id cannot be mapped to an index
    """
    embedding_dim = app.state.config.get("embedding_dim", 64)

    # Try to get embedding from model
    if state.model is not None:
        # Get user index
        if user_id in state.user_id_to_index:
            user_idx = state.user_id_to_index[user_id]
        else:
            # Unknown user - use a default index (0) or hash-based
            # In production, you might want to handle new users differently
            logger.warning(f"User {user_id} not in mapping, using hash-based index")
            user_idx = hash(user_id) % state.model.num_users

        # Extract embedding from model
        # NCF has two embedding paths: GMF and MLP
        # We'll use the MLP embedding as it's typically richer
        with torch.no_grad():
            user_idx_tensor = torch.tensor([user_idx], dtype=torch.long).to(state.device)

            # Get MLP embedding (combining both paths for richer representation)
            user_emb_mlp = state.model.user_embedding_mlp(user_idx_tensor)
            user_emb_gmf = state.model.user_embedding_gmf(user_idx_tensor)

            # Combine embeddings (average or concatenate)
            # Using average for simplicity, but concatenation is also valid
            user_embedding = (user_emb_mlp + user_emb_gmf) / 2.0

            # Convert to numpy
            embedding = user_embedding.cpu().numpy().flatten().astype("float32")

            return embedding

    # Fallback: try feature store
    if state.feature_store is not None:
        try:
            from datetime import datetime

            user_features = state.feature_store.get_user_features(user_id, datetime.now())

            # Extract embedding from features if available
            if "embedding" in user_features:
                embedding = np.array(user_features["embedding"], dtype="float32")
                if embedding.shape[0] == embedding_dim:
                    return embedding
        except Exception as e:
            logger.warning(f"Failed to get embedding from feature store: {e}")

    # Final fallback: return random embedding
    logger.warning(f"Using random embedding for user {user_id}")
    return np.random.randn(embedding_dim).astype("float32")


async def retrieve_candidates(user_id: str, k: int = 100) -> List[str]:
    """
    Stage 1: Retrieve top-k candidates using FAISS.

    Args:
        user_id: User identifier
        k: Number of candidates to retrieve

    Returns:
        List of movie IDs (candidates)
    """
    if state.faiss_index is None or state.faiss_index.ntotal == 0:
        logger.warning("FAISS index not available, returning empty candidates")
        return []

    # Get user embedding
    user_embedding = get_user_embedding(user_id)
    user_embedding = user_embedding.reshape(1, -1)

    # Search in FAISS index
    distances, indices = state.faiss_index.search(user_embedding, k)

    # Convert indices to movie IDs
    candidate_movie_ids = []
    for idx in indices[0]:
        if idx in state.index_to_movie_id:
            candidate_movie_ids.append(state.index_to_movie_id[idx])

    return candidate_movie_ids


async def rank_candidates(user_id: str, candidate_movie_ids: List[str]) -> List[Tuple[str, float]]:
    """
    Stage 2: Rank candidates using NCF model.

    Args:
        user_id: User identifier
        candidate_movie_ids: List of candidate movie IDs

    Returns:
        List of (movie_id, score) tuples sorted by score descending
    """
    if state.model is None:
        logger.warning("Model not available, returning random scores")
        return [(movie_id, np.random.rand()) for movie_id in candidate_movie_ids]

    if not candidate_movie_ids:
        return []

    # Convert user_id and movie_ids to integer indices
    # Get user index
    if user_id in state.user_id_to_index:
        user_idx = state.user_id_to_index[user_id]
    else:
        # Unknown user - use hash-based mapping with model's num_users
        logger.warning(f"User {user_id} not in mapping, using hash-based index")
        if state.model is not None:
            user_idx = hash(user_id) % state.model.num_users
        else:
            user_idx = hash(user_id) % 10000  # Fallback

    # Get movie indices
    movie_indices = []
    for movie_id in candidate_movie_ids:
        if movie_id in state.movie_id_to_index:
            movie_indices.append(state.movie_id_to_index[movie_id])
        else:
            # Unknown movie - use hash-based mapping
            logger.warning(f"Movie {movie_id} not in mapping, using hash-based index")
            if state.model is not None:
                movie_idx = hash(movie_id) % state.model.num_items
            else:
                movie_idx = hash(movie_id) % 5000  # Fallback
            movie_indices.append(movie_idx)

    # Batch prediction
    user_ids_tensor = torch.tensor([user_idx] * len(movie_indices), dtype=torch.long)
    movie_ids_tensor = torch.tensor(movie_indices, dtype=torch.long)

    state.model.eval()
    with torch.no_grad():
        user_ids_tensor = user_ids_tensor.to(state.device)
        movie_ids_tensor = movie_ids_tensor.to(state.device)

        # Get predictions
        predictions = state.model.predict(user_ids_tensor, movie_ids_tensor)
        scores = predictions.cpu().numpy().flatten()

    # Combine movie IDs with scores and sort
    ranked = list(zip(candidate_movie_ids, scores))
    ranked.sort(key=lambda x: x[1], reverse=True)

    return ranked


async def get_fallback_recommendations(user_id: str, k: int = 10) -> List[Tuple[str, float]]:
    """
    Get fallback recommendations when no personalized recommendations available.

    Fallback order:
    1. Popular in user's region (if available)
    2. Trending globally
    3. Random popular movies

    Args:
        user_id: User identifier
        k: Number of recommendations

    Returns:
        List of (movie_id, score) tuples
    """
    # Try trending movies first
    if state.trending_movies:
        movies = state.trending_movies[:k]
        return [(movie_id, 0.7) for movie_id in movies]

    # Fall back to popular movies
    if state.popular_movies:
        movies = state.popular_movies[:k]
        return [(movie_id, 0.6) for movie_id in movies]

    # Last resort: random (in production, this should query database)
    logger.warning(f"No fallback movies available for user {user_id}")
    return []


async def get_cached_recommendations(user_id: str, k: int) -> Optional[List[RecommendationItem]]:
    """
    Get recommendations from Redis cache.

    Args:
        user_id: User identifier
        k: Number of recommendations

    Returns:
        List of recommendations if cached, None otherwise
    """
    if state.redis_client is None:
        return None

    try:
        cache_key = f"recommendations:{user_id}:{k}"
        cached = state.redis_client.get(cache_key)

        if cached:
            CACHE_HITS.labels(type="recommendations").inc()
            data = json.loads(cached)
            return [RecommendationItem(**item) for item in data]
        else:
            CACHE_MISSES.labels(type="recommendations").inc()
            return None

    except Exception as e:
        logger.warning(f"Cache read failed: {e}")
        return None


async def cache_recommendations(
    user_id: str, k: int, recommendations: List[RecommendationItem], ttl: int = 300
) -> None:
    """
    Cache recommendations in Redis.

    Args:
        user_id: User identifier
        k: Number of recommendations
        recommendations: List of recommendations to cache
        ttl: Time to live in seconds (default: 300 = 5 minutes)
    """
    if state.redis_client is None:
        return

    try:
        cache_key = f"recommendations:{user_id}:{k}"
        data = [rec.dict() for rec in recommendations]
        state.redis_client.setex(cache_key, ttl, json.dumps(data))
    except Exception as e:
        logger.warning(f"Cache write failed: {e}")


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationResponse,
    summary="Get personalized recommendations",
    description="""
    Get personalized movie recommendations for a user using two-stage process:
    1. Retrieval: FAISS-based approximate nearest neighbor search
    2. Ranking: NCF model-based scoring

    Results are cached in Redis for 5 minutes by default.
    """,
    tags=["Recommendations"],
)
async def get_recommendations(
    user_id: str,
    k: int = Query(10, ge=1, le=100, description="Number of recommendations"),
    force_refresh: bool = Query(False, description="Force refresh cache"),
    request: Request = None,
) -> RecommendationResponse:
    """
    Get personalized recommendations for a user.

    Args:
        user_id: User identifier
        k: Number of recommendations to return
        force_refresh: Whether to bypass cache

    Returns:
        RecommendationResponse with recommendations and timing metrics

    Raises:
        HTTPException: If recommendation generation fails
    """
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        # Check cache first (unless force_refresh)
        cached_recommendations = None
        if not force_refresh:
            cached_recommendations = await get_cached_recommendations(user_id, k)
            if cached_recommendations:
                total_time_ms = (time.time() - start_time) * 1000
                REQUEST_COUNTER.labels(endpoint="get_recommendations", status="success").inc()
                REQUEST_DURATION.labels(endpoint="get_recommendations").observe(
                    total_time_ms / 1000
                )
                ACTIVE_REQUESTS.dec()

                logger.info(
                    "Recommendations retrieved from cache",
                    extra={
                        "user_id": user_id,
                        "k": k,
                        "total_time_ms": total_time_ms,
                        "cached": True,
                    },
                )

                return RecommendationResponse(
                    user_id=user_id,
                    recommendations=cached_recommendations[:k],
                    cached=True,
                    retrieval_time_ms=0.0,
                    ranking_time_ms=0.0,
                    total_time_ms=total_time_ms,
                )

        # Stage 1: Retrieval
        retrieval_start = time.time()
        candidates = await retrieve_candidates(user_id, k=100)
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Stage 2: Ranking
        ranking_start = time.time()
        if candidates:
            ranked = await rank_candidates(user_id, candidates)
            recommendations = [
                RecommendationItem(movie_id=movie_id, score=float(score))
                for movie_id, score in ranked[:k]
            ]
        else:
            # Fallback strategies
            logger.info(f"No candidates found for user {user_id}, using fallback")
            fallback = await get_fallback_recommendations(user_id, k)
            recommendations = [
                RecommendationItem(movie_id=movie_id, score=float(score))
                for movie_id, score in fallback[:k]
            ]
        ranking_time = (time.time() - ranking_start) * 1000

        # Cache results
        await cache_recommendations(
            user_id, k, recommendations, ttl=app.state.config.get("cache_ttl", 300)
        )

        total_time = (time.time() - start_time) * 1000

        # Log metrics
        REQUEST_COUNTER.labels(endpoint="get_recommendations", status="success").inc()
        REQUEST_DURATION.labels(endpoint="get_recommendations").observe(total_time / 1000)

        # Log request
        logger.info(
            "Recommendations generated",
            extra={
                "user_id": user_id,
                "k": k,
                "total_time_ms": total_time,
                "retrieval_time_ms": retrieval_time,
                "ranking_time_ms": ranking_time,
                "cached": False,
            },
        )

        ACTIVE_REQUESTS.dec()

        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            cached=False,
            retrieval_time_ms=retrieval_time,
            ranking_time_ms=ranking_time,
            total_time_ms=total_time,
        )

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="get_recommendations", status="error").inc()
        ACTIVE_REQUESTS.dec()
        logger.error(
            f"Failed to generate recommendations: {e}", extra={"user_id": user_id}, exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Failed to generate recommendations: {str(e)}")


@app.post(
    "/recommend/batch",
    response_model=BatchRecommendationResponse,
    summary="Get batch recommendations",
    description="""
    Get personalized recommendations for multiple users in a single request.
    Uses batch prediction for efficiency.
    """,
    tags=["Recommendations"],
)
async def get_batch_recommendations(
    request_body: BatchRecommendationRequest,
) -> BatchRecommendationResponse:
    """
    Get recommendations for multiple users.

    Args:
        request_body: BatchRecommendationRequest with user_ids and parameters

    Returns:
        BatchRecommendationResponse with recommendations for each user
    """
    start_time = time.time()
    ACTIVE_REQUESTS.inc()

    try:
        user_ids = request_body.user_ids
        k = request_body.k
        force_refresh = request_body.force_refresh

        results = {}

        # Process each user
        for user_id in user_ids:
            try:
                # Reuse single recommendation endpoint logic
                response = await get_recommendations(user_id, k, force_refresh)
                results[user_id] = response.recommendations
            except Exception as e:
                logger.warning(f"Failed to get recommendations for user {user_id}: {e}")
                # Use fallback for this user
                fallback = await get_fallback_recommendations(user_id, k)
                results[user_id] = [
                    RecommendationItem(movie_id=movie_id, score=float(score))
                    for movie_id, score in fallback[:k]
                ]

        total_time = (time.time() - start_time) * 1000

        REQUEST_COUNTER.labels(endpoint="batch_recommendations", status="success").inc()
        REQUEST_DURATION.labels(endpoint="batch_recommendations").observe(total_time / 1000)
        ACTIVE_REQUESTS.dec()

        logger.info(
            "Batch recommendations generated",
            extra={"num_users": len(user_ids), "total_time_ms": total_time},
        )

        return BatchRecommendationResponse(recommendations=results, total_time_ms=total_time)

    except Exception as e:
        REQUEST_COUNTER.labels(endpoint="batch_recommendations", status="error").inc()
        ACTIVE_REQUESTS.dec()
        logger.error(f"Failed to generate batch recommendations: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Failed to generate batch recommendations: {str(e)}"
        )


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    health = {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "model": state.model is not None,
            "redis": state.redis_client is not None
            and (state.redis_client.ping() if state.redis_client else False),
            "faiss": state.faiss_index is not None and state.faiss_index.ntotal > 0,
        },
    }

    if all(health["components"].values()):
        return health
    else:
        health["status"] = "degraded"
        return JSONResponse(status_code=503, content=health)


@app.get("/metrics", tags=["Metrics"])
async def get_metrics() -> Dict[str, Any]:
    """Get Prometheus metrics summary."""
    # This would typically be exposed via Prometheus client library
    # For now, return basic metrics
    return {"message": "Metrics available at /metrics endpoint (Prometheus format)"}
