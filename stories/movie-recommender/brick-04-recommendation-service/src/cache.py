"""Redis-based cache manager for recommendation service.

This module provides a high-level cache abstraction with JSON serialization,
compression, statistics tracking, and graceful degradation when Redis is
unavailable.

Features:
- JSON serialization for complex objects
- Automatic compression for large values (>10KB)
- Cache statistics (hits, misses, hit rate)
- Pattern-based invalidation
- Graceful degradation when Redis unavailable
- Key patterns for recommendations, user features, movie features
"""

import logging
import json
import gzip
import pickle
from typing import Any, Optional, Callable, Dict
from pathlib import Path
import hashlib
import redis
from collections import defaultdict
from threading import Lock
import time

logger = logging.getLogger(__name__)

# Compression threshold (bytes)
COMPRESSION_THRESHOLD = 10 * 1024  # 10KB


class CacheManager:
    """
    Redis-based cache manager with compression and statistics.

    Provides a high-level caching interface with automatic serialization,
    compression for large values, cache statistics, and graceful degradation.

    Args:
        redis_client: Redis client instance (can be None for graceful degradation)
        enable_compression: Whether to compress large values (default: True)
        compression_threshold: Minimum size in bytes to trigger compression (default: 10KB)

    Attributes:
        redis_client: Redis client instance
        stats: Dictionary tracking cache statistics
        _lock: Thread lock for statistics updates

    Example:
        >>> redis_client = redis.Redis(host='localhost', port=6379)
        >>> cache = CacheManager(redis_client)
        >>>
        >>> # Simple get/set
        >>> cache.set("key", {"data": "value"}, ttl=300)
        >>> value = cache.get("key")
        >>>
        >>> # Get or compute pattern
        >>> value = cache.get_or_compute(
        ...     "expensive_key",
        ...     lambda: expensive_computation(),
        ...     ttl=300
        ... )
    """

    def __init__(
        self,
        redis_client: Optional[redis.Redis],
        enable_compression: bool = True,
        compression_threshold: int = COMPRESSION_THRESHOLD
    ) -> None:
        """Initialize cache manager."""
        self.redis_client = redis_client
        self.enable_compression = enable_compression
        self.compression_threshold = compression_threshold

        # Cache statistics
        self.stats = defaultdict(int)
        self.stats['total_requests'] = 0
        self.stats['hits'] = 0
        self.stats['misses'] = 0
        self.stats['sets'] = 0
        self.stats['deletes'] = 0
        self.stats['errors'] = 0
        self._stats_lock = Lock()

        # Test Redis connection
        self._redis_available = False
        if self.redis_client is not None:
            try:
                self.redis_client.ping()
                self._redis_available = True
                logger.info("Cache manager initialized with Redis connection")
            except Exception as e:
                logger.warning(f"Redis unavailable: {e}, operating in degraded mode")
                self._redis_available = False
        else:
            logger.warning("No Redis client provided, operating in degraded mode")

    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Automatically deserializes JSON and decompresses if needed.

        Args:
            key: Cache key

        Returns:
            Cached value if found, None otherwise

        Example:
            >>> value = cache.get("recommendations:user_123:10")
            >>> if value:
            ...     print(f"Found {len(value)} recommendations")
        """
        if not self._redis_available:
            self._increment_stat('misses')
            return None

        try:
            cached_data = self.redis_client.get(key)

            if cached_data is None:
                self._increment_stat('misses')
                logger.debug(f"Cache miss for key: {key}")
                return None

            # Deserialize
            value = self._deserialize(cached_data)

            self._increment_stat('hits')
            logger.debug(f"Cache hit for key: {key}")

            return value

        except Exception as e:
            self._increment_stat('errors')
            logger.warning(f"Cache get error for key {key}: {e}")
            return None

    def set(
        self,
        key: str,
        value: Any,
        ttl: int = 300
    ) -> bool:
        """
        Set value in cache with TTL.

        Automatically serializes to JSON and compresses large values.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds (default: 300 = 5 minutes)

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = cache.set(
            ...     "recommendations:user_123:10",
            ...     [{"movie_id": "m1", "score": 0.9}],
            ...     ttl=300
            ... )
        """
        if not self._redis_available:
            return False

        try:
            # Serialize
            serialized = self._serialize(value)

            # Set with TTL
            self.redis_client.setex(key, ttl, serialized)

            self._increment_stat('sets')
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")

            return True

        except Exception as e:
            self._increment_stat('errors')
            logger.warning(f"Cache set error for key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key to delete

        Returns:
            True if deleted, False otherwise

        Example:
            >>> cache.delete("recommendations:user_123:10")
        """
        if not self._redis_available:
            return False

        try:
            result = self.redis_client.delete(key)
            self._increment_stat('deletes')
            logger.debug(f"Deleted cache key: {key}")
            return result > 0

        except Exception as e:
            self._increment_stat('errors')
            logger.warning(f"Cache delete error for key {key}: {e}")
            return False

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Any],
        ttl: int = 300
    ) -> Any:
        """
        Get value from cache or compute and cache it.

        Thread-safe pattern for cache-aside. If value not in cache,
        calls compute_fn() and caches the result.

        Args:
            key: Cache key
            compute_fn: Function to compute value if not cached
            ttl: Time to live in seconds (default: 300)

        Returns:
            Cached or computed value

        Example:
            >>> recommendations = cache.get_or_compute(
            ...     f"recommendations:{user_id}:{k}",
            ...     lambda: generate_recommendations(user_id, k),
            ...     ttl=300
            ... )
        """
        # Try to get from cache
        cached_value = self.get(key)

        if cached_value is not None:
            return cached_value

        # Compute value
        try:
            value = compute_fn()

            # Cache it
            self.set(key, value, ttl)

            return value

        except Exception as e:
            logger.error(f"Error computing value for key {key}: {e}", exc_info=True)
            raise

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all keys matching pattern.

        Uses SCAN for safe iteration over large key sets.

        Args:
            pattern: Redis key pattern (e.g., "recommendations:user_*:*")

        Returns:
            Number of keys deleted

        Example:
            >>> deleted = cache.invalidate_pattern("recommendations:user_123:*")
            >>> print(f"Deleted {deleted} keys")
        """
        if not self._redis_available:
            return 0

        try:
            deleted_count = 0
            cursor = 0

            # Use SCAN to iterate over keys matching pattern
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor=cursor,
                    match=pattern,
                    count=100  # Batch size
                )

                if keys:
                    deleted = self.redis_client.delete(*keys)
                    deleted_count += deleted
                    logger.debug(f"Deleted {deleted} keys matching pattern: {pattern}")

                if cursor == 0:
                    break

            if deleted_count > 0:
                self._increment_stat('deletes', amount=deleted_count)
                logger.info(f"Invalidated {deleted_count} keys matching pattern: {pattern}")

            return deleted_count

        except Exception as e:
            self._increment_stat('errors')
            logger.warning(f"Error invalidating pattern {pattern}: {e}")
            return 0

    def _serialize(self, value: Any) -> bytes:
        """
        Serialize value to bytes with optional compression.

        Args:
            value: Value to serialize

        Returns:
            Serialized bytes (potentially compressed)
        """
        # Serialize to JSON string
        json_str = json.dumps(value, default=str)
        json_bytes = json_str.encode('utf-8')

        # Compress if enabled and value is large enough
        if self.enable_compression and len(json_bytes) > self.compression_threshold:
            compressed = gzip.compress(json_bytes)

            # Only use compression if it actually saves space
            if len(compressed) < len(json_bytes):
                # Prepend marker to indicate compression
                marker = b'GZIP:'
                return marker + compressed
            else:
                return json_bytes
        else:
            return json_bytes

    def _deserialize(self, data: bytes) -> Any:
        """
        Deserialize bytes to value, handling compression.

        Args:
            data: Serialized bytes (potentially compressed)

        Returns:
            Deserialized value
        """
        # Check for compression marker
        if data.startswith(b'GZIP:'):
            # Decompress
            compressed = data[5:]  # Remove marker
            json_bytes = gzip.decompress(compressed)
        else:
            json_bytes = data

        # Deserialize JSON
        json_str = json_bytes.decode('utf-8')
        return json.loads(json_str)

    def _increment_stat(self, stat_name: str, amount: int = 1) -> None:
        """Thread-safe statistic increment."""
        with self._stats_lock:
            self.stats[stat_name] += amount
            if stat_name in ('hits', 'misses'):
                self.stats['total_requests'] = self.stats['hits'] + self.stats['misses']

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics including:
            - hits: Number of cache hits
            - misses: Number of cache misses
            - total_requests: Total get requests
            - hit_rate: Cache hit rate (0-1)
            - sets: Number of set operations
            - deletes: Number of delete operations
            - errors: Number of errors
            - redis_available: Whether Redis is available

        Example:
            >>> stats = cache.get_stats()
            >>> print(f"Hit rate: {stats['hit_rate']:.2%}")
        """
        with self._stats_lock:
            total = self.stats['total_requests']
            hits = self.stats['hits']

            hit_rate = hits / total if total > 0 else 0.0

            return {
                'hits': hits,
                'misses': self.stats['misses'],
                'total_requests': total,
                'hit_rate': hit_rate,
                'sets': self.stats['sets'],
                'deletes': self.stats['deletes'],
                'errors': self.stats['errors'],
                'redis_available': self._redis_available
            }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        with self._stats_lock:
            self.stats.clear()
            self.stats['total_requests'] = 0
            self.stats['hits'] = 0
            self.stats['misses'] = 0
            self.stats['sets'] = 0
            self.stats['deletes'] = 0
            self.stats['errors'] = 0

        logger.info("Cache statistics reset")

    # Convenience methods for common key patterns

    def get_recommendations(
        self,
        user_id: str,
        k: int
    ) -> Optional[Any]:
        """
        Get cached recommendations for a user.

        Args:
            user_id: User identifier
            k: Number of recommendations

        Returns:
            Cached recommendations or None
        """
        key = f"recommendations:{user_id}:{k}"
        return self.get(key)

    def set_recommendations(
        self,
        user_id: str,
        k: int,
        recommendations: Any,
        ttl: int = 300
    ) -> bool:
        """
        Cache recommendations for a user.

        Args:
            user_id: User identifier
            k: Number of recommendations
            recommendations: Recommendations to cache
            ttl: Time to live in seconds (default: 300)

        Returns:
            True if cached successfully
        """
        key = f"recommendations:{user_id}:{k}"
        return self.set(key, recommendations, ttl)

    def invalidate_user_recommendations(self, user_id: str) -> int:
        """
        Invalidate all recommendations for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of keys deleted
        """
        pattern = f"recommendations:{user_id}:*"
        return self.invalidate_pattern(pattern)

    def get_user_features(self, user_id: str) -> Optional[Any]:
        """
        Get cached user features.

        Args:
            user_id: User identifier

        Returns:
            Cached user features or None
        """
        key = f"user_features:{user_id}"
        return self.get(key)

    def set_user_features(
        self,
        user_id: str,
        features: Any,
        ttl: int = 3600  # User features change less frequently
    ) -> bool:
        """
        Cache user features.

        Args:
            user_id: User identifier
            features: User features to cache
            ttl: Time to live in seconds (default: 3600 = 1 hour)

        Returns:
            True if cached successfully
        """
        key = f"user_features:{user_id}"
        return self.set(key, features, ttl)

    def get_movie_features(self, movie_id: str) -> Optional[Any]:
        """
        Get cached movie features.

        Args:
            movie_id: Movie identifier

        Returns:
            Cached movie features or None
        """
        key = f"movie_features:{movie_id}"
        return self.get(key)

    def set_movie_features(
        self,
        movie_id: str,
        features: Any,
        ttl: int = 86400  # Movie features change rarely
    ) -> bool:
        """
        Cache movie features.

        Args:
            movie_id: Movie identifier
            features: Movie features to cache
            ttl: Time to live in seconds (default: 86400 = 24 hours)

        Returns:
            True if cached successfully
        """
        key = f"movie_features:{movie_id}"
        return self.set(key, features, ttl)

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on cache.

        Returns:
            Dictionary with health status
        """
        if not self._redis_available:
            return {
                'status': 'degraded',
                'redis_available': False,
                'message': 'Redis unavailable, operating in degraded mode'
            }

        try:
            # Test connection
            start_time = time.time()
            self.redis_client.ping()
            latency_ms = (time.time() - start_time) * 1000

            # Get stats
            stats = self.get_stats()

            return {
                'status': 'healthy',
                'redis_available': True,
                'latency_ms': latency_ms,
                'hit_rate': stats['hit_rate'],
                'stats': stats
            }

        except Exception as e:
            return {
                'status': 'unhealthy',
                'redis_available': False,
                'error': str(e)
            }
