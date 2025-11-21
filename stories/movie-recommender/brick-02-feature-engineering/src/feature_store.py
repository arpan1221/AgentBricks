"""Feature store implementation using DuckDB.

This module provides a feature store for storing and retrieving user features,
movie features, and interaction features with point-in-time correctness. It uses
DuckDB for efficient storage and querying of time-series feature data.
"""

import logging
import duckdb
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import json
from pathlib import Path
from contextlib import contextmanager
import threading

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Feature store for storing and retrieving features with point-in-time correctness.

    Uses DuckDB to store features in a long format (entity_id, feature_name,
    feature_value, as_of_date) which enables efficient point-in-time queries.
    Features are versioned by as_of_date to support temporal correctness.

    Attributes:
        db_path: Path to DuckDB database file
        conn: DuckDB connection (thread-local)
        _lock: Thread lock for connection management

    Example:
        >>> store = FeatureStore("features.db")
        >>> store.create_tables()
        >>>
        >>> # Save user features
        >>> user_features = {
        ...     "total_watch_time": 36000.0,
        ...     "watch_count_last_7days": 5
        ... }
        >>> store.save_user_features("user_123", user_features, datetime.now())
        >>>
        >>> # Retrieve features as of a date
        >>> features = store.get_user_features("user_123", datetime.now())
        >>> print(features["total_watch_time"])
        36000.0
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        """
        Initialize feature store.

        Args:
            db_path: Path to DuckDB database file. If file exists, opens it;
                    otherwise creates new database.

        Raises:
            ValueError: If db_path is invalid
            Exception: If database connection fails

        Example:
            >>> store = FeatureStore("/path/to/features.db")
            >>> store.create_tables()
        """
        self.db_path = Path(db_path)

        # Thread-local storage for DuckDB connections
        # DuckDB connections are not thread-safe, so we use thread-local storage
        self._local = threading.local()
        self._lock = threading.Lock()

        # Ensure parent directory exists
        if self.db_path.parent:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Test connection
        try:
            conn = self._get_connection()
            conn.execute("SELECT 1").fetchall()
            logger.info(
                f"Feature store initialized at {self.db_path}",
                extra={"db_path": str(self.db_path)}
            )
        except Exception as e:
            logger.error(f"Failed to initialize feature store: {e}", exc_info=True)
            raise

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """
        Get thread-local DuckDB connection.

        Returns:
            DuckDB connection for current thread

        Raises:
            Exception: If connection creation fails
        """
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            try:
                self._local.conn = duckdb.connect(str(self.db_path))
                # Set pragmas for better performance
                self._local.conn.execute("PRAGMA threads=4")
                self._local.conn.execute("PRAGMA enable_checkpoint_on_shutdown")
                logger.debug(f"Created new DuckDB connection for thread")
            except Exception as e:
                logger.error(f"Failed to create DuckDB connection: {e}", exc_info=True)
                raise

        return self._local.conn

    @contextmanager
    def _transaction(self):
        """
        Context manager for database transactions.

        Ensures operations are atomic and handles rollback on errors.

        Yields:
            DuckDB connection

        Example:
            >>> with store._transaction() as conn:
            ...     conn.execute("INSERT INTO ...")
        """
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Transaction failed, rolling back: {e}", exc_info=True)
            raise

    def create_tables(self) -> None:
        """
        Create feature store tables with indexes.

        Creates three tables:
        - user_features: Stores user features (long format)
        - movie_features: Stores movie features (long format)
        - interaction_features: Stores interaction features (wide format)

        Also creates indexes for efficient point-in-time queries.

        Raises:
            Exception: If table creation fails

        Example:
            >>> store = FeatureStore("features.db")
            >>> store.create_tables()
        """
        conn = self._get_connection()

        try:
            # Drop tables if they exist (for development/testing)
            # In production, you might want to use IF NOT EXISTS instead
            conn.execute("DROP TABLE IF EXISTS user_features")
            conn.execute("DROP TABLE IF EXISTS movie_features")
            conn.execute("DROP TABLE IF EXISTS interaction_features")
            conn.execute("DROP TABLE IF EXISTS feature_versions")

            # Create user_features table (long format)
            conn.execute("""
                CREATE TABLE user_features (
                    user_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value VARCHAR,
                    feature_type VARCHAR,
                    as_of_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, feature_name, as_of_date)
                )
            """)

            # Create movie_features table (long format)
            conn.execute("""
                CREATE TABLE movie_features (
                    movie_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value VARCHAR,
                    feature_type VARCHAR,
                    as_of_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (movie_id, feature_name, as_of_date)
                )
            """)

            # Create interaction_features table (wide format for batch retrieval)
            conn.execute("""
                CREATE TABLE interaction_features (
                    user_id VARCHAR NOT NULL,
                    movie_id VARCHAR NOT NULL,
                    feature_name VARCHAR NOT NULL,
                    feature_value VARCHAR,
                    feature_type VARCHAR,
                    as_of_date TIMESTAMP NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (user_id, movie_id, feature_name, as_of_date)
                )
            """)

            # Create feature_versions table for tracking feature versions
            conn.execute("""
                CREATE TABLE feature_versions (
                    entity_type VARCHAR NOT NULL,
                    entity_id VARCHAR NOT NULL,
                    as_of_date TIMESTAMP NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (entity_type, entity_id, as_of_date)
                )
            """)

            # Create indexes for efficient point-in-time queries
            # Index on (entity_id, as_of_date) for fast lookups
            conn.execute("""
                CREATE INDEX idx_user_features_lookup
                ON user_features(user_id, as_of_date DESC)
            """)

            conn.execute("""
                CREATE INDEX idx_movie_features_lookup
                ON movie_features(movie_id, as_of_date DESC)
            """)

            conn.execute("""
                CREATE INDEX idx_interaction_features_lookup
                ON interaction_features(user_id, movie_id, as_of_date DESC)
            """)

            # Index on as_of_date for time-range queries
            conn.execute("""
                CREATE INDEX idx_user_features_date
                ON user_features(as_of_date)
            """)

            conn.execute("""
                CREATE INDEX idx_movie_features_date
                ON movie_features(as_of_date)
            """)

            conn.commit()

            logger.info(
                "Feature store tables created successfully",
                extra={"db_path": str(self.db_path)}
            )

        except Exception as e:
            logger.error(f"Failed to create tables: {e}", exc_info=True)
            raise

    def _serialize_value(self, value: Any) -> tuple[str, str]:
        """
        Serialize feature value to string with type information.

        Args:
            value: Feature value (can be int, float, str, bool, None, list, dict)

        Returns:
            Tuple of (serialized_value, value_type)

        Example:
            >>> store._serialize_value(123)
            ('123', 'int')
            >>> store._serialize_value([1, 2, 3])
            ('[1, 2, 3]', 'list')
        """
        if value is None:
            return ("NULL", "null")
        elif isinstance(value, (int, float, bool)):
            return (str(value), type(value).__name__)
        elif isinstance(value, str):
            return (value, "str")
        elif isinstance(value, (list, dict)):
            return (json.dumps(value), type(value).__name__)
        else:
            # For other types, convert to string
            return (str(value), type(value).__name__)

    def _deserialize_value(self, value: str, value_type: Optional[str]) -> Any:
        """
        Deserialize feature value from string based on type.

        Args:
            value: Serialized feature value
            value_type: Type of the value (int, float, str, bool, list, dict, null)

        Returns:
            Deserialized feature value

        Example:
            >>> store._deserialize_value("123", "int")
            123
            >>> store._deserialize_value("[1, 2, 3]", "list")
            [1, 2, 3]
        """
        if value is None or value == "NULL" or value_type == "null":
            return None

        try:
            if value_type == "int":
                return int(value)
            elif value_type == "float":
                return float(value)
            elif value_type == "bool":
                return value.lower() in ("true", "1", "yes")
            elif value_type == "list":
                return json.loads(value)
            elif value_type == "dict":
                return json.loads(value)
            else:
                # Default to string
                return value
        except (ValueError, json.JSONDecodeError) as e:
            logger.warning(
                f"Failed to deserialize value {value} as {value_type}: {e}",
                extra={"value": value, "value_type": value_type}
            )
            return value  # Return as string if deserialization fails

    def save_user_features(
        self,
        user_id: str,
        features_dict: Dict[str, Any],
        as_of_date: datetime
    ) -> None:
        """
        Save user features to feature store.

        Stores features in long format with point-in-time versioning.
        Overwrites existing features for the same (user_id, feature_name, as_of_date).

        Args:
            user_id: Unique user identifier
            features_dict: Dictionary mapping feature names to values
            as_of_date: Timestamp for point-in-time correctness

        Raises:
            ValueError: If user_id or as_of_date is invalid
            Exception: If save operation fails

        Example:
            >>> store.save_user_features(
            ...     "user_123",
            ...     {"total_watch_time": 36000.0, "watch_count_last_7days": 5},
            ...     datetime.now()
            ... )
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")

        if not isinstance(as_of_date, datetime):
            raise ValueError("as_of_date must be a datetime object")

        if not features_dict:
            logger.warning(f"No features provided for user {user_id}")
            return

        conn = self._get_connection()

        try:
            with self._transaction():
                # Delete existing features for this user and as_of_date
                # (allows updating features for the same timestamp)
                conn.execute("""
                    DELETE FROM user_features
                    WHERE user_id = ? AND as_of_date = ?
                """, [user_id, as_of_date])

                # Insert new features
                rows = []
                for feature_name, feature_value in features_dict.items():
                    serialized_value, value_type = self._serialize_value(feature_value)
                    rows.append({
                        "user_id": user_id,
                        "feature_name": feature_name,
                        "feature_value": serialized_value,
                        "feature_type": value_type,
                        "as_of_date": as_of_date
                    })

                if rows:
                    df = pd.DataFrame(rows)
                    conn.execute("""
                        INSERT INTO user_features
                        (user_id, feature_name, feature_value, feature_type, as_of_date)
                        SELECT * FROM df
                    """)

                # Update version tracking
                conn.execute("""
                    INSERT OR REPLACE INTO feature_versions
                    (entity_type, entity_id, as_of_date, version)
                    VALUES ('user', ?, ?,
                        COALESCE(
                            (SELECT MAX(version) FROM feature_versions
                             WHERE entity_type = 'user' AND entity_id = ?) + 1,
                            1
                        )
                    )
                """, [user_id, as_of_date, user_id])

                logger.debug(
                    f"Saved {len(features_dict)} features for user {user_id}",
                    extra={
                        "user_id": user_id,
                        "as_of_date": as_of_date.isoformat(),
                        "feature_count": len(features_dict)
                    }
                )

        except Exception as e:
            logger.error(
                f"Failed to save user features: {e}",
                extra={"user_id": user_id},
                exc_info=True
            )
            raise

    def save_movie_features(
        self,
        movie_id: str,
        features_dict: Dict[str, Any],
        as_of_date: datetime
    ) -> None:
        """
        Save movie features to feature store.

        Stores features in long format with point-in-time versioning.
        Similar to save_user_features but for movies.

        Args:
            movie_id: Unique movie identifier
            features_dict: Dictionary mapping feature names to values
            as_of_date: Timestamp for point-in-time correctness

        Raises:
            ValueError: If movie_id or as_of_date is invalid
            Exception: If save operation fails

        Example:
            >>> store.save_movie_features(
            ...     "movie_456",
            ...     {"total_views": 1000, "avg_rating": 4.5},
            ...     datetime.now()
            ... )
        """
        if not movie_id:
            raise ValueError("movie_id cannot be empty")

        if not isinstance(as_of_date, datetime):
            raise ValueError("as_of_date must be a datetime object")

        if not features_dict:
            logger.warning(f"No features provided for movie {movie_id}")
            return

        conn = self._get_connection()

        try:
            with self._transaction():
                # Delete existing features for this movie and as_of_date
                conn.execute("""
                    DELETE FROM movie_features
                    WHERE movie_id = ? AND as_of_date = ?
                """, [movie_id, as_of_date])

                # Insert new features
                rows = []
                for feature_name, feature_value in features_dict.items():
                    serialized_value, value_type = self._serialize_value(feature_value)
                    rows.append({
                        "movie_id": movie_id,
                        "feature_name": feature_name,
                        "feature_value": serialized_value,
                        "feature_type": value_type,
                        "as_of_date": as_of_date
                    })

                if rows:
                    df = pd.DataFrame(rows)
                    conn.execute("""
                        INSERT INTO movie_features
                        (movie_id, feature_name, feature_value, feature_type, as_of_date)
                        SELECT * FROM df
                    """)

                # Update version tracking
                conn.execute("""
                    INSERT OR REPLACE INTO feature_versions
                    (entity_type, entity_id, as_of_date, version)
                    VALUES ('movie', ?, ?,
                        COALESCE(
                            (SELECT MAX(version) FROM feature_versions
                             WHERE entity_type = 'movie' AND entity_id = ?) + 1,
                            1
                        )
                    )
                """, [movie_id, as_of_date, movie_id])

                logger.debug(
                    f"Saved {len(features_dict)} features for movie {movie_id}",
                    extra={
                        "movie_id": movie_id,
                        "as_of_date": as_of_date.isoformat(),
                        "feature_count": len(features_dict)
                    }
                )

        except Exception as e:
            logger.error(
                f"Failed to save movie features: {e}",
                extra={"movie_id": movie_id},
                exc_info=True
            )
            raise

    def get_user_features(
        self,
        user_id: str,
        as_of_date: datetime
    ) -> Dict[str, Any]:
        """
        Get user features as of a specific date (point-in-time query).

        Retrieves the latest features available on or before as_of_date.
        Returns empty dict if no features found.

        Args:
            user_id: Unique user identifier
            as_of_date: Get features as of this timestamp

        Returns:
            Dictionary mapping feature names to values. Empty dict if no features.

        Raises:
            ValueError: If user_id or as_of_date is invalid
            Exception: If query fails

        Example:
            >>> features = store.get_user_features("user_123", datetime.now())
            >>> print(features.get("total_watch_time"))
            36000.0
        """
        if not user_id:
            raise ValueError("user_id cannot be empty")

        if not isinstance(as_of_date, datetime):
            raise ValueError("as_of_date must be a datetime object")

        conn = self._get_connection()

        try:
            # Point-in-time query: get latest features on or before as_of_date
            result = conn.execute("""
                SELECT feature_name, feature_value, feature_type
                FROM (
                    SELECT
                        feature_name,
                        feature_value,
                        feature_type,
                        ROW_NUMBER() OVER (
                            PARTITION BY feature_name
                            ORDER BY as_of_date DESC
                        ) as rn
                    FROM user_features
                    WHERE user_id = ? AND as_of_date <= ?
                )
                WHERE rn = 1
            """, [user_id, as_of_date]).fetchdf()

            # Convert long format to wide format (dict)
            features = {}
            if not result.empty:
                for _, row in result.iterrows():
                    feature_name = row["feature_name"]
                    feature_value = row["feature_value"]
                    feature_type = row.get("feature_type")
                    features[feature_name] = self._deserialize_value(
                        feature_value, feature_type
                    )

            logger.debug(
                f"Retrieved {len(features)} features for user {user_id}",
                extra={
                    "user_id": user_id,
                    "as_of_date": as_of_date.isoformat(),
                    "feature_count": len(features)
                }
            )

            return features

        except Exception as e:
            logger.error(
                f"Failed to get user features: {e}",
                extra={"user_id": user_id},
                exc_info=True
            )
            raise

    def get_movie_features(
        self,
        movie_id: str,
        as_of_date: datetime
    ) -> Dict[str, Any]:
        """
        Get movie features as of a specific date (point-in-time query).

        Retrieves the latest features available on or before as_of_date.
        Returns empty dict if no features found.

        Args:
            movie_id: Unique movie identifier
            as_of_date: Get features as of this timestamp

        Returns:
            Dictionary mapping feature names to values. Empty dict if no features.

        Raises:
            ValueError: If movie_id or as_of_date is invalid
            Exception: If query fails

        Example:
            >>> features = store.get_movie_features("movie_456", datetime.now())
            >>> print(features.get("total_views"))
            1000
        """
        if not movie_id:
            raise ValueError("movie_id cannot be empty")

        if not isinstance(as_of_date, datetime):
            raise ValueError("as_of_date must be a datetime object")

        conn = self._get_connection()

        try:
            # Point-in-time query: get latest features on or before as_of_date
            result = conn.execute("""
                SELECT feature_name, feature_value, feature_type
                FROM (
                    SELECT
                        feature_name,
                        feature_value,
                        feature_type,
                        ROW_NUMBER() OVER (
                            PARTITION BY feature_name
                            ORDER BY as_of_date DESC
                        ) as rn
                    FROM movie_features
                    WHERE movie_id = ? AND as_of_date <= ?
                )
                WHERE rn = 1
            """, [movie_id, as_of_date]).fetchdf()

            # Convert long format to wide format (dict)
            features = {}
            if not result.empty:
                for _, row in result.iterrows():
                    feature_name = row["feature_name"]
                    feature_value = row["feature_value"]
                    feature_type = row.get("feature_type")
                    features[feature_name] = self._deserialize_value(
                        feature_value, feature_type
                    )

            logger.debug(
                f"Retrieved {len(features)} features for movie {movie_id}",
                extra={
                    "movie_id": movie_id,
                    "as_of_date": as_of_date.isoformat(),
                    "feature_count": len(features)
                }
            )

            return features

        except Exception as e:
            logger.error(
                f"Failed to get movie features: {e}",
                extra={"movie_id": movie_id},
                exc_info=True
            )
            raise

    def get_features_batch(
        self,
        as_of_date: datetime,
        user_ids: Optional[List[str]] = None,
        movie_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get features for multiple users and/or movies in batch.

        Efficiently retrieves features for multiple entities in a single query.
        Returns a DataFrame with columns: entity_id, entity_type, feature_name,
        feature_value, as_of_date.

        Args:
            as_of_date: Get features as of this timestamp
            user_ids: Optional list of user IDs. If None, all users are included.
            movie_ids: Optional list of movie IDs. If None, all movies are included.

        Returns:
            DataFrame with columns:
                - entity_id: User or movie ID
                - entity_type: 'user' or 'movie'
                - feature_name: Feature name
                - feature_value: Feature value (deserialized)
                - as_of_date: Timestamp of the feature
            Empty DataFrame if no features found.

        Raises:
            ValueError: If as_of_date is invalid or both user_ids and movie_ids are None
            Exception: If query fails

        Example:
            >>> df = store.get_features_batch(
            ...     user_ids=["user_123", "user_456"],
            ...     movie_ids=["movie_789"],
            ...     as_of_date=datetime.now()
            ... )
            >>> print(df.head())
        """
        if not isinstance(as_of_date, datetime):
            raise ValueError("as_of_date must be a datetime object")

        if not user_ids and not movie_ids:
            raise ValueError("At least one of user_ids or movie_ids must be provided")

        conn = self._get_connection()

        try:
            queries = []
            params = []

            # Build query for user features
            if user_ids:
                placeholders = ",".join(["?"] * len(user_ids))
                user_query = f"""
                    SELECT
                        user_id as entity_id,
                        'user' as entity_type,
                        feature_name,
                        feature_value,
                        feature_type,
                        as_of_date
                    FROM (
                        SELECT
                            user_id,
                            feature_name,
                            feature_value,
                            feature_type,
                            as_of_date,
                            ROW_NUMBER() OVER (
                                PARTITION BY user_id, feature_name
                                ORDER BY as_of_date DESC
                            ) as rn
                        FROM user_features
                        WHERE user_id IN ({placeholders})
                            AND as_of_date <= ?
                    )
                    WHERE rn = 1
                """
                queries.append(user_query)
                params.extend(user_ids + [as_of_date])

            # Build query for movie features
            if movie_ids:
                placeholders = ",".join(["?"] * len(movie_ids))
                movie_query = f"""
                    SELECT
                        movie_id as entity_id,
                        'movie' as entity_type,
                        feature_name,
                        feature_value,
                        feature_type,
                        as_of_date
                    FROM (
                        SELECT
                            movie_id,
                            feature_name,
                            feature_value,
                            feature_type,
                            as_of_date,
                            ROW_NUMBER() OVER (
                                PARTITION BY movie_id, feature_name
                                ORDER BY as_of_date DESC
                            ) as rn
                        FROM movie_features
                        WHERE movie_id IN ({placeholders})
                            AND as_of_date <= ?
                    )
                    WHERE rn = 1
                """
                queries.append(movie_query)
                params.extend(movie_ids + [as_of_date])

            # Combine queries
            combined_query = " UNION ALL ".join(queries)

            # Execute query
            result = conn.execute(combined_query, params).fetchdf()

            # Deserialize feature values
            if not result.empty and "feature_value" in result.columns:
                result["feature_value"] = result.apply(
                    lambda row: self._deserialize_value(
                        row["feature_value"],
                        row.get("feature_type")
                    ),
                    axis=1
                )
                # Drop feature_type column as it's no longer needed
                if "feature_type" in result.columns:
                    result = result.drop(columns=["feature_type"])

            logger.debug(
                f"Retrieved batch features",
                extra={
                    "user_count": len(user_ids) if user_ids else 0,
                    "movie_count": len(movie_ids) if movie_ids else 0,
                    "as_of_date": as_of_date.isoformat(),
                    "row_count": len(result)
                }
            )

            return result

        except Exception as e:
            logger.error(
                f"Failed to get batch features: {e}",
                exc_info=True
            )
            raise

    def close(self) -> None:
        """
        Close database connections.

        Closes all thread-local connections. Should be called when done with
        the feature store.

        Example:
            >>> store.close()
        """
        try:
            if hasattr(self._local, 'conn') and self._local.conn is not None:
                self._local.conn.close()
                self._local.conn = None
                logger.debug("Closed DuckDB connection")
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
