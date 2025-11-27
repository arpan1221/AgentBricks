"""Integration tests for feature store operations.

Tests DuckDB feature store integration with real data.
"""

from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pytest


@pytest.mark.integration
class TestFeatureStoreIntegration:
    """Test feature store operations."""

    @pytest.fixture
    def feature_store_path(self, tmp_path: Path) -> Path:
        """Create a temporary feature store database."""
        db_path = tmp_path / "feature_store.duckdb"
        return db_path

    @pytest.fixture
    def feature_store(self, feature_store_path: Path) -> duckdb.DuckDBPyConnection:
        """Create a feature store connection."""
        conn = duckdb.connect(str(feature_store_path))

        # Create tables
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_features (
                user_id VARCHAR PRIMARY KEY,
                total_watch_time INTEGER,
                avg_watch_time FLOAT,
                favorite_genre VARCHAR,
                days_since_last_active INTEGER,
                updated_at TIMESTAMP
            )
        """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS movie_features (
                movie_id VARCHAR PRIMARY KEY,
                total_views INTEGER,
                avg_rating FLOAT,
                genre VARCHAR,
                release_year INTEGER,
                updated_at TIMESTAMP
            )
        """
        )

        yield conn
        conn.close()

    def test_store_user_features(self, feature_store: duckdb.DuckDBPyConnection):
        """Test storing user features in feature store."""
        user_id = "test_user_features"
        features = {
            "user_id": user_id,
            "total_watch_time": 7200,
            "avg_watch_time": 3600.0,
            "favorite_genre": "action",
            "days_since_last_active": 1,
            "updated_at": datetime.now(timezone.utc),
        }

        # Insert features
        feature_store.execute(
            """
            INSERT INTO user_features
            (user_id, total_watch_time, avg_watch_time, favorite_genre,
             days_since_last_active, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                features["user_id"],
                features["total_watch_time"],
                features["avg_watch_time"],
                features["favorite_genre"],
                features["days_since_last_active"],
                features["updated_at"],
            ],
        )

        # Retrieve features
        result = feature_store.execute(
            """
            SELECT * FROM user_features WHERE user_id = ?
        """,
            [user_id],
        ).fetchone()

        assert result is not None
        assert result[0] == user_id
        assert result[1] == features["total_watch_time"]
        assert result[2] == features["avg_watch_time"]
        assert result[3] == features["favorite_genre"]

    def test_store_movie_features(self, feature_store: duckdb.DuckDBPyConnection):
        """Test storing movie features in feature store."""
        movie_id = "test_movie_features"
        features = {
            "movie_id": movie_id,
            "total_views": 1000,
            "avg_rating": 4.5,
            "genre": "action",
            "release_year": 2020,
            "updated_at": datetime.now(timezone.utc),
        }

        # Insert features
        feature_store.execute(
            """
            INSERT INTO movie_features
            (movie_id, total_views, avg_rating, genre, release_year, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [
                features["movie_id"],
                features["total_views"],
                features["avg_rating"],
                features["genre"],
                features["release_year"],
                features["updated_at"],
            ],
        )

        # Retrieve features
        result = feature_store.execute(
            """
            SELECT * FROM movie_features WHERE movie_id = ?
        """,
            [movie_id],
        ).fetchone()

        assert result is not None
        assert result[0] == movie_id
        assert result[1] == features["total_views"]
        assert result[2] == features["avg_rating"]
        assert result[3] == features["genre"]

    def test_query_features_with_pandas(self, feature_store: duckdb.DuckDBPyConnection):
        """Test querying features using pandas."""
        # Insert test data
        user_ids = [f"user_{i}" for i in range(5)]
        for i, user_id in enumerate(user_ids):
            feature_store.execute(
                """
                INSERT INTO user_features
                (user_id, total_watch_time, avg_watch_time, favorite_genre,
                 days_since_last_active, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [user_id, 1000 * (i + 1), 500.0 * (i + 1), "action", 1, datetime.now(timezone.utc)],
            )

        # Query using pandas
        df = feature_store.execute(
            """
            SELECT * FROM user_features WHERE total_watch_time > 2000
        """
        ).df()

        assert len(df) > 0
        assert "user_id" in df.columns
        assert "total_watch_time" in df.columns
        assert all(df["total_watch_time"] > 2000)

    def test_update_features(self, feature_store: duckdb.DuckDBPyConnection):
        """Test updating existing features."""
        user_id = "test_user_update"

        # Insert initial features
        feature_store.execute(
            """
            INSERT INTO user_features
            (user_id, total_watch_time, avg_watch_time, favorite_genre,
             days_since_last_active, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            [user_id, 1000, 500.0, "action", 1, datetime.now(timezone.utc)],
        )

        # Update features
        new_watch_time = 2000
        feature_store.execute(
            """
            UPDATE user_features
            SET total_watch_time = ?, updated_at = ?
            WHERE user_id = ?
        """,
            [new_watch_time, datetime.now(timezone.utc), user_id],
        )

        # Verify update
        result = feature_store.execute(
            """
            SELECT total_watch_time FROM user_features WHERE user_id = ?
        """,
            [user_id],
        ).fetchone()

        assert result is not None
        assert result[0] == new_watch_time
