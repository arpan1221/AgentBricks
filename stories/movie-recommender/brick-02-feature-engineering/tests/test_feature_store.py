"""Unit tests for feature store."""

import pytest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd

from src.feature_store import FeatureStore


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield db_path

    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def feature_store(temp_db):
    """Create a feature store instance for testing."""
    store = FeatureStore(temp_db)
    store.create_tables()
    yield store
    store.close()


class TestFeatureStoreInitialization:
    """Tests for FeatureStore initialization."""

    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database file."""
        store = FeatureStore(temp_db)
        store.close()

        assert os.path.exists(temp_db)

    def test_init_with_invalid_path_raises_error(self):
        """Test that invalid path raises error."""
        # Try to create database in non-existent directory
        invalid_path = "/non/existent/path/features.db"

        with pytest.raises((ValueError, Exception)):
            FeatureStore(invalid_path)

    def test_create_tables_creates_all_tables(self, feature_store):
        """Test that create_tables creates all required tables."""
        conn = feature_store._get_connection()

        # Check that tables exist
        tables = conn.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
        """).fetchall()

        table_names = [t[0] for t in tables]
        assert "user_features" in table_names
        assert "movie_features" in table_names
        assert "interaction_features" in table_names
        assert "feature_versions" in table_names


class TestSaveUserFeatures:
    """Tests for save_user_features method."""

    def test_save_user_features_stores_features(self, feature_store):
        """Test that save_user_features stores features correctly."""
        user_id = "user_123"
        features = {
            "total_watch_time": 36000.0,
            "watch_count_last_7days": 5,
            "avg_rating_given": 4.5
        }
        as_of_date = datetime.now()

        feature_store.save_user_features(user_id, features, as_of_date)

        # Verify features were saved
        retrieved = feature_store.get_user_features(user_id, as_of_date)
        assert retrieved["total_watch_time"] == 36000.0
        assert retrieved["watch_count_last_7days"] == 5
        assert retrieved["avg_rating_given"] == 4.5

    def test_save_user_features_overwrites_existing(self, feature_store):
        """Test that saving again overwrites existing features."""
        user_id = "user_123"
        as_of_date = datetime.now()

        # Save initial features
        feature_store.save_user_features(
            user_id,
            {"total_watch_time": 36000.0},
            as_of_date
        )

        # Save updated features
        feature_store.save_user_features(
            user_id,
            {"total_watch_time": 72000.0},
            as_of_date
        )

        # Verify updated value
        retrieved = feature_store.get_user_features(user_id, as_of_date)
        assert retrieved["total_watch_time"] == 72000.0

    def test_save_user_features_with_empty_dict_handles_gracefully(self, feature_store):
        """Test that empty features dict is handled gracefully."""
        user_id = "user_123"
        as_of_date = datetime.now()

        # Should not raise error
        feature_store.save_user_features(user_id, {}, as_of_date)

        # Should return empty dict
        retrieved = feature_store.get_user_features(user_id, as_of_date)
        assert retrieved == {}

    def test_save_user_features_with_invalid_user_id_raises_error(self, feature_store):
        """Test that invalid user_id raises error."""
        with pytest.raises(ValueError):
            feature_store.save_user_features("", {"total_watch_time": 100}, datetime.now())

    def test_save_user_features_with_various_types(self, feature_store):
        """Test that various feature types are saved correctly."""
        user_id = "user_123"
        as_of_date = datetime.now()

        features = {
            "total_watch_time": 36000.0,  # float
            "watch_count": 10,  # int
            "favorite_genre": "Action",  # str
            "is_active": True,  # bool
            "genres_watched": ["Action", "Sci-Fi"],  # list
            "preferences": {"theme": "dark"},  # dict
            "optional_feature": None  # None
        }

        feature_store.save_user_features(user_id, features, as_of_date)
        retrieved = feature_store.get_user_features(user_id, as_of_date)

        assert retrieved["total_watch_time"] == 36000.0
        assert retrieved["watch_count"] == 10
        assert retrieved["favorite_genre"] == "Action"
        assert retrieved["is_active"] is True
        assert retrieved["genres_watched"] == ["Action", "Sci-Fi"]
        assert retrieved["preferences"] == {"theme": "dark"}
        assert retrieved["optional_feature"] is None


class TestSaveMovieFeatures:
    """Tests for save_movie_features method."""

    def test_save_movie_features_stores_features(self, feature_store):
        """Test that save_movie_features stores features correctly."""
        movie_id = "movie_456"
        features = {
            "total_views": 1000,
            "avg_rating": 4.5,
            "completion_rate": 0.85
        }
        as_of_date = datetime.now()

        feature_store.save_movie_features(movie_id, features, as_of_date)

        # Verify features were saved
        retrieved = feature_store.get_movie_features(movie_id, as_of_date)
        assert retrieved["total_views"] == 1000
        assert retrieved["avg_rating"] == 4.5
        assert retrieved["completion_rate"] == 0.85


class TestGetUserFeatures:
    """Tests for get_user_features method."""

    def test_get_user_features_returns_features(self, feature_store):
        """Test that get_user_features returns saved features."""
        user_id = "user_123"
        features = {"total_watch_time": 36000.0}
        as_of_date = datetime.now()

        feature_store.save_user_features(user_id, features, as_of_date)
        retrieved = feature_store.get_user_features(user_id, as_of_date)

        assert retrieved["total_watch_time"] == 36000.0

    def test_get_user_features_point_in_time_query(self, feature_store):
        """Test point-in-time correctness of get_user_features."""
        user_id = "user_123"
        base_date = datetime(2024, 1, 1)

        # Save features for different dates
        feature_store.save_user_features(
            user_id,
            {"total_watch_time": 10000.0},
            base_date
        )

        feature_store.save_user_features(
            user_id,
            {"total_watch_time": 20000.0},
            base_date + timedelta(days=7)
        )

        # Query as of earlier date should return earlier features
        features_early = feature_store.get_user_features(
            user_id,
            base_date + timedelta(days=3)
        )
        assert features_early["total_watch_time"] == 10000.0

        # Query as of later date should return later features
        features_late = feature_store.get_user_features(
            user_id,
            base_date + timedelta(days=10)
        )
        assert features_late["total_watch_time"] == 20000.0

    def test_get_user_features_returns_empty_for_new_user(self, feature_store):
        """Test that new user returns empty dict."""
        user_id = "new_user"
        as_of_date = datetime.now()

        features = feature_store.get_user_features(user_id, as_of_date)
        assert features == {}

    def test_get_user_features_handles_missing_features_gracefully(self, feature_store):
        """Test that missing features are handled gracefully."""
        user_id = "user_123"
        as_of_date = datetime.now()

        # Save only one feature
        feature_store.save_user_features(
            user_id,
            {"total_watch_time": 36000.0},
            as_of_date
        )

        # Get all features (missing ones should not cause error)
        features = feature_store.get_user_features(user_id, as_of_date)
        assert "total_watch_time" in features
        assert "watch_count_last_7days" not in features  # Not saved


class TestGetMovieFeatures:
    """Tests for get_movie_features method."""

    def test_get_movie_features_returns_features(self, feature_store):
        """Test that get_movie_features returns saved features."""
        movie_id = "movie_456"
        features = {"total_views": 1000}
        as_of_date = datetime.now()

        feature_store.save_movie_features(movie_id, features, as_of_date)
        retrieved = feature_store.get_movie_features(movie_id, as_of_date)

        assert retrieved["total_views"] == 1000


class TestGetFeaturesBatch:
    """Tests for get_features_batch method."""

    def test_get_features_batch_returns_user_features(self, feature_store):
        """Test batch retrieval of user features."""
        user_ids = ["user_1", "user_2"]
        as_of_date = datetime.now()

        # Save features for users
        feature_store.save_user_features(
            "user_1",
            {"total_watch_time": 10000.0},
            as_of_date
        )
        feature_store.save_user_features(
            "user_2",
            {"total_watch_time": 20000.0},
            as_of_date
        )

        # Get batch features
        df = feature_store.get_features_batch(
            as_of_date=as_of_date,
            user_ids=user_ids
        )

        assert len(df) > 0
        assert "entity_id" in df.columns
        assert "entity_type" in df.columns
        assert "feature_name" in df.columns
        assert "feature_value" in df.columns

        # Check that both users are present
        user_1_features = df[df["entity_id"] == "user_1"]
        user_2_features = df[df["entity_id"] == "user_2"]

        assert len(user_1_features) > 0
        assert len(user_2_features) > 0

    def test_get_features_batch_returns_movie_features(self, feature_store):
        """Test batch retrieval of movie features."""
        movie_ids = ["movie_1", "movie_2"]
        as_of_date = datetime.now()

        # Save features for movies
        feature_store.save_movie_features(
            "movie_1",
            {"total_views": 1000},
            as_of_date
        )
        feature_store.save_movie_features(
            "movie_2",
            {"total_views": 2000},
            as_of_date
        )

        # Get batch features
        df = feature_store.get_features_batch(
            as_of_date=as_of_date,
            movie_ids=movie_ids
        )

        assert len(df) > 0
        movie_1_features = df[df["entity_id"] == "movie_1"]
        movie_2_features = df[df["entity_id"] == "movie_2"]

        assert len(movie_1_features) > 0
        assert len(movie_2_features) > 0

    def test_get_features_batch_returns_both_user_and_movie_features(self, feature_store):
        """Test batch retrieval of both user and movie features."""
        user_ids = ["user_1"]
        movie_ids = ["movie_1"]
        as_of_date = datetime.now()

        feature_store.save_user_features(
            "user_1",
            {"total_watch_time": 10000.0},
            as_of_date
        )
        feature_store.save_movie_features(
            "movie_1",
            {"total_views": 1000},
            as_of_date
        )

        df = feature_store.get_features_batch(
            as_of_date=as_of_date,
            user_ids=user_ids,
            movie_ids=movie_ids
        )

        # Should have both user and movie features
        user_features = df[df["entity_type"] == "user"]
        movie_features = df[df["entity_type"] == "movie"]

        assert len(user_features) > 0
        assert len(movie_features) > 0

    def test_get_features_batch_with_empty_ids_raises_error(self, feature_store):
        """Test that batch query with empty IDs raises error."""
        as_of_date = datetime.now()

        with pytest.raises(ValueError):
            feature_store.get_features_batch(
                as_of_date=as_of_date,
                user_ids=None,
                movie_ids=None
            )


class TestContextManager:
    """Tests for context manager support."""

    def test_feature_store_context_manager(self, temp_db):
        """Test that FeatureStore works as context manager."""
        with FeatureStore(temp_db) as store:
            store.create_tables()
            store.save_user_features(
                "user_123",
                {"total_watch_time": 36000.0},
                datetime.now()
            )
            # Should auto-close on exit
        # Connection should be closed after context exit
