"""Feature engineering pipeline for movie recommendation system.

This module provides a feature engineering pipeline that computes user features,
movie features, and interaction features with temporal correctness (point-in-time
accuracy) to prevent data leakage in machine learning models.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering pipeline with temporal correctness.

    Computes user, movie, and interaction features while ensuring point-in-time
    correctness (no future data leakage). All features are computed as-of a
    specific timestamp to simulate real-time feature computation.

    Attributes:
        default_user_features: Default feature values for new users
        default_movie_features: Default feature values for new movies

    Example:
        >>> engineer = FeatureEngineer()
        >>> as_of_date = datetime.now()
        >>>
        >>> # Compute user features
        >>> user_features = engineer.compute_user_features(
        ...     user_id="user_123",
        ...     interactions_df=df,
        ...     as_of_date=as_of_date
        ... )
        >>> print(user_features["total_watch_time"])
        36000
        >>>
        >>> # Compute movie features
        >>> movie_features = engineer.compute_movie_features(
        ...     movie_id="movie_456",
        ...     interactions_df=df,
        ...     as_of_date=as_of_date
        ... )
        >>> print(movie_features["total_views"])
        150
    """

    def __init__(self) -> None:
        """Initialize feature engineer with default values."""
        self.default_user_features = {
            "total_watch_time": 0.0,
            "avg_watch_time_per_movie": 0.0,
            "watch_count_last_7days": 0,
            "watch_count_last_30days": 0,
            "watch_count_last_90days": 0,
            "unique_genres_watched": 0,
            "favorite_genre": None,
            "avg_rating_given": None,
            "skip_rate": 0.0,
            "peak_activity_hour": None,
            "days_since_last_watch": None,
        }

        self.default_movie_features = {
            "total_views": 0,
            "unique_viewers": 0,
            "avg_rating": None,
            "completion_rate": 0.0,
            "popularity_trend": 0.0,
            "days_since_release": None,
        }

        logger.debug("FeatureEngineer initialized")

    def compute_user_features(
        self,
        user_id: str,
        interactions_df: pd.DataFrame,
        as_of_date: datetime
    ) -> Dict[str, Any]:
        """
        Compute user features as of a specific date.

        Computes comprehensive user features from interaction history, ensuring
        temporal correctness by only using data available up to the as_of_date.

        Args:
            user_id: Unique user identifier
            interactions_df: DataFrame with columns:
                - user_id
                - movie_id
                - timestamp
                - watch_time_seconds
                - rating (optional)
                - event_type (view, skip, etc.)
                - genres (list or comma-separated string)
            as_of_date: Compute features as of this timestamp (point-in-time)

        Returns:
            Dictionary with computed features:
                - total_watch_time: Total seconds watched (float)
                - avg_watch_time_per_movie: Average watch time per movie (float)
                - watch_count_last_7days: Views in last 7 days (int)
                - watch_count_last_30days: Views in last 30 days (int)
                - watch_count_last_90days: Views in last 90 days (int)
                - unique_genres_watched: Number of unique genres (int)
                - favorite_genre: Most watched genre (str or None)
                - avg_rating_given: Average rating user gave (float or None)
                - skip_rate: Ratio of skips to total interactions (float)
                - peak_activity_hour: Hour with most activity (int or None)
                - days_since_last_watch: Days since last view (int or None)

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = pd.DataFrame({
            ...     "user_id": ["user_123"] * 10,
            ...     "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            ...     "watch_time_seconds": [3600] * 10,
            ...     "event_type": ["view"] * 10
            ... })
            >>> features = engineer.compute_user_features(
            ...     "user_123",
            ...     df,
            ...     datetime(2024, 1, 15)
            ... )
            >>> print(features["total_watch_time"])
            36000.0
        """
        if interactions_df.empty:
            logger.debug(
                f"No interactions found for user {user_id}, returning defaults"
            )
            return self.default_user_features.copy()

        # Filter interactions for this user
        user_interactions = interactions_df[
            interactions_df["user_id"] == user_id
        ].copy()

        if user_interactions.empty:
            logger.debug(
                f"User {user_id} not found in interactions, returning defaults"
            )
            return self.default_user_features.copy()

        # CRITICAL: Filter to only use data available up to as_of_date
        # This ensures temporal correctness (no future data leakage)
        user_interactions = user_interactions[
            pd.to_datetime(user_interactions["timestamp"]) <= as_of_date
        ].copy()

        if user_interactions.empty:
            logger.debug(
                f"No interactions before {as_of_date} for user {user_id}, "
                "returning defaults"
            )
            return self.default_user_features.copy()

        # Ensure timestamp is datetime
        user_interactions["timestamp"] = pd.to_datetime(
            user_interactions["timestamp"]
        )

        # Initialize features dict
        features = {}

        # 1. Total watch time
        total_watch_time = user_interactions[
            user_interactions["event_type"] == "view"
        ]["watch_time_seconds"].sum()
        features["total_watch_time"] = float(total_watch_time)

        # 2. Average watch time per movie
        view_events = user_interactions[
            user_interactions["event_type"] == "view"
        ]
        if len(view_events) > 0:
            avg_watch_time = view_events["watch_time_seconds"].mean()
            features["avg_watch_time_per_movie"] = float(avg_watch_time)
        else:
            features["avg_watch_time_per_movie"] = 0.0

        # 3. Watch counts by time window
        time_windows = {
            "watch_count_last_7days": 7,
            "watch_count_last_30days": 30,
            "watch_count_last_90days": 90,
        }

        for feature_name, days in time_windows.items():
            cutoff_date = as_of_date - timedelta(days=days)
            recent_interactions = user_interactions[
                (user_interactions["event_type"] == "view") &
                (user_interactions["timestamp"] > cutoff_date)
            ]
            features[feature_name] = len(recent_interactions)

        # 4. Unique genres watched
        if "genres" in user_interactions.columns:
            view_events = user_interactions[
                user_interactions["event_type"] == "view"
            ]
            if len(view_events) > 0:
                # Handle genres as list or string
                all_genres = []
                for genres in view_events["genres"].dropna():
                    if isinstance(genres, str):
                        # Comma-separated or single string
                        all_genres.extend([g.strip() for g in genres.split(",")])
                    elif isinstance(genres, list):
                        all_genres.extend(genres)

                unique_genres = len(set(all_genres)) if all_genres else 0
                features["unique_genres_watched"] = unique_genres
            else:
                features["unique_genres_watched"] = 0
        else:
            features["unique_genres_watched"] = 0

        # 5. Favorite genre (most watched)
        if "genres" in user_interactions.columns:
            view_events = user_interactions[
                user_interactions["event_type"] == "view"
            ]
            if len(view_events) > 0:
                genre_counts = {}
                for genres in view_events["genres"].dropna():
                    genre_list = (
                        [g.strip() for g in genres.split(",")]
                        if isinstance(genres, str)
                        else (genres if isinstance(genres, list) else [])
                    )
                    for genre in genre_list:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

                if genre_counts:
                    favorite_genre = max(genre_counts.items(), key=lambda x: x[1])[0]
                    features["favorite_genre"] = favorite_genre
                else:
                    features["favorite_genre"] = None
            else:
                features["favorite_genre"] = None
        else:
            features["favorite_genre"] = None

        # 6. Average rating given
        if "rating" in user_interactions.columns:
            ratings = user_interactions[
                user_interactions["rating"].notna()
            ]["rating"]
            if len(ratings) > 0:
                features["avg_rating_given"] = float(ratings.mean())
            else:
                features["avg_rating_given"] = None
        else:
            features["avg_rating_given"] = None

        # 7. Skip rate
        total_interactions = len(user_interactions)
        skip_count = len(
            user_interactions[user_interactions["event_type"] == "skip"]
        )
        features["skip_rate"] = (
            float(skip_count / total_interactions) if total_interactions > 0 else 0.0
        )

        # 8. Peak activity hour
        if len(user_interactions) > 0:
            user_interactions["hour"] = user_interactions["timestamp"].dt.hour
            hour_counts = user_interactions["hour"].value_counts()
            if len(hour_counts) > 0:
                features["peak_activity_hour"] = int(hour_counts.index[0])
            else:
                features["peak_activity_hour"] = None
        else:
            features["peak_activity_hour"] = None

        # 9. Days since last watch
        view_events = user_interactions[
            user_interactions["event_type"] == "view"
        ]
        if len(view_events) > 0:
            last_watch = view_events["timestamp"].max()
            days_since = (as_of_date - last_watch.to_pydatetime()).days
            features["days_since_last_watch"] = int(days_since)
        else:
            features["days_since_last_watch"] = None

        logger.debug(
            f"Computed user features for {user_id}",
            extra={
                "user_id": user_id,
                "as_of_date": as_of_date.isoformat(),
                "total_watch_time": features["total_watch_time"],
            }
        )

        return features

    def compute_movie_features(
        self,
        movie_id: str,
        interactions_df: pd.DataFrame,
        as_of_date: datetime,
        movie_release_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Compute movie features as of a specific date.

        Computes movie features from interaction history with temporal
        correctness. Handles cold-start movies (new releases) with defaults.

        Args:
            movie_id: Unique movie identifier
            interactions_df: DataFrame with columns:
                - movie_id
                - user_id
                - timestamp
                - watch_time_seconds
                - rating (optional)
                - event_type
                - movie_duration_seconds (optional, for completion rate)
            as_of_date: Compute features as of this timestamp
            movie_release_date: Optional movie release date for computing
                              days_since_release

        Returns:
            Dictionary with computed features:
                - total_views: Total number of views (int)
                - unique_viewers: Number of unique users who viewed (int)
                - avg_rating: Average rating received (float or None)
                - completion_rate: Average completion rate (float)
                - popularity_trend: Change in views (last 7d vs previous 7d) (float)
                - days_since_release: Days since movie release (int or None)

        Example:
            >>> engineer = FeatureEngineer()
            >>> df = pd.DataFrame({
            ...     "movie_id": ["movie_456"] * 20,
            ...     "user_id": [f"user_{i}" for i in range(20)],
            ...     "timestamp": pd.date_range("2024-01-01", periods=20, freq="D"),
            ...     "watch_time_seconds": [1800] * 20,
            ...     "rating": [4.0] * 20,
            ...     "event_type": ["view"] * 20
            ... })
            >>> features = engineer.compute_movie_features(
            ...     "movie_456",
            ...     df,
            ...     datetime(2024, 1, 15)
            ... )
            >>> print(features["total_views"])
            15
        """
        if interactions_df.empty:
            logger.debug(
                f"No interactions found for movie {movie_id}, returning defaults"
            )
            return self.default_movie_features.copy()

        # Filter interactions for this movie
        movie_interactions = interactions_df[
            interactions_df["movie_id"] == movie_id
        ].copy()

        if movie_interactions.empty:
            logger.debug(
                f"Movie {movie_id} not found in interactions, returning defaults"
            )
            features = self.default_movie_features.copy()
            # Set days_since_release if provided
            if movie_release_date:
                days = (as_of_date - movie_release_date).days
                features["days_since_release"] = int(days)
            return features

        # CRITICAL: Filter to only use data available up to as_of_date
        movie_interactions = movie_interactions[
            pd.to_datetime(movie_interactions["timestamp"]) <= as_of_date
        ].copy()

        if movie_interactions.empty:
            logger.debug(
                f"No interactions before {as_of_date} for movie {movie_id}, "
                "returning defaults"
            )
            features = self.default_movie_features.copy()
            if movie_release_date:
                days = (as_of_date - movie_release_date).days
                features["days_since_release"] = int(days)
            return features

        # Ensure timestamp is datetime
        movie_interactions["timestamp"] = pd.to_datetime(
            movie_interactions["timestamp"]
        )

        # Initialize features dict
        features = {}

        # 1. Total views
        view_events = movie_interactions[
            movie_interactions["event_type"] == "view"
        ]
        features["total_views"] = len(view_events)

        # 2. Unique viewers
        if len(view_events) > 0:
            features["unique_viewers"] = int(view_events["user_id"].nunique())
        else:
            features["unique_viewers"] = 0

        # 3. Average rating
        if "rating" in movie_interactions.columns:
            ratings = view_events[view_events["rating"].notna()]["rating"]
            if len(ratings) > 0:
                features["avg_rating"] = float(ratings.mean())
            else:
                features["avg_rating"] = None
        else:
            features["avg_rating"] = None

        # 4. Completion rate (watch_time / movie_duration)
        if "movie_duration_seconds" in movie_interactions.columns and len(view_events) > 0:
            # Get movie duration (should be same for all rows)
            duration = movie_interactions["movie_duration_seconds"].iloc[0]
            if duration and duration > 0:
                completion_rates = (
                    view_events["watch_time_seconds"] / duration
                ).clip(upper=1.0)  # Cap at 1.0
                features["completion_rate"] = float(completion_rates.mean())
            else:
                features["completion_rate"] = 0.0
        else:
            features["completion_rate"] = 0.0

        # 5. Popularity trend (last 7 days vs previous 7 days)
        if len(view_events) >= 1:
            last_7d_cutoff = as_of_date - timedelta(days=7)
            prev_7d_cutoff = as_of_date - timedelta(days=14)

            last_7d_views = len(
                view_events[view_events["timestamp"] > last_7d_cutoff]
            )
            prev_7d_views = len(
                view_events[
                    (view_events["timestamp"] > prev_7d_cutoff) &
                    (view_events["timestamp"] <= last_7d_cutoff)
                ]
            )

            # Calculate trend (positive = increasing, negative = decreasing)
            if prev_7d_views > 0:
                trend = (last_7d_views - prev_7d_views) / prev_7d_views
            else:
                # No previous views - if current > 0, trend is 1.0 (new popularity)
                trend = 1.0 if last_7d_views > 0 else 0.0

            features["popularity_trend"] = float(trend)
        else:
            features["popularity_trend"] = 0.0

        # 6. Days since release
        if movie_release_date:
            days = (as_of_date - movie_release_date).days
            features["days_since_release"] = int(days)
        else:
            features["days_since_release"] = None

        logger.debug(
            f"Computed movie features for {movie_id}",
            extra={
                "movie_id": movie_id,
                "as_of_date": as_of_date.isoformat(),
                "total_views": features["total_views"],
            }
        )

        return features

    def compute_interaction_features(
        self,
        user_id: str,
        movie_id: str,
        context: Dict[str, Any],
        feature_store: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compute interaction-specific features combining user, movie, and context.

        Combines user features, movie features, and contextual information to
        create interaction-level features for model training/inference.

        Args:
            user_id: Unique user identifier
            movie_id: Unique movie identifier
            context: Context dictionary with keys:
                - timestamp: Interaction timestamp
                - hour: Hour of day (0-23)
                - day_of_week: Day of week (0-6)
                - genres: List of movie genres
            feature_store: Dictionary mapping entity_id -> features dict
                          Should contain both user and movie features

        Returns:
            Dictionary with interaction features:
                - user_movie_genre_match: User's preference for movie genres (float)
                - time_since_user_last_active: Hours since last activity (float or None)
                - movie_popularity_percentile: Movie popularity percentile (float)
                - user_activity_level: Activity level (str: "high", "medium", "low")

        Example:
            >>> engineer = FeatureEngineer()
            >>> feature_store = {
            ...     "user_123": {"favorite_genre": "Action", "watch_count_last_30days": 20},
            ...     "movie_456": {"total_views": 1000, "avg_rating": 4.5}
            ... }
            >>> context = {
            ...     "timestamp": datetime.now(),
            ...     "hour": 20,
            ...     "genres": ["Action", "Sci-Fi"]
            ... }
            >>> features = engineer.compute_interaction_features(
            ...     "user_123",
            ...     "movie_456",
            ...     context,
            ...     feature_store
            ... )
            >>> print(features["user_movie_genre_match"])
            1.0
        """
        features = {}

        # Get user and movie features from feature store
        user_features = feature_store.get(user_id, {})
        movie_features = feature_store.get(movie_id, {})

        # 1. User-movie genre match
        # Check if user's favorite genre matches any of the movie's genres
        user_favorite_genre = user_features.get("favorite_genre")
        movie_genres = context.get("genres", [])

        if user_favorite_genre and movie_genres:
            # Convert movie_genres to list if string
            if isinstance(movie_genres, str):
                movie_genres = [g.strip() for g in movie_genres.split(",")]

            # Perfect match = 1.0, no match = 0.0
            if user_favorite_genre in movie_genres:
                features["user_movie_genre_match"] = 1.0
            else:
                # Check for partial match (e.g., user likes "Action", movie has "Action-Adventure")
                match_score = 0.0
                for genre in movie_genres:
                    if user_favorite_genre.lower() in genre.lower() or \
                       genre.lower() in user_favorite_genre.lower():
                        match_score = 0.5  # Partial match
                        break
                features["user_movie_genre_match"] = match_score
        else:
            features["user_movie_genre_match"] = 0.0

        # 2. Time since user last active
        if "days_since_last_watch" in user_features:
            days_since = user_features["days_since_last_watch"]
            if days_since is not None:
                features["time_since_user_last_active"] = float(days_since * 24)
            else:
                features["time_since_user_last_active"] = None
        else:
            features["time_since_user_last_active"] = None

        # 3. Movie popularity percentile
        # This would typically require comparing against all movies
        # For now, compute a simple percentile based on total_views
        # In production, this would use a precomputed percentile from all movies
        total_views = movie_features.get("total_views", 0)

        # Simple bucketing (in production, use actual percentiles)
        if total_views == 0:
            features["movie_popularity_percentile"] = 0.0
        elif total_views < 10:
            features["movie_popularity_percentile"] = 0.25
        elif total_views < 100:
            features["movie_popularity_percentile"] = 0.5
        elif total_views < 1000:
            features["movie_popularity_percentile"] = 0.75
        else:
            features["movie_popularity_percentile"] = 0.95

        # 4. User activity level
        watch_count_30d = user_features.get("watch_count_last_30days", 0)

        if watch_count_30d >= 20:
            features["user_activity_level"] = "high"
        elif watch_count_30d >= 5:
            features["user_activity_level"] = "medium"
        else:
            features["user_activity_level"] = "low"

        logger.debug(
            f"Computed interaction features",
            extra={
                "user_id": user_id,
                "movie_id": movie_id,
                "genre_match": features["user_movie_genre_match"],
                "activity_level": features["user_activity_level"],
            }
        )

        return features
