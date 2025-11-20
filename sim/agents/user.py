"""Synthetic user agent with realistic preferences and behaviors."""

import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Age groups for user segmentation
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]

# Common regions
REGIONS = ["US-West", "US-East", "EU", "APAC", "US-Central", "LATAM"]


@dataclass
class UserAgent:
    """
    Synthetic user agent with realistic preferences and behaviors.

    Represents a synthetic user in the AgentBricks simulation with age group,
    region, preference vector, and activity patterns. Used to generate realistic
    interaction behaviors for training recommendation systems.

    Attributes:
        user_id: Unique identifier for the user
        age_group: User's age group (one of: "18-24", "25-34", "35-44", "45-54", "55+")
        region: User's geographic region (e.g., "US-West", "EU")
        preference_vector: 50-dimensional preference embedding vector
        activity_pattern: Dictionary with peak hours and session length info

    Example:
        >>> # Create user with random preferences
        >>> user = UserAgent(user_id="user_123")
        >>> print(user.age_group)
        '25-34'
        >>>
        >>> # Create user with specific attributes
        >>> user = UserAgent(
        ...     user_id="user_456",
        ...     age_group="18-24",
        ...     region="US-West"
        ... )
        >>>
        >>> # Simulate watch behavior
        >>> movie_embedding = np.random.rand(50)
        >>> context = {"time_of_day": "evening", "day_of_week": "Friday"}
        >>> behavior = user.simulate_watch_behavior(movie_embedding, context)
        >>> print(behavior["will_watch"])
        True
        >>> print(behavior["watch_time_seconds"])
        3600
    """

    user_id: str
    age_group: Optional[str] = None
    region: Optional[str] = None
    preference_vector: npt.NDArray[np.float64] = field(init=False)
    activity_pattern: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Initialize user attributes after dataclass initialization.

        Generates random age_group, region, preference_vector, and activity_pattern
        if not explicitly provided.

        Raises:
            ValueError: If provided age_group or region is invalid
        """
        # Validate and set age_group
        if self.age_group is None:
            self.age_group = np.random.choice(AGE_GROUPS)
        elif self.age_group not in AGE_GROUPS:
            raise ValueError(
                f"Invalid age_group: {self.age_group}. "
                f"Must be one of: {AGE_GROUPS}"
            )

        # Validate and set region
        if self.region is None:
            self.region = np.random.choice(REGIONS)
        elif self.region not in REGIONS:
            logger.warning(
                f"Unknown region '{self.region}' provided. "
                f"Common regions: {REGIONS}. Proceeding with provided region."
            )

        # Generate preference vector
        self.preference_vector = self.generate_preference_vector()

        # Generate activity pattern if not provided
        if not self.activity_pattern:
            self.activity_pattern = self._generate_activity_pattern()

        logger.debug(
            f"Created user agent",
            extra={
                "user_id": self.user_id,
                "age_group": self.age_group,
                "region": self.region,
            }
        )

    def generate_preference_vector(
        self,
        n_dims: int = 50,
        seed: Optional[int] = None
    ) -> npt.NDArray[np.float64]:
        """
        Generate a random preference vector using Gaussian distribution.

        Creates a normalized preference embedding vector that represents the
        user's preferences across different dimensions (genres, themes, etc.).
        Uses a seeded random number generator if seed is provided for reproducibility.

        Args:
            n_dims: Dimension of the preference vector (default: 50)
            seed: Optional random seed for reproducibility

        Returns:
            Normalized preference vector of shape (n_dims,)

        Example:
            >>> user = UserAgent(user_id="user_123")
            >>> pref_vector = user.generate_preference_vector(n_dims=50)
            >>> print(pref_vector.shape)
            (50,)
            >>> print(np.linalg.norm(pref_vector))
            1.0
        """
        if n_dims <= 0:
            raise ValueError(f"n_dims must be positive, got {n_dims}")

        # Use seeded random if provided (for reproducibility)
        rng = np.random.default_rng(seed)

        # Generate vector from standard normal distribution
        vector = rng.normal(loc=0.0, scale=1.0, size=n_dims)

        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        else:
            # Fallback: use uniform distribution if norm is zero (rare)
            logger.warning(
                f"Preference vector norm was zero for user {self.user_id}, "
                "using uniform distribution fallback"
            )
            vector = rng.uniform(low=-1.0, high=1.0, size=n_dims)
            vector = vector / np.linalg.norm(vector)

        logger.debug(
            f"Generated preference vector",
            extra={
                "user_id": self.user_id,
                "n_dims": n_dims,
                "norm": float(np.linalg.norm(vector)),
            }
        )

        return vector.astype(np.float64)

    def simulate_watch_behavior(
        self,
        movie_embedding: npt.NDArray[np.float64],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Simulate user's watch behavior for a given movie.

        Determines whether the user will watch a movie, how long they'll watch,
        whether they'll rate it, and what rating they'll give. Behavior is
        influenced by preference similarity, context (time, day), and user
        characteristics (age group, activity pattern).

        Args:
            movie_embedding: Movie's embedding vector (should match preference_vector
                           dimension)
            context: Optional context dictionary with keys:
                - time_of_day: "morning", "afternoon", "evening", "night"
                - day_of_week: "Monday", "Tuesday", etc.
                - device: "mobile", "tablet", "desktop"
                - is_weekend: bool

        Returns:
            Dictionary with keys:
                - will_watch: bool - Whether user will watch the movie
                - watch_time_seconds: int - Watch duration in seconds
                - will_rate: bool - Whether user will rate the movie
                - rating: Optional[int] - Rating (1-5) if will_rate is True

        Raises:
            ValueError: If movie_embedding dimension doesn't match preference_vector
            ValueError: If movie_embedding is empty

        Example:
            >>> user = UserAgent(user_id="user_123")
            >>> movie_emb = np.random.rand(50)
            >>> context = {"time_of_day": "evening", "is_weekend": True}
            >>> behavior = user.simulate_watch_behavior(movie_emb, context)
            >>> print(behavior["will_watch"])
            True
            >>> print(f"Watch time: {behavior['watch_time_seconds']} seconds")
            Watch time: 3600 seconds
        """
        if context is None:
            context = {}

        # Validate movie embedding
        if movie_embedding.size == 0:
            raise ValueError("movie_embedding cannot be empty")

        if movie_embedding.shape[0] != self.preference_vector.shape[0]:
            raise ValueError(
                f"movie_embedding dimension {movie_embedding.shape[0]} "
                f"does not match preference_vector dimension "
                f"{self.preference_vector.shape[0]}"
            )

        # Compute preference similarity (cosine similarity)
        similarity = np.dot(self.preference_vector, movie_embedding)

        # Base watch probability from similarity
        base_probability = (similarity + 1.0) / 2.0  # Normalize to [0, 1]

        # Context modifiers
        context_modifier = self._compute_context_modifier(context)
        watch_probability = np.clip(
            base_probability * context_modifier,
            a_min=0.0,
            a_max=1.0
        )

        # Decide if user will watch
        will_watch = np.random.random() < watch_probability

        # If watching, determine watch time and rating
        watch_time_seconds = 0
        will_rate = False
        rating: Optional[int] = None

        if will_watch:
            # Watch time based on similarity and age group
            base_watch_time = self._estimate_watch_time(similarity)
            watch_time_seconds = int(base_watch_time)

            # Rating probability increases with watch completion
            completion_rate = min(watch_time_seconds / 3600.0, 1.0)  # Assume 1hr full movie
            rating_probability = 0.3 + (completion_rate * 0.5)  # 30-80% chance

            will_rate = np.random.random() < rating_probability

            if will_rate:
                # Rating correlates with similarity
                base_rating = 2.5 + (similarity * 2.5)  # Map [-1, 1] to [0, 5]
                rating_float = np.clip(
                    base_rating + np.random.normal(0, 0.5),
                    a_min=1.0,
                    a_max=5.0
                )
                rating = int(round(rating_float))

        result = {
            "will_watch": will_watch,
            "watch_time_seconds": watch_time_seconds,
            "will_rate": will_rate,
            "rating": rating,
        }

        logger.debug(
            f"Simulated watch behavior",
            extra={
                "user_id": self.user_id,
                "similarity": float(similarity),
                "will_watch": will_watch,
                "watch_time_seconds": watch_time_seconds,
            }
        )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert user agent to dictionary for serialization.

        Serializes the user agent to a dictionary format that can be easily
        stored in databases or JSON files. Preference vector is converted to
        a list.

        Returns:
            Dictionary representation of the user agent

        Example:
            >>> user = UserAgent(user_id="user_123")
            >>> user_dict = user.to_dict()
            >>> print(user_dict["user_id"])
            'user_123'
            >>> print(len(user_dict["preference_vector"]))
            50
        """
        return {
            "user_id": self.user_id,
            "age_group": self.age_group,
            "region": self.region,
            "preference_vector": self.preference_vector.tolist(),
            "activity_pattern": self.activity_pattern,
        }

    def _generate_activity_pattern(self) -> Dict[str, Any]:
        """
        Generate realistic activity pattern based on age group.

        Creates activity patterns with peak hours and average session length
        that vary by age group.

        Returns:
            Dictionary with keys:
                - peak_hours: List of peak activity hours (0-23)
                - avg_session_length_seconds: Average session length in seconds
                - weekday_activity: Relative activity on weekdays (0-1)
                - weekend_activity: Relative activity on weekends (0-1)
        """
        # Age-based patterns
        age_patterns = {
            "18-24": {
                "peak_hours": [18, 19, 20, 21, 22],
                "avg_session_length": 3600,  # 1 hour
                "weekday_activity": 0.6,
                "weekend_activity": 0.9,
            },
            "25-34": {
                "peak_hours": [19, 20, 21, 22],
                "avg_session_length": 4200,  # 1.17 hours
                "weekday_activity": 0.7,
                "weekend_activity": 0.85,
            },
            "35-44": {
                "peak_hours": [20, 21, 22],
                "avg_session_length": 3300,  # 55 minutes
                "weekday_activity": 0.65,
                "weekend_activity": 0.8,
            },
            "45-54": {
                "peak_hours": [19, 20, 21],
                "avg_session_length": 2700,  # 45 minutes
                "weekday_activity": 0.6,
                "weekend_activity": 0.75,
            },
            "55+": {
                "peak_hours": [14, 15, 20, 21],
                "avg_session_length": 2400,  # 40 minutes
                "weekday_activity": 0.7,
                "weekend_activity": 0.7,
            },
        }

        pattern = age_patterns.get(self.age_group, age_patterns["25-34"])

        return {
            "peak_hours": pattern["peak_hours"],
            "avg_session_length_seconds": pattern["avg_session_length"],
            "weekday_activity": pattern["weekday_activity"],
            "weekend_activity": pattern["weekend_activity"],
        }

    def _compute_context_modifier(self, context: Dict[str, Any]) -> float:
        """
        Compute context modifier for watch probability.

        Adjusts watch probability based on time of day, day of week, and
        other contextual factors.

        Args:
            context: Context dictionary with optional keys:
                - time_of_day: "morning", "afternoon", "evening", "night"
                - day_of_week: Day name (e.g., "Monday")
                - is_weekend: Boolean (optional if day_of_week provided)

        Returns:
            Multiplier for base watch probability (typically 0.5 to 2.0)
        """
        modifier = 1.0

        # Time of day modifier
        time_of_day = context.get("time_of_day", "").lower()
        time_modifiers = {
            "morning": 0.6,
            "afternoon": 0.8,
            "evening": 1.2,
            "night": 1.0,
        }
        modifier *= time_modifiers.get(time_of_day, 1.0)

        # Determine if weekend from day_of_week or is_weekend
        is_weekend = False
        if "day_of_week" in context:
            day = context["day_of_week"]
            is_weekend = day.lower() in ["saturday", "sunday"]
        elif "is_weekend" in context:
            is_weekend = context["is_weekend"]

        # Apply weekend modifier if weekend
        if is_weekend:
            modifier *= 1.15

        # Age group activity pattern modifier
        activity = (
            self.activity_pattern["weekend_activity"]
            if is_weekend
            else self.activity_pattern["weekday_activity"]
        )
        modifier *= activity

        return modifier

    def _estimate_watch_time(self, similarity: float) -> float:
        """
        Estimate watch time based on preference similarity.

        Higher similarity leads to longer watch times. Watch time also varies
        by age group (older users may watch shorter, younger may binge).

        Args:
            similarity: Cosine similarity between preference and movie embedding

        Returns:
            Estimated watch time in seconds
        """
        # Base watch time from similarity (0-3600 seconds for full movie)
        base_time = 1800 + (similarity + 1.0) * 900  # 30 min to 60 min base

        # Age group adjustment
        age_adjustments = {
            "18-24": 1.15,  # Watch longer
            "25-34": 1.1,
            "35-44": 1.0,
            "45-54": 0.9,
            "55+": 0.85,  # Watch shorter
        }

        adjustment = age_adjustments.get(self.age_group, 1.0)
        watch_time = base_time * adjustment

        # Add some randomness (±20%)
        noise = np.random.normal(1.0, 0.2)
        watch_time = watch_time * noise

        # Clip to reasonable bounds (5 minutes to 2 hours)
        watch_time = np.clip(watch_time, a_min=300.0, a_max=7200.0)

        return watch_time
