"""Interaction simulation rules for user-movie interactions."""

import logging
from typing import Optional, Dict, Any
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Type imports (avoid circular imports by using TYPE_CHECKING)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sim.agents.user import UserAgent
    from sim.items.movie import Movie

# Weekend days (0=Monday, 6=Sunday)
WEEKEND_DAYS = [5, 6]  # Saturday, Sunday


def calculate_watch_probability(
    user: "UserAgent",
    movie: "Movie",
    context: Optional[Dict[str, Any]] = None
) -> float:
    """
    Calculate probability that a user will watch a movie.

    Computes the watch probability based on:
    - Cosine similarity between user preferences and movie embedding
    - Temporal effects (time of day, day of week)
    - Random noise for realism

    Args:
        user: UserAgent instance with preference vector
        movie: Movie instance with embedding vector
        context: Optional context dictionary with keys:
            - hour: int (0-23) - Hour of day
            - day_of_week: int (0-6, Monday=0) or str - Day of week
            - time_of_day: str - "morning", "afternoon", "evening", "night"
            - is_weekend: bool - Whether it's a weekend

    Returns:
        Watch probability in range [0, 1]

    Raises:
        ValueError: If user or movie is None
        ValueError: If context contains invalid values

    Example:
        >>> from sim.agents.user import UserAgent
        >>> from sim.items.movie import Movie
        >>> import numpy as np
        >>>
        >>> # Create user and movie
        >>> user = UserAgent(user_id="u1")
        >>> movie = Movie(
        ...     movie_id="m1",
        ...     title="Test Movie",
        ...     genres=["Action"],
        ...     release_year=2020,
        ...     duration_minutes=120
        ... )
        >>>
        >>> # Calculate probability
        >>> context = {"hour": 20, "day_of_week": 5, "is_weekend": True}
        >>> prob = calculate_watch_probability(user, movie, context)
        >>> assert 0.0 <= prob <= 1.0
        >>> print(f"Watch probability: {prob:.3f}")
        Watch probability: 0.456
    """
    # Input validation
    if user is None:
        raise ValueError("user cannot be None")
    if movie is None:
        raise ValueError("movie cannot be None")

    if context is None:
        context = {}

    # Validate context if provided
    if "hour" in context:
        hour = context["hour"]
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            raise ValueError(f"hour must be int in [0, 23], got {hour}")

    if "day_of_week" in context:
        day = context["day_of_week"]
        if isinstance(day, str):
            # Convert string day to int
            day_map = {
                "monday": 0, "tuesday": 1, "wednesday": 2,
                "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
            }
            day = day_map.get(day.lower(), None)
        if day is None or not isinstance(day, int) or not (0 <= day <= 6):
            raise ValueError(
                f"day_of_week must be int in [0, 6] or valid day name, "
                f"got {context['day_of_week']}"
            )

    # Calculate base probability from similarity
    similarity = movie.calculate_similarity(user.preference_vector)

    # Map similarity [-1, 1] to base probability [0.1, 0.9]
    # Higher similarity = higher probability, but never 0 or 1
    base_probability = 0.5 + (similarity * 0.4)
    base_probability = np.clip(base_probability, a_min=0.1, a_max=0.9)

    # Apply temporal effects
    hour = context.get("hour", 12)  # Default to noon
    day_of_week = context.get("day_of_week", 0)  # Default to Monday

    # Convert day_of_week if it's a string
    if isinstance(day_of_week, str):
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2,
            "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
        }
        day_of_week = day_map.get(day_of_week.lower(), 0)

    adjusted_probability = apply_temporal_effects(
        base_probability,
        hour,
        day_of_week
    )

    # Apply user's context modifier if available
    if context:
        try:
            user_context_modifier = user._compute_context_modifier(context)
            adjusted_probability = adjusted_probability * user_context_modifier
            adjusted_probability = np.clip(
                adjusted_probability,
                a_min=0.0,
                a_max=1.0
            )
        except Exception as e:
            logger.warning(
                f"Failed to apply user context modifier: {e}",
                extra={"user_id": user.user_id, "movie_id": movie.movie_id}
            )

    # Add random noise (σ=0.1) for realism
    noise = np.random.normal(loc=0.0, scale=0.1)
    final_probability = adjusted_probability + noise

    # Clip to valid probability range [0, 1]
    final_probability = np.clip(final_probability, a_min=0.0, a_max=1.0)

    logger.debug(
        f"Calculated watch probability",
        extra={
            "user_id": user.user_id,
            "movie_id": movie.movie_id,
            "similarity": float(similarity),
            "base_prob": float(base_probability),
            "final_prob": float(final_probability),
        }
    )

    return float(final_probability)


def simulate_watch_time(
    user: "UserAgent",
    movie: "Movie",
    did_watch: bool
) -> int:
    """
    Simulate watch time for a user watching a movie.

    If user didn't watch, returns 0. Otherwise, samples from a lognormal
    distribution based on movie duration and user's typical session length.
    Watch time is capped at the movie's duration.

    Args:
        user: UserAgent instance with activity pattern
        movie: Movie instance with duration
        did_watch: Whether the user actually watched the movie

    Returns:
        Watch time in seconds (0 if didn't watch)

    Raises:
        ValueError: If user or movie is None
        ValueError: If movie duration is invalid

    Example:
        >>> from sim.agents.user import UserAgent
        >>> from sim.items.movie import Movie
        >>>
        >>> # Create user and movie
        >>> user = UserAgent(user_id="u1", age_group="25-34")
        >>> movie = Movie(
        ...     movie_id="m1",
        ...     title="Test Movie",
        ...     genres=["Action"],
        ...     release_year=2020,
        ...     duration_minutes=120  # 2 hours = 7200 seconds
        ... )
        >>>
        >>> # Simulate watch time if watched
        >>> watch_time = simulate_watch_time(user, movie, did_watch=True)
        >>> assert 0 <= watch_time <= 7200
        >>> print(f"Watch time: {watch_time} seconds ({watch_time/60:.1f} minutes)")
        Watch time: 3600 seconds (60.0 minutes)
        >>>
        >>> # If didn't watch, returns 0
        >>> watch_time = simulate_watch_time(user, movie, did_watch=False)
        >>> assert watch_time == 0
    """
    # Input validation
    if user is None:
        raise ValueError("user cannot be None")
    if movie is None:
        raise ValueError("movie cannot be None")

    if not did_watch:
        return 0

    # Get movie duration in seconds
    movie_duration_seconds = movie.duration_minutes * 60

    if movie_duration_seconds <= 0:
        raise ValueError(
            f"movie duration must be positive, got {movie.duration_minutes} minutes"
        )

    # Get user's typical session length
    user_avg_session = user.activity_pattern.get(
        "avg_session_length_seconds",
        3600  # Default 1 hour
    )

    # Estimate mean watch time based on:
    # - User's typical session length
    # - Movie duration (user won't watch longer than movie)
    # - Take the minimum as a reasonable expectation
    expected_watch_time = min(user_avg_session, movie_duration_seconds * 0.8)

    # Use lognormal distribution for realistic watch times
    # Parameters: mean = expected_watch_time, but we need to convert
    # For lognormal: if X ~ LogNormal(μ, σ), then E[X] = exp(μ + σ²/2)
    # We set σ = 0.5 for reasonable variance
    sigma = 0.5
    mu = np.log(expected_watch_time) - (sigma ** 2) / 2

    # Sample watch time from lognormal distribution
    watch_time_seconds = np.random.lognormal(mean=mu, sigma=sigma)

    # Cap at movie duration (user can't watch longer than movie length)
    watch_time_seconds = min(watch_time_seconds, movie_duration_seconds)

    # Ensure minimum of 30 seconds (brief interaction)
    watch_time_seconds = max(watch_time_seconds, 30)

    watch_time_seconds = int(watch_time_seconds)

    logger.debug(
        f"Simulated watch time",
        extra={
            "user_id": user.user_id,
            "movie_id": movie.movie_id,
            "watch_time_seconds": watch_time_seconds,
            "movie_duration_seconds": movie_duration_seconds,
            "completion_rate": watch_time_seconds / movie_duration_seconds,
        }
    )

    return watch_time_seconds


def simulate_rating(
    user: "UserAgent",
    movie: "Movie",
    watch_time: int
) -> Optional[int]:
    """
    Simulate user rating for a movie based on watch behavior.

    Users only rate movies if they watched at least 30% of the movie.
    Rating is based on preference similarity with added noise.
    Higher similarity and longer watch time lead to higher ratings.

    Args:
        user: UserAgent instance with preference vector
        movie: Movie instance with embedding
        watch_time: Watch time in seconds

    Returns:
        Rating as integer 1-5, or None if user didn't watch enough

    Raises:
        ValueError: If user or movie is None
        ValueError: If watch_time is negative

    Example:
        >>> from sim.agents.user import UserAgent
        >>> from sim.items.movie import Movie
        >>>
        >>> # Create user and movie
        >>> user = UserAgent(user_id="u1")
        >>> movie = Movie(
        ...     movie_id="m1",
        ...     title="Test Movie",
        ...     genres=["Action"],
        ...     release_year=2020,
        ...     duration_minutes=120  # 7200 seconds
        ... )
        >>>
        >>> # Rate with sufficient watch time (50% of movie)
        >>> rating = simulate_rating(user, movie, watch_time=3600)
        >>> if rating is not None:
        ...     assert 1 <= rating <= 5
        ...     print(f"Rating: {rating}")
        Rating: 4
        >>>
        >>> # With insufficient watch time (< 30%), returns None
        >>> rating = simulate_rating(user, movie, watch_time=1000)
        >>> assert rating is None
    """
    # Input validation
    if user is None:
        raise ValueError("user cannot be None")
    if movie is None:
        raise ValueError("movie cannot be None")
    if watch_time < 0:
        raise ValueError(f"watch_time cannot be negative, got {watch_time}")

    # Movie duration in seconds
    movie_duration_seconds = movie.duration_minutes * 60

    # Check if user watched enough to rate (at least 30% of movie)
    minimum_watch_for_rating = 0.3 * movie_duration_seconds

    if watch_time < minimum_watch_for_rating:
        logger.debug(
            f"Watch time insufficient for rating",
            extra={
                "user_id": user.user_id,
                "movie_id": movie.movie_id,
                "watch_time": watch_time,
                "minimum_required": minimum_watch_for_rating,
            }
        )
        return None

    # Calculate base rating from similarity
    similarity = movie.calculate_similarity(user.preference_vector)

    # Map similarity [-1, 1] to base rating [1.5, 4.5]
    # This gives room for noise to push to 1-5 range
    base_rating = 3.0 + (similarity * 1.5)

    # Adjust based on watch time (longer watch = slightly higher rating)
    completion_rate = min(watch_time / movie_duration_seconds, 1.0)
    watch_time_bonus = (completion_rate - 0.3) * 0.7  # Max +0.5 for full watch
    base_rating += watch_time_bonus

    # Add noise (σ=0.8 for reasonable variance)
    rating_float = base_rating + np.random.normal(loc=0.0, scale=0.8)

    # Clip to valid rating range [1, 5] and round
    rating = int(round(np.clip(rating_float, a_min=1.0, a_max=5.0)))

    logger.debug(
        f"Simulated rating",
        extra={
            "user_id": user.user_id,
            "movie_id": movie.movie_id,
            "similarity": float(similarity),
            "completion_rate": completion_rate,
            "rating": rating,
        }
    )

    return rating


def apply_temporal_effects(
    base_probability: float,
    hour: int,
    day_of_week: int
) -> float:
    """
    Apply temporal effects to base watch probability.

    Adjusts probability based on:
    - Time of day (higher in evening hours 18-23)
    - Day of week (higher on weekends)

    Args:
        base_probability: Base probability in [0, 1]
        hour: Hour of day (0-23)
        day_of_week: Day of week (0-6, Monday=0, Sunday=6)

    Returns:
        Adjusted probability in [0, 1]

    Raises:
        ValueError: If base_probability is out of [0, 1] range
        ValueError: If hour is out of [0, 23] range
        ValueError: If day_of_week is out of [0, 6] range

    Example:
        >>> # Evening on weekend - higher probability
        >>> prob = apply_temporal_effects(0.5, hour=20, day_of_week=6)
        >>> assert prob > 0.5
        >>> print(f"Evening weekend probability: {prob:.3f}")
        Evening weekend probability: 0.680
        >>>
        >>> # Morning on weekday - lower probability
        >>> prob = apply_temporal_effects(0.5, hour=8, day_of_week=1)
        >>> assert prob < 0.5
        >>> print(f"Morning weekday probability: {prob:.3f}")
        Morning weekday probability: 0.300
    """
    # Input validation
    if not (0.0 <= base_probability <= 1.0):
        raise ValueError(
            f"base_probability must be in [0, 1], got {base_probability}"
        )

    if not (0 <= hour <= 23):
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    if not (0 <= day_of_week <= 6):
        raise ValueError(f"day_of_week must be in [0, 6], got {day_of_week}")

    adjusted_probability = base_probability

    # Time of day multiplier
    # Higher probability in evening (18-23)
    # Lower in morning (6-12)
    # Medium in afternoon (12-18)
    # Low at night (0-6)
    if 18 <= hour <= 23:
        # Evening: peak viewing time
        time_multiplier = 1.4
    elif 12 <= hour < 18:
        # Afternoon: moderate viewing
        time_multiplier = 1.1
    elif 6 <= hour < 12:
        # Morning: low viewing
        time_multiplier = 0.6
    else:
        # Late night/early morning: very low viewing
        time_multiplier = 0.4

    adjusted_probability *= time_multiplier

    # Weekend multiplier
    # Higher probability on weekends (Saturday=5, Sunday=6)
    is_weekend = day_of_week in WEEKEND_DAYS
    if is_weekend:
        weekend_multiplier = 1.25
        adjusted_probability *= weekend_multiplier

    # Clip to valid probability range
    adjusted_probability = np.clip(adjusted_probability, a_min=0.0, a_max=1.0)

    logger.debug(
        f"Applied temporal effects",
        extra={
            "base_prob": base_probability,
            "hour": hour,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "adjusted_prob": adjusted_probability,
        }
    )

    return float(adjusted_probability)
