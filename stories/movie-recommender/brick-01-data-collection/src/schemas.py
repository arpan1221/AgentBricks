"""Event schemas for movie recommendation data collection."""

import logging
from datetime import datetime, timezone
from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

logger = logging.getLogger(__name__)

# Valid device types
DEVICE_TYPES = Literal["mobile", "tablet", "desktop", "tv"]

# Maximum watch time in seconds (24 hours)
MAX_WATCH_TIME_SECONDS = 24 * 60 * 60


class ViewEvent(BaseModel):
    """
    User view event schema.

    Represents a user watching a movie. Captures view duration and
    contextual information for recommendation system training.

    Attributes:
        user_id: Unique user identifier (UUID or alphanumeric)
        movie_id: Unique movie identifier
        timestamp: Event timestamp (UTC)
        watch_time_seconds: Duration watched in seconds (0 to 86400)
        session_id: Optional session identifier for grouping events
        device: Optional device type used for viewing

    Validations:
        - watch_time_seconds must be non-negative
        - watch_time_seconds cannot exceed 24 hours (86400 seconds)
        - timestamp cannot be in the future
        - device must be one of: "mobile", "tablet", "desktop", "tv"

    Example (Valid):
        >>> from datetime import datetime, timezone
        >>> event = ViewEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     timestamp=datetime.now(timezone.utc),
        ...     watch_time_seconds=3600,
        ...     session_id="session_789",
        ...     device="desktop"
        ... )
        >>> print(event.watch_time_seconds)
        3600

    Example (Invalid - watch time too long):
        >>> event = ViewEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     timestamp=datetime.now(timezone.utc),
        ...     watch_time_seconds=100000  # > 24 hours
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: watch_time_seconds cannot exceed 86400 seconds (24 hours)

    Example (Invalid - future timestamp):
        >>> from datetime import timedelta
        >>> future_time = datetime.now(timezone.utc) + timedelta(days=1)
        >>> event = ViewEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     timestamp=future_time,
        ...     watch_time_seconds=3600
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: timestamp cannot be in the future
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier (UUID or alphanumeric)",
        min_length=1,
        max_length=255
    )
    movie_id: str = Field(
        ...,
        description="Unique movie identifier",
        min_length=1,
        max_length=255
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp (UTC)"
    )
    watch_time_seconds: int = Field(
        ...,
        description="Duration watched in seconds",
        ge=0
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session identifier for grouping events",
        max_length=255
    )
    device: Optional[DEVICE_TYPES] = Field(
        default=None,
        description="Device type used for viewing"
    )

    @field_validator("watch_time_seconds")
    @classmethod
    def validate_watch_time(cls, v: int) -> int:
        """
        Validate watch time doesn't exceed 24 hours.

        Args:
            v: Watch time in seconds

        Returns:
            Validated watch time

        Raises:
            ValueError: If watch time exceeds 24 hours
        """
        if v > MAX_WATCH_TIME_SECONDS:
            raise ValueError(
                f"watch_time_seconds cannot exceed {MAX_WATCH_TIME_SECONDS} "
                f"seconds (24 hours), got {v}"
            )
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        """
        Validate timestamp is not in the future.

        Args:
            v: Event timestamp

        Returns:
            Validated timestamp

        Raises:
            ValueError: If timestamp is in the future
        """
        now = datetime.now(timezone.utc)

        # Handle timezone-aware and naive datetimes
        if v.tzinfo is None:
            # Assume UTC if timezone-naive
            v = v.replace(tzinfo=timezone.utc)
            logger.warning(
                "Received timezone-naive timestamp, assuming UTC"
            )

        if v > now:
            raise ValueError(
                f"timestamp cannot be in the future, got {v}, "
                f"current time: {now}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "timestamp": "2024-01-15T20:30:00Z",
                "watch_time_seconds": 3600,
                "session_id": "session_789",
                "device": "desktop"
            }
        }
    }


class RatingEvent(BaseModel):
    """
    User rating event schema.

    Represents a user rating a movie. Captures explicit feedback
    for training recommendation models.

    Attributes:
        user_id: Unique user identifier
        movie_id: Unique movie identifier
        rating: Rating value (1-5, where 5 is highest)
        timestamp: Event timestamp (UTC)

    Validations:
        - rating must be between 1 and 5 (inclusive)
        - timestamp cannot be in the future

    Example (Valid):
        >>> from datetime import datetime, timezone
        >>> event = RatingEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     rating=5,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> print(event.rating)
        5

    Example (Invalid - rating out of range):
        >>> event = RatingEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     rating=6,  # Invalid: must be 1-5
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: rating must be between 1 and 5, got 6

    Example (Invalid - rating too low):
        >>> event = RatingEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     rating=0,  # Invalid: must be 1-5
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: rating must be between 1 and 5, got 0
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier",
        min_length=1,
        max_length=255
    )
    movie_id: str = Field(
        ...,
        description="Unique movie identifier",
        min_length=1,
        max_length=255
    )
    rating: int = Field(
        ...,
        description="Rating value (1-5, where 5 is highest)",
        ge=1,
        le=5
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp (UTC)"
    )

    @field_validator("rating")
    @classmethod
    def validate_rating_range(cls, v: int) -> int:
        """
        Validate rating is in valid range [1, 5].

        Args:
            v: Rating value

        Returns:
            Validated rating

        Raises:
            ValueError: If rating is outside [1, 5] range
        """
        if not (1 <= v <= 5):
            raise ValueError(f"rating must be between 1 and 5, got {v}")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        """
        Validate timestamp is not in the future.

        Args:
            v: Event timestamp

        Returns:
            Validated timestamp

        Raises:
            ValueError: If timestamp is in the future
        """
        now = datetime.now(timezone.utc)

        # Handle timezone-aware and naive datetimes
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
            logger.warning(
                "Received timezone-naive timestamp, assuming UTC"
            )

        if v > now:
            raise ValueError(
                f"timestamp cannot be in the future, got {v}, "
                f"current time: {now}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "rating": 4,
                "timestamp": "2024-01-15T20:45:00Z"
            }
        }
    }


class SearchEvent(BaseModel):
    """
    User search event schema.

    Represents a user searching for movies. Captures search queries
    and results for improving recommendation relevance.

    Attributes:
        user_id: Unique user identifier
        query: Search query string
        results_count: Number of results returned
        timestamp: Event timestamp (UTC)

    Validations:
        - query cannot be empty
        - results_count must be non-negative
        - timestamp cannot be in the future

    Example (Valid):
        >>> from datetime import datetime, timezone
        >>> event = SearchEvent(
        ...     user_id="user_123",
        ...     query="action movies",
        ...     results_count=25,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> print(event.query)
        'action movies'

    Example (Invalid - empty query):
        >>> event = SearchEvent(
        ...     user_id="user_123",
        ...     query="",  # Invalid: cannot be empty
        ...     results_count=25,
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: query cannot be empty
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier",
        min_length=1,
        max_length=255
    )
    query: str = Field(
        ...,
        description="Search query string",
        min_length=1,
        max_length=1000
    )
    results_count: int = Field(
        ...,
        description="Number of results returned",
        ge=0
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp (UTC)"
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty(cls, v: str) -> str:
        """
        Validate query is not empty.

        Args:
            v: Search query string

        Returns:
            Validated query (stripped of whitespace)

        Raises:
            ValueError: If query is empty after stripping
        """
        v = v.strip()
        if not v:
            raise ValueError("query cannot be empty")
        return v

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        """
        Validate timestamp is not in the future.

        Args:
            v: Event timestamp

        Returns:
            Validated timestamp

        Raises:
            ValueError: If timestamp is in the future
        """
        now = datetime.now(timezone.utc)

        # Handle timezone-aware and naive datetimes
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
            logger.warning(
                "Received timezone-naive timestamp, assuming UTC"
            )

        if v > now:
            raise ValueError(
                f"timestamp cannot be in the future, got {v}, "
                f"current time: {now}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "query": "sci-fi thriller",
                "results_count": 42,
                "timestamp": "2024-01-15T19:00:00Z"
            }
        }
    }


class SkipEvent(BaseModel):
    """
    User skip event schema.

    Represents a user skipping a movie before completion. Captures
    negative feedback signals for recommendation improvement.

    Attributes:
        user_id: Unique user identifier
        movie_id: Unique movie identifier
        watch_duration_seconds: How long user watched before skipping
        movie_duration_seconds: Total movie duration in seconds
        timestamp: Event timestamp (UTC)

    Validations:
        - watch_duration_seconds must be non-negative
        - movie_duration_seconds must be positive
        - watch_duration_seconds cannot exceed movie_duration_seconds
        - timestamp cannot be in the future

    Example (Valid):
        >>> from datetime import datetime, timezone
        >>> event = SkipEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     watch_duration_seconds=300,  # Watched 5 minutes
        ...     movie_duration_seconds=7200,  # Movie is 2 hours
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        >>> print(event.watch_duration_seconds)
        300

    Example (Invalid - watched longer than movie):
        >>> event = SkipEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     watch_duration_seconds=8000,  # Invalid: > movie duration
        ...     movie_duration_seconds=7200,  # Movie is 2 hours
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: watch_duration_seconds (8000) cannot exceed ...

    Example (Invalid - movie duration must be positive):
        >>> event = SkipEvent(
        ...     user_id="user_123",
        ...     movie_id="movie_456",
        ...     watch_duration_seconds=300,
        ...     movie_duration_seconds=0,  # Invalid: must be positive
        ...     timestamp=datetime.now(timezone.utc)
        ... )
        Traceback (most recent call last):
        ...
        ValidationError: movie_duration_seconds must be positive
    """

    user_id: str = Field(
        ...,
        description="Unique user identifier",
        min_length=1,
        max_length=255
    )
    movie_id: str = Field(
        ...,
        description="Unique movie identifier",
        min_length=1,
        max_length=255
    )
    watch_duration_seconds: int = Field(
        ...,
        description="How long user watched before skipping (seconds)",
        ge=0
    )
    movie_duration_seconds: int = Field(
        ...,
        description="Total movie duration in seconds",
        gt=0
    )
    timestamp: datetime = Field(
        ...,
        description="Event timestamp (UTC)"
    )

    @field_validator("movie_duration_seconds")
    @classmethod
    def validate_movie_duration_positive(cls, v: int) -> int:
        """
        Validate movie duration is positive.

        Args:
            v: Movie duration in seconds

        Returns:
            Validated movie duration

        Raises:
            ValueError: If movie duration is not positive
        """
        if v <= 0:
            raise ValueError(
                f"movie_duration_seconds must be positive, got {v}"
            )
        return v

    @model_validator(mode="after")
    def validate_skip_before_end(self) -> "SkipEvent":
        """
        Validate that watch duration doesn't exceed movie duration.

        A skip must happen before the movie ends. Watch duration
        cannot exceed movie duration.

        Returns:
            Validated SkipEvent instance

        Raises:
            ValueError: If watch duration exceeds movie duration
        """
        if self.watch_duration_seconds > self.movie_duration_seconds:
            raise ValueError(
                f"watch_duration_seconds ({self.watch_duration_seconds}) "
                f"cannot exceed movie_duration_seconds "
                f"({self.movie_duration_seconds}). "
                f"A skip must happen before the movie ends."
            )
        return self

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_not_future(cls, v: datetime) -> datetime:
        """
        Validate timestamp is not in the future.

        Args:
            v: Event timestamp

        Returns:
            Validated timestamp

        Raises:
            ValueError: If timestamp is in the future
        """
        now = datetime.now(timezone.utc)

        # Handle timezone-aware and naive datetimes
        if v.tzinfo is None:
            v = v.replace(tzinfo=timezone.utc)
            logger.warning(
                "Received timezone-naive timestamp, assuming UTC"
            )

        if v > now:
            raise ValueError(
                f"timestamp cannot be in the future, got {v}, "
                f"current time: {now}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "user_123",
                "movie_id": "movie_456",
                "watch_duration_seconds": 600,
                "movie_duration_seconds": 5400,
                "timestamp": "2024-01-15T20:10:00Z"
            }
        }
    }
