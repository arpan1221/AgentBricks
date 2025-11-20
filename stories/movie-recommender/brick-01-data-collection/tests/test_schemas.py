"""Tests for Pydantic event schemas."""

import pytest
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from src.schemas import ViewEvent, RatingEvent, SearchEvent, SkipEvent


class TestViewEvent:
    """Tests for ViewEvent schema."""

    def test_view_event_with_valid_data_succeeds(self):
        """Test that valid view event data is accepted."""
        event = ViewEvent(
            user_id="user_123",
            movie_id="movie_456",
            timestamp=datetime.now(timezone.utc),
            watch_time_seconds=3600,
            session_id="session_789",
            device="desktop"
        )

        assert event.user_id == "user_123"
        assert event.movie_id == "movie_456"
        assert event.watch_time_seconds == 3600
        assert event.session_id == "session_789"
        assert event.device == "desktop"

    def test_view_event_with_minimal_data_succeeds(self):
        """Test that view event with minimal required fields succeeds."""
        event = ViewEvent(
            user_id="user_123",
            movie_id="movie_456",
            timestamp=datetime.now(timezone.utc),
            watch_time_seconds=0
        )

        assert event.user_id == "user_123"
        assert event.session_id is None
        assert event.device is None

    def test_view_event_with_zero_watch_time_succeeds(self):
        """Test that zero watch time is accepted."""
        event = ViewEvent(
            user_id="user_123",
            movie_id="movie_456",
            timestamp=datetime.now(timezone.utc),
            watch_time_seconds=0
        )

        assert event.watch_time_seconds == 0

    def test_view_event_with_max_watch_time_succeeds(self):
        """Test that 24-hour watch time is accepted."""
        max_watch_time = 24 * 60 * 60  # 86400 seconds
        event = ViewEvent(
            user_id="user_123",
            movie_id="movie_456",
            timestamp=datetime.now(timezone.utc),
            watch_time_seconds=max_watch_time
        )

        assert event.watch_time_seconds == max_watch_time

    def test_view_event_with_watch_time_exceeding_24_hours_raises_error(self):
        """Test that watch time exceeding 24 hours raises validation error."""
        max_watch_time = 24 * 60 * 60  # 86400 seconds

        with pytest.raises(ValidationError) as exc_info:
            ViewEvent(
                user_id="user_123",
                movie_id="movie_456",
                timestamp=datetime.now(timezone.utc),
                watch_time_seconds=max_watch_time + 1
            )

        assert "cannot exceed" in str(exc_info.value).lower()

    def test_view_event_with_negative_watch_time_raises_error(self):
        """Test that negative watch time raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ViewEvent(
                user_id="user_123",
                movie_id="movie_456",
                timestamp=datetime.now(timezone.utc),
                watch_time_seconds=-1
            )

        assert "greater than or equal to 0" in str(exc_info.value)

    def test_view_event_with_future_timestamp_raises_error(self):
        """Test that future timestamp raises validation error."""
        future_time = datetime.now(timezone.utc) + timedelta(days=1)

        with pytest.raises(ValidationError) as exc_info:
            ViewEvent(
                user_id="user_123",
                movie_id="movie_456",
                timestamp=future_time,
                watch_time_seconds=3600
            )

        assert "future" in str(exc_info.value).lower()

    @pytest.mark.parametrize("device", ["mobile", "tablet", "desktop", "tv"])
    def test_view_event_with_valid_devices_succeeds(self, device):
        """Test that all valid device types are accepted."""
        event = ViewEvent(
            user_id="user_123",
            movie_id="movie_456",
            timestamp=datetime.now(timezone.utc),
            watch_time_seconds=3600,
            device=device
        )

        assert event.device == device

    def test_view_event_with_invalid_device_raises_error(self):
        """Test that invalid device type raises validation error."""
        with pytest.raises(ValidationError):
            ViewEvent(
                user_id="user_123",
                movie_id="movie_456",
                timestamp=datetime.now(timezone.utc),
                watch_time_seconds=3600,
                device="invalid_device"
            )


class TestRatingEvent:
    """Tests for RatingEvent schema."""

    @pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
    def test_rating_event_with_valid_ratings_succeeds(self, rating):
        """Test that all valid rating values (1-5) are accepted."""
        event = RatingEvent(
            user_id="user_123",
            movie_id="movie_456",
            rating=rating,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.rating == rating

    def test_rating_event_with_minimal_rating_succeeds(self):
        """Test that minimum rating (1) is accepted."""
        event = RatingEvent(
            user_id="user_123",
            movie_id="movie_456",
            rating=1,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.rating == 1

    def test_rating_event_with_maximum_rating_succeeds(self):
        """Test that maximum rating (5) is accepted."""
        event = RatingEvent(
            user_id="user_123",
            movie_id="movie_456",
            rating=5,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.rating == 5

    @pytest.mark.parametrize("invalid_rating", [0, 6, -1, 10])
    def test_rating_event_with_invalid_ratings_raises_error(self, invalid_rating):
        """Test that invalid rating values raise validation error."""
        with pytest.raises(ValidationError) as exc_info:
            RatingEvent(
                user_id="user_123",
                movie_id="movie_456",
                rating=invalid_rating,
                timestamp=datetime.now(timezone.utc)
            )

        # Pydantic v2 returns different error messages - check for validation error
        error_str = str(exc_info.value).lower()
        assert ("greater than or equal to 1" in error_str or
                "less than or equal to 5" in error_str or
                "between 1 and 5" in error_str)

    def test_rating_event_with_future_timestamp_raises_error(self):
        """Test that future timestamp raises validation error."""
        future_time = datetime.now(timezone.utc) + timedelta(days=1)

        with pytest.raises(ValidationError):
            RatingEvent(
                user_id="user_123",
                movie_id="movie_456",
                rating=5,
                timestamp=future_time
            )


class TestSearchEvent:
    """Tests for SearchEvent schema."""

    def test_search_event_with_valid_data_succeeds(self):
        """Test that valid search event data is accepted."""
        event = SearchEvent(
            user_id="user_123",
            query="action movies",
            results_count=25,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.user_id == "user_123"
        assert event.query == "action movies"
        assert event.results_count == 25

    def test_search_event_with_zero_results_succeeds(self):
        """Test that zero results count is accepted."""
        event = SearchEvent(
            user_id="user_123",
            query="nonexistent movie",
            results_count=0,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.results_count == 0

    def test_search_event_with_negative_results_raises_error(self):
        """Test that negative results count raises validation error."""
        with pytest.raises(ValidationError):
            SearchEvent(
                user_id="user_123",
                query="test",
                results_count=-1,
                timestamp=datetime.now(timezone.utc)
            )

    def test_search_event_with_empty_query_raises_error(self):
        """Test that empty query raises validation error."""
        with pytest.raises(ValidationError):
            SearchEvent(
                user_id="user_123",
                query="",
                results_count=25,
                timestamp=datetime.now(timezone.utc)
            )

    def test_search_event_with_whitespace_only_query_raises_error(self):
        """Test that whitespace-only query raises validation error."""
        with pytest.raises(ValidationError):
            SearchEvent(
                user_id="user_123",
                query="   ",
                results_count=25,
                timestamp=datetime.now(timezone.utc)
            )

    def test_search_event_with_future_timestamp_raises_error(self):
        """Test that future timestamp raises validation error."""
        future_time = datetime.now(timezone.utc) + timedelta(days=1)

        with pytest.raises(ValidationError):
            SearchEvent(
                user_id="user_123",
                query="test",
                results_count=25,
                timestamp=future_time
            )


class TestSkipEvent:
    """Tests for SkipEvent schema."""

    def test_skip_event_with_valid_data_succeeds(self):
        """Test that valid skip event data is accepted."""
        event = SkipEvent(
            user_id="user_123",
            movie_id="movie_456",
            watch_duration_seconds=600,
            movie_duration_seconds=5400,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.user_id == "user_123"
        assert event.watch_duration_seconds == 600
        assert event.movie_duration_seconds == 5400

    def test_skip_event_with_zero_watch_duration_succeeds(self):
        """Test that zero watch duration is accepted."""
        event = SkipEvent(
            user_id="user_123",
            movie_id="movie_456",
            watch_duration_seconds=0,
            movie_duration_seconds=5400,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.watch_duration_seconds == 0

    def test_skip_event_with_watch_duration_equals_movie_duration_succeeds(self):
        """Test that watch duration equal to movie duration is accepted."""
        duration = 5400
        event = SkipEvent(
            user_id="user_123",
            movie_id="movie_456",
            watch_duration_seconds=duration,
            movie_duration_seconds=duration,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.watch_duration_seconds == duration

    def test_skip_event_with_watch_duration_exceeding_movie_duration_raises_error(self):
        """Test that watch duration exceeding movie duration raises error."""
        with pytest.raises(ValidationError) as exc_info:
            SkipEvent(
                user_id="user_123",
                movie_id="movie_456",
                watch_duration_seconds=5401,
                movie_duration_seconds=5400,
                timestamp=datetime.now(timezone.utc)
            )

        assert "cannot exceed" in str(exc_info.value).lower()

    def test_skip_event_with_negative_watch_duration_raises_error(self):
        """Test that negative watch duration raises validation error."""
        with pytest.raises(ValidationError):
            SkipEvent(
                user_id="user_123",
                movie_id="movie_456",
                watch_duration_seconds=-1,
                movie_duration_seconds=5400,
                timestamp=datetime.now(timezone.utc)
            )

    def test_skip_event_with_zero_movie_duration_raises_error(self):
        """Test that zero movie duration raises validation error."""
        with pytest.raises(ValidationError):
            SkipEvent(
                user_id="user_123",
                movie_id="movie_456",
                watch_duration_seconds=0,
                movie_duration_seconds=0,
                timestamp=datetime.now(timezone.utc)
            )

    def test_skip_event_with_negative_movie_duration_raises_error(self):
        """Test that negative movie duration raises validation error."""
        with pytest.raises(ValidationError):
            SkipEvent(
                user_id="user_123",
                movie_id="movie_456",
                watch_duration_seconds=0,
                movie_duration_seconds=-1,
                timestamp=datetime.now(timezone.utc)
            )

    def test_skip_event_with_future_timestamp_raises_error(self):
        """Test that future timestamp raises validation error."""
        future_time = datetime.now(timezone.utc) + timedelta(days=1)

        with pytest.raises(ValidationError):
            SkipEvent(
                user_id="user_123",
                movie_id="movie_456",
                watch_duration_seconds=600,
                movie_duration_seconds=5400,
                timestamp=future_time
            )

    @pytest.mark.parametrize("watch_duration,movie_duration", [
        (100, 5400),  # Watched 100 seconds of 90-minute movie
        (3000, 7200),  # Watched 50 minutes of 2-hour movie
        (5400, 5400),  # Watched full 90-minute movie (edge case)
    ])
    def test_skip_event_with_valid_durations_succeeds(self, watch_duration, movie_duration):
        """Test various valid duration combinations."""
        event = SkipEvent(
            user_id="user_123",
            movie_id="movie_456",
            watch_duration_seconds=watch_duration,
            movie_duration_seconds=movie_duration,
            timestamp=datetime.now(timezone.utc)
        )

        assert event.watch_duration_seconds == watch_duration
        assert event.movie_duration_seconds == movie_duration
