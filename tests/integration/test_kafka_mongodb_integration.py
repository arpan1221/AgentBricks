"""Integration tests for Kafka -> MongoDB flow.

Tests that events consumed from Kafka are properly stored in MongoDB.
Note: This assumes there's a consumer service that reads from Kafka
and writes to MongoDB. For now, we test the MongoDB storage directly.
"""

from datetime import datetime, timezone

import pytest
from pymongo import MongoClient


@pytest.mark.integration
class TestKafkaToMongoDBIntegration:
    """Test Kafka to MongoDB event storage."""

    def test_event_stored_in_mongodb(self, mongo_client: MongoClient, wait_for_services):
        """Test that events can be stored in MongoDB."""
        db = mongo_client.agentbricks
        collection = db.events

        # Create test event
        test_event = {
            "event_id": "test_event_mongodb",
            "event_type": "view",
            "user_id": "test_user_mongodb",
            "movie_id": "test_movie_mongodb",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "watch_time_seconds": 3600,
            "created_at": datetime.now(timezone.utc),
        }

        # Insert event
        result = collection.insert_one(test_event)
        assert result.inserted_id is not None

        # Retrieve event
        retrieved_event = collection.find_one({"event_id": "test_event_mongodb"})
        assert retrieved_event is not None
        assert retrieved_event["user_id"] == test_event["user_id"]
        assert retrieved_event["movie_id"] == test_event["movie_id"]
        assert retrieved_event["event_type"] == "view"

        # Cleanup
        collection.delete_one({"event_id": "test_event_mongodb"})

    def test_multiple_events_stored_in_mongodb(self, mongo_client: MongoClient, wait_for_services):
        """Test that multiple events can be stored in MongoDB."""
        db = mongo_client.agentbricks
        collection = db.events

        # Create multiple test events
        test_events = []
        for i in range(5):
            event = {
                "event_id": f"test_event_mongodb_{i}",
                "event_type": "view",
                "user_id": f"user_{i}",
                "movie_id": f"movie_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600,
                "created_at": datetime.now(timezone.utc),
            }
            test_events.append(event)

        # Insert events
        result = collection.insert_many(test_events)
        assert len(result.inserted_ids) == 5

        # Retrieve events
        retrieved_events = list(collection.find({"event_id": {"$regex": "^test_event_mongodb_"}}))
        assert len(retrieved_events) == 5

        # Verify all events
        user_ids = {event["user_id"] for event in retrieved_events}
        assert len(user_ids) == 5

        # Cleanup
        collection.delete_many({"event_id": {"$regex": "^test_event_mongodb_"}})

    def test_event_query_by_user_id(self, mongo_client: MongoClient, wait_for_services):
        """Test querying events by user_id."""
        db = mongo_client.agentbricks
        collection = db.events

        # Create test events for a specific user
        user_id = "test_user_query"
        test_events = []
        for i in range(3):
            event = {
                "event_id": f"test_event_query_{i}",
                "event_type": "view",
                "user_id": user_id,
                "movie_id": f"movie_{i}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "watch_time_seconds": 3600,
                "created_at": datetime.now(timezone.utc),
            }
            test_events.append(event)

        # Insert events
        collection.insert_many(test_events)

        # Query by user_id
        user_events = list(collection.find({"user_id": user_id}))
        assert len(user_events) >= 3

        # Verify all events belong to the user
        for event in user_events:
            assert event["user_id"] == user_id

        # Cleanup
        collection.delete_many({"event_id": {"$regex": "^test_event_query_"}})
