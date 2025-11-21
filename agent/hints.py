"""Hint system with graduated levels.

This module provides hints at different levels of specificity.
"""

from typing import Dict

# Hint database organized by task number and level
HINTS: Dict[int, Dict[int, str]] = {
    1: {
        1: """
Think about what you need to set up before writing code. Consider:
- Project structure (directories for src, tests, etc.)
- Dependencies (requirements.txt)
- Development environment (Docker, virtual environment)
- Configuration files
        """,
        2: """
Here's a more specific approach:
1. Create the directory structure: `src/`, `tests/`, `config/`
2. Create a `requirements.txt` with FastAPI, Pydantic, Kafka dependencies
3. Set up a `Dockerfile` and `docker-compose.yml` for local development
4. Create a `.env.template` file for environment variables
5. Initialize a basic FastAPI app in `src/api.py`
        """,
        3: """
```python
# Project structure:
brick-01-data-collection/
├── src/
│   ├── __init__.py
│   ├── api.py          # FastAPI app
│   ├── schemas.py      # Pydantic models
│   └── kafka_producer.py
├── tests/
│   ├── __init__.py
│   └── test_api.py
├── requirements.txt
├── Dockerfile
└── docker-compose.yml

# requirements.txt:
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
aiokafka==0.10.0
python-dotenv==1.0.0

# Basic api.py structure:
from fastapi import FastAPI
app = FastAPI()

@app.get("/health")
async def health():
    return {"status": "ok"}
```
        """,
    },
    2: {
        1: """
Consider what information you need to capture for each event type.
Think about:
- What data is essential?
- What should be validated?
- What are the data types?
        """,
        2: """
For event schemas, you'll need:
1. ViewEvent: user_id, movie_id, timestamp, watch_time (optional)
2. RatingEvent: user_id, movie_id, rating (1-5), timestamp
3. SearchEvent: user_id, query, timestamp, results_count (optional)
4. SkipEvent: user_id, movie_id, timestamp, reason (optional)

Use Pydantic BaseModel with Field() for validation.
        """,
        3: """
```python
from pydantic import BaseModel, Field
from datetime import datetime

class ViewEvent(BaseModel):
    user_id: str = Field(..., description="User identifier")
    movie_id: str = Field(..., description="Movie identifier")
    timestamp: datetime = Field(default_factory=datetime.now)
    watch_time: Optional[int] = Field(None, ge=0, description="Watch time in seconds")

class RatingEvent(BaseModel):
    user_id: str
    movie_id: str
    rating: int = Field(..., ge=1, le=5)
    timestamp: datetime = Field(default_factory=datetime.now)
```
        """,
    },
    3: {
        1: """
Think about REST API design principles:
- Use appropriate HTTP methods (POST for creating events)
- Design clear URL paths
- Return proper status codes
- Handle errors gracefully
        """,
        2: """
For FastAPI endpoints:
1. Use `@app.post("/events/{event_type}")` decorator
2. Accept Pydantic models as request bodies
3. Return 200 on success, 400 on validation errors
4. Use dependency injection for shared resources (like Kafka producer)
5. Add request/response logging
        """,
        3: """
```python
from fastapi import FastAPI, HTTPException, status
from src.schemas import ViewEvent

app = FastAPI()

@app.post("/events/view", status_code=status.HTTP_200_OK)
async def create_view_event(event: ViewEvent):
    try:
        # Validate event
        # Send to Kafka
        # Return success
        return {"status": "success", "event_id": "..."}
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
```
        """,
    },
}


def get_hint(task_number: int, level: int) -> str:
    """Get a hint for a specific task at a given level.

    Args:
        task_number: The task number
        level: Hint level (1=gentle, 2=specific, 3=pseudocode)

    Returns:
        str: The hint text
    """
    if task_number not in HINTS:
        return f"""
I don't have specific hints for Task {task_number} yet.

General guidance:
- Review the task description carefully
- Check the acceptance criteria
- Look at similar tasks in other bricks
- Use `agentbricks ask <question>` for more help
        """

    task_hints = HINTS[task_number]

    if level not in task_hints:
        # Return the highest available level
        max_level = max(task_hints.keys())
        return task_hints[max_level]

    return task_hints[level]
