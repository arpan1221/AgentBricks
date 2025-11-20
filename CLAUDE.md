# AgentBricks Project Context

This file provides context for AI assistants (Claude, Cursor AI, etc.) working on the AgentBricks project.

## Project Overview

**AgentBricks** is an open-source, story-driven learning platform that teaches ML system design through hands-on building. Learners build production-grade recommendation systems for millions of synthetic agents, gaining real-world experience without real user data.

### Core Concept
- **Synthetic Universe**: Millions of simulated users with realistic behaviors
- **Story-Driven**: Learning paths presented as engaging narratives
- **Modular "Bricks"**: Each system component is a self-contained learning module
- **Production-Grade**: Real Kafka, real databases, real ML models, real cloud deployment
- **Portfolio-Ready**: Every project becomes a showcase piece for careers

## Project Architecture

### High-Level System Flow
```
Synthetic Agents → Event Generation → Kafka Ingestion → Feature Engineering 
→ Model Training → Model Registry → Recommendation Serving → Monitoring
```

### Technology Stack

**Core Languages:**
- Python 3.11+ (primary)
- SQL (DuckDB, PostgreSQL)
- YAML (configuration)
- Bash (scripts)

**Data Infrastructure:**
- Kafka (event streaming)
- MongoDB (event storage)
- DuckDB (feature store)
- Redis (caching)
- PostgreSQL (metadata)

**ML Stack:**
- PyTorch (model training)
- scikit-learn (preprocessing)
- FAISS (vector search)
- MLflow (model registry)
- Prefect/Airflow (orchestration)

**API & Serving:**
- FastAPI (API framework)
- Uvicorn (ASGI server)
- Pydantic v2 (validation)
- Docker (containerization)

**Monitoring:**
- Prometheus (metrics)
- Grafana (visualization)
- Structured logging (JSON)

**Cloud & IaC:**
- Terraform (infrastructure)
- Docker Compose (local dev)
- Kubernetes (production, optional)
- AWS/GCP/Azure (multi-cloud support)

**Development:**
- pytest (testing)
- Black (formatting)
- Ruff (linting)
- mypy (type checking)
- pre-commit (git hooks)

## Repository Structure

```
agentbricks/
├── sim/                    # Synthetic agent simulation engine
├── bricks/                 # Reusable system components
├── stories/                # Story-driven learning arcs
│   └── movie-recommender/  # First story arc (6 bricks)
├── api/                    # Shared API components
├── agent/                  # AI mentor system
├── infra/                  # Infrastructure as code
├── docs/                   # Documentation
├── tests/                  # Test suite
└── scripts/               # Utility scripts
```

## Coding Standards

### Python Style
- **Formatting**: Black with line length 100
- **Linting**: Ruff (select=["E", "F", "I"])
- **Type Hints**: Required on all functions and class methods
- **Docstrings**: Google style, comprehensive
- **Naming**:
  - Functions/variables: `snake_case`
  - Classes: `PascalCase`
  - Constants: `UPPER_CASE`
  - Private: `_leading_underscore`

### Code Quality Requirements
- **Test Coverage**: >80% for all modules
- **Type Checking**: Pass mypy strict mode
- **Documentation**: Every public function/class must have docstrings
- **Error Handling**: Explicit, never bare `except:`
- **Logging**: Structured JSON logs with appropriate levels

### Commit Messages
Follow Conventional Commits:
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: feat, fix, docs, style, refactor, test, chore, perf

**Examples**:
```
feat(brick-01): add event validation middleware
fix(sim): correct preference vector generation
docs(readme): update installation instructions
```

## Design Principles

### 1. Modularity First
- Every component should be self-contained
- Clear interfaces between modules
- Minimal coupling, high cohesion

### 2. Production-Grade Code
- This is NOT tutorial code
- Handle edge cases
- Robust error handling
- Comprehensive logging
- Performance considerations

### 3. Educational Value
- Code should be readable and well-documented
- Concepts explained in comments and docs
- Clear examples in docstrings
- Architecture decisions documented

### 4. Story-Driven
- Every brick has narrative context
- Learning objectives clearly stated
- Tasks build on each other logically
- Celebrate completions

### 5. Beginner-Friendly (But Not Simplistic)
- Assume basic Python knowledge
- Explain system design concepts
- Provide hints, not solutions
- Progressive difficulty

## Current Story Arc: Movie Recommender

### Brick 01: Data Collection Service
- **Focus**: Event streaming, API design, Kafka
- **Components**: FastAPI service, Kafka producer, schema validation
- **Learning**: Event-driven architecture, data quality at ingestion

### Brick 02: Feature Engineering
- **Focus**: Feature computation, temporal correctness, feature store
- **Components**: Feature pipeline, DuckDB feature store
- **Learning**: ML feature engineering, point-in-time correctness

### Brick 03: Model Training
- **Focus**: Neural Collaborative Filtering, training pipeline, model registry
- **Components**: PyTorch NCF model, training loop, MLflow integration
- **Learning**: Recommendation models, training best practices

### Brick 04: Recommendation Service
- **Focus**: Low-latency serving, two-stage retrieval+ranking, caching
- **Components**: FastAPI service, FAISS retrieval, Redis caching
- **Learning**: Production ML serving, latency optimization

### Brick 05: Monitoring
- **Focus**: Observability, metrics, alerting
- **Components**: Prometheus, Grafana dashboards, alert rules
- **Learning**: System monitoring, SLOs, on-call readiness

### Brick 06: Orchestration
- **Focus**: Pipeline automation, scheduling, dependency management
- **Components**: Airflow DAGs, end-to-end pipeline
- **Learning**: MLOps, workflow orchestration

## AI Assistant Guidelines

### When Generating Code

**DO:**
- Include comprehensive docstrings (Google style)
- Add type hints to all functions/methods
- Write defensive code with validation
- Include example usage in docstrings
- Add structured logging at appropriate levels
- Consider edge cases and error scenarios
- Make code self-documenting with clear variable names
- Add inline comments for complex logic

**DON'T:**
- Use bare `except:` statements
- Leave TODO comments in production code
- Hardcode configuration values
- Skip input validation
- Ignore type hints
- Write functions >50 lines (refactor if needed)
- Use magic numbers (define constants)

### When Generating Tests

**DO:**
- Write tests for happy path AND edge cases
- Use descriptive test names: `test_<function>_with_<condition>_returns_<expected>`
- Use pytest fixtures for test data
- Mock external dependencies
- Test error conditions explicitly
- Aim for >80% coverage
- Include docstrings explaining what's being tested

**DON'T:**
- Skip negative test cases
- Test implementation details (test behavior)
- Leave commented-out tests
- Use sleep() for timing (use proper async/mock)

### When Generating Documentation

**DO:**
- Write for early-career developers
- Include concrete examples
- Add architecture diagrams (ASCII or Mermaid)
- Explain the "why", not just the "what"
- Include troubleshooting sections
- Add links to external resources
- Use proper markdown formatting

**DON'T:**
- Assume deep knowledge of technologies
- Skip setup instructions
- Leave broken links
- Use jargon without explanation

### When Debugging

**DO:**
- Read error messages carefully and completely
- Check types and validation first
- Verify configuration/environment variables
- Check logs for context
- Test components in isolation
- Use the scientific method (hypothesis, test, refine)

**DON'T:**
- Guess randomly
- Make multiple changes at once
- Skip reading documentation
- Ignore warnings

## Performance Targets

### API Latency
- P50: <20ms
- P95: <100ms
- P99: <200ms

### Throughput
- Minimum: 100 req/sec per instance
- Target: 500 req/sec per instance

### Model Training
- Small dataset (100K interactions): <5 minutes
- Medium dataset (1M interactions): <30 minutes
- Large dataset (10M interactions): <2 hours

### Feature Engineering
- Batch features: Process 1M events in <10 minutes
- Real-time features: <10ms per user

## Common Patterns

### Async FastAPI Endpoint
```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)
app = FastAPI()

class EventRequest(BaseModel):
    user_id: str
    movie_id: str
    timestamp: datetime

@app.post("/events/view")
async def create_view_event(event: EventRequest):
    """
    Record a movie view event.
    
    Args:
        event: View event details
        
    Returns:
        dict: Success response with event_id
        
    Raises:
        HTTPException: If event processing fails
    """
    try:
        # Validate
        # Process
        # Store
        logger.info(f"Processed view event", extra={
            "user_id": event.user_id,
            "movie_id": event.movie_id
        })
        return {"status": "success", "event_id": "..."}
    except Exception as e:
        logger.error(f"Failed to process event: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Event processing failed")
```

### Feature Engineering with Temporal Correctness
```python
def compute_user_features(
    user_id: str,
    interactions_df: pd.DataFrame,
    as_of_date: datetime
) -> dict:
    """
    Compute user features with point-in-time correctness.
    
    Args:
        user_id: User identifier
        interactions_df: All user interactions
        as_of_date: Compute features as of this date (no future data)
        
    Returns:
        dict: User features
    """
    # Filter to only data before as_of_date
    historical_data = interactions_df[
        interactions_df['timestamp'] <= as_of_date
    ]
    
    # Compute features
    features = {
        'total_watch_time': historical_data['watch_time'].sum(),
        'avg_watch_time': historical_data['watch_time'].mean(),
        # ...
    }
    
    return features
```

### Model Training Loop
```python
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        optimizer: Optimizer instance
        device: Training device (cpu/cuda)
        
    Returns:
        float: Average loss for epoch
    """
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        user_ids, movie_ids, labels = batch
        user_ids = user_ids.to(device)
        movie_ids = movie_ids.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(user_ids, movie_ids)
        loss = F.binary_cross_entropy_with_logits(predictions, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

## References

### System Design
- Based on production ML recommendation systems
- Two-stage retrieval+ranking architecture
- Cold-start strategies (geographic, popularity-based)
- Negative sampling for implicit feedback
- Real-time and batch feature computation

### ML Concepts
- Collaborative Filtering (Neural Collaborative Filtering - NCF)
- Matrix Factorization
- Embedding-based retrieval (FAISS)
- Learning to Rank

### Infrastructure Patterns
- Event-driven architecture
- CQRS (Command Query Responsibility Segregation)
- Feature stores
- Model registries
- Microservices

## Key Decisions & Rationale

### Why Synthetic Data?
- No privacy concerns
- Unlimited scale
- Reproducible experiments
- Control over data distributions
- Educational use case

### Why Kafka?
- Industry-standard event streaming
- Handles high throughput
- Durability and replay capabilities
- Decouples producers and consumers

### Why DuckDB for Feature Store?
- Embedded, no server needed
- SQL interface (familiar)
- Excellent for analytics workloads
- Point-in-time queries
- Easy for learners to understand

### Why FastAPI?
- Modern, async Python framework
- Automatic OpenAPI docs
- Pydantic validation
- High performance
- Great developer experience

### Why PyTorch?
- Industry standard for deep learning
- Dynamic computation graphs
- Excellent documentation
- Strong community support

## Future Roadmap

### Planned Story Arcs
1. ✅ Movie Recommender (current)
2. 🔜 TikTok-style For You Page
3. 🔜 Amazon Product Ranking Engine
4. 🔜 Uber Dispatch & Matching
5. 🔜 Yelp Local Search
6. 🔜 Spotify Playlist Recommender

### Planned Features
- Web dashboard for progress tracking
- Advanced AI mentor with code review
- Video tutorials for each brick
- Community showcase
- Multi-language support
- Enterprise features (teams, analytics)

## Questions to Ask Before Generating Code

1. **Purpose**: What is this code's specific purpose in the learning journey?
2. **Prerequisites**: What should the learner have completed before this?
3. **Learning Objective**: What concept does this teach?
4. **Production Quality**: Is this production-grade code with proper error handling?
5. **Documentation**: Does this have comprehensive docstrings and comments?
6. **Testing**: What tests should accompany this code?
7. **Edge Cases**: What could go wrong, and how is it handled?
8. **Performance**: Are there performance considerations?

## Maintaining Context

When working on AgentBricks:
- Refer back to this file for standards and patterns
- Check the Master Execution Plan for timeline and priorities
- Use the Checkpoint Guide to verify completion
- Follow the Copy-Paste Prompts for consistency
- Keep the story-driven learning approach central

---

**Last Updated**: 2025
**Version**: 0.1.0
**Maintainer**: Solo founder (building with Cursor + Claude)