# AgentBricks - MCP Project Rules

## Project Overview

**Name**: AgentBricks
**Type**: Open-source educational platform
**Purpose**: Teach production ML system design through story-driven learning
**Audience**: Early-career software engineers, bootcamp grads, self-taught developers
**License**: MIT

## Project Structure

```
agentbricks/
├── sim/                          # Synthetic agent simulation engine
│   ├── agents/                   # User agent models
│   ├── items/                    # Movie/item models
│   └── interactions/             # Interaction rules and generation
├── bricks/                       # Reusable system components
│   ├── ingestion/                # Data collection patterns
│   ├── features/                 # Feature engineering utilities
│   ├── training/                 # Model training utilities
│   ├── serving/                  # Serving layer components
│   ├── monitoring/               # Observability utilities
│   └── orchestration/            # Workflow components
├── stories/                      # Story-driven learning arcs
│   └── movie-recommender/        # First story arc (6 bricks)
│       ├── brick-01-data-collection/
│       ├── brick-02-feature-engineering/
│       ├── brick-03-model-training/
│       ├── brick-04-recommendation-service/
│       ├── brick-05-monitoring/
│       └── brick-06-orchestration/
├── agent/                        # AI mentor system
├── infra/                        # Infrastructure as code
├── docs/                         # Documentation
└── tests/                        # Test suite
```

## Coding Standards

### Python Style Guide
- **Python Version**: 3.11+
- **Formatter**: Black (line-length=100)
- **Linter**: Ruff (select=["E", "F", "I"])
- **Type Checker**: mypy (strict mode)
- **Docstring Style**: Google format
- **Import Order**: stdlib, third-party, local (enforced by Ruff)

### Naming Conventions
- **Modules**: `lowercase_with_underscores.py`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Variables**: `snake_case`
- **Constants**: `UPPER_CASE`
- **Private**: `_leading_underscore`
- **Protected**: `_single_leading_underscore`

### Code Quality Requirements
- **Type Hints**: Required on all function signatures
- **Docstrings**: Required on all public functions/classes
- **Test Coverage**: Minimum 80% per module
- **Max Function Length**: 50 lines (refactor if longer)
- **Max Line Length**: 100 characters
- **Error Handling**: Explicit exceptions, never bare `except:`

## Technology Stack Constraints

### Required Technologies
- **Language**: Python 3.11+
- **Data Infrastructure**: Kafka, MongoDB, DuckDB, Redis
- **ML Stack**: PyTorch, scikit-learn, FAISS, MLflow
- **API Framework**: FastAPI with Pydantic v2
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Prometheus, Grafana
- **Orchestration**: Airflow or Prefect
- **Testing**: pytest with pytest-asyncio, pytest-cov

### Technology Decisions
- **Kafka over RabbitMQ**: Industry standard, better for high throughput
- **FastAPI over Flask**: Modern, async, automatic docs, Pydantic validation
- **DuckDB over SQLite**: Better for analytics, columnar storage
- **PyTorch over TensorFlow**: More Pythonic, dynamic graphs, better for learning
- **FAISS over Annoy**: Better performance, more algorithms, GPU support
- **Prometheus/Grafana**: Industry standard observability stack

## Architecture Principles

### System Design
1. **Event-Driven Architecture**: Use Kafka for async event processing
2. **Two-Stage Serving**: Retrieval (FAISS) → Ranking (ML model)
3. **CQRS**: Separate read and write models
4. **Microservices**: Each brick is independently deployable
5. **Feature Store Pattern**: Centralized feature management
6. **Model Registry**: Versioned model artifacts with MLflow

### Scalability Patterns
- Horizontal scaling with stateless services
- Caching layer (Redis) for frequently accessed data
- Batch + streaming feature computation
- Async API endpoints
- Connection pooling for databases

### Performance Targets
- **API Latency**: P50 < 20ms, P95 < 100ms, P99 < 200ms
- **Throughput**: Minimum 100 req/sec per instance
- **Model Training**: <30 minutes for 1M interactions
- **Feature Engineering**: <10 minutes for 1M events
- **Cache Hit Rate**: >50%

## Testing Requirements

### Test Types
1. **Unit Tests**: Test individual functions/classes in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete user flows
4. **Performance Tests**: Verify latency and throughput targets

### Test Organization
```
tests/
├── unit/
│   ├── test_agents.py
│   ├── test_models.py
│   └── test_features.py
├── integration/
│   ├── test_api_kafka.py
│   └── test_training_pipeline.py
└── e2e/
    └── test_movie_recommender_flow.py
```

### Test Naming Convention
- Pattern: `test_<function>_with_<condition>_<expected_result>`
- Example: `test_compute_features_with_new_user_returns_defaults`

### Coverage Requirements
- Minimum 80% line coverage per module
- Critical paths must have 100% coverage
- All edge cases must be tested
- All error paths must be tested

## Documentation Standards

### Code Documentation
- All public functions/classes require Google-style docstrings
- Include Args, Returns, Raises, Examples sections
- Complex logic requires inline comments explaining "why"
- Reference papers/articles for algorithms

### Project Documentation
1. **README.md**: Project overview, quick start, features
2. **CONTRIBUTING.md**: How to contribute, development workflow
3. **Architecture docs**: System design, data flows, diagrams
4. **API docs**: Generated from FastAPI (OpenAPI)
5. **Brick READMEs**: Story context, tasks, acceptance criteria

### Documentation Updates
- Update docs in same PR as code changes
- Keep architecture docs synchronized with implementation
- Update CHANGELOG.md for all releases
- Version API documentation

## Git Workflow

### Branch Strategy
- **main**: Stable, production-ready code
- **dev**: Integration branch for features
- **feature/\***: New features or bricks
- **bugfix/\***: Bug fixes
- **docs/\***: Documentation updates
- **refactor/\***: Code improvements without feature changes

### Commit Message Format (Conventional Commits)
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvement

**Examples**:
```
feat(brick-01): add request validation middleware

Implement Pydantic validation for all event types.
Includes comprehensive error messages and logging.

Closes #42
```

### Pull Request Process
1. Create feature branch from `dev`
2. Implement changes with tests
3. Ensure all tests pass locally
4. Push and create PR to `dev`
5. PR must include:
   - Clear description of changes
   - Link to related issue
   - Test evidence (screenshots, logs)
   - Updated documentation
6. Require at least one review
7. All CI checks must pass
8. Squash and merge to `dev`

## CI/CD Requirements

### Continuous Integration
Must pass before merge:
- [ ] All tests pass (unit, integration)
- [ ] Coverage >80%
- [ ] Black formatting check
- [ ] Ruff linting (no errors)
- [ ] mypy type checking (strict)
- [ ] Docker image builds successfully
- [ ] No security vulnerabilities (Bandit, Safety)

### Continuous Deployment
- Merge to `main` triggers release process
- Docker images tagged and pushed
- Documentation deployed
- GitHub release created

## Security Standards

### Never Commit
- API keys, passwords, secrets
- `.env` files with real credentials
- Private keys or certificates
- Database connection strings with passwords
- Personal identifiable information (PII)

### Required Practices
- Use environment variables for all secrets
- Validate all user inputs (Pydantic models)
- Parameterize all database queries
- Rate limit API endpoints
- Implement authentication/authorization for admin endpoints
- Keep dependencies updated (Dependabot)
- Run security scans in CI

## Dependency Management

### Core Dependencies
```
# Web Framework
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0

# Data Infrastructure
aiokafka==0.10.0
pymongo==4.6.0
redis==5.0.1

# ML Stack
torch==2.1.0
scikit-learn==1.3.2
faiss-cpu==1.7.4
mlflow==2.9.1

# Data Processing
pandas==2.1.3
numpy==1.26.2

# Orchestration
prefect==2.14.9

# Monitoring
prometheus-client==0.19.0

# Testing
pytest==7.4.3
pytest-cov==4.1.0
pytest-asyncio==0.21.1
```

### Update Policy
- Security updates: Apply immediately
- Minor updates: Monthly review and update
- Major updates: Evaluate breaking changes, update carefully

## Performance Optimization

### Required Optimizations
- Use async/await for I/O operations
- Implement connection pooling (DB, Redis)
- Cache frequently accessed data (Redis)
- Batch database operations
- Use appropriate indexes
- Profile before optimizing
- Load test before deployment

### Monitoring & Observability
- Structured JSON logging (timestamp, level, message, context)
- Prometheus metrics for all services
- Grafana dashboards for visualization
- Distributed tracing (request IDs)
- Alert rules for SLA violations

## Story Arc Development

### Brick Structure
Each brick must include:
1. **README.md**: Story, objectives, tasks, hints
2. **src/**: Implementation code
3. **tests/**: Comprehensive test suite
4. **docker-compose.yml**: Local development setup
5. **solution/**: Reference implementation
6. **docs/**: Additional documentation

### Learning Objectives
- Each brick teaches specific system design concepts
- Progressive difficulty (builds on previous bricks)
- Connects to real-world ML systems
- Includes "why" along with "how"

### Acceptance Criteria
- All tests pass (>80% coverage)
- Meets performance targets
- Docker Compose works on fresh clone
- Documentation complete and clear
- Reference solution provided

## Release Process

### Versioning (Semantic Versioning)
- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes

### Release Checklist
- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped (pyproject.toml)
- [ ] Git tag created (vX.Y.Z)
- [ ] GitHub release created
- [ ] Docker images published
- [ ] Announcement (LinkedIn, Twitter)

## Community Guidelines

### Code of Conduct
- Be respectful and inclusive
- Welcome beginners
- Provide constructive feedback
- Assume good intentions
- No harassment or discrimination

### Contributing
- Issues before PRs for significant changes
- Follow coding standards
- Include tests with all changes
- Update documentation
- Respond to review comments

### Recognition
- All contributors listed in README
- Highlight contributions in releases
- Monthly contributor spotlight

## Project Maintenance

### Regular Tasks
- **Daily**: Respond to issues, review PRs
- **Weekly**: Update dependencies, review metrics
- **Monthly**: Plan new features, community updates
- **Quarterly**: Major feature releases

### Issue Triage
- **bug**: Critical (fix immediately), High (fix this sprint), Normal (backlog)
- **feature**: Label by story arc or component
- **good-first-issue**: For new contributors
- **help-wanted**: Complex issues needing expertise

## Future Roadmap

### Planned Story Arcs
1. ✅ Movie Recommender (v0.1)
2. 🔜 TikTok-style For You Page (v0.2)
3. 🔜 Amazon Product Ranking (v0.3)
4. 🔜 Uber Dispatch System (v0.4)
5. 🔜 Yelp Local Search (v0.5)

### Planned Features
- Web dashboard for progress tracking
- Advanced AI mentor with code review
- Video tutorials
- Multi-language support
- Cloud deployment templates (AWS, GCP, Azure)

---

**These rules ensure consistency, quality, and maintainability across the AgentBricks project. All contributors must follow these standards.**
