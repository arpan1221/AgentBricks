# AgentBricks - MCP Project Commands

These commands operate at the project level and can be triggered automatically based on context or explicitly by the AI assistant.

## Automated Context Loading

### `@load-project-context`
**When**: Start of any conversation about AgentBricks
**Action**:
- Load CLAUDE.md into context
- Load current brick README if working on specific brick
- Load relevant architecture docs
- Check current git branch and recent commits

---

### `@check-file-changes`
**When**: User asks for review or mentions changes
**Action**:
- Review git diff
- Identify changed files
- Load those specific files into context
- Check if tests were updated alongside code

---

### `@load-related-code`
**When**: Discussing a specific component
**Action**:
- Load the main file being discussed
- Load related imports and dependencies
- Load corresponding tests
- Load relevant documentation

---

## Code Quality Automation

### `@run-linting`
**When**: Before suggesting code is ready / User asks for review
**Action**:
- Simulate Black formatting check
- Simulate Ruff linting
- Simulate mypy type checking
- Report issues found
- Suggest fixes

---

### `@check-test-coverage`
**When**: Checkpoint validation / User asks about coverage
**Action**:
- Analyze test files
- Estimate coverage based on code/test ratio
- Identify untested code paths
- Suggest additional tests

---

### `@validate-docstrings`
**When**: Code review / Checkpoint
**Action**:
- Check all public functions have docstrings
- Verify Google style format
- Check for Args, Returns, Examples sections
- Report missing documentation

---

## Architecture Validation

### `@verify-brick-structure`
**When**: User creates new brick / Brick completion check
**Action**:
- Verify directory structure matches template
- Check required files exist (README, src/, tests/, docker-compose.yml)
- Validate naming conventions
- Report missing components

---

### `@check-dependencies`
**When**: User adds imports / Integration issues
**Action**:
- Verify dependency exists in requirements.txt
- Check version compatibility
- Identify missing dependencies
- Suggest additions to requirements.txt

---

### `@validate-api-design`
**When**: Creating/modifying API endpoints
**Action**:
- Check RESTful conventions
- Verify Pydantic models used for validation
- Check status codes are appropriate
- Verify async endpoints where needed
- Check error handling present

---

## Performance Analysis

### `@estimate-latency`
**When**: Performance discussions / Optimization requests
**Action**:
- Analyze code patterns
- Identify potential bottlenecks
- Estimate approximate latency
- Suggest optimizations

**Example Analysis**:
```
🔍 Latency Analysis: Recommendation Endpoint

Current Implementation:
├─ Database query: ~50ms
├─ Feature computation: ~30ms
├─ FAISS retrieval: ~10ms
├─ Model inference: ~40ms
└─ Response formatting: ~5ms

Estimated Total: ~135ms (exceeds 100ms P95 target)

Bottlenecks:
1. Database query (37% of time)
2. Model inference (30% of time)

Optimization Suggestions:
1. Add Redis caching for user features (save ~50ms)
2. Batch model predictions (save ~20ms)
3. Use connection pooling (save ~10ms)

Estimated After Optimization: ~55ms ✅
```

---

### `@analyze-scalability`
**When**: Architecture discussions / Deployment planning
**Action**:
- Identify scalability concerns
- Check for stateless design
- Verify caching strategy
- Assess database query patterns
- Suggest horizontal scaling approach

---

## Testing Automation

### `@generate-test-cases`
**When**: User creates new function / Asks for test help
**Action**:
- Analyze function signature
- Identify edge cases
- Generate test templates
- Suggest mock strategies

**Example Output**:
```python
# Generated test cases for: compute_user_features()

def test_compute_user_features_with_valid_data_returns_features():
    """Test feature computation with typical user."""
    # Arrange
    user_id = "user_123"
    interactions = create_sample_interactions()

    # Act
    features = compute_user_features(user_id, interactions, datetime.now())

    # Assert
    assert 'total_watch_time' in features
    assert features['total_watch_time'] > 0

def test_compute_user_features_with_new_user_returns_defaults():
    """Test feature computation for user with no history."""
    # Edge case: new user

def test_compute_user_features_with_future_date_raises_error():
    """Test temporal correctness validation."""
    # Edge case: invalid date

def test_compute_user_features_with_empty_interactions_handles_gracefully():
    """Test handling of empty data."""
    # Edge case: no interactions
```

---

### `@suggest-integration-tests`
**When**: Component integration / Brick completion
**Action**:
- Identify integration points
- Suggest cross-component tests
- Generate test scenarios
- Show example test structure

---

## Documentation Generation

### `@generate-readme`
**When**: New brick creation / README incomplete
**Action**:
- Generate brick README template
- Include story context placeholder
- Add learning objectives structure
- Create tasks checklist template
- Add hints section template

---

### `@generate-architecture-doc`
**When**: Architecture discussions / New components
**Action**:
- Generate architecture documentation
- Create Mermaid/ASCII diagrams
- Document data flows
- Explain design decisions

---

### `@update-changelog`
**When**: Feature completion / Release preparation
**Action**:
- Generate CHANGELOG entry
- Categorize changes (Added, Changed, Fixed)
- Format according to Keep a Changelog
- Suggest version bump

---

## Error Analysis

### `@diagnose-error`
**When**: User shares error message / Debugging session
**Action**:
- Parse error message
- Identify error type
- Search codebase for likely cause
- Suggest fixes with explanations
- Provide prevention advice

**Example**:
```
🔍 Error Diagnosis

Error: ValidationError in EventSchema
Location: stories/.../brick-01/.../schemas.py:23
Type: Pydantic validation error

Root Cause:
Field 'timestamp' expected datetime, received string.

Why This Happened:
JSON doesn't have a native datetime type, so timestamps come as strings.
You need to parse the string into a datetime object.

Fix:
from datetime import datetime
from pydantic import validator

class EventSchema(BaseModel):
    timestamp: datetime

    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v)
        return v

Prevention:
Always validate datetime fields with custom validators when accepting
from external sources (API, Kafka, etc.).

Testing:
Add test case:
def test_event_schema_accepts_string_timestamp():
    event = EventSchema(timestamp="2024-01-15T10:30:00Z", ...)
    assert isinstance(event.timestamp, datetime)
```

---

## Refactoring Assistance

### `@suggest-refactoring`
**When**: Code review / Complexity warnings
**Action**:
- Identify code smells
- Suggest refactoring patterns
- Show before/after examples
- Maintain functionality

**Example**:
```
🔧 Refactoring Suggestion

Issue: Function `process_events()` is 87 lines (exceeds 50 line limit)

Suggested Refactoring:
Break into smaller functions:

Before:
def process_events(events):
    # 87 lines of mixed concerns

After:
def process_events(events):
    validated = validate_events(events)
    transformed = transform_events(validated)
    stored = store_events(transformed)
    return stored

def validate_events(events): ...
def transform_events(events): ...
def store_events(events): ...

Benefits:
- Each function has single responsibility
- Easier to test individually
- More maintainable
- Better named (self-documenting)
```

---

## Progress Tracking

### `@calculate-completion`
**When**: Progress queries / Checkpoint validation
**Action**:
- Count completed tasks
- Calculate percentage complete
- Estimate time remaining
- Identify blockers

---

### `@check-milestone`
**When**: Brick completion / Major feature done
**Action**:
- Verify milestone criteria met
- Generate completion summary
- Suggest celebration message
- Show next milestone

---

## Dependency Analysis

### `@check-circular-dependencies`
**When**: Import issues / Architecture review
**Action**:
- Analyze import graph
- Identify circular dependencies
- Suggest restructuring
- Show dependency tree

---

### `@audit-dependencies`
**When**: Security review / Dependency updates
**Action**:
- Check for outdated packages
- Identify security vulnerabilities
- Suggest upgrade path
- Show compatibility issues

---

## Deployment Validation

### `@validate-docker-config`
**When**: Docker setup / Deployment preparation
**Action**:
- Check Dockerfile best practices
- Verify multi-stage build
- Check for security issues
- Validate docker-compose.yml

---

### `@check-production-readiness`
**When**: Pre-deployment / Release checklist
**Action**:
- Verify all tests pass
- Check configuration for production
- Verify secrets are externalized
- Check monitoring is configured
- Validate error handling
- Check scaling capabilities

**Example Checklist**:
```
🚀 Production Readiness Check: Brick 04 Recommendation Service

✅ Code Quality
  ✅ All tests pass (coverage 84%)
  ✅ No linting errors
  ✅ Type checking passes

✅ Configuration
  ✅ All secrets in environment variables
  ✅ Configuration file for each environment
  ✅ Graceful shutdown implemented

⚠️ Performance
  ✅ Latency targets met (P95: 87ms)
  ⚠️ Throughput: 95 req/sec (target: 100)
  ✅ Caching implemented

✅ Observability
  ✅ Prometheus metrics exposed
  ✅ Structured logging implemented
  ✅ Health check endpoint present

❌ Security
  ✅ Input validation on all endpoints
  ❌ Rate limiting not implemented
  ✅ No secrets in code

Status: Nearly Ready (90%)

Blockers:
1. Implement rate limiting (15 min task)
2. Optimize to reach 100 req/sec (30 min task)

Estimated Time to Production: 45 minutes
```

---

## Code Search & Navigation

### `@find-pattern`
**When**: User asks "where is X used" / Code navigation
**Action**:
- Search codebase for pattern
- Show usage locations
- Provide context for each usage
- Suggest related code

---

### `@find-similar-code`
**When**: User needs examples / Learning from existing code
**Action**:
- Find similar implementations in codebase
- Show patterns and variations
- Explain differences
- Suggest best approach

---

## Learning Support

### `@explain-code-section`
**When**: User asks about existing code / Code review
**Action**:
- Analyze code section
- Explain what it does
- Explain why it's designed this way
- Show alternatives if applicable
- Connect to learning objectives

---

### `@compare-approaches`
**When**: Architecture decisions / Multiple solutions discussed
**Action**:
- Create comparison matrix
- Show pros/cons of each approach
- Explain project's choice
- Provide examples of each

**Example**:
```
⚖️ Comparison: Feature Store Options

| Aspect          | DuckDB (Chosen) | Redis         | PostgreSQL    |
|-----------------|-----------------|---------------|---------------|
| Setup           | ✅ Embedded     | 🟡 Server     | 🟡 Server     |
| Query Language  | ✅ SQL          | ❌ Commands   | ✅ SQL        |
| Analytics       | ✅ Optimized    | ❌ Limited    | 🟡 Good       |
| Point-in-time   | ✅ Easy         | ❌ Complex    | ✅ Easy       |
| Learning Curve  | ✅ Familiar     | 🟡 Moderate   | ✅ Familiar   |
| Production      | 🟡 Good         | ✅ Excellent  | ✅ Excellent  |

Why DuckDB for AgentBricks:
1. Educational: Students familiar with SQL
2. Simplicity: No server setup needed
3. Analytics: Optimized for feature queries
4. Point-in-time: Natural with SQL WHERE clauses

For Production: Consider migrating to Feast or Redis depending on scale.
```

---

## Automated Suggestions

### `@suggest-next-step`
**When**: User completes task / Asks what's next
**Action**:
- Analyze current progress
- Suggest logical next step
- Provide rationale
- Estimate time needed

---

### `@identify-risks`
**When**: Architecture decisions / Pre-deployment
**Action**:
- Analyze current implementation
- Identify potential issues
- Assess risk levels
- Suggest mitigations

**Example**:
```
⚠️ Risk Assessment: Current Implementation

🔴 High Risk:
1. No rate limiting on API endpoints
   Impact: Vulnerable to DDoS
   Mitigation: Add slowapi or custom rate limiter

2. Database connection pool not configured
   Impact: Exhausted connections under load
   Mitigation: Add connection pooling (10 lines)

🟡 Medium Risk:
1. No circuit breaker for external services
   Impact: Cascading failures
   Mitigation: Add tenacity retry with backoff

🟢 Low Risk:
1. Cache eviction strategy could be more sophisticated
   Impact: Suboptimal cache hit rate
   Mitigation: Can optimize later

Priority: Address High risks before deployment (ETA: 30 mins)
```

---

## Project-Wide Operations

### `@update-all-docs`
**When**: Major changes / Release preparation
**Action**:
- Check all README files
- Verify architecture docs match code
- Update API documentation
- Check links are valid
- Update version references

---

### `@generate-release-notes`
**When**: Release preparation / Version bump
**Action**:
- Analyze git commits since last release
- Categorize changes
- Generate release notes
- Highlight breaking changes
- List contributors

---

**These project commands enable intelligent, context-aware assistance throughout the AgentBricks development process.**
