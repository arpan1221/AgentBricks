# AgentBricks - MCP User Commands

These are custom commands the user can invoke for common tasks. Format: `/command` or natural language trigger.

## Development Workflow Commands

### `/start-brick <number>`
**Trigger**: "start brick 1", "begin brick 2"
**Action**:
- Display brick story introduction
- Show learning objectives
- List all tasks
- Create feature branch: `feature/brick-0X-<name>`
- Show first task to complete

**Example Response**:
```
🧱 Starting Brick 01: Data Collection Service

📖 Story Context:
Welcome to MovieWorld! You've been hired as a Platform Engineer...

🎯 Learning Objectives:
- Event-driven architecture
- API design with FastAPI
- Kafka integration
...

📋 Tasks (15 total):
✅ Task 1: Setup development environment
⬜ Task 2: Define event schemas
...

🌿 Created branch: feature/brick-01-data-collection

💡 Next Step: Complete Task 1 - Setup development environment
Type /show-task 1 for details.
```

---

### `/show-task <number>`
**Trigger**: "show task 3", "what is task 5"
**Action**:
- Display detailed task instructions
- Show acceptance criteria
- Provide hints (collapsible)
- Link to relevant docs

**Example Response**:
```
📋 Task 3: Implement API Endpoints

Description:
Create FastAPI endpoints for all event types (view, rating, search, skip).
Each endpoint should validate input using Pydantic models and return
appropriate status codes.

Acceptance Criteria:
✅ POST /events/view accepts ViewEvent schema
✅ POST /events/rating accepts RatingEvent schema
✅ All endpoints return 200 on success
✅ Validation errors return 400 with details
✅ OpenAPI docs auto-generated at /docs

💡 Hints:
[Click to expand]
- Use Pydantic BaseModel for schemas
- FastAPI decorators: @app.post("/endpoint")
- Include type hints for auto docs

📚 Resources:
- FastAPI Documentation: https://fastapi.tiangolo.com
- Pydantic Models: https://docs.pydantic.dev
```

---

### `/complete-task <number>`
**Trigger**: "completed task 2", "finished task 5"
**Action**:
- Mark task as complete
- Run automated checks if applicable
- Show progress
- Suggest next task

---

### `/ask <question>`
**Trigger**: Natural questions about the project
**Action**:
- Search CLAUDE.md and documentation
- Provide relevant answer with context
- Show code examples if applicable
- Link to detailed docs

**Example**:
```
User: /ask how do I implement caching in the recommendation service?

Response:
🔍 Caching in AgentBricks Recommendation Service

For Brick 04, implement a two-level caching strategy:

1. **Redis Cache** (primary):
   - Cache recommendations by user_id
   - TTL: 5 minutes
   - Key pattern: `recommendations:{user_id}:{k}`

2. **Local Memory Cache** (fallback):
   - If Redis unavailable
   - LRU cache with max 1000 entries

Example implementation:
[Code snippet]

📍 Location: stories/movie-recommender/brick-04.../src/cache.py
📚 See also: Checkpoint Guide - Brick 04
```

---

### `/hint <task-number> <level>`
**Trigger**: "give me a hint for task 3", "hint level 2 for task 5"
**Action**:
- Provide graduated hints (1=gentle, 2=specific, 3=detailed)
- Never give full solution
- Encourage thinking

**Example**:
```
💡 Hint (Level 1) for Task 3: API Endpoints

Think about the HTTP method and path structure. What RESTful convention
makes sense for submitting events?

Need more help? Try /hint 3 2
```

---

### `/review`
**Trigger**: "review my code", "check my implementation"
**Action**:
- Analyze recent changes
- Check against coding standards
- Suggest improvements
- Verify tests exist
- Check documentation

**Example Response**:
```
🔍 Code Review: Brick 01 Implementation

✅ Good:
- All functions have type hints
- Pydantic models are well-structured
- Error handling is explicit

💡 Suggestions:
1. Add docstrings to helper functions (lines 45-60)
2. Consider adding integration test for Kafka producer
3. Log levels: Use DEBUG for verbose, INFO for key events

📊 Test Coverage: 78% (needs 2% more for >80% target)
Missing coverage: src/kafka_producer.py lines 103-107

Overall: Strong implementation! Address the suggestions and you're ready.
```

---

### `/test`
**Trigger**: "run tests", "test my code"
**Action**:
- Run relevant test suite
- Show results with coverage
- Highlight failures
- Suggest fixes for failures

---

### `/checkpoint`
**Trigger**: "am I ready to move on?", "check if brick complete"
**Action**:
- Verify all acceptance criteria met
- Check test coverage
- Validate documentation
- Confirm Docker setup works
- Give pass/fail with details

**Example Response**:
```
✅ Checkpoint: Brick 01 - Data Collection Service

Acceptance Criteria:
✅ All 4 event types supported
✅ Schema validation rejects invalid payloads
✅ Events successfully written to Kafka
✅ API responds <50ms P95
❌ Handles 1000 req/sec (currently 850 req/sec)
✅ Test coverage 82%
✅ Docker Compose setup works

Status: Almost Ready! ⚠️
Fix: Optimize API to handle 1000 req/sec. Consider:
- Add Redis caching
- Use connection pooling
- Profile with cProfile to find bottleneck

Run /optimize for suggestions.
```

---

### `/submit-brick`
**Trigger**: "submit brick 1", "I'm done with brick 2"
**Action**:
- Run full checkpoint validation
- Generate completion certificate
- Show next steps
- Celebrate milestone

---

## Code Generation Commands

### `/generate <component> <name>`
**Trigger**: "generate api endpoint", "create pydantic model"
**Action**: Generate boilerplate code with proper structure

**Examples**:
```
/generate api-endpoint /events/favorite
/generate pydantic-model FavoriteEvent
/generate test test_favorite_event
/generate docker-compose brick-04
```

---

### `/scaffold <brick-number>`
**Trigger**: "scaffold brick 3"
**Action**:
- Create full brick directory structure
- Generate placeholder files
- Add initial README template
- Create docker-compose skeleton

---

## Debugging Commands

### `/debug <error>`
**Trigger**: "debug", "I'm getting this error"
**Action**:
- Analyze error message
- Identify root cause
- Provide fix with explanation
- Suggest prevention

**Example**:
```
User: /debug "AttributeError: 'NoneType' object has no attribute 'get'"

Response:
🐛 Debugging: AttributeError

Root Cause:
You're trying to call .get() on a variable that is None. This typically
happens when:
1. Database query returns no results
2. Dictionary lookup fails
3. API response is empty

Likely Location:
Check your code where you're accessing user_features or movie_features.

Fix:
[Code showing proper None handling]

Prevention:
Always validate data exists before accessing:
if result is not None:
    value = result.get('key')
```

---

### `/optimize <component>`
**Trigger**: "optimize the api", "make this faster"
**Action**:
- Profile the component
- Identify bottlenecks
- Suggest optimizations
- Show before/after benchmarks

---

## Documentation Commands

### `/docs <topic>`
**Trigger**: "show docs on kafka", "documentation for faiss"
**Action**:
- Search project documentation
- Show relevant sections
- Link to external resources
- Provide examples

---

### `/example <pattern>`
**Trigger**: "show example async endpoint", "example feature engineering"
**Action**:
- Provide code examples from project
- Explain each part
- Show usage
- Link to full implementation

---

## Progress Tracking Commands

### `/progress`
**Trigger**: "show my progress", "how am I doing"
**Action**:
- Display completed bricks/tasks
- Show current brick progress
- Estimate time remaining
- Celebrate milestones

**Example Response**:
```
📊 Your AgentBricks Progress

Story Arc: Movie Recommender
━━━━━━━━━━━━━━━━━━━━━━━━ 50% (3/6 bricks)

✅ Brick 01: Data Collection (completed 2 days ago)
✅ Brick 02: Feature Engineering (completed 1 day ago)
✅ Brick 03: Model Training (completed today) 🎉
⏳ Brick 04: Recommendation Service (in progress - 6/10 tasks)
⬜ Brick 05: Monitoring
⬜ Brick 06: Orchestration

⏱️ Time Invested: 24 hours
🎯 Estimated Completion: 3-4 days
🏆 Achievements: First Model Trained 🎊

Keep going! You're halfway there! 💪
```

---

### `/roadmap`
**Trigger**: "show roadmap", "what's next"
**Action**:
- Display project timeline
- Show current phase
- List upcoming features
- Link to detailed roadmap

---

## Learning Support Commands

### `/explain <concept>`
**Trigger**: "explain collaborative filtering", "what is FAISS"
**Action**:
- Provide clear explanation
- Use analogies
- Show code example
- Link to resources

**Example**:
```
User: /explain two-stage retrieval and ranking

Response:
📚 Two-Stage Retrieval + Ranking

Think of it like shopping:

Stage 1 - Retrieval (Fast & Broad):
Like walking into a store and seeing displays. You quickly scan 100+ items
that might interest you based on basic criteria (genre, popularity).
- Uses: FAISS (approximate nearest neighbor)
- Speed: <10ms
- Goal: High recall (don't miss good items)

Stage 2 - Ranking (Slow & Precise):
Like carefully comparing your top 10 items. You consider all features to
rank them perfectly.
- Uses: ML model (NCF in our case)
- Speed: <50ms
- Goal: High precision (best items first)

Why Two Stages?
- Can't run complex ML model on millions of items (too slow)
- Can't rank accurately with simple retrieval (not personalized enough)
- Together: Fast + Accurate ⚡

In AgentBricks Brick 04:
[Code example showing both stages]

📖 Read more: docs/architecture/serving-layer.md
```

---

### `/compare <option1> <option2>`
**Trigger**: "compare FastAPI and Flask", "pytorch vs tensorflow"
**Action**:
- Compare technologies/approaches
- Show pros/cons
- Explain project choice
- Provide examples

---

## Project Setup Commands

### `/setup`
**Trigger**: "setup project", "initialize environment"
**Action**:
- Guide through setup steps
- Check prerequisites
- Verify installation
- Run setup script

---

### `/verify`
**Trigger**: "verify setup", "check my environment"
**Action**:
- Run verification checks
- Show status of each component
- Identify issues
- Provide fixes

**Example Response**:
```
🔍 Environment Verification

✅ Python 3.11.5 installed
✅ Virtual environment activated
✅ All dependencies installed
✅ Docker running
✅ Docker Compose available
⚠️ Port 8000 already in use
❌ Kafka not running

Issues Found: 2

Fixes:
1. Port 8000 conflict:
   lsof -i :8000  # Find process
   kill -9 <PID>  # Kill process

2. Start Kafka:
   cd stories/movie-recommender/brick-01-data-collection
   docker-compose up -d kafka

Overall Status: 85% Ready ⚠️
Fix the issues above and run /verify again.
```

---

## Community Commands

### `/contribute`
**Trigger**: "how can I contribute", "I want to help"
**Action**:
- Show contribution opportunities
- Link to CONTRIBUTING.md
- Suggest good first issues
- Explain PR process

---

### `/showcase`
**Trigger**: "showcase my work", "share my project"
**Action**:
- Help prepare showcase
- Generate README highlights
- Create demo instructions
- Provide sharing templates

---

## Miscellaneous Commands

### `/motivate`
**Trigger**: "I'm stuck", "this is hard", "losing motivation"
**Action**:
- Provide encouragement
- Break down problem
- Show progress made
- Connect to career goals

**Example**:
```
💪 You've Got This!

I know it feels challenging right now, but look how far you've come:

✅ You've built a production Kafka pipeline
✅ You've implemented ML feature engineering
✅ You've trained a real recommendation model

These are skills that:
- Take most engineers months to learn
- Are in HIGH demand in industry
- Directly apply to jobs at Netflix, Spotify, etc.

Current Challenge: Serving layer optimization
Break it down:
1. First, get it working (even if slow) ✓
2. Then, measure the bottleneck (use profiling)
3. Finally, optimize (add caching)

You're on step 2 - which means you're 66% done with this challenge!

🎯 Remember Why You Started:
Build a portfolio project that gets you interviews.
You're building that right now!

Take a 15-minute break, then tackle step 2. You've got this! 🚀
```

---

### `/tldr`
**Trigger**: "summarize this", "give me the short version"
**Action**:
- Provide concise summary
- Highlight key points
- Show action items

---

### `/help`
**Trigger**: "help", "show commands"
**Action**:
- List all available commands
- Show command syntax
- Provide examples
- Link to full documentation

---

## Command Aliases

Users can use natural language instead of exact commands:

- "I'm done with brick 1" → `/submit-brick 1`
- "What should I do next?" → `/show-task next`
- "Check if I can move on" → `/checkpoint`
- "How's my progress?" → `/progress`
- "I need help with this error" → `/debug`
- "Show me how to do X" → `/example X`
- "Explain X to me" → `/explain X`

---

**These commands provide guided assistance throughout the AgentBricks learning journey.**
