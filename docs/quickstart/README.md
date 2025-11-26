# Quick Start Guide

Get up and running with AgentBricks in 5 minutes! This guide will help you set up your development environment and start your first brick.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.11+**: Check with `python --version` or `python3 --version`
- **Docker & Docker Compose**: Check with `docker --version` and `docker compose version`
- **Git**: Check with `git --version`
- **8GB RAM minimum**: Recommended 16GB for smooth operation
- **10GB free disk space**: For Docker images, data, and dependencies

### Verifying Prerequisites

```bash
# Check Python version
python --version  # Should be 3.11 or higher

# Check Docker
docker --version
docker compose version

# Check Git
git --version

# Check available disk space (Linux/Mac)
df -h

# Check available RAM (Linux/Mac)
free -h
```

## Installation (5 minutes)

### Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/agentbricks.git
cd agentbricks
```

> **Note**: Replace `yourusername` with the actual GitHub username or organization.

### Step 2: Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pre-commit install
```

> **Tip**: Keep your virtual environment activated for all subsequent commands.

### Step 3: Generate Synthetic Data

```bash
cd sim

# Generate synthetic users, movies, and interactions
python generate.py generate-all --users 1000 --movies 500 --days 7

# This creates:
# - 1000 synthetic users
# - 500 movies
# - 7 days of interaction events
```

> **Note**: For faster setup, you can use smaller numbers (e.g., `--users 100 --movies 50 --days 1`). For production-like testing, use larger numbers.

### Step 4: Start Infrastructure

```bash
# Navigate to brick-01 directory
cd ../stories/movie-recommender/brick-01-data-collection

# Start all services (Kafka, MongoDB, API)
docker compose up -d

# Wait for services to be ready (about 30 seconds)
sleep 30
```

> **Note**: On first run, Docker will download images which may take a few minutes.

### Step 5: Verify Installation

```bash
# Check services are running
docker compose ps

# You should see:
# - kafka (running)
# - mongodb (running)
# - api (running)

# Test API health endpoint
curl http://localhost:8000/health

# Expected response:
# {"status":"healthy"}

# Check API documentation
# Open in browser: http://localhost:8000/docs
```

## Your First Brick (10 minutes)

Now that everything is set up, let's start with Brick 01: Data Collection Service.

### Using the CLI

```bash
# Make sure you're in the project root
cd ../../../../

# Start Brick 01
agentbricks start-brick 1

# This will:
# - Display the brick story and objectives
# - Create a feature branch
# - Show you the tasks to complete
```

### Manual Setup

If you prefer to work without the CLI:

```bash
# Navigate to brick directory
cd stories/movie-recommender/brick-01-data-collection

# Create feature branch
git checkout -b feature/brick-01-data-collection

# Review the README
cat README.md

# Start with Task 1
# Follow the instructions in the brick's README
```

### Complete Your First Task

1. **Review the Story**: Read the brick's README to understand the context
2. **Check Objectives**: Understand what you'll learn
3. **Start Task 1**: Set up the development environment
4. **Get Help**: Use `agentbricks hint 1` if you're stuck
5. **Ask Questions**: Use `agentbricks ask "your question"` for guidance

### Example: Sending Your First Event

```bash
# Send a view event to the API
curl -X POST http://localhost:8000/events/view \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "movie_id": "movie_456",
    "timestamp": "2025-01-15T10:00:00Z"
  }'

# Expected response:
# {"status":"success","event_id":"..."}
```

## Common Issues

### Docker not starting

**Symptoms**: `docker compose up` fails or services don't start

**Solutions**:
- Ensure Docker daemon is running: `docker ps`
- Check ports are available:
  - Port 8000 (API)
  - Port 9092 (Kafka)
  - Port 27017 (MongoDB)
- On Linux, you may need `sudo` or add user to docker group
- Restart Docker Desktop if using macOS/Windows

```bash
# Check if ports are in use
# On macOS/Linux:
lsof -i :8000
lsof -i :9092
lsof -i :27017

# On Windows:
netstat -ano | findstr :8000
```

### Python dependencies failing

**Symptoms**: `pip install` errors or import failures

**Solutions**:
- Use Python 3.11 or higher
- Upgrade pip: `python -m pip install --upgrade pip setuptools wheel`
- Use virtual environment (don't install globally)
- Clear pip cache: `pip cache purge`
- Try installing one package at a time to identify the issue

```bash
# Verify Python version
python --version

# Upgrade pip
python -m pip install --upgrade pip

# Reinstall dependencies
pip install --force-reinstall -r requirements.txt
```

### Kafka connection errors

**Symptoms**: API can't connect to Kafka

**Solutions**:
- Wait longer for Kafka to start (can take 30-60 seconds)
- Check Kafka logs: `docker compose logs kafka`
- Restart services: `docker compose restart`
- Ensure Docker has enough resources (RAM, CPU)

```bash
# Check Kafka logs
docker compose logs kafka

# Restart Kafka
docker compose restart kafka

# Wait and check again
sleep 30
docker compose ps
```

### Import errors in Python

**Symptoms**: `ModuleNotFoundError` or `ImportError`

**Solutions**:
- Ensure virtual environment is activated
- Reinstall package: `pip install -e .`
- Check PYTHONPATH is set correctly
- Verify you're in the correct directory

```bash
# Verify virtual environment
which python  # Should show path to venv

# Reinstall in development mode
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"
```

### Port already in use

**Symptoms**: "Address already in use" errors

**Solutions**:
- Stop other services using the ports
- Change ports in `docker-compose.yml`
- Find and kill process using the port

```bash
# Find process using port (macOS/Linux)
lsof -ti:8000 | xargs kill -9

# Or change ports in docker-compose.yml
# Edit the ports section to use different ports
```

## Next Steps

Now that you're set up, here's what to do next:

### 1. Complete Brick 01
- Follow the tasks in the brick's README
- Use `agentbricks check-progress` to track your progress
- Submit when complete: `agentbricks submit-brick`

### 2. Explore Documentation
- [Architecture Overview](../architecture/overview.md): Understand the system design
- [Story Arcs](../story-arcs/): Learn about different learning paths
- [Best Practices](../best-practices/): Coding standards and patterns

### 3. Join the Community
- **Discord**: [Join our Discord server](#) (link to be added)
- **GitHub Discussions**: Ask questions and share ideas
- **Issues**: Report bugs or request features

### 4. Continue Learning
- Complete all 6 bricks in the Movie Recommender arc
- Build your portfolio with completed projects
- Contribute back to the project

### 5. Advanced Usage

```bash
# Generate more data for testing
cd sim
python generate.py generate-all --users 10000 --movies 1000 --days 30

# Run tests
pytest tests/ -v

# Check code quality
black --check .
ruff check .
mypy agent/

# Review your code
agentbricks review --path .
```

## Getting Help

If you're stuck:

1. **Check the Documentation**: Most questions are answered in the docs
2. **Use the CLI**: `agentbricks ask "your question"`
3. **Get Hints**: `agentbricks hint <task_number>`
4. **Search Issues**: Check existing GitHub issues
5. **Ask the Community**: Post in Discord or GitHub Discussions

## Troubleshooting Checklist

- [ ] Python 3.11+ installed and in PATH
- [ ] Virtual environment created and activated
- [ ] All dependencies installed successfully
- [ ] Docker Desktop running (if on macOS/Windows)
- [ ] Docker daemon accessible
- [ ] Required ports (8000, 9092, 27017) are free
- [ ] Sufficient disk space (10GB+)
- [ ] Sufficient RAM (8GB+)
- [ ] Git repository cloned correctly
- [ ] Synthetic data generated successfully
- [ ] Docker services started and healthy

## What's Next?

Congratulations! You're ready to start building. Here's your learning path:

1. **Brick 01**: Data Collection Service (Event streaming, API design)
2. **Brick 02**: Feature Engineering (Feature stores, temporal correctness)
3. **Brick 03**: Model Training (NCF, training pipelines, MLflow)
4. **Brick 04**: Recommendation Service (Low-latency serving, caching)
5. **Brick 05**: Monitoring (Observability, metrics, alerting)
6. **Brick 06**: Orchestration (Airflow, end-to-end pipelines)

Each brick builds on the previous one, teaching you production ML system design step by step.

---

**Happy Building! 🧱**

For more help, see the [full documentation](../README.md) or [open an issue](https://github.com/yourusername/agentbricks/issues).
