# Testing and Running the Movie Recommender Story Arc

This guide explains how to test and run the entire Movie Recommender story arc end-to-end.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Running Individual Bricks](#running-individual-bricks)
- [Running the Full Pipeline](#running-the-full-pipeline)
- [Testing the System](#testing-the-system)
- [Troubleshooting](#troubleshooting)

## Prerequisites

Before running the system, ensure you have:

1. **Python 3.11+** installed
2. **Docker & Docker Compose** installed and running
3. **8GB+ RAM** available
4. **10GB+ free disk space**
5. **All dependencies** installed (see [Quick Start Guide](../../docs/quickstart/README.md))

### Verify Prerequisites

```bash
# Check Python version
python --version  # Should be 3.11 or higher

# Check Docker
docker --version
docker compose version

# Check Docker is running
docker ps

# Verify dependencies
pip list | grep -E "(fastapi|kafka|pymongo|pytorch|mlflow|redis|duckdb)"
```

## Quick Start

### Automated Setup and Test

Use the provided script to set up and test everything:

```bash
# From project root
./scripts/test_movie_recommender.sh

# Or with Python
python scripts/test_movie_recommender.py
```

This script will:
1. Check prerequisites
2. Generate synthetic data
3. Start all infrastructure services
4. Run unit tests for each brick
5. Run integration tests
6. Run end-to-end tests
7. Display a summary report

### Manual Setup

If you prefer manual setup:

```bash
# 1. Generate synthetic data
cd sim
python generate.py generate-all --users 1000 --movies 500 --days 7
cd ..

# 2. Start infrastructure (Brick 01)
cd stories/movie-recommender/brick-01-data-collection
docker compose up -d
cd ../../..

# 3. Wait for services to be ready
sleep 30

# 4. Verify services
curl http://localhost:8000/health
```

## Running Individual Bricks

### Brick 01: Data Collection Service

**Start Services:**
```bash
cd stories/movie-recommender/brick-01-data-collection
docker compose up -d
```

**Verify:**
```bash
# Check services are running
docker compose ps

# Test API health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs
```

**Send Test Events:**
```bash
# Send a view event
curl -X POST http://localhost:8000/events/view \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "movie_id": "movie_456",
    "timestamp": "2025-01-15T10:00:00Z"
  }'

# Send a rating event
curl -X POST http://localhost:8000/events/rating \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "movie_id": "movie_456",
    "rating": 4.5,
    "timestamp": "2025-01-15T10:05:00Z"
  }'
```

**Run Tests:**
```bash
cd stories/movie-recommender/brick-01-data-collection
pytest tests/ -v
```

**Stop Services:**
```bash
docker compose down
```

### Brick 02: Feature Engineering

**Run Feature Pipeline:**
```bash
cd stories/movie-recommender/brick-02-feature-engineering

# Set up feature store
python -c "from src.feature_store import FeatureStore; fs = FeatureStore(); print('Feature store initialized')"

# Run feature computation (example)
python -c "
from src.feature_pipeline import FeatureEngineer
fe = FeatureEngineer()
# Add your feature computation logic here
"
```

**Run Tests:**
```bash
pytest tests/ -v
```

### Brick 03: Model Training

**Train Model:**
```bash
cd stories/movie-recommender/brick-03-model-training

# Start MLflow (if not already running)
mlflow ui --port 5000 &

# Train model
python src/train.py \
  --config config/config.yaml \
  --output-dir ./models

# Check MLflow UI
open http://localhost:5000
```

**Run Tests:**
```bash
pytest tests/ -v  # If tests exist
```

### Brick 04: Recommendation Service

**Start Services:**
```bash
cd stories/movie-recommender/brick-04-recommendation-service
docker compose up -d
```

**Verify:**
```bash
# Check services
docker compose ps

# Test health endpoint
curl http://localhost:8001/health

# Get recommendations
curl -X POST http://localhost:8001/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10
  }'
```

**Run Tests:**
```bash
pytest tests/ -v  # If tests exist
```

**Stop Services:**
```bash
docker compose down
```

### Brick 05: Monitoring

**Start Prometheus and Grafana:**
```bash
cd stories/movie-recommender/brick-05-monitoring

# Start Prometheus (if docker-compose exists)
docker compose up -d

# Or run Prometheus directly
prometheus --config.file=prometheus/prometheus.yml
```

**View Dashboards:**
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (if configured)

**Run Tests:**
```bash
pytest tests/ -v  # If tests exist
```

### Brick 06: Orchestration

**Setup Airflow:**
```bash
cd stories/movie-recommender/brick-06-orchestration

# Initialize Airflow (first time only)
export AIRFLOW_HOME=$(pwd)/airflow_home
airflow db init
airflow users create \
  --username admin \
  --password admin \
  --role Admin \
  --email admin@example.com

# Start Airflow
airflow webserver --port 8080 &
airflow scheduler &
```

**Access Airflow UI:**
- Open http://localhost:8080
- Login with username: `admin`, password: `admin`
- Find the `movie_recommender_pipeline` DAG
- Trigger a manual run

**Run Tests:**
```bash
pytest tests/ -v  # If tests exist
```

## Running the Full Pipeline

### Option 1: Using the Orchestration Script

```bash
# From project root
python scripts/run_full_pipeline.py
```

This script will:
1. Generate synthetic data
2. Start all infrastructure
3. Run feature engineering
4. Train the model
5. Start the recommendation service
6. Run end-to-end validation

### Option 2: Manual Step-by-Step

#### Step 1: Generate Data
```bash
cd sim
python generate.py generate-all --users 10000 --movies 5000 --days 30
cd ..
```

#### Step 2: Start Infrastructure (Brick 01)
```bash
cd stories/movie-recommender/brick-01-data-collection
docker compose up -d
cd ../../..
```

Wait for services to be ready:
```bash
# Wait for Kafka
timeout 60 bash -c 'until docker exec agentbricks-kafka kafka-broker-api-versions --bootstrap-server localhost:9092; do sleep 2; done'

# Wait for MongoDB
timeout 60 bash -c 'until docker exec agentbricks-mongodb mongosh --eval "db.adminCommand(\"ping\")"; do sleep 2; done'

# Wait for API
timeout 60 bash -c 'until curl -f http://localhost:8000/health; do sleep 2; done'
```

#### Step 3: Ingest Events
```bash
# Send events to the API (or use the simulation)
cd sim
python generate.py generate-interactions --count 10000
cd ..
```

#### Step 4: Extract Events from Kafka
```bash
# Use Kafka consumer to extract events
# Or use the orchestration DAG's extract_events task
```

#### Step 5: Compute Features (Brick 02)
```bash
cd stories/movie-recommender/brick-02-feature-engineering
python -m src.feature_pipeline
cd ../../..
```

#### Step 6: Train Model (Brick 03)
```bash
cd stories/movie-recommender/brick-03-model-training

# Start MLflow
mlflow ui --port 5000 &

# Train model
python src/train.py --config config/config.yaml

cd ../../..
```

#### Step 7: Start Recommendation Service (Brick 04)
```bash
cd stories/movie-recommender/brick-04-recommendation-service
docker compose up -d
cd ../../..
```

#### Step 8: Verify End-to-End
```bash
# Get recommendations
curl -X POST http://localhost:8001/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_123",
    "num_recommendations": 10
  }'
```

### Option 3: Using Airflow DAG (Brick 06)

If you have Airflow set up:

1. Start Airflow (see Brick 06 section above)
2. Open Airflow UI: http://localhost:8080
3. Find `movie_recommender_pipeline` DAG
4. Trigger a manual run
5. Monitor task execution

## Testing the System

### Unit Tests

Run unit tests for each brick:

```bash
# Brick 01
cd stories/movie-recommender/brick-01-data-collection
pytest tests/ -v
cd ../../..

# Brick 02
cd stories/movie-recommender/brick-02-feature-engineering
pytest tests/ -v
cd ../../..

# All bricks
pytest stories/movie-recommender/*/tests/ -v
```

### Integration Tests

Run integration tests that test component interactions:

```bash
# From project root
pytest tests/integration/ -v
```

### End-to-End Tests

Run end-to-end tests that validate the complete flow:

```bash
# From project root
pytest tests/e2e/ -v

# Or use the test script
./scripts/test_movie_recommender.sh --e2e-only
```

### Performance Tests

Test system performance:

```bash
# Load test the API
cd stories/movie-recommender/brick-01-data-collection
python -m pytest tests/ -v -k "performance" --benchmark-only

# Or use a load testing tool
# Install: pip install locust
locust -f tests/load_test.py --host=http://localhost:8000
```

### Test Coverage

Generate coverage report:

```bash
# Install coverage tool
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# View report
open htmlcov/index.html
```

## Troubleshooting

### Services Not Starting

**Issue:** Docker services fail to start

**Solutions:**
```bash
# Check Docker is running
docker ps

# Check port conflicts
lsof -i :8000  # API
lsof -i :9092  # Kafka
lsof -i :27017  # MongoDB
lsof -i :6379   # Redis

# Check Docker logs
docker compose logs

# Restart services
docker compose down
docker compose up -d
```

### Kafka Connection Errors

**Issue:** Cannot connect to Kafka

**Solutions:**
```bash
# Wait longer for Kafka to start (can take 60+ seconds)
sleep 60

# Check Kafka logs
docker compose logs kafka

# Verify Kafka is ready
docker exec agentbricks-kafka kafka-broker-api-versions --bootstrap-server localhost:9092

# Restart Kafka
docker compose restart kafka
```

### MongoDB Connection Errors

**Issue:** Cannot connect to MongoDB

**Solutions:**
```bash
# Check MongoDB logs
docker compose logs mongodb

# Verify MongoDB is ready
docker exec agentbricks-mongodb mongosh --eval "db.adminCommand('ping')"

# Check credentials
# Default: username=admin, password=password
```

### Model Training Fails

**Issue:** Model training errors

**Solutions:**
```bash
# Check feature store exists
ls -la stories/movie-recommender/brick-02-feature-engineering/data/

# Verify MLflow is running
curl http://localhost:5000/health

# Check training logs
tail -f stories/movie-recommender/brick-03-model-training/logs/training.log
```

### Recommendation Service Errors

**Issue:** Recommendations fail

**Solutions:**
```bash
# Check model file exists
ls -la stories/movie-recommender/brick-04-recommendation-service/models/

# Verify Redis is running
docker exec movie-recommender-redis redis-cli ping

# Check service logs
docker compose logs recommendation-service
```

### Port Conflicts

**Issue:** Port already in use

**Solutions:**
```bash
# Find process using port
lsof -ti:8000 | xargs kill -9  # API
lsof -ti:9092 | xargs kill -9  # Kafka
lsof -ti:27017 | xargs kill -9  # MongoDB

# Or change ports in docker-compose.yml
```

### Out of Memory

**Issue:** System runs out of memory

**Solutions:**
```bash
# Reduce data size
python sim/generate.py generate-all --users 1000 --movies 500 --days 7

# Increase Docker memory limit
# Docker Desktop: Settings > Resources > Memory

# Stop unused services
docker compose down
```

## Cleanup

### Stop All Services

```bash
# Stop Brick 01 services
cd stories/movie-recommender/brick-01-data-collection
docker compose down -v
cd ../../..

# Stop Brick 04 services
cd stories/movie-recommender/brick-04-recommendation-service
docker compose down -v
cd ../../..

# Stop Brick 05 services (if any)
cd stories/movie-recommender/brick-05-monitoring
docker compose down -v
cd ../../..

# Stop Airflow (Brick 06)
pkill -f "airflow webserver"
pkill -f "airflow scheduler"
```

### Remove All Data

```bash
# Remove Docker volumes
docker volume prune -f

# Remove generated data
rm -rf sim/data/*
rm -rf stories/movie-recommender/*/data/*
rm -rf stories/movie-recommender/*/models/*
rm -rf stories/movie-recommender/*/logs/*
```

## Next Steps

After successfully running and testing the system:

1. **Explore the Code**: Review each brick's implementation
2. **Modify Components**: Try changing features, models, or configurations
3. **Add Features**: Extend the system with new capabilities
4. **Scale Up**: Test with larger datasets
5. **Deploy**: Use the infrastructure code to deploy to cloud

## Additional Resources

- [Quick Start Guide](../../docs/quickstart/README.md)
- [Architecture Overview](../../docs/architecture/overview.md)
- [Contributing Guide](../../CONTRIBUTING.md)
- [Brick 01 README](brick-01-data-collection/README.md)
- [Brick 06 Orchestration README](brick-06-orchestration/README.md)

---

**Happy Testing! 🧪**
