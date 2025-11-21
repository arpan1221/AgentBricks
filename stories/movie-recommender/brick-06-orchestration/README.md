# Brick 06: Orchestration

This brick implements workflow orchestration using Apache Airflow for the movie recommender pipeline.

## Overview

The orchestration layer coordinates all components of the recommendation system:
- Data generation
- Event extraction from Kafka
- Feature engineering
- Model training (weekly)
- Model evaluation
- Model deployment (conditional)
- Notifications

## DAG: `movie_recommender_pipeline`

### Schedule
- **Frequency:** Daily at 2:00 AM UTC
- **Catchup:** Disabled (no backfills)
- **Max Active Runs:** 1

### Tasks

#### 1. `generate_synthetic_data`
- **Type:** BashOperator
- **Description:** Generates synthetic user, movie, and interaction data
- **Command:** Runs `sim/generate.py generate-all`
- **SLA:** 2 hours
- **Schedule:** Daily

#### 2. `extract_events`
- **Type:** PythonOperator
- **Description:** Extracts events from Kafka topics and stores in data lake
- **SLA:** 1 hour
- **Dependencies:** `generate_synthetic_data`

#### 3. `compute_features`
- **Type:** PythonOperator
- **Description:** Runs feature engineering pipeline
- **SLA:** 3 hours
- **Dependencies:** `extract_events`

#### 4. `train_model`
- **Type:** PythonOperator
- **Description:** Trains NCF model using features
- **SLA:** 4 hours
- **Dependencies:** `compute_features`
- **Schedule:** Weekly (executes daily but only trains on Sundays)

#### 5. `evaluate_model`
- **Type:** PythonOperator
- **Description:** Evaluates model on test set
- **SLA:** 1 hour
- **Dependencies:** `train_model`

#### 6. `deploy_model_check`
- **Type:** BranchPythonOperator
- **Description:** Checks if metrics meet deployment threshold
- **Thresholds:**
  - AUC ≥ 0.75
  - NDCG@10 ≥ 0.50
  - Hit Rate@10 ≥ 0.40
- **Branches:** `deploy_model` (if passed) or `skip_deploy` (if failed)
- **Dependencies:** `evaluate_model`

#### 7. `deploy_model`
- **Type:** PythonOperator
- **Description:** Promotes model to production in MLflow
- **SLA:** 1 hour
- **Dependencies:** `deploy_model_check` (conditional)

#### 8. `skip_deploy`
- **Type:** EmptyOperator
- **Description:** Placeholder for skipped deployment
- **Dependencies:** `deploy_model_check` (conditional)

#### 9. `alert_success`
- **Type:** SlackWebhookOperator
- **Description:** Sends success notification to Slack
- **Dependencies:** `deploy_model` OR `skip_deploy`

#### 10. `alert_failure`
- **Type:** SlackWebhookOperator
- **Description:** Sends failure notification to Slack
- **Trigger:** On any task failure

### Configuration

#### Default Arguments
```python
{
    "owner": "ml-platform-team",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=6),  # Overall DAG SLA
    "execution_timeout": timedelta(hours=8),
    "email": ["ml-platform-alerts@example.com"],
    "email_on_failure": True,
}
```

#### Environment Variables
- `KAFKA_BOOTSTRAP_SERVERS`: Kafka broker addresses (default: `localhost:9092`)
- `DATA_LAKE_PATH`: Path to data lake storage (default: `/data/events`)
- `FEATURE_STORE_PATH`: Path to feature store database (default: `/data/features/feature_store.duckdb`)
- `MLFLOW_TRACKING_URI`: MLflow tracking server URI (default: `http://localhost:5000`)

#### Airflow Connections
- `slack_default`: Slack webhook connection
  - Connection Type: HTTP
  - Host: `hooks.slack.com`
  - Extra: `{"webhook_token": "YOUR_WEBHOOK_TOKEN"}`

### Task Dependencies

```
generate_synthetic_data
    ↓
extract_events
    ↓
compute_features
    ↓
train_model (weekly)
    ↓
evaluate_model
    ↓
deploy_model_check (branch)
    ├──→ deploy_model ──┐
    └──→ skip_deploy ───┼──→ alert_success
                        │
alert_failure ←─────────┘ (on failure)
```

### Retry Logic

- **Retries:** 2 attempts per task
- **Retry Delay:** 5 minutes between retries
- **Execution Timeout:** 8 hours per task (prevents hanging tasks)

### SLA Monitoring

Each task has individual SLA monitoring:
- `generate_synthetic_data`: 2 hours
- `extract_events`: 1 hour
- `compute_features`: 3 hours
- `train_model`: 4 hours
- `evaluate_model`: 1 hour
- `deploy_model`: 1 hour

Overall DAG SLA: 6 hours

### Notifications

#### Success Notification
- **Channel:** Slack (via webhook)
- **Trigger:** When pipeline completes successfully (deploy or skip)
- **Content:** DAG ID, run ID, execution date

#### Failure Notification
- **Channel:** Slack + Email
- **Trigger:** On any task failure
- **Content:** DAG ID, run ID, execution date, failed task ID

### Usage

#### Local Development

1. Install Airflow:
   ```bash
   pip install apache-airflow==2.7.0
   pip install apache-airflow-providers-slack==7.0.0
   pip install apache-airflow-providers-email==1.0.0
   ```

2. Set environment variables:
   ```bash
   export KAFKA_BOOTSTRAP_SERVERS=localhost:9092
   export DATA_LAKE_PATH=/data/events
   export FEATURE_STORE_PATH=/data/features/feature_store.duckdb
   export MLFLOW_TRACKING_URI=http://localhost:5000
   ```

3. Initialize Airflow:
   ```bash
   airflow db init
   airflow users create --username admin --password admin --role Admin --email admin@example.com
   ```

4. Start Airflow:
   ```bash
   airflow webserver --port 8080
   airflow scheduler
   ```

5. Copy DAG to Airflow DAGs folder:
   ```bash
   cp dags/movie_recommender_pipeline.py $AIRFLOW_HOME/dags/
   ```

#### Production Deployment

1. Use Docker Compose or Kubernetes for Airflow deployment
2. Configure connections and variables in Airflow UI
3. Set up proper authentication and authorization
4. Configure external data stores (S3, GCS, etc.)
5. Set up monitoring and alerting

### Monitoring

Monitor the DAG execution:
- **Airflow UI:** http://localhost:8080
- **DAG View:** See task status, logs, XCom values
- **Tree View:** Visualize task dependencies and execution
- **Graph View:** Understand task relationships

### Troubleshooting

#### Common Issues

1. **Task Failures:**
   - Check task logs in Airflow UI
   - Verify environment variables are set correctly
   - Ensure dependencies (Kafka, MLflow) are accessible

2. **SLA Misses:**
   - Review task execution times
   - Optimize slow tasks
   - Adjust SLA thresholds if needed

3. **Deployment Failures:**
   - Check model metrics in XCom
   - Verify MLflow connectivity
   - Ensure serving layer is accessible

### Extending the Pipeline

To add new tasks:
1. Define task function or operator
2. Add task to DAG
3. Set task dependencies
4. Update SLA if needed
5. Add to success/failure notifications

### Resources

- [Apache Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Operators](https://airflow.apache.org/docs/apache-airflow/stable/operators/index.html)
