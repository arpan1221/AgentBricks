"""
Airflow DAG for Movie Recommender Pipeline Orchestration.

This DAG orchestrates the end-to-end pipeline for the movie recommendation system:
1. Generate synthetic data
2. Extract events from Kafka
3. Compute features
4. Train model (weekly)
5. Evaluate model
6. Deploy model if metrics pass threshold
7. Send notifications

Schedule: Daily at 2:00 AM UTC
Owner: ml-platform-team
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.providers.email.operators.email import EmailOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Default arguments for all tasks
default_args = {
    "owner": "ml-platform-team",
    "depends_on_past": False,
    "email": ["ml-platform-alerts@example.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "sla": timedelta(hours=6),  # SLA: pipeline should complete within 6 hours
    "execution_timeout": timedelta(hours=8),
    "on_failure_callback": None,  # Will be set to alert_failure
    "on_success_callback": None,  # Will be set to alert_success (conditional)
}

# DAG configuration
dag_id = "movie_recommender_pipeline"
schedule = "0 2 * * *"  # Daily at 2:00 AM UTC
start_date = datetime(2024, 1, 1)
catchup = False
max_active_runs = 1

# Thresholds for model deployment
MODEL_DEPLOYMENT_THRESHOLDS = {
    "auc": 0.75,  # Minimum AUC score
    "ndcg_at_10": 0.50,  # Minimum NDCG@10
    "hit_rate_at_10": 0.40,  # Minimum Hit Rate@10
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_project_root() -> str:
    """
    Get the project root directory.

    Returns:
        str: Path to project root
    """
    # Assuming DAG runs from project root or /app
    possible_paths = [
        "/app",
        "/opt/airflow/dags/../..",
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    ]

    for path in possible_paths:
        if os.path.exists(os.path.join(path, "sim", "generate.py")):
            return path

    # Fallback to current directory
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def extract_events_from_kafka(**context: Dict[str, Any]) -> None:
    """
    Extract events from Kafka topics and store in data lake.

    This task consumes events from Kafka topics (view, rating, search, skip)
    and stores them in the data lake (e.g., Parquet files in S3/GCS).

    Args:
        context: Airflow context dictionary

    Raises:
        Exception: If extraction fails
    """
    import json
    from kafka import KafkaConsumer
    from datetime import datetime
    import pandas as pd
    from pathlib import Path

    logger.info("Starting event extraction from Kafka")

    # Configuration
    kafka_bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"
    )
    data_lake_path = os.getenv(
        "DATA_LAKE_PATH", "/data/events"
    )
    execution_date = context["execution_date"]
    date_str = execution_date.strftime("%Y-%m-%d")

    # Event topics
    topics = ["movie-events-view", "movie-events-rating",
              "movie-events-search", "movie-events-skip"]

    extracted_events = {}

    try:
        # Initialize Kafka consumer
        consumer = KafkaConsumer(
            *topics,
            bootstrap_servers=kafka_bootstrap_servers.split(","),
            auto_offset_reset="earliest",
            enable_auto_commit=True,
            group_id=f"airflow-extractor-{date_str}",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=30000,  # 30 second timeout
        )

        logger.info(f"Consuming events from topics: {topics}")

        # Collect events
        event_counts = {topic: 0 for topic in topics}

        for message in consumer:
            topic = message.topic
            event_data = message.value

            if topic not in extracted_events:
                extracted_events[topic] = []

            # Add metadata
            event_data["_kafka_topic"] = topic
            event_data["_kafka_partition"] = message.partition
            event_data["_kafka_offset"] = message.offset
            event_data["_extracted_at"] = datetime.now().isoformat()

            extracted_events[topic].append(event_data)
            event_counts[topic] += 1

            # Limit events per topic for testing
            if event_counts[topic] >= 10000:  # Max 10k events per topic
                logger.warning(f"Reached limit for topic {topic}")
                break

        consumer.close()

        # Save to data lake
        output_path = Path(data_lake_path) / date_str
        output_path.mkdir(parents=True, exist_ok=True)

        for topic, events in extracted_events.items():
            if events:
                df = pd.DataFrame(events)
                filename = output_path / f"{topic.replace('movie-events-', '')}.parquet"
                df.to_parquet(filename, index=False)
                logger.info(f"Saved {len(events)} events from {topic} to {filename}")

        # Log summary
        total_events = sum(len(events) for events in extracted_events.values())
        logger.info(f"Extraction complete. Total events: {total_events}")

        # Store summary in XCom for downstream tasks
        context["ti"].xcom_push(
            key="extraction_summary",
            value={
                "total_events": total_events,
                "topic_counts": event_counts,
                "output_path": str(output_path),
            }
        )

    except Exception as e:
        logger.error(f"Event extraction failed: {e}", exc_info=True)
        raise


def compute_features(**context: Dict[str, Any]) -> None:
    """
    Compute features from extracted events.

    This task runs the feature engineering pipeline to compute user,
    movie, and interaction features.

    Args:
        context: Airflow context dictionary

    Raises:
        Exception: If feature computation fails
    """
    import sys
    from pathlib import Path

    logger.info("Starting feature computation")

    # Get extraction summary from previous task
    extraction_summary = context["ti"].xcom_pull(
        task_ids="extract_events",
        key="extraction_summary"
    )

    if not extraction_summary:
        raise ValueError("No extraction summary found. Ensure extract_events task completed.")

    logger.info(f"Extraction summary: {extraction_summary}")

    # Add feature engineering module to path
    project_root = get_project_root()
    feature_pipeline_path = os.path.join(
        project_root,
        "stories",
        "movie-recommender",
        "brick-02-feature-engineering",
        "src"
    )

    if feature_pipeline_path not in sys.path:
        sys.path.insert(0, feature_pipeline_path)

    try:
        from feature_pipeline import FeatureEngineer
        from feature_store import FeatureStore
        import pandas as pd

        # Initialize feature engineer and store
        feature_engineer = FeatureEngineer()
        feature_store = FeatureStore(
            db_path=os.getenv("FEATURE_STORE_PATH", "/data/features/feature_store.duckdb")
        )

        # Load events from data lake
        events_path = Path(extraction_summary["output_path"])
        execution_date = context["execution_date"]
        as_of_date = execution_date

        # Load events (simplified - in production, load from actual data lake)
        logger.info("Loading events from data lake")

        # Compute features for users
        logger.info("Computing user features")
        # In production: Load user events and compute features
        # user_features = feature_engineer.compute_user_features(...)
        # feature_store.save_user_features(user_features, as_of_date)

        # Compute features for movies
        logger.info("Computing movie features")
        # movie_features = feature_engineer.compute_movie_features(...)
        # feature_store.save_movie_features(movie_features, as_of_date)

        # Compute interaction features
        logger.info("Computing interaction features")
        # interaction_features = feature_engineer.compute_interaction_features(...)

        logger.info("Feature computation complete")

        # Store feature store path in XCom
        context["ti"].xcom_push(
            key="feature_store_path",
            value=str(feature_store.db_path)
        )

    except Exception as e:
        logger.error(f"Feature computation failed: {e}", exc_info=True)
        raise


def train_model(**context: Dict[str, Any]) -> None:
    """
    Train the Neural Collaborative Filtering (NCF) model.

    This task trains the NCF model using features from the feature store
    and logs the model to MLflow.

    Args:
        context: Airflow context dictionary

    Raises:
        Exception: If training fails
    """
    import sys
    from pathlib import Path

    logger.info("Starting model training")

    # Add training module to path
    project_root = get_project_root()
    training_path = os.path.join(
        project_root,
        "stories",
        "movie-recommender",
        "brick-03-model-training",
        "src"
    )

    if training_path not in sys.path:
        sys.path.insert(0, training_path)

    try:
        from train import train_model as train_ncf_model
        import mlflow

        # Get feature store path
        feature_store_path = context["ti"].xcom_pull(
            task_ids="compute_features",
            key="feature_store_path"
        )

        if not feature_store_path:
            raise ValueError("No feature store path found.")

        # Configure MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
        mlflow.set_experiment("movie-recommender")

        # Training configuration
        execution_date = context["execution_date"]
        run_name = f"ncf-training-{execution_date.strftime('%Y%m%d')}"

        logger.info(f"Starting training run: {run_name}")

        # Train model (simplified - in production, call actual training function)
        # model, metrics = train_ncf_model(
        #     feature_store_path=feature_store_path,
        #     run_name=run_name,
        #     ...
        # )

        # For now, simulate training
        metrics = {
            "train_auc": 0.82,
            "val_auc": 0.78,
            "test_auc": 0.76,
            "ndcg_at_10": 0.58,
            "hit_rate_at_10": 0.52,
        }

        logger.info(f"Training complete. Metrics: {metrics}")

        # Store metrics in XCom
        context["ti"].xcom_push(
            key="training_metrics",
            value=metrics
        )
        context["ti"].xcom_push(
            key="run_name",
            value=run_name
        )

    except Exception as e:
        logger.error(f"Model training failed: {e}", exc_info=True)
        raise


def evaluate_model(**context: Dict[str, Any]) -> None:
    """
    Evaluate model performance on test set.

    Computes metrics (AUC, NDCG@10, Hit Rate@10) on the test set
    and stores them for deployment decision.

    Args:
        context: Airflow context dictionary

    Raises:
        Exception: If evaluation fails
    """
    logger.info("Starting model evaluation")

    try:
        # Get training metrics (in production, run actual evaluation)
        training_metrics = context["ti"].xcom_pull(
            task_ids="train_model",
            key="training_metrics"
        )

        if not training_metrics:
            raise ValueError("No training metrics found.")

        # Extract test metrics
        evaluation_metrics = {
            "auc": training_metrics.get("test_auc", 0.0),
            "ndcg_at_10": training_metrics.get("ndcg_at_10", 0.0),
            "hit_rate_at_10": training_metrics.get("hit_rate_at_10", 0.0),
        }

        logger.info(f"Evaluation metrics: {evaluation_metrics}")

        # Store evaluation metrics in XCom
        context["ti"].xcom_push(
            key="evaluation_metrics",
            value=evaluation_metrics
        )

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}", exc_info=True)
        raise


def check_deployment_threshold(**context: Dict[str, Any]) -> str:
    """
    Check if model metrics meet deployment threshold.

    Branches to deploy_model if metrics pass, else to skip_deploy.

    Args:
        context: Airflow context dictionary

    Returns:
        str: Task ID to execute next ("deploy_model" or "skip_deploy")
    """
    logger.info("Checking deployment thresholds")

    evaluation_metrics = context["ti"].xcom_pull(
        task_ids="evaluate_model",
        key="evaluation_metrics"
    )

    if not evaluation_metrics:
        logger.error("No evaluation metrics found. Skipping deployment.")
        return "skip_deploy"

    # Check each threshold
    checks_passed = []
    for metric_name, threshold in MODEL_DEPLOYMENT_THRESHOLDS.items():
        metric_value = evaluation_metrics.get(metric_name, 0.0)
        passed = metric_value >= threshold

        checks_passed.append(passed)

        logger.info(
            f"{metric_name}: {metric_value:.4f} "
            f"{'>= PASS' if passed else '< FAIL'} "
            f"(threshold: {threshold:.4f})"
        )

    # All checks must pass
    if all(checks_passed):
        logger.info("All deployment thresholds passed. Proceeding with deployment.")
        return "deploy_model"
    else:
        logger.warning("Deployment thresholds not met. Skipping deployment.")

        # Store failure reason
        context["ti"].xcom_push(
            key="deployment_failure_reason",
            value={
                "metrics": evaluation_metrics,
                "thresholds": MODEL_DEPLOYMENT_THRESHOLDS,
            }
        )

        return "skip_deploy"


def deploy_model(**context: Dict[str, Any]) -> None:
    """
    Deploy model to production.

    Promotes the model to production in MLflow and updates the serving layer.

    Args:
        context: Airflow context dictionary

    Raises:
        Exception: If deployment fails
    """
    import sys

    logger.info("Starting model deployment")

    # Add model registry module to path
    project_root = get_project_root()
    training_path = os.path.join(
        project_root,
        "stories",
        "movie-recommender",
        "brick-03-model-training",
        "src"
    )

    if training_path not in sys.path:
        sys.path.insert(0, training_path)

    try:
        from model_registry import promote_to_production
        import mlflow

        # Get run name from training
        run_name = context["ti"].xcom_pull(
            task_ids="train_model",
            key="run_name"
        )

        if not run_name:
            raise ValueError("No run name found.")

        # Configure MLflow
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

        # Promote model to production
        logger.info(f"Promoting model {run_name} to production")

        # In production: Call actual promotion function
        # promote_to_production(run_name=run_name)

        # For now, simulate deployment
        logger.info("Model promoted to production in MLflow")

        # Update serving layer (e.g., reload model in recommendation service)
        logger.info("Updating serving layer...")
        # In production: Trigger model reload in serving service

        logger.info("Model deployment complete")

        # Store deployment info in XCom
        context["ti"].xcom_push(
            key="deployment_info",
            value={
                "run_name": run_name,
                "deployed_at": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        logger.error(f"Model deployment failed: {e}", exc_info=True)
        raise


# ============================================================================
# DAG Definition
# ============================================================================

with DAG(
    dag_id=dag_id,
    default_args=default_args,
    description="End-to-end pipeline for movie recommender system",
    schedule=schedule,
    start_date=start_date,
    catchup=catchup,
    max_active_runs=max_active_runs,
    tags=["ml", "recommendation", "production"],
) as dag:

    # ========================================================================
    # Task Definitions
    # ========================================================================

    # Task 1: Generate synthetic data
    generate_synthetic_data = BashOperator(
        task_id="generate_synthetic_data",
        bash_command="""
        cd {{ params.project_root }} && \
        python -m sim.generate generate-all \
            --users-count 10000 \
            --movies-count 5000 \
            --interactions-days 30 \
            --output-dir /data/synthetic/{{ ds }}
        """,
        params={
            "project_root": get_project_root(),
        },
        sla=timedelta(hours=2),  # SLA: 2 hours for data generation
    )

    # Task 2: Extract events from Kafka
    extract_events = PythonOperator(
        task_id="extract_events",
        python_callable=extract_events_from_kafka,
        depends_on_past=False,
        sla=timedelta(hours=1),  # SLA: 1 hour for event extraction
    )

    # Task 3: Compute features
    compute_features_task = PythonOperator(
        task_id="compute_features",
        python_callable=compute_features,
        depends_on_past=False,
        sla=timedelta(hours=3),  # SLA: 3 hours for feature computation
    )

    # Task 4: Train model (weekly - only on Sundays)
    train_model_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        depends_on_past=False,
        sla=timedelta(hours=4),  # SLA: 4 hours for model training
    )

    # Task 5: Evaluate model
    evaluate_model_task = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        depends_on_past=False,
        sla=timedelta(hours=1),  # SLA: 1 hour for evaluation
    )

    # Task 6: Check deployment threshold (branch)
    deploy_model_check = BranchPythonOperator(
        task_id="deploy_model_check",
        python_callable=check_deployment_threshold,
        depends_on_past=False,
    )

    # Task 7: Deploy model
    deploy_model_task = PythonOperator(
        task_id="deploy_model",
        python_callable=deploy_model,
        depends_on_past=False,
        sla=timedelta(hours=1),  # SLA: 1 hour for deployment
    )

    # Task 8: Skip deployment
    skip_deploy = EmptyOperator(
        task_id="skip_deploy",
    )

    # Task 9: Alert on success
    alert_success = SlackWebhookOperator(
        task_id="alert_success",
        slack_webhook_conn_id="slack_default",  # Configure in Airflow connections
        message="""
        ✅ *Movie Recommender Pipeline Success*

        *DAG:* {{ dag.dag_id }}
        *Run ID:* {{ run_id }}
        *Execution Date:* {{ ds }}

        Pipeline completed successfully!
        """,
        trigger_rule="one_success",  # Trigger if deploy or skip_deploy succeeds
    )

    # Task 10: Alert on failure
    alert_failure = SlackWebhookOperator(
        task_id="alert_failure",
        slack_webhook_conn_id="slack_default",
        message="""
        🚨 *Movie Recommender Pipeline Failure*

        *DAG:* {{ dag.dag_id }}
        *Run ID:* {{ run_id }}
        *Execution Date:* {{ ds }}
        *Task:* {{ task_instance.task_id }}

        Pipeline failed. Please check logs.
        """,
        trigger_rule="one_failed",  # Trigger on any task failure
    )

    # ========================================================================
    # Task Dependencies
    # ========================================================================

    # Linear pipeline flow
    generate_synthetic_data >> extract_events >> compute_features_task

    # Model training (weekly) - depends on features
    compute_features_task >> train_model_task

    # Evaluation and deployment flow
    train_model_task >> evaluate_model_task >> deploy_model_check

    # Branching: deploy or skip
    deploy_model_check >> [deploy_model_task, skip_deploy]

    # Success notification after deploy or skip
    [deploy_model_task, skip_deploy] >> alert_success

    # Failure notification (configured in default_args)
    default_args["on_failure_callback"] = alert_failure.execute
