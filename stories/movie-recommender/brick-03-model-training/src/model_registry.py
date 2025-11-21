"""MLflow model registry for NCF model versioning and management.

This module provides functions for managing model lifecycle with MLflow,
including model registration, versioning, promotion to production, and
model comparison.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import scipy.stats as stats

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelRegistryError(Exception):
    """Custom exception for model registry errors."""
    pass


def init_mlflow(
    tracking_uri: str,
    experiment_name: str
) -> str:
    """
    Initialize MLflow tracking and create experiment if needed.

    Sets up MLflow tracking URI and creates experiment if it doesn't exist.

    Args:
        tracking_uri: MLflow tracking URI (e.g., "file:./mlruns" or "http://localhost:5000")
        experiment_name: Name of the experiment to create or use

    Returns:
        Experiment ID as string

    Raises:
        ModelRegistryError: If MLflow initialization fails

    Example:
        >>> experiment_id = init_mlflow("file:./mlruns", "ncf_recommender")
        >>> print(f"Experiment ID: {experiment_id}")
    """
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")

        # Get or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                # Create new experiment
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        except MlflowException as e:
            logger.error(f"Failed to get/create experiment: {e}", exc_info=True)
            raise ModelRegistryError(f"Failed to initialize experiment: {e}") from e

        # Set active experiment
        mlflow.set_experiment(experiment_name)

        return experiment_id

    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {e}", exc_info=True)
        raise ModelRegistryError(f"MLflow initialization failed: {e}") from e


def log_model(
    model: torch.nn.Module,
    metrics: Dict[str, float],
    hyperparams: Dict[str, Any],
    model_name: str,
    run_id: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
    artifacts_dir: Optional[str] = None
) -> str:
    """
    Log PyTorch model to MLflow with metrics and hyperparameters.

    Registers model in MLflow model registry with versioning, tags, and metadata.

    Args:
        model: PyTorch model to log
        metrics: Dictionary of evaluation metrics (e.g., {"auc": 0.85, "ndcg_at_10": 0.72})
        hyperparams: Dictionary of hyperparameters used for training
        model_name: Name for the registered model
        run_id: Optional MLflow run ID (if None, uses current active run)
        tags: Optional dictionary of tags to add to model
        artifacts_dir: Optional directory with additional artifacts to log

    Returns:
        Model version as string

    Raises:
        ModelRegistryError: If logging fails

    Example:
        >>> model = NCF(num_users=1000, num_items=500)
        >>> metrics = {"auc": 0.85, "ndcg_at_10": 0.72}
        >>> hyperparams = {"embedding_dim": 64, "lr": 0.001}
        >>> version = log_model(model, metrics, hyperparams, "ncf_v1")
        >>> print(f"Model version: {version}")
    """
    try:
        # Start run if not already active
        active_run = mlflow.active_run()
        if run_id is None and active_run is None:
            mlflow.start_run()
            active_run = mlflow.active_run()
            close_run = True
        else:
            close_run = False

        try:
            # Log hyperparameters
            if hyperparams:
                # Convert lists/dicts to strings for MLflow
                processed_hyperparams = {}
                for key, value in hyperparams.items():
                    if isinstance(value, (list, dict)):
                        processed_hyperparams[key] = str(value)
                    else:
                        processed_hyperparams[key] = value
                mlflow.log_params(processed_hyperparams)
                logger.debug(f"Logged {len(hyperparams)} hyperparameters")

            # Log metrics
            if metrics:
                mlflow.log_metrics(metrics)
                logger.debug(f"Logged {len(metrics)} metrics")

            # Add tags
            default_tags = {
                "timestamp": datetime.now().isoformat(),
                "model_type": "ncf",
                "framework": "pytorch"
            }
            if tags:
                default_tags.update(tags)

            for key, value in default_tags.items():
                mlflow.set_tag(key, value)

            # Log PyTorch model
            mlflow.pytorch.log_model(
                pytorch_model=model,
                artifact_path="model",
                registered_model_name=model_name
            )
            logger.info(f"Logged PyTorch model: {model_name}")

            # Log additional artifacts if provided
            if artifacts_dir and Path(artifacts_dir).exists():
                mlflow.log_artifacts(artifacts_dir)
                logger.debug(f"Logged artifacts from: {artifacts_dir}")

            # Get model version from registry
            client = MlflowClient()
            latest_version = client.get_latest_versions(model_name, stages=["None"])

            if latest_version:
                model_version = latest_version[0].version
                logger.info(f"Model registered as version {model_version}")
            else:
                # If model doesn't exist in registry, it will be created
                model_version = "1"
                logger.info(f"Created new model in registry: {model_name}")

            # Add metadata to model version
            if latest_version:
                client.set_model_version_tag(
                    name=model_name,
                    version=latest_version[0].version,
                    key="registered_at",
                    value=datetime.now().isoformat()
                )

            return model_version

        finally:
            if close_run:
                mlflow.end_run()

    except MlflowException as e:
        logger.error(f"MLflow error during model logging: {e}", exc_info=True)
        raise ModelRegistryError(f"Failed to log model: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during model logging: {e}", exc_info=True)
        raise ModelRegistryError(f"Model logging failed: {e}") from e


def load_latest_model(
    model_name: str,
    stage: Optional[str] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Load latest model version from MLflow model registry.

    Fetches the latest version of a registered model and loads it into memory.

    Args:
        model_name: Name of the registered model
        stage: Model stage to load from (e.g., "Staging", "Production", "None").
               If None, loads latest version regardless of stage.
        device: Device to load model on. If None, uses model's original device.

    Returns:
        Tuple of (loaded_model, model_info_dict) where model_info contains:
            - version: Model version
            - stage: Model stage
            - run_id: MLflow run ID
            - metrics: Model metrics
            - hyperparams: Model hyperparameters

    Raises:
        ModelRegistryError: If model not found or loading fails

    Example:
        >>> model, info = load_latest_model("ncf_v1", stage="Production")
        >>> print(f"Loaded model version {info['version']}")
        >>> print(f"AUC: {info['metrics']['auc']}")
    """
    try:
        client = MlflowClient()

        # Get model version
        if stage:
            versions = client.get_latest_versions(model_name, stages=[stage])
            if not versions:
                raise ModelRegistryError(
                    f"No model found with name '{model_name}' in stage '{stage}'"
                )
            model_version = versions[0]
        else:
            # Get latest version regardless of stage
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                raise ModelRegistryError(f"No model found with name '{model_name}'")
            # Sort by version number and get latest
            versions = sorted(versions, key=lambda v: int(v.version), reverse=True)
            model_version = versions[0]

        # Get model URI
        model_uri = f"models:/{model_name}/{model_version.version}"

        # Load model
        try:
            model = mlflow.pytorch.load_model(model_uri, map_location=device)
            logger.info(f"Loaded model {model_name} version {model_version.version}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_uri}: {e}", exc_info=True)
            raise ModelRegistryError(f"Model loading failed: {e}") from e

        # Get model metadata
        model_info = {
            'version': model_version.version,
            'stage': model_version.current_stage,
            'run_id': model_version.run_id,
            'created_at': model_version.creation_timestamp,
            'description': model_version.description
        }

        # Get run information
        try:
            run = mlflow.get_run(model_version.run_id)
            model_info['metrics'] = run.data.metrics
            model_info['hyperparams'] = run.data.params
            model_info['tags'] = run.data.tags
        except Exception as e:
            logger.warning(f"Failed to get run info: {e}")
            model_info['metrics'] = {}
            model_info['hyperparams'] = {}
            model_info['tags'] = {}

        logger.debug(
            f"Loaded model metadata",
            extra={
                "model_name": model_name,
                "version": model_version.version,
                "stage": model_version.current_stage
            }
        )

        return model, model_info

    except ModelRegistryError:
        raise
    except MlflowException as e:
        logger.error(f"MLflow error during model loading: {e}", exc_info=True)
        raise ModelRegistryError(f"Failed to load model: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during model loading: {e}", exc_info=True)
        raise ModelRegistryError(f"Model loading failed: {e}") from e


def promote_to_production(
    model_name: str,
    version: str,
    archive_previous: bool = True
) -> None:
    """
    Promote model version to production stage.

    Transitions model to "Production" stage and optionally archives previous
    production models to "Archived" stage.

    Args:
        model_name: Name of the registered model
        version: Model version to promote
        archive_previous: Whether to archive previous production models (default: True)

    Raises:
        ModelRegistryError: If promotion fails

    Example:
        >>> promote_to_production("ncf_v1", version="5", archive_previous=True)
        >>> # Model version 5 is now in Production, version 4 is Archived
    """
    try:
        client = MlflowClient()

        # Verify model version exists
        try:
            model_version = client.get_model_version(model_name, version)
        except MlflowException as e:
            raise ModelRegistryError(
                f"Model version {version} not found for model '{model_name}': {e}"
            ) from e

        # Archive previous production models if requested
        if archive_previous:
            try:
                production_versions = client.get_latest_versions(
                    model_name,
                    stages=["Production"]
                )

                for prod_version in production_versions:
                    if prod_version.version != version:
                        client.transition_model_version_stage(
                            name=model_name,
                            version=prod_version.version,
                            stage="Archived"
                        )
                        logger.info(
                            f"Archived previous production model version {prod_version.version}"
                        )
            except MlflowException as e:
                logger.warning(f"Failed to archive previous production models: {e}")

        # Promote to production
        try:
            client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production"
            )
            logger.info(
                f"Promoted model {model_name} version {version} to Production",
                extra={
                    "model_name": model_name,
                    "version": version,
                    "stage": "Production"
                }
            )
        except MlflowException as e:
            logger.error(f"Failed to promote model to production: {e}", exc_info=True)
            raise ModelRegistryError(f"Failed to promote model: {e}") from e

    except ModelRegistryError:
        raise
    except MlflowException as e:
        logger.error(f"MLflow error during promotion: {e}", exc_info=True)
        raise ModelRegistryError(f"Model promotion failed: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during promotion: {e}", exc_info=True)
        raise ModelRegistryError(f"Model promotion failed: {e}") from e


def compare_models(
    model_name: str,
    version_1: str,
    version_2: str,
    test_data: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None
) -> pd.DataFrame:
    """
    Compare two model versions side-by-side.

    Compares metrics, hyperparameters, and optionally performance on test data
    with statistical significance tests.

    Args:
        model_name: Name of the registered model
        version_1: First model version to compare
        version_2: Second model version to compare
        test_data: Optional tuple of (user_ids, item_ids, labels) for
                   performance comparison. If None, only compares logged metrics.

    Returns:
        DataFrame with comparison results including:
            - Metric names
            - Version 1 values
            - Version 2 values
            - Difference
            - Statistical significance (if test_data provided)

    Raises:
        ModelRegistryError: If comparison fails

    Example:
        >>> comparison_df = compare_models(
        ...     "ncf_v1",
        ...     version_1="4",
        ...     version_2="5",
        ...     test_data=(user_ids, item_ids, labels)
        ... )
        >>> print(comparison_df)
    """
    try:
        client = MlflowClient()

        # Load model metadata for both versions
        try:
            model_v1 = client.get_model_version(model_name, version_1)
            model_v2 = client.get_model_version(model_name, version_2)
        except MlflowException as e:
            raise ModelRegistryError(f"Failed to get model versions: {e}") from e

        # Get run information
        try:
            run_v1 = mlflow.get_run(model_v1.run_id)
            run_v2 = mlflow.get_run(model_v2.run_id)
        except MlflowException as e:
            raise ModelRegistryError(f"Failed to get run information: {e}") from e

        # Collect metrics
        metrics_v1 = run_v1.data.metrics
        metrics_v2 = run_v2.data.metrics

        # Get all unique metric names
        all_metrics = set(metrics_v1.keys()) | set(metrics_v2.keys())

        # Build comparison DataFrame
        comparison_data = []

        for metric_name in sorted(all_metrics):
            value_v1 = metrics_v1.get(metric_name, None)
            value_v2 = metrics_v2.get(metric_name, None)

            # Calculate difference
            if value_v1 is not None and value_v2 is not None:
                try:
                    diff = float(value_v2) - float(value_v1)
                    pct_change = (diff / float(value_v1)) * 100 if float(value_v1) != 0 else None
                except (ValueError, TypeError):
                    diff = None
                    pct_change = None
            else:
                diff = None
                pct_change = None

            comparison_data.append({
                'metric': metric_name,
                'version_1': value_v1,
                'version_2': value_v2,
                'difference': diff,
                'percent_change': pct_change
            })

        comparison_df = pd.DataFrame(comparison_data)

        # If test data provided, compute statistical significance
        if test_data is not None:
            try:
                user_ids, item_ids, labels = test_data

                # Load both models
                model_v1_obj, _ = load_latest_model(model_name, stage=None)
                model_v2_obj, _ = load_latest_model(model_name, stage=None)

                # Ensure models are in eval mode
                model_v1_obj.eval()
                model_v2_obj.eval()

                # Get predictions
                with torch.no_grad():
                    preds_v1 = model_v1_obj.predict(user_ids, item_ids).cpu().numpy().flatten()
                    preds_v2 = model_v2_obj.predict(user_ids, item_ids).cpu().numpy().flatten()

                # Perform paired t-test
                try:
                    t_stat, p_value = stats.ttest_rel(preds_v1, preds_v2)
                    is_significant = p_value < 0.05

                    # Add to comparison DataFrame
                    comparison_df['statistical_test'] = 'paired_ttest'
                    comparison_df['p_value'] = p_value if len(comparison_df) == 1 else None
                    comparison_df['p_value'].iloc[0] = p_value
                    comparison_df['statistically_significant'] = (
                        is_significant if len(comparison_df) == 1 else None
                    )
                    comparison_df['statistically_significant'].iloc[0] = is_significant

                    logger.info(
                        f"Statistical test: p-value={p_value:.4f}, "
                        f"significant={is_significant}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to perform statistical test: {e}")

            except Exception as e:
                logger.warning(f"Failed to compute test data comparison: {e}")

        # Add model metadata
        comparison_df['model_name'] = model_name
        comparison_df['version_1'] = version_1
        comparison_df['version_2'] = version_2
        comparison_df['comparison_date'] = datetime.now().isoformat()

        logger.info(
            f"Compared model versions {version_1} and {version_2}",
            extra={
                "model_name": model_name,
                "metrics_compared": len(all_metrics)
            }
        )

        return comparison_df

    except ModelRegistryError:
        raise
    except MlflowException as e:
        logger.error(f"MLflow error during model comparison: {e}", exc_info=True)
        raise ModelRegistryError(f"Failed to compare models: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error during model comparison: {e}", exc_info=True)
        raise ModelRegistryError(f"Model comparison failed: {e}") from e


def list_models(
    experiment_name: Optional[str] = None,
    stage: Optional[str] = None
) -> pd.DataFrame:
    """
    List all registered models or models in a specific stage.

    Args:
        experiment_name: Optional experiment name to filter models
        stage: Optional model stage to filter by (e.g., "Production", "Staging")

    Returns:
        DataFrame with model information (name, version, stage, metrics)

    Example:
        >>> models_df = list_models(stage="Production")
        >>> print(models_df)
    """
    try:
        client = MlflowClient()

        if stage:
            models = client.search_model_versions(f"stage='{stage}'")
        else:
            models = client.search_registered_models()

        model_data = []

        for model in models:
            if hasattr(model, 'name'):
                # Registered model
                model_data.append({
                    'name': model.name,
                    'description': model.description,
                    'tags': model.tags
                })
            else:
                # Model version
                model_data.append({
                    'name': model.name,
                    'version': model.version,
                    'stage': model.current_stage,
                    'run_id': model.run_id
                })

        df = pd.DataFrame(model_data)

        logger.debug(f"Listed {len(df)} models")
        return df

    except MlflowException as e:
        logger.error(f"Failed to list models: {e}", exc_info=True)
        raise ModelRegistryError(f"Failed to list models: {e}") from e
