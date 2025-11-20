"""Training script for Neural Collaborative Filtering (NCF) model.

This module provides functions for training the NCF model with proper
data splitting, negative sampling, evaluation metrics, and experiment tracking.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class InteractionDataset(Dataset):
    """
    Dataset class for user-item interactions.

    Handles positive interactions and generates negative samples for training.

    Args:
        interactions: DataFrame with columns: user_id, item_id, label (0 or 1)
        num_users: Total number of users
        num_items: Total number of items
        num_negatives: Number of negative samples per positive (default: 4)
        is_training: Whether dataset is for training (generates negatives)

    Example:
        >>> dataset = InteractionDataset(train_df, num_users=1000, num_items=500)
        >>> loader = DataLoader(dataset, batch_size=32, shuffle=True)
    """

    def __init__(
        self,
        interactions: pd.DataFrame,
        num_users: int,
        num_items: int,
        num_negatives: int = 4,
        is_training: bool = True
    ) -> None:
        """Initialize interaction dataset."""
        self.num_users = num_users
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.is_training = is_training

        # Filter positive interactions (label == 1)
        positive_interactions = interactions[interactions['label'] == 1].copy()

        # Build user-item set for efficient negative sampling
        if is_training:
            self.user_item_set = set(
                zip(positive_interactions['user_id'], positive_interactions['item_id'])
            )

        self.interactions = positive_interactions.reset_index(drop=True)

        logger.debug(
            f"Dataset initialized",
            extra={
                "num_interactions": len(self.interactions),
                "num_negatives": num_negatives,
                "is_training": is_training
            }
        )

    def __len__(self) -> int:
        """Return dataset length."""
        if self.is_training:
            return len(self.interactions) * (1 + self.num_negatives)
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get interaction at index.

        For training: returns positive + negatives
        For evaluation: returns only positive interactions

        Returns:
            Tuple of (user_ids, item_ids, labels)
        """
        if self.is_training:
            # Determine if this is a positive or negative sample
            positive_idx = idx // (1 + self.num_negatives)
            negative_idx = idx % (1 + self.num_negatives)

            row = self.interactions.iloc[positive_idx]
            user_id = int(row['user_id'])

            if negative_idx == 0:
                # Positive sample
                item_id = int(row['item_id'])
                label = 1.0
            else:
                # Negative sample
                item_id = self._sample_negative_item(user_id)
                label = 0.0
        else:
            # Evaluation: only positive interactions
            row = self.interactions.iloc[idx]
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            label = 1.0

        return (
            torch.tensor(user_id, dtype=torch.long),
            torch.tensor(item_id, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )

    def _sample_negative_item(self, user_id: int) -> int:
        """
        Sample a negative item for a user (item user hasn't interacted with).

        Args:
            user_id: User ID

        Returns:
            Negative item ID
        """
        max_attempts = 100
        for _ in range(max_attempts):
            item_id = np.random.randint(0, self.num_items)
            if (user_id, item_id) not in self.user_item_set:
                return item_id

        # Fallback: return random item if no suitable negative found
        logger.warning(f"Could not find negative for user {user_id}, using random")
        return np.random.randint(0, self.num_items)


def load_data(
    interactions_path: str,
    split_date: datetime
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split interaction data temporally.

    Ensures no data leakage by using temporal split. All data before split_date
    is used for training/validation, and data after split_date is used for testing.

    Args:
        interactions_path: Path to interactions CSV/Parquet file
        split_date: Date to split train/val and test sets

    Returns:
        Tuple of (train_df, val_df, test_df)

    Raises:
        FileNotFoundError: If interactions file doesn't exist
        ValueError: If data format is invalid

    Example:
        >>> train_df, val_df, test_df = load_data(
        ...     "data/interactions.parquet",
        ...     datetime(2024, 1, 1)
        ... )
    """
    path = Path(interactions_path)

    if not path.exists():
        raise FileNotFoundError(f"Interactions file not found: {interactions_path}")

    # Load data
    if path.suffix == '.parquet':
        df = pd.read_parquet(path)
    elif path.suffix == '.csv':
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    # Ensure timestamp column is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ensure user_id and item_id are integers
    if 'user_id' not in df.columns or 'item_id' not in df.columns:
        raise ValueError("DataFrame must contain 'user_id' and 'item_id' columns")

    # Create label column (1 for views, 0 for skips if available)
    if 'label' not in df.columns:
        df['label'] = 1  # All interactions are positive by default
        if 'event_type' in df.columns:
            df['label'] = (df['event_type'] == 'view').astype(int)

    # Temporal split: test set is after split_date
    test_df = df[df['timestamp'] > split_date].copy()
    train_val_df = df[df['timestamp'] <= split_date].copy()

    # Split train/val: use 80/20 split on train_val data
    # Use last 20% of data chronologically for validation
    train_val_df = train_val_df.sort_values('timestamp')
    split_idx = int(len(train_val_df) * 0.8)

    train_df = train_val_df.iloc[:split_idx].copy()
    val_df = train_val_df.iloc[split_idx:].copy()

    logger.info(
        f"Data loaded and split",
        extra={
            "total_interactions": len(df),
            "train_size": len(train_df),
            "val_size": len(val_df),
            "test_size": len(test_df),
            "split_date": split_date.isoformat()
        }
    )

    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    num_users: int,
    num_items: int,
    batch_size: int = 256,
    num_workers: int = 4,
    num_negatives: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training and validation.

    Creates DataLoaders with negative sampling for training data.

    Args:
        train_df: Training interactions DataFrame
        val_df: Validation interactions DataFrame
        num_users: Total number of users
        num_items: Total number of items
        batch_size: Batch size for training (default: 256)
        num_workers: Number of worker processes (default: 4)
        num_negatives: Number of negative samples per positive (default: 4)

    Returns:
        Tuple of (train_loader, val_loader)

    Example:
        >>> train_loader, val_loader = create_dataloaders(
        ...     train_df, val_df, num_users=1000, num_items=500,
        ...     batch_size=128, num_negatives=4
        ... )
    """
    # Create datasets
    train_dataset = InteractionDataset(
        train_df,
        num_users=num_users,
        num_items=num_items,
        num_negatives=num_negatives,
        is_training=True
    )

    val_dataset = InteractionDataset(
        val_df,
        num_users=num_users,
        num_items=num_items,
        num_negatives=0,  # No negatives for validation
        is_training=False
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Faster GPU transfer
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    logger.info(
        f"DataLoaders created",
        extra={
            "train_batches": len(train_loader),
            "val_batches": len(val_loader),
            "batch_size": batch_size,
            "num_negatives": num_negatives
        }
    )

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_grad_norm: float = 1.0
) -> float:
    """
    Train model for one epoch.

    Args:
        model: NCF model to train
        dataloader: Training data loader
        optimizer: Optimizer for updating weights
        criterion: Loss function
        device: Device to train on
        max_grad_norm: Maximum gradient norm for clipping (default: 1.0)

    Returns:
        Average training loss for the epoch

    Example:
        >>> avg_loss = train_epoch(
        ...     model, train_loader, optimizer, criterion, device
        ... )
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)

    for user_ids, item_ids, labels in progress_bar:
        # Move to device
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        labels = labels.to(device).unsqueeze(1)  # (batch_size, 1)

        # Forward pass
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)

        # Compute loss
        loss = criterion(predictions, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    logger.debug(
        f"Training epoch completed",
        extra={
            "avg_loss": avg_loss,
            "num_batches": num_batches
        }
    )

    return avg_loss


def compute_ndcg_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    Compute NDCG@k (Normalized Discounted Cumulative Gain).

    Args:
        scores: Prediction scores, shape (num_items,)
        labels: Ground truth labels (binary), shape (num_items,)
        k: Number of top items to consider (default: 10)

    Returns:
        NDCG@k score
    """
    # Get top k items
    top_k_idx = np.argsort(scores)[::-1][:k]
    top_k_labels = labels[top_k_idx]

    # Compute DCG
    dcg = np.sum(top_k_labels / np.log2(np.arange(2, len(top_k_labels) + 2)))

    # Compute ideal DCG
    ideal_labels = np.sort(labels)[::-1][:k]
    idcg = np.sum(ideal_labels / np.log2(np.arange(2, len(ideal_labels) + 2)))

    # Avoid division by zero
    if idcg == 0:
        return 0.0

    return dcg / idcg


def compute_hit_rate_at_k(scores: np.ndarray, labels: np.ndarray, k: int = 10) -> float:
    """
    Compute Hit Rate@k.

    Hit Rate@k = 1 if at least one relevant item is in top k, else 0.

    Args:
        scores: Prediction scores, shape (num_items,)
        labels: Ground truth labels (binary), shape (num_items,)
        k: Number of top items to consider (default: 10)

    Returns:
        Hit Rate@k (0 or 1)
    """
    top_k_idx = np.argsort(scores)[::-1][:k]
    return 1.0 if np.any(labels[top_k_idx] == 1) else 0.0


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    num_items: int,
    k: int = 10
) -> Dict[str, float]:
    """
    Evaluate model on validation/test set.

    Computes AUC, NDCG@k, and Hit Rate@k metrics.

    Args:
        model: NCF model to evaluate
        dataloader: Validation/test data loader
        device: Device to evaluate on
        num_items: Total number of items (for NDCG/HitRate computation)
        k: Top k items for ranking metrics (default: 10)

    Returns:
        Dictionary with metrics: {'auc', 'ndcg_at_k', 'hit_rate_at_k'}

    Example:
        >>> metrics = evaluate(model, val_loader, device, num_items=500)
        >>> print(f"AUC: {metrics['auc']:.4f}")
    """
    model.eval()

    all_predictions = []
    all_labels = []
    all_user_ids = []
    all_item_ids = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)

        for user_ids, item_ids, labels in progress_bar:
            user_ids = user_ids.to(device)
            item_ids = item_ids.to(device)

            # Get predictions
            predictions = model.predict(user_ids, item_ids)

            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(labels.numpy())
            all_user_ids.extend(user_ids.cpu().numpy())
            all_item_ids.extend(item_ids.numpy())

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_user_ids = np.array(all_user_ids)
    all_item_ids = np.array(all_item_ids)

    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_predictions)
    except ValueError:
        # Handle edge case where all labels are same class
        auc = 0.5

    # Compute NDCG@k and Hit Rate@k per user
    ndcg_scores = []
    hit_rate_scores = []

    unique_users = np.unique(all_user_ids)

    for user_id in unique_users:
        user_mask = all_user_ids == user_id
        user_scores = all_predictions[user_mask]
        user_labels = all_labels[user_mask]

        # For NDCG/HitRate, we need scores for all items for this user
        # For simplicity, we'll compute on the items in the evaluation set
        # In production, you'd want to rank all items
        ndcg = compute_ndcg_at_k(user_scores, user_labels, k=k)
        hit_rate = compute_hit_rate_at_k(user_scores, user_labels, k=k)

        ndcg_scores.append(ndcg)
        hit_rate_scores.append(hit_rate)

    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    avg_hit_rate = np.mean(hit_rate_scores) if hit_rate_scores else 0.0

    metrics = {
        'auc': float(auc),
        f'ndcg_at_{k}': float(avg_ndcg),
        f'hit_rate_at_{k}': float(avg_hit_rate)
    }

    logger.info(
        f"Evaluation completed",
        extra=metrics
    )

    return metrics


def train_model(config: Dict[str, Any]) -> nn.Module:
    """
    Main training function for NCF model.

    Trains model with early stopping, learning rate scheduling, MLflow logging,
    and checkpoint saving.

    Args:
        config: Configuration dictionary with keys:
            - data: Dictionary with 'interactions_path', 'split_date'
            - model: Dictionary with 'num_users', 'num_items', 'embedding_dim', 'mlp_layers'
            - training: Dictionary with 'lr', 'batch_size', 'epochs', 'num_negatives'
            - paths: Dictionary with 'checkpoint_dir', 'mlflow_uri'
            - device: Device string ('cpu' or 'cuda')

    Returns:
        Trained model

    Raises:
        Exception: If training fails

    Example:
        >>> config = {
        ...     "data": {"interactions_path": "data/interactions.parquet", ...},
        ...     "model": {"num_users": 1000, "num_items": 500, ...},
        ...     ...
        ... }
        >>> model = train_model(config)
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Extract configuration
    data_config = config.get('data', {})
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    paths_config = config.get('paths', {})
    device_str = config.get('device', 'cpu')

    device = torch.device(device_str)
    logger.info(f"Training on device: {device}")

    # Load data
    logger.info("Loading data...")
    train_df, val_df, test_df = load_data(
        interactions_path=data_config['interactions_path'],
        split_date=pd.to_datetime(data_config['split_date'])
    )

    # Get model dimensions
    num_users = model_config['num_users']
    num_items = model_config['num_items']

    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        num_users=num_users,
        num_items=num_items,
        batch_size=training_config.get('batch_size', 256),
        num_workers=training_config.get('num_workers', 4),
        num_negatives=training_config.get('num_negatives', 4)
    )

    # Initialize model
    from src.model import NCF

    model = NCF(
        num_users=num_users,
        num_items=num_items,
        embedding_dim=model_config.get('embedding_dim', 64),
        mlp_layers=model_config.get('mlp_layers', [128, 64, 32]),
        dropout=model_config.get('dropout', 0.2),
        use_batch_norm=model_config.get('use_batch_norm', True)
    ).to(device)

    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=training_config.get('lr', 0.001),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize AUC
        factor=0.5,
        patience=2,
        verbose=True
    )

    # MLflow setup
    mlflow_uri = paths_config.get('mlflow_uri', 'file:./mlruns')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment("ncf_training")

    # Training loop
    best_val_auc = 0.0
    patience = training_config.get('early_stopping_patience', 3)
    patience_counter = 0
    epochs = training_config.get('epochs', 10)
    checkpoint_dir = Path(paths_config.get('checkpoint_dir', './checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params({
            'embedding_dim': model_config.get('embedding_dim', 64),
            'mlp_layers': str(model_config.get('mlp_layers', [128, 64, 32])),
            'learning_rate': training_config.get('lr', 0.001),
            'batch_size': training_config.get('batch_size', 256),
            'num_negatives': training_config.get('num_negatives', 4),
            'dropout': model_config.get('dropout', 0.2)
        })

        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch + 1}/{epochs}")

            # Train
            train_loss = train_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device
            )

            # Validate
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                num_items=num_items
            )

            # Update learning rate
            scheduler.step(val_metrics['auc'])

            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_auc': val_metrics['auc'],
                'val_ndcg_at_10': val_metrics['ndcg_at_10'],
                'val_hit_rate_at_10': val_metrics['hit_rate_at_10'],
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)

            logger.info(
                f"Epoch {epoch + 1} - "
                f"Train Loss: {train_loss:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}, "
                f"Val NDCG@10: {val_metrics['ndcg_at_10']:.4f}"
            )

            # Early stopping
            if val_metrics['auc'] > best_val_auc:
                best_val_auc = val_metrics['auc']
                patience_counter = 0

                # Save best model
                checkpoint_path = checkpoint_dir / f"best_model_epoch_{epoch + 1}.pth"
                model.save(
                    str(checkpoint_path),
                    metadata={
                        'epoch': epoch + 1,
                        'val_auc': val_metrics['auc'],
                        'val_ndcg_at_10': val_metrics['ndcg_at_10'],
                        'config': config
                    }
                )

                # Log model to MLflow
                mlflow.pytorch.log_model(model, "model")

                logger.info(f"Saved best model with AUC: {best_val_auc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        best_checkpoint = checkpoint_dir / f"best_model_epoch_*.pth"
        checkpoints = list(checkpoint_dir.glob("best_model_epoch_*.pth"))
        if checkpoints:
            best_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            model = NCF.load(str(best_checkpoint), device=device)
            logger.info(f"Loaded best model from {best_checkpoint}")

        # Final evaluation on test set
        test_loader, _ = create_dataloaders(
            train_df=test_df,
            val_df=test_df.iloc[:0],  # Empty validation
            num_users=num_users,
            num_items=num_items,
            batch_size=training_config.get('batch_size', 256),
            num_negatives=0
        )

        test_metrics = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            num_items=num_items
        )

        mlflow.log_metrics({
            'test_auc': test_metrics['auc'],
            'test_ndcg_at_10': test_metrics['ndcg_at_10'],
            'test_hit_rate_at_10': test_metrics['hit_rate_at_10']
        })

        logger.info(f"Test Metrics: {test_metrics}")

    return model
