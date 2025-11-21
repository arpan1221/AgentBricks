"""Neural Collaborative Filtering (NCF) model implementation.

This module implements the NCF model from the paper:
"Neural Collaborative Filtering" by He et al. (WWW 2017)

The model combines Generalized Matrix Factorization (GMF) and Multi-Layer
Perceptron (MLP) paths to learn user-item interactions.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.

    Implements the architecture from He et al. (2017) that combines:
    1. Generalized Matrix Factorization (GMF) path: learns linear interactions
    2. Multi-Layer Perceptron (MLP) path: learns non-linear interactions

    The model fuses both paths to make predictions:
        y_hat = sigmoid(W^T * [h_GMF, h_MLP] + b)

    where:
        h_GMF = user_emb_gmf ⊙ item_emb_gmf (element-wise product)
        h_MLP = MLP([user_emb_mlp, item_emb_mlp])

    Args:
        num_users: Number of unique users in the dataset
        num_items: Number of unique items (movies) in the dataset
        embedding_dim: Dimension of embedding vectors (default: 64)
        mlp_layers: List of hidden layer sizes for MLP path (default: [128, 64, 32])
        dropout: Dropout rate for regularization (default: 0.2)
        use_batch_norm: Whether to use batch normalization (default: True)

    Attributes:
        num_users: Number of users
        num_items: Number of items
        embedding_dim: Embedding dimension
        mlp_layers: MLP layer configuration

    Example:
        >>> model = NCF(
        ...     num_users=10000,
        ...     num_items=5000,
        ...     embedding_dim=64,
        ...     mlp_layers=[128, 64, 32]
        ... )
        >>> user_ids = torch.tensor([0, 1, 2])
        >>> item_ids = torch.tensor([10, 20, 30])
        >>> scores = model(user_ids, item_ids)
        >>> print(scores.shape)
        torch.Size([3, 1])
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        mlp_layers: Optional[List[int]] = None,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ) -> None:
        """Initialize NCF model."""
        super().__init__()

        if num_users <= 0 or num_items <= 0:
            raise ValueError("num_users and num_items must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if dropout < 0 or dropout >= 1:
            raise ValueError("dropout must be in [0, 1)")

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.mlp_layers = mlp_layers if mlp_layers is not None else [128, 64, 32]
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # GMF path embeddings
        # These learn linear interactions through element-wise product
        # h_GMF = p_u ⊙ q_i where ⊙ is element-wise product
        self.user_embedding_gmf = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )
        self.item_embedding_gmf = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        # MLP path embeddings
        # These are concatenated and passed through MLP layers
        # h_MLP = MLP([p_u, q_i]) where [·, ·] is concatenation
        self.user_embedding_mlp = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim
        )
        self.item_embedding_mlp = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim
        )

        # MLP layers for non-linear feature learning
        # Input dimension is 2 * embedding_dim (concatenated user + item embeddings)
        mlp_modules = []
        input_size = embedding_dim * 2

        for layer_size in self.mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))

            if self.use_batch_norm:
                mlp_modules.append(nn.BatchNorm1d(layer_size))

            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(self.dropout))

            input_size = layer_size

        self.mlp = nn.Sequential(*mlp_modules)

        # Final output layer that fuses GMF and MLP paths
        # h_final = [h_GMF, h_MLP]
        # y_hat = σ(W^T * h_final + b)
        # where σ is sigmoid activation
        output_size = embedding_dim + self.mlp_layers[-1]
        self.output_layer = nn.Linear(output_size, 1)

        # Initialize weights using He initialization (for ReLU activations)
        self._init_weights()

        logger.info(
            f"NCF model initialized",
            extra={
                "num_users": num_users,
                "num_items": num_items,
                "embedding_dim": embedding_dim,
                "mlp_layers": self.mlp_layers,
                "dropout": dropout
            }
        )

    def _init_weights(self) -> None:
        """
        Initialize model weights using He initialization.

        He initialization is optimal for ReLU activations:
            W ~ N(0, √(2/n_in))

        where n_in is the number of input neurons to the layer.

        Embeddings are initialized from uniform distribution:
            U(-1/√d, 1/√d) where d is embedding dimension
        """
        # Initialize GMF embeddings
        nn.init.normal_(self.user_embedding_gmf.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding_gmf.weight, mean=0.0, std=0.01)

        # Initialize MLP embeddings
        nn.init.normal_(self.user_embedding_mlp.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embedding_mlp.weight, mean=0.0, std=0.01)

        # Initialize MLP layers with He initialization
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_out', nonlinearity='relu')
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

        logger.debug("Model weights initialized using He initialization")

    def forward(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass through the NCF model.

        Computes predictions by fusing GMF and MLP paths:
            1. GMF path: h_GMF = user_emb_gmf ⊙ item_emb_gmf
            2. MLP path: h_MLP = MLP([user_emb_mlp, item_emb_mlp])
            3. Fusion: y_hat = σ(W^T * [h_GMF, h_MLP] + b)

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)

        Returns:
            Prediction scores, shape (batch_size, 1). Values are in [0, 1]
            after sigmoid activation.

        Raises:
            ValueError: If input shapes are invalid

        Example:
            >>> model = NCF(num_users=1000, num_items=500)
            >>> user_ids = torch.tensor([0, 1, 2])
            >>> item_ids = torch.tensor([10, 20, 30])
            >>> scores = model(user_ids, item_ids)
            >>> print(scores.shape)
            torch.Size([3, 1])
        """
        if user_ids.dim() != 1 or item_ids.dim() != 1:
            raise ValueError(
                f"user_ids and item_ids must be 1D tensors, "
                f"got shapes {user_ids.shape} and {item_ids.shape}"
            )

        if len(user_ids) != len(item_ids):
            raise ValueError(
                f"user_ids and item_ids must have same length, "
                f"got {len(user_ids)} and {len(item_ids)}"
            )

        batch_size = len(user_ids)

        # GMF path: element-wise product of embeddings
        # h_GMF = p_u ⊙ q_i
        # This captures linear interactions between users and items
        user_emb_gmf = self.user_embedding_gmf(user_ids)  # (batch_size, embedding_dim)
        item_emb_gmf = self.item_embedding_gmf(item_ids)  # (batch_size, embedding_dim)
        gmf_output = user_emb_gmf * item_emb_gmf  # (batch_size, embedding_dim)

        # MLP path: concatenate embeddings and pass through MLP
        # h_MLP = MLP([p_u, q_i])
        # This captures non-linear interactions between users and items
        user_emb_mlp = self.user_embedding_mlp(user_ids)  # (batch_size, embedding_dim)
        item_emb_mlp = self.item_embedding_mlp(item_ids)  # (batch_size, embedding_dim)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)  # (batch_size, 2*embedding_dim)
        mlp_output = self.mlp(mlp_input)  # (batch_size, mlp_layers[-1])

        # Fuse GMF and MLP paths
        # h_final = [h_GMF, h_MLP]
        # Concatenate along feature dimension
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)  # (batch_size, embedding_dim + mlp_layers[-1])

        # Final prediction layer with sigmoid activation
        # y_hat = σ(W^T * h_final + b)
        # Sigmoid ensures output is in [0, 1] for binary classification
        output = self.output_layer(concat_output)  # (batch_size, 1)
        output = torch.sigmoid(output)  # Apply sigmoid to get probabilities

        return output

    def predict(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Prediction method for inference (evaluation mode).

        Wrapper around forward() that ensures model is in eval mode and
        disables gradient computation for efficiency.

        Args:
            user_ids: Tensor of user IDs, shape (batch_size,)
            item_ids: Tensor of item IDs, shape (batch_size,)

        Returns:
            Prediction scores, shape (batch_size, 1)

        Example:
            >>> model = NCF(num_users=1000, num_items=500)
            >>> model.eval()  # Set to evaluation mode
            >>> user_ids = torch.tensor([0, 1, 2])
            >>> item_ids = torch.tensor([10, 20, 30])
            >>> scores = model.predict(user_ids, item_ids)
            >>> print(scores)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(user_ids, item_ids)

    def save(
        self,
        filepath: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Save model state and metadata to file.

        Saves model state dict, model configuration, and optional metadata
        for later loading.

        Args:
            filepath: Path to save the model (should end with .pth or .pt)
            metadata: Optional dictionary with additional metadata to save
                     (e.g., training hyperparameters, performance metrics)

        Raises:
            ValueError: If filepath is invalid
            Exception: If save operation fails

        Example:
            >>> model = NCF(num_users=1000, num_items=500)
            >>> model.save(
            ...     "models/ncf_v1.pth",
            ...     metadata={"epoch": 10, "train_loss": 0.5}
            ... )
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_config': {
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_dim': self.embedding_dim,
                'mlp_layers': self.mlp_layers,
                'dropout': self.dropout,
                'use_batch_norm': self.use_batch_norm
            }
        }

        if metadata:
            checkpoint['metadata'] = metadata

        torch.save(checkpoint, filepath)

        logger.info(
            f"Model saved to {filepath}",
            extra={"filepath": str(filepath), "metadata_keys": list(metadata.keys()) if metadata else []}
        )

    @classmethod
    def load(
        cls,
        filepath: str,
        device: Optional[torch.device] = None
    ) -> "NCF":
        """
        Load model from saved checkpoint.

        Creates model instance and loads weights from checkpoint file.

        Args:
            filepath: Path to saved model checkpoint
            device: Optional device to load model on. If None, uses CPU.
                   Can be 'cpu', 'cuda', or torch.device object.

        Returns:
            Loaded NCF model instance

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            Exception: If loading fails

        Example:
            >>> model = NCF.load("models/ncf_v1.pth", device="cuda")
            >>> model.eval()
            >>> scores = model.predict(user_ids, item_ids)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {filepath}")

        if device is None:
            device = torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)

        checkpoint = torch.load(filepath, map_location=device)

        # Extract model configuration
        config = checkpoint['model_config']
        model = cls(
            num_users=config['num_users'],
            num_items=config['num_items'],
            embedding_dim=config['embedding_dim'],
            mlp_layers=config['mlp_layers'],
            dropout=config.get('dropout', 0.2),
            use_batch_norm=config.get('use_batch_norm', True)
        )

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        logger.info(
            f"Model loaded from {filepath}",
            extra={
                "filepath": str(filepath),
                "device": str(device),
                "has_metadata": 'metadata' in checkpoint
            }
        )

        return model

    def get_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Get metadata from saved checkpoint without loading entire model.

        Useful for inspecting training history, hyperparameters, etc.

        Args:
            checkpoint_path: Path to saved model checkpoint

        Returns:
            Dictionary with metadata, or empty dict if no metadata found

        Example:
            >>> metadata = model.get_metadata("models/ncf_v1.pth")
            >>> print(metadata.get("epoch"))
            10
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint.get('metadata', {})

    def get_num_parameters(self) -> int:
        """
        Get total number of trainable parameters in the model.

        Returns:
            Total number of parameters

        Example:
            >>> model = NCF(num_users=1000, num_items=500)
            >>> num_params = model.get_num_parameters()
            >>> print(f"Model has {num_params:,} parameters")
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to_device(self, device: str) -> "NCF":
        """
        Move model to specified device and return self.

        Convenience method for device management.

        Args:
            device: Device string ('cpu' or 'cuda') or torch.device

        Returns:
            Self (for method chaining)

        Example:
            >>> model = NCF(num_users=1000, num_items=500)
            >>> model = model.to_device("cuda")
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.to(device)
        logger.debug(f"Model moved to device: {device}")
        return self
