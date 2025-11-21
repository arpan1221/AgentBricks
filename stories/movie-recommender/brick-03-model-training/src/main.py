"""Main entry point for NCF model training with Hydra configuration.

This script uses Hydra for configuration management and provides a CLI
interface for training the NCF model.
"""

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from typing import Dict, Any

from src.train import train_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main training function with Hydra configuration.

    Args:
        cfg: Hydra configuration object

    Example:
        Run training with default config:
            python src/main.py

        Override hyperparameters:
            python src/main.py training.lr=0.0001 training.batch_size=512

        Use GPU:
            python src/main.py device=cuda
    """
    # Convert Hydra config to regular dict
    config = OmegaConf.to_container(cfg, resolve=True)

    logger.info("Starting NCF model training")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        # Train model
        model = train_model(config)

        logger.info("Training completed successfully")
        logger.info(f"Model parameters: {model.get_num_parameters():,}")

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
