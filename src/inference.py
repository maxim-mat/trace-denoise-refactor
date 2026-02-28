import logging
import pickle as pkl
from pathlib import Path
from typing import Optional

import torch
import lightning as L

from src.dataset import TracesDataModule
from src.denoisers import ConditionalUnetDenoiser
from src.diffusion import DiffusionLightningModule, DDPM, DDIM
from src.utils.config import Config
from src.utils.setup_utils import load_data, create_datamodule, create_denoiser, create_diffusion, create_model, create_loggers


logger = logging.getLogger(__name__)


def run_inference(cfg: Config):
    """
    Main inference entry point.
    
    Loads a trained model and generates samples or runs evaluation.
    
    Args:
        cfg: Configuration object
    """
    L.seed_everything(cfg.seed, workers=True)
    
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required for inference")
    
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load data for conditioning or evaluation
    logger.info("Loading dataset from %s", cfg.data.path)
    labels, data = load_data(cfg)
    datamodule = create_datamodule(cfg, labels, data)
    datamodule.setup()
    
    logger.info(
        "Dataset loaded: %d samples, num_classes=%d",
        len(datamodule.full_dataset),
        cfg.data.num_classes,
    )

    model = DiffusionLightningModule.load_from_experiment(checkpoint_path)

    # Run evaluation on test set
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        logger=False,
    )
    
    logger.info("Running evaluation on test set...")
    results = trainer.predict(model, datamodule=datamodule)
    
    logger.info("Test results: %s", results)
