import logging
from datetime import datetime
import pickle as pkl
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    BatchSizeFinder,
    LearningRateFinder,
)
from lightning.pytorch.loggers import Logger, MLFlowLogger, TensorBoardLogger, WandbLogger
from typing import List

from src.dataset import TracesDataModule
from src.denoisers import ConditionalUnetDenoiser, ConditionalUnetGraphDenoiser, ConditionalUnetMatrixDenoiser
from src.diffusion import DiffusionLightningModule, DDPM, DDIM
from src.utils.config import Config
from src.utils.setup_utils import load_data, create_datamodule, create_denoiser, create_diffusion, create_loggers
from src.utils.initialization import create_model


logger = logging.getLogger(__name__)

def _create_callbacks(cfg: Config, save_dir: Path, start_time: str) -> list:
    """Create Lightning callbacks."""
    callbacks = []
    
    checkpoint_dir = Path(save_dir, "checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{start_time}-{cfg.logging.experiment_name}-{cfg.logging.run_name}-{cfg.logging.version}-{{epoch:03d}}-{{val_loss:.4f}}",
            mode="min",
            monitor="val_loss",
            save_top_k=cfg.callbacks.save_top_k,
            save_last=cfg.callbacks.save_last,
            verbose=True,
        )
    )

    callbacks.append(BatchSizeFinder(
            mode="binsearch",
            margin=0.1,
        )
    )

    callbacks.append(LearningRateFinder(
            min_lr=1e-6,
            max_lr=1e-3,
            num_training_steps=100,
            mode="exponential",
        )
    )
    
    if cfg.callbacks.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.early_stopping_monitor,
                patience=cfg.callbacks.early_stopping_patience,
                mode=cfg.callbacks.early_stopping_mode,
                verbose=True,
            )
        )
    
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    callbacks.append(RichProgressBar())
    
    return callbacks


def train(cfg: Config):
    """
    Main training entry point.
    
    Sets up DataModule, Model, Logger, Callbacks and runs Lightning Trainer.
    """
    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("Loading dataset from %s", cfg.data.path)
    labels, data = load_data(cfg)
    
    # Create datamodule
    datamodule = create_datamodule(cfg, labels, data)
    datamodule.setup()
    
    logger.info(
        "Dataset: train=%d, val=%d, test=%d, num_classes=%d",
        len(datamodule.train_dataset),
        len(datamodule.val_dataset),
        len(datamodule.test_dataset),
        cfg.data.num_classes,
    )
    
    # Create model components
    denoiser = create_denoiser(cfg, datamodule.flow_matrix, datamodule.graph_data)
    diffusion = create_diffusion(cfg.diffusion.sampler, cfg)
    eval_diffusion = create_diffusion("ddim", cfg) if cfg.diffusion.eval_use_ddim else diffusion
    model = create_model(cfg, denoiser, diffusion, eval_diffusion)

    save_dir = Path(cfg.logging.save_dir, cfg.logging.experiment_name, cfg.logging.run_name)
    save_dir.mkdir(parents=True, exist_ok=True)

    exp_loggers = create_loggers(cfg, save_dir)
    
    callbacks = _create_callbacks(cfg, save_dir, start_time)
    
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        deterministic=cfg.trainer.deterministic,
        logger=exp_loggers if exp_loggers else False,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
        fast_dev_run=cfg.trainer.fast_dev_run,
    )
    
    # Log hyperparameters to every logger
    if exp_loggers:
        from omegaconf import OmegaConf
        hparams = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        for lg in trainer.loggers:
            lg.log_hyperparams(hparams)
    
    # Train (with optional resume)
    logger.info("Starting training...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=cfg.resume_from,
    )
    
    # Test
    logger.info("Running test evaluation...")
    trainer.test(model, datamodule=datamodule, ckpt_path="best")
    
    logger.info("Training complete. Best checkpoint: %s", trainer.checkpoint_callback.best_model_path)
