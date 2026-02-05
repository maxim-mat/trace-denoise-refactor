import logging
import os
import pickle as pkl
from pathlib import Path
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import Logger, MLFlowLogger, TensorBoardLogger, WandbLogger

from src.dataset import TracesDataModule
from src.denoisers import ConditionalUnetDenoiser
from src.diffusion import DiffusionLightningModule, DDPM, DDIM
from src.utils.config import Config


logger = logging.getLogger(__name__)


def _load_data(cfg: Config):
    """Load dataset from pickle file."""
    with open(cfg.data.path, "rb") as f:
        payload = pkl.load(f)
    labels = payload.get("target")
    data = payload.get("stochastic")
    if labels is None:
        raise ValueError("Dataset payload missing 'target' key")
    if data is None:
        raise ValueError("Dataset payload missing 'stochastic' key")
    return labels, data


def _create_datamodule(cfg: Config, labels, data) -> TracesDataModule:
    """Create Lightning DataModule."""
    return TracesDataModule(
        labels=labels,
        data=data,
        final_channels=cfg.data.num_classes,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        pin_memory=cfg.data.pin_memory,
        target_length=cfg.data.target_length,
        seed=cfg.seed,
    )


def _create_denoiser(cfg: Config, max_input_dim: int):
    """Create denoiser model based on config."""
    if cfg.model.type == "unet":
        return ConditionalUnetDenoiser(
            in_ch=cfg.data.num_classes,
            out_ch=cfg.data.num_classes,
            max_input_dim=max_input_dim,
            time_dim=cfg.model.time_dim,
        )
    # TODO: Add unet_matrix and unet_graph when converted
    raise ValueError(f"Unknown model type: {cfg.model.type}")


def _create_diffusion(cfg: Config) -> Optional[DDPM | DDIM]:
    """Create diffusion process based on config."""
    if cfg.diffusion.sampler == "ddpm":
        return DDPM(
            noise_steps=cfg.diffusion.noise_steps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
        )
    elif cfg.diffusion.sampler == "ddim":
        return DDIM(
            noise_steps=cfg.diffusion.noise_steps,
            inference_steps=cfg.diffusion.ddim_inference_steps,
            eta=cfg.diffusion.ddim_eta,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
        )
    return None


def _create_model(cfg: Config, denoiser, diffusion) -> DiffusionLightningModule:
    """Create DiffusionLightningModule."""
    return DiffusionLightningModule(
        denoiser=denoiser,
        diffusion=diffusion,
        noise_steps=cfg.diffusion.noise_steps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        learning_rate=cfg.optimizer.learning_rate,
        loss_type=cfg.diffusion.loss_type,
        denoiser_output=cfg.diffusion.denoiser_output,
        conditional_dropout=cfg.diffusion.conditional_dropout,
    )


def _create_logger(cfg: Config) -> Optional[Logger]:
    """Create experiment logger based on config."""
    save_dir = Path(cfg.logging.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if cfg.logging.logger == "tensorboard":
        return TensorBoardLogger(
            save_dir=str(save_dir),
            name=cfg.logging.project_name,
            version=cfg.logging.experiment_name,
        )
    elif cfg.logging.logger == "mlflow":
        return MLFlowLogger(
            experiment_name=cfg.logging.project_name,
            run_name=cfg.logging.experiment_name,
            tracking_uri=cfg.logging.mlflow_tracking_uri or "mlruns",
            save_dir=str(save_dir),
        )
    elif cfg.logging.logger == "wandb":
        return WandbLogger(
            project=cfg.logging.project_name,
            name=cfg.logging.experiment_name,
            save_dir=str(save_dir),
            entity=cfg.logging.wandb_entity,
            offline=cfg.logging.wandb_offline,
        )
    elif cfg.logging.logger == "none":
        return None
    raise ValueError(f"Unknown logger: {cfg.logging.logger}")


def _create_callbacks(cfg: Config, save_dir: Path) -> list:
    """Create Lightning callbacks."""
    callbacks = []
    
    # Checkpointing - best model
    checkpoint_dir = save_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks.append(
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename="best-{epoch:03d}-{val_loss:.4f}",
            monitor=cfg.callbacks.early_stopping_monitor,
            mode=cfg.callbacks.early_stopping_mode,
            save_top_k=cfg.callbacks.save_top_k,
            save_last=cfg.callbacks.save_last,
            verbose=True,
        )
    )
    
    # Early stopping
    if cfg.callbacks.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.callbacks.early_stopping_monitor,
                patience=cfg.callbacks.early_stopping_patience,
                mode=cfg.callbacks.early_stopping_mode,
                verbose=True,
            )
        )
    
    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))
    
    # Progress bar
    callbacks.append(RichProgressBar())
    
    return callbacks


def train(cfg: Config):
    """
    Main training entry point.
    
    Sets up DataModule, Model, Logger, Callbacks and runs Lightning Trainer.
    """
    # Seed everything for reproducibility
    L.seed_everything(cfg.seed, workers=True)
    
    # Load data
    logger.info("Loading dataset from %s", cfg.data.path)
    labels, data = _load_data(cfg)
    
    # Create datamodule
    datamodule = _create_datamodule(cfg, labels, data)
    datamodule.setup()
    
    # Get max sequence length from dataset
    max_input_dim = cfg.model.max_input_dim
    if max_input_dim is None or max_input_dim == 0:
        max_input_dim = datamodule.full_dataset.max_sequence_length
        # Round up to nearest power of 8 for efficient computation
        max_input_dim = ((max_input_dim + 7) // 8) * 8
    
    logger.info(
        "Dataset: train=%d, val=%d, test=%d, max_seq_len=%d, num_classes=%d",
        len(datamodule.train_dataset),
        len(datamodule.val_dataset),
        len(datamodule.test_dataset),
        max_input_dim,
        cfg.data.num_classes,
    )
    
    # Create model components
    denoiser = _create_denoiser(cfg, max_input_dim)
    diffusion = _create_diffusion(cfg)
    model = _create_model(cfg, denoiser, diffusion)
    
    # Create logger
    exp_logger = _create_logger(cfg)
    
    # Determine save directory
    if exp_logger is not None and hasattr(exp_logger, 'log_dir'):
        save_dir = Path(exp_logger.log_dir)
    else:
        save_dir = Path(cfg.logging.save_dir) / cfg.logging.project_name
        if cfg.logging.experiment_name:
            save_dir = save_dir / cfg.logging.experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create callbacks
    callbacks = _create_callbacks(cfg, save_dir)
    
    # Create trainer
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
        logger=exp_logger,
        callbacks=callbacks,
        default_root_dir=str(save_dir),
    )
    
    # Log hyperparameters
    if exp_logger is not None:
        # Save config as hyperparameters
        from omegaconf import OmegaConf
        hparams = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
        trainer.logger.log_hyperparams(hparams)
    
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
    
    return trainer, model, datamodule
