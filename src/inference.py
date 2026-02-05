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


def _create_denoiser_from_config(cfg: Config, max_input_dim: int):
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


def load_model_from_checkpoint(
    checkpoint_path: str,
    cfg: Config,
    max_input_dim: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> DiffusionLightningModule:
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        cfg: Config for reconstructing denoiser and optionally overriding sampler
        max_input_dim: Maximum input dimension for denoiser
        device: Device to load model on
        
    Returns:
        Loaded DiffusionLightningModule
    """
    logger.info("Loading model from checkpoint: %s", checkpoint_path)
    
    # Create denoiser architecture from config
    denoiser = _create_denoiser_from_config(cfg, max_input_dim)
    
    # Load model with the denoiser
    model = DiffusionLightningModule.load_from_checkpoint_with_denoiser(
        checkpoint_path=checkpoint_path,
        denoiser=denoiser,
        map_location=device,
    )
    
    # Optionally switch to a different sampler for inference (e.g., DDIM for speed)
    if cfg.diffusion.sampler == "ddim":
        logger.info("Switching to DDIM sampler with %d inference steps", cfg.diffusion.ddim_inference_steps)
        model.set_diffusion(
            DDIM(
                noise_steps=model.noise_steps,
                inference_steps=cfg.diffusion.ddim_inference_steps,
                eta=cfg.diffusion.ddim_eta,
                beta_start=model.beta_start,
                beta_end=model.beta_end,
                device=device,
            )
        )
    
    model.eval()
    model.to(device)
    return model


def sample(
    model: DiffusionLightningModule,
    batch_size: int,
    num_classes: int,
    sequence_length: int,
    conditioning: Optional[torch.Tensor] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> torch.Tensor:
    """
    Generate samples from the diffusion model.
    
    Args:
        model: Trained DiffusionLightningModule
        batch_size: Number of samples to generate
        num_classes: Number of classes (channels)
        sequence_length: Length of sequences to generate
        conditioning: Optional conditioning tensor
        device: Device to run on
        
    Returns:
        Generated samples tensor of shape (batch_size, num_classes, sequence_length)
    """
    model.eval()
    model.to(device)
    
    if conditioning is not None:
        conditioning = conditioning.to(device)
    
    with torch.no_grad():
        samples = model.sample(
            batch_size=batch_size,
            shape=(num_classes, sequence_length),
            y=conditioning,
        )
    
    return samples


def run_inference(cfg: Config):
    """
    Main inference entry point.
    
    Loads a trained model and generates samples or runs evaluation.
    
    Args:
        cfg: Configuration object
    """
    L.seed_everything(cfg.seed, workers=True)
    
    # Check if verbose trajectory is enabled in config
    verbose_trajectory = cfg.metrics.verbose_trajectory
    
    if cfg.checkpoint_path is None:
        raise ValueError("checkpoint_path is required for inference")
    
    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load data for conditioning or evaluation
    logger.info("Loading dataset from %s", cfg.data.path)
    labels, data = _load_data(cfg)
    datamodule = _create_datamodule(cfg, labels, data)
    datamodule.setup()
    
    max_input_dim = cfg.model.max_input_dim
    if max_input_dim is None or max_input_dim == 0:
        max_input_dim = datamodule.full_dataset.max_sequence_length
        max_input_dim = ((max_input_dim + 7) // 8) * 8
    
    # Load model
    model = load_model_from_checkpoint(
        checkpoint_path=str(checkpoint_path),
        cfg=cfg,
        max_input_dim=max_input_dim,
        device=device,
    )
    
    logger.info(
        "Dataset loaded: %d samples, max_seq_len=%d, num_classes=%d",
        len(datamodule.full_dataset),
        max_input_dim,
        cfg.data.num_classes,
    )
    
    # Enable verbose test mode if requested
    if verbose_trajectory:
        logger.info("Enabling verbose trajectory analysis...")
        model.enable_verbose_test(
            metrics=cfg.metrics.test,
            save_every=cfg.metrics.trajectory_save_every,
        )
    
    # Run evaluation on test set
    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        logger=False,
    )
    
    logger.info("Running evaluation on test set...")
    results = trainer.test(model, datamodule=datamodule)
    
    logger.info("Test results: %s", results)
    
    # If verbose mode was enabled, save trajectory analysis
    if verbose_trajectory and model.trajectory_results:
        output_dir = Path(cfg.logging.save_dir) / "inference_outputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw trajectory results
        trajectory_path = output_dir / "trajectory_results.pkl"
        with open(trajectory_path, "wb") as f:
            pkl.dump(model.trajectory_results, f)
        logger.info("Trajectory results saved to %s", trajectory_path)
        
        # Also save as DataFrame if pandas available
        try:
            df = model.get_trajectory_dataframe()
            csv_path = output_dir / "trajectory_metrics.csv"
            df.to_csv(csv_path, index=False)
            logger.info("Trajectory metrics CSV saved to %s", csv_path)
            
            # Log summary stats
            logger.info("\nTrajectory Metrics Summary (by timestep):")
            summary = df.groupby("timestep").mean(numeric_only=True)
            logger.info("\n%s", summary.to_string())
        except ImportError:
            logger.warning("pandas not available, skipping CSV export")
    
    # Generate some samples for inspection
    logger.info("Generating sample outputs...")
    test_loader = datamodule.test_dataloader()
    batch = next(iter(test_loader))
    labels_batch, data_batch = batch
    
    # Use test data as conditioning
    conditioning = data_batch.permute(0, 2, 1).float().to(device)
    
    samples = sample(
        model=model,
        batch_size=min(cfg.data.batch_size, 8),
        num_classes=cfg.data.num_classes,
        sequence_length=max_input_dim,
        conditioning=conditioning[:min(cfg.data.batch_size, 8)],
        device=device,
    )
    
    logger.info("Generated samples shape: %s", samples.shape)
    
    # Save samples
    output_dir = Path(cfg.logging.save_dir) / "inference_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "samples.pkl"
    with open(output_path, "wb") as f:
        pkl.dump({
            "samples": samples.cpu(),
            "conditioning": conditioning.cpu(),
            "labels": labels_batch.cpu(),
        }, f)
    
    logger.info("Samples saved to %s", output_path)
    
    return model, samples
