"""Utility functions for setup steps in train and inference."""

from src.utils.config import Config
from src.dataset import TracesDataModule
from src.denoisers import ConditionalUnetDenoiser, ConditionalUnetGraphDenoiser, ConditionalUnetMatrixDenoiser
from src.diffusion import DDPM, DDIM
from lightning.pytorch.loggers import Logger, MLFlowLogger, TensorBoardLogger, WandbLogger
from typing import List, Optional
import pickle as pkl
from pathlib import Path

def load_data(cfg: Config):
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


def create_datamodule(cfg: Config, labels, data) -> TracesDataModule:
    """Create Lightning DataModule."""
    get_flow_matrix = cfg.model.type == "unet_matrix"
    get_graph_data = cfg.model.type == "unet_graph"
    if (get_flow_matrix or get_graph_data) and cfg.process.method is None:
        raise ValueError("Process discovery is required for flow matrix or graph data")
    
    return TracesDataModule(
        labels=labels,
        data=data,
        final_channels=cfg.data.num_classes,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
        test_split=cfg.data.test_split,
        padding_value=cfg.data.padding_value,
        pin_memory=cfg.data.pin_memory,
        target_length=cfg.data.target_length,
        use_padding_mask=cfg.data.use_padding_mask,
        seed=cfg.seed,
        get_flow_matrix=get_flow_matrix,
        get_graph_data=get_graph_data,
        process_discovery_method=cfg.process.method,
        remove_duplicates=cfg.process.remove_duplicates,
        activity_names=cfg.process.activity_names,
    )


def create_denoiser(cfg: Config, flow_matrix=None, graph_data=None):
    """Create denoiser model based on config."""
    if cfg.model.type == "unet":
        return ConditionalUnetDenoiser(
            in_ch=cfg.model.n_channels,
            out_ch=cfg.model.n_channels,
            time_dim=cfg.model.time_dim,
        )
    elif cfg.model.type == "unet_matrix":
        return ConditionalUnetMatrixDenoiser(
            in_ch=cfg.model.n_channels,
            out_ch=cfg.model.n_channels,
            time_dim=cfg.model.time_dim,
            transition_dim=cfg.model.flow_matrix_dim,
            flow_matrix=flow_matrix,
            latent_flow_matrix=cfg.model.latent_matrix,
            matrix_out_channels=cfg.model.matrix_out_channels,
        )
    elif cfg.model.type == "unet_graph":
        return ConditionalUnetGraphDenoiser(
            in_ch=cfg.model.n_channels,
            out_ch=cfg.model.n_channels,
            time_dim=cfg.model.time_dim,
            graph_data=graph_data,
            embedding_dim=cfg.model.node_embedding_dim,
            hidden_dim=cfg.model.graph_hidden_dim,
            pooling=cfg.model.pooling,
        )
    else:
        raise ValueError(f"Unknown model type: {cfg.model.type}")


def create_diffusion(sampler, cfg: Config) -> Optional[DDPM | DDIM]:
    """Create diffusion process based on config."""
    if sampler == "ddpm":
        return DDPM(
            noise_steps=cfg.diffusion.noise_steps,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
        )
    elif sampler == "ddim":
        return DDIM(
            noise_steps=cfg.diffusion.noise_steps,
            inference_steps=cfg.diffusion.ddim_inference_steps,
            eta=cfg.diffusion.ddim_eta,
            beta_start=cfg.diffusion.beta_start,
            beta_end=cfg.diffusion.beta_end,
        )
    return None


def create_loggers(cfg: Config, save_dir: Path) -> List[Logger]:
    """Create experiment loggers based on config.
    
    Lightning Trainer accepts a list of loggers and broadcasts every
    ``self.log`` / ``self.log_dict`` call to all of them automatically.
    """
    loggers: List[Logger] = []
    for name in cfg.logging.loggers:
        if name == "tensorboard":
            loggers.append(TensorBoardLogger(
                save_dir=str(save_dir),
                name=cfg.logging.project_name,
                version=cfg.logging.version,
            ))
        elif name == "mlflow":
            loggers.append(MLFlowLogger(
                experiment_name=cfg.logging.experiment_name,
                run_name=cfg.logging.run_name,
                tracking_uri=cfg.logging.mlflow_tracking_uri or "mlruns",
                save_dir=str(save_dir),
            ))
        elif name == "wandb":
            loggers.append(WandbLogger(
                entity=cfg.logging.wandb_entity,
                project=cfg.logging.project_name,
                offline=cfg.logging.wandb_offline,
                name=cfg.logging.run_name,
                version=cfg.logging.version,
            ))
        else:
            raise ValueError(f"Unknown logger: {name}")

    return loggers
