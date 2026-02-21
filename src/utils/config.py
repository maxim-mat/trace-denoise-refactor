from dataclasses import dataclass, field
from typing import Optional, Literal, List
from omegaconf import MISSING


@dataclass
class DataConfig:
    """Data configuration."""
    path: str = MISSING
    num_classes: int = MISSING
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False  # recommnded False for local, True for server
    # if set , all traces will be padded to this length
    target_length: Optional[int] = None


@dataclass
class ModelConfig:
    """Model/denoiser configuration."""
    type: str = "unet"  # "unet", "unet_matrix", "unet_graph"
    loss_function: str = "cross_entropy"  # "mse", "l1", "cross_entropy", "hybrid"
    gamma: Optional[float] = 1.0  # weight of main loss for hybrid loss
    time_dim: int = 128
    conditional_dropout: Optional[float] = 0.2
    # Parameters for advanced denoisers
    latent_matrix: Optional[bool] = True
    transition_dim: Optional[int] = 100  # shape of flow matrix
    node_embedding_dim: Optional[int] = 128
    graph_hidden_dim: Optional[int] = 128
    pooling: Optional[str] = None  # "mean", "max", "add"


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampler: str = "ddpm"  # "ddpm", "ddim"
    denoiser_output: str = "original"  # "noise", "original"
    eval_use_ddim: bool = False
    # DDIM specific
    ddim_inference_steps: Optional[int] = 50
    ddim_eta: Optional[float] = 0.0


@dataclass
class ProcessConfig:
    """Process discovery configuration."""
    method: Optional[str] = None  # "inductive", "heuristic", "fuzzy"
    remove_duplicates: Optional[bool] = True
    activity_names: Optional[list[str]] = None


@dataclass
class TrainerConfig:
    """Lightning Trainer configuration."""
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"  # "32", "16-mixed", "bf16-mixed"
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    deterministic: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    method: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    scheduler: str = "cosine"  # "cosine", "step", "none"
    warmup_epochs: Optional[int] = 0


@dataclass
class CallbacksConfig:
    """Callbacks configuration."""
    early_stopping: bool = True
    early_stopping_patience: int = 10
    early_stopping_monitor: str = "val_loss"
    early_stopping_mode: str = "min"
    save_top_k: int = 3
    save_last: bool = True


@dataclass
class LoggingConfig:
    """Experiment logging configuration."""
    logger: str = "tensorboard"  # "tensorboard", "mlflow", "wandb", "none"
    project_name: str = "trace-denoise"
    experiment_name: Optional[str] = None
    save_dir: str = MISSING
    # MLflow specific
    mlflow_tracking_uri: Optional[str] = None
    # W&B specific
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False


@dataclass
class MetricsConfig:
    """
    Metrics configuration.
    
    IMPORTANT: Metrics are computed on FULL reverse diffusion samples, not
    single-step denoising predictions. This means validation/test runs the
    complete sampling process (Algorithm 2) which is computationally expensive.
    
    Available metrics:
    - accuracy: Micro-averaged accuracy
    - accuracy_macro: Macro-averaged accuracy
    - precision: Macro-averaged precision
    - recall: Macro-averaged recall
    - f1: Macro-averaged F1 score
    - auroc: Macro-averaged AUROC (may fail if not all classes present)
    - safe_auroc: AUROC that returns -1 instead of failing
    - auroc_weighted: Weighted AUROC
    - wasserstein: Wasserstein-1 distance between sequences
    - confusion_matrix: Normalized confusion matrix
    """
    # Metrics to compute during validation (requires full reverse diffusion)
    val: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "safe_auroc"])
    # Metrics to compute during testing
    test: List[str] = field(default_factory=lambda: ["accuracy", "precision", "recall", "f1", "safe_auroc", "wasserstein"])
    verbose_trajectory: bool = False
    trajectory_save_every: int = 5


@dataclass
class Config:
    """Root configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    process: Optional[ProcessConfig] = field(default_factory=ProcessConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    seed: int = 42
    
    # Resume/checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    checkpoint_path: Optional[str] = None  # For inference: path to model checkpoint
