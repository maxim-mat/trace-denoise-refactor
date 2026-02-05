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
    pin_memory: bool = True
    target_length: Optional[int] = None


@dataclass
class ModelConfig:
    """Model/denoiser configuration."""
    type: Literal["unet", "unet_matrix", "unet_graph"] = "unet"
    time_dim: int = 128
    max_input_dim: int = MISSING  # Usually set from data


@dataclass
class DiffusionConfig:
    """Diffusion process configuration."""
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sampler: Literal["ddpm", "ddim"] = "ddpm"
    ddim_inference_steps: int = 50
    ddim_eta: float = 0.0
    denoiser_output: Literal["noise", "original"] = "original"
    conditional_dropout: float = 0.0
    loss_type: Literal["mse", "l1", "cross_entropy"] = "cross_entropy"


@dataclass
class TrainerConfig:
    """Lightning Trainer configuration."""
    max_epochs: int = 100
    accelerator: str = "auto"
    devices: int = 1
    precision: Literal["32", "16-mixed", "bf16-mixed"] = "32"
    gradient_clip_val: Optional[float] = 1.0
    accumulate_grad_batches: int = 1
    val_check_interval: float = 1.0
    log_every_n_steps: int = 50
    deterministic: bool = False


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    warmup_epochs: int = 0


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
    logger: Literal["tensorboard", "mlflow", "wandb", "none"] = "tensorboard"
    project_name: str = "trace-denoise"
    experiment_name: Optional[str] = None
    save_dir: str = "outputs"
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
    # Run full evaluation every N epochs (0 = never during training, only at test)
    eval_every_n_epochs: int = 10
    # Use DDIM for faster evaluation sampling (recommended)
    eval_use_ddim: bool = True
    # Number of DDIM steps for evaluation (fewer = faster but potentially lower quality)
    eval_ddim_steps: int = 50
    # Verbose trajectory evaluation during inference (analyze reverse diffusion at each step)
    verbose_trajectory: bool = False
    # Evaluate trajectory metrics every N steps (1 = all steps, higher = faster)
    trajectory_save_every: int = 5


@dataclass
class Config:
    """Root configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    callbacks: CallbacksConfig = field(default_factory=CallbacksConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    
    mode: Literal["train", "inference"] = "train"
    seed: int = 42
    
    # Resume/checkpoint
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    checkpoint_path: Optional[str] = None  # For inference: path to model checkpoint
