import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

from .base_diffusion import BaseDiffusion
from src.modules.learnable_hybrid_loss import LearnableHybridLoss
from src.modules.hybrid_loss import HybridLoss
from src.utils.metrics import create_metric_collection


class DiffusionLightningModule(L.LightningModule):
    """
    Lightning module that wraps a denoiser with a diffusion process.
    
    Handles the forward diffusion (noising) during training and 
    reverse diffusion (sampling) during inference.
    
    Supports denoisers with multiple outputs following the convention that
    the PRIMARY diffusion output is ALWAYS FIRST. Auxiliary outputs (e.g.,
    matrix predictions) follow.
    
    IMPORTANT: Metrics are computed on the FULL reverse diffusion output,
    not on single-step denoising predictions. This means:
    - Training loss: computed on single-step predictions (efficient)
    - Validation/Test metrics: computed after running full reverse diffusion (expensive)
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        diffusion: BaseDiffusion,
        eval_diffusion: BaseDiffusion,
        optimizer: Literal["adamw", "adam", "sgd"] = "adamw",
        learning_rate: Optional[float] = 1e-4,
        weight_decay: Optional[float] = 0.0,
        scheduler: Literal["cosine", "step", "none"] = "cosine",
        warmup_epochs: Optional[int] = 0,
        loss_type: Literal["mse", "l1", "cross_entropy", "hybrid", "learnable_hybrid"] = "cross_entropy",
        gamma: float = 1.0,
        denoiser_output: Literal["noise", "original"] = "original",
        conditional_dropout: float = 0.0,
        denoiser_config: Optional[Dict[str, Any]] = None,
        padding_value: Optional[int] = None,
        # Metrics configuration
        num_classes: Optional[int] = None,
        val_metrics: Optional[List[str]] = None,
        test_metrics: Optional[List[str]] = None,
        # Verbose test mode (trajectory analysis during inference)
        verbose_test: bool = False,  # Enable trajectory evaluation during test
        trajectory_metrics: Optional[List[str]] = None,  # Metrics to compute along trajectory
        trajectory_save_every: int = 10,  # Evaluate every N steps along trajectory
    ):
        """
        Args:
            denoiser: Neural network that predicts noise or original from noisy input.
                      Returns tensor or tuple (primary_output, *auxiliary_outputs).
            diffusion: Diffusion process (DDPM, DDIM, etc.). If None, creates DDPM.
            noise_steps: Number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function for primary output
            denoiser_output: What the denoiser predicts - 'noise' or 'original'
            conditional_dropout: Probability of dropping conditioning during training
            auxiliary_losses: List of auxiliary loss configs for multi-output denoisers.
                Each dict has: output_index, loss_type, weight, target.
                Example for matrix denoiser: [{"output_index": 1, "loss_type": "bce_logits", 
                    "weight": 0.5, "target": "labels"}]
            denoiser_config: Config dict for reconstructing denoiser from checkpoint
            num_classes: Number of classes for metrics (required if using metrics)
            val_metrics: List of metric names to compute during validation
            test_metrics: List of metric names to compute during testing
            eval_every_n_epochs: Run full metrics evaluation every N epochs (0 = never)
            eval_use_ddim: Whether to use DDIM for faster evaluation sampling
            eval_ddim_steps: Number of DDIM steps when eval_use_ddim=True
            verbose_test: Enable trajectory evaluation during test (for inference analysis)
            trajectory_metrics: Metrics to compute at each point along reverse diffusion
            trajectory_save_every: Evaluate trajectory every N steps (1 = all steps)
        """
        super().__init__()
        
        # Store denoiser config for checkpoint loading
        if denoiser_config is None and denoiser is not None:
            denoiser_config = {
                "class": denoiser.__class__.__name__,
                "module": denoiser.__class__.__module__,
            }
            if hasattr(denoiser, 'time_dim'):
                denoiser_config["time_dim"] = denoiser.time_dim
            if hasattr(denoiser, 'max_input_dim'):
                denoiser_config["max_input_dim"] = denoiser.max_input_dim
        
        self.save_hyperparameters(ignore=['denoiser', 'diffusion', 'eval_diffusion'])
        
        self.denoiser = denoiser
        self.optimizer = optimizer
        self.diffusion = diffusion
        self.eval_diffusion = eval_diffusion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.denoiser_output = denoiser_output
        self.conditional_dropout = conditional_dropout
        self.gamma = gamma
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.verbose_test = verbose_test
        self.trajectory_metrics = trajectory_metrics or ["accuracy"]
        self.trajectory_save_every = trajectory_save_every
        
        self.loss_fn = self._create_loss_fn(loss_type)
        self.padding_value = padding_value
        
        # Setup metrics
        self._setup_metrics(
            num_classes=num_classes,
            val_metrics=val_metrics or [],
            test_metrics=test_metrics or [],
        )

        # Storage for trajectory analysis results (populated during verbose test)
        self.trajectory_results: List[Dict[str, Any]] = []

    def _create_loss_fn(self, loss_type: str) -> nn.Module:
        """Create a loss function from type string."""
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "hybrid":
            return HybridLoss(nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), self.gamma)
        elif loss_type == "learnable_hybrid":
            return LearnableHybridLoss(nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss(), self.gamma)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def _setup_metrics(
        self,
        num_classes: Optional[int],
        val_metrics: List[str],
        test_metrics: List[str],
    ):
        """Initialize metric collections for validation and test only."""
        self.val_metrics = None
        self.test_metrics = None
        
        if num_classes is None or num_classes <= 0:
            return
        
        if val_metrics:
            self.val_metrics = create_metric_collection(
                val_metrics, num_classes=num_classes, prefix="val/"
            )
        if test_metrics:
            self.test_metrics = create_metric_collection(
                test_metrics, num_classes=num_classes, prefix="test/"
            )

    def configure_optimizers(self):
        if self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        
        if self.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs if self.trainer else 100,
                eta_min=0.01 * self.learning_rate,
            )
        elif self.scheduler == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=50,
            )
        elif self.scheduler == "none":
            return optimizer
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def enable_verbose_test(
        self,
        metrics: Optional[List[str]] = None,
        save_every: int = 1,
    ):
        """Enable verbose test mode for trajectory analysis during inference."""
        self.verbose_test = True
        if metrics is not None:
            self.trajectory_metrics = metrics
        self.trajectory_save_every = save_every
        self.trajectory_results = []
    
    def disable_verbose_test(self):
        """Disable verbose test mode."""
        self.verbose_test = False
    
    def get_trajectory_dataframe(self):
        """Convert trajectory results to a pandas DataFrame for analysis."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for get_trajectory_dataframe()")
        
        rows = []
        for result in self.trajectory_results:
            batch_idx = result["batch_idx"]
            for step_data in result["trajectory"]:
                row = {"batch_idx": batch_idx, **step_data}
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass through denoiser."""
        return self.denoiser(x, t, y)
    
    def _update_metrics(
        self,
        sampled: torch.Tensor,
        labels: torch.Tensor,
        metrics_collection,
        stage: str,
    ):
        """Update metrics based on full reverse diffusion samples vs ground truth."""
        if metrics_collection is None:
            return
        
        # Get predictions as class probabilities and indices
        pred_probs = F.softmax(sampled, dim=1)  # (B, C, L)
        pred_classes = torch.argmax(pred_probs, dim=1)  # (B, L)
        
        # Flatten for metrics: (B*L, C) and (B*L,)
        batch_size, num_classes, seq_len = sampled.shape
        pred_probs_flat = pred_probs.permute(0, 2, 1).reshape(-1, num_classes)
        pred_classes_flat = pred_classes.reshape(-1)
        labels_flat = labels.reshape(-1).long()
        
        # Update metrics
        for name, metric in metrics_collection.items():
            metric_name = name.replace(f"{stage}/", "")
            if metric_name in ["auroc", "safe_auroc", "auroc_weighted"]:
                metric.update(pred_probs_flat, labels_flat)
            elif metric_name == "wasserstein":
                metric.update(pred_classes, labels.long())
            else:
                metric.update(pred_classes_flat, labels_flat)
    
    def _run_full_reverse_diffusion(
        self,
        conditioning: torch.Tensor,
        shape: tuple,
    ) -> torch.Tensor:
        """Run full reverse diffusion process for evaluation."""
        batch_size = conditioning.shape[0]
        model = self._get_primary_output_model()
        
        return self.eval_diffusion.sample(
            model=model,
            sample_size=batch_size,
            shape=shape,
            y=conditioning,
            denoiser_output=self.denoiser_output,
        )

    def training_step(self, batch, batch_idx):
        """
        Training step with forward diffusion (Algorithm 1).
        
        Computes single-step denoising loss. Supports multi-output denoisers
        by extracting primary output for loss computation.
        
        Expected batch format: (labels, data) where data is (B, L, C).
        """
        x, y = batch
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.noise_data(x, t)
        
        if torch.rand(1).item() < self.conditional_dropout:
            y = None
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x

        if self.loss_type in {"hybrid", "learnable_hybrid"}:
            target = (target, self.denoiser.gt_flow_matrix)
        
        denoiser_out = self.denoiser(x_t, t, y)
        if self.loss_type in {"cross_entropy", "hybrid", "learnable_hybrid"}:
            loss = self.loss_fn(denoiser_out, target, ignore_index=self.padding_value)
        else:
            loss = self.loss_fn(denoiser_out, target)
        self.log("train_loss", loss, prog_bar=True)
        if self.loss_type == "hybrid":
            self.log("mixture_ratio", self.gamma, prog_bar=False)
        elif self.loss_type == "learnable_hybrid":
            self.log("mixture_ratio", torch.sigmoid(self.gamma), prog_bar=False)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step."""
        grad_norm = self._compute_grad_norm()
        if grad_norm is not None:
            self.log("grad_norm", grad_norm, prog_bar=False)
    
    def _compute_grad_norm(self, norm_type: float = 2.0) -> Optional[torch.Tensor]:
        """Compute the total gradient norm across all parameters."""
        parameters = [p for p in self.parameters() if p.grad is not None]
        if len(parameters) == 0:
            return None
        
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
            norm_type
        )
        return total_norm
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        - Every epoch: Compute single-step loss (cheap, for monitoring)
        - Every N epochs: Run full reverse diffusion and compute metrics (expensive)
        """
        x, y = batch
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        
        t = self.diffusion.sample_timesteps(x.shape[0])
        x_t, noise = self.diffusion.noise_data(x, t)
        
        if torch.rand(1).item() < self.conditional_dropout:
            y = None
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x

        if self.loss_type in {"hybrid", "learnable_hybrid"}:
            target = (target, self.denoiser.gt_flow_matrix)
        
        denoiser_out = self.denoiser(x_t, t, y)
        if self.loss_type in {"cross_entropy", "hybrid", "learnable_hybrid"}:
            loss = self.loss_fn(denoiser_out, target, ignore_index=self.padding_value)
        else:
            loss = self.loss_fn(denoiser_out, target)
        
        self.log("train_loss", loss, prog_bar=True)
        if self.loss_type == "hybrid":
            self.log("mixture_ratio", self.gamma, prog_bar=False)
        elif self.loss_type == "learnable_hybrid":
            self.log("mixture_ratio", torch.sigmoid(self.gamma), prog_bar=False)

        # Run full reverse diffusion for metrics only on eval epochs
        if self.val_metrics is not None:
            shape = (x.shape[1], x.shape[2])
            
            with torch.no_grad():
                sampled = self._run_full_reverse_diffusion(conditioning=x, shape=shape)
            
            self._update_metrics(sampled, x, self.val_metrics, "val")
        
        return loss
    
    def on_validation_epoch_end(self):
        """Log validation metrics at epoch end (only on eval epochs)."""
        if self.val_metrics is not None and self._is_eval_epoch():
            metrics = self.val_metrics.compute()
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            self.val_metrics.reset()
    
    def test_step(self, batch, batch_idx):
        """
        Test step with full reverse diffusion for metrics.
        
        When verbose_test is enabled, also evaluates metrics at each point
        along the reverse diffusion trajectory for analysis.
        """
        x, y = batch
        x = x.permute(0, 2, 1).float()  # (B, C, L)
        
        x_pred = self.eval_diffusion.sample(self.denosier, x.shape[0], (x.shape[1], x.shape[2]), y, self.denoiser_output)
        
        # Verbose test: evaluate entire trajectory
        if self.verbose_test and self.num_classes is not None:
            batch_trajectory = self.evaluate_trajectory(
                ground_truth=x,
                labels=labels,
                conditioning=x,
                metric_names=self.trajectory_metrics,
                save_every=self.trajectory_save_every,
            )
            self.trajectory_results.append({
                "batch_idx": batch_idx,
                "trajectory": batch_trajectory,
            })
        # Standard test: just final sample metrics
        elif self.test_metrics is not None:
            shape = (x.shape[1], x.shape[2])
            
            with torch.no_grad():
                sampled = self._run_full_reverse_diffusion(conditioning=x, shape=shape)
            
            self._update_metrics(sampled, x, labels, self.test_metrics, "test")
        
        return loss
    
    def on_test_epoch_start(self):
        """Clear trajectory results at the start of test epoch."""
        self.trajectory_results = []
    
    def on_test_epoch_end(self):
        """Log test metrics at epoch end."""
        if self.test_metrics is not None:
            metrics = self.test_metrics.compute()
            self.log_dict(metrics, prog_bar=True, sync_dist=True)
            self.test_metrics.reset()
        
        if self.verbose_test and self.trajectory_results:
            self._log_trajectory_summary()
    
    def _log_trajectory_summary(self):
        """Aggregate and log trajectory metrics across all test batches."""
        if not self.trajectory_results:
            return
        
        all_trajectories = [r["trajectory"] for r in self.trajectory_results]
        timesteps = [step["timestep"] for step in all_trajectories[0]]
        
        aggregated = {}
        for step_idx, t in enumerate(timesteps):
            step_metrics = {}
            for traj in all_trajectories:
                if step_idx < len(traj):
                    for key, value in traj[step_idx].items():
                        if key not in ["timestep", "step"]:
                            if key not in step_metrics:
                                step_metrics[key] = []
                            step_metrics[key].append(value)
            
            for key, values in step_metrics.items():
                metric_key = f"traj_t{t}/{key}"
                if metric_key not in aggregated:
                    aggregated[metric_key] = sum(values) / len(values)
        
        if aggregated:
            self.log_dict(aggregated, sync_dist=True)
        
        # Log final timestep metrics prominently
        if all_trajectories and all_trajectories[0]:
            final_step = all_trajectories[0][-1]
            final_metrics = {}
            for key in final_step:
                if key not in ["timestep", "step"]:
                    values = [t[-1][key] for t in all_trajectories if t and key in t[-1]]
                    if values:
                        final_metrics[f"traj_final/{key}"] = sum(values) / len(values)
            if final_metrics:
                self.log_dict(final_metrics, prog_bar=True, sync_dist=True)
    
    def sample(
        self,
        batch_size: int,
        shape: tuple,
        y: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using reverse diffusion."""
        model = self._get_primary_output_model()
        return self.diffusion.sample(
            model=model,
            sample_size=batch_size,
            shape=shape,
            y=y,
            denoiser_output=self.denoiser_output,
        )
    
    def sample_with_trajectory(
        self,
        batch_size: int,
        shape: tuple,
        y: Optional[torch.Tensor] = None,
        save_every: int = 1,
        use_eval_diffusion: bool = True,
    ):
        """Generate samples and return the entire trajectory."""
        model = self._get_primary_output_model()
        diffusion = self.eval_diffusion if use_eval_diffusion else self.diffusion
        return diffusion.sample_with_trajectory(
            model=model,
            sample_size=batch_size,
            shape=shape,
            y=y,
            denoiser_output=self.denoiser_output,
            save_every=save_every,
        )
    
    def sample_generator(
        self,
        batch_size: int,
        shape: tuple,
        y: Optional[torch.Tensor] = None,
        use_eval_diffusion: bool = True,
    ):
        """Generator that yields (timestep, x_t) at each reverse diffusion step."""
        model = self._get_primary_output_model()
        diffusion = self.eval_diffusion if use_eval_diffusion else self.diffusion
        yield from diffusion.reverse_diffusion_generator(
            model=model,
            sample_size=batch_size,
            shape=shape,
            y=y,
            denoiser_output=self.denoiser_output,
        )
    
    def evaluate_trajectory(
        self,
        ground_truth: torch.Tensor,
        labels: torch.Tensor,
        conditioning: torch.Tensor,
        metric_names: List[str],
        save_every: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Evaluate metrics at every point along the reverse diffusion trajectory.
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be set to evaluate trajectory")
        
        results = []
        shape = (ground_truth.shape[1], ground_truth.shape[2])
        batch_size = ground_truth.shape[0]
        
        metrics = create_metric_collection(
            metric_names,
            num_classes=self.num_classes,
            prefix="traj/",
        )
        metrics = metrics.to(self.device)
        
        for i, (t, x_t) in enumerate(self.sample_generator(
            batch_size=batch_size,
            shape=shape,
            y=conditioning,
            use_eval_diffusion=True,
        )):
            if i % save_every != 0:
                continue
            
            metrics.reset()
            self._update_metrics(x_t, ground_truth, labels, metrics, "traj")
            
            step_metrics = metrics.compute()
            results.append({
                "timestep": t,
                "step": i,
                **{k.replace("traj/", ""): v.item() if hasattr(v, 'item') else v 
                   for k, v in step_metrics.items()}
            })
        
        return results
    
    @classmethod
    def load_from_checkpoint_with_denoiser(
        cls,
        checkpoint_path: str,
        denoiser: nn.Module,
        map_location=None,
        **kwargs,
    ):
        """
        Load model from checkpoint with a provided denoiser.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint.get("hyper_parameters", {})
        model = cls(denoiser=denoiser, **hparams, **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model
