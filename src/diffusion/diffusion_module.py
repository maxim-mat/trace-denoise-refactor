import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Optional, Dict, Any, List

from .base_diffusion import BaseDiffusion
from .ddpm import DDPM
from .ddim import DDIM
from src.utils.metrics import create_metric_collection


class DiffusionLightningModule(L.LightningModule):
    """
    Lightning module that wraps a denoiser with a diffusion process.
    
    Handles the forward diffusion (noising) during training and 
    reverse diffusion (sampling) during inference.
    
    IMPORTANT: Metrics are computed on the FULL reverse diffusion output,
    not on single-step denoising predictions. This means:
    - Training loss: computed on single-step predictions (efficient)
    - Validation/Test metrics: computed after running full reverse diffusion (expensive)
    
    This is necessary because for inverse problems, the quality of the solution
    can only be assessed after the complete sampling process.
    """
    
    def __init__(
        self,
        denoiser: nn.Module,
        diffusion: Optional[BaseDiffusion] = None,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        learning_rate: float = 1e-4,
        loss_type: Literal["mse", "l1", "cross_entropy"] = "cross_entropy",
        denoiser_output: Literal["noise", "original"] = "original",
        conditional_dropout: float = 0.0,
        # Denoiser config for checkpoint restoration
        denoiser_config: Optional[Dict[str, Any]] = None,
        # Metrics configuration
        num_classes: Optional[int] = None,
        val_metrics: Optional[List[str]] = None,
        test_metrics: Optional[List[str]] = None,
        # Evaluation settings (full reverse diffusion is expensive)
        eval_every_n_epochs: int = 10,  # Run full eval every N epochs (0 = never)
        eval_use_ddim: bool = True,  # Use DDIM for faster evaluation
        eval_ddim_steps: int = 50,  # Number of DDIM steps for evaluation
        # Verbose test mode (trajectory analysis during inference)
        verbose_test: bool = False,  # Enable trajectory evaluation during test
        trajectory_metrics: Optional[List[str]] = None,  # Metrics to compute along trajectory
        trajectory_save_every: int = 1,  # Evaluate every N steps along trajectory
    ):
        """
        Args:
            denoiser: Neural network that predicts noise or original from noisy input
            diffusion: Diffusion process (DDPM, DDIM, etc.). If None, creates DDPM.
            noise_steps: Number of diffusion steps
            beta_start: Starting value for noise schedule
            beta_end: Ending value for noise schedule
            learning_rate: Learning rate for optimizer
            loss_type: Type of loss function
            denoiser_output: What the denoiser predicts - 'noise' or 'original'
            conditional_dropout: Probability of dropping conditioning during training
            denoiser_config: Config dict for reconstructing denoiser from checkpoint
            num_classes: Number of classes for metrics (required if using metrics)
            val_metrics: List of metric names to compute during validation
            test_metrics: List of metric names to compute during testing
            eval_every_n_epochs: Run full metrics evaluation every N epochs (0 = never during training)
            eval_use_ddim: Whether to use DDIM for faster evaluation sampling
            eval_ddim_steps: Number of DDIM steps when eval_use_ddim=True
            verbose_test: Enable trajectory evaluation during test (for inference analysis)
            trajectory_metrics: Metrics to compute at each point along reverse diffusion
            trajectory_save_every: Evaluate trajectory every N steps (1 = all steps)
        """
        super().__init__()
        
        # Store denoiser config for checkpoint loading
        # Extract from denoiser if not provided
        if denoiser_config is None and denoiser is not None:
            denoiser_config = {
                "class": denoiser.__class__.__name__,
                "module": denoiser.__class__.__module__,
            }
            # Try to get init args from denoiser if it has them
            if hasattr(denoiser, 'time_dim'):
                denoiser_config["time_dim"] = denoiser.time_dim
            if hasattr(denoiser, 'max_input_dim'):
                denoiser_config["max_input_dim"] = denoiser.max_input_dim
        
        self.save_hyperparameters(ignore=['denoiser', 'diffusion'])
        
        self.denoiser = denoiser
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.learning_rate = learning_rate
        self.denoiser_output = denoiser_output
        self.conditional_dropout = conditional_dropout
        self.loss_type = loss_type
        self.num_classes = num_classes
        self.eval_every_n_epochs = eval_every_n_epochs
        self.eval_use_ddim = eval_use_ddim
        self.eval_ddim_steps = eval_ddim_steps
        self.verbose_test = verbose_test
        self.trajectory_metrics = trajectory_metrics or ["accuracy"]
        self.trajectory_save_every = trajectory_save_every
        
        # Storage for trajectory analysis results (populated during verbose test)
        self.trajectory_results: List[Dict[str, Any]] = []
        
        # Set up loss function
        if loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "l1":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "cross_entropy":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Store diffusion process (will be initialized on first forward if None)
        self._diffusion = diffusion
        self._eval_diffusion = None  # Separate diffusion for evaluation (DDIM)
        
        # Noise schedule parameters (registered as buffers for device handling)
        self._setup_noise_schedule()
        
        # Setup metrics
        self._setup_metrics(
            num_classes=num_classes,
            val_metrics=val_metrics or [],
            test_metrics=test_metrics or [],
        )
    
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
        
        # Note: No train_metrics - computing metrics during training would require
        # running full reverse diffusion which is too expensive
        if val_metrics:
            self.val_metrics = create_metric_collection(
                val_metrics, num_classes=num_classes, prefix="val/"
            )
        if test_metrics:
            self.test_metrics = create_metric_collection(
                test_metrics, num_classes=num_classes, prefix="test/"
            )
    
    def _setup_noise_schedule(self):
        """Set up noise schedule as buffers for automatic device placement."""
        beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        alpha = 1.0 - beta
        alpha_hat = torch.cumprod(alpha, dim=0)
        
        self.register_buffer('beta', beta)
        self.register_buffer('alpha', alpha)
        self.register_buffer('alpha_hat', alpha_hat)
    
    @property
    def diffusion(self) -> BaseDiffusion:
        """Get diffusion process, creating default DDPM if not set."""
        if self._diffusion is None:
            self._diffusion = DDPM(
                noise_steps=self.noise_steps,
                beta_start=self.beta_start,
                beta_end=self.beta_end,
                device=self.device,
            )
        # Update device if needed
        if hasattr(self._diffusion, 'device') and self._diffusion.device != self.device:
            self._diffusion.device = self.device
            self._diffusion.beta = self._diffusion.beta.to(self.device)
            self._diffusion.alpha = self._diffusion.alpha.to(self.device)
            self._diffusion.alpha_hat = self._diffusion.alpha_hat.to(self.device)
        return self._diffusion
    
    @property
    def eval_diffusion(self) -> BaseDiffusion:
        """Get diffusion process for evaluation (DDIM for speed if configured)."""
        if self._eval_diffusion is None:
            if self.eval_use_ddim:
                self._eval_diffusion = DDIM(
                    noise_steps=self.noise_steps,
                    inference_steps=self.eval_ddim_steps,
                    eta=0.0,  # Deterministic for evaluation
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                    device=self.device,
                )
            else:
                self._eval_diffusion = self.diffusion
        # Update device if needed
        if hasattr(self._eval_diffusion, 'device') and self._eval_diffusion.device != self.device:
            self._eval_diffusion.device = self.device
            self._eval_diffusion.beta = self._eval_diffusion.beta.to(self.device)
            self._eval_diffusion.alpha = self._eval_diffusion.alpha.to(self.device)
            self._eval_diffusion.alpha_hat = self._eval_diffusion.alpha_hat.to(self.device)
        return self._eval_diffusion
    
    def set_diffusion(self, diffusion: BaseDiffusion):
        """Set a different diffusion process (e.g., switch from DDPM to DDIM)."""
        self._diffusion = diffusion
    
    def set_eval_diffusion(self, diffusion: BaseDiffusion):
        """Set a different diffusion process for evaluation."""
        self._eval_diffusion = diffusion
    
    def enable_verbose_test(
        self,
        metrics: Optional[List[str]] = None,
        save_every: int = 1,
    ):
        """
        Enable verbose test mode for trajectory analysis during inference.
        
        Call this before running trainer.test() to analyze the reverse
        diffusion process at each timestep.
        
        Args:
            metrics: Metrics to compute (defaults to current trajectory_metrics)
            save_every: Evaluate every N steps (1 = all steps)
        """
        self.verbose_test = True
        if metrics is not None:
            self.trajectory_metrics = metrics
        self.trajectory_save_every = save_every
        self.trajectory_results = []
    
    def disable_verbose_test(self):
        """Disable verbose test mode."""
        self.verbose_test = False
    
    def get_trajectory_dataframe(self):
        """
        Convert trajectory results to a pandas DataFrame for analysis.
        
        Returns:
            DataFrame with columns: batch_idx, step, timestep, and all metrics
        """
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
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(1, self.noise_steps, (batch_size,), device=self.device)
    
    def noise_data(self, x: torch.Tensor, t: torch.Tensor):
        """
        Forward diffusion - add noise to data.
        
        Args:
            x: Clean data tensor (B, C, L)
            t: Timesteps (B,)
            
        Returns:
            Tuple of (noisy_x, noise)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        x_t = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_t, eps
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None):
        """Forward pass through denoiser."""
        return self.denoiser(x, t, y)
    
    def _update_metrics(
        self,
        sampled: torch.Tensor,
        ground_truth: torch.Tensor,
        labels: torch.Tensor,
        metrics_collection,
        stage: str,
    ):
        """
        Update metrics based on full reverse diffusion samples vs ground truth.
        
        Args:
            sampled: Sampled output from full reverse diffusion (B, C, L) - logits
            ground_truth: Ground truth clean data (B, C, L)
            labels: Ground truth class labels (B, L)
            metrics_collection: MetricCollection to update
            stage: 'val' or 'test'
        """
        if metrics_collection is None:
            return
        
        # sampled shape: (B, C, L) - these are logits from the reverse diffusion
        # labels shape: (B, L) - ground truth class indices
        
        # Get predictions as class probabilities and indices
        pred_probs = F.softmax(sampled, dim=1)  # (B, C, L)
        pred_classes = torch.argmax(pred_probs, dim=1)  # (B, L)
        
        # Flatten for metrics: (B*L, C) and (B*L,)
        batch_size, num_classes, seq_len = sampled.shape
        pred_probs_flat = pred_probs.permute(0, 2, 1).reshape(-1, num_classes)  # (B*L, C)
        pred_classes_flat = pred_classes.reshape(-1)  # (B*L,)
        labels_flat = labels.reshape(-1).long()  # (B*L,)
        
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
        """
        Run full reverse diffusion process for evaluation.
        
        Args:
            conditioning: Conditioning tensor (B, C, L) - the noisy/corrupted input
            shape: Shape of output (C, L)
            
        Returns:
            Sampled output (B, C, L)
        """
        batch_size = conditioning.shape[0]
        
        # Use eval_diffusion (DDIM for speed if configured)
        return self.eval_diffusion.sample(
            model=self.denoiser,
            sample_size=batch_size,
            shape=shape,
            y=conditioning,
            denoiser_output=self.denoiser_output,
        )

    def training_step(self, batch, batch_idx):
        """
        Training step with forward diffusion (Algorithm 1).
        
        Computes single-step denoising loss only. Metrics require full reverse
        diffusion and are computed during validation/test.
        
        Expected batch format: (labels, data) where data is (B, L, C).
        """
        labels, data = batch
        
        # Transpose from (B, L, C) to (B, C, L) for conv1d
        x = data.permute(0, 2, 1).float()
        
        # Sample timesteps
        t = self.sample_timesteps(x.shape[0])
        
        # Forward diffusion: add noise (Algorithm 1, line 4-5)
        x_t, noise = self.noise_data(x, t)
        
        # Apply conditional dropout
        y = None
        if self.conditional_dropout > 0 and torch.rand(1).item() < self.conditional_dropout:
            y = None
        else:
            # Use the original data as conditioning
            y = x
        
        # Predict (single denoising step)
        predicted = self.denoiser(x_t, t, y)
        
        # Compute loss based on what denoiser predicts
        if self.denoiser_output == "noise":
            target = noise
        else:  # "original"
            target = x
        
        loss = self.loss_fn(predicted, target)
        
        self.log("train_loss", loss, prog_bar=True)
        
        return loss
    
    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step."""
        # Compute gradient norm across all parameters
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
    
    def _is_eval_epoch(self) -> bool:
        """Check if current epoch should run full evaluation."""
        if self.eval_every_n_epochs <= 0:
            return False
        # current_epoch is 0-indexed, so epoch 0, N, 2N, etc. are eval epochs
        return self.current_epoch % self.eval_every_n_epochs == 0
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        - Every epoch: Compute single-step loss (cheap, for monitoring)
        - Every N epochs: Run full reverse diffusion and compute metrics (expensive)
        """
        labels, data = batch
        x = data.permute(0, 2, 1).float()  # Ground truth: (B, C, L)
        
        # Always compute single-step loss for monitoring training progress
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_data(x, t)
        predicted = self.denoiser(x_t, t, x)  # y=x for conditioning
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x
        
        loss = self.loss_fn(predicted, target)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        # Run full reverse diffusion for metrics only on eval epochs
        is_eval_epoch = self._is_eval_epoch()
        if self.val_metrics is not None and is_eval_epoch:
            shape = (x.shape[1], x.shape[2])  # (C, L)
            
            # Sample from full reverse diffusion with conditioning
            with torch.no_grad():
                sampled = self._run_full_reverse_diffusion(
                    conditioning=x,  # Use ground truth as conditioning
                    shape=shape,
                )
            
            # Update metrics: compare sampled output to ground truth
            self._update_metrics(sampled, x, labels, self.val_metrics, "val")
        
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
        labels, data = batch
        x = data.permute(0, 2, 1).float()  # Ground truth: (B, C, L)
        
        # Compute single-step loss
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_data(x, t)
        predicted = self.denoiser(x_t, t, x)
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x
        
        loss = self.loss_fn(predicted, target)
        self.log("test_loss", loss, prog_bar=True, sync_dist=True)
        
        # Verbose test: evaluate entire trajectory
        if self.verbose_test and self.num_classes is not None:
            batch_trajectory = self.evaluate_trajectory(
                ground_truth=x,
                labels=labels,
                conditioning=x,
                metric_names=self.trajectory_metrics,
                save_every=self.trajectory_save_every,
            )
            # Store with batch info for later analysis
            self.trajectory_results.append({
                "batch_idx": batch_idx,
                "trajectory": batch_trajectory,
            })
        # Standard test: just final sample metrics
        elif self.test_metrics is not None:
            shape = (x.shape[1], x.shape[2])  # (C, L)
            
            with torch.no_grad():
                sampled = self._run_full_reverse_diffusion(
                    conditioning=x,
                    shape=shape,
                )
            
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
        
            # Log aggregated trajectory metrics if verbose test was enabled
        if self.verbose_test and self.trajectory_results:
            self._log_trajectory_summary()
    
    def _log_trajectory_summary(self):
        """
        Aggregate and log trajectory metrics across all test batches.
        
        Computes mean metrics at each timestep across batches.
        """
        if not self.trajectory_results:
            return
        
        # Collect all trajectories
        all_trajectories = [r["trajectory"] for r in self.trajectory_results]
        
        # Find common timesteps (should be the same across batches)
        timesteps = [step["timestep"] for step in all_trajectories[0]]
        
        # Aggregate metrics per timestep
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
            
            # Compute mean for each metric
            for key, values in step_metrics.items():
                metric_key = f"traj_t{t}/{key}"
                if metric_key not in aggregated:
                    aggregated[metric_key] = sum(values) / len(values)
        
        # Log aggregated trajectory metrics
        if aggregated:
            self.log_dict(aggregated, sync_dist=True)
        
        # Also log final timestep metrics prominently
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
        """
        Generate samples using reverse diffusion.
        
        Args:
            batch_size: Number of samples to generate
            shape: Shape of each sample (C, L)
            y: Optional conditioning tensor
            
        Returns:
            Generated samples
        """
        return self.diffusion.sample(
            model=self.denoiser,
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
        """
        Generate samples and return the entire trajectory.
        
        Useful for:
        - Visualizing the reverse process
        - Analyzing intermediate quality
        - Research and debugging
        
        Args:
            batch_size: Number of samples to generate
            shape: Shape of each sample (C, L)
            y: Optional conditioning tensor
            save_every: Save every N steps (1 = all steps)
            use_eval_diffusion: Use eval_diffusion (DDIM) for speed
            
        Returns:
            Tuple of (final_sample, trajectory) where trajectory is list of (timestep, x_t)
        """
        diffusion = self.eval_diffusion if use_eval_diffusion else self.diffusion
        return diffusion.sample_with_trajectory(
            model=self.denoiser,
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
        """
        Generator that yields (timestep, x_t) at each reverse diffusion step.
        
        Memory-efficient way to inspect the reverse process without storing
        all intermediate states.
        
        Args:
            batch_size: Number of samples to generate
            shape: Shape of each sample (C, L)
            y: Optional conditioning tensor
            use_eval_diffusion: Use eval_diffusion (DDIM) for speed
            
        Yields:
            Tuple of (timestep, x_t) at each step
        """
        diffusion = self.eval_diffusion if use_eval_diffusion else self.diffusion
        yield from diffusion.reverse_diffusion_generator(
            model=self.denoiser,
            batch_size=batch_size,
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
        
        This is useful for understanding how sample quality evolves during
        the reverse process.
        
        Args:
            ground_truth: Ground truth clean data (B, C, L)
            labels: Ground truth class labels (B, L)
            conditioning: Conditioning tensor (B, C, L)
            metric_names: List of metric names to compute
            save_every: Evaluate every N steps (1 = all steps)
            
        Returns:
            List of dicts, each containing timestep and metrics at that step
        """
        if self.num_classes is None:
            raise ValueError("num_classes must be set to evaluate trajectory")
        
        results = []
        shape = (ground_truth.shape[1], ground_truth.shape[2])
        batch_size = ground_truth.shape[0]
        
        # Create a fresh metric collection for trajectory evaluation
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
            
            # Reset metrics for this timestep
            metrics.reset()
            
            # Update metrics with current sample
            self._update_metrics(x_t, ground_truth, labels, metrics, "traj")
            
            # Compute and store
            step_metrics = metrics.compute()
            results.append({
                "timestep": t,
                "step": i,
                **{k.replace("traj/", ""): v.item() if hasattr(v, 'item') else v 
                   for k, v in step_metrics.items()}
            })
        
        return results
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs if self.trainer else 100,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    
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
        
        This is useful when the denoiser architecture needs to be 
        reconstructed externally (e.g., from config).
        
        Args:
            checkpoint_path: Path to checkpoint file
            denoiser: Pre-constructed denoiser module
            map_location: Device to load checkpoint to
            **kwargs: Additional kwargs passed to load_from_checkpoint
            
        Returns:
            Loaded DiffusionLightningModule with denoiser weights restored
        """
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        
        # Get hyperparameters
        hparams = checkpoint.get("hyper_parameters", {})
        
        # Create instance with provided denoiser
        model = cls(denoiser=denoiser, **hparams, **kwargs)
        
        # Load state dict
        model.load_state_dict(checkpoint["state_dict"])
        
        return model