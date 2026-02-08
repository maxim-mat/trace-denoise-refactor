import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from typing import Literal, Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

from .base_diffusion import BaseDiffusion
from .ddpm import DDPM
from .ddim import DDIM
from src.utils.metrics import create_metric_collection


@dataclass
class AuxiliaryLossSpec:
    """Runtime specification for an auxiliary loss (parsed from config)."""
    output_index: int  # Index in auxiliary outputs tuple
    loss_fn: nn.Module  # Loss function instance
    weight: float  # Weight for this loss
    target: str  # "ground_truth" or "labels"


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
        diffusion: Optional[BaseDiffusion] = None,
        noise_steps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        learning_rate: float = 1e-4,
        loss_type: Literal["mse", "l1", "cross_entropy"] = "cross_entropy",
        denoiser_output: Literal["noise", "original"] = "original",
        conditional_dropout: float = 0.0,
        # Auxiliary losses for multi-output denoisers
        auxiliary_losses: Optional[List[Dict[str, Any]]] = None,
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
        
        # Set up primary loss function
        self.loss_fn = self._create_loss_fn(loss_type)
        
        # Set up auxiliary losses for multi-output denoisers
        self.auxiliary_loss_specs: List[AuxiliaryLossSpec] = []
        if auxiliary_losses:
            for aux_cfg in auxiliary_losses:
                self.auxiliary_loss_specs.append(AuxiliaryLossSpec(
                    output_index=aux_cfg.get("output_index", 1),
                    loss_fn=self._create_loss_fn(aux_cfg.get("loss_type", "mse")),
                    weight=aux_cfg.get("weight", 1.0),
                    target=aux_cfg.get("target", "ground_truth"),
                ))
        
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
    
    def _create_loss_fn(self, loss_type: str) -> nn.Module:
        """Create a loss function from type string."""
        if loss_type == "mse":
            return nn.MSELoss()
        elif loss_type == "l1":
            return nn.L1Loss()
        elif loss_type == "cross_entropy":
            return nn.CrossEntropyLoss()
        elif loss_type == "bce":
            return nn.BCELoss()
        elif loss_type == "bce_logits":
            return nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
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
                    eta=0.0,
                    beta_start=self.beta_start,
                    beta_end=self.beta_end,
                    device=self.device,
                )
            else:
                self._eval_diffusion = self.diffusion
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
    
    def _extract_primary_output(
        self,
        denoiser_output: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        """
        Extract the primary output from denoiser prediction.
        
        Handles both single-tensor outputs and multi-output denoisers.
        Convention: primary diffusion output is always first.
        
        Args:
            denoiser_output: Either a tensor or a tuple (primary, *auxiliaries)
            
        Returns:
            Tuple of (primary_output, auxiliary_outputs) where auxiliary_outputs
            may be None for single-output denoisers.
        """
        if isinstance(denoiser_output, tuple):
            return denoiser_output[0], denoiser_output[1:]
        return denoiser_output, None
    
    def _compute_auxiliary_loss(
        self,
        auxiliary_outputs: Optional[Tuple[torch.Tensor, ...]],
        ground_truth: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary losses based on configuration.
        
        Losses are configured via auxiliary_losses parameter, not in the denoiser.
        This keeps denoisers as pure architectures.
        
        Args:
            auxiliary_outputs: Tuple of auxiliary outputs from denoiser
            ground_truth: The original clean data (x)
            labels: Batch labels (for auxiliary outputs that compare against labels)
            
        Returns:
            Total weighted auxiliary loss (0 if no auxiliary outputs or no configs)
        """
        if auxiliary_outputs is None or not self.auxiliary_loss_specs:
            return torch.tensor(0.0, device=self.device)
        
        total_aux_loss = torch.tensor(0.0, device=self.device)
        
        for spec in self.auxiliary_loss_specs:
            # Get the auxiliary output by index (1-indexed: 1 means first auxiliary)
            aux_idx = spec.output_index - 1
            if aux_idx < 0 or aux_idx >= len(auxiliary_outputs):
                continue
            
            aux_output = auxiliary_outputs[aux_idx]
            
            # Determine target based on config
            if spec.target == "ground_truth":
                target = ground_truth
            else:  # "labels"
                # For matrix denoiser: labels may need to be broadcast to match output shape
                target = labels
                # Handle shape mismatch (e.g., matrix output shape differs from labels)
                if aux_output.shape != target.shape:
                    # Try to broadcast labels to match auxiliary output
                    if aux_output.dim() == 4 and target.dim() <= 2:
                        # Matrix case: aux_output is (B, C, H, W), labels might need expansion
                        target = target.unsqueeze(1).expand_as(aux_output).float()
                    elif aux_output.dim() == 4:
                        target = target.expand_as(aux_output).float()
            
            # Compute weighted loss
            aux_loss = spec.loss_fn(aux_output, target)
            total_aux_loss = total_aux_loss + spec.weight * aux_loss
        
        return total_aux_loss
    
    def _update_metrics(
        self,
        sampled: torch.Tensor,
        ground_truth: torch.Tensor,
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
    
    def _get_primary_output_model(self):
        """
        Get a wrapper around the denoiser that always returns primary output only.
        
        This is needed for sampling, where the diffusion process expects
        a single tensor output from the model.
        """
        denoiser = self.denoiser
        extract_fn = self._extract_primary_output
        
        class PrimaryOutputWrapper:
            """Wrapper that extracts primary output from multi-output denoisers."""
            def __init__(self, model, extract_primary):
                self.model = model
                self.extract_primary = extract_primary
            
            def __call__(self, x, t, y=None):
                output = self.model(x, t, y)
                primary, _ = self.extract_primary(output)
                return primary
            
            def eval(self):
                self.model.eval()
                return self
            
            def train(self, mode=True):
                self.model.train(mode)
                return self
        
        return PrimaryOutputWrapper(denoiser, extract_fn)
    
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
        labels, data = batch
        x = data.permute(0, 2, 1).float()  # (B, C, L)
        
        # Sample timesteps
        t = self.sample_timesteps(x.shape[0])
        
        # Forward diffusion: add noise
        x_t, noise = self.noise_data(x, t)
        
        # Apply conditional dropout
        y = None
        if self.conditional_dropout > 0 and torch.rand(1).item() < self.conditional_dropout:
            y = None
        else:
            y = x
        
        # Get target based on denoiser_output setting
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x
        
        # Predict and extract primary output
        denoiser_out = self.denoiser(x_t, t, y)
        predicted, auxiliary_outputs = self._extract_primary_output(denoiser_out)
        
        # Compute primary loss
        primary_loss = self.loss_fn(predicted, target)
        
        # Compute auxiliary loss based on config (not from denoiser)
        auxiliary_loss = self._compute_auxiliary_loss(auxiliary_outputs, x, labels)
        
        # Total loss
        loss = primary_loss + auxiliary_loss
        
        self.log("train_loss", loss, prog_bar=True)
        if self.auxiliary_loss_specs:
            self.log("train_primary_loss", primary_loss, prog_bar=False)
            self.log("train_auxiliary_loss", auxiliary_loss, prog_bar=False)
        
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
    
    def _is_eval_epoch(self) -> bool:
        """Check if current epoch should run full evaluation."""
        if self.eval_every_n_epochs <= 0:
            return False
        return self.current_epoch % self.eval_every_n_epochs == 0
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        
        - Every epoch: Compute single-step loss (cheap, for monitoring)
        - Every N epochs: Run full reverse diffusion and compute metrics (expensive)
        """
        labels, data = batch
        x = data.permute(0, 2, 1).float()
        
        # Compute single-step loss
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_data(x, t)
        
        denoiser_out = self.denoiser(x_t, t, x)
        predicted, _ = self._extract_primary_output(denoiser_out)
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x
        
        loss = self.loss_fn(predicted, target)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        
        # Run full reverse diffusion for metrics only on eval epochs
        if self.val_metrics is not None and self._is_eval_epoch():
            shape = (x.shape[1], x.shape[2])
            
            with torch.no_grad():
                sampled = self._run_full_reverse_diffusion(conditioning=x, shape=shape)
            
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
        x = data.permute(0, 2, 1).float()
        
        # Compute single-step loss
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_data(x, t)
        
        denoiser_out = self.denoiser(x_t, t, x)
        predicted, _ = self._extract_primary_output(denoiser_out)
        
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
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        hparams = checkpoint.get("hyper_parameters", {})
        model = cls(denoiser=denoiser, **hparams, **kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        return model
