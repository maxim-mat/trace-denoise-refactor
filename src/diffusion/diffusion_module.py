import torch
import torch.nn as nn
import lightning as L
from typing import Literal, Optional, Dict, Any

from .base_diffusion import BaseDiffusion
from .ddpm import DDPM
from .ddim import DDIM


class DiffusionLightningModule(L.LightningModule):
    """
    Lightning module that wraps a denoiser with a diffusion process.
    
    Handles the forward diffusion (noising) during training and 
    reverse diffusion (sampling) during inference.
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
        
        # Noise schedule parameters (registered as buffers for device handling)
        self._setup_noise_schedule()
    
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
    
    def set_diffusion(self, diffusion: BaseDiffusion):
        """Set a different diffusion process (e.g., switch from DDPM to DDIM)."""
        self._diffusion = diffusion
    
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
    
    def training_step(self, batch, batch_idx):
        """
        Training step with forward diffusion.
        
        Expected batch format: (labels, data) where data is (B, L, C).
        Data is transposed to (B, C, L) for the model.
        """
        labels, data = batch
        
        # Transpose from (B, L, C) to (B, C, L) for conv1d
        x = data.permute(0, 2, 1).float()
        
        # Sample timesteps
        t = self.sample_timesteps(x.shape[0])
        
        # Forward diffusion: add noise
        x_t, noise = self.noise_data(x, t)
        
        # Apply conditional dropout
        y = None
        if self.conditional_dropout > 0 and torch.rand(1).item() < self.conditional_dropout:
            y = None
        else:
            # Use the original data as conditioning (for denoising task)
            y = x
        
        # Predict
        predicted = self.denoiser(x_t, t, y)
        
        # Compute loss based on what denoiser predicts
        if self.denoiser_output == "noise":
            target = noise
        else:  # "original"
            target = x
        
        loss = self.loss_fn(predicted, target)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        labels, data = batch
        x = data.permute(0, 2, 1).float()
        
        t = self.sample_timesteps(x.shape[0])
        x_t, noise = self.noise_data(x, t)
        
        # No dropout during validation
        y = x
        
        predicted = self.denoiser(x_t, t, y)
        
        if self.denoiser_output == "noise":
            target = noise
        else:
            target = x
        
        loss = self.loss_fn(predicted, target)
        
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step - same as validation."""
        return self.validation_step(batch, batch_idx)
    
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