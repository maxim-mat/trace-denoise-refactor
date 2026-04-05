import torch
from typing import List, Tuple
from .base_diffusion import BaseDiffusion


class DDIM(BaseDiffusion):
    """
    Denoising Diffusion Implicit Models (DDIM).
    
    Uses subsampled timesteps for faster inference.
    eta=0 gives deterministic sampling, eta=1 gives DDPM-like stochasticity.
    """
    
    def __init__(
        self,
        noise_steps=1000,
        inference_steps=50,
        eta=0.0,
        beta_start=1e-4,
        beta_end=0.02,
    ):
        super().__init__(noise_steps, beta_start, beta_end)
        self.inference_steps = inference_steps
        self.eta = eta
        
        # Create subsampled timestep sequence
        step_ratio = self.noise_steps // inference_steps
        self.timesteps = list(range(0, self.noise_steps, step_ratio))
        if self.timesteps[-1] != self.noise_steps - 1:
            self.timesteps.append(self.noise_steps - 1)
        self.timesteps = sorted(self.timesteps, reverse=True)  # Descending order

    def get_timestep_pairs(self) -> List[Tuple[int, int]]:
        """Return subsampled timestep pairs."""
        # Pair each timestep with the next one (going backwards)
        pairs = []
        for i in range(len(self.timesteps) - 1):
            pairs.append((self.timesteps[i], self.timesteps[i + 1]))
        # Final step to 0
        if self.timesteps[-1] != 0:
            pairs.append((self.timesteps[-1], 0))
        return pairs

    def denoising_step(
        self,
        x_t: torch.Tensor,
        t: int,
        t_prev: int,
        model_output: torch.Tensor,
        denoiser_output: str,
        batch_size: int,
    ) -> torch.Tensor:
        """
        DDIM reverse step: x_{t_prev} from x_t.
        
        x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1 - ᾱ_{t-1} - σ²) * ε + σ * z
        
        where σ² = η² * (1-ᾱ_{t-1})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t-1})
        """
        t_tensor = torch.full((batch_size,), t, dtype=torch.long, device=self.beta.device)
        t_prev_tensor = torch.full((batch_size,), max(t_prev, 0), dtype=torch.long, device=self.beta.device)
        
        alpha_hat = self.alpha_hat[t_tensor][:, None, None]
        alpha_hat_prev = self.alpha_hat[t_prev_tensor][:, None, None] if t_prev > 0 else torch.ones_like(alpha_hat)
        
        # Add noise except at final step
        if t > 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        # DDIM variance: σ² = η² * (1-ᾱ_{t-1})/(1-ᾱ_t) * (1 - ᾱ_t/ᾱ_{t-1})
        sigma_sq = (self.eta ** 2) * ((1 - alpha_hat_prev) / (1 - alpha_hat)) * (1 - (alpha_hat / alpha_hat_prev))
        sigma_sq = torch.clamp(sigma_sq, min=0)  # Numerical stability
        
        if denoiser_output == 'noise':
            # Predict x_0 from noise prediction: x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
            pred_x0 = (x_t - torch.sqrt(1 - alpha_hat) * model_output) / torch.sqrt(alpha_hat)
            # Direction pointing to x_t
            pred_eps = model_output
        elif denoiser_output == 'original':
            # Model directly predicts x_0
            pred_x0 = model_output
            # Compute implied noise: ε = (x_t - √ᾱ_t * x̂_0) / √(1-ᾱ_t)
            pred_eps = (x_t - torch.sqrt(alpha_hat) * pred_x0) / torch.sqrt(1 - alpha_hat)
        else:
            raise ValueError(f"Unknown denoiser_output: {denoiser_output}")
        
        # DDIM update: x_{t-1} = √ᾱ_{t-1} * x̂_0 + √(1 - ᾱ_{t-1} - σ²) * ε + σ * z
        x_prev = torch.sqrt(alpha_hat_prev) * pred_x0 \
            + torch.sqrt(1 - alpha_hat_prev - sigma_sq) * pred_eps \
            + torch.sqrt(sigma_sq) * noise
        
        return x_prev
