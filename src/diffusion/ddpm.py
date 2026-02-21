import torch
from typing import List, Tuple
from .base_diffusion import BaseDiffusion


class DDPM(BaseDiffusion):
    """
    Denoising Diffusion Probabilistic Models (DDPM).
    
    Uses all timesteps for reverse diffusion (slower but higher quality).
    """
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__(noise_steps, beta_start, beta_end)

    def get_timestep_pairs(self) -> List[Tuple[int, int]]:
        """Return all timestep pairs from T-1 down to 0."""
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0)]
        return [(t, t - 1) for t in range(self.noise_steps - 1, 0, -1)]

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
        DDPM reverse step: x_{t-1} from x_t.
        
        For denoiser_output='noise':
            x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
            
        For denoiser_output='original':
            x_{t-1} = (√ᾱ_{t-1} * β_t)/(1-ᾱ_t) * x̂_0 + (√α_t * (1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t + σ_t * z
        """
        t_tensor = torch.full((batch_size,), t, dtype=torch.long)
        t_prev_tensor = torch.full((batch_size,), t_prev, dtype=torch.long)
        
        alpha = self.alpha[t_tensor][:, None, None]
        alpha_hat = self.alpha_hat[t_tensor][:, None, None]
        alpha_hat_prev = self.alpha_hat[t_prev_tensor][:, None, None] if t_prev >= 0 else torch.ones_like(alpha_hat)
        beta = self.beta[t_tensor][:, None, None]
        
        # Add noise except at final step
        if t > 1:
            noise = torch.randn_like(x_t)
        else:
            noise = torch.zeros_like(x_t)
        
        # Posterior variance: σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
        posterior_variance = ((1 - alpha_hat_prev) / (1 - alpha_hat)) * beta
        
        if denoiser_output == 'noise':
            # x_{t-1} = 1/√α_t * (x_t - β_t/√(1-ᾱ_t) * ε_θ) + σ_t * z
            x_prev = (1 / torch.sqrt(alpha)) * (x_t - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * model_output) \
                + torch.sqrt(posterior_variance) * noise
        elif denoiser_output == 'original':
            # Posterior mean: μ̃_t = (√ᾱ_{t-1} * β_t)/(1-ᾱ_t) * x̂_0 + (√α_t * (1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
            x_prev = (torch.sqrt(alpha_hat_prev) * beta / (1 - alpha_hat)) * model_output \
                + ((torch.sqrt(alpha) * (1 - alpha_hat_prev)) / (1 - alpha_hat)) * x_t \
                + torch.sqrt(posterior_variance) * noise
        else:
            raise ValueError(f"Unknown denoiser_output: {denoiser_output}")
        
        return x_prev
