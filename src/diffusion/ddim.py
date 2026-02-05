import torch
from .base_diffusion import BaseDiffusion
from typing import override


class DDIM(BaseDiffusion):
    def __init__(self, noise_steps=1000, inference_steps=50, eta=0.0, 
                 beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__(noise_steps, beta_start, beta_end, device)
        self.inference_steps = inference_steps
        self.eta = eta
        step_ratio = self.noise_steps // inference_steps
        self.timesteps = list(range(1, self.noise_steps, step_ratio))
        if self.timesteps[-1] != self.noise_steps - 1:
            self.timesteps.append(self.noise_steps - 1)

    @override
    def sample(self, model, sample_size, shape, y=None, denoiser_output='noise'):
        model.eval()
        with torch.no_grad():
            x = torch.randn((sample_size, *shape)).to(self.device)
            for i, i_prev in zip(reversed(self.timesteps), list(reversed(self.timesteps))[1:]):
                t_tensor = (torch.ones(sample_size) * i).long().to(self.device)
                t_prev = (torch.ones(sample_size) * i_prev).long().to(self.device)
                predicted = model(x, t_tensor, y)  # x_0 hat if denoiser output is "original", epsilon_hat if "noise"
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                sigma_sq = (self.eta ** 2) * ((1 - alpha_hat_prev) / (1 - alpha_hat)) * (1 - (alpha_hat / alpha_hat_prev))
                if denoiser_output == 'noise':
                    # DDIM reverse process: x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1 - ᾱ_{t-1} - σ_t²) · ε_θ + σ_t · z
                    # where x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √ᾱ_t
                    # and σ_t² = η² · (1-ᾱ_{t-1})/(1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{t-1})
                    x = torch.sqrt(alpha_hat_prev) * ((x - torch.sqrt(1 - alpha_hat) * predicted) / torch.sqrt(alpha_hat)) \
                        + torch.sqrt(1 - alpha_hat_prev - sigma_sq) * predicted + torch.sqrt(sigma_sq) * noise
                elif denoiser_output == 'original':
                    # DDIM reverse process: x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1 - ᾱ_{t-1} - σ_t²) · ε + σ_t · z
                    # where x̂_0 = predicted, ε = (x_t - √ᾱ_t · x̂_0) / √(1-ᾱ_t)
                    # and σ_t² = η² · (1-ᾱ_{t-1})/(1-ᾱ_t) · (1 - ᾱ_t/ᾱ_{t-1})
                    x = torch.sqrt(alpha_hat_prev) * predicted \
                        + torch.sqrt(1 - alpha_hat_prev - sigma_sq) * ((x - torch.sqrt(alpha_hat) * predicted) / torch.sqrt(1 - alpha_hat)) \
                        + torch.sqrt(sigma_sq) * noise
        model.train()
        return x
