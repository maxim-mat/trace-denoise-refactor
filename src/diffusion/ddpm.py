import torch
from .base_diffusion import BaseDiffusion
from typing import override


class DDPM(BaseDiffusion):
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__(noise_steps, beta_start, beta_end, device)

    @override
    def sample(self, model, sample_size, shape, y=None, denoiser_output='noise'):
        model.eval()
        with torch.no_grad():
            x = torch.randn((sample_size, *shape)).to(self.device)
            for i in reversed(range(1, self.noise_steps)):
                t_tensor = (torch.ones(sample_size) * i).long().to(self.device)
                t_prev = (torch.ones(sample_size) * (i - 1)).long().to(self.device)
                predicted = model(x, t_tensor, y)  # x_0 hat if denoiser output is "original", epsilon_hat if "noise"
                alpha = self.alpha[t_tensor][:, None, None]
                alpha_hat = self.alpha_hat[t_tensor][:, None, None]
                alpha_hat_prev = self.alpha_hat[t_prev][:, None, None]
                beta = self.beta[t_tensor][:, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                if denoiser_output == 'noise':
                    # DDPM reverse process: x_{t-1} = 1/sqrt(α_t) * (x_t - β_t/sqrt(1-ᾱ_t) * ε_θ) + σ_t * z
                    # where σ_t² = β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted) \
                        + torch.sqrt(((1 - alpha_hat_prev) / (1 - alpha_hat)) * beta) * noise
                elif denoiser_output == 'original':
                    # Posterior mean: μ̃_t = (√ᾱ_{t-1}·β_t)/(1-ᾱ_t) * x_0 + (√α_t·(1-ᾱ_{t-1}))/(1-ᾱ_t) * x_t
                    # Posterior variance: β̃_t = (1-ᾱ_{t-1})/(1-ᾱ_t) * β_t
                    x = (torch.sqrt(alpha_hat_prev) * beta / (1 - alpha_hat)) * predicted \
                        + ((torch.sqrt(alpha) * (1 - alpha_hat_prev)) / (1 - alpha_hat)) * x \
                        + torch.sqrt(((1 - alpha_hat_prev) / (1 - alpha_hat)) * beta) * noise
        model.train()
        return x
