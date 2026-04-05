import torch
import torch.nn as nn
from abc import abstractmethod, ABC
from typing import Generator, Tuple, List, Optional


class BaseDiffusion(nn.Module, ABC):
    """
    Base class for diffusion processes.
    
    Subclasses only need to implement:
    - get_timestep_pairs(): Define the timestep sequence
    - denoising_step(): The actual denoising formula
    
    The sampling loop and generator are implemented generically here.
    Inherits from nn.Module so Lightning manages device placement of schedule tensors.
    """
    
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.register_buffer('beta', self.prepare_noise_schedule())
        self.register_buffer('alpha', 1. - self.beta)
        self.register_buffer('alpha_hat', torch.cumprod(self.alpha, dim=0))

    def prepare_noise_schedule(self) -> torch.Tensor:
        """Create linear noise schedule."""
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n_timesteps: int) -> torch.Tensor:
        """Sample random timesteps for training."""
        return torch.randint(low=1, high=self.noise_steps, size=(n_timesteps,), device=self.beta.device)

    def noise_data(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion - add noise to data.
        
        Args:
            x: Clean data (B, C, L)
            t: Timesteps (B,)
            
        Returns:
            Tuple of (noisy_x, noise)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    @abstractmethod
    def get_timestep_pairs(self) -> List[Tuple[int, int]]:
        """
        Return list of (t, t_prev) pairs for reverse diffusion.
        
        For DDPM: [(T-1, T-2), (T-2, T-3), ..., (1, 0)]
        For DDIM: Subsampled timesteps based on inference_steps
        """
        pass

    @abstractmethod
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
        Single denoising step: compute x_{t_prev} from x_t.
        
        Args:
            x_t: Current noisy sample (B, C, L)
            t: Current timestep
            t_prev: Previous timestep (target)
            model_output: Denoiser prediction (noise or original depending on denoiser_output)
            denoiser_output: 'noise' or 'original'
            batch_size: Batch size for tensor creation
            
        Returns:
            x_{t_prev}: Denoised sample at previous timestep
        """
        pass

    def reverse_diffusion_generator(
        self,
        model,
        batch_size: int,
        shape: Tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        denoiser_output: str = 'noise',
    ) -> Generator[Tuple[int, torch.Tensor], None, None]:
        """
        Generator that yields (timestep, x_t) at each reverse diffusion step.
        
        Useful for:
        - Visualizing the reverse process
        - Computing metrics at intermediate steps
        - Early stopping based on intermediate quality
        
        Args:
            model: Denoiser model
            batch_size: Number of samples to generate
            shape: Shape of each sample (C, L)
            y: Optional conditioning tensor
            denoiser_output: What the model predicts ('noise' or 'original')
            
        Yields:
            Tuple of (timestep, x_t) at each step, starting from pure noise
        """
        # Start with pure noise
        x = torch.randn((batch_size, *shape), device=self.beta.device)
        
        # Yield initial noise state
        yield self.noise_steps, x
        
        # Iterate through timestep pairs
        for t, t_prev in self.get_timestep_pairs():
            t_tensor = torch.full((batch_size,), t, device=self.beta.device, dtype=torch.long)
            
            # Get model prediction
            with torch.no_grad():
                model_output = model(x, t_tensor, y)
            
            # Denoising step
            x = self.denoising_step(x, t, t_prev, model_output, denoiser_output, batch_size)
            
            yield t_prev, x

    def sample(
        self,
        model,
        sample_size: int,
        shape: Tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        denoiser_output: str = 'noise',
    ) -> torch.Tensor:
        """
        Run full reverse diffusion and return final sample.
        
        Args:
            model: Denoiser model
            sample_size: Number of samples to generate
            shape: Shape of each sample (C, L)
            y: Optional conditioning tensor
            denoiser_output: What the model predicts ('noise' or 'original')
            
        Returns:
            Final denoised samples (B, C, L)
        """
        model.eval()
        
        # Consume generator to get final sample
        for t, x in self.reverse_diffusion_generator(model, sample_size, shape, y, denoiser_output):
            final_x = x
        
        model.train()
        return final_x
    
    def sample_with_trajectory(
        self,
        model,
        sample_size: int,
        shape: Tuple[int, ...],
        y: Optional[torch.Tensor] = None,
        denoiser_output: str = 'noise',
        save_every: int = 1,
    ) -> Tuple[torch.Tensor, List[Tuple[int, torch.Tensor]]]:
        """
        Run reverse diffusion and return both final sample and trajectory.
        
        Args:
            model: Denoiser model
            sample_size: Number of samples
            shape: Shape of each sample
            y: Optional conditioning
            denoiser_output: 'noise' or 'original'
            save_every: Save every N steps (1 = all steps)
            
        Returns:
            Tuple of (final_sample, trajectory) where trajectory is list of (t, x_t)
        """
        model.eval()
        
        trajectory = []
        for i, (t, x) in enumerate(self.reverse_diffusion_generator(model, sample_size, shape, y, denoiser_output)):
            if i % save_every == 0:
                trajectory.append((t, x.clone()))
            final_x = x
        
        model.train()
        return final_x, trajectory
