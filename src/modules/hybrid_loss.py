import torch.nn as nn
from typing import Optional, Tuple
import torch

class HybridLoss(nn.Module):
    def __init__(
        self, 
        first_loss: nn.Module, 
        second_loss: nn.Module,
        gamma: Optional[float] = 0.5,
    ):
        super(HybridLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.register_buffer('gamma', torch.tensor(gamma))

    def forward(
        self, 
        predicted: Tuple[torch.Tensor, Optional[torch.Tensor]], 
        target: Tuple[torch.Tensor, Optional[torch.Tensor]],
        padding_mask: Optional[torch.Tensor] = None,
    ):
        predicted_primary, predicted_auxiliary = predicted
        target_primary, target_auxiliary = target

        primary_loss = self.first_loss(predicted_primary, target_primary)
        if padding_mask is not None:
            # mask-based reduction: per-element loss already from reduction='none'
            primary_loss = (primary_loss * padding_mask).sum() / padding_mask.sum().clamp(min=1)

        if predicted_auxiliary is not None:
            return self.gamma * primary_loss + (1 - self.gamma) * self.second_loss(predicted_auxiliary, target_auxiliary)
        else:
            return primary_loss
