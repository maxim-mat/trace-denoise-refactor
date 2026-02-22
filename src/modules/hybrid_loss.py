import torch.nn as nn
from typing import Optional, Tuple
import torch

class HybridLoss(nn.Module):
    def __init__(
        self, 
        first_loss: nn.Module, 
        second_loss: nn.Module,
        gamma: Optional[float] = 0.5,
        ignore_index: Optional[int] = None,
    ):
        super(HybridLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.gamma = gamma
        self.register_buffer('gamma', torch.tensor(gamma))
        self.ignore_index = ignore_index

    def forward(
        self, 
        predicted: Tuple[torch.Tensor, Optional[torch.Tensor]], 
        target: Tuple[torch.Tensor, Optional[torch.Tensor]],
        ignore_index: Optional[int] = None,
    ):
        predicted_primary, predicted_auxiliary = predicted
        target_primary, target_auxiliary = target
        if predicted_auxiliary is not None:
            return self.gamma * self.first_loss(predicted_primary, target_primary, ignore_index=ignore_index) + \
                (1 - self.gamma) * self.second_loss(predicted_auxiliary, target_auxiliary)
        else:
            return self.first_loss(predicted_primary, target_primary, ignore_index=ignore_index)
