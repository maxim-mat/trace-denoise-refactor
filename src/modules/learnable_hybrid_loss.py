import torch.nn as nn
from typing import Optional, Tuple
import torch

class LearnableHybridLoss(nn.Module):
    def __init__(
        self, 
        first_loss: nn.Module, 
        second_loss: nn.Module, 
        gamma: Optional[float] = 0,  # logit
    ):
        super(LearnableHybridLoss, self).__init__()
        self.first_loss = first_loss
        self.second_loss = second_loss
        self.gamma = nn.Parameter(torch.tensor(gamma))

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
            primary_loss = (primary_loss * padding_mask).sum() / padding_mask.sum().clamp(min=1)

        if predicted_auxiliary is not None and target_auxiliary is not None:
            scale = torch.sigmoid(self.gamma)
            return scale * primary_loss + (1 - scale) * self.second_loss(predicted_auxiliary, target_auxiliary)
        else:
            return primary_loss
