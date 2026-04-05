import torch
import torch.nn as nn
import torch.nn.functional as F
from .double_conv import DoubleConv


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        # Pad upsampled x to match skip connection length (preserves full resolution)
        diff = skip_x.shape[-1] - x.shape[-1]
        if diff > 0:
            x = F.pad(x, (0, diff))
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        return x + emb
