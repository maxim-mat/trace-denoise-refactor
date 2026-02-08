import torch
import torch.nn as nn
from .double_conv_2d import DoubleConv2d


class Up2d(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, scale_factor=2):
        super().__init__()
        self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv2d(in_channels, in_channels, residual=True),
            DoubleConv2d(in_channels, out_channels, in_channels // 2),
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
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t).unsqueeze(2).unsqueeze(2).repeat(1, 1, x.shape[2], x.shape[3])
        return x + emb
