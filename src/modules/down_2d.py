import torch.nn as nn
from .double_conv_2d import DoubleConv2d


class Down2d(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, down_rate=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(down_rate),
            DoubleConv2d(in_channels, in_channels, residual=True),
            DoubleConv2d(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t).unsqueeze(2).unsqueeze(2).repeat(1, 1, x.shape[2], x.shape[3])
        return x + emb
