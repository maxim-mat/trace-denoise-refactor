import torch.nn as nn
from .double_conv import DoubleConv


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256, down_rate=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(down_rate),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
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
        emb = self.emb_layer(t).unsqueeze(2).repeat(1, 1, x.shape[2])
        return x + emb
