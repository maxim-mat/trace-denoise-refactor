import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, residual=False):
        # in ch start at 19 (number of features)
        super().__init__()
        mid_ch = out_ch if mid_ch is None else mid_ch
        self.residual = residual
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_ch),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)
