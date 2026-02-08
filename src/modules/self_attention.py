import torch.nn as nn


class SelfAttention(nn.Module):
    """
    Self-attention module for 1D sequences.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: 2)
    """
    def __init__(self, channels, num_heads=2):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # x: (B, C, L) -> (B, L, C) for attention
        batch_size, channels, seq_len = x.shape
        x = x.permute(0, 2, 1)  # (B, L, C)
        
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        
        # (B, L, C) -> (B, C, L)
        return attention_value.permute(0, 2, 1)
