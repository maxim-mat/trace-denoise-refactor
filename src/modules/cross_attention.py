import torch.nn as nn


class CrossAttention(nn.Module):
    """
    Cross-attention module for 1D sequences.
    
    Args:
        channels: Number of input/output channels
        num_heads: Number of attention heads (default: 2)
    """
    def __init__(self, channels, num_heads=2):
        super(CrossAttention, self).__init__()
        self.channels = channels
        self.mha = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_cross = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, query, key, value, key_padding_mask=None):
        # Inputs: (B, C, L) -> (B, L, C) for attention
        # key_padding_mask: (B, L_key) bool, True = real, False = padding
        batch_size, channels, seq_len = query.shape
        
        query = query.permute(0, 2, 1)  # (B, L, C)
        key = key.permute(0, 2, 1)
        value = value.permute(0, 2, 1)

        # nn.MultiheadAttention expects True = ignore position
        attn_mask = ~key_padding_mask if key_padding_mask is not None else None

        query_ln = self.ln(query)
        key_ln = self.ln(key)
        value_ln = self.ln(value)

        attention_value, _ = self.mha(query_ln, key_ln, value_ln, key_padding_mask=attn_mask)
        attention_value = attention_value + query  # Residual connection
        attention_value = self.ff_cross(attention_value) + attention_value  # Residual after feed-forward

        # (B, L, C) -> (B, C, L)
        return attention_value.permute(0, 2, 1)
