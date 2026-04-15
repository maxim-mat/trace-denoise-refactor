import torch
import torch.nn as nn
from src.modules import DoubleConv, Up, Down, SelfAttention


from src.utils.nn_utils import _downsample_mask

class ConditionalUnetDenoiser(nn.Module):
    """
    Conditional U-Net denoiser architecture.
    
    Pure nn.Module - does not contain training logic.
    Use with DiffusionLightningModule for training.
    """
    
    def __init__(
        self,
        in_ch,
        out_ch,
        time_dim=128,
    ):
        super().__init__()
        self.time_dim = time_dim

        # Main path
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128, emb_dim=time_dim)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256, emb_dim=time_dim)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256, emb_dim=time_dim)
        self.sa3 = SelfAttention(256)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128, emb_dim=time_dim)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64, emb_dim=time_dim)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64, emb_dim=time_dim)
        self.sa6 = SelfAttention(64)

        # Conditioning path
        self.inc_cond = DoubleConv(in_ch, 64)
        self.down1_cond = Down(64, 128, emb_dim=time_dim)
        self.sa1_cond = SelfAttention(128)
        self.down2_cond = Down(128, 256, emb_dim=time_dim)
        self.sa2_cond = SelfAttention(256)
        self.down3_cond = Down(256, 256, emb_dim=time_dim)
        self.sa3_cond = SelfAttention(256)

        self.bot1_cond = DoubleConv(256, 512)
        self.bot2_cond = DoubleConv(512, 512)
        self.bot3_cond = DoubleConv(512, 256)

        self.up1_cond = Up(512, 128, emb_dim=time_dim)
        self.sa4_cond = SelfAttention(128)
        self.up2_cond = Up(256, 64, emb_dim=time_dim)
        self.sa5_cond = SelfAttention(64)
        self.up3_cond = Up(128, 64, emb_dim=time_dim)
        self.sa6_cond = SelfAttention(64)
        
        self.outc = nn.Conv1d(64, out_ch, kernel_size=1)

    def pos_encoding(self, t, channels, device):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _forward_uncond(self, x, t, mask=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        # Downsample mask at each level (L -> L//2 -> L//4 -> L//8)
        mask1 = mask                             # L
        mask2 = _downsample_mask(mask1)          # L//2
        mask3 = _downsample_mask(mask2)          # L//4
        mask4 = _downsample_mask(mask3)          # L//8

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2, key_padding_mask=mask2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3, key_padding_mask=mask3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4, key_padding_mask=mask4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x, key_padding_mask=mask3)
        x = self.up2(x, x2, t)
        x = self.sa5(x, key_padding_mask=mask2)
        x = self.up3(x, x1, t)
        x = self.sa6(x, key_padding_mask=mask1)
        x = self.outc(x)
        return x

    def _forward_cond(self, x, y, t, mask=None):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        mask1 = mask
        mask2 = _downsample_mask(mask1)
        mask3 = _downsample_mask(mask2)
        mask4 = _downsample_mask(mask3)

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        x2 = self.down1(x1 + y1, t)
        y2 = self.down1_cond(x1 + y1, t)
        y2 = self.sa1_cond(y2, key_padding_mask=mask2)
        x2 = self.sa1(x2, key_padding_mask=mask2)
        x3 = self.down2(x2 + y2, t)
        x3 = self.sa2(x3, key_padding_mask=mask3)
        y3 = self.down2_cond(x2 + y2, t)
        y3 = self.sa2_cond(y3, key_padding_mask=mask3)
        x4 = self.down3(x3 + y3, t)
        x4 = self.sa3(x4, key_padding_mask=mask4)
        y4 = self.down3_cond(x3 + y3, t)
        y4 = self.sa3_cond(y4, key_padding_mask=mask4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)

        y = self.up1_cond(x4 + y4, y3, t)
        x = self.up1(x4 + y4, x3, t)
        x = self.sa4(x, key_padding_mask=mask3)
        y = self.sa4_cond(y, key_padding_mask=mask3)
        x_next = self.up2(x + y, x2, t)
        y_next = self.up2_cond(x + y, y2, t)
        y_next = self.sa5_cond(y_next, key_padding_mask=mask2)
        x_next = self.sa5(x_next, key_padding_mask=mask2)
        x = self.up3(x_next + y_next, x1, t)
        y = self.up3_cond(x_next + y_next, y1, t)
        y = self.sa6_cond(y, key_padding_mask=mask1)
        x = self.sa6(x, key_padding_mask=mask1)
        x = self.outc(x + y)

        return x

    def forward(self, x, t, y=None, mask=None, *args, **kwargs):
        """
        Forward pass for denoising.
        
        Args:
            x: Noisy input tensor (B, C, L)
            t: Diffusion timestep (B,)
            y: Optional conditioning tensor (B, C, L)
            mask: Optional padding mask (B, L), True = real, False = padding
            
        Returns:
            Denoised output tensor (B, C, L)
        """
        if y is not None:
            return self._forward_cond(x, y, t, mask=mask)
        return self._forward_uncond(x, t, mask=mask)
