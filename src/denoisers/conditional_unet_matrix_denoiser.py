import torch
import torch.nn as nn

from src.modules.double_conv import DoubleConv
from src.modules.double_conv_2d import DoubleConv2d
from src.modules.up import Up
from src.modules.up_2d import Up2d
from src.modules.down import Down
from src.modules.down_2d import Down2d
from src.modules.self_attention import SelfAttention
from src.modules.cross_attention import CrossAttention


class ConditionalUnetMatrixDenoiser(nn.Module):
    """
    Conditional U-Net denoiser with transition matrix auxiliary output.
    
    This denoiser jointly predicts:
    1. The denoised trace sequence (primary output)
    2. A transition matrix (auxiliary output for regularization)
    
    The forward method returns (x_hat, m_hat) following the convention that
    the primary diffusion output is always first.
    
    Args:
        in_ch: Number of input channels (num_classes)
        out_ch: Number of output channels (num_classes)
        transition_dim: Dimension of the transition matrix
        transition_matrix: Optional pre-defined transition matrix
        time_dim: Dimension of time embeddings
        matrix_out_channels: Output channels for matrix prediction
    """
    
    def __init__(
        self,
        in_ch,
        out_ch,
        flow_matrix_dim,
        flow_matrix=None,
        time_dim=128,
        matrix_out_channels=1,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.num_classes = in_ch
        self.flow_matrix_dim = flow_matrix_dim
        
        # Learnable transition matrix if not provided
        if self.flow_matrix_dim is not None:
            self.register_buffer('flow_matrix', self.flow_matrix_dim)
        else:
            self.flow_matrix = nn.Parameter(
                torch.randn(1, in_ch + 1, self.flow_matrix_dim, self.flow_matrix_dim)
            )

        # Main sequence U-Net path
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

        # Matrix U-Net path
        self.inc_mat = DoubleConv2d(in_ch + 1, 64)
        self.down1_mat = Down2d(64, 128, emb_dim=time_dim, down_rate=8)
        self.sa1_mat = SelfAttention(128)
        self.down2_mat = Down2d(128, 256, emb_dim=time_dim)
        self.sa2_mat = SelfAttention(256)
        self.down3_mat = Down2d(256, 256, emb_dim=time_dim)
        self.sa3_mat = SelfAttention(256)
        
        self.bot1_mat = DoubleConv2d(256, 512)
        self.bot2_mat = DoubleConv2d(512, 512)
        self.bot3_mat = DoubleConv2d(512, 256)
        
        self.up1_mat = Up2d(512, 128, emb_dim=time_dim)
        self.sa4_mat = SelfAttention(128)
        self.up2_mat = Up2d(256, 64, emb_dim=time_dim)
        self.sa5_mat = SelfAttention(64)
        self.up3_mat = Up2d(128, 64, emb_dim=time_dim, scale_factor=8)
        self.sa6_mat = SelfAttention(64)

        # Cross-attention: sequence <-> matrix
        self.casm1 = CrossAttention(128)
        self.casm2 = CrossAttention(256)
        self.casm3 = CrossAttention(256)
        self.casm4 = CrossAttention(128)
        self.casm5 = CrossAttention(64)
        self.casm6 = CrossAttention(64)

        self.casm1_cond = CrossAttention(128)
        self.casm2_cond = CrossAttention(256)
        self.casm3_cond = CrossAttention(256)
        self.casm4_cond = CrossAttention(128)
        self.casm5_cond = CrossAttention(64)
        self.casm6_cond = CrossAttention(64)

        self.cams1 = CrossAttention(128)
        self.cams2 = CrossAttention(256)
        self.cams3 = CrossAttention(256)
        self.cams4 = CrossAttention(128)
        self.cams5 = CrossAttention(64)
        self.cams6 = CrossAttention(64)

        self.outc_mat = nn.Conv2d(64, matrix_out_channels, kernel_size=1)
        self.outc = nn.Conv1d(64, out_ch, kernel_size=1)

    def pos_encoding(self, t, channels, device):
        """Generate sinusoidal positional encoding."""
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def _forward_uncond_mat(self, x, m, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)
        batch_dim = x.shape[0]

        x1 = self.inc(x)
        m1 = self.inc_mat(m)

        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        m2 = self.down1_mat(m1, t)
        m2 = m2.view(batch_dim, m2.shape[1], -1)
        m2 = self.sa1_mat(m2)
        m2_ca = self.cams1(m2, x2, x2).view(m2.size(0), m2.size(1),
                                            self.transition_dim // 8, self.transition_dim // 8)
        x2_ca = self.casm1(x2, m2, m2)

        x3 = self.down2(x2_ca, t)
        x3 = self.sa2(x3)
        m3 = self.down2_mat(m2_ca, t)
        m3 = m3.view(batch_dim, m3.shape[1], -1)
        m3 = self.sa2_mat(m3)
        m3_ca = self.cams2(m3, x3, x3).view(m3.size(0), m3.size(1),
                                            self.transition_dim // 16, self.transition_dim // 16)
        x3_ca = self.casm2(x3, m3, m3)

        x4 = self.down3(x3_ca, t)
        x4 = self.sa3(x4)
        m4 = self.down3_mat(m3_ca, t)
        m4 = m4.view(batch_dim, m4.shape[1], -1)
        m4 = self.sa3_mat(m4)
        m4_ca = self.cams3(m4, x4, x4).view(m4.size(0), m4.size(1),
                                            self.transition_dim // 32, self.transition_dim // 32)
        x4_ca = self.casm3(x4, m4, m4)

        x4 = self.bot1(x4_ca)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        m4 = self.bot1_mat(m4_ca)
        m4 = self.bot2_mat(m4)
        m4 = self.bot3_mat(m4)

        x = self.up1(x4, x3_ca, t)
        x = self.sa4(x)
        m = self.up1_mat(m4, m3_ca, t)
        m = m.view(batch_dim, m.shape[1], -1)
        m = self.sa4_mat(m)
        m_ca = self.cams4(m, x, x).view(m.size(0), m.size(1),
                                        self.transition_dim // 16, self.transition_dim // 16)
        x_ca = self.casm4(x, m, m)

        x_next = self.up2(x_ca, x2_ca, t)
        x_next = self.sa5(x_next)
        m_next = self.up2_mat(m_ca, m2_ca, t)
        m_next = m_next.view(batch_dim, m_next.shape[1], -1)
        m_next = self.sa5_mat(m_next)
        m_next_ca = self.cams5(m_next, x_next, x_next).view(m_next.size(0), m_next.size(1),
                                                           self.transition_dim // 8,
                                                           self.transition_dim // 8)
        x_next_ca = self.casm5(x_next, m_next, m_next)

        x = self.up3(x_next_ca, x1, t)
        x = self.sa6(x)
        m = self.up3_mat(m_next_ca, m1.repeat(x.shape[0], 1, 1, 1), t)

        m = self.outc_mat(m)
        x = self.outc(x)

        return x, m

    def _forward_cond_mat(self, x, y, m, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)
        batch_dim = x.shape[0]

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        m1 = self.inc_mat(m)

        x2 = self.down1(x1 + y1, t)
        x2 = self.sa1(x2)
        y2 = self.down1_cond(x1 + y1, t)
        y2 = self.sa1_cond(y2)
        m2 = self.down1_mat(m1, t)
        m2 = m2.view(batch_dim, m2.shape[1], -1)
        m2 = self.sa1_mat(m2)
        m2_ca = self.cams1(m2, x2 + y2, x2 + y2).view(m2.size(0), m2.size(1),
                                                      self.transition_dim // 8, self.transition_dim // 8)
        x2_ca = self.casm1(x2 + y2, m2, m2)
        y2_ca = self.casm1_cond(x2 + y2, m2, m2)

        x3 = self.down2(x2_ca + y2_ca, t)
        x3 = self.sa2(x3)
        y3 = self.down2_cond(x2_ca + y2_ca, t)
        y3 = self.sa2_cond(y3)
        m3 = self.down2_mat(m2_ca, t)
        m3 = m3.view(batch_dim, m3.shape[1], -1)
        m3 = self.sa2_mat(m3)
        m3_ca = self.cams2(m3, x3 + y3, x3 + y3).view(m3.size(0), m3.size(1),
                                                      self.transition_dim // 16, self.transition_dim // 16)
        x3_ca = self.casm2(x3 + y3, m3, m3)
        y3_ca = self.casm2_cond(x3 + y3, m3, m3)

        x4 = self.down3(x3_ca + y3_ca, t)
        x4 = self.sa3(x4)
        y4 = self.down3_cond(x3_ca + y3_ca, t)
        y4 = self.sa3_cond(y4)
        m4 = self.down3_mat(m3_ca, t)
        m4 = m4.view(batch_dim, m4.shape[1], -1)
        m4 = self.sa3_mat(m4)
        m4_ca = self.cams3(m4, x4 + y4, x4 + y4).view(m4.size(0), m4.size(1),
                                                      self.transition_dim // 32, self.transition_dim // 32)
        x4_ca = self.casm3(x4 + y4, m4, m4)
        y4_ca = self.casm3_cond(x4 + y4, m4, m4)

        x4 = self.bot1(x4_ca)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4_ca)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)
        m4 = self.bot1_mat(m4_ca)
        m4 = self.bot2_mat(m4)
        m4 = self.bot3_mat(m4)

        x = self.up1(x4 + y4, x3_ca, t)
        x = self.sa4(x)
        y = self.up1(x4 + y4, y3_ca, t)
        y = self.sa4_cond(y)
        m = self.up1_mat(m4, m3_ca, t)
        m = m.view(batch_dim, m.shape[1], -1)
        m = self.sa4_mat(m)
        m_ca = self.cams4(m, x + y, x + y).view(m.size(0), m.size(1),
                                                self.transition_dim // 16, self.transition_dim // 16)
        x_ca = self.casm4(x + y, m, m)
        y_ca = self.casm4_cond(x + y, m, m)

        x_next = self.up2(x_ca + y_ca, x2_ca, t)
        x_next = self.sa5(x_next)
        y_next = self.up2_cond(x_ca + y_ca, y2_ca, t)
        y_next = self.sa5_cond(y_next)
        m_next = self.up2_mat(m_ca, m2_ca, t)
        m_next = m_next.view(batch_dim, m_next.shape[1], -1)
        m_next = self.sa5_mat(m_next)
        m_next_ca = self.cams5(m_next, x_next + y_next, x_next + y_next).view(m_next.size(0), m_next.size(1),
                                                                              self.transition_dim // 8,
                                                                              self.transition_dim // 8)
        x_next_ca = self.casm5(x_next + y_next, m_next, m_next)
        y_next_ca = self.casm5_cond(x_next + y_next, m_next, m_next)

        x = self.up3(x_next_ca + y_next_ca, x1, t)
        x = self.sa6(x)
        y = self.up3(x_next_ca + y_next_ca, y1, t)
        y = self.sa6(y)
        m = self.up3_mat(m_next_ca, m1.repeat(x.shape[0], 1, 1, 1), t)

        m = self.outc_mat(m)
        x = self.outc(x + y)

        return x, m

    def _forward_uncond(self, x, t):
        """Unconditional forward without matrix."""
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        x = self.outc(x)
        return x

    def _forward_cond(self, x, y, t):
        """Conditional forward without matrix."""
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        y1 = self.inc_cond(y)
        x1 = self.inc(x)
        x2 = self.down1(x1 + y1, t)
        y2 = self.down1_cond(x1 + y1, t)
        y2 = self.sa1_cond(y2)
        x2 = self.sa1(x2)
        x3 = self.down2(x2 + y2, t)
        x3 = self.sa2(x3)
        y3 = self.down2_cond(x2 + y2, t)
        y3 = self.sa2_cond(y3)
        x4 = self.down3(x3 + y3, t)
        x4 = self.sa3(x4)
        y4 = self.down3_cond(x3 + y3, t)
        y4 = self.sa3_cond(y4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)

        y = self.up1(x4 + y4, y3, t)
        x = self.up1(x4 + y4, x3, t)
        x = self.sa4(x)
        y = self.sa4(y)
        x_next = self.up2(x + y, x2, t)
        y_next = self.up2(x + y, y2, t)
        y_next = self.sa5(y_next)
        x_next = self.sa5(x_next)
        x = self.up3(x_next + y_next, x1, t)
        y = self.up3(x_next + y_next, y1, t)
        y = self.sa6(y)
        x = self.sa6(x)
        x = self.outc(x + y)

        return x

    def forward(self, x, t, y=None, use_matrix=True):
        """
        Forward pass for denoising.
        
        Args:
            x: Noisy input tensor (B, C, L)
            t: Diffusion timestep (B,)
            y: Optional conditioning tensor (B, C, L)
            use_matrix: Whether to use matrix branch (enables auxiliary output)
            
        Returns:
            If use_matrix=True: (x_hat, m_hat) - trace and matrix predictions
            If use_matrix=False: x_hat - trace prediction only
            
            Primary output (trace) is always first.
        """
        if use_matrix:
            if y is not None:
                return self._forward_cond_mat(x, y, self.flow_matrix, t)
            else:
                return self._forward_uncond_mat(x, self.flow_matrix, t)
        else:
            if y is not None:
                return self._forward_cond(x, y, t)
            else:
                return self._forward_uncond(x, t)
