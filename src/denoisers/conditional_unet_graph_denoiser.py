import torch
import torch.nn as nn

from src.modules.double_conv import DoubleConv
from src.modules.self_attention import SelfAttention
from src.modules.down import Down
from src.modules.up import Up
from src.modules.graph_encoder import GraphEncoder
from src.modules.cross_attention import CrossAttention


class ConditionalUnetGraphDenoiser(nn.Module):
    """
    Conditional U-Net denoiser with graph-based conditioning.
    
    Uses a graph neural network to encode structural information (e.g., process model)
    and injects it into the U-Net via cross-attention or additive fusion.
    
    Args:
        in_ch: Number of input channels (num_classes)
        out_ch: Number of output channels (num_classes)
        num_nodes: Number of nodes in the graph
        graph_data: Graph structure data (edge_index, etc.)
        embedding_dim: Dimension of node embeddings
        hidden_dim: Hidden dimension in GNN
        pooling: Pooling method ('mean', 'max', None for no pooling)
        time_dim: Dimension of time embeddings
    """
    
    def __init__(
        self,
        in_ch,
        out_ch,
        graph_data,
        embedding_dim=128,
        hidden_dim=128,
        pooling=None,
        time_dim=128,
    ):
        super().__init__()
        self.time_dim = time_dim
        self.graph_data = graph_data
        self.gnn_pooling = pooling
        self.num_nodes = self.graph_data.num_nodes

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

        # Graph encoders at different scales
        self.genc1 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=1, output_dim=64, pooling=pooling)
        self.genc2 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=2, output_dim=128, pooling=pooling)
        self.genc3 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=3, output_dim=256, pooling=pooling)
        self.genc4 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=3, output_dim=256, pooling=pooling)
        self.genc5 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=2, output_dim=128, pooling=pooling)
        self.genc6 = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=1, output_dim=64, pooling=pooling)

        self.genc1_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=1, output_dim=64, pooling=pooling)
        self.genc2_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=2, output_dim=128, pooling=pooling)
        self.genc3_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=3, output_dim=256, pooling=pooling)
        self.genc4_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=3, output_dim=256, pooling=pooling)
        self.genc5_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=2, output_dim=128, pooling=pooling)
        self.genc6_cond = GraphEncoder(self.num_nodes, embedding_dim, hidden_dim, num_layers=1, output_dim=64, pooling=pooling)

        # Cross-attention layers for graph fusion (when not using pooling)
        if self.gnn_pooling is None:
            self.ca1 = CrossAttention(128)
            self.ca2 = CrossAttention(256)
            self.ca3 = CrossAttention(256)
            self.ca4 = CrossAttention(128)
            self.ca5 = CrossAttention(64)
            self.ca6 = CrossAttention(64)

            self.ca1_cond = CrossAttention(128)
            self.ca2_cond = CrossAttention(256)
            self.ca3_cond = CrossAttention(256)
            self.ca4_cond = CrossAttention(128)
            self.ca5_cond = CrossAttention(64)
            self.ca6_cond = CrossAttention(64)

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

    def _forward_uncond(self, x, t):
        """Unconditional forward without graph."""
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
        """Conditional forward without graph."""
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

    def _forward_uncond_graph(self, x, t):
        """Unconditional forward with graph fusion."""
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        batch_size = x.size(0)
        g = self.graph_data

        x1 = self.inc(x)
        if self.gnn_pooling is None:
            x2 = self.down1(x1, t)
            x2 = self.sa1(x2)
            g2 = self.genc2(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x2 = self.ca1(x2, g2, g2)

            x3 = self.down2(x2, t)
            x3 = self.sa2(x3)
            g3 = self.genc3(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x3 = self.ca2(x3, g3, g3)

            x4 = self.down3(x3, t)
            x4 = self.sa3(x4)
            g4 = self.genc4(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x4 = self.ca3(x4, g4, g4)
        else:
            x2 = self.down1(x1, t)
            x2 = self.sa1(x2)

            g2 = self.genc2(g).view(1, -1, 1).repeat(batch_size, 1, x2.size(2))
            x3 = self.down2(x2 + g2, t)
            x3 = self.sa2(x3)

            g3 = self.genc3(g).view(1, -1, 1).repeat(batch_size, 1, x3.size(2))
            x4 = self.down3(x3 + g3, t)
            x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        if self.gnn_pooling is None:
            x = self.up1(x4, x3, t)
            x = self.sa4(x)

            g5 = self.genc5(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.ca4(x, g5, g5)
            x = self.up2(x, x2, t)
            x = self.sa5(x)

            g6 = self.genc6(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.ca5(x, g6, g6)
            x = self.up3(x, x1, t)
            x = self.sa6(x)
        else:
            g4 = self.genc4(g).view(1, -1, 1).repeat(batch_size, 1, x4.size(2))
            x = self.up1(x4 + g4, x3, t)
            x = self.sa4(x)

            g5 = self.genc5(g).view(1, -1, 1).repeat(batch_size, 1, x.size(2))
            x = self.up2(x + g5, x2, t)
            x = self.sa5(x)

            g6 = self.genc6(g).view(1, -1, 1).repeat(batch_size, 1, x.size(2))
            x = self.up3(x + g6, x1, t)
            x = self.sa6(x)

        x = self.outc(x)
        return x

    def _forward_cond_graph(self, x, y, t):
        """Conditional forward with graph fusion."""
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim, x.device)

        batch_size = x.size(0)
        g = self.graph_data

        x1 = self.inc(x)
        y1 = self.inc_cond(y)
        if self.gnn_pooling is None:
            x2 = self.down1(x1 + y1, t)
            x2 = self.sa1(x2)
            y2 = self.down1_cond(y1 + x1, t)
            y2 = self.sa1_cond(y2)
            g2 = self.genc2(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            g2_cond = self.genc2_cond(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x2 = self.ca1(x2, g2, g2)
            y2 = self.ca1_cond(y2, g2_cond, g2_cond)

            x3 = self.down2(x2 + y2, t)
            x3 = self.sa2(x3)
            y3 = self.down2_cond(y2 + x2, t)
            y3 = self.sa2_cond(y3)
            g3 = self.genc3(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            g3_cond = self.genc3_cond(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x3 = self.ca2(x3, g3, g3)
            y3 = self.ca2_cond(y3, g3_cond, g3_cond)

            x4 = self.down3(x3 + y3, t)
            x4 = self.sa3(x4)
            y4 = self.down3_cond(y3 + x3, t)
            y4 = self.sa3_cond(y4)
            g4 = self.genc4(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            g4_cond = self.genc4_cond(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x4 = self.ca3(x4, g4, g4)
            y4 = self.ca3_cond(y4, g4_cond, g4_cond)
        else:
            x2 = self.down1(x1 + y1, t)
            x2 = self.sa1(x2)
            y2 = self.down1_cond(y1 + x1, t)
            y2 = self.sa1_cond(y2)
            g2 = self.genc2(g).view(1, -1, 1).repeat(batch_size, 1, x2.size(2))
            g2_cond = self.genc2_cond(g).view(1, -1, 1).repeat(batch_size, 1, y2.size(2))

            x3 = self.down2(x2 + y2 + g2, t)
            x3 = self.sa2(x3)
            y3 = self.down2_cond(y2 + x2 + g2_cond, t)
            y3 = self.sa2_cond(y3)
            g3 = self.genc3(g).view(1, -1, 1).repeat(batch_size, 1, x3.size(2))
            g3_cond = self.genc3_cond(g).view(1, -1, 1).repeat(batch_size, 1, y3.size(2))

            x4 = self.down3(x3 + y3 + g3, t)
            x4 = self.sa3(x4)
            y4 = self.down3_cond(y3 + x3 + g3_cond, t)
            y4 = self.sa3_cond(y4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)
        y4 = self.bot1_cond(y4)
        y4 = self.bot2_cond(y4)
        y4 = self.bot3_cond(y4)

        if self.gnn_pooling is None:
            x = self.up1(x4 + y4, x3, t)
            x = self.sa4(x)
            y = self.up1_cond(y4 + x4, y3, t)
            y = self.sa4_cond(y)
            g5 = self.genc5(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            g5_cond = self.genc5_cond(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.ca4(x, g5, g5)
            y = self.ca4_cond(y, g5_cond, g5_cond)

            x_next = self.up2(x + y, x2, t)
            x_next = self.sa5(x_next)
            y = self.up2_cond(y + x, y2, t)
            y = self.sa5_cond(y)
            g6 = self.genc6(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            g6_cond = self.genc6_cond(g).view(-1, g.num_nodes).unsqueeze(0).repeat(batch_size, 1, 1)
            x = self.ca5(x_next, g6, g6)
            y = self.ca5_cond(y, g6_cond, g6_cond)

            x_next = self.up3(x + y, x1, t)
            y = self.up3_cond(y + x, y1, t)
            x = self.sa6(x_next)
            y = self.sa6_cond(y)
        else:
            g4 = self.genc4(g).view(1, -1, 1).repeat(batch_size, 1, x4.size(2))
            g4_cond = self.genc4_cond(g).view(1, -1, 1).repeat(batch_size, 1, y4.size(2))
            x = self.up1(x4 + y4 + g4, x3, t)
            x = self.sa4(x)
            y = self.up1_cond(y4 + x4 + g4_cond, y3, t)
            y = self.sa4_cond(y)

            g5 = self.genc5(g).view(1, -1, 1).repeat(batch_size, 1, x.size(2))
            g5_cond = self.genc5_cond(g).view(1, -1, 1).repeat(batch_size, 1, y.size(2))
            x_next = self.up2(x + y + g5, x2, t)
            x_next = self.sa5(x_next)
            y = self.up2_cond(y + x + g5_cond, y2, t)
            y = self.sa5_cond(y)

            g6 = self.genc6(g).view(1, -1, 1).repeat(batch_size, 1, x_next.size(2))
            g6_cond = self.genc6_cond(g).view(1, -1, 1).repeat(batch_size, 1, y.size(2))
            x = self.up3(x_next + y + g6, x1, t)
            x = self.sa6(x)
            y = self.up3_cond(y + x_next + g6_cond, y1, t)
            y = self.sa6_cond(y)

        x = self.outc(x + y)
        return x

    def forward(self, x, t, y=None, use_graph=True):
        """
        Forward pass for denoising.
        
        Args:
            x: Noisy input tensor (B, C, L)
            t: Diffusion timestep (B,)
            y: Optional conditioning tensor (B, C, L)
            use_graph: Whether to use graph conditioning
            
        Returns:
            x_hat: Denoised output tensor (B, C, L)
        """
        if use_graph:
            if y is not None:
                return self._forward_cond_graph(x, y, t)
            else:
                return self._forward_uncond_graph(x, t)
        else:
            if y is not None:
                return self._forward_cond(x, y, t)
            else:
                return self._forward_uncond(x, t)
