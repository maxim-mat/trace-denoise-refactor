import torch
import torch.nn as nn
from torch_geometric.nn import (
    GINConv, SAGEConv, GCNConv, GATv2Conv,
    global_add_pool, global_mean_pool
)
# not implemented
# from modules.GPSLayer import GPSLayer


def build_gnn_layer(kind: str, in_dim: int, out_dim: int, dropout: float, n_heads: int=4):
    """Factory for arbitrary GNN layers."""
    kind = kind.lower()

    if kind == "gin":
        mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU()
        )
        return GINConv(mlp)

    elif kind == "graphsage":
        return SAGEConv(in_dim, out_dim)

    elif kind == "gcn":
        return GCNConv(in_dim, out_dim)

    elif kind == "gat":
        heads = n_heads
        assert out_dim % heads == 0
        return GATv2Conv(in_dim, out_dim // heads, heads=heads, dropout=dropout)
    # not implemented
    # elif kind == "gps":
    #     return GPSLayer(dim=in_dim,
    #             edge_dim=in_dim,
    #             num_heads=n_heads,
    #             dropout=dropout)

    else:
        raise ValueError(f"Unknown conv type: {kind}")


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_nodes,
        embedding_dim,
        hidden_dim,
        output_dim,
        num_layers=5,
        dropout_prob=0.1,
        pooling='add',
        conv_type="gin",
        attention_heads=4,  # optional and relevant only for gat and gps
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_nodes, embedding_dim)

        # Pooling
        if pooling == "add":
            self.pool_func = global_add_pool
        elif pooling == "mean":
            self.pool_func = global_mean_pool
        elif pooling is None:
            self.pool_func = None
        else:
            raise ValueError(f"Invalid pooling={pooling}")

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_dim = embedding_dim if i == 0 else hidden_dim
            self.convs.append(build_gnn_layer(conv_type, in_dim, hidden_dim, dropout_prob))

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        if x.dim() == 2:
            x = x.squeeze(1)

        x = self.embedding(x)

        for conv in self.convs:
            x = conv(x, edge_index)

        if self.pool_func is not None:
            batch = data.batch if data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.pool_func(x, batch=batch)

        x = self.out(x)
        return x
