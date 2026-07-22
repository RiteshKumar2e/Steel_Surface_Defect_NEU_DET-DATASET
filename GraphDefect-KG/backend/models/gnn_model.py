"""Additional GNN architectures: GraphSAGE and Graph Attention Network (GAT).

Both are implemented with native PyTorch message passing (reusing the scatter
helpers from :mod:`gcn_model`) so they run without the optional torch-scatter
extension. GAT additionally exposes per-edge attention weights, which the graph
explainer surfaces as edge importance.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, settings
from .gcn_model import global_pool, scatter_add, scatter_mean


class SAGEConv(nn.Module):
    """GraphSAGE layer with mean aggregation and a self+neighbour transform."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin_self = nn.Linear(in_dim, out_dim)
        self.lin_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        row, col = edge_index[0], edge_index[1]
        msg = x[col]
        if edge_weight is not None:
            msg = msg * edge_weight.unsqueeze(-1)
        agg = scatter_mean(msg, row, num_nodes)
        return self.lin_self(x) + self.lin_neigh(agg)


class GATConv(nn.Module):
    """Single-layer multi-head graph attention with exposed attention weights."""

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.lin = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.att_src = nn.Parameter(torch.empty(1, heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(1, heads, out_dim))
        self.bias = nn.Parameter(torch.zeros(heads * out_dim))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        self._last_attention: Optional[torch.Tensor] = None

    def forward(self, x, edge_index, edge_weight=None):
        num_nodes = x.size(0)
        h = self.lin(x).view(num_nodes, self.heads, self.out_dim)
        row, col = edge_index[0], edge_index[1]        # row=target, col=source
        alpha = (h[col] * self.att_src).sum(-1) + (h[row] * self.att_dst).sum(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        # softmax over incoming edges per target node
        alpha = alpha - scatter_add(alpha.exp(), row, num_nodes)[row].clamp(min=1e-9).log()
        alpha = alpha.exp()
        denom = scatter_add(alpha, row, num_nodes)[row].clamp(min=1e-9)
        alpha = alpha / denom                          # (E, heads)
        self._last_attention = alpha.mean(dim=1).detach()   # (E,) mean over heads
        msg = h[col] * alpha.unsqueeze(-1)             # (E, heads, out_dim)
        out = torch.zeros(num_nodes, self.heads, self.out_dim, device=x.device)
        idx = row.view(-1, 1, 1).expand_as(msg)
        out = out.scatter_add_(0, idx, msg)
        return out.reshape(num_nodes, self.heads * self.out_dim) + self.bias

    @property
    def attention(self) -> Optional[torch.Tensor]:
        return self._last_attention


class GNN(nn.Module):
    """Configurable GNN (``graphsage`` or ``gat``) with a pooled classifier head."""

    def __init__(self, in_dim: int, arch: str = "graphsage",
                 cfg: Optional[ModelConfig] = None, num_classes: int = 6):
        super().__init__()
        cfg = cfg or settings.model
        self.arch = arch.lower()
        h = cfg.gnn_hidden
        self.input_proj = nn.Linear(in_dim, h)
        self.dropout = cfg.dropout
        self.edge_attention: List[torch.Tensor] = []

        if self.arch == "gat":
            per_head = h // cfg.gat_heads
            self.conv1 = GATConv(h, per_head, heads=cfg.gat_heads)
            self.conv2 = GATConv(h, per_head, heads=cfg.gat_heads)
        elif self.arch in ("graphsage", "sage"):
            self.conv1 = SAGEConv(h, h)
            self.conv2 = SAGEConv(h, h)
        else:
            raise ValueError(f"Unknown GNN arch: {arch}")

        self.bn1 = nn.BatchNorm1d(h)
        self.bn2 = nn.BatchNorm1d(h)
        self.embed_dim = 2 * h
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, num_classes),
        )

    def forward(self, x, edge_index, edge_weight=None, batch=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.edge_attention = []
        h = F.gelu(self.input_proj(x))
        h = F.gelu(self.bn1(self.conv1(h, edge_index, edge_weight)))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.gelu(self.bn2(self.conv2(h, edge_index, edge_weight)))
        if self.arch == "gat":
            self.edge_attention = [
                self.conv1.attention, self.conv2.attention,
            ]
        graph_embed = global_pool(h, batch)
        return self.classifier(graph_embed), graph_embed

    def edge_importance(self) -> Optional[torch.Tensor]:
        """Mean attention over layers (GAT only); ``None`` for GraphSAGE."""
        atts = [a for a in self.edge_attention if a is not None]
        if not atts:
            return None
        return torch.stack(atts).mean(0)
