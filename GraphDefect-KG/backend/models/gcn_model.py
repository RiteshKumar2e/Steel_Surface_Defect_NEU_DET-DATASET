"""Graph Convolutional Network for graph-level defect classification.

Implemented with native PyTorch message passing (``index_add_`` based scatter)
so it runs even when the optional ``torch-scatter``/``torch-sparse`` extensions
fail to load. Interfaces accept the standard ``(x, edge_index, edge_weight,
batch)`` signature used by PyTorch Geometric, so a PyG ``Data``/``Batch`` object
can be passed field-by-field.

Architecture (per spec):
  input projection -> N x [GCNConv -> BN -> GELU -> Dropout (+residual)]
  -> global mean+max pooling -> MLP classifier -> 6-class logits.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, settings


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum ``src`` rows into ``dim_size`` buckets given by ``index``."""
    out = src.new_zeros((dim_size,) + src.shape[1:])
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    summed = scatter_add(src, index, dim_size)
    count = scatter_add(torch.ones_like(index, dtype=src.dtype), index, dim_size).clamp(min=1)
    return summed / count.view(-1, *([1] * (src.dim() - 1)))


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = src.new_full((dim_size,) + src.shape[1:], float("-inf"))
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    out = out.scatter_reduce(0, idx, src, reduce="amax", include_self=True)
    return torch.nan_to_num(out, neginf=0.0)


def gcn_norm(edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int):
    """Symmetric normalisation coefficients for GCN aggregation."""
    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(edge_weight, row, num_nodes).clamp(min=1e-12)
    d_inv_sqrt = deg.pow(-0.5)
    return d_inv_sqrt[row] * edge_weight * d_inv_sqrt[col]


class GCNConv(nn.Module):
    """Single graph-convolution layer with edge weights."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight):
        num_nodes = x.size(0)
        h = self.lin(x)
        norm = gcn_norm(edge_index, edge_weight, num_nodes)
        row, col = edge_index[0], edge_index[1]
        messages = h[col] * norm.unsqueeze(-1)
        return scatter_add(messages, row, num_nodes)


def global_pool(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    """Concatenated global mean + max pooling."""
    if batch is None:
        batch = x.new_zeros(x.size(0), dtype=torch.long)
    n_graphs = int(batch.max().item()) + 1
    mean = scatter_mean(x, batch, n_graphs)
    mx = scatter_max(x, batch, n_graphs)
    return torch.cat([mean, mx], dim=-1)


class GCN(nn.Module):
    """Multi-layer GCN with residual connections and a pooled classifier head."""

    def __init__(self, in_dim: int, cfg: Optional[ModelConfig] = None,
                 num_classes: int = 6, return_embedding: bool = True):
        super().__init__()
        cfg = cfg or settings.model
        h = cfg.gcn_hidden
        self.input_proj = nn.Linear(in_dim, h)
        self.convs = nn.ModuleList([GCNConv(h, h) for _ in range(cfg.gcn_layers)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(cfg.gcn_layers)])
        self.dropout = cfg.dropout
        self.embed_dim = 2 * h          # mean+max pooling doubles width
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, h), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(h, num_classes),
        )
        self.return_embedding = return_embedding

    def forward(self, x, edge_index, edge_weight=None, batch=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        h = F.gelu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            h_new = conv(h, edge_index, edge_weight)
            h_new = bn(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            h = h + h_new                       # residual
        graph_embed = global_pool(h, batch)
        logits = self.classifier(graph_embed)
        if self.return_embedding:
            return logits, graph_embed
        return logits, h                        # node embeddings

    def node_embeddings(self, x, edge_index, edge_weight=None) -> torch.Tensor:
        """Return per-node hidden features (used for node-importance analysis)."""
        if edge_weight is None:
            edge_weight = x.new_ones(edge_index.size(1))
        h = F.gelu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            h = h + F.gelu(bn(conv(h, edge_index, edge_weight)))
        return h
