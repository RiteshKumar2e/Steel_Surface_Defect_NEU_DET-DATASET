"""KNN-based graph construction from region/patch node features.

Turns a set of node feature vectors (MobileNetV2 patch embeddings fused with
handcrafted descriptors) into a connected graph suitable for a GCN/GNN. Supports
Euclidean / cosine / combined (feature + spatial) distances, produces an
adjacency matrix + edge weights, adds self-loops, and guarantees no isolated
nodes. This is graph *construction* — a standalone KNN classifier lives in
``knn_classifier`` below for baseline comparison.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import GraphConfig, settings
from ..utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class RegionGraph:
    """Container for a single-image region graph."""

    node_features: np.ndarray                 # (N, F)
    node_meta: List[Dict]                     # per-node metadata (type, bbox, ...)
    edge_index: np.ndarray                    # (2, E) int
    edge_weight: np.ndarray                   # (E,) float in (0, 1]
    positions: np.ndarray                     # (N, 2) normalised centres
    adjacency: np.ndarray = field(default=None)  # (N, N) dense weights

    @property
    def num_nodes(self) -> int:
        return int(self.node_features.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    def to_torch(self):
        """Return a PyTorch-Geometric ``Data`` object (or a light dict fallback)."""
        import torch
        x = torch.from_numpy(self.node_features.astype(np.float32))
        edge_index = torch.from_numpy(self.edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(self.edge_weight.astype(np.float32))
        try:
            from torch_geometric.data import Data
            return Data(x=x, edge_index=edge_index, edge_attr=edge_weight)
        except Exception:
            return {"x": x, "edge_index": edge_index, "edge_weight": edge_weight}


def _pairwise_distance(feats: np.ndarray, metric: str) -> np.ndarray:
    """Dense pairwise distance matrix in [0, 1] (lower = more similar)."""
    from sklearn.metrics import pairwise_distances

    if metric == "cosine":
        d = pairwise_distances(feats, metric="cosine")      # in [0, 2]
        return d / 2.0
    if metric == "euclidean":
        d = pairwise_distances(feats, metric="euclidean")
        return d / (d.max() + 1e-9)
    raise ValueError(f"Unknown metric: {metric}")


def build_knn_graph(
    node_features: np.ndarray,
    node_meta: List[Dict],
    positions: np.ndarray,
    cfg: Optional[GraphConfig] = None,
) -> RegionGraph:
    """Construct a KNN graph over ``node_features``.

    Parameters
    ----------
    node_features : (N, F) fused node descriptors.
    node_meta     : per-node metadata dicts (kept for visualisation).
    positions     : (N, 2) normalised patch centres for spatial distance.
    """
    cfg = cfg or settings.graph
    n = node_features.shape[0]
    if n < 2:
        # Degenerate graph: single self-loop.
        return RegionGraph(
            node_features=node_features,
            node_meta=node_meta,
            edge_index=np.array([[0], [0]], dtype=np.int64),
            edge_weight=np.array([1.0], dtype=np.float32),
            positions=positions,
            adjacency=np.ones((n, n), dtype=np.float32),
        )

    k = min(cfg.knn_k, n - 1)

    if cfg.metric == "combined":
        feat_d = _pairwise_distance(node_features, "cosine")
        pos_d = _pairwise_distance(positions, "euclidean")
        dist = (1 - cfg.spatial_weight) * feat_d + cfg.spatial_weight * pos_d
    else:
        dist = _pairwise_distance(node_features, cfg.metric)

    np.fill_diagonal(dist, np.inf)                 # exclude self for neighbour search
    adjacency = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        nn_idx = np.argsort(dist[i])[:k]
        for j in nn_idx:
            w = float(np.exp(-dist[i, j]))         # similarity weight in (0, 1]
            adjacency[i, j] = max(adjacency[i, j], w)

    if cfg.symmetric:
        adjacency = np.maximum(adjacency, adjacency.T)

    # Guarantee no isolated node: connect to its single nearest neighbour.
    for i in range(n):
        if adjacency[i].sum() == 0:
            j = int(np.argmin(dist[i]))
            adjacency[i, j] = adjacency[j, i] = float(np.exp(-dist[i, j]))

    if cfg.add_self_loops:
        np.fill_diagonal(adjacency, 1.0)

    src, dst = np.nonzero(adjacency)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_weight = adjacency[src, dst].astype(np.float32)

    return RegionGraph(
        node_features=node_features,
        node_meta=node_meta,
        edge_index=edge_index,
        edge_weight=edge_weight,
        positions=positions,
        adjacency=adjacency,
    )


def normalized_adjacency(adjacency: np.ndarray) -> np.ndarray:
    """Symmetric normalisation D^-1/2 (A) D^-1/2 (for reference / analysis)."""
    deg = adjacency.sum(axis=1)
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0.0
    return adjacency * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
