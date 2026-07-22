"""The proposed hybrid GraphDefect-KG classifier.

Fuses six complementary representations with a learnable gating mechanism:

    MobileNetV2 global embedding
  + MobileNetV2 patch embedding (mean-pooled)
  + handcrafted visual descriptor
  + GCN graph embedding
  + GNN (GraphSAGE/GAT) graph embedding
  + knowledge-graph semantic embedding
        -> gated fusion -> FC -> 6-class head

Each component is projected to a common width, a softmax gate weights the
components, and the gated concatenation feeds a small MLP classifier. The gate
values are exposed so the explanation can report which representation the model
relied on most.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig, settings
from .gcn_model import GCN
from .gnn_model import GNN

COMPONENTS = ["cnn_global", "cnn_patch", "handcrafted", "gcn", "gnn", "kg"]


@dataclass
class HybridInputs:
    """Bundles the tensors required for a hybrid forward pass (single graph)."""
    node_x: torch.Tensor          # (N, F) region node features
    edge_index: torch.Tensor      # (2, E)
    edge_weight: torch.Tensor     # (E,)
    cnn_global: torch.Tensor      # (1280,)
    cnn_patch: torch.Tensor       # (1280,) mean patch embedding
    handcrafted: torch.Tensor     # (H,)
    kg_embed: torch.Tensor        # (K,)


class GatedFusion(nn.Module):
    """Project components to a shared dim and combine them with a softmax gate."""

    def __init__(self, dims: Dict[str, int], shared: int):
        super().__init__()
        self.proj = nn.ModuleDict(
            {name: nn.Sequential(nn.Linear(d, shared), nn.LayerNorm(shared), nn.GELU())
             for name, d in dims.items()}
        )
        self.gate = nn.Linear(shared, 1)
        self.shared = shared
        self.last_gates: Optional[torch.Tensor] = None

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        projected = [self.proj[name](feats[name]) for name in COMPONENTS]
        stack = torch.stack(projected, dim=1)              # (B, C, shared)
        gate_logits = self.gate(stack).squeeze(-1)         # (B, C)
        gates = F.softmax(gate_logits, dim=1)
        self.last_gates = gates.detach()
        gated = stack * gates.unsqueeze(-1)
        return gated.reshape(gated.size(0), -1)            # (B, C*shared)


class HybridDefectModel(nn.Module):
    """Complete proposed model. Runs GCN + GNN internally and fuses everything."""

    def __init__(self, node_dim: int, handcrafted_dim: int, kg_dim: int,
                 cfg: Optional[ModelConfig] = None, num_classes: int = 6):
        super().__init__()
        cfg = cfg or settings.model
        self.cfg = cfg
        self.num_classes = num_classes

        self.gcn = GCN(node_dim, cfg, num_classes=num_classes)
        self.gnn = GNN(node_dim, arch="gat", cfg=cfg, num_classes=num_classes)

        shared = cfg.projected_dim
        dims = {
            "cnn_global": cfg.cnn_embed_dim,
            "cnn_patch": cfg.cnn_embed_dim,
            "handcrafted": handcrafted_dim,
            "gcn": self.gcn.embed_dim,
            "gnn": self.gnn.embed_dim,
            "kg": kg_dim,
        }
        self.fusion = GatedFusion(dims, shared)
        fused_dim = shared * len(COMPONENTS)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, cfg.fusion_hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.fusion_hidden, num_classes),
        )

    def forward(self, inp: HybridInputs) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch = torch.zeros(inp.node_x.size(0), dtype=torch.long, device=inp.node_x.device)
        _, gcn_embed = self.gcn(inp.node_x, inp.edge_index, inp.edge_weight, batch)
        _, gnn_embed = self.gnn(inp.node_x, inp.edge_index, inp.edge_weight, batch)

        feats = {
            "cnn_global": inp.cnn_global.unsqueeze(0),
            "cnn_patch": inp.cnn_patch.unsqueeze(0),
            "handcrafted": inp.handcrafted.unsqueeze(0),
            "gcn": gcn_embed,
            "gnn": gnn_embed,
            "kg": inp.kg_embed.unsqueeze(0),
        }
        fused = self.fusion(feats)
        logits = self.classifier(fused)
        aux = {
            "gcn_embed": gcn_embed,
            "gnn_embed": gnn_embed,
            "gates": self.fusion.last_gates,
            "edge_attention": self.gnn.edge_importance(),
        }
        return logits, aux

    def component_importance(self) -> Dict[str, float]:
        """Return the latest per-component gate weights as a plain dict."""
        if self.fusion.last_gates is None:
            return {c: 0.0 for c in COMPONENTS}
        g = self.fusion.last_gates.squeeze(0).cpu().tolist()
        return {c: round(float(v), 4) for c, v in zip(COMPONENTS, g)}
