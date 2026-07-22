"""MobileNetV2 deep feature extractor.

Wraps ``torchvision`` MobileNetV2 to produce:
  * a global image embedding (1280-d penultimate features),
  * patch-level embeddings (one per preprocessing patch),
  * an optional low-dimensional projection with attention refinement.

The extractor never makes the final class decision on its own — its outputs
become node attributes in the graph. Supports frozen-backbone and fine-tuning
modes and falls back to CPU automatically.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..config import ModelConfig, settings
from ..utils.helpers import get_logger
from ..utils.preprocessing import to_cnn_tensor

logger = get_logger(__name__)


class FeatureProjector(nn.Module):
    """Linear projection + optional self-attention refinement of node features."""

    def __init__(self, in_dim: int, out_dim: int, use_attention: bool = True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )
        self.use_attention = use_attention
        if use_attention:
            self.attn = nn.MultiheadAttention(out_dim, num_heads=4, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (N, in_dim)
        h = self.proj(x)
        if self.use_attention and h.size(0) > 1:
            h_seq = h.unsqueeze(0)                      # (1, N, out_dim)
            refined, _ = self.attn(h_seq, h_seq, h_seq)
            h = (h + refined.squeeze(0))                # residual
        return h


class MobileNetV2FeatureExtractor(nn.Module):
    """Backbone feature extractor with global + patch embeddings."""

    def __init__(self, cfg: Optional[ModelConfig] = None, device: Optional[str] = None):
        super().__init__()
        self.cfg = cfg or settings.model
        self.device = device or settings.resolve_device()

        from torchvision import models

        try:
            weights = (
                models.MobileNet_V2_Weights.IMAGENET1K_V1
                if self.cfg.pretrained else None
            )
            backbone = models.mobilenet_v2(weights=weights)
            self.pretrained_loaded = self.cfg.pretrained
        except Exception as exc:  # offline / no weights cache
            logger.warning("Could not load pretrained weights (%s); using random init", exc)
            backbone = models.mobilenet_v2(weights=None)
            self.pretrained_loaded = False

        self.features = backbone.features              # conv feature maps
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed_dim = self.cfg.cnn_embed_dim
        self.projector = FeatureProjector(
            self.embed_dim, self.cfg.projected_dim, use_attention=True
        )

        if self.cfg.freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.to(self.device)
        self.eval()

    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def _backbone_embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return the 1280-d pooled embedding for a batch of CNN tensors."""
        fmap = self.features(x.to(self.device))        # (B, 1280, h, w)
        pooled = self.pool(fmap).flatten(1)            # (B, 1280)
        return pooled

    @torch.no_grad()
    def global_embedding(self, resized_bgr: np.ndarray) -> np.ndarray:
        """1280-d embedding for the whole (resized) image."""
        x = to_cnn_tensor(resized_bgr)
        return self._backbone_embed(x).cpu().numpy()[0]

    @torch.no_grad()
    def patch_embeddings(self, patches: List[np.ndarray]) -> np.ndarray:
        """Stacked 1280-d embeddings, one per patch image (grayscale or BGR)."""
        tensors = []
        for p in patches:
            if p.ndim == 2:
                p = np.stack([p] * 3, axis=-1)
            tensors.append(to_cnn_tensor(p))
        batch = torch.cat(tensors, dim=0)
        return self._backbone_embed(batch).cpu().numpy()

    @torch.no_grad()
    def project(self, embeddings: np.ndarray) -> np.ndarray:
        """Project ``(N, 1280)`` embeddings to ``(N, projected_dim)``."""
        t = torch.from_numpy(np.atleast_2d(embeddings).astype(np.float32)).to(self.device)
        return self.projector(t).cpu().numpy()

    def info(self) -> dict:
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "backbone": "MobileNetV2",
            "pretrained": self.pretrained_loaded,
            "frozen": self.cfg.freeze_backbone,
            "embed_dim": self.embed_dim,
            "projected_dim": self.cfg.projected_dim,
            "parameters": int(n_params),
            "device": self.device,
        }
