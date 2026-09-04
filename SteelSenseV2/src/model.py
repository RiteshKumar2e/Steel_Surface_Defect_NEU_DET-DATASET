"""SteelSense-BiLSTM v2 and its order-free control.

Reviewer 1 Q6 -- "if order is irrelevant, justify the recurrent encoder" -- is
answered by shipping the control model in the same file, trained by the same
loop, on the same tokens:

    SteelSenseBiLSTM : embedding -> 2-layer BiLSTM -> {attention, max, mean}
                       pooling -> LayerNorm -> MLP head
    TokenDeepSets    : embedding -> per-token MLP -> {attention, max, mean}
                       pooling -> LayerNorm -> MLP head   (permutation
                       invariant by construction; no recurrence)

The two differ ONLY in the sequence encoder. If the BiLSTM does not beat the
DeepSets control, the recurrent encoder is not earning its parameters and the
paper should say so.

`forward(..., return_attention=True)` exposes the attention distribution, which
interpret.py aggregates per class (Reviewer 2, Q5).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim // 2)
        self.score = nn.Linear(dim // 2, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        s = self.score(torch.tanh(self.proj(h))).squeeze(-1)      # (B, L)
        s = s.masked_fill(~mask, torch.finfo(s.dtype).min)
        a = torch.softmax(s, dim=1)
        return torch.bmm(a.unsqueeze(1), h).squeeze(1), a


class _PooledHead(nn.Module):
    """Shared tail: pooled view(s) -> LayerNorm -> MLP -> logits.

    `pooling` selects which of the three pooled views feed the head:
    "all" (attention + max + mean, concatenated -- the deployed configuration),
    or any single one of "attention" / "max" / "mean". The single-view modes
    exist only to answer the ablation Reviewer 1 Q1 and Reviewer 2 Q6 ask for
    -- do attention pooling, max pooling and mean pooling each earn their
    parameters, or would one alone do the same job.
    """

    def __init__(self, enc_dim: int, num_classes: int, dropout: float, pooling: str = "all"):
        super().__init__()
        if pooling not in ("all", "attention", "max", "mean"):
            raise ValueError(f"unknown pooling mode {pooling!r}")
        self.pooling = pooling
        self.att = AttentionPool(enc_dim) if pooling in ("all", "attention") else None
        n_views = 3 if pooling == "all" else 1
        self.norm = nn.LayerNorm(enc_dim * n_views)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim * n_views, enc_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(enc_dim, num_classes),
        )

    def forward(self, h: torch.Tensor, mask: torch.Tensor):
        m = mask.unsqueeze(-1)
        views = []
        attn = None
        if self.pooling in ("all", "attention"):
            a_pool, attn = self.att(h, mask)
            views.append(a_pool)
        if self.pooling in ("all", "max"):
            mx = h.masked_fill(~m, torch.finfo(h.dtype).min).max(dim=1).values
            views.append(mx)
        if self.pooling in ("all", "mean"):
            mean = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
            views.append(mean)
        z = self.norm(torch.cat(views, dim=-1))
        return self.head(z), attn


class SteelSenseBiLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 96,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.30,
        pooling: str = "all",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.tail = _PooledHead(hidden_dim * 2, num_classes, dropout, pooling=pooling)

    def forward(self, ids: torch.Tensor, return_attention: bool = False):
        mask = ids.ne(0)
        h, _ = self.lstm(self.emb_drop(self.emb(ids)))
        logits, attn = self.tail(h, mask)
        return (logits, attn) if return_attention else logits


class TokenDeepSets(nn.Module):
    """Permutation-invariant control. Same embedding, same pooled head, no
    recurrence -- the encoder is a position-wise MLP."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 96,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.30,
        pooling: str = "all",
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.emb_drop = nn.Dropout(dropout * 0.5)
        layers = []
        d = embed_dim
        for _ in range(num_layers):
            layers += [nn.Linear(d, hidden_dim * 2), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim * 2
        self.enc = nn.Sequential(*layers)
        self.tail = _PooledHead(hidden_dim * 2, num_classes, dropout, pooling=pooling)

    def forward(self, ids: torch.Tensor, return_attention: bool = False):
        mask = ids.ne(0)
        h = self.enc(self.emb_drop(self.emb(ids)))
        logits, attn = self.tail(h, mask)
        return (logits, attn) if return_attention else logits


class ScalarEmbed(nn.Module):
    """Continuous analogue of `nn.Embedding`: position *i* (always feature
    *i*) gets its own learned scalar-to-vector projection, in place of a
    lookup table indexed by a discretized bin id."""

    def __init__(self, n_features: int, embed_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, F) standardized floats
        return x.unsqueeze(-1) * self.weight + self.bias  # (B, F, E)


class NumericBiLSTM(nn.Module):
    """Same architecture family as `SteelSenseBiLSTM` -- same encoder, same
    pooled head -- but fed the RAW standardized descriptor vector directly,
    with no discretization and no tokenizer anywhere in the path. Holding the
    encoder fixed and swapping only the input representation isolates what
    text-prompt discretization itself contributes (Reviewer 2 Q6, Reviewer 3
    Q7/Q8), which the tabular ML baselines (different model family entirely)
    cannot answer on their own.

    `vocab_size` is reused as `n_features` here so `build_model` can build
    every arch through one call site; there is no vocabulary for this arch.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 96,
        hidden_dim: int = 192,
        num_layers: int = 2,
        dropout: float = 0.30,
        pooling: str = "all",
    ):
        super().__init__()
        n_features = vocab_size
        self.scalar_embed = ScalarEmbed(n_features, embed_dim)
        self.emb_drop = nn.Dropout(dropout * 0.5)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.tail = _PooledHead(hidden_dim * 2, num_classes, dropout, pooling=pooling)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        h, _ = self.lstm(self.emb_drop(self.scalar_embed(x)))
        logits, attn = self.tail(h, mask)
        return (logits, attn) if return_attention else logits


ARCHS = {
    "bilstm": SteelSenseBiLSTM,
    "deepsets": TokenDeepSets,
    "bilstm_numeric": NumericBiLSTM,
}


def build_model(arch: str, vocab_size: int, num_classes: int, cfg) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch}; choose from {sorted(ARCHS)}")
    return ARCHS[arch](
        vocab_size=vocab_size,
        num_classes=num_classes,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pooling=getattr(cfg, "pooling", "all"),
    )


def count_parameters(model: nn.Module) -> Tuple[int, float]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
    return n, mb
