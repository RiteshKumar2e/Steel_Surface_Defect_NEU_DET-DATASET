"""Discretization of the descriptor vector (Reviewer 1, Q7 and Q8).

Two questions the original submission did not answer:

Q7  "Why three semantic levels?"  -> `n_bins` is a swept hyper-parameter, and
    both `quantile` and `uniform` (equal-width) edge rules are implemented so
    the sweep in run_ablations.py can compare them under identical splits.

Q8  "What happens when a value falls outside the fitted range at inference?"
    -> The edges are fitted on the TRAIN split only, so out-of-range values at
    val/test time are expected and must have a declared policy:

        clamp      : the value is assigned to the first/last bin. The bin token
                     stays in-vocabulary, so no <unk> is ever produced by a
                     numeric feature. Occurrences are COUNTED per feature and
                     written to the run record.
        oor_token  : the value gets a dedicated `<feat>_oorlo` / `<feat>_oorhi`
                     token, which is added to the vocabulary at fit time (with
                     zero training occurrences) so it is never <unk> either.

    Either way the rate is reported. `clamp` is the default because a value
    slightly beyond the training minimum is far closer in meaning to bin 0 than
    to an unrelated symbol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from config import BinConfig


@dataclass
class Discretizer:
    n_bins: int
    strategy: str
    oor_policy: str
    names: List[str] = field(default_factory=list)
    edges: Dict[str, np.ndarray] = field(default_factory=dict)
    # Populated by transform(); reset with reset_counters().
    oor_low: Dict[str, int] = field(default_factory=dict)
    oor_high: Dict[str, int] = field(default_factory=dict)
    n_seen: int = 0

    # -- fitting ------------------------------------------------------------
    @classmethod
    def fit(cls, X: np.ndarray, names: List[str], cfg: BinConfig) -> "Discretizer":
        """X must contain TRAIN ROWS ONLY."""
        d = cls(n_bins=cfg.n_bins, strategy=cfg.strategy, oor_policy=cfg.oor_policy,
                names=list(names))
        for j, nm in enumerate(names):
            col = X[:, j]
            if cfg.strategy == "quantile":
                qs = np.linspace(0, 100, cfg.n_bins + 1)[1:-1]
                e = np.percentile(col, qs)
                # Constant / near-constant features collapse; keep them strictly
                # increasing so np.searchsorted stays well defined.
                e = np.unique(e)
                if e.size < cfg.n_bins - 1:
                    lo, hi = float(col.min()), float(col.max())
                    if hi - lo < 1e-9:
                        hi = lo + 1e-6
                    e = np.linspace(lo, hi, cfg.n_bins + 1)[1:-1]
            elif cfg.strategy == "uniform":
                lo, hi = float(col.min()), float(col.max())
                if hi - lo < 1e-9:
                    hi = lo + 1e-6
                e = np.linspace(lo, hi, cfg.n_bins + 1)[1:-1]
            else:
                raise ValueError(f"unknown strategy {cfg.strategy}")
            d.edges[nm] = np.asarray(e, dtype=np.float64)
        d.train_min = {nm: float(X[:, j].min()) for j, nm in enumerate(names)}
        d.train_max = {nm: float(X[:, j].max()) for j, nm in enumerate(names)}
        d.reset_counters()
        return d

    def reset_counters(self) -> None:
        self.oor_low = {n: 0 for n in self.names}
        self.oor_high = {n: 0 for n in self.names}
        self.n_seen = 0

    # -- transform ----------------------------------------------------------
    def transform(self, X: np.ndarray, count_oor: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (bin_index, oor_flag).

        bin_index: (N, F) int8 in [0, n_bins)
        oor_flag : (N, F) int8 in {-1, 0, +1} -- below range, in range, above.
        """
        N, F = X.shape
        B = np.zeros((N, F), dtype=np.int16)
        O = np.zeros((N, F), dtype=np.int8)
        for j, nm in enumerate(self.names):
            col = X[:, j].astype(np.float64)
            B[:, j] = np.searchsorted(self.edges[nm], col, side="right")
            lo = col < self.train_min[nm]
            hi = col > self.train_max[nm]
            O[lo, j] = -1
            O[hi, j] = 1
            if count_oor:
                self.oor_low[nm] += int(lo.sum())
                self.oor_high[nm] += int(hi.sum())
        if count_oor:
            self.n_seen += N
        np.clip(B, 0, self.n_bins - 1, out=B)
        return B.astype(np.int16), O

    # -- reporting ----------------------------------------------------------
    def oor_report(self) -> dict:
        if self.n_seen == 0:
            return {"n_seen": 0}
        per: Dict[str, dict] = {}
        tot_lo = tot_hi = 0
        for nm in self.names:
            lo, hi = self.oor_low[nm], self.oor_high[nm]
            tot_lo += lo
            tot_hi += hi
            if lo or hi:
                per[nm] = {
                    "below": lo,
                    "above": hi,
                    "rate_pct": round(100.0 * (lo + hi) / self.n_seen, 3),
                }
        n_feat = len(self.names)
        return {
            "policy": self.oor_policy,
            "n_rows_seen": self.n_seen,
            "n_features": n_feat,
            "total_value_checks": self.n_seen * n_feat,
            "n_out_of_range_values": tot_lo + tot_hi,
            "overall_rate_pct": round(
                100.0 * (tot_lo + tot_hi) / (self.n_seen * n_feat), 4
            ),
            "mean_oor_features_per_image": round((tot_lo + tot_hi) / self.n_seen, 3),
            "per_feature": dict(
                sorted(per.items(), key=lambda kv: -kv[1]["rate_pct"])[:25]
            ),
        }

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "n_bins": self.n_bins,
            "strategy": self.strategy,
            "oor_policy": self.oor_policy,
            "names": self.names,
            "edges": {k: v.tolist() for k, v in self.edges.items()},
            "train_min": self.train_min,
            "train_max": self.train_max,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> "Discretizer":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        d = cls(p["n_bins"], p["strategy"], p["oor_policy"], p["names"])
        d.edges = {k: np.asarray(v, dtype=np.float64) for k, v in p["edges"].items()}
        d.train_min = p["train_min"]
        d.train_max = p["train_max"]
        d.reset_counters()
        return d
