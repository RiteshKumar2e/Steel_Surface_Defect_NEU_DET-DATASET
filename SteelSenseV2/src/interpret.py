"""Interpretability evidence (Reviewer 2, Q5).

"Interpretable" is a claim about the model, so it needs a measurement, and one
measurement is not enough because attention alone is famously not explanation.
Two independent views are produced and their agreement is reported:

  1. ATTENTION. The pooling attention distribution over the token sequence,
     averaged per true class over the test split. Because token position i is
     always feature i (canonical order), an attention weight is directly
     attributable to a named descriptor.

  2. PERMUTATION IMPORTANCE. For each descriptor, its token is resampled across
     the test split (breaking its association with the label while keeping the
     marginal distribution intact) and the drop in macro-F1 is recorded, over
     `n_repeats` shuffles. This is a causal measure of what the model uses; it
     needs no assumption about attention being faithful.

  3. AGREEMENT. Spearman correlation between the two rankings. High agreement
     is the evidence that the attention map can be shown to an operator as an
     explanation; low agreement means the attention figure must not be
     presented as one, and the paper should say so.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import stats as sstats
from sklearn.metrics import f1_score

from engine import ensemble_probs


@torch.no_grad()
def attention_by_class(
    model,
    snapshots: Sequence[Dict],
    ids: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    classes: Sequence[str],
    batch: int = 256,
) -> Dict:
    """Mean attention weight per (class, feature), averaged over snapshots."""
    n_feat = len(feature_names)
    acc = np.zeros((len(classes), n_feat), dtype=np.float64)
    cnt = np.zeros(len(classes), dtype=np.int64)

    for sd in snapshots:
        model.load_state_dict(sd)
        model.eval()
        for i in range(0, len(ids), batch):
            xb = torch.from_numpy(ids[i : i + batch])
            _, attn = model(xb, return_attention=True)
            a = attn.cpu().numpy()[:, 1 : n_feat + 1]   # drop <cls>
            for r, lab in enumerate(y[i : i + batch]):
                acc[lab] += a[r]
                cnt[lab] += 1
    mean = acc / np.maximum(cnt, 1)[:, None]

    out = {"classes": list(classes), "features": list(feature_names)}
    out["mean_attention"] = mean.tolist()
    out["top_by_class"] = {
        classes[c]: [
            {"feature": feature_names[j], "attention": round(float(mean[c, j]), 5)}
            for j in np.argsort(-mean[c])[:10]
        ]
        for c in range(len(classes))
    }
    glob = mean.mean(axis=0)
    out["global_ranking"] = [
        {"feature": feature_names[j], "attention": round(float(glob[j]), 5)}
        for j in np.argsort(-glob)
    ]
    # A class-DISCRIMINATIVE view: how far a class's attention departs from the
    # dataset mean. This is what actually distinguishes the classes; raw
    # attention is dominated by features every class attends to.
    dev = mean - glob[None, :]
    out["most_distinctive_by_class"] = {
        classes[c]: [
            {"feature": feature_names[j], "delta_vs_mean": round(float(dev[c, j]), 5)}
            for j in np.argsort(-dev[c])[:8]
        ]
        for c in range(len(classes))
    }
    return out


def permutation_importance(
    model,
    snapshots: Sequence[Dict],
    ids: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    classes: Sequence[str],
    n_repeats: int = 5,
    seed: int = 0,
    verbose: bool = True,
) -> Dict:
    """Drop in test macro-F1 when one feature's token column is resampled."""
    rng = np.random.RandomState(seed)
    base_probs = ensemble_probs(model, snapshots, ids)
    base = float(f1_score(y, base_probs.argmax(1), average="macro", zero_division=0))

    rows = []
    for j, name in enumerate(feature_names):
        col = j + 1  # position 0 is <cls>
        drops = []
        for _ in range(n_repeats):
            perm = ids.copy()
            perm[:, col] = perm[rng.permutation(len(perm)), col]
            p = ensemble_probs(model, snapshots, perm)
            drops.append(base - float(f1_score(y, p.argmax(1), average="macro", zero_division=0)))
        rows.append(
            {
                "feature": name,
                "mean_drop": float(np.mean(drops)),
                "sd_drop": float(np.std(drops, ddof=1)) if n_repeats > 1 else 0.0,
            }
        )
        if verbose and (j + 1) % 20 == 0:
            print(f"  permutation importance {j+1}/{len(feature_names)}")

    rows.sort(key=lambda r: -r["mean_drop"])
    return {
        "baseline_macro_f1": base,
        "n_repeats": n_repeats,
        "ranking": rows,
        "top20": rows[:20],
    }


def agreement(attn: Dict, perm: Dict) -> Dict:
    """Spearman rank correlation between the attention and permutation views."""
    a_map = {d["feature"]: d["attention"] for d in attn["global_ranking"]}
    p_map = {d["feature"]: d["mean_drop"] for d in perm["ranking"]}
    common = [f for f in a_map if f in p_map]
    a = np.array([a_map[f] for f in common])
    p = np.array([p_map[f] for f in common])
    rho, pv = sstats.spearmanr(a, p)
    top_a = {d["feature"] for d in attn["global_ranking"][:15]}
    top_p = {d["feature"] for d in perm["ranking"][:15]}
    return {
        "n_features": len(common),
        "spearman_rho": float(rho),
        "p_value": float(pv),
        "top15_overlap": len(top_a & top_p),
        "top15_shared_features": sorted(top_a & top_p),
        "interpretation": (
            "attention ranking tracks causal importance"
            if rho is not None and rho > 0.5
            else "attention ranking does NOT track causal importance; do not "
                 "present the attention map as an explanation"
        ),
    }
