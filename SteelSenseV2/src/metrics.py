"""Metrics and significance testing (Reviewer 1, Q9 and Q10).

Accuracy alone is not reportable on either dataset: SteelDefectX is imbalanced
(Inclusion is ~34% of the six-class subset, and the 24-class version is far
worse), so `evaluate()` always returns macro-F1, balanced accuracy, per-class
precision/recall/F1 and the full confusion matrix alongside accuracy.

Two kinds of significance test are provided and they answer different questions:

    mcnemar()      -- on ONE test set, are two models' error patterns different?
                      This is the right test for "model A beats model B on this
                      test set" because the predictions are paired per image.
    paired_over_seeds() -- across N seeds, is the mean difference non-zero?
                      Paired t-test plus Wilcoxon signed-rank, because with
                      five seeds the normality assumption is not checkable.

Confidence intervals on macro-F1 come from a stratified bootstrap over test
images (10k resamples by default), which is what makes a 0.4-point gap on a
360-image test set interpretable.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats as sstats
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Sequence[str],
    y_prob: Optional[np.ndarray] = None,
) -> Dict:
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(np.mean(p)),
        "macro_recall": float(np.mean(r)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
        "n_test": int(len(y_true)),
        "n_errors": int((y_true != y_pred).sum()),
        "per_class": {
            classes[i]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f[i]),
                "support": int(s[i]),
            }
            for i in labels
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classes": list(classes),
    }
    if y_prob is not None:
        eps = 1e-12
        pr = np.clip(y_prob[np.arange(len(y_true)), y_true], eps, 1.0)
        out["nll"] = float(-np.log(pr).mean())
        conf = y_prob.max(axis=1)
        out["mean_confidence"] = float(conf.mean())
        out["ece"] = float(expected_calibration_error(y_true, y_pred, conf))
    return out


def expected_calibration_error(
    y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, n_bins: int = 10
) -> float:
    correct = (y_true == y_pred).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = "macro_f1",
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(y_true)
    fn = {
        "macro_f1": lambda a, b: f1_score(a, b, average="macro", zero_division=0),
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }[metric]
    point = float(fn(y_true, y_pred))
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        vals[i] = fn(y_true[idx], y_pred[idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Dict:
    """Exact McNemar (binomial) on the discordant pairs.

    b = A right / B wrong, c = A wrong / B right. The exact test is used rather
    than the chi-square approximation because b + c is often below 25 here.
    """
    a_ok = pred_a == y_true
    b_ok = pred_b == y_true
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "test": "exact", "note": "identical errors"}
    p = float(min(1.0, 2.0 * sstats.binom.cdf(min(b, c), n, 0.5)))
    return {
        "b_a_right_b_wrong": b,
        "c_a_wrong_b_right": c,
        "p_value": p,
        "test": "exact binomial (two-sided)",
        "significant_at_0.05": bool(p < 0.05),
    }


def paired_over_seeds(a: Sequence[float], b: Sequence[float]) -> Dict:
    """Paired comparison across seeds. `a` and `b` must be seed-aligned."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return {"n_seeds": int(a.size), "note": "need >= 2 paired seeds"}
    d = a - b
    out = {
        "n_seeds": int(a.size),
        "mean_a": float(a.mean()),
        "sd_a": float(a.std(ddof=1)),
        "mean_b": float(b.mean()),
        "sd_b": float(b.std(ddof=1)),
        "mean_diff": float(d.mean()),
        "sd_diff": float(d.std(ddof=1)),
    }
    if np.allclose(d, 0):
        out.update({"t_p_value": 1.0, "wilcoxon_p_value": 1.0, "cohens_dz": 0.0})
        return out
    t, p = sstats.ttest_rel(a, b)
    out["t_stat"] = float(t)
    out["t_p_value"] = float(p)
    out["cohens_dz"] = float(d.mean() / (d.std(ddof=1) + 1e-12))
    try:
        w, pw = sstats.wilcoxon(a, b)
        out["wilcoxon_p_value"] = float(pw)
    except ValueError:
        out["wilcoxon_p_value"] = None
    out["significant_at_0.05"] = bool(p < 0.05)
    return out


def aggregate_seeds(runs: List[Dict], keys: Sequence[str] = None) -> Dict:
    """mean +- SD over seeds for the scalar metrics (Reviewer 1, Q9)."""
    keys = keys or [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "weighted_f1",
        "macro_precision",
        "macro_recall",
    ]
    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        v = np.array([r[k] for r in runs if k in r], dtype=float)
        if v.size == 0:
            continue
        out[k] = {
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "n": int(v.size),
            "values": [float(x) for x in v],
        }
    return out


def per_class_over_seeds(runs: List[Dict], classes: Sequence[str]) -> Dict:
    out = {}
    for c in classes:
        for m in ("precision", "recall", "f1"):
            v = np.array([r["per_class"][c][m] for r in runs], dtype=float)
            out.setdefault(c, {})[m] = {
                "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            }
        out[c]["support"] = runs[0]["per_class"][c]["support"]
    return out


def confusion_pair(cm: Sequence[Sequence[int]], classes: Sequence[str], a: str, b: str) -> Dict:
    """Reviewer 1 Q3 asks specifically for crazing vs rolled-in scale."""
    idx = {c.lower(): i for i, c in enumerate(classes)}
    ia, ib = idx.get(a.lower()), idx.get(b.lower())
    if ia is None or ib is None:
        return {"note": f"{a} or {b} not in {list(classes)}"}
    cm = np.asarray(cm)
    return {
        "pair": [a, b],
        f"{a}_as_{b}": int(cm[ia, ib]),
        f"{b}_as_{a}": int(cm[ib, ia]),
        f"{a}_support": int(cm[ia].sum()),
        f"{b}_support": int(cm[ib].sum()),
        "pairwise_error_rate_pct": round(
            100.0 * (cm[ia, ib] + cm[ib, ia]) / max(1, cm[ia].sum() + cm[ib].sum()), 3
        ),
    }
