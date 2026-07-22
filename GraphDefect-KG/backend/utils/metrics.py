"""Evaluation metrics used by the notebook and the (optional) training code.

Thin, dependency-light wrappers around scikit-learn so the same metric names are
used consistently in the notebook, tests and API model-info endpoint.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

import numpy as np


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 (macro & weighted) and optional AUC."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    if y_proba is not None:
        try:
            out["roc_auc_ovr"] = float(
                roc_auc_score(y_true, y_proba, multi_class="ovr", labels=labels)
            )
        except Exception:
            out["roc_auc_ovr"] = float("nan")
    return out


def confusion(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    from sklearn.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def count_parameters(model) -> int:
    """Number of trainable parameters in a torch module."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def time_inference(fn, *args, repeats: int = 5, **kwargs) -> Dict[str, float]:
    """Return mean/std inference time (ms) over ``repeats`` calls."""
    times: List[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(*args, **kwargs)
        times.append((time.perf_counter() - start) * 1000.0)
    arr = np.array(times)
    return {"mean_ms": float(arr.mean()), "std_ms": float(arr.std())}


def iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection-over-union for two ``[x0, y0, x1, y1]`` boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter + 1e-9
    return float(inter / union)
