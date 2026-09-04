"""Training and evaluation loop.

Selection rule, stated once and enforced here: checkpoints are ranked by
VALIDATION macro-F1. The snapshot ensemble is the top-k of that ranking. The
test split is passed to `evaluate_model` exactly once, after `fit` returns, and
`fit` never receives it.

This is the difference that matters relative to the original submission, where
the top-5 checkpoints were selected on the same 20% used to report the headline
accuracy.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, TrainConfig
from metrics import evaluate
from model import build_model, count_parameters


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _loader(ids: np.ndarray, y: np.ndarray, bs: int, shuffle: bool, seed: int = 0):
    ds = TensorDataset(torch.from_numpy(ids), torch.from_numpy(y))
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=g if shuffle else None)


@dataclass
class FitResult:
    snapshots: List[Dict]
    val_scores: List[float]
    history: List[Dict]
    n_params: int
    size_mb: float
    train_seconds: float
    epochs_run: int


@torch.no_grad()
def _predict_probs(model: nn.Module, ids: np.ndarray, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        batch = torch.from_numpy(ids[i : i + bs])
        out.append(torch.softmax(model(batch), dim=-1).cpu().numpy())
    return np.concatenate(out, axis=0)


def ensemble_probs(
    model: nn.Module, snapshots: Sequence[Dict], ids: np.ndarray, bs: int = 256
) -> np.ndarray:
    acc = None
    for sd in snapshots:
        model.load_state_dict(sd)
        p = _predict_probs(model, ids, bs)
        acc = p if acc is None else acc + p
    return acc / len(snapshots)


def fit(
    train_ids: np.ndarray,
    train_y: np.ndarray,
    val_ids: np.ndarray,
    val_y: np.ndarray,
    vocab_size: int,
    num_classes: int,
    mcfg: ModelConfig,
    tcfg: TrainConfig,
    classes: Sequence[str],
    seed: int = 42,
    verbose: bool = True,
) -> Tuple[nn.Module, FitResult]:
    set_seed(seed)
    model = build_model(mcfg.arch, vocab_size, num_classes, mcfg)
    n_params, size_mb = count_parameters(model)

    weight = None
    if tcfg.class_weighting:
        counts = np.bincount(train_y, minlength=num_classes).astype(np.float64)
        w = counts.sum() / (num_classes * np.maximum(counts, 1))
        weight = torch.tensor(w, dtype=torch.float32)

    crit = nn.CrossEntropyLoss(weight=weight, label_smoothing=tcfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    dl = _loader(train_ids, train_y, tcfg.batch_size, True, seed)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=tcfg.lr, epochs=tcfg.epochs,
        steps_per_epoch=max(1, len(dl)), pct_start=tcfg.pct_start,
    )

    snapshots: List[Tuple[float, Dict]] = []
    history: List[Dict] = []
    best, since_best = -1.0, 0
    t0 = time.perf_counter()
    epochs_run = 0

    for ep in range(tcfg.epochs):
        model.train()
        tot, n = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
            opt.step()
            sched.step()
            tot += float(loss.item()) * len(yb)
            n += len(yb)
        epochs_run = ep + 1

        # ---- validation only ------------------------------------------
        vp = _predict_probs(model, val_ids)
        vm = evaluate(val_y, vp.argmax(1), classes, vp)
        score = vm["macro_f1"] if tcfg.select_metric == "val_macro_f1" else vm["accuracy"]
        history.append(
            {"epoch": ep + 1, "train_loss": tot / max(1, n),
             "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"]}
        )
        snapshots.append(
            (score, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        )
        snapshots = sorted(snapshots, key=lambda s: -s[0])[: tcfg.ensemble_size]

        if score > best + 1e-5:
            best, since_best = score, 0
        else:
            since_best += 1
        if verbose and ((ep + 1) % 5 == 0 or ep == 0):
            print(f"  ep {ep+1:3d}  loss {tot/max(1,n):.4f}  "
                  f"val_acc {vm['accuracy']:.4f}  val_macroF1 {vm['macro_f1']:.4f}")
        if since_best >= tcfg.early_stop_patience:
            if verbose:
                print(f"  early stop at epoch {ep+1} (no val gain for "
                      f"{tcfg.early_stop_patience} epochs)")
            break

    res = FitResult(
        snapshots=[sd for _, sd in snapshots],
        val_scores=[float(s) for s, _ in snapshots],
        history=history,
        n_params=n_params,
        size_mb=size_mb,
        train_seconds=time.perf_counter() - t0,
        epochs_run=epochs_run,
    )
    return model, res


def evaluate_model(
    model: nn.Module,
    res: FitResult,
    ids: np.ndarray,
    y: np.ndarray,
    classes: Sequence[str],
    use_ensemble: bool = True,
) -> Tuple[Dict, np.ndarray]:
    """The single, final touch of the held-out split."""
    if use_ensemble:
        probs = ensemble_probs(model, res.snapshots, ids)
    else:
        model.load_state_dict(res.snapshots[0])
        probs = _predict_probs(model, ids)
    m = evaluate(y, probs.argmax(1), classes, probs)
    m["ensemble"] = bool(use_ensemble)
    m["ensemble_size"] = len(res.snapshots) if use_ensemble else 1
    m["val_scores_of_snapshots"] = res.val_scores
    m["n_params"] = res.n_params
    m["size_mb"] = round(res.size_mb, 3)
    m["train_seconds"] = round(res.train_seconds, 2)
    m["epochs_run"] = res.epochs_run
    return m, probs
