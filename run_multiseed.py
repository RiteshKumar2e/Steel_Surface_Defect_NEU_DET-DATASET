#!/usr/bin/env python
"""Multi-seed evaluation of the architecture ladder and the backbone baselines.

Why this script exists
----------------------
Every accuracy currently in the manuscript comes from ONE seed on a 360-image
test set. At that size one test image is 0.28 accuracy points, and the existing
single-seed numbers swing far more than that between runs that should be nearly
identical:

    "full" model, seed 42, 15+15 epochs  ->  99.72   (backbone_comparison.csv)
    "full" model, seed 42, 30+30 epochs  ->  93.33   (table_ablation.csv)

    AMFF channel-only  -> 100.00 |  AMFF spatial-only -> 82.78
    SEAM rates (1,3,5) ->  93.33 |  SEAM rates (1,3)  ->  99.44

Those variants differ by a handful of parameters, so a 17-point spread is run
variance, not architecture. No ranking computed from one seed is defensible, in
either direction. CFG.model_seeds already declares five seeds; only one was ever
run. This script runs all of them and reports mean +- sd.

    python run_multiseed.py
    MS_VARIANTS=baseline,fpn_amff,full python run_multiseed.py
    MS_SEEDS=42,1337,2026 MS_EF=15 MS_ET=15 python run_multiseed.py
    MS_BACKBONES=MobileNetV2,ResNet50 python run_multiseed.py

Rules kept from run_backbone_comparison.py: the frozen split is never
regenerated, selection is on validation only, metrics are recomputed from saved
predictions, results are written after every run so an interrupt resumes, and a
failed run is recorded as failed rather than given a number.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import platform
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

# On a CPU-only machine TensorFlow does not always pick up every core. Set the
# thread pools explicitly before TF is imported anywhere else; measured on this
# project a step costs ~9 s at batch 32, so a bad thread count is expensive.
_CORES = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(_CORES))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(_CORES))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "new_model_code.ipynb"
OUT = ROOT / "results_multiseed"
PRED_DIR = OUT / "predictions"
for d in (OUT, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

ROWS_CSV = OUT / "runs.csv"


def _env_list(name, default):
    raw = os.environ.get(name, "")
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or default


# ---------------------------------------------------------------------------
# load the project's own pipeline (same mechanism as run_backbone_comparison.py)
# ---------------------------------------------------------------------------
def load_pipeline():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("QUICK"):
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(src, f"nb_cell_{i}", "exec"), ns)
    return ns


print("Loading project pipeline from new_model_code.ipynb ...", flush=True)
NS = load_pipeline()
CFG = NS["CFG"]
tf = NS["tf"]
TRAIN_DF, VAL_DF, TEST_DF = NS["TRAIN_DF"], NS["VAL_DF"], NS["TEST_DF"]
CLASS_NAMES = list(CFG.class_names)
TEST_TRUE = np.array([CFG.class_to_index[c] for c in TEST_DF["class"]], dtype=int)

SEEDS = [int(s) for s in _env_list("MS_SEEDS", [str(s) for s in CFG.model_seeds])]
VARIANTS = _env_list("MS_VARIANTS", list(NS["ARCH_VARIANTS"].keys()))
BACKBONES = _env_list("MS_BACKBONES", [])
EPOCHS_FROZEN = int(os.environ.get("MS_EF", CFG.epochs_frozen))
EPOCHS_FINETUNE = int(os.environ.get("MS_ET", CFG.epochs_finetune))

print(f"  split: train={len(TRAIN_DF)} val={len(VAL_DF)} test={len(TEST_DF)}")
print(f"  seeds: {SEEDS}")
print(f"  variants: {VARIANTS}")
print(f"  backbones: {BACKBONES or '(none)'}")
print(f"  schedule: {EPOCHS_FROZEN} frozen + {EPOCHS_FINETUNE} finetune")
print(f"  device: {'GPU' if tf.config.list_physical_devices('GPU') else f'CPU ({_CORES} cores)'}")
print(f"  runs to perform: {(len(VARIANTS) + len(BACKBONES)) * len(SEEDS)}\n")


# ---------------------------------------------------------------------------
# metrics, recomputed from the saved prediction vector alone
# ---------------------------------------------------------------------------
def metrics_from_predictions(y_true, y_pred):
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)
    labels = list(range(len(CLASS_NAMES)))
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100,
        "precision_macro": float(pm) * 100,
        "recall_macro": float(rm) * 100,
        "f1_macro": float(fm) * 100,
        "errors": int((y_true != y_pred).sum()),
        "confusion_matrix": cm.tolist(),
    }


def pred_path(name, seed):
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    return PRED_DIR / f"{safe}_s{seed}.npz"


def load_rows():
    if ROWS_CSV.exists():
        return pd.read_csv(ROWS_CSV).to_dict("records")
    return []


def save_rows(rows):
    pd.DataFrame(rows).to_csv(ROWS_CSV, index=False)


# ---------------------------------------------------------------------------
# one run
# ---------------------------------------------------------------------------
def train_variant(variant, seed):
    model, hist, _ = NS["train_one"](
        variant, seed=seed, epochs_frozen=EPOCHS_FROZEN,
        epochs_finetune=EPOCHS_FINETUNE, verbose=0,
        tag=f"ms_{variant}_s{seed}")
    return model, hist.get("train_seconds", 0)


def train_backbone(arch, seed):
    model, secs = NS["train_baseline"](
        arch, seed=seed, epochs_frozen=EPOCHS_FROZEN,
        epochs_finetune=EPOCHS_FINETUNE, verbose=0)
    return model, secs


def run_one(name, seed, trainer):
    p = pred_path(name, seed)
    if p.exists():                      # resume rather than retrain
        d = np.load(p)
        return d["y_pred"], dict(metrics_from_predictions(TEST_TRUE, d["y_pred"]),
                                 train_seconds=float(d["train_seconds"]),
                                 params=int(d["params"])), True

    t0 = time.time()
    model, secs = trainer(name, seed)
    probs, y_true = NS["predict_split"](model, TEST_DF)
    assert np.array_equal(y_true, TEST_TRUE), "test label order changed"
    y_pred = probs.argmax(1)
    params = int(model.count_params())
    np.savez_compressed(p, y_pred=y_pred, probs=probs, y_true=TEST_TRUE,
                        train_seconds=secs or (time.time() - t0), params=params)
    del model
    tf.keras.backend.clear_session()
    m = metrics_from_predictions(TEST_TRUE, y_pred)
    m.update(train_seconds=float(secs or 0), params=params)
    return y_pred, m, False


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def aggregate(rows):
    df = pd.DataFrame(rows)
    df = df[df.status == "ok"]
    if df.empty:
        return df
    g = df.groupby("model")
    out = pd.DataFrame({
        "seeds": g.size(),
        "acc_mean": g.accuracy.mean(), "acc_sd": g.accuracy.std(ddof=1),
        "acc_min": g.accuracy.min(), "acc_max": g.accuracy.max(),
        "f1_mean": g.f1_macro.mean(), "f1_sd": g.f1_macro.std(ddof=1),
        "errors_mean": g.errors.mean(),
        "params": g.params.max(),
    }).reset_index().sort_values("acc_mean", ascending=False)
    return out


def paired_tests(rows):
    """Paired comparison across seeds. Pairing removes the seed effect, which is
    the dominant source of variation here, so it is far more sensitive than
    comparing two independent means."""
    from scipy import stats
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    if df.empty:
        return pd.DataFrame()
    wide = df.pivot_table(index="seed", columns="model", values="accuracy")
    wide = wide.dropna(axis=1, how="any")
    res = []
    for a, b in combinations(wide.columns, 2):
        d = wide[a] - wide[b]
        if len(d) < 2 or np.allclose(d, 0):
            t, p = np.nan, np.nan
        else:
            t, p = stats.ttest_rel(wide[a], wide[b])
        res.append({"model_a": a, "model_b": b, "n_seeds": len(d),
                    "mean_diff": d.mean(), "sd_diff": d.std(ddof=1),
                    "t": t, "p_value": p})
    return pd.DataFrame(res).sort_values("mean_diff", ascending=False)


def mcnemar_pooled(rows):
    """McNemar on predictions pooled over seeds, for every model pair. Uses the
    exact binomial test, which is the correct one when the discordant count is
    small -- and here it will be."""
    from scipy import stats
    df = pd.DataFrame([r for r in rows if r["status"] == "ok"])
    if df.empty:
        return pd.DataFrame()
    correct = {}
    for name, sub in df.groupby("model"):
        seeds = sorted(sub.seed)
        vecs = []
        for s in seeds:
            p = pred_path(name, s)
            if p.exists():
                vecs.append(np.load(p)["y_pred"] == TEST_TRUE)
        if vecs:
            correct[name] = np.concatenate(vecs)
    res = []
    for a, b in combinations(sorted(correct), 2):
        ca, cb = correct[a], correct[b]
        if len(ca) != len(cb):
            continue
        n01 = int((~ca & cb).sum())      # b right, a wrong
        n10 = int((ca & ~cb).sum())      # a right, b wrong
        n = n01 + n10
        p = stats.binomtest(n10, n, 0.5).pvalue if n else 1.0
        res.append({"model_a": a, "model_b": b, "a_only_correct": n10,
                    "b_only_correct": n01, "discordant": n, "p_value": p})
    return pd.DataFrame(res).sort_values("p_value")


# ---------------------------------------------------------------------------
def main():
    rows = load_rows()
    done = {(r["model"], r["seed"]) for r in rows if r["status"] == "ok"}

    jobs = ([(v, "variant") for v in VARIANTS]
            + [(b, "backbone") for b in BACKBONES])

    todo = [(n, k, s) for n, k in jobs for s in SEEDS if (n, s) not in done]
    n_todo = len(todo)
    durations, completed = [], 0

    for name, kind in jobs:
        trainer = ((lambda n, s: train_variant(n, s)) if kind == "variant"
                   else (lambda n, s: train_backbone(n, s)))
        for seed in SEEDS:
            if (name, seed) in done:
                print(f"[skip] {name:<22} seed {seed}  (already recorded)")
                continue
            eta = ""
            if durations:
                remaining = (n_todo - completed) * float(np.median(durations))
                eta = f"  eta {remaining/3600:.1f}h"
            print(f"[run ] {name:<22} seed {seed}  "
                  f"({completed + 1}/{n_todo}){eta} ...", end="", flush=True)
            t0 = time.time()
            try:
                _, m, cached = run_one(name, seed, trainer)
                rows.append({"model": name, "kind": kind, "seed": seed,
                             "status": "ok", **{k: v for k, v in m.items()
                                                if k != "confusion_matrix"}})
                took = time.time() - t0
                if not cached:
                    durations.append(took)
                print(f" acc {m['accuracy']:.2f}%  ({m['errors']} errors)"
                      f"  [{took/60:.0f} min]"
                      + ("  [from cache]" if cached else ""))
            except Exception as e:
                traceback.print_exc()
                rows.append({"model": name, "kind": kind, "seed": seed,
                             "status": "failed", "error": repr(e)})
                print(f" FAILED: {e}")
            completed += 1
            save_rows(rows)

    print("\n" + "=" * 78)
    agg = aggregate(rows)
    if agg.empty:
        print("no successful runs")
        return
    agg.to_csv(OUT / "summary_mean_sd.csv", index=False)
    print("Mean +- sd over seeds (accuracy, %):\n")
    for _, r in agg.iterrows():
        sd = "  n/a" if pd.isna(r.acc_sd) else f"{r.acc_sd:5.2f}"
        print(f"  {r['model']:<24} {r.acc_mean:6.2f} +- {sd}   "
              f"[{r.acc_min:.2f}, {r.acc_max:.2f}]   n={int(r.seeds)}")

    pt = paired_tests(rows)
    if not pt.empty:
        pt.to_csv(OUT / "paired_ttests.csv", index=False)
        print("\nPaired t-tests across seeds (top 10 by mean difference):\n")
        for _, r in pt.head(10).iterrows():
            print(f"  {r.model_a:<20} - {r.model_b:<20} "
                  f"{r.mean_diff:+6.2f} pts   p={r.p_value:.4f}")

    mc = mcnemar_pooled(rows)
    if not mc.empty:
        mc.to_csv(OUT / "mcnemar_pooled.csv", index=False)
        print("\nMcNemar on pooled predictions (10 most significant):\n")
        for _, r in mc.head(10).iterrows():
            print(f"  {r.model_a:<20} vs {r.model_b:<20} "
                  f"{r.a_only_correct:3d}/{r.b_only_correct:3d} discordant  "
                  f"p={r.p_value:.4f}")

    (OUT / "config.json").write_text(json.dumps({
        "seeds": SEEDS, "variants": VARIANTS, "backbones": BACKBONES,
        "epochs_frozen": EPOCHS_FROZEN, "epochs_finetune": EPOCHS_FINETUNE,
        "batch_size": CFG.batch_size, "split_csv": str(CFG.split_csv),
        "n_train": len(TRAIN_DF), "n_val": len(VAL_DF), "n_test": len(TEST_DF),
        "platform": platform.platform(), "tensorflow": tf.__version__,
    }, indent=2), encoding="utf-8")
    print(f"\nwritten -> {OUT}")


if __name__ == "__main__":
    main()
