#!/usr/bin/env python
"""Backbone comparison on NEU-DET, run end to end from the project pipeline.

    python run_backbone_comparison.py                 # all models
    BB_ONLY=MobileNetV2,ResNet18 python run_backbone_comparison.py
    BB_EF=8 BB_ET=8 python run_backbone_comparison.py # override the schedule

Every number this produces comes from executing the project's own code on the
project's own frozen split. Nothing is copied, estimated or hand-entered.

Design rules, all of them load-bearing:

  * The split is read from paper_results/splits/split_v1.csv and never
    regenerated. Every model sees the identical train/val/test images.
  * Checkpoint selection and early stopping use VALIDATION only. The test split
    is touched once, at the end, to produce predictions.
  * All metrics are derived from the SAVED PREDICTIONS, not from any in-memory
    value, so every reported number can be recomputed from disk.
  * Results are written after each model, so an interrupted run keeps what it
    already produced and resumes rather than starting over.
  * A model that fails is recorded as failed. It is never given a number.
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
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT / "new_model_code.ipynb"
RESULTS = ROOT / "results"
CKPT_DIR = RESULTS / "checkpoints"
CM_DIR = RESULTS / "confusion_matrices"
PRED_DIR = RESULTS / "predictions"
for d in (RESULTS, CKPT_DIR, CM_DIR, PRED_DIR):
    d.mkdir(parents=True, exist_ok=True)

EPOCHS_FROZEN = int(os.environ.get("BB_EF", 8))
EPOCHS_FINETUNE = int(os.environ.get("BB_ET", 8))
SEED = int(os.environ.get("BB_SEED", 42))
ONLY = [m.strip() for m in os.environ.get("BB_ONLY", "").split(",") if m.strip()]

# Models trained through the project's Keras pipeline.
KERAS_MODELS = [
    "ResNet18", "ResNet34", "ResNet50", "VGG16", "DenseNet121",
    "EfficientNetV2B0", "EfficientNetB0", "MobileNet", "MobileNetV3Large",
    "MobileNetV2", "ShuffleNetV2",
]
# Models with no Keras port. Official timm implementations + ImageNet weights.
TIMM_MODELS = {
    "RepViT-M1":    "repvit_m1.dist_in1k",
    "EdgeNeXt-S":   "edgenext_small.usi_in1k",
    "FastViT-SA12": "fastvit_sa12.apple_in1k",
}
PROPOSED = "Proposed (AMFF-CNN)"


# ---------------------------------------------------------------------------
# load the project's own pipeline
# ---------------------------------------------------------------------------
def load_pipeline():
    """Execute the notebook's code cells to obtain the exact project pipeline."""
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict = {}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.lstrip().startswith("QUICK"):      # skip the orchestration cell
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
DISPLAY = [NS["CLASS_DISPLAY"][c] for c in CLASS_NAMES]

print(f"  split: train={len(TRAIN_DF)}  val={len(VAL_DF)}  test={len(TEST_DF)}")
print(f"  input: {CFG.input_shape}   classes: {CLASS_NAMES}")
print(f"  schedule: {EPOCHS_FROZEN} frozen + {EPOCHS_FINETUNE} finetune, seed {SEED}")

# The exact test identity every model must be scored against.
TEST_STEMS = list(TEST_DF["stem"])
TEST_TRUE = np.array([CFG.class_to_index[c] for c in TEST_DF["class"]], dtype=int)
TRAIN_VAL_STEMS = set(TRAIN_DF["stem"]) | set(VAL_DF["stem"])

CONFIG_RECORD = {
    "seed": SEED,
    "epochs_frozen": EPOCHS_FROZEN,
    "epochs_finetune": EPOCHS_FINETUNE,
    "batch_size": CFG.batch_size,
    "optimizer": "Adam",
    "lr_frozen": CFG.lr_frozen,
    "lr_finetune": CFG.lr_finetune,
    "early_stopping": f"val_accuracy, patience {CFG.early_stop_patience}",
    "input_shape": list(CFG.input_shape),
    "classes": CLASS_NAMES,
    "split_csv": str(CFG.split_csv),
    "n_train": len(TRAIN_DF), "n_val": len(VAL_DF), "n_test": len(TEST_DF),
    "augmentation": "random H/V flip, +/-10% brightness (train split only)",
    "platform": platform.platform(),
    "processor": platform.processor(),
    "tensorflow": tf.__version__,
    "note": ("Every model uses the identical frozen split, preprocessing and "
             "schedule. ImageNet initialisation where weights exist; random "
             "init otherwise, recorded per model in the 'init' field."),
}


# ---------------------------------------------------------------------------
# metrics, derived only from saved predictions
# ---------------------------------------------------------------------------
def metrics_from_predictions(y_true, y_pred):
    """All reported metrics, computed from the prediction vector alone."""
    from sklearn.metrics import (confusion_matrix, precision_recall_fscore_support,
                                 accuracy_score)
    labels = list(range(len(CLASS_NAMES)))
    p, r, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    pm, rm, fm, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Class-wise accuracy = recall for that class = diagonal / row sum.
    per_class_acc = np.divide(np.diag(cm), np.maximum(cm.sum(1), 1),
                              dtype=float) * 100
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)) * 100,
        "precision_macro": float(pm) * 100,
        "recall_macro": float(rm) * 100,
        "f1_macro": float(fm) * 100,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            CLASS_NAMES[i]: {"precision": float(p[i]) * 100,
                             "recall": float(r[i]) * 100,
                             "f1": float(f1[i]) * 100,
                             "support": int(sup[i]),
                             "class_accuracy": float(per_class_acc[i])}
            for i in range(len(CLASS_NAMES))},
    }


def efficiency_of_keras(model, tag):
    try:
        flops = NS["count_flops"](model)
        macs = round(flops["macs_m"], 2)
        fl = round(flops["flops_m"], 2)
    except Exception:
        macs = fl = None
    try:
        lat = NS["measure_latency"](model, "/CPU:0", warmup=10, runs=50)
        latency, fps = round(lat["latency_ms_median"], 2), round(lat["fps"], 2)
    except Exception:
        latency = fps = None
    total = int(model.count_params())
    trainable = int(sum(np.prod(v.shape) for v in model.trainable_weights))
    return {"params_total": total, "params_trainable": trainable,
            "macs_m": macs, "flops_m": fl,
            "size_mib": round(total * 4 / 1024 ** 2, 2),
            "latency_ms": latency, "fps": fps}


# ---------------------------------------------------------------------------
# Keras track -- the project's own training function
# ---------------------------------------------------------------------------
def run_keras(arch):
    model, secs = NS["train_baseline"](arch, seed=SEED,
                                       epochs_frozen=EPOCHS_FROZEN,
                                       epochs_finetune=EPOCHS_FINETUNE,
                                       verbose=0)
    probs, y_true = NS["predict_split"](model, TEST_DF)
    assert np.array_equal(y_true, TEST_TRUE), "test label order changed"
    eff = efficiency_of_keras(model, arch)
    eff["init"] = ("ImageNet" if NS["BACKBONE_PRETRAINED"].get(arch, True)
                   else "scratch")
    eff["train_seconds"] = round(secs)
    ckpt = CFG.subdir("checkpoints") / f"baseline_{arch}_s{SEED}.weights.h5"
    if ckpt.exists():
        model.save_weights(str(CKPT_DIR / f"{arch}.weights.h5"))
    del model
    tf.keras.backend.clear_session()
    return probs, eff


def run_proposed():
    model, hist, _ = NS["train_one"]("full", seed=SEED,
                                     epochs_frozen=EPOCHS_FROZEN,
                                     epochs_finetune=EPOCHS_FINETUNE,
                                     verbose=0, tag=f"cmp_proposed_s{SEED}")
    probs, y_true = NS["predict_split"](model, TEST_DF)
    assert np.array_equal(y_true, TEST_TRUE), "test label order changed"
    eff = efficiency_of_keras(model, "proposed")
    eff["init"] = "ImageNet"
    eff["train_seconds"] = round(hist.get("train_seconds", 0))
    model.save_weights(str(CKPT_DIR / "Proposed.weights.h5"))
    del model
    tf.keras.backend.clear_session()
    return probs, eff


# ---------------------------------------------------------------------------
# timm / PyTorch track -- same split, same preprocessing contract
# ---------------------------------------------------------------------------
def run_timm(label, timm_name):
    import torch
    import torch.nn as nn
    import timm
    from timm.data import resolve_data_config
    import cv2

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    model = timm.create_model(timm_name, pretrained=True,
                              num_classes=len(CLASS_NAMES))
    cfg = resolve_data_config({}, model=model)
    mean = np.array(cfg.get("mean", (0.485, 0.456, 0.406)), np.float32)
    std = np.array(cfg.get("std", (0.229, 0.224, 0.225)), np.float32)

    S = CFG.img_size

    def load(df, train, rng=None):
        xs, ys = [], []
        for _, r in df.iterrows():
            img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (S, S)).astype(np.float32)
            if train and rng is not None:
                if rng.random() < 0.5:
                    img = img[:, ::-1]
                if rng.random() < 0.5:
                    img = img[::-1, :]
                img = np.clip(img * rng.uniform(0.9, 1.1), 0, 255)
            x = np.repeat(img[..., None], 3, -1) / 255.0     # grayscale -> 3ch
            x = (x - mean) / std                             # model's own norm
            xs.append(x.transpose(2, 0, 1))
            ys.append(CFG.class_to_index[r["class"]])
        return (torch.tensor(np.ascontiguousarray(xs), dtype=torch.float32),
                torch.tensor(ys, dtype=torch.long))

    rng = np.random.default_rng(SEED)
    xv, yv = load(VAL_DF, False)
    xt, yt = load(TEST_DF, False)
    assert yt.numpy().tolist() == TEST_TRUE.tolist(), "test label order changed"

    crit = nn.CrossEntropyLoss()
    bs = CFG.batch_size
    best_val, best_state, patience = -1.0, None, 0
    t0 = time.time()

    for phase, (n_ep, lr, freeze) in enumerate(
            [(EPOCHS_FROZEN, CFG.lr_frozen, True),
             (EPOCHS_FINETUNE, CFG.lr_finetune, False)]):
        # Phase 1 freezes the backbone and trains the classifier, mirroring the
        # Keras path; phase 2 unfreezes at the lower rate.
        for name, prm in model.named_parameters():
            head = any(k in name for k in ("head", "classifier", "fc"))
            prm.requires_grad = head if freeze else True
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                               lr=lr)
        for ep in range(n_ep):
            model.train()
            xtr, ytr = load(TRAIN_DF, True, rng)
            perm = torch.randperm(len(xtr))
            for i in range(0, len(perm), bs):
                idx = perm[i:i + bs]
                opt.zero_grad()
                loss = crit(model(xtr[idx]), ytr[idx])
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                pv = torch.cat([model(xv[i:i + bs]) for i in range(0, len(xv), bs)])
            acc = (pv.argmax(1) == yv).float().mean().item()
            print(f"      phase{phase + 1} ep{ep + 1}/{n_ep} val_acc={acc:.4f}",
                  flush=True)
            if acc > best_val:                       # selection on VALIDATION
                best_val = acc
                best_state = {k: v.detach().clone()
                              for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= CFG.early_stop_patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)
    secs = time.time() - t0

    model.eval()
    with torch.no_grad():
        logits = torch.cat([model(xt[i:i + bs]) for i in range(0, len(xt), bs)])
        probs = torch.softmax(logits, 1).numpy()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        from fvcore.nn import FlopCountAnalysis
        macs = FlopCountAnalysis(model, torch.zeros(1, 3, S, S)).total() / 1e6
    except Exception:
        macs = None
    with torch.no_grad():
        dummy = torch.zeros(1, 3, S, S)
        for _ in range(10):
            model(dummy)
        ts = []
        for _ in range(50):
            s = time.perf_counter(); model(dummy); ts.append((time.perf_counter() - s) * 1000)
    lat = float(np.median(ts))

    torch.save(model.state_dict(), CKPT_DIR / f"{label}.pt")
    return probs, {"params_total": int(total), "params_trainable": int(trainable),
                   "macs_m": round(macs, 2) if macs else None,
                   "flops_m": None,
                   "size_mib": round(total * 4 / 1024 ** 2, 2),
                   "latency_ms": round(lat, 2), "fps": round(1000.0 / lat, 2),
                   "init": "ImageNet", "train_seconds": round(secs),
                   "framework": "pytorch/timm"}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def record_path(label):
    return RESULTS / f"_model_{label.replace(' ', '_').replace('(', '').replace(')', '')}.json"


def evaluate(label, runner):
    """Train, predict, verify, and persist one model. Never fabricates."""
    out = record_path(label)
    if out.exists():
        print(f"  [skip] {label} already done")
        return json.loads(out.read_text())

    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    try:
        probs, eff = runner()
    except Exception as exc:
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)
        rec = {"model": label, "status": "failed",
               "error": f"{type(exc).__name__}: {exc}"}
        out.write_text(json.dumps(rec, indent=2))
        return rec

    probs = np.asarray(probs, dtype=float)
    y_pred = probs.argmax(1)
    np.savez_compressed(PRED_DIR / f"{label.replace(' ', '_')}.npz",
                        probs=probs, y_pred=y_pred, y_true=TEST_TRUE,
                        stems=np.array(TEST_STEMS))

    m = metrics_from_predictions(TEST_TRUE, y_pred)
    rec = {"model": label, "status": "ok", **eff, **m,
           "wall_seconds": round(time.time() - t0)}

    cm = np.array(m["confusion_matrix"])
    assert cm.sum() == len(TEST_TRUE), "confusion matrix does not sum to test size"
    assert abs(m["f1_macro"] -
               np.mean([v["f1"] for v in m["per_class"].values()])) < 1e-6
    pd.DataFrame(cm, index=DISPLAY, columns=DISPLAY).to_csv(
        CM_DIR / f"{label.replace(' ', '_')}.csv")

    out.write_text(json.dumps(rec, indent=2))
    print(f"  acc={m['accuracy']:.2f}  macroF1={m['f1_macro']:.2f}  "
          f"params={eff['params_total']:,}  init={eff.get('init')}  "
          f"({rec['wall_seconds']}s)", flush=True)
    return rec


def main():
    jobs = []
    for a in KERAS_MODELS:
        jobs.append((a, lambda a=a: run_keras(a)))
    for label, tname in TIMM_MODELS.items():
        jobs.append((label, lambda l=label, t=tname: run_timm(l, t)))
    jobs.append((PROPOSED, run_proposed))
    if ONLY:
        jobs = [j for j in jobs if j[0] in ONLY]

    print(f"\n{len(jobs)} model(s) queued\n" + "=" * 60)
    records = [evaluate(label, fn) for label, fn in jobs]

    ok = [r for r in records if r.get("status") == "ok"]
    failed = [r for r in records if r.get("status") != "ok"]
    if not ok:
        print("\nNo model completed. Nothing written.")
        return

    (RESULTS / "config.json").write_text(json.dumps(CONFIG_RECORD, indent=2))
    (RESULTS / "backbone_comparison.json").write_text(
        json.dumps(records, indent=2))

    main_rows, per_rows, eff_rows = [], [], []
    for r in ok:
        row = {"Model": r["model"], "Init": r.get("init", "?"),
               "Accuracy": round(r["accuracy"], 2),
               "MacroPrecision": round(r["precision_macro"], 2),
               "MacroRecall": round(r["recall_macro"], 2),
               "MacroF1": round(r["f1_macro"], 2)}
        for c, d in zip(CLASS_NAMES, DISPLAY):
            row[d] = round(r["per_class"][c]["class_accuracy"], 2)
        main_rows.append(row)

        pr = {"Model": r["model"]}
        for c, d in zip(CLASS_NAMES, DISPLAY):
            pr[f"{d} F1"] = round(r["per_class"][c]["f1"], 2)
        pr["Macro F1"] = round(r["f1_macro"], 2)
        per_rows.append(pr)

        eff_rows.append({"Model": r["model"], "Init": r.get("init", "?"),
                         "Params": r["params_total"],
                         "Trainable": r["params_trainable"],
                         "MACs (M)": r.get("macs_m"),
                         "Size (MiB)": r.get("size_mib"),
                         "Latency (ms)": r.get("latency_ms"),
                         "FPS": r.get("fps"),
                         "Framework": r.get("framework", "tensorflow")})

    dfm = pd.DataFrame(main_rows)
    dfp = pd.DataFrame(per_rows)
    dfe = pd.DataFrame(eff_rows)
    dfm.to_csv(RESULTS / "backbone_comparison.csv", index=False)
    dfp.to_csv(RESULTS / "per_class_results.csv", index=False)
    dfe.to_csv(RESULTS / "efficiency_results.csv", index=False)

    write_latex(dfm, dfp, dfe)

    print("\n" + "=" * 78)
    print("FINAL BACKBONE COMPARISON")
    print("=" * 78)
    hdr = f"{'Model':<22}{'Init':>9}{'Acc':>8}{'MacroF1':>9}{'Params':>12}{'MACs':>9}{'FPS':>8}"
    print(hdr); print("-" * 78)
    for r in ok:
        macs = r.get("macs_m"); fps = r.get("fps")
        print(f"{r['model']:<22}{r.get('init','?'):>9}{r['accuracy']:>8.2f}"
              f"{r['f1_macro']:>9.2f}{r['params_total']:>12,}"
              f"{(f'{macs:.0f}' if macs else 'N/A'):>9}"
              f"{(f'{fps:.1f}' if fps else 'N/A'):>8}")
    print("=" * 78)

    summarise(ok, failed)


def _bold_idx(series, higher_better=True):
    s = pd.to_numeric(series, errors="coerce")
    if s.dropna().empty:
        return -1
    return int(s.idxmax() if higher_better else s.idxmin())


def _fmt(v, dec=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}".replace(",", "{,}")
    return f"{float(v):.{dec}f}"


def write_latex(dfm, dfp, dfe):
    """Emit LaTeX with the best value in each column bolded programmatically."""
    def table(df, cols, higher, caption, label, first="Model"):
        best = {c: _bold_idx(df[c], higher.get(c, True)) for c in cols}
        out = ["\\begin{table*}[t]", "\\centering", f"\\caption{{{caption}}}",
               f"\\label{{{label}}}", "\\renewcommand{\\arraystretch}{1.15}",
               "\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule",
               " & ".join([f"\\textbf{{{first}}}"] +
                          [f"\\textbf{{{c}}}" for c in cols]) + r" \\",
               "\\midrule"]
        for i, row in df.iterrows():
            cells = [str(row[first]).replace("_", r"\_")]
            for c in cols:
                v = _fmt(row[c])
                cells.append(f"\\textbf{{{v}}}" if i == best[c] and v != "N/A" else v)
            out.append(" & ".join(cells) + r" \\")
        out += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", "", ""]
        return "\n".join(out)

    cls_cols = [c for c in dfm.columns if c not in
                ("Model", "Init", "Accuracy", "MacroPrecision",
                 "MacroRecall", "MacroF1")]
    tex = ["% Generated by run_backbone_comparison.py -- measured results only.",
           "% Bold marks the best value per column, determined programmatically.",
           "% Requires: \\usepackage{booktabs}", ""]
    tex.append(table(dfm, ["Accuracy"] + cls_cols,
                     {c: True for c in ["Accuracy"] + cls_cols},
                     "Class-wise classification accuracy (\\%) of different "
                     "backbone architectures on the NEU-DET test split. All "
                     "models share the same frozen split, preprocessing, "
                     "schedule and seed. Bold marks the best value per column.",
                     "tab:backbone_comparison"))
    tex.append(table(dfm, ["Accuracy", "MacroPrecision", "MacroRecall", "MacroF1"],
                     {"Accuracy": True, "MacroPrecision": True,
                      "MacroRecall": True, "MacroF1": True},
                     "Overall classification metrics (\\%) on the NEU-DET test "
                     "split. Bold marks the best value per column.",
                     "tab:backbone_overall"))
    tex.append(table(dfe, ["Params", "MACs (M)", "Latency (ms)", "FPS"],
                     {"Params": False, "MACs (M)": False,
                      "Latency (ms)": False, "FPS": True},
                     "Computational cost, measured on identical hardware at "
                     "batch size 1. Lower is better for parameters, MACs and "
                     "latency; higher is better for FPS.",
                     "tab:backbone_efficiency"))
    tex.append(table(dfp, [c for c in dfp.columns if c != "Model"],
                     {c: True for c in dfp.columns if c != "Model"},
                     "Per-class F1 score (\\%) on the NEU-DET test split.",
                     "tab:backbone_per_class"))
    (RESULTS / "backbone_tables.tex").write_text("\n".join(tex), encoding="utf-8")
    print(f"\nLaTeX -> {RESULTS / 'backbone_tables.tex'}")


def summarise(ok, failed):
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    best = lambda key, hi=True: (max if hi else min)(ok, key=lambda r: r[key])
    b = best("accuracy"); print(f"Highest accuracy   : {b['model']} ({b['accuracy']:.2f}%)")
    b = best("f1_macro"); print(f"Highest macro-F1   : {b['model']} ({b['f1_macro']:.2f}%)")
    b = best("params_total", False); print(f"Fewest parameters  : {b['model']} ({b['params_total']:,})")
    have = [r for r in ok if r.get("macs_m")]
    if have:
        b = min(have, key=lambda r: r["macs_m"]); print(f"Lowest MACs        : {b['model']} ({b['macs_m']:.0f} M)")
    have = [r for r in ok if r.get("fps")]
    if have:
        b = max(have, key=lambda r: r["fps"]); print(f"Highest CPU FPS    : {b['model']} ({b['fps']:.1f})")

    print("\nBest model per defect class (class-wise accuracy):")
    for c, d in zip(CLASS_NAMES, DISPLAY):
        b = max(ok, key=lambda r: r["per_class"][c]["class_accuracy"])
        print(f"  {d:<18} {b['model']:<24} {b['per_class'][c]['class_accuracy']:.2f}%")

    prop = next((r for r in ok if r["model"] == PROPOSED), None)
    if prop:
        lighter = [r for r in ok if r["params_total"] < prop["params_total"]]
        better = [r for r in lighter if r["accuracy"] > prop["accuracy"]]
        print(f"\nProposed: {prop['accuracy']:.2f}% accuracy at "
              f"{prop['params_total']:,} parameters.")
        print(f"  Models both smaller AND more accurate: "
              f"{[r['model'] for r in better] or 'none'}")

    if failed:
        print("\nFAILED (no numbers produced, nothing fabricated):")
        for r in failed:
            print(f"  {r['model']}: {r['error']}")

    print(f"\nFiles written under {RESULTS}/")
    for f in ["backbone_comparison.csv", "backbone_comparison.json",
              "per_class_results.csv", "efficiency_results.csv",
              "backbone_tables.tex", "config.json"]:
        print(f"  {f}")
    print(f"  confusion_matrices/  predictions/  checkpoints/")


if __name__ == "__main__":
    main()
