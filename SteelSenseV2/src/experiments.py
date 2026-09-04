"""High-level experiment drivers. The notebooks call these.

Each function owns one reviewer question end to end and returns a JSON-ready
dict, so a notebook cell is a single call plus a table render and nothing about
the protocol lives in the notebook.

    run_seeds            R1 Q9   multi-seed main result, mean +- SD
    run_order_ablation   R1 Q6   token order and the permutation-invariant control
    run_bin_sweep        R1 Q7   bin count and quantile vs equal-width
    compare_to_baselines R1 Q4/Q5 paired tests against the strongest baseline
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from config import ExperimentConfig
from engine import evaluate_model, fit
from metrics import (
    aggregate_seeds,
    bootstrap_ci,
    confusion_pair,
    mcnemar,
    paired_over_seeds,
    per_class_over_seeds,
)
from pipeline import Bundle, build_bundle


def _fit_eval(
    bundle: Bundle, cfg: ExperimentConfig, seed: int,
    arch: str = None, verbose=True, pooling: str = None,
):
    mcfg = copy.deepcopy(cfg.model)
    if arch:
        mcfg.arch = arch
    if pooling:
        mcfg.pooling = pooling
    model, res = fit(
        bundle.train.ids, bundle.train.y,
        bundle.val.ids, bundle.val.y,
        vocab_size=len(bundle.tokenizer),
        num_classes=len(bundle.classes),
        mcfg=mcfg, tcfg=cfg.train, classes=bundle.classes,
        seed=seed, verbose=verbose,
    )
    m, probs = evaluate_model(model, res, bundle.test.ids, bundle.test.y, bundle.classes)
    m["seed"] = seed
    m["arch"] = mcfg.arch
    m["pooling"] = mcfg.pooling
    return model, res, m, probs


def _fit_eval_numeric(
    train_X: np.ndarray, train_y: np.ndarray,
    val_X: np.ndarray, val_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    classes: Sequence[str], cfg: ExperimentConfig, seed: int,
    pooling: str = "all", verbose: bool = True,
):
    """Same protocol as `_fit_eval`, but for the `bilstm_numeric` control,
    which takes the standardized raw descriptor matrix instead of token ids
    (Reviewer 2 Q6, Reviewer 3 Q7/Q8 -- see `model.NumericBiLSTM`)."""
    mcfg = copy.deepcopy(cfg.model)
    mcfg.arch = "bilstm_numeric"
    mcfg.pooling = pooling
    model, res = fit(
        train_X.astype(np.float32), train_y,
        val_X.astype(np.float32), val_y,
        vocab_size=train_X.shape[1],  # reused as n_features; see model.NumericBiLSTM
        num_classes=len(classes),
        mcfg=mcfg, tcfg=cfg.train, classes=classes,
        seed=seed, verbose=verbose,
    )
    m, probs = evaluate_model(model, res, test_X.astype(np.float32), test_y, classes)
    m["seed"] = seed
    m["arch"] = "bilstm_numeric"
    m["pooling"] = pooling
    return model, res, m, probs


# ---------------------------------------------------------------------------
# R1 Q9 -- multi-seed main result
# ---------------------------------------------------------------------------


def run_seeds(
    bundle: Bundle,
    cfg: ExperimentConfig,
    seeds: Optional[Sequence[int]] = None,
    arch: Optional[str] = None,
    verbose: bool = True,
) -> Dict:
    seeds = list(seeds or cfg.seeds)
    runs, all_probs, keep = [], [], None
    for s in seeds:
        if verbose:
            print(f"[seed {s}] training {arch or cfg.model.arch} ...")
        model, res, m, probs = _fit_eval(bundle, cfg, s, arch=arch, verbose=verbose)
        runs.append(m)
        all_probs.append(probs)
        if keep is None:
            keep = (model, res)
        if verbose:
            print(f"[seed {s}] TEST acc {m['accuracy']:.4f}  "
                  f"macroF1 {m['macro_f1']:.4f}  balAcc {m['balanced_accuracy']:.4f}")

    agg = aggregate_seeds(runs)
    mean_probs = np.mean(all_probs, axis=0)
    pred = mean_probs.argmax(1)
    f1_pt, f1_lo, f1_hi = bootstrap_ci(bundle.test.y, pred, "macro_f1", n_boot=5000)
    acc_pt, acc_lo, acc_hi = bootstrap_ci(bundle.test.y, pred, "accuracy", n_boot=5000)

    out = {
        "dataset": cfg.dataset,
        "arch": arch or cfg.model.arch,
        "seeds": seeds,
        "n_train_rows": int(len(bundle.train.y)),
        "n_train_images": int(len(bundle.train.paths)),
        "n_val_images": int(len(bundle.val.y)),
        "n_test_images": int(len(bundle.test.y)),
        "classes": bundle.classes,
        "per_seed": runs,
        "aggregate": agg,
        "per_class_over_seeds": per_class_over_seeds(runs, bundle.classes),
        "seed_averaged_prediction": {
            "accuracy": acc_pt,
            "accuracy_ci95": [acc_lo, acc_hi],
            "macro_f1": f1_pt,
            "macro_f1_ci95": [f1_lo, f1_hi],
        },
        "out_of_range_report": bundle.oor_report,
        "vocab_size": len(bundle.tokenizer),
        "sequence_length": bundle.tokenizer.max_len,
        "n_features": len(bundle.feature_names),
        "selection_rule": "top-k checkpoints by VALIDATION macro-F1; test scored once",
    }
    # Confusion matrix over the seed-averaged prediction, which is what the
    # CI above is computed from.
    from metrics import evaluate

    out["seed_averaged_prediction"]["metrics"] = evaluate(
        bundle.test.y, pred, bundle.classes, mean_probs
    )
    out["_probs"] = mean_probs
    out["_model"] = keep
    return out


# ---------------------------------------------------------------------------
# R1 Q6 -- does token order matter?
# ---------------------------------------------------------------------------


def run_order_ablation(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    seeds: Optional[Sequence[int]] = None,
    verbose: bool = True,
) -> Dict:
    """Four conditions that between them settle whether recurrence is earned.

        A  bilstm   / canonical    the deployed model
        B  bilstm   / shuffled     one fixed random order, same for every sample
        C  bilstm   / per_sample   a different order per sample (order destroyed)
        D  deepsets / canonical    permutation-invariant control, no recurrence

    Reading: A == B says the specific order is arbitrary. A == C says the model
    is not using order at all. A == D says the recurrent encoder buys nothing
    over an order-free encoder of the same width, and the architecture claim in
    the paper must be withdrawn.
    """
    seeds = list(seeds or cfg.seeds)[:3]
    conditions = [
        ("A_bilstm_canonical", "bilstm", "canonical"),
        ("B_bilstm_fixed_shuffle", "bilstm", "shuffled"),
        ("C_bilstm_per_sample_shuffle", "bilstm", "per_sample"),
        ("D_deepsets_canonical", "deepsets", "canonical"),
    ]
    results: Dict[str, Dict] = {}
    scores: Dict[str, List[float]] = {}
    for key, arch, order in conditions:
        if verbose:
            print(f"\n=== {key} (arch={arch}, order={order}) ===")
        b = build_bundle(df, cfg, order=order, order_seed=1234, verbose=False)
        runs = []
        for s in seeds:
            _, _, m, _ = _fit_eval(b, cfg, s, arch=arch, verbose=False)
            runs.append(m)
            if verbose:
                print(f"  seed {s}: acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f}")
        results[key] = {
            "arch": arch,
            "order": order,
            "per_seed": runs,
            "aggregate": aggregate_seeds(runs),
            "n_params": runs[0]["n_params"],
        }
        scores[key] = [r["macro_f1"] for r in runs]

    base = "A_bilstm_canonical"
    results["paired_tests_vs_A"] = {
        k: paired_over_seeds(scores[base], scores[k]) for k in scores if k != base
    }
    a = np.mean(scores[base])
    results["verdict"] = {
        "canonical_vs_per_sample_shuffle_macro_f1_gap": round(
            a - float(np.mean(scores["C_bilstm_per_sample_shuffle"])), 4
        ),
        "canonical_vs_deepsets_macro_f1_gap": round(
            a - float(np.mean(scores["D_deepsets_canonical"])), 4
        ),
        "how_to_read": (
            "A gap near zero against C means the sequence order carries no "
            "information. A gap near zero against D means the recurrent "
            "encoder is not needed and an order-free encoder should be used."
        ),
    }
    return results


# ---------------------------------------------------------------------------
# R1 Q7 -- how many bins, and which edge rule?
# ---------------------------------------------------------------------------


def run_bin_sweep(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    bin_counts: Sequence[int] = (2, 3, 5, 7, 10, 15),
    strategies: Sequence[str] = ("quantile", "uniform"),
    seeds: Optional[Sequence[int]] = None,
    epochs: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    batch_size: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict:
    """Bin-count / edge-rule sweep (R1 Q7).

    `epochs` / `early_stop_patience` / `batch_size`, if given, override
    `cfg.train` for the SWEEP ONLY (a deep copy -- the `cfg` passed in is
    never mutated). This is a relative comparison across bin configurations,
    not the headline result, so a lighter training budget than `run_seeds`
    uses is legitimate: checkpoint selection still uses validation macro-F1
    only, and the split/augmentation/bin-edge-fitting order is untouched.

    `checkpoint_path`, if given, makes the sweep resumable: every finished
    (strategy, n_bins) row is flushed to disk immediately, and re-calling
    with the same bin_counts/strategies/seeds skips whatever already
    succeeded instead of re-training it. A combo that raises is recorded as
    a failure row and does not abort the rest of the sweep -- both matter
    for a grid this size on CPU, where one bad combo or one interrupted
    kernel used to mean losing everything already computed.
    """
    seeds = list(seeds or cfg.seeds)[:3]
    sweep_cfg = copy.deepcopy(cfg)
    if epochs is not None:
        sweep_cfg.train.epochs = epochs
    if early_stop_patience is not None:
        sweep_cfg.train.early_stop_patience = early_stop_patience
    if batch_size is not None:
        sweep_cfg.train.batch_size = batch_size

    rows: List[Dict] = []
    done: Dict[Tuple[str, int], Dict] = {}
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if saved.get("seeds") == seeds:
                saved_rows = saved.get("rows", [])
                for r in saved_rows:
                    if not r.get("failed"):  # retry failures, never skip them permanently
                        done[(r["strategy"], r["n_bins"])] = r
                rows = list(done.values())
                n_failed_prev = sum(1 for r in saved_rows if r.get("failed"))
                if verbose and (rows or n_failed_prev):
                    print(f"[bin_sweep] resuming from checkpoint: "
                          f"{len(rows)} combo(s) already done"
                          + (f", {n_failed_prev} previously-failed combo(s) will be retried"
                             if n_failed_prev else ""))
            elif verbose:
                print("[bin_sweep] checkpoint seeds differ from this call -- "
                      "ignoring it and starting fresh")

    def _flush():
        if checkpoint_path is None:
            return
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = checkpoint_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"seeds": seeds, "rows": rows}, indent=2),
                        encoding="utf-8")
        tmp.replace(checkpoint_path)

    combos = [(strat, k) for strat in strategies for k in bin_counts]
    todo = [c for c in combos if c not in done]
    t_start = time.perf_counter()
    for i, (strat, k) in enumerate(todo):
        t0 = time.perf_counter()
        try:
            b = build_bundle(df, sweep_cfg, n_bins=k, strategy=strat, verbose=False)
            runs = []
            for s in seeds:
                _, _, m, _ = _fit_eval(b, sweep_cfg, s, verbose=False)
                runs.append(m)
            agg = aggregate_seeds(runs)
            row = {
                "strategy": strat,
                "n_bins": k,
                "vocab_size": len(b.tokenizer),
                "test_accuracy_mean": agg["accuracy"]["mean"],
                "test_accuracy_sd": agg["accuracy"]["sd"],
                "macro_f1_mean": agg["macro_f1"]["mean"],
                "macro_f1_sd": agg["macro_f1"]["sd"],
                "balanced_accuracy_mean": agg["balanced_accuracy"]["mean"],
                "test_oor_rate_pct": b.oor_report["test"]["overall_rate_pct"],
                "n_seeds": len(seeds),
            }
            msg = (f"  {strat:9s} bins={k:3d} vocab={row['vocab_size']:5d} "
                   f"macroF1 {row['macro_f1_mean']:.4f} "
                   f"+- {row['macro_f1_sd']:.4f}  "
                   f"OOR {row['test_oor_rate_pct']}%")
        except Exception as e:  # one bad combo must not cost the rest of the sweep
            row = {"strategy": strat, "n_bins": k, "n_seeds": len(seeds),
                   "failed": True, "error": str(e)}
            msg = f"  {strat:9s} bins={k:3d}  FAILED: {e}"

        rows.append(row)
        _flush()

        elapsed = time.perf_counter() - t_start
        done_now = i + 1
        eta_min = (elapsed / done_now) * (len(todo) - done_now) / 60.0
        if verbose:
            print(f"{msg}  [{time.perf_counter() - t0:.0f}s, "
                  f"{done_now}/{len(todo)} this run, ETA {eta_min:.1f} min]")

    ok_rows = [r for r in rows if not r.get("failed")]
    best = max(ok_rows, key=lambda r: r["macro_f1_mean"]) if ok_rows else None
    return {
        "sweep": rows,
        "best": best,
        "note": (
            "The original submission used three levels (Low/Medium/High) with no "
            "justification. This sweep is the justification, or the reason to "
            "change it."
        ),
    }


# ---------------------------------------------------------------------------
# R1 Q1, R2 Q6, R3 Q7/Q8 -- which component earns its parameters?
# ---------------------------------------------------------------------------


def run_component_ablation(
    bundle: Bundle,
    cfg: ExperimentConfig,
    seeds: Optional[Sequence[int]] = None,
    epochs: Optional[int] = None,
    early_stop_patience: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    """Six conditions, everything else held fixed at the main run's split,
    seeds, epochs and checkpoint-selection rule (validation macro-F1; test
    scored once per condition):

        full_5snapshot_ensemble        tokens + BiLSTM + attention+max+mean
                                        pooling + 5-snapshot ensemble (the
                                        deployed configuration; same protocol
                                        as `run_seeds`)
        no_ensemble_best_checkpoint    the SAME fit as above, rescored with
                                        only its single best-validation
                                        checkpoint -- no retraining, this is
                                        exactly what "without snapshot
                                        ensembling" means
        pool_attention_only            tokens + BiLSTM, attention pooling only
        pool_max_only                  tokens + BiLSTM, max pooling only
        pool_mean_only                 tokens + BiLSTM, mean pooling only
        no_discretization_numeric      RAW standardized descriptors (fit on
                                        TRAIN only) + the SAME encoder and
                                        pooled head, via `model.NumericBiLSTM`
                                        -- no tokenizer, no discretizer,
                                        anywhere in this condition's path

    This is the direct answer to "which component is responsible for the
    reported performance" (Reviewer 1 Q1, Reviewer 2 Q6, Reviewer 3 Q7/Q8):
    the tabular ML baselines in `baselines_tabular.py` already show what a
    completely different model family does on the raw vector, but they
    cannot isolate discretization itself, because trees and a generic MLP
    are not the deployed encoder. `no_discretization_numeric` is.
    """
    seeds = list(seeds or cfg.seeds)[:3]
    abl_cfg = copy.deepcopy(cfg)
    if epochs is not None:
        abl_cfg.train.epochs = epochs
    if early_stop_patience is not None:
        abl_cfg.train.early_stop_patience = early_stop_patience

    # Standardized raw descriptors, fit on TRAIN ONLY -- same rule as the
    # discretizer and the tokenizer schema everywhere else in this repo.
    mu = bundle.train.X.mean(0, keepdims=True)
    sd = bundle.train.X.std(0, keepdims=True) + 1e-6
    Xtr = (bundle.train.X - mu) / sd
    Xva = (bundle.val.X - mu) / sd
    Xte = (bundle.test.X - mu) / sd

    conditions: Dict[str, List[Dict]] = {}

    full_runs: List[Dict] = []
    no_ens_runs: List[Dict] = []
    for s in seeds:
        model, res, m, _ = _fit_eval(bundle, abl_cfg, s, verbose=False)
        full_runs.append(m)
        m_single, _ = evaluate_model(
            model, res, bundle.test.ids, bundle.test.y, bundle.classes, use_ensemble=False
        )
        m_single["seed"] = s
        m_single["arch"] = m["arch"]
        m_single["pooling"] = m["pooling"]
        no_ens_runs.append(m_single)
    conditions["full_5snapshot_ensemble"] = full_runs
    conditions["no_ensemble_best_checkpoint"] = no_ens_runs
    if verbose:
        for name in ("full_5snapshot_ensemble", "no_ensemble_best_checkpoint"):
            f1 = np.mean([r["macro_f1"] for r in conditions[name]])
            print(f"  {name:30s} macroF1 {f1:.4f}")

    for pooling, key in (
        ("attention", "pool_attention_only"),
        ("max", "pool_max_only"),
        ("mean", "pool_mean_only"),
    ):
        runs = [
            _fit_eval(bundle, abl_cfg, s, pooling=pooling, verbose=False)[2] for s in seeds
        ]
        conditions[key] = runs
        if verbose:
            f1 = np.mean([r["macro_f1"] for r in runs])
            print(f"  {key:30s} macroF1 {f1:.4f}")

    numeric_runs = [
        _fit_eval_numeric(
            Xtr, bundle.train.y, Xva, bundle.val.y, Xte, bundle.test.y,
            bundle.classes, abl_cfg, s, pooling="all", verbose=False,
        )[2]
        for s in seeds
    ]
    conditions["no_discretization_numeric"] = numeric_runs
    if verbose:
        f1 = np.mean([r["macro_f1"] for r in numeric_runs])
        print(f"  {'no_discretization_numeric':30s} macroF1 {f1:.4f}")

    summary = []
    for name, runs in conditions.items():
        agg = aggregate_seeds(runs)
        summary.append(
            {
                "condition": name,
                "macro_f1_mean": agg["macro_f1"]["mean"],
                "macro_f1_sd": agg["macro_f1"]["sd"],
                "accuracy_mean": agg["accuracy"]["mean"],
                "accuracy_sd": agg["accuracy"]["sd"],
                "n_params": runs[0]["n_params"],
                "n_seeds": len(runs),
            }
        )

    full_scores = [r["macro_f1"] for r in conditions["full_5snapshot_ensemble"]]
    paired = {
        name: paired_over_seeds(full_scores, [r["macro_f1"] for r in runs])
        for name, runs in conditions.items()
        if name != "full_5snapshot_ensemble"
    }
    return {
        "conditions": conditions,
        "summary": summary,
        "paired_tests_vs_full": paired,
        "note": (
            "full_5snapshot_ensemble is the deployed configuration; every other row "
            "changes exactly one component (ensembling, one pooling view, or "
            "discretization) and holds the split, seeds, epochs and "
            "checkpoint-selection rule fixed."
        ),
    }


# ---------------------------------------------------------------------------
# R1 Q4/Q5 -- paired comparison against the strongest baseline
# ---------------------------------------------------------------------------


def compare_to_baselines(
    steelsense: Dict,
    baseline_results: Dict[str, Dict],
    baseline_probs: Dict[str, np.ndarray],
    y_test: np.ndarray,
    classes: Sequence[str],
) -> Dict:
    ours_pred = steelsense["_probs"].argmax(1)
    scored = {
        k: v for k, v in baseline_results.items()
        if "macro_f1" in v and "failed" not in v
    }
    if not scored:
        return {"note": "no baseline completed"}
    best_name = max(scored, key=lambda k: scored[k]["macro_f1"])
    best_pred = baseline_probs[best_name].argmax(1)

    table = [
        {
            "model": "SteelSense-BiLSTM",
            "representation": "binned descriptors as text prompt",
            "accuracy": steelsense["aggregate"]["accuracy"]["mean"],
            "accuracy_sd": steelsense["aggregate"]["accuracy"]["sd"],
            "macro_f1": steelsense["aggregate"]["macro_f1"]["mean"],
            "macro_f1_sd": steelsense["aggregate"]["macro_f1"]["sd"],
            "balanced_accuracy": steelsense["aggregate"]["balanced_accuracy"]["mean"],
            "n_params": steelsense["per_seed"][0]["n_params"],
        }
    ]
    for k, v in scored.items():
        table.append(
            {
                "model": k,
                "representation": v.get("representation", ""),
                "accuracy": v["accuracy"],
                "accuracy_sd": None,
                "macro_f1": v["macro_f1"],
                "macro_f1_sd": None,
                "balanced_accuracy": v["balanced_accuracy"],
                "n_params": v.get("n_params"),
            }
        )

    return {
        "strongest_baseline": best_name,
        "strongest_baseline_macro_f1": scored[best_name]["macro_f1"],
        "steelsense_macro_f1_mean": steelsense["aggregate"]["macro_f1"]["mean"],
        "mcnemar_vs_strongest": mcnemar(y_test, ours_pred, best_pred),
        "comparison_table": table,
        "note": (
            "McNemar is computed on the seed-averaged SteelSense prediction "
            "against the strongest baseline on the identical test images."
        ),
    }


def save_json(path: Path, payload: Dict) -> None:
    """Strips the private numpy/model handles before writing."""
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean(payload), f, indent=2)
