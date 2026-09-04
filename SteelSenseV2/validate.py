"""Fast end-to-end smoke test of every module the notebooks call.

Runs the whole protocol on a small subsample with tiny schedules, so a broken
code path surfaces in a couple of minutes instead of after a full run.

    python validate.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
import pandas as pd
import torch

from config import DATASETS, ExperimentConfig, RESULTS_DIR, SplitConfig

OK, FAIL = [], []


def check(name):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n{'='*70}\n[CHECK] {name}\n{'='*70}", flush=True)
            try:
                r = fn(*a, **k)
                OK.append(name)
                print(f"[PASS] {name}")
                return r
            except Exception as e:
                FAIL.append((name, repr(e)))
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc()
                return None
        return wrapped
    return deco


def subsample(df: pd.DataFrame, per_class_per_split: int = 12) -> pd.DataFrame:
    return (
        df.groupby(["class", "split"], group_keys=False)
        .apply(lambda g: g.head(per_class_per_split))
        .reset_index(drop=True)
    )


def main() -> int:
    import splits, pipeline, experiments, metrics

    cfg = ExperimentConfig(dataset="neu_det", tag="_validate")
    cfg.train.epochs = 3
    cfg.train.aug_variants = 1
    cfg.train.ensemble_size = 2
    cfg.seeds = [42, 7]

    df_full = splits.load_split("neu_det", RESULTS_DIR / "splits")
    df = subsample(df_full, 12)
    print("subsample:", df.split.value_counts().to_dict())

    bundle = check("pipeline.build_bundle")(pipeline.build_bundle)(df, cfg, verbose=False)
    if bundle is None:
        return 1
    print(f"  features={len(bundle.feature_names)} vocab={len(bundle.tokenizer)} "
          f"train_rows={len(bundle.train.y)} test={len(bundle.test.y)}")
    print("  OOR test rate:", bundle.oor_report["test"]["overall_rate_pct"], "%")

    main_res = check("experiments.run_seeds")(experiments.run_seeds)(
        bundle, cfg, verbose=False
    )
    if main_res:
        a = main_res["aggregate"]
        print(f"  acc {a['accuracy']['mean']:.4f}+-{a['accuracy']['sd']:.4f}  "
              f"macroF1 {a['macro_f1']['mean']:.4f}")

    cm = np.array(main_res["seed_averaged_prediction"]["metrics"]["confusion_matrix"])
    check("metrics.confusion_pair")(metrics.confusion_pair)(
        cm, bundle.classes, "crazing", "rolled-in_scale"
    )

    # --- tabular baselines -------------------------------------------------
    import baselines_tabular as BT

    tab = check("baselines_tabular.run_all")(BT.run_all)(
        bundle.train.X, bundle.train.y, bundle.val.X, bundle.val.y,
        bundle.test.X, bundle.test.y, bundle.classes, seed=42, verbose=True,
    )
    tab_res, tab_probs = tab if tab else ({}, {})

    # --- CNN baseline (1 epoch, 1 model) -----------------------------------
    import baselines_cnn as BC

    cls_idx = {c: i for i, c in enumerate(bundle.classes)}
    to_y = lambda d: np.array([cls_idx[c] for c in d["class"]], dtype=np.int64)
    tr, va, te = df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]
    cnn = check("baselines_cnn.train_backbone")(BC.train_backbone)(
        "MobileNetV3-Small", tr["path"].tolist(), to_y(tr), va["path"].tolist(), to_y(va),
        te["path"].tolist(), to_y(te), bundle.classes, seed=42, epochs=1,
        img_size=128, verbose=True,
    )
    cnn_res = {"MobileNetV3-Small": cnn[0]} if cnn else {}
    cnn_probs = {"MobileNetV3-Small": cnn[1]} if cnn else {}

    check("experiments.compare_to_baselines")(experiments.compare_to_baselines)(
        main_res, {**tab_res, **cnn_res}, {**tab_probs, **cnn_probs},
        bundle.test.y, bundle.classes,
    )

    # --- complexity --------------------------------------------------------
    from complexity import profile_model
    from model import build_model

    ss_model = build_model("bilstm", len(bundle.tokenizer), len(bundle.classes), cfg.model)
    cx = check("complexity.profile_model")(profile_model)(
        "SteelSense-BiLSTM", ss_model, torch.from_numpy(bundle.test.ids[:1]),
        is_token_model=True, repeats=20,
    )
    if cx:
        print(f"  params={cx['n_params']:,} size={cx['size_mb']}MB "
              f"MMACs={cx['mmacs']} lat={cx['latency_bs1']['median_ms']}ms")

    # --- ablations ---------------------------------------------------------
    check("experiments.run_order_ablation")(experiments.run_order_ablation)(
        df, cfg, seeds=[42, 7], verbose=True
    )
    check("experiments.run_bin_sweep")(experiments.run_bin_sweep)(
        df, cfg, bin_counts=(3, 5), strategies=("quantile", "uniform"),
        seeds=[42], verbose=True,
    )
    check("experiments.run_component_ablation")(experiments.run_component_ablation)(
        bundle, cfg, seeds=[42, 7], epochs=2, early_stop_patience=2, verbose=True,
    )

    # --- interpretability --------------------------------------------------
    import interpret

    ss, res = main_res["_model"]
    attn = check("interpret.attention_by_class")(interpret.attention_by_class)(
        ss, res.snapshots, bundle.test.ids, bundle.test.y,
        bundle.feature_names, bundle.classes,
    )
    perm = check("interpret.permutation_importance")(interpret.permutation_importance)(
        ss, res.snapshots, bundle.test.ids[:60], bundle.test.y[:60],
        bundle.feature_names[:8], bundle.classes, n_repeats=1, verbose=False,
    )
    if attn and perm:
        check("interpret.agreement")(interpret.agreement)(attn, perm)

    # --- localization ------------------------------------------------------
    import localize

    te2 = df[df.split == "test"].reset_index(drop=True)
    probs = main_res["_probs"]
    loc = check("localize.run_localization")(localize.run_localization)(
        te2["path"].tolist(),
        [bundle.classes[i] for i in probs.argmax(1)],
        probs.max(1).tolist(),
        DATASETS["neu_det"].annotation_dir, cfg.spec().img_size,
        bundle.classes, verbose=False,
    )
    if loc:
        print(f"  DET AP50={loc['DET_real_detector']['AP50']} "
              f"AP75={loc['DET_real_detector']['AP75']} "
              f"mAP={loc['DET_real_detector']['mAP@[.5:.95]']}")
        print(f"  ORACLE AP50={loc['ORACLE_classification_ceiling']['AP50']} "
              f"(threshold-invariant by construction)")
        print(f"  mean IoU={loc['DET_iou_histogram']['mean_iou']} "
              f"fallback={loc['proposal_fallback_rate_pct']}%")
        qual = loc.get("DET_qualitative_examples")
        if qual:
            print(f"  qualitative examples: {len(qual['good'])} good, {len(qual['bad'])} bad")

    # --- profiling ---------------------------------------------------------
    import profile_stages as PS

    prof = check("profile_stages.profile_pipeline")(PS.profile_pipeline)(
        bundle.test.paths[0], cfg.features, bundle.discretizer, bundle.tokenizer,
        ss, res.snapshots, bundle.feature_names, repeats=20, include_localization=True,
    )
    if prof:
        print("  totals:", prof["totals"])
    check("profile_stages.speedup_report")(PS.speedup_report)(
        bundle.test.paths[:16], cfg.features, thread_counts=(1, 2, 4)
    )

    # --- serialization -----------------------------------------------------
    check("experiments.save_json")(experiments.save_json)(
        cfg.out_dir() / "validate.json",
        {"main": main_res, "baselines": tab_res, "localization": loc, "profile": prof},
    )

    print(f"\n{'='*70}\nPASSED {len(OK)}   FAILED {len(FAIL)}\n{'='*70}")
    for n, e in FAIL:
        print(f"  FAIL {n}: {e}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
