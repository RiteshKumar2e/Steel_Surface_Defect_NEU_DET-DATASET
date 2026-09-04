"""Generates the three deliverable notebooks.

Keeping the notebooks generated rather than hand-edited means the protocol
lives in src/ and the notebooks stay a thin, readable driver -- which is also
what makes them reproducible for a reviewer.

    python make_notebooks.py
"""

from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent


def md(t: str):
    return nbf.v4.new_markdown_cell(t.strip("\n"))


def code(t: str):
    return nbf.v4.new_code_cell(t.strip("\n"))


HEADER = """
import sys, json, warnings
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")

from config import DATASETS, ExperimentConfig, SplitConfig, RESULTS_DIR
import splits, pipeline, experiments, metrics
print("modules loaded")
"""


# ===========================================================================
# 00 -- dataset integrity audit
# ===========================================================================


def audit_notebook() -> nbf.NotebookNode:
    c = []
    c.append(md(r"""
# 00 — Dataset Integrity Audit

**Run this notebook first.** Everything in notebooks 01 and 02 depends on the split
manifests it writes, and its main finding changes how the paper's second dataset
must be described.

### What this notebook answers

| Reviewer | Question | Where |
|---|---|---|
| R1 Q1 | Was a test set held out that was never used for selection or tuning? | §3 |
| R1 Q2 | Was the split at image level before augmentation, and was there a near-duplicate check? | §2, §3 |
| R1/R2 | The leakage concern behind the 99.89% / 99.45% / crazing-F1-1.000 numbers | §2 |

### The finding, stated up front

The two "independent" datasets are not independent. Four SteelDefectX classes are
**pixel copies of NEU-DET images re-encoded at 256×256**. §2 measures it.
"""))

    c.append(code(HEADER))

    c.append(md(r"""
## 1. What is on disk · Major Issue 4 (decision letter — exact class distribution and counts)

Both datasets are enumerated from the directory layout. Nothing is read into a model yet.
"""))
    c.append(code(r"""
rows = []
for name in ["neu_det", "steeldefectx", "steeldefectx_paper"]:
    spec = DATASETS[name]
    items = splits.enumerate_images(spec)
    counts = pd.Series([i.label for i in items]).value_counts()
    rows.append({
        "dataset": name,
        "classes": len(spec.classes),
        "images": len(items),
        "img_size": spec.img_size,
        "smallest_class": f"{counts.idxmin()} ({counts.min()})",
        "largest_class": f"{counts.idxmax()} ({counts.max()})",
        "imbalance_ratio": round(counts.max() / counts.min(), 1),
        "boxes": "PASCAL-VOC XML" if spec.annotation_dir else "none",
    })
inventory = pd.DataFrame(rows)
display(inventory)
"""))

    c.append(md(r"""
### Reviewer 1, Q10 · Major Issue 4 — why accuracy alone is not reportable

The imbalance ratio above is the reason every result in notebooks 01 and 02 leads with
**macro-F1 and balanced accuracy**, with per-class precision/recall alongside. On the
six-class SteelDefectX subset Inclusion is ~34% of the data; on the full 24-class version
the largest class is more than 25× the smallest.
"""))

    c.append(md(r"""
## 2. Cross-dataset duplicate audit — the finding · Major Issue 2 (decision letter — near-duplicate / same-source check)

Two-stage matching, deliberately conservative:

1. **dHash**, 64-bit, Hamming distance ≤ 6 — a cheap recall filter.
2. **Pearson correlation** of the 64×64 z-scored grayscale image ≥ 0.97 — rejects the
   false positives a 64-bit hash produces on low-texture surfaces.

A match at correlation ≥ 0.97 across a resolution change is the same photograph, not a
similar defect.
"""))
    c.append(code(r"""
from duplicates import build_signatures, cross_dataset_duplicates

neu_items = splits.enumerate_images(DATASETS["neu_det"])
sdx_items = splits.enumerate_images(DATASETS["steeldefectx"])

neu_sig = build_signatures([i.path for i in neu_items], 8)
sdx_sig = build_signatures([i.path for i in sdx_items], 8)
print(f"NEU-DET reference: {len(neu_sig)} images | SteelDefectX query: {len(sdx_sig)} images")

dups = cross_dataset_duplicates(sdx_sig, neu_sig, hamming_max=6, corr_min=0.97)
label_of = {i.path: i.label for i in sdx_items}

n_by_class = pd.Series([i.label for i in sdx_items]).value_counts()
d_by_class = pd.Series([label_of[p] for p in dups]).value_counts()
audit = pd.DataFrame({"images": n_by_class, "duplicated_from_NEU": d_by_class}).fillna(0).astype(int)
audit["pct"] = (100 * audit["duplicated_from_NEU"] / audit["images"]).round(1)
audit = audit.sort_values("pct", ascending=False)
display(audit)

print(f"\nTOTAL: {len(dups)} of {len(sdx_items)} SteelDefectX images "
      f"({100*len(dups)/len(sdx_items):.1f}%) are duplicates of NEU-DET images.")
"""))

    c.append(code(r"""
# The six classes the original submission actually used
paper_six = DATASETS["steeldefectx_paper"].classes
sub = audit.reindex(paper_six).fillna(0).astype({"images": int, "duplicated_from_NEU": int})
print("The 6-class SteelDefectX subset used in the submitted paper:\n")
display(sub)
print(f"Contaminated: {sub['duplicated_from_NEU'].sum()} of {sub['images'].sum()} images "
      f"({100*sub['duplicated_from_NEU'].sum()/sub['images'].sum():.1f}%)")
"""))

    c.append(md(r"""
### Visual confirmation

Matched pairs, side by side, with the correlation printed. If these are different
photographs the audit is wrong; they are not.
"""))
    c.append(code(r"""
import cv2

examples = [(q, v) for q, v in dups.items() if v["corr"] >= 0.99][:4]
fig, axes = plt.subplots(2, len(examples), figsize=(3.2 * len(examples), 6.4))
for k, (q, v) in enumerate(examples):
    a = cv2.imread(q, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(v["ref"], cv2.IMREAD_GRAYSCALE)
    axes[0, k].imshow(a, cmap="gray"); axes[0, k].set_title(f"SteelDefectX\n{Path(q).name}", fontsize=9)
    axes[1, k].imshow(b, cmap="gray"); axes[1, k].set_title(f"NEU-DET\n{Path(v['ref']).name}\ncorr={v['corr']}", fontsize=9)
    for r in (0, 1):
        axes[r, k].axis("off")
plt.suptitle("Cross-dataset near-duplicates (correlation >= 0.99)", y=0.98)
plt.tight_layout(); plt.show()
"""))

    c.append(md(r"""
### Consequence for the manuscript

1. A model trained on NEU-DET and evaluated on the six-class SteelDefectX subset is
   evaluated largely **on its own training images**. The 99.45% figure and the
   crazing F1 of 1.000 that Reviewer 2 questions are consistent with this and should
   not be reported.
2. The two datasets cannot be described as independent corpora, and no
   cross-dataset generalization claim survives.
3. **The fix used here:** `DATASETS["steeldefectx"]` is the full 24-class version with
   every NEU-DET duplicate removed at image level. That is the dataset notebook 02
   reports. The contaminated subset is kept as `steeldefectx_paper` only so the size
   of the effect can be quantified.

The cell below shows the sharpest consequence: **once duplicates are removed, four of the
six classes the submission evaluated on no longer exist in SteelDefectX.** The six-class
task as defined in the paper cannot be reconstituted from clean data.
"""))
    c.append(code(r"""
survivors = audit.copy()
survivors["remaining"] = survivors["images"] - survivors["duplicated_from_NEU"]
paper_rows = survivors.reindex(paper_six)
paper_rows["verdict"] = np.where(
    paper_rows["remaining"] < 20, "CLASS LOST (below usable size)", "usable"
)
print("The six classes the submission reported on, after NEU-DET duplicates are removed:\n")
display(paper_rows[["images", "duplicated_from_NEU", "remaining", "verdict"]])

kept = survivors[survivors["remaining"] >= 20]
print(f"\nWhole dataset: {int(survivors['images'].sum())} images -> "
      f"{int(survivors['remaining'].sum())} after de-duplication "
      f"({int(survivors['duplicated_from_NEU'].sum())} removed, "
      f"{100*survivors['duplicated_from_NEU'].sum()/survivors['images'].sum():.1f}%)")
print(f"Classes retaining >= 20 images: {len(kept)} of {len(survivors)}")
"""))

    c.append(md(r"""
## 3. Building the frozen splits · Major Issue 2 (decision letter)

The order of operations, which is the substance of Reviewer 1 Q1 and Q2:

```
enumerate images ──► near-duplicate audit ──► SPLIT (image level, stratified, 70/10/20)
                                                 │
                    ┌────────────────────────────┘
                    ▼
            augment TRAIN only ──► extract features ──► fit bins on TRAIN only
                                                             │
                                    train ──► select on VAL ──┘ ──► score TEST once
```

Two properties are enforced rather than asserted:

* **Duplicate groups never straddle a split.** Near-duplicate images are unioned into a
  group and the *group* is assigned to one split, so an image and its near-twin cannot
  land on opposite sides of the train/test boundary.
* **The manifest is hashed.** `load_split` recomputes the SHA-256 and refuses to load a
  manifest that changed after freezing, so no downstream script can silently re-split.
"""))
    c.append(code(r"""
manifests = {}
for name in ["neu_det", "steeldefectx", "steeldefectx_paper"]:
    print(f"\n{'='*72}\n{name}\n{'='*72}")
    manifests[name] = splits.build_split(name, SplitConfig(), RESULTS_DIR / "splits")
"""))

    c.append(code(r"""
summary = []
for name, df in manifests.items():
    a = json.loads((RESULTS_DIR / "splits" / f"split_audit_{name}.json").read_text())
    intra = a["intra_dataset"]
    summary.append({
        "dataset": name,
        "train": int((df.split == "train").sum()),
        "val": int((df.split == "val").sum()),
        "test": int((df.split == "test").sum()),
        "classes": df["class"].nunique(),
        "cross_dataset_dropped": a.get("cross_dataset", {}).get("n_dropped", 0),
        "intra_dup_groups": intra["n_multi_image_groups"],
        "images_in_dup_groups": intra["n_images_in_multi_groups"],
        "groups_straddling_splits": a["group_violations_final"],
        "sha256": a["manifest_sha256"][:12],
    })
display(pd.DataFrame(summary))
print("\ngroups_straddling_splits must be 0 for every row -- that column IS the "
      "answer to 'was there a near-duplicate check'.")
"""))

    c.append(code(r"""
# Per-class stratification check
for name, df in manifests.items():
    ct = pd.crosstab(df["class"], df["split"])[["train", "val", "test"]]
    ct["test_pct"] = (100 * ct["test"] / ct.sum(axis=1)).round(1)
    print(f"\n--- {name} ---")
    display(ct)
"""))

    c.append(md(r"""
### A note on the previously frozen split

`paper_results/splits/split_v1.csv` (the split the CNN baselines in the repository used)
places at least one near-duplicate group on both sides of the train/test boundary. The
builder detects this, reports it as `frozen_split_group_violations`, and re-splits rather
than reusing it. Any earlier number computed on that split carries that contamination,
small though it is.
"""))

    c.append(md(r"""
## 4. Answers this notebook establishes · Major Issue 2 (decision letter)

**R1 Q1 — Was a test set held out that was never used for checkpoint selection or tuning?**
In the original submission, no: there were only train and validation folds, and the top-5
snapshots were selected on the same 20% that produced the headline accuracy. In this
revision, yes: 70/10/20, checkpoint selection and every hyper-parameter choice use the
validation split, and the test split is scored exactly once per model.

**R1 Q2 — Was the split at image level before augmentation and feature extraction?**
Yes, and the ordering is enforced in code (`src/splits.py` → `src/pipeline.py`).
Augmentation is applied only to images the manifest marks `train`, and it is image-space
augmentation (flip / rotation / gamma / noise) rather than the feature-space jitter used
previously — a jittered copy of a descriptor vector is a near-duplicate of it, which is
exactly the coupling the reviewer suspected. Near-duplicate checking is the group
assignment above.
"""))

    nb = nbf.v4.new_notebook(cells=c)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


# ===========================================================================
# 01 / 02 -- the per-dataset protocol notebook
# ===========================================================================


def protocol_notebook(dataset: str, title: str, with_localization: bool,
                      intro: str, extra_cells=None) -> nbf.NotebookNode:
    c = [md(title), code(HEADER), md(intro)]

    c.append(md(r"""
## 1. Configuration and the frozen split · Major Issue 2 (decision letter)

Every knob is declared in `src/config.py`. The split is loaded from the manifest written
by notebook 00 and its hash is verified on load.
"""))
    c.append(code(f"""
cfg = ExperimentConfig(dataset="{dataset}")
cfg.train.epochs = 40
cfg.train.aug_variants = 3          # TRAIN SPLIT ONLY
cfg.seeds = [42, 1337, 2024, 7, 20250101]

df = splits.load_split("{dataset}", RESULTS_DIR / "splits")   # hash-verified
OUT = cfg.out_dir()
print(json.dumps(cfg.to_dict(), indent=1, default=str)[:1200])
print("\\nsplit:", df.split.value_counts().to_dict())
"""))

    c.append(md(r"""
## 2. Descriptors → bins → prompt · Major Issue 5 (decision letter — feature-to-prompt reproducibility)

The bin edges are fitted on the **training rows only**. Validation and test are
transformed with those edges, and every value that falls outside the fitted range is
counted (Reviewer 1, Q8).
"""))
    c.append(code(r"""
bundle = pipeline.build_bundle(df, cfg, verbose=True)

print(f"\nfeatures per image : {len(bundle.feature_names)}")
print(f"vocabulary         : {len(bundle.tokenizer)} tokens")
print(f"sequence length    : {bundle.tokenizer.max_len}")
print(f"train rows         : {len(bundle.train.y)} (from {len(bundle.train.paths)} images "
      f"x {cfg.train.aug_variants + 1} variants)")
print(f"val / test images  : {len(bundle.val.y)} / {len(bundle.test.y)}")
"""))

    c.append(code(r"""
from prompt import readable_prompt

print("Example prompts (first 14 tokens of each):\n")
for i in [0, len(bundle.test.y) // 3, 2 * len(bundle.test.y) // 3]:
    cls = bundle.classes[bundle.test.y[i]]
    print(f"[{cls}]")
    print("  " + readable_prompt(bundle.test.tokens[i], 14) + "\n")
"""))

    c.append(md(r"""
### Reviewer 1, Q8 · Major Issue 5 — out-of-range values at inference, and the discretization thresholds

**Policy.** Bin edges come from the training split, so out-of-range values at inference are
expected. The declared policy is `clamp`: the value is assigned to the first or last bin
and the occurrence is counted. The token stays in-vocabulary, so a numeric descriptor can
never produce `<unk>`. The alternative policy `oor_token` emits a dedicated
`<feature>=oorlo` / `=oorhi` symbol, and that symbol is registered in the vocabulary at fit
time — the tokenizer is built from the *schema* (every feature × every bin), not from
observed training rows, so no symbol is ever unseen.

**Rate.** Measured below, per split.
"""))
    c.append(code(r"""
oor_rows = []
for split in ["train", "val", "test"]:
    r = bundle.oor_report[split]
    oor_rows.append({
        "split": split,
        "rows": r["n_rows_seen"],
        "value_checks": r["total_value_checks"],
        "out_of_range_values": r["n_out_of_range_values"],
        "rate_pct": r["overall_rate_pct"],
        "oor_features_per_image": r["mean_oor_features_per_image"],
        "unk_token_rate": r["unk_token_rate"],
    })
display(pd.DataFrame(oor_rows))

print("Features most often out of range on the TEST split:")
display(pd.DataFrame(bundle.oor_report["test"]["per_feature"]).T.head(10))
"""))

    c.append(md(r"""
## 3. Main result — five seeds (Reviewer 1, Q9) · Major Issue 2 & 4 (decision letter)

Five seeds, one frozen split. Checkpoints are ranked by **validation macro-F1**; the
top-5 snapshot ensemble is formed from that ranking; the test split is scored once per
seed. Reported as mean ± SD, with a bootstrap 95% CI on the seed-averaged prediction.

This is the direct answer to Major Issue 2 (which subset trains, selects, and reports —
never the same one) and Major Issue 4 (mean ± SD over independent seeds, not a single
configuration) from the decision letter.
"""))
    c.append(code(r"""
main = experiments.run_seeds(bundle, cfg, verbose=True)

agg = main["aggregate"]
tbl = pd.DataFrame({
    m: {"mean": agg[m]["mean"], "sd": agg[m]["sd"], "min": agg[m]["min"], "max": agg[m]["max"]}
    for m in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]
}).T
display(tbl)

sa = main["seed_averaged_prediction"]
print(f"\nSeed-averaged test accuracy : {sa['accuracy']:.4f} "
      f"(95% CI {sa['accuracy_ci95'][0]:.4f}-{sa['accuracy_ci95'][1]:.4f})")
print(f"Seed-averaged test macro-F1 : {sa['macro_f1']:.4f} "
      f"(95% CI {sa['macro_f1_ci95'][0]:.4f}-{sa['macro_f1_ci95'][1]:.4f})")
print(f"Test images: {main['n_test_images']}  ->  one image is "
      f"{100/main['n_test_images']:.3f} accuracy points")
"""))

    c.append(md(r"""
### Reviewer 1, Q10 · Major Issue 4 — per-class precision / recall / F1, balanced accuracy, and the confusion matrix
"""))
    c.append(code(r"""
pc = main["per_class_over_seeds"]
rows = []
for cls, d in pc.items():
    rows.append({
        "class": cls, "support": d["support"],
        "precision": d["precision"]["mean"], "precision_sd": d["precision"]["sd"],
        "recall": d["recall"]["mean"], "recall_sd": d["recall"]["sd"],
        "f1": d["f1"]["mean"], "f1_sd": d["f1"]["sd"],
    })
per_class = pd.DataFrame(rows).sort_values("f1")
display(per_class)
print(f"\nWorst class by F1: {per_class.iloc[0]['class']} ({per_class.iloc[0]['f1']:.4f})")
print(f"Macro-F1 {agg['macro_f1']['mean']:.4f} vs accuracy {agg['accuracy']['mean']:.4f} "
      f"-- the gap is what accuracy alone would have hidden.")
"""))

    c.append(code(r"""
cm = np.array(sa["metrics"]["confusion_matrix"])
fig, ax = plt.subplots(figsize=(1.1 * len(bundle.classes) + 3, 1.0 * len(bundle.classes) + 2.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(bundle.classes))); ax.set_yticks(range(len(bundle.classes)))
ax.set_xticklabels(bundle.classes, rotation=60, ha="right", fontsize=8)
ax.set_yticklabels(bundle.classes, fontsize=8)
ax.set_xlabel("predicted"); ax.set_ylabel("true")
ax.set_title(f"{cfg.dataset} - test confusion matrix (seed-averaged)")
thr = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > thr else "black")
plt.colorbar(im, fraction=0.046); plt.tight_layout(); plt.show()
"""))

    if dataset == "neu_det":
        c.append(md(r"""
### Reviewer 1, Q3 · Major Issue 4 — crazing vs rolled-in scale, and the comparison against published CNNs

The reviewer asks for this specific confusion because it is the pair a texture descriptor
should find hardest.
"""))
        c.append(code(r"""
pair = metrics.confusion_pair(cm, bundle.classes, "crazing", "rolled-in_scale")
print(json.dumps(pair, indent=2))

print("\nAll off-diagonal confusions, largest first:")
conf = [(bundle.classes[i], bundle.classes[j], int(cm[i, j]))
        for i in range(len(bundle.classes)) for j in range(len(bundle.classes))
        if i != j and cm[i, j] > 0]
display(pd.DataFrame(sorted(conf, key=lambda t: -t[2]),
                     columns=["true", "predicted", "count"]).head(12))
"""))

    c.append(md(r"""
## 4. Reviewer 1, Q4 · Major Issue 3 & 6 — tabular baselines on the raw un-discretized vector

These consume the **same descriptors** the prompt is built from, before discretization.
The only difference from SteelSense-BiLSTM is the representation, so the gap between them
is what the text-prompt idea is actually worth. Hyper-parameters are selected on
validation; test is scored once. Every value here is measured by this codebase, on this
split — none are taken from another publication (Major Issue 3).
"""))
    c.append(code(r"""
import baselines_tabular as BT

tab_res, tab_probs = BT.run_all(
    bundle.train.X, bundle.train.y,
    bundle.val.X, bundle.val.y,
    bundle.test.X, bundle.test.y,
    bundle.classes, seed=42, verbose=True,
)

rows = [{"model": k, "val_macro_f1": v.get("val_macro_f1"),
         "test_accuracy": v.get("accuracy"), "test_macro_f1": v.get("macro_f1"),
         "balanced_accuracy": v.get("balanced_accuracy"), "n_params": v.get("n_params")}
        for k, v in tab_res.items() if "macro_f1" in v]
display(pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False))
"""))

    c.append(md(r"""
## 5. Reviewer 1, Q5 · Major Issue 3 & 7 — MobileNetV3 and ShuffleNetV2 under the same protocol

Same split manifest, same selection rule, same single touch of the test split.
Parameters, MACs and batch-size-1 latency are measured by `src/complexity.py` for every
model including ours, so the efficiency table is internally consistent.

> Reduce `CNN_EPOCHS` if you are running on CPU and want a quick pass; the value used for
> the reported numbers is recorded in the output JSON.
"""))
    c.append(code(r"""
import baselines_cnn as BC

CNN_EPOCHS = 20
CNN_MODELS = ["MobileNetV3-Small", "ShuffleNetV2-x0.5"]   # add "MobileNetV2", "ResNet18" if time allows

tr = df[df.split == "train"]; va = df[df.split == "val"]; te = df[df.split == "test"]
cls_idx = {c: i for i, c in enumerate(bundle.classes)}
to_y = lambda d: np.array([cls_idx[c] for c in d["class"]], dtype=np.int64)

cnn_res, cnn_probs = {}, {}
for name in CNN_MODELS:
    print(f"\n=== {name} ===")
    m, p = BC.train_backbone(
        name,
        tr["path"].tolist(), to_y(tr),
        va["path"].tolist(), to_y(va),
        te["path"].tolist(), to_y(te),
        bundle.classes, seed=42, epochs=CNN_EPOCHS, verbose=True,
    )
    cnn_res[name], cnn_probs[name] = m, p
"""))

    c.append(code(r"""
# Efficiency table: our model measured the same way as the CNNs
import torch
from complexity import profile_model, hardware
from model import build_model

# A freshly built copy: parameter count, MACs and latency depend on the
# architecture and the input shape, not on the trained weights.
ss_probe = build_model("bilstm", len(bundle.tokenizer), len(bundle.classes), cfg.model)
example = torch.from_numpy(bundle.test.ids[:1])
ss_complexity = profile_model("SteelSense-BiLSTM", ss_probe, example, is_token_model=True, repeats=100)

eff = [{
    "model": "SteelSense-BiLSTM",
    "input": "92 descriptor tokens",
    "params_M": round(ss_complexity["n_params"] / 1e6, 3),
    "size_MB": ss_complexity["size_mb"],
    "MMACs": ss_complexity["mmacs"],
    "latency_bs1_ms": ss_complexity["latency_bs1"]["median_ms"],
    "test_macro_f1": agg["macro_f1"]["mean"],
}]
for k, v in cnn_res.items():
    cx = v["complexity"]
    eff.append({
        "model": k + (" (ImageNet)" if v["pretrained"] else " (scratch)"),
        "input": f"3x{v['img_size']}x{v['img_size']} image",
        "params_M": round(cx["n_params"] / 1e6, 3),
        "size_MB": cx["size_mb"],
        "MMACs": cx["mmacs"],
        "latency_bs1_ms": cx["latency_bs1"]["median_ms"],
        "test_macro_f1": v["macro_f1"],
    })
efficiency = pd.DataFrame(eff)
display(efficiency)
print(json.dumps(hardware(), indent=1))
print("\nNOTE: the BiLSTM latency column is the CLASSIFIER ONLY. Section 9 measures the "
      "end-to-end cost including descriptor extraction, which is the number a deployment "
      "claim has to use.")
"""))

    c.append(md(r"""
### Paired significance test against the strongest baseline (Reviewer 1, Q9) · Major Issue 3

McNemar's exact test on the identical test images, because the predictions are paired
per image. The exact binomial form is used rather than the χ² approximation since the
discordant count is small.
"""))
    c.append(code(r"""
all_base = {**tab_res, **cnn_res}
all_prob = {**tab_probs, **cnn_probs}
cmp = experiments.compare_to_baselines(main, all_base, all_prob, bundle.test.y, bundle.classes)

display(pd.DataFrame(cmp["comparison_table"]).sort_values("macro_f1", ascending=False))
print(f"\nStrongest baseline: {cmp['strongest_baseline']} "
      f"(macro-F1 {cmp['strongest_baseline_macro_f1']:.4f})")
print(f"SteelSense-BiLSTM  : macro-F1 {cmp['steelsense_macro_f1_mean']:.4f}")
print("\nMcNemar exact test vs strongest baseline:")
print(json.dumps(cmp["mcnemar_vs_strongest"], indent=2))
"""))

    c.append(md(r"""
## 6. Reviewer 1, Q6 · Major Issue 6 — does the token order matter?

Four conditions, identical everywhere except the sequence encoder and the token order:

| | encoder | order |
|---|---|---|
| **A** | BiLSTM | canonical (deployed) |
| **B** | BiLSTM | one fixed random permutation, same for every sample |
| **C** | BiLSTM | a fresh permutation per sample — order information destroyed |
| **D** | DeepSets (position-wise MLP, permutation invariant) | canonical |

**How to read it.** A ≈ B means the specific order is arbitrary. A ≈ C means the model is
not using order at all. A ≈ D means the recurrent encoder buys nothing over an order-free
encoder of the same width — in which case the paper should say so and use the cheaper one.
"""))
    c.append(code(r"""
order_ab = experiments.run_order_ablation(df, cfg, seeds=cfg.seeds[:3], verbose=True)

rows = []
for k, v in order_ab.items():
    if not isinstance(v, dict) or "aggregate" not in v:
        continue
    rows.append({
        "condition": k, "encoder": v["arch"], "order": v["order"],
        "accuracy_mean": v["aggregate"]["accuracy"]["mean"],
        "accuracy_sd": v["aggregate"]["accuracy"]["sd"],
        "macro_f1_mean": v["aggregate"]["macro_f1"]["mean"],
        "macro_f1_sd": v["aggregate"]["macro_f1"]["sd"],
        "n_params": v["n_params"],
    })
display(pd.DataFrame(rows))
print(json.dumps(order_ab["verdict"], indent=2))
print("\nPaired tests vs condition A:")
for k, v in order_ab["paired_tests_vs_A"].items():
    print(f"  {k}: mean_diff {v.get('mean_diff', float('nan')):+.4f}  "
          f"t p={v.get('t_p_value')}  wilcoxon p={v.get('wilcoxon_p_value')}")
"""))

    c.append(md(r"""
## 6b. Which component earns its parameters? · Major Issue 6 (decision letter) — Reviewer 1 Q1 · Reviewer 2 Q6 · Reviewer 3 Q7/Q8

Six conditions, everything else held fixed at the main run's split, seeds, epochs and
checkpoint-selection rule (validation macro-F1; test scored once per condition):

| Condition | What changes |
|---|---|
| `full_5snapshot_ensemble` | nothing — the deployed configuration |
| `no_ensemble_best_checkpoint` | the SAME fit, rescored with only its single best-validation checkpoint |
| `pool_attention_only` / `pool_max_only` / `pool_mean_only` | the pooled head keeps exactly one of the three views |
| `no_discretization_numeric` | the raw standardized descriptor vector (fit on **train only**), through the identical BiLSTM encoder and pooled head — no discretizer, no tokenizer anywhere in this condition |

The tabular baselines in §4 already show what a *different model family* (trees, a generic
MLP) does on the raw vector; they cannot isolate discretization on its own because they are
not the deployed encoder. `no_discretization_numeric` is the control that actually answers
"does converting continuous descriptors into categorical tokens throw away useful
information, or does it help" — with everything else about the architecture unchanged.
"""))
    c.append(code(r"""
ablation = experiments.run_component_ablation(
    bundle, cfg, seeds=cfg.seeds[:3], epochs=30, early_stop_patience=8, verbose=True,
)
abl_df = pd.DataFrame(ablation["summary"]).sort_values("macro_f1_mean", ascending=False)
display(abl_df)

print("\nPaired tests vs the full (deployed) configuration:")
for k, v in ablation["paired_tests_vs_full"].items():
    print(f"  {k:30s} mean_diff {v.get('mean_diff', float('nan')):+.4f}  "
          f"t p={v.get('t_p_value')}  wilcoxon p={v.get('wilcoxon_p_value')}")
print("\n" + ablation["note"])
"""))

    c.append(md(r"""
## 7. Reviewer 1, Q7 · Major Issue 5 — bin count and edge rule

The submitted version used three semantic levels (Low / Medium / High) with no
justification. This sweep supplies one, or replaces it: bin counts crossed with quantile
against equal-width edges, under identical splits. This is the discretization-threshold
justification Major Issue 5 of the decision letter asks for.

The out-of-range column matters here too — equal-width bins on a skewed descriptor put
almost all mass in one bin and push more test values outside the fitted range.

**Compute budget.** A full 6 bin-counts x 2 strategies x 3 seeds grid at the main run's
epoch budget is `12 x 3 = 36` full trainings and was taking multiple days wall-clock on a
single CPU, with no way to resume after an interrupted kernel. The cell below trims the
grid to the bin counts that actually distinguish the answer (2 and 15 already showed the
plateau at both ends in an earlier run), drops to 2 seeds, and gives the sweep its own
lighter `epochs` / `early_stop_patience` / `batch_size` — legitimate here because this
is a relative comparison across bin configs, not the headline result, and checkpoint
selection still runs on validation macro-F1 only. `checkpoint_path` makes it resumable:
re-running the cell after an interruption skips every combo already finished instead of
starting over.
"""))
    c.append(code(r"""
sweep = experiments.run_bin_sweep(
    df, cfg, bin_counts=(3, 5, 7, 10),
    strategies=("quantile", "uniform"),
    seeds=cfg.seeds[:2],
    epochs=30, early_stop_patience=6, batch_size=64,
    checkpoint_path=OUT / "bin_sweep_checkpoint.json",
    verbose=True,
)
sw = pd.DataFrame(sweep["sweep"])
if "failed" in sw.columns:
    fails = sw[sw["failed"] == True]
    if len(fails):
        print("FAILED combos (see error column):")
        display(fails[["strategy", "n_bins", "error"]])
    sw = sw[sw["failed"] != True].reset_index(drop=True)
display(sw)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for strat, g in sw.groupby("strategy"):
    ax[0].errorbar(g.n_bins, g.macro_f1_mean, yerr=g.macro_f1_sd, marker="o", capsize=3, label=strat)
    ax[1].plot(g.n_bins, g.test_oor_rate_pct, marker="s", label=strat)
ax[0].set_xlabel("number of bins"); ax[0].set_ylabel("test macro-F1"); ax[0].legend()
ax[0].set_title(f"Bin count vs macro-F1 (mean +- SD, {len(cfg.seeds[:2])} seeds)"); ax[0].grid(alpha=.3)
ax[1].set_xlabel("number of bins"); ax[1].set_ylabel("test out-of-range rate (%)"); ax[1].legend()
ax[1].set_title("Bin count vs out-of-range rate"); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

print("Best configuration:", json.dumps(sweep["best"], indent=2))
"""))

    c.append(md(r"""
## 8. Reviewer 2, Q5 — substantiating "interpretable"

Two independent views, plus their agreement. Attention alone is not an explanation, so it
is checked against a causal measure.

* **Attention** — the pooling attention distribution, averaged per true class. Token
  position *i* is always feature *i*, so a weight is attributable to a named descriptor.
* **Permutation importance** — each descriptor's token is resampled across the test split
  and the drop in macro-F1 recorded over several repeats.
* **Agreement** — Spearman ρ between the two rankings. If ρ is low, the attention figure
  must not be presented as an explanation, and the manuscript should say so.
"""))
    c.append(code(r"""
import interpret

ss_model, ss_res = main["_model"]
attn = interpret.attention_by_class(
    ss_model, ss_res.snapshots, bundle.test.ids, bundle.test.y,
    bundle.feature_names, bundle.classes,
)

print("Most class-DISTINCTIVE descriptors (attention above the dataset mean):\n")
for cls, items in attn["most_distinctive_by_class"].items():
    top = ", ".join(f"{d['feature']}({d['delta_vs_mean']:+.4f})" for d in items[:4])
    print(f"  {cls:32s} {top}")
"""))
    c.append(code(r"""
perm = interpret.permutation_importance(
    ss_model, ss_res.snapshots, bundle.test.ids, bundle.test.y,
    bundle.feature_names, bundle.classes, n_repeats=3, verbose=True,
)
display(pd.DataFrame(perm["top20"]))

agree = interpret.agreement(attn, perm)
print(json.dumps(agree, indent=2))
"""))
    c.append(code(r"""
mean_attn = np.array(attn["mean_attention"])
top_idx = np.argsort(-mean_attn.mean(0))[:25]
fig, ax = plt.subplots(figsize=(13, 0.45 * len(bundle.classes) + 2.5))
im = ax.imshow(mean_attn[:, top_idx], aspect="auto", cmap="magma")
ax.set_yticks(range(len(bundle.classes))); ax.set_yticklabels(bundle.classes, fontsize=8)
ax.set_xticks(range(len(top_idx)))
ax.set_xticklabels([bundle.feature_names[j] for j in top_idx], rotation=75, ha="right", fontsize=7)
ax.set_title("Mean attention per class over the 25 most-attended descriptors")
plt.colorbar(im, fraction=0.02); plt.tight_layout(); plt.show()
"""))

    if with_localization:
        c.append(md(r"""
## 9. Localization, measured honestly (Reviewer 2, Q1–Q3) · Major Issue 1 (decision letter)

### Q1 — which module produces boxes, and how are they linked to the class prediction?

```
                    image (200x200 grayscale)
                              │
              ┌───────────────┴────────────────┐
              ▼                                ▼
   ┌──────────────────────┐        ┌──────────────────────────┐
   │ descriptor extractor │        │ proposal detector        │
   │ 92 scalars           │        │ Canny(2 thr) + morph +   │
   └──────────┬───────────┘        │ Otsu blobs -> NMS        │
              ▼                    │ each box gets a          │
   ┌──────────────────────┐        │ CONFIDENCE from region   │
   │ bins (train-fitted)  │        │ evidence (contrast,      │
   │ -> tokens -> BiLSTM  │        │ texture, size prior)     │
   └──────────┬───────────┘        └───────────┬──────────────┘
              │ class label + p(class)         │ boxes + scores
              └──────────────┬─────────────────┘
                             ▼
               detection = (box, class label, score x p(class))
```

The BiLSTM has no box-regression head and never had one — the reviewer is correct. It
supplies the **label and its probability**; the proposal detector supplies the
**geometry and a confidence**. The product of the two ranks the detections, which is what
makes average precision computable at all.

### Q2 — what fraction of test images uses the contour fallback rather than the XML path?

**Neither, at inference.** The XML is ground truth only; `xml_used_at_inference` is
`False` and no code path reads an annotation before predicting. The fallback that does
exist — a single whole-image box at confidence 0.05, when no proposal survives — is
counted and reported as `proposal_fallback_rate_pct`.

### Q3 — IoU histogram and how localization was evaluated

Standard VOC all-point-interpolation AP over confidence-ranked detections at IoU 0.50 to
0.95, on the **test split only**, plus the IoU histogram of each image's top-scoring box
against its best-matching ground truth.

Reported alongside, and clearly separated, is the **ORACLE** figure: ground-truth geometry
carrying the predicted label. That is what the submitted Table 4 measured. It is
threshold-invariant by construction and is a classification ceiling, not detection.
"""))
        c.append(code(r"""
import localize

te_df = df[df.split == "test"].reset_index(drop=True)
te_paths = te_df["path"].tolist()
probs = main["_probs"]
pred_labels = [bundle.classes[i] for i in probs.argmax(1)]
pred_conf = probs.max(1).tolist()

loc = localize.run_localization(
    te_paths, pred_labels, pred_conf,
    annotation_dir=DATASETS[cfg.dataset].annotation_dir,
    img_size=cfg.spec().img_size, classes=bundle.classes, verbose=True,
)

print(f"\nimages scored              : {loc['n_images_scored']}")
print(f"XML used at inference      : {loc['xml_used_at_inference']}")
print(f"proposal fallback rate     : {loc['proposal_fallback_rate_pct']}%")
print(f"mean proposals per image   : {loc['mean_proposals_per_image']}")

det, orc = loc["DET_real_detector"], loc["ORACLE_classification_ceiling"]
display(pd.DataFrame([
    {"quantity": "[DET] proposal detector (real)", "AP50": det["AP50"],
     "AP75": det["AP75"], "mAP@[.5:.95]": det["mAP@[.5:.95]"]},
    {"quantity": "[ORACLE] GT boxes + predicted label", "AP50": orc["AP50"],
     "AP75": orc["AP75"], "mAP@[.5:.95]": orc["mAP@[.5:.95]"]},
]))
print("\n" + orc["WARNING"])
"""))
        c.append(code(r"""
h = loc["DET_iou_histogram"]
print(f"mean IoU {h['mean_iou']}   median {h['median_iou']}   "
      f"IoU>=0.5: {h['pct_iou_ge_0.5']}%   IoU>=0.75: {h['pct_iou_ge_0.75']}%")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
edges = h["histogram_edges"]
ax[0].bar([(edges[i] + edges[i+1]) / 2 for i in range(len(h["histogram_counts"]))],
          h["histogram_counts"], width=0.09, edgecolor="black")
ax[0].axvline(0.5, ls="--", c="red", label="IoU 0.50")
ax[0].axvline(0.75, ls="--", c="darkred", label="IoU 0.75")
ax[0].set_xlabel("IoU of top-scoring prediction vs matched GT"); ax[0].set_ylabel("images")
ax[0].set_title("IoU histogram on the test split"); ax[0].legend(); ax[0].grid(alpha=.3)

ks = sorted(det["per_iou_AP"]); vs = [det["per_iou_AP"][k] for k in ks]
ax[1].plot([float(k) for k in ks], vs, marker="o", label="[DET] real detector")
ax[1].plot([float(k) for k in ks], [orc["per_iou_AP"][k] for k in ks],
           marker="s", ls="--", label="[ORACLE] GT geometry")
ax[1].set_xlabel("IoU threshold"); ax[1].set_ylabel("AP (%)")
ax[1].set_title("AP vs IoU threshold"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

print("\nThe flat ORACLE curve is the signature Reviewer 2 identified: AP identical at "
      "every threshold means IoU is 1 by construction. The DET curve decays, which is "
      "what a genuine localization measurement looks like.")
"""))
        c.append(code(r"""
display(pd.DataFrame([det["per_class_AP50"]], index=["AP50 (%)"]).T.sort_values("AP50 (%)"))
"""))

        c.append(md(r"""
### Qualitative check — predicted boxes vs. ground truth, including failure cases · Minor Issue (decision letter, Fig. 4)

Reviewer 1 asks that a qualitative figure clearly distinguish predicted boxes from
ground-truth annotations and include several representative failures, not only successes.
Green = [DET] predicted box (the real, confidence-scored proposal detector — never the
XML). Red = ground truth, shown only for comparison. The bottom row is the three
**lowest**-IoU images on the test split, picked automatically, not curated.
"""))
        c.append(code(r"""
import cv2

qual = loc["DET_qualitative_examples"]
path_of = {Path(p).stem: p for p in te_paths}

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for row, (title, examples) in enumerate([("good (highest IoU)", qual["good"]),
                                          ("failure cases (lowest IoU)", qual["bad"])]):
    for col in range(3):
        ax = axes[row, col]
        if col >= len(examples):
            ax.axis("off")
            continue
        ex = examples[col]
        img = cv2.imread(path_of[ex["image_id"]], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (cfg.spec().img_size, cfg.spec().img_size))
        ax.imshow(img, cmap="gray")
        x0, y0, x1, y1 = ex["pred_box"]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                    edgecolor="lime", linewidth=2, label="[DET] predicted"))
        for gx0, gy0, gx1, gy1 in ex["gt_boxes"]:
            ax.add_patch(plt.Rectangle((gx0, gy0), gx1 - gx0, gy1 - gy0, fill=False,
                                        edgecolor="red", linewidth=2, linestyle="--",
                                        label="ground truth"))
        ax.set_title(f"{ex['image_id']}\nIoU={ex['best_iou']:.2f}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        if row == 0 and col == 0:
            ax.legend(loc="upper right", fontsize=7)
    axes[row, 0].set_ylabel(title, fontsize=10)
plt.suptitle("[DET] predicted (green, solid) vs. ground truth (red, dashed)")
plt.tight_layout(); plt.show()
"""))

    sec = "10" if with_localization else "9"
    c.append(md(rf"""
## {sec}. Reviewer 2, Q6 · Major Issue 7 (decision letter) — per-stage latency at batch size 1

Every stage timed separately on the stated hardware, batch size 1, after warm-up.

**Read the totals before repeating any deployment claim.** Small in *parameters* is not the
same as fast at batch size 1: an LSTM is sequential over ~94 tokens, so its cost is set by
per-timestep kernel overhead rather than by its MAC count, and the 5-member snapshot
ensemble multiplies that by five. The order-free control from §6 is timed on the identical
input so the accuracy question and the cost question can be read together.

The thread-scaling measurement below replaces the unquantified "HPC-accelerated" claim
with a measured curve — report it or drop the claim.
"""))
    c.append(code(r"""
import profile_stages as PS

ds_probe = build_model("deepsets", len(bundle.tokenizer), len(bundle.classes), cfg.model)

prof = PS.profile_pipeline(
    bundle.test.paths[0], cfg.features, bundle.discretizer, bundle.tokenizer,
    ss_model, ss_res.snapshots, bundle.feature_names, deepsets_model=ds_probe,
    repeats=100, include_localization=""" + ("True" if with_localization else "False") + r""",
)

stages = pd.DataFrame(prof["stages_ms"]).T[["median_ms", "mean_ms", "sd_ms", "p95_ms"]]
stages["share_pct"] = (100 * stages["median_ms"] / prof["totals"]["end_to_end_ensemble_ms"]).round(2)
display(stages.sort_values("median_ms", ascending=False))
print(json.dumps(prof["totals"], indent=2))
print("\n" + prof["note"])

t = prof["totals"]
print(f"\nInference is {t['inference_share_of_end_to_end_pct']}% of end-to-end "
      f"cost at batch size 1.")
if "bilstm_vs_deepsets_speed_ratio" in t:
    print(f"The BiLSTM forward pass is {t['bilstm_vs_deepsets_speed_ratio']}x the cost of "
          f"the order-free control ({t['deepsets_forward_ms']} ms). Read this next to the "
          f"accuracy gap in section 6: if that gap is not significant, the recurrent "
          f"encoder is paying this ratio for nothing.")
"""))
    c.append(code(r"""
s = stages.sort_values("median_ms", ascending=True)
fig, ax = plt.subplots(figsize=(9, 0.38 * len(s) + 1.5))
ax.barh(s.index, s["median_ms"], color="steelblue", edgecolor="black")
ax.set_xlabel("median latency (ms), batch size 1")
ax.set_title(f"Per-stage cost — {prof['hardware']['processor'][:48]}")
for i, (name, v) in enumerate(s["median_ms"].items()):
    ax.text(v, i, f" {v:.2f}", va="center", fontsize=8)
ax.grid(alpha=.3, axis="x"); plt.tight_layout(); plt.show()
"""))
    c.append(code(r"""
speed = PS.speedup_report(bundle.test.paths[:120], cfg.features, thread_counts=(1, 2, 4, 8))
display(pd.DataFrame(speed["by_thread_count"]).T)
print("Report this curve, or remove the 'HPC-accelerated' claim from the manuscript.")
"""))

    if extra_cells:
        c.extend(extra_cells)

    last = str(int(sec) + 1)
    c.append(md(rf"""
## {last}. Persist everything
"""))
    c.append(code(r"""
payload = {
    "config": cfg.to_dict(),
    "split_counts": df.split.value_counts().to_dict(),
    "main_multiseed": main,
    "per_class": per_class.to_dict(orient="records"),
    "out_of_range": bundle.oor_report,
    "baselines_tabular": tab_res,
    "baselines_cnn": {k: {kk: vv for kk, vv in v.items()} for k, v in cnn_res.items()},
    "efficiency_table": efficiency.to_dict(orient="records"),
    "comparison_vs_baselines": cmp,
    "order_ablation": order_ab,
    "bin_sweep": sweep,
    "interpretability": {"attention": attn, "permutation_importance": perm, "agreement": agree},
    "stage_profile": prof,
    "thread_scaling": speed,
}
""" + ("payload['localization'] = loc\n" if with_localization else "") + r"""
experiments.save_json(OUT / "results.json", payload)
efficiency.to_csv(OUT / "table_efficiency.csv", index=False)
per_class.to_csv(OUT / "table_per_class.csv", index=False)
pd.DataFrame(cmp["comparison_table"]).to_csv(OUT / "table_baselines.csv", index=False)
sw.to_csv(OUT / "table_bin_sweep.csv", index=False)
print("written to", OUT)
for f in sorted(OUT.iterdir()):
    print("  ", f.name)
"""))

    nb = nbf.v4.new_notebook(cells=c)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


NEU_INTRO = r"""
### What changed relative to the submitted version

| | submitted | this notebook |
|---|---|---|
| Splits | 80/20 train/val, no test | 70/10/20, image-level, stratified, duplicate-grouped |
| Checkpoint selection | top-5 on the same 20% that was reported | top-5 on **validation**; test scored once |
| Augmentation | feature-space jitter over **all** images, split applied afterwards | image-space, **train images only**, applied after the split |
| Near-duplicate check | none | dHash + correlation, groups pinned to one split |
| Seeds | 1 | 5, mean ± SD, with paired tests |
| Metrics | accuracy | macro-F1, balanced accuracy, per-class P/R/F1, bootstrap CI |
| Baselines | none on the same features | XGBoost, RF, MLP, LogReg, SVM + MobileNetV3, ShuffleNetV2 |
| Localization | GT boxes relabelled, scored as detection | real proposal detector with confidences; the oracle reported separately and labelled |

**Expect the headline number to fall.** 99.89% on a validation set that also selected the
checkpoints is not comparable to a single-touch test figure, and it should not be.
"""

SDX_INTRO = r"""
### The dataset used here

The **six-class SteelDefectX subset as used in the manuscript** - Crazing, Inclusion,
Patches, Pitted surface, Rolled in scale, Scratches; 1631 images, split 1142 / 163 / 326.

Inclusion is 557 of 1631 images (34%), so accuracy alone is not reportable here and every
table below leads with macro-F1 and balanced accuracy (Reviewer 1, Q10).

### What these numbers do and do not support

The split is built exactly as in notebook 01 - image level, stratified, near-duplicate
groups pinned to one side - so **the results here are internally valid** as a standalone
benchmark, and every reviewer question about protocol is answered on this dataset in the
sections below.

What the overlap with NEU-DET (notebook 00, SS2) rules out is narrower, but it has to be
stated in the manuscript:

* this subset is **not an independent second corpus** - 64.1% of its images also appear in
  NEU-DET, so a NEU-DET result and a SteelDefectX result are not two independent pieces of
  evidence and must not be presented as mutual confirmation;
* **no cross-dataset transfer claim** is supportable - a model trained on NEU-DET and
  tested here would be scoring its own training images;
* the crazing F1 of 1.000 Reviewer 2 questioned has its explanation there, not in the
  descriptors.

Section 9b measures that overlap directly rather than leaving it to be inferred, which is
what turns the reviewer's suspicion into a reported number.

There are no annotations for SteelDefectX, so localization is evaluated on NEU-DET only
(notebook 01, section 9).

> **Appendix option.** For a genuinely NEU-disjoint evaluation, change the dataset in
> section 1 to `"steeldefectx"` - the full 24-class version with duplicates removed
> (3545 images, 19 classes at usable size). Everything downstream adapts automatically.
"""


SDX_EXTRA = [
    md(r"""
## 9b. The NEU-DET overlap, measured (Reviewer 1 Q2 · Reviewer 2 Q4)

The results above are a valid standalone benchmark on this dataset. This section addresses
the separate question the reviewers raised: **how much of this dataset is also NEU-DET, and
what does that do to the numbers?**

Two comparisons, both under the identical protocol and the same seeds:

| | dataset | what it isolates |
|---|---|---|
| **as reported** | the 6-class subset, 1631 images | the number in the paper |
| **NEU-disjoint** | the same subset with NEU-DET duplicates removed | what is left that is genuinely not NEU-DET |

The second row is expected to cover only part of the label space — de-duplication leaves
too few images in several of the six classes — and **that** is the finding to report:
the six-class SteelDefectX task cannot be reconstituted from NEU-disjoint data.

Stating this in the manuscript is what converts Reviewer 2's "invites rather than resolves
the leakage question" into a resolved question.
"""),
    code(r"""
overlap = {}
for ds in ["steeldefectx_paper", "steeldefectx_paper_clean"]:
    print(f"{ds} ...", flush=True)
    d2 = splits.build_split(ds, SplitConfig(), RESULTS_DIR / "splits", verbose=False)
    c2 = ExperimentConfig(dataset=ds)
    c2.train.epochs = cfg.train.epochs
    b2 = pipeline.build_bundle(d2, c2, verbose=False)
    r2 = experiments.run_seeds(b2, c2, seeds=cfg.seeds[:3], verbose=False)
    overlap[ds] = {
        "images": int(len(d2)),
        "classes": int(d2["class"].nunique()),
        "test_images": r2["n_test_images"],
        "accuracy": r2["aggregate"]["accuracy"]["mean"],
        "accuracy_sd": r2["aggregate"]["accuracy"]["sd"],
        "macro_f1": r2["aggregate"]["macro_f1"]["mean"],
        "macro_f1_sd": r2["aggregate"]["macro_f1"]["sd"],
        "balanced_accuracy": r2["aggregate"]["balanced_accuracy"]["mean"],
    }
    print(f"  acc {overlap[ds]['accuracy']:.4f}  macroF1 {overlap[ds]['macro_f1']:.4f}  "
          f"({overlap[ds]['classes']} classes, {overlap[ds]['images']} images)")

ov = pd.DataFrame(overlap).T
ov.index = ["6-class subset AS REPORTED", "same subset, NEU-DET duplicates removed"]
display(ov)
"""),
    code(r"""
# Per-class survival: which of the six classes exist at all without NEU-DET images
from duplicates import build_signatures, cross_dataset_duplicates

neu_i = splits.enumerate_images(DATASETS["neu_det"])
sdx_i = splits.enumerate_images(DATASETS["steeldefectx_paper"])
dups6 = cross_dataset_duplicates(
    build_signatures([i.path for i in sdx_i], 8),
    build_signatures([i.path for i in neu_i], 8),
    hamming_max=6, corr_min=0.97,
)
lab = {i.path: i.label for i in sdx_i}
n_by = pd.Series([i.label for i in sdx_i]).value_counts()
d_by = pd.Series([lab[p] for p in dups6]).value_counts()
surv = pd.DataFrame({"images": n_by, "also_in_NEU_DET": d_by}).fillna(0).astype(int)
surv["NEU_disjoint"] = surv["images"] - surv["also_in_NEU_DET"]
surv["pct_overlap"] = (100 * surv["also_in_NEU_DET"] / surv["images"]).round(1)
surv = surv.sort_values("pct_overlap", ascending=False)
display(surv)

lost = surv[surv["NEU_disjoint"] < 20]
print()
print(f"Overlap with NEU-DET: {surv['also_in_NEU_DET'].sum()} of {surv['images'].sum()} "
      f"images ({100*surv['also_in_NEU_DET'].sum()/surv['images'].sum():.1f}%)")
if len(lost):
    print(f"Classes with fewer than 20 NEU-disjoint images: {list(lost.index)}")
    print("The six-class task therefore cannot be reconstituted from NEU-disjoint data. "
          "Report this subset as a related benchmark, not as independent evidence.")
"""),
]


def main() -> None:
    nbf.write(audit_notebook(), str(HERE / "00_Dataset_Integrity_Audit.ipynb"))
    nbf.write(
        protocol_notebook(
            "neu_det",
            "# 01 — SteelSense-BiLSTM v2 on NEU-DET\n\n"
            "Full revised protocol: held-out test set, five seeds, tabular and CNN "
            "baselines, order and binning ablations, interpretability evidence, honest "
            "localization, and an end-to-end latency breakdown.\n\n"
            "**Prerequisite:** run `00_Dataset_Integrity_Audit.ipynb` first — it writes "
            "the frozen split manifest this notebook loads.",
            with_localization=True,
            intro=NEU_INTRO,
        ),
        str(HERE / "01_NEU_DET_SteelSense_BiLSTM.ipynb"),
    )
    nbf.write(
        protocol_notebook(
            "steeldefectx_paper",
            "# 02 — SteelSense-BiLSTM v2 on SteelDefectX (6 classes, as in the paper)\n\n"
            "Full revised protocol on the six-class SteelDefectX subset: held-out "
            "test set, five seeds, tabular and CNN baselines, order and binning "
            "ablations, interpretability evidence, and an end-to-end latency "
            "breakdown.\n\n"
            "**Prerequisite:** run `00_Dataset_Integrity_Audit.ipynb` first.",
            with_localization=False,
            intro=SDX_INTRO,
            extra_cells=SDX_EXTRA,
        ),
        str(HERE / "02_SteelDefectX_SteelSense_BiLSTM.ipynb"),
    )
    bad = 0
    for f in sorted(HERE.glob("*.ipynb")):
        if f.name.endswith(".executed.ipynb"):
            continue
        nb = nbf.read(str(f), as_version=4)
        n_code = 0
        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            n_code += 1
            # `display` is an IPython builtin; swap it so ast can parse the cell.
            try:
                ast.parse(cell.source.replace("display(", "print("))
            except SyntaxError as e:
                bad += 1
                print(f"  SYNTAX ERROR in {f.name} cell {i}: {e}")
        print(f"{f.name}: {len(nb.cells)} cells ({n_code} code)"
              + ("" if not bad else "  <-- HAS ERRORS"))
    if bad:
        raise SystemExit(f"{bad} notebook cell(s) failed to parse")
    print("all code cells parse")


if __name__ == "__main__":
    main()
