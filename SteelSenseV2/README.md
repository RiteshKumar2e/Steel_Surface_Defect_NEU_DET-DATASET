# SteelSense-BiLSTM v2 — revised protocol for NEU-DET and SteelDefectX

A rebuild of the SteelSense-BiLSTM pipeline that addresses the review of the original
submission. The model idea is unchanged — handcrafted descriptors are discretized into
tokens and classified by a small BiLSTM — but the **evaluation protocol is rebuilt from
the split outward**, and the claims that could not survive measurement are withdrawn
rather than defended.

`REVIEWER_RESPONSE.md` maps all 17 reviewer questions to the notebook section and module
that answers each one.

---

## Run order

```bash
python make_notebooks.py      # regenerate the notebooks (already generated)
python warm_cache.py          # one-time: build split manifests + feature cache (~15 min)
python validate.py            # fast smoke test of every code path (~5 min)
```

Then, in order:

| Notebook | What it does |
|---|---|
| `00_Dataset_Integrity_Audit.ipynb` | **Run first.** Duplicate audits, and writes the frozen, hash-verified split manifests every other notebook loads. |
| `01_NEU_DET_SteelSense_BiLSTM.ipynb` | Full protocol on NEU-DET, including localization and the latency breakdown. |
| `02_SteelDefectX_SteelSense_BiLSTM.ipynb` | Full protocol on the **six-class SteelDefectX subset used in the paper**, plus §9b measuring its NEU-DET overlap. |

Results land in `results/<dataset>/main/` as `results.json` plus CSV tables.

---

## The dataset overlap finding

Four SteelDefectX classes are **NEU-DET images re-encoded at 256x256** - verified at pixel
level, e.g. `cracking_01.jpg` and `crazing_1.jpg` correlate at 0.999.

| SteelDefectX class | images | also in NEU-DET |
|---|---|---|
| Patches | 210 | 210 (100%) |
| Pitted surface | 210 | 210 (100%) |
| Rolled in scale | 210 | 210 (100%) |
| Crazing | 210 | 209 (99.5%) |
| Inclusion | 557 | ~207 (37%) |
| Scratches | 234 | 0 |

Across the six-class subset: **1046 of 1631 images (64.1%)** also appear in NEU-DET.
Across all 24 classes: 1326 of 4871 (27.2%).

**What this does and does not invalidate.** The duplication is *cross-dataset*, not within
SteelDefectX. Notebook 02's split is image-level with duplicate groups pinned to one side,
so results on the six-class subset are internally valid as a standalone benchmark. What the
overlap rules out is narrower and must be stated in the manuscript:

* the subset is **not an independent second corpus** - NEU-DET and SteelDefectX results are
  not mutual confirmation;
* **no cross-dataset transfer claim** survives - training on NEU-DET and testing here means
  scoring training images;
* the crazing F1 of 1.000 Reviewer 2 flagged is explained by the overlap, not by the
  descriptors.

Notebook 02 reports the six-class subset as the primary result and measures the overlap
effect in §9b. A NEU-disjoint appendix configuration (`"steeldefectx"`, 3545 images across
19 classes) is one config line away.

## What changed relative to the submitted version

| | submitted | this repository |
|---|---|---|
| Splits | 80/20 train/val, no test set | 70/10/20, image-level, stratified, duplicate-grouped, hash-frozen |
| Checkpoint selection | top-5 snapshots on the same 20% that was reported | top-5 on **validation**; test scored once |
| Augmentation | feature-space jitter over **all** images, split applied afterwards | image-space, **train images only**, applied after the split |
| Near-duplicate check | none | dHash + correlation, within and across datasets |
| Seeds | 1 | 5, mean ± SD, bootstrap CI, McNemar + paired tests |
| Metrics | accuracy | macro-F1, balanced accuracy, per-class P/R/F1, κ, ECE |
| Baselines | none on the same features | XGBoost, RF, MLP, LogReg, SVM + MobileNetV3-Small, ShuffleNetV2 |
| Order claim | asserted | tested against a permutation-invariant control |
| Binning | 3 levels, unjustified | swept 2–15 × {quantile, equal-width} |
| Out-of-range values | undefined | declared policy, schema-fitted vocabulary, measured rate |
| Localization | GT boxes relabelled, scored as detection | real detector with confidences; oracle reported separately and labelled |
| Latency | "lightweight, HPC-accelerated" | 16-stage breakdown at batch size 1 + measured thread scaling |

**Expect the headline accuracy to fall.** 99.89% measured on the split that also selected
the checkpoints is not comparable to a single-touch test figure, and it should not be.

**Expect the efficiency story to change too.** On the CPU this was built on, the BiLSTM
forward pass at batch size 1 costs more than the entire descriptor extraction stage --
an LSTM is sequential over ~94 tokens, so per-timestep overhead dominates and the MAC
count does not predict the wall clock. 1.75M parameters is not by itself a deployment
argument; §9-10 of the notebooks measures what actually is.

---

## Pipeline

```
split manifest (images, hash-verified)
        │
        ├── augment TRAIN ONLY (flip / rotate / gamma / noise, image-space)
        ▼
   92 descriptors per image  ── GLCM(2 dist × 4 angles), LBP, Canny/Sobel/Laplacian,
        │                        contour geometry, FFT bands, quadrant statistics
        ├──────────────────────────────► tabular baselines consume THIS (raw floats)
        ▼
   discretizer FITTED ON TRAIN ONLY  ── quantile or equal-width, OOR counted
        ▼
   tokens  "cnt_n=b4 glcm_d1_contrast_mean=b2 ..."   (vocabulary fitted from the schema)
        ▼
   SteelSense-BiLSTM  ── embedding → 2-layer BiLSTM → {attention, max, mean} pooling
        │                → LayerNorm → MLP head
        ▼
   select on VAL ──► score TEST once
```

For localization the class prediction is combined with a separate proposal detector:

```
image ──┬──► descriptors ──► BiLSTM ──► class label + p(class) ──┐
        │                                                        ├──► (box, label, score)
        └──► proposals (Canny×2 + morphology + Otsu, NMS) ───────┘
             each with a confidence from contrast / texture / size prior
```

---

## Layout

```
src/
  config.py        dataset registry, split/feature/bin/model/train configs
  duplicates.py    dHash + correlation matching; intra- and cross-dataset audits
  splits.py        image-level stratified 3-way split, duplicate grouping, manifests
  features.py      92 descriptors + image-space augmentation + cached batch extraction
  discretize.py    train-fitted binning, OOR policy and counters
  prompt.py        tokens, order modes (canonical/shuffled/per-sample), tokenizer
  model.py         SteelSenseBiLSTM + TokenDeepSets (order-free control)
  engine.py        training loop, val-only selection, snapshot ensemble, single test touch
  metrics.py       macro-F1, balanced acc, per-class, bootstrap CI, McNemar, paired tests
  pipeline.py      assembles the split into tensors in the legal order
  experiments.py   multi-seed run, order ablation, bin sweep, baseline comparison
  baselines_tabular.py   XGBoost / RF / MLP / LogReg / SVM on the raw vector
  baselines_cnn.py       MobileNetV3-Small / ShuffleNetV2 / MobileNetV2 / ResNet18
  complexity.py    params, MACs (thop + analytic LSTM), batch-size-1 latency, hardware
  interpret.py     per-class attention, permutation importance, and their agreement
  localize.py      proposal detector with confidences, VOC AP, IoU histogram, oracle
  profile_stages.py  16-stage latency breakdown + measured thread scaling
```

## Requirements

`torch`, `torchvision`, `numpy`, `scipy`, `scikit-learn`, `scikit-image`, `opencv-python`,
`pandas`, `matplotlib`, `xgboost`, `thop`, `nbformat`. All present in the environment this
was built in; CPU-only is fine and is what the latency numbers were measured on.
