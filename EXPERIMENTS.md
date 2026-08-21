# Experiment Suite — what is implemented, what is measured

**Companion to** [`PAPER_GUIDE.md`](PAPER_GUIDE.md) · **Code:** [`new_model_code.ipynb`](new_model_code.ipynb)
**Regenerate LaTeX:** `python make_tables.py` → [`paper_tables.tex`](paper_tables.tex)

| Marker | Meaning                                                      |
| ------ | ------------------------------------------------------------ |
| ✅     | Implemented**and** measured. Safe to publish.          |
| 🔧     | Implemented and verified to build,**not yet trained**. |
| ⚠️   | A result that constrains what the paper may claim.           |

> Every parameter count below was obtained by building the model. Every accuracy
> and AP was obtained by running it. Nothing here is estimated.

---

## Section 5 structure

```
5.1  Computational Complexity
5.2  Comparison with Representative CNN and Lightweight Models
5.3  Comparison with Existing Attention Mechanisms
5.4  Ablation Study
     5.4.1  Component-wise Ablation
     5.4.2  CSAF-specific Ablation
5.5  Per-Class Performance and Confusion Analysis
5.6  CAM-based Localization Results
5.7  Error Analysis and Limitations
```

---

## 5.1 Computational Complexity ✅

Batch 1, 200×200×3, CPU-only (AMD64, 2 physical cores, no GPU), median of 200 timed
runs after 20 warm-up runs. Model size is fp32 parameter memory, **not** checkpoint
size — a checkpoint saved after training also carries Adam's moment slots and
measures 33.3 MiB against 12.8 MiB of actual weights.

| Configuration                     | Params              | MACs (M)       | Size (MiB)      | Latency (ms)    | p95 (ms)         | CPU FPS        |
| --------------------------------- | ------------------- | -------------- | --------------- | --------------- | ---------------- | -------------- |
| MobileNetV2 backbone only         | 2,423,110           | 270            | —              | —              | —               | 27.3           |
| Proposed, 32 pyramid ch.          | 2,382,680           | 349            | —              | —              | —               | 19.9           |
| Proposed, 64 pyramid ch.          | 2,607,400           | 563            | —              | —              | —               | 17.5           |
| **Proposed, 128 (default)** | **3,364,064** | **1373** | **12.83** | **97.58** | **215.87** | **10.2** |
| Proposed detector                 | 3,966,821           | —             | —              | —              | —               | —             |

Full model: 2746.34 MFLOPs = 1373.17 MMACs. State which convention you report.

⚠️ **The pyramid dominates cost, not the backbone.** It adds only +0.94 M parameters
but **5× the MACs**, because AMFF and the 3×3 smoothing convolutions run at P2
resolution (50×50). Quoting "3.4 M parameters, lightweight" and stopping there tells
less than half the story — and that is exactly what Reviewer 4 (c7) was probing.

⚠️ **10 FPS on CPU is not real-time.** Drop the claim, or report it against hardware
where it holds. Note `fpn_channels=32` costs about the same as the bare backbone.

> Re-measure on an unloaded machine. The same configuration measured 27.3 and 16.3
> FPS on two separate runs during development because other work was running.

---

## 5.2 Comparison with Representative Models 🔧

Ten backbones, three families. All trained by us on the identical frozen split with
the same augmentation, schedule and seed; each receives its **own canonical
`preprocess_input`**, so no baseline is handicapped by a preprocessing mismatch.

| Model                  | Init              | Params     | Accuracy  | Macro-F1  | MACs      | Latency   | FPS       |
| ---------------------- | ----------------- | ---------- | --------- | --------- | --------- | --------- | --------- |
| ResNet50               | ImageNet          | 23,600,006 | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| VGG16                  | ImageNet          | 14,717,766 | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| DenseNet121            | ImageNet          | 7,043,654  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| EfficientNetV2B0       | ImageNet          | 5,926,998  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| EfficientNetB0         | ImageNet          | 4,057,257  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| MobileNet              | ImageNet          | 3,235,014  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| MobileNetV3Large       | ImageNet          | 3,002,118  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| MobileNetV2            | ImageNet          | 2,265,670  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| **ResNet18**     | **scratch** | 11,189,190 | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| **ShuffleNetV2** | **scratch** | 1,275,934  | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` | `[RUN]` |
| **Proposed**     | ImageNet          | 3,364,064  | `[RUN]` | `[RUN]` | 1373      | 97.58     | 10.2      |

### ⚠️ The `Init` column is not optional

ResNet18 and ShuffleNetV2 ship **no ImageNet weights** in `tf.keras.applications`, so
they start from random initialisation while every other row starts from ImageNet.
That is a real disadvantage. Reporting them next to pretrained rows without saying so
is the same class of unfair comparison Reviewer 4 objected to.

Both are faithful re-implementations — ResNet18 as 2-2-2-2 BasicBlocks, ShuffleNetV2
1.0× with channel-split and channel-shuffle units. Parameter counts (11.19 M and
1.28 M) match the published architectures, which is the check that they are right.

One implementation detail worth stating: for scratch rows the backbone is **not
frozen** in phase 1. Freezing a randomly initialised backbone would leave only the
classifier head able to learn.

**RepViT and EdgeNeXt are absent.** They have no Keras port. `timm 1.0.19` is
installed and does provide them pretrained, but in PyTorch — adding them would mean
maintaining a second training framework. Say this in the paper rather than leaving a
silent gap.

---

## 5.3 Comparison with Attention Mechanisms 🔧

Only the block at the FPN lateral-fusion point changes. Backbone, pyramid, CSAF,
SEAM, partition, schedule and seed are all held fixed.

| Attention                          | Params    | Δ vs CBAM        | Accuracy  | Macro-F1  | AP50      |
| ---------------------------------- | --------- | ----------------- | --------- | --------- | --------- |
| None (concat + 1×1)               | 3,301,895 | −13,017          | `[RUN]` | `[RUN]` | `[RUN]` |
| ECA (Wang et al., 2020)            | 3,301,910 | −13,002          | `[RUN]` | `[RUN]` | `[RUN]` |
| SE (Hu et al., 2018)               | 3,314,615 | −297             | `[RUN]` | `[RUN]` | `[RUN]` |
| CBAM (Woo et al., 2018)            | 3,314,912 | —                | `[RUN]` | `[RUN]` | `[RUN]` |
| Coordinate Att. (Hou et al., 2021) | 3,321,287 | +6,375            | `[RUN]` | `[RUN]` | `[RUN]` |
| EMA (Ouyang et al., 2023)          | 3,794,951 | +480,039          | `[RUN]` | `[RUN]` | `[RUN]` |
| **AMFF (ours)**              | 3,364,064 | **+49,152** | `[RUN]` | `[RUN]` | `[RUN]` |

⚠️ **AMFF is not parameter-matched.** It feeds three tensors into the fusion
convolution where CBAM feeds two, so its 1×1 is 1.5× wider — **+49,152 parameters**.
A small AMFF win may simply be the extra capacity. If AMFF leads CBAM by less than
the multi-seed standard deviation, report it as *"comparable to CBAM at slightly
higher cost"*. That is the defensible reading.

Note EMA is by far the heaviest row (+480 K over CBAM); if it wins, its cost must be
reported alongside.

---

## 5.4.1 Component-wise Ablation ✅ (partly)

| Variant              | FPN | AMFF | CSAF | SEAM | Params    | Accuracy         | AP50      | mAP       |
| -------------------- | --- | ---- | ---- | ---- | --------- | ---------------- | --------- | --------- |
| MobileNetV2 baseline | –  | –   | –   | –   | 2,423,110 | **100.00** | 8.34      | 3.70      |
| + FPN                | ✓  | –   | –   | –   | 3,196,102 | 99.72            | 7.83      | 3.26      |
| + FPN + AMFF         | ✓  | ✓   | –   | –   | 3,358,111 | 99.17            | 9.12      | 4.12      |
| + FPN + CSAF         | ✓  | –   | ✓   | –   | 3,196,678 | `[RUN]`        | `[RUN]` | `[RUN]` |
| + FPN + AMFF + CSAF  | ✓  | ✓   | ✓   | –   | 3,358,687 | 98.61            | 9.14      | 4.58      |
| Full model (+ SEAM)  | ✓  | ✓   | ✓   | ✓   | 3,364,064 | 99.72            | 9.64      | 3.99      |

*(3+2 epochs, single seed, test split. `+ FPN + CSAF` is a newly added row.)*

### ⚠️ Two things this table already establishes

**1. Accuracy is saturated. The ablation cannot use it.** The bare backbone reaches
**100.00 %** on the held-out test split, and every added module makes accuracy
slightly *worse*. A pilot over 12 architectural variants put 10 of them at exactly
100 % validation accuracy. There is no headroom, so no module can demonstrate a
contribution on accuracy. **Report ablations on AP50** and say why.

**2. CSAF contributes +0.02 AP50 over AMFF.** The central novelty claim currently has
no experimental support. Section 5.4.2 exists to test it properly; if that fails too,
withdraw the claim rather than defend it — defending it is what Reviewer 2 objected to.

Note FPN alone *reduces* AP50 (8.34 → 7.83). Only with AMFF does it recover.

---

## 5.4.2 CSAF-specific Ablation 🔧 — the novelty test

This is the most important new experiment. Each step changes **exactly one property**
of the fusion rule, so the paper's claim is isolated rather than bundled.

| Fusion strategy             | Params    | Accuracy  | Macro-F1  | AP50      | What the step tests                   |
| --------------------------- | --------- | --------- | --------- | --------- | ------------------------------------- |
| FPN + addition              | 3,297,952 | `[RUN]` | `[RUN]` | `[RUN]` | classic FPN rule                      |
| FPN + concatenation         | 3,363,488 | `[RUN]` | `[RUN]` | `[RUN]` | fixed proportion                      |
| FPN + fixed learned weights | 3,297,956 | `[RUN]` | `[RUN]` | `[RUN]` | ← does*learning* it help?          |
| FPN + per-sample gating     | 3,314,912 | `[RUN]` | `[RUN]` | `[RUN]` | ← does*per-sample* help?           |
| **FPN + CSAF (ours)** | 3,364,064 | `[RUN]` | `[RUN]` | `[RUN]` | ←**cross-scale + competition** |

How to read it:

| Comparison              | Isolates                                                        |
| ----------------------- | --------------------------------------------------------------- |
| concat → fixed         | Is a**learned** proportion better than a fixed one?       |
| fixed → local          | Does making it**per-sample** help?                        |
| **local → csaf** | **Does cross-scale input plus softmax competition help?** |

**The last row is the paper's actual claim.** `local` is adaptive per-sample gating
where each level's gate comes from *its own* descriptor with a sigmoid — adaptive,
but neither cross-scale nor competitive. If `local` matches `csaf`, the contribution
is ordinary gating and the cross-scale claim must go.

This also gives a clean contrast with the closest prior work: MobileSteelNet's MSFF
is `interpolate → concat → 1×1`, i.e. exactly the `concat` row — a proportion fixed
at design time and identical for every image.

`_interpret_ablation()` prints all three deltas automatically.

---

## 5.5 Per-Class Performance and Confusion Analysis ✅

Confusion matrices for all five ablation variants:
`paper_results/figures/ablation_confusion_grid.png` (+ `.pdf`), raw counts in
`ablation_confusion_matrices.json`.

| Variant    | Errors / 360 | Confusions                                         |
| ---------- | ------------ | -------------------------------------------------- |
| Baseline   | **0**  | none — perfect                                    |
| + FPN      | 1            | Inclusion → Pitted surface                        |
| + AMFF     | 3            | Scratches → Inclusion                             |
| + CSAF     | 5            | Crazing → Patches (1), Inclusion → Scratches (4) |
| Full model | 1            | Patches → Crazing                                 |

⚠️ **Do not present this as ablation evidence.** The baseline is already perfect and
errors *increase* as modules are added. Its honest use is as **saturation evidence**:

> *"A MobileNetV2 baseline already classifies the held-out test split without error
> (Figure X); classification accuracy therefore cannot discriminate between
> architectural variants on NEU-DET."*

That supports the metric choice in 5.4 and is a methodological point in your favour.

---

## 5.6 CAM-based Localization Results ✅

Grad-CAM → threshold → morphology → connected components → boxes → scored regions.
No box supervision at training time. Evaluated with the **same** metric code and the
same verified VOC ground truth as the detector.

| Method                          | Supervision       | Params    | AP50            | AP75            | mAP50:95        |
| ------------------------------- | ----------------- | --------- | --------------- | --------------- | --------------- |
| CAM-based localization          | image labels only | 3,364,064 | 9.71            | 3.62            | 4.30            |
| **Detection head (ours)** | bounding boxes    | 3,966,821 | **64.77** | **18.89** | **27.20** |

Per-class AP50:

| Class           | Detection | CAM-based |
| --------------- | --------- | --------- |
| Pitted surface  | 84.8 | 57.6      |
| Scratches       | 77.7 | 0.3       |
| Patches         | 84.8 | 0.1       |
| Inclusion       | 65.1 | 0.0       |
| Rolled-in scale | 43.3 | 0.0       |
| Crazing         | 35.1 | 0.1       |

**Three of the five classes that scored zero without box supervision now exceed 80.**

⚠️ **Report AP75 and mAP alongside AP50.** AP75 (18.89) is far below AP50 (64.77):
boxes are being found but their edges are imprecise — the signature of under-trained
box regression. Quoting AP50 alone hides it, and a reviewer who asks for AP75 will
find it. *(18+22 epochs, cosine decay, deeper backbone unfreeze. The earlier 12+8-epoch
constant-rate run scored AP50 63.88 / AP75 17.02 / mAP 27.01.)*

⚠️ Precision at the reported operating point is 2.25 % with recall 95.4 %, because
the score threshold (0.02) was selected **on validation** to maximise AP. That is
correct for AP, but if the paper quotes precision/recall it must choose and state a
sensible operating point instead.

---

## 5.7 Error Analysis and Limitations ✅

### Why CAM-based localization fails, quantified

| Class                    | Median GT box (% of image) | Boxes > 50 % of image | Boxes/image | CAM AP50       |
| ------------------------ | -------------------------- | --------------------- | ----------- | -------------- |
| **Pitted surface** | **55.5**             | **58.3 %**      | 1.63        | **57.6** |
| Crazing                  | 21.1                       | 1.2 %                 | 2.31        | 0.1            |
| Rolled-in scale          | 12.3                       | 0.2 %                 | 2.09        | 0.0            |
| Patches                  | 9.3                        | 0.3 %                 | 2.91        | 0.1            |
| Scratches                | 7.5                        | 0.0 %                 | 2.07        | 0.3            |
| Inclusion                | 4.1                        | 0.0 %                 | 2.95        | 0.0            |

**AP tracks ground-truth box size almost perfectly.** The one class that works is the
one where "highlight everything" is the right answer. For the other five, defects are
small and multiple, and a class activation map cannot separate instances.

Tested, not assumed: a **72-configuration** sweep on validation (thresholds, minimum
area, morphology kernel, Otsu) moved AP50 by **−0.01**. The limitation is the
supervision signal, not the post-processing.

> *"CAM-based localization succeeds only where the annotation approaches full-frame
> extent. Its per-class AP correlates with median ground-truth box area, indicating
> that class activation maps cannot resolve the small, multi-instance defects that
> dominate NEU-DET."*

### Inclusion ↔ Pitted surface

The reference NEU-DET literature reports heavy Inclusion/Pitted confusion. **We do
not observe it in classification** — our baseline makes zero errors. The reason is
protocol, not architecture: that work trains on 90 images from scratch, we train on
1260 with ImageNet initialisation.

It *does* appear in localization, where the two are the extremes of the box-size
distribution (4.1 % vs 55.5 % of the frame), and detection AP splits accordingly
(67.2 vs 83.3). That is the honest place to discuss it.

### Stated limitations

1. Classification is saturated on NEU-DET; the dataset cannot discriminate between
   architectural variants on accuracy.
2. AP75 lags AP50 substantially — box regression is under-trained on the available
   compute (2 CPU cores, no GPU).
3. CAM-based localization is viable only for large, single-instance defects.
4. Measured throughput does not support a real-time claim on CPU.

---

## What still has to be run

| Experiment                        | Runs         | Est. on this CPU |
| --------------------------------- | ------------ | ---------------- |
| 5.2 backbone comparison           | 10           | ~4 h             |
| 5.3 attention comparison          | 7            | ~3 h             |
| 5.4.1 missing`+ FPN + CSAF` row | 1            | ~25 min          |
| 5.4.2 CSAF fusion strategies      | 5            | ~2 h             |
| Multi-seed final model (5 seeds)  | 5            | ~2 h             |
| YOLOv8n / YOLO11n / YOLO12n       | 3            | ~2 h             |
| **Total**                   | **31** | **~13 h**  |

```python
# in the notebook, after Section 13
CLS = run_classification_comparison(seed=42)
ABL = run_ablation(seed=42, with_localization=True, studies=("A","B","C","D","E"))
YOL = run_yolo_baselines()
```

then `python make_tables.py` to fill `paper_tables.tex`.

⚠️ **On a GPU this is roughly 30–40 minutes rather than 13 hours.** The notebook is
portable and needs no changes. Given the number of runs still outstanding, that is
the difference between finishing today and finishing next week.

---

## The honest state of the contribution

Supported by measurement right now:

1. A reproducible evaluation protocol — frozen split, verified VOC ground truth only,
   classification and localization reported separately.
2. Classification saturates on NEU-DET, so architectural ablations must be judged on
   localization metrics. Demonstrated over 12 variants.
3. An explanation of when CAM-based localization works and when it cannot, quantified
   against ground-truth box statistics.
4. A lightweight anchor-free detector: **AP50 64.77** at 3.97 M parameters, versus
   9.71 without box supervision.

Not yet supported:

5. CSAF as a contribution. Currently +0.02 AP50. Section 5.4.2 is the test that
   decides it, and it has not been run.
6. AMFF as a contribution. Not parameter-matched to its controls; Section 5.3 decides.

Items 1–4 are a publishable paper on their own. Items 5–6 are claims awaiting
evidence — write them only if the evidence arrives.
