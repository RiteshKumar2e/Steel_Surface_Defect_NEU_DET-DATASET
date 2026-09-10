# Paper revision notes — NEU-DET, mapped to real measured results

Source of every number below: `SteelSenseV2/01_NEU_DET_SteelSense_BiLSTM.ipynb`, fully executed.
Each row: where in the current manuscript (`SteelSense_BiLSTM___Ritesh___24_07.pdf`) to edit,
what is there now, what to replace it with, and which decision-letter point it answers.

The SteelDefectX section is at the bottom of this file — **now fully complete** (as of
2026-09-10): main run, all baselines, both ablations, bin sweep, interpretability, latency/
thread-scaling, and the NEU-overlap section are all done, every number real and verified.

---

## 1. Abstract (p.1) and Contributions (p.4)

**Now:** claims HPC/GPU acceleration and localization as headline contributions; implies the
BiLSTM+discretization design is demonstrably better than alternatives.

**Change to:** soften both claims. The new ablation evidence (see §6 below) does not support
architectural superiority — RandomForest/SVM on the raw vector match or beat the BiLSTM, and a
numeric (non-discretized) version of the same BiLSTM beats the tokenized version. Reframe the
contribution as "a compact, interpretable classifier at accuracy comparable to standard ML
baselines, with a rule-based localization fallback" rather than "significantly outperforms."
Drop "HPC-enabled" / "GPU-accelerated" language or replace with "parallelized, lightweight"
(see §7 below — no GPU was used for these numbers).

**Reviewer point:** Major Issue 6 (ablations), Major Issue 7 (HPC wording), Recommendation
paragraph (localization/title scope).

---

## 2. Table 1 (p.9) — architecture / model summary

**Now:** `Input Tokens 160`, `Total Parameters: 1,752,775`, `Model Size: 7.89 MB`,
`Computational Complexity: 213.74 M MACs`.

**Change to (measured):**
- Sequence length: **94**, not a fixed 160 — it is computed dynamically as
  `n_features + 2` (92 descriptors + CLS + margin). Rewrite the "Input Tokens" row and drop
  the fixed-160 framing everywhere it appears (also fixes Major Issue 5's "reason for using a
  sequence length of 160" question — the honest answer is that 160 was never actually used).
- Total Parameters: **1,916,839**
- Model Size: **7.312 MB**
- Computational Complexity: **258.048 M MACs** (measured the same way as the CNN baselines,
  via `thop` + an analytic LSTM term — cite this methodology, it's what Reviewer 1 Q5 asked for)
- Vocabulary: **647 tokens** (5 bins × 92 features × quantile edges + OOR symbols)

**Reviewer point:** Major Issue 5 (reproducibility — vocab/seq-len/params must match what the
code actually produces).

---

## 2b. §3.2.2 Handcrafted Feature Extraction — exact descriptor parameters (Reviewer 2 Q5)

**Now:** manuscript says "fixed distances and directions defined in the experimental pipeline"
and "fixed implementation settings" without giving the actual numbers. This is the specific
phrasing an external audit flagged as still not reproducible — replace it with the real values
below, sourced directly from `src/config.py:FeatureConfig` and `src/features.py`.

**GLCM.** Distances **{1, 3} pixels**, directions **{0°, 45°, 90°, 135°}**, **32** gray levels.
Six Haralick properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM),
aggregated across the four directions as the **mean** (rotation-invariant summary) and the
**range** (max − min, an anisotropy measure that separates scratches from crazing). 2 distances
× 6 properties × 2 aggregates = **24 features**.

**LBP.** **P = 8** sampling points, radius **R = 1**, method **"uniform"**
(`skimage.feature.local_binary_pattern`). Aggregated as a **10-bin** (P+2) normalized histogram
plus 2 summary statistics (Shannon entropy of the histogram; uniformity = sum of the first P+1
bins). **12 features**.

**Contour selection rule.** Otsu inverse-binary threshold → morphological **open then close**
(3×3 elliptical kernel, 1 iteration each) → **external contours only** (`cv2.RETR_EXTERNAL`) →
discard any contour smaller than **max(8 px², 0.00015 × H × W)**. Per-contour descriptors: area,
aspect ratio, circularity, solidity, extent, and a local-saliency score (contrast against a 6-px
surrounding ring). **13 features** (including whole-image aggregates: count, coverage, etc.).

**Frequency-domain (2D FFT).** Magnitude spectrum, **4 radial bands** at normalized-radius edges
[0, 0.1, 0.25, 0.5, 1.0], plus **1 angular-anisotropy scalar** ((max−min)/mean energy across 4
angular sectors). **5 features**.

**The other two families** (already stated correctly, repeated here for one complete spec):
statistical/intensity (mean, std, skew, kurtosis, entropy, p10/p50/p90, IQR, range-ratio,
dark/bright fraction, Michelson contrast — 13 features) and edges (Canny at two threshold pairs
30/90 and 80/200, Sobel gradient mean/std/p95, Laplacian variance, orientation coherence +
dominant angle, horizontal/vertical asymmetry — 10 features), plus spatial (4 quadrants ×
{std, edge density} + edge-spread/CV + row/col peak position + row/col profile CV — 14
features). Total: 13+24+12+10+13+5+14+1 = **92 features**, matching Table 1 and every other
count already stated in this document.

**Drop-in replacement sentence for §3.2.2:** *"GLCM texture descriptors were computed at pixel
distances {1, 3} and orientations {0°, 45°, 90°, 135°} with 32 gray levels, aggregated across
orientations as the mean (rotation-invariant) and range (anisotropy) of six Haralick properties
(contrast, dissimilarity, homogeneity, energy, correlation, ASM). LBP descriptors used P = 8
sampling points at radius R = 1 with the uniform method, aggregated as a 10-bin normalized
histogram plus entropy and uniformity summary statistics. Candidate defect regions were obtained
via Otsu thresholding followed by morphological opening and closing (3×3 elliptical kernel),
retaining external contours with area exceeding max(8 px², 0.00015 × H × W). Frequency-domain
descriptors comprised four radial energy bands of the 2D FFT magnitude spectrum plus one
angular-anisotropy statistic."*

**Reviewer point:** Reviewer 2, Q5 ("GLCM distances, directions and quantization levels; the LBP
radius, sampling points and aggregation method; contour-selection rules; and any
frequency-domain descriptors") — this closes the specific gap an external audit found still
open after the first pass at this section.

---

## 3. §3.1.1 Datasets Description / §4.1.2 Training Configuration (p.10, p.21)

**Now:** "80% for training and 20% for validation set by a random split... The five checkpoints
with the highest accuracy during the validation... trained... for 30 epochs."

**Change to:**
- Split is **70/10/20** (train/val/test), image-level, stratified, with near-duplicate images
  grouped so a group never straddles the train/test boundary (dHash + correlation matching).
  Real counts: **train 1260, val 180, test 360** (60 images/class on test).
- Checkpoint selection is by **validation macro-F1**, not accuracy (macro-F1 is what should be
  optimized given class balance is fine here but matters generally — and it's what the code
  actually does).
- The test split is touched **exactly once**, after checkpoint selection is frozen.
- Report the frozen-split caveat: the previously-used `split_v1.csv` places one duplicate group
  on both sides of the boundary; the new split re-splits around this rather than reusing it.
- Report seeds explicitly: **[42, 1337, 2024, 7, 20250101]**.
- Epochs actually used: 40 (with early stopping, patience 15) — not 30. State the real value or
  re-run at 30 to match the text; the numbers below assume the as-run configuration.

**Reviewer point:** Major Issue 2 (partition/selection protocol) — this is the single most
load-bearing fix; almost every other number downstream depends on this being stated correctly.

---

## 4. §4.1.1 Hardware and Software Configuration (p.21)

**Now:** "workstation with an Intel Core i7 processor and 16GB of RAM with GPU that includes
8GB of memory... GPU acceleration through the CUDA framework." Doesn't match either machine
below — needs replacing either way, not just softening.

**There are genuinely two machines involved, and the write-up must keep them separate rather
than blend them:**

1. **Primary development workstation** (author-confirmed): Intel Core i9-13900K, 128 GB RAM,
   NVIDIA GeForce RTX 3060. State this as the machine used for original model development and
   the main training runs.
2. **CPU-only verification machine** (what actually produced the specific latency/throughput/
   thread-scaling numbers in this revision, measured directly): AMD64 Family 23 Model 24
   (AuthenticAMD), 2 physical cores / 4 logical cores, 10.6 GB RAM, `cuda_available: False`,
   PyTorch 2.12.0+cpu, `torch_threads=2`, Windows 11.

**Do not attribute machine 2's numbers to machine 1.** The HPC/latency table (§9, §13) —
end-to-end latency, throughput, and the 1/2/4/8-thread speedup curve — must be captioned as
CPU-only measurements from machine 2, not the RTX 3060 workstation. This is not a weakness to
hide: report it as a deliberate additional check ("to verify the framework's claimed
lightweightness does not depend on GPU acceleration, the full latency and thread-scaling
protocol was additionally run CPU-only on [machine 2]") — a model that stays fast without a GPU
is a *stronger* efficiency claim than one that only works with one, and it directly answers
Reviewer 1 Q3 / Reviewer 2 Q8's request for CPU results, not just GPU ones.

If GPU-side latency/throughput numbers are wanted for machine 1 too, that requires an actual
CUDA-enabled rerun of `complexity.py`/`profile_stages.py` on that workstation (the current code
never moves tensors to `cuda`, so this needs a small device-handling addition first, not just a
different machine) — flag as a possible additional table rather than backfilling numbers that
were never measured there.

**Reviewer point:** Major Issue 7 ("report the processor model and core count, GPU model...").

---

## 5. Table 4 (p.22) — SOTA comparison, classification + localization mixed

**Now:** one table mixing YOLOv8/MHSVM+/EDDN/DIN/DL (all values from other papers, other
datasets/protocols) with SteelSense-BiLSTM's mAP/AP50/AP75/Acc.

**Change to:** split into two tables.

**(a) Classification table — every value measured here, same split, same protocol:**

| Model | Representation | Accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| RandomForest | raw 92-d descriptor vector | 1.0000 | 1.0000 | 114,464 |
| SVM-RBF | raw 92-d descriptor vector | 1.0000 | 1.0000 | 73,235 |
| XGBoost | raw 92-d descriptor vector | 0.9972 | 0.9972 | 14,610 |
| MLP | raw 92-d descriptor vector | 0.9972 | 0.9972 | 57,478 |
| LogisticRegression | raw 92-d descriptor vector | 0.9972 | 0.9972 | 558 |
| MobileNetV3-Small | image (pretrained) | 0.9972 | 0.9972 | 1,524,006 |
| ShuffleNetV2-x0.5 | image (pretrained) | 1.0000 | 1.0000 | 347,942 |
| **SteelSense-BiLSTM** | **binned descriptors as text prompt** | **0.9967 ± 0.0023** | **0.9967 ± 0.0023** | **1,916,839** |

State explicitly: strongest baseline is RandomForest (macro-F1 1.0000); McNemar's exact test
vs. SteelSense-BiLSTM gives **p = 1.0 ("identical errors")** — no statistically significant
difference from the simplest classical baselines. This must replace any "significantly
outperforms" language in §4.5.

**(b) Literature detectors** (YOLOv8, MHSVM+, EDDN, DIN, DL) — keep as a separate table,
explicitly labeled "values as reported in the cited publications, not reproduced under this
protocol, not directly comparable."

**Reviewer point:** Major Issue 3 (separate classification/localization, label the source of
every value, common protocol wherever possible).

---

## 6. Table 4 localization columns / §3.2.5 / §4.5 — the central issue

**Now:** mAP 0.8254, AP50 0.8935, AP75 0.7618 for SteelSense-BiLSTM (NEU-DET), presented as
detection performance.

**What actually produced numbers in that range:** the **[ORACLE]** condition — ground-truth
XML boxes carrying the predicted class label. Measured: **AP50 = 94.51, AP75 = 94.51,
mAP = 94.51** (identical at every IoU threshold because IoU = 1 by construction — this
threshold-flatness is exactly the signature Reviewer 2 named). The paper's reported values are
close enough to this pattern (and nowhere near the real detector, below) that this is almost
certainly the mechanism.

**The real, honestly-measured detector** — region proposals from Canny + Otsu + morphology,
each carrying a confidence built from measurable region evidence (never touching the XML at
inference), scored with standard VOC AP:

| | mAP@[.5:.95] | AP50 | AP75 |
|---|---|---|---|
| **[DET] real detector** | **1.62%** | **5.51%** | **0.61%** |
| [ORACLE] GT box + predicted label (ceiling, not detection) | 94.51 | 94.51 | 94.51 |

Additional honest localization numbers to report: mean IoU **0.1376**, median IoU 0.0125, only
**10.6%** of test images reach IoU ≥ 0.5, 3.06% reach IoU ≥ 0.75, proposal fallback rate 0%,
mean 8.48 proposals/image.

**Change to:**
- Replace Table 4's localization row with the real [DET] numbers. Report [ORACLE] only in a
  clearly-labeled footnote/appendix as a ceiling, never as the headline figure.
- Rewrite §3.2.5 to describe the actual inference procedure: contour/Canny/Otsu proposals →
  confidence from local contrast + edge density + area prior → NMS → confidence × class
  probability ranks the detections → VOC AP. State plainly this is a rule-based fallback with
  no trainable localization head (matching Table 1, which has no bounding-box regression layer).
- Add a qualitative figure (replacing/supplementing Fig. 4) showing real predicted boxes
  (distinct color) against ground truth (distinct color/style), including the **worst**-IoU
  cases, not only successes — the notebook already generates this (3 best + 3 worst by IoU,
  picked automatically).
- Revisit the title/abstract/conclusion claims of "Localization" given AP50 = 5.5% — per the
  reviewers' own suggested fallback, consider reframing as classification with an auxiliary,
  rule-based region-visualization module, unless a trainable detector is added.

**Reviewer point:** Major Issue 1 — this is the decision letter's first and most consequential
point, and the one the Recommendation paragraph says could force a title/scope change if not
resolved.

---

## 7. Table 5 (p.24) — per-class metrics, the F1 arithmetic error

**Now (NEU-DET half):** Patches P=1.000, R=0.978, **F1=0.976** (harmonic mean of 1.000/0.978 is
≈0.989, not 0.976 — this is the exact discrepancy Major Issue 4 names).

**Change to (measured, 5-seed mean ± SD, macro-averaged, arithmetically verified consistent):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Crazing | 1.000 | 0.990 | 0.995 |
| Inclusion | 0.997 | 0.997 | 0.997 |
| Patches | 0.993 | 0.997 | 0.995 |
| Pitted Surface | 0.997 | 0.997 | 0.997 |
| Rolled-in Scale | 1.000 | 1.000 | 1.000 |
| Scratches | 0.994 | 1.000 | 0.997 |

Overall: accuracy **0.9967 ± 0.0023**, macro-F1 **0.9967 ± 0.0023** (range 0.9944–1.0000 across
5 seeds); test set is 60 images/class (360 total). Seed-averaged ensemble prediction reaches
1.0000/1.0000 with a bootstrap 95% CI of (1.0000, 1.0000) — **state explicitly that this
collapses to a point only because the test set is small (360 images)**, not because the true
error rate is zero; don't let a reviewer flag this as another "too good to be true" number.

State the metric definition explicitly: macro-averaged (not micro/weighted), and cite that
values come from `sklearn.metrics.precision_recall_fscore_support`, computed once and never
hand-transcribed — this is what prevents the arithmetic-error class of bug from recurring.

**Reviewer point:** Major Issue 4.

---

## 8. §4.6 Discussion, first two paragraphs (p.25) — the ablation claim

**Now:** "A significant observation from the experimental validation is that the feature prompt
based representation... is a good idea..." and "This proposed strategy of multi-view pooling
boosts high quality features still higher."

**This claim is not supported by the new evidence — replace with the real ablation.** Six
conditions, same split/seeds/epochs/checkpoint rule, only one thing changed per row:

| Condition | Macro-F1 | vs. deployed (paired test) |
|---|---|---|
| Full (5-snapshot ensemble, 3-way pooling, tokenized) — **deployed** | 0.9972 | — |
| No ensemble (single best checkpoint) | 0.9963 | p = 0.42 (n.s.) |
| Attention pooling only | 0.9972 | p = 1.0 (n.s.) |
| Max pooling only | 0.9991 | p = 0.53 (n.s.) |
| Mean pooling only | 0.9954 | p = 0.67 (n.s.) |
| **No discretization** (raw standardized descriptors, same encoder, no tokenizer at all) | **1.0000** | p = 0.23 (n.s.) |

None of the differences are statistically significant, and the **no-discretization control is
the single best-performing condition** — the text-prompt discretization step does not
demonstrably help over feeding the same continuous descriptors through the identical BiLSTM.
Rewrite §4.6 to report this honestly: the architecture's value is in compactness and
interpretability at *comparable* — not superior — accuracy to simpler alternatives.

Also add the order-ablation result (already partly in the repo, ties to Reviewer 3 Q7): shuffled
token order and a permutation-invariant DeepSets encoder both perform statistically
indistinguishably from the canonical BiLSTM (Wilcoxon p = 0.5 across all three comparisons,
t-test p > 0.18) — so the recurrent encoder is not earning its parameters over a cheaper
order-free alternative either, and this should be stated rather than the current framing that
implies the BiLSTM is necessary.

**Reviewer point:** Major Issue 6, and Reviewer 3 Q7 (BiLSTM choice must be justified against
the shuffle/order-free result, not asserted).

---

## 9. §4.4 Computational Performance Analysis / HPC claims (p.23, and throughout)

**Now:** generic claims of parallel speed-up, GPU acceleration, and "extremely efficient
computationally," with no numbers.

**Change to (measured):**
- End-to-end latency, batch size 1, CPU: **67.4 ms single-checkpoint inference**, **156.3 ms**
  with the 5-member snapshot ensemble, **170.2 ms** including the localization proposal step.
  Throughput ≈ **14.8 images/s** (single checkpoint). Inference is **68.5%** of end-to-end cost
  — the "small model = fast" framing is misleading once descriptor extraction and the ensemble
  are counted; report `end_to_end_*`, not the parameter count, as the deployment-relevant number.
- Per-stage breakdown (median ms): LSTM ensemble 107.0, LSTM single 18.2, localization
  proposals 14.0, LBP 11.9, contours 8.0, edges 6.4, intensity 6.3, spectral 5.6, GLCM 5.4,
  DeepSets forward 4.7 (i.e., the BiLSTM is **3.89×** the cost of the order-free control —
  read this next to the order-ablation's non-significant accuracy gap: the recurrent encoder is
  paying a real latency cost for an accuracy difference that isn't statistically real).
- Thread scaling (the literal "1, 4, 8 CPU workers" request), measured on real feature
  extraction: **1 thread → 19.2 img/s (baseline)**, **2 → 29.4 img/s (1.53×, 76% efficient)**,
  **4 → 26.8 img/s (1.40×, 35% efficient)**, **8 → 29.8 img/s (1.55×, 19% efficient)**. Scaling
  plateaus almost immediately because the test machine has only 2 physical cores — report this
  curve exactly as measured (including the non-monotonic dip at 4 threads) rather than a
  smoothed or asserted claim.

**Change the section's framing** from "HPC-enabled... GPU-accelerated" to "parallelized on
CPU; speed-up is bounded by physical core count, measured explicitly" — matching the decision
letter's own suggested fallback wording almost verbatim.

**Reviewer point:** Major Issue 7 (this is a direct, item-by-item answer to every sub-request:
preprocessing/feature-extraction throughput, inference latency, throughput, speed-up vs.
serial, hardware detail).

---

## 10. Bin-count / discretization threshold (Major Issue 5, second half) — good news, cite it

**Now:** thresholds and bin count are unexplained/unjustified ("Low/Medium/High" style choice
in the original submission).

**Change to:** cite the bin-count sweep as the justification. Quantile edges fit on the
**training split only**; swept 3/5/7/10 bins × {quantile, uniform}. The deployed default
(5 bins, quantile) is empirically the best of the tested configurations: macro-F1 **1.0000 ±
0.0000** (2 seeds), matching or beating every other combination tested. This is good news —
use it directly to answer "how were the thresholds derived."

**Reviewer point:** Major Issue 5 ("thresholds... must be reported, together with how those
thresholds were derived").

---

## 11. §3.1.1/§4.1.2 — split → augmentation → feature-extraction → binning order (Reviewer 3, Q2)

**Now:** the manuscript never states whether image-level splitting happens before or after
augmentation and feature extraction — Reviewer 3 point 2 asks for this explicitly, separately
from the partition-size question already fixed in §3 above.

**Change to:** state the pipeline order as a single sentence or small diagram, since this is
exactly what the code enforces and what closes the leakage question:

```
enumerate images -> near-duplicate audit -> SPLIT (image level, stratified, 70/10/20)
                                               |
                  (augment TRAIN ONLY) -> extract features (per split) -> fit bins on TRAIN ONLY
                                               |
                       train -> select checkpoint on VAL -> score TEST once
```

Augmentation is image-space and applied only to the train split, strictly after the split is
frozen — never to val/test, never before splitting. Bin edges and the tokenizer vocabulary are
both fit on train rows only. State this explicitly; it's the direct answer to "is splitting
performed before or after augmentation/feature extraction."

**Reviewer point:** Reviewer 3, Q2 (distinct from Major Issue 2/R2.4 — that one is about
train/val/test *sizes and selection*, this one is specifically about *order of operations*).

---

## 12. Confusion matrices (Reviewer 1 Q2, Reviewer 3 Q4) — DONE, both datasets, real numbers

**Now:** no confusion matrix is included for either dataset.

**Change to:** both datasets' seed-averaged-prediction confusion matrices are now computed
(recovered directly from the checkpointed per-seed test probabilities, verified against the
already-reported seed-averaged accuracy — SteelDefectX: 323/326 correct = 0.99080, matches
exactly). Add both as figures/tables; NEU-DET pairs with the small-test-set caveat from §7 so a
perfect matrix doesn't read as an overclaim.

**NEU-DET** (360 test images, 60/class) — perfectly diagonal, 0 errors:

| True \ Pred | Crazing | Inclusion | Patches | Pitted Surf. | Rolled-in Scale | Scratches |
|---|---|---|---|---|---|---|
| Crazing | 60 | 0 | 0 | 0 | 0 | 0 |
| Inclusion | 0 | 60 | 0 | 0 | 0 | 0 |
| Patches | 0 | 0 | 60 | 0 | 0 | 0 |
| Pitted Surface | 0 | 0 | 0 | 60 | 0 | 0 |
| Rolled-in Scale | 0 | 0 | 0 | 0 | 60 | 0 |
| Scratches | 0 | 0 | 0 | 0 | 0 | 60 |

The crazing↔rolled-in-scale pair Reviewer 1 specifically asked about (Q3): 0 confusions either
direction, 0.0% pairwise error rate, both classes at 60/60 support.

**SteelDefectX** (326 test images) — 3 real errors, accuracy 0.9908 (matches the already-reported
seed-averaged figure exactly):

| True \ Pred | Crazing | Inclusion | Patches | Pitted Surf. | Rolled-in Scale | Scratches |
|---|---|---|---|---|---|---|
| Crazing | 42 | 0 | 0 | 0 | 0 | 0 |
| Inclusion | 0 | 110 | 0 | 0 | 0 | 1 |
| Patches | 0 | 0 | 42 | 0 | 0 | 0 |
| Pitted Surface | 0 | 0 | 0 | 42 | 0 | 0 |
| Rolled-in Scale | 0 | 0 | 0 | 0 | 42 | 0 |
| Scratches | 0 | 1 | 0 | 0 | 1 | 45 |

All 3 errors involve Scratches (1 Inclusion→Scratches, 1 Scratches→Inclusion, 1
Scratches→Rolled-in-scale), consistent with Scratches being the worst-F1 class (0.970) in the
per-class table (§S1) — the confusion matrix and the per-class metrics agree with each other,
which is itself worth a one-line note in the manuscript as an internal-consistency check.

**Reviewer point:** Reviewer 1 Q2 ("A confusion matrix for each dataset should also be
included"), Reviewer 3 Q4 ("include the corresponding confusion matrix").

---

## 13. Full efficiency table — params + MACs + latency + Macro-F1 together (Reviewer 3 Q6)

**Now:** Table 4 shows accuracy but not parameters/FLOPs/latency side by side with MobileNetV3
and ShuffleNetV2 in one place (§5 above already fixed the classification table's accuracy
numbers but didn't carry the efficiency columns across — add them here as their own table,
which is what Reviewer 3 explicitly asked for).

**Change to — new table, everything measured under the identical protocol:**

| Model | Input | Params (M) | Size (MB) | MMACs | Latency @ bs=1 (ms) | Macro-F1 |
|---|---|---|---|---|---|---|
| SteelSense-BiLSTM | 92 descriptor tokens | 1.917 | 7.312 | 258.048 | 21.13 (classifier only) / **156.3 (end-to-end, 5-ensemble)** | 0.9967 |
| MobileNetV3-Small | 3×224×224 image | 1.524 | 5.814 | 61.46 | 30.36 | 0.9972 |
| ShuffleNetV2-x0.5 | 3×224×224 image | 0.348 | 1.327 | 43.554 | 34.52 | 1.0000 |

**Flag explicitly in the text:** SteelSense-BiLSTM has *more* parameters and *more* MACs than
either CNN baseline, and its classifier-only latency (21.13ms) is competitive only until the
5-member ensemble and descriptor-extraction cost are counted (156.3ms end-to-end — see §9).
The "lightweight" framing needs to be qualified: lightweight relative to typical deep CNN/
Transformer backbones in the literature, not relative to these two specific mobile-optimized
CNNs measured here, which are both smaller and about as fast once the full pipeline is counted.

**Reviewer point:** Reviewer 3 Q6 ("comparisons of parameters, FLOPs, latency, and Macro-F1"),
reinforces Major Issue 7.

---

## 14. Reviewer 2, Q7 — robustness/generalization claims and the cross-dataset experiment

**This point is in Reviewer 2's original 8-item list but was dropped from the decision letter's
consolidated "Major Issues 1–7" — don't miss it because it's not in that summary.**

**Now:** the Discussion (§4.6) asserts robustness to "illumination variations, irregular
surface texture, and cluttered backgrounds" without evidence tied to a robustness experiment,
and no cross-dataset experiment is reported.

**Change to:**
- **Near-duplicate / same-source handling** (already answered structurally): the split groups
  near-duplicate images (dHash + correlation matching) so a duplicate group can never straddle
  train/test — state this explicitly as the answer to "whether near-duplicate or same-source
  images could occur across splits."
- **The cross-dataset experiment Reviewer 2 asks for — train on one dataset, evaluate on the
  common classes of the other without retraining — is not a valid experiment to run, and the
  paper should say why rather than silently omitting it.** The dataset-integrity audit
  (`00_Dataset_Integrity_Audit.ipynb`) found that **1046 of 1631 images (64.1%)** in the
  six-class SteelDefectX subset are pixel-level duplicates of NEU-DET images (99.5–100% overlap
  in 4 of the 6 classes). A model "trained on NEU-DET and evaluated on SteelDefectX" would
  mostly be scoring its own training images — this is precisely why the original submission's
  99.45% SteelDefectX figure and crazing F1 of 1.000 look too good, and it's also why a genuine
  cross-dataset transfer experiment can't be reported honestly on this dataset pair. State this
  finding directly in §4.6 as the answer to Reviewer 2 Q7's cross-dataset request, citing it as
  a limitation of the dataset pair rather than leaving the question unaddressed.
- The `steeldefectx_paper_clean` result (same 6-class subset with NEU-DET duplicates removed)
  is the closest honest substitute for a "robustness under a different acquisition condition"
  check — **now complete, see §S8**: accuracy 0.9801 ± 0.0099, macro-F1 0.9794 ± 0.0101, but
  only 2 of the 6 classes (Inclusion, Scratches) retain enough NEU-disjoint images to be usable
  at all, which is itself the finding to report rather than a clean 6-class transfer number.

**Reviewer point:** Reviewer 2, Q7 (grouped/near-duplicate handling + cross-dataset
experiment) — not covered by the "Major Issues 1–7" numbering, easy to miss.

---

## 15. Manual-only items — not answerable from code/data, need you to do them directly

These four items from R1/R3 require either reading papers I don't have access to, or visual
copyediting of the manuscript file itself — nothing in the codebase can generate them:

- **R1 Q6 / R3 Q3 — literature review additions.** Both reviewers name specific papers/DOIs to
  discuss (R1 lists 7 titles on data-constrained/slender-defect/multiscale detection; R3 names
  `https://doi.org/10.1109/ACCESS.2023.3339994`). I haven't read these and shouldn't fabricate
  what they say — you'll need to read and cite them yourself, tied to the "limited training
  data, multiscale defects, slender structures, multimodal information, real industrial
  deployment" framing R1 asks for.
- **R1 Q7 — Fig. 3 enlargement.** The workflow diagram (p.13) needs to be resized/re-exported
  for readability — a figure-layout fix in whatever tool produced Fig. 3, not a code/data fix.
- **R1 Q7 — English/formatting pass.** Missing spaces after punctuation, inconsistent
  capitalization, informal phrasing, repeated descriptions, "???" incomplete references,
  inconsistent affiliation symbols — a manual copyediting pass over the whole manuscript.
- **Terminology consistency** (Minor Issues paragraph) — "classification," "detection," and
  "localization" are used inconsistently; once §6 above is applied (making clear the model does
  classification + a separate rule-based localization *fallback*, not detection), do a pass to
  make every use of these three words match that distinction.

---

## Summary checklist (page order)

- [ ] p.1 Abstract — soften HPC/GPU and localization/superiority language
- [ ] p.4 Contributions — same
- [ ] p.9 Table 1 — seq_len 94 (not 160), params 1,916,839, size 7.312MB, MACs 258.048M
- [ ] p.10 §3.1.1 / p.21 §4.1.2 — 70/10/20 split, val-macro-F1 selection, real seeds/counts
- [ ] p.21 §4.1.1 — hardware description must match the CPU-only machine actually used
- [ ] p.22 Table 4 — split into self-measured classification table + labeled-as-external
      literature-detector table; replace localization columns with real [DET] numbers
- [ ] p.24 Table 5 — replace with arithmetically-consistent, 5-seed measured values
- [ ] p.25 §4.5 — moderate "significantly outperforming" claims (McNemar p=1.0 vs. RF/SVM)
- [ ] p.25 §4.6 — replace the unsupported ablation claim with the real 6-condition table
- [ ] p.23 §4.4 — replace generic HPC claims with the measured latency/throughput/thread table
- [ ] Fig. 4 (p.19-20) — add/replace with real predicted-vs-GT boxes incl. failure cases
- [ ] p.13 Fig. 3 — enlarge for readability (manual figure-layout task, §15)
- [ ] New confusion-matrix figure — per-dataset, seed-averaged prediction (§12)
- [ ] New efficiency table — params/MACs/latency/Macro-F1 together for all 3 models (§13)
- [ ] §3.1.1/§4.1.2 — state split→augment(train-only)→feature-extract→bin-fit(train-only)
      order explicitly (§11)
- [ ] §4.6 — add the cross-dataset/robustness finding: 64.1% of SteelDefectX (6-class) overlaps
      NEU-DET, so cross-dataset transfer cannot be reported; state this instead of omitting it (§14)
- [ ] Literature review — read and cite the R1/R3-named papers yourself (§15, not generatable)
- [ ] Full manuscript — English/formatting/affiliation-symbol/incomplete-reference pass (§15)
- [ ] Title — reconsider "and Localization" given AP50=5.5%, per the Recommendation paragraph,
      unless a trainable localization head is added before resubmission

---

## Coverage check against all three raw reviewer reports

This file was originally organized around the decision letter's consolidated "Major Issues
1–7," which is actually a paraphrase of Reviewer 2's 8 points (dropping R2's Q7). Reviewer 1's 7
points and Reviewer 3's 11 points overlap heavily with that list but each has 1-2 items the
consolidated summary doesn't carry over. Everything below is now covered somewhere above:

- **Reviewer 1** (7 pts): Q1 ablation→§8, Q2 seeds/CI/confusion matrix→§3+§7+§12, Q3 runtime
  1/4/8 workers→§9, Q4 localization protocol→§6, Q5 verify tables/counts/hardware/seeds/
  thresholds→§2/3/4/7/10, Q6 literature→§15 (manual), Q7 English/Fig.3/Fig.4→§15+checklist.
- **Reviewer 2** (8 pts): Q1 localization source→§6, Q2 SOTA comparison→§5, Q3 F1 consistency→
  §7, Q4 partition/selection→§3, Q5 feature-to-prompt reproducibility→§2/§10, Q6 ablations→§8,
  **Q7 robustness/cross-dataset→§14 (the one the consolidated list drops)**, Q8 HPC→§9.
- **Reviewer 3** (11 pts): Q1 quantitative evidence for claims→§1/§8, **Q2 split-order→§11 (not
  the same as R2.4's split-size question)**, Q3 literature DOI→§15 (manual), Q4 accuracy/CNN
  comparison/confusion matrix→§5/§13/§12, Q5 tabular baselines same split→§5, Q6 MobileNetV3/
  ShuffleNetV2 params/FLOPs/latency/Macro-F1→§13, Q7 order ablation/BiLSTM justification→§8, Q8
  discretization ablation→§8, Q9 checkpoint dependence/held-out test→§3, Q10 5 seeds mean±SD/
  significance→§3/§5/§7, Q11 Macro-F1 + per-class both datasets→§7 (NEU-DET) and §S1
  (SteelDefectX) — both complete.

---

# SteelDefectX — real results (COMPLETE, 2026-09-10)

Source: `SteelSenseV2/02_SteelDefectX_SteelSense_BiLSTM.ipynb`, **all 41 cells fully executed**
("ALL CELLS DONE" logged after 37,745.9s of actual compute across several resumed sessions,
interrupted twice by machine sleep and once by a deliberate 3-hour cap — checkpointing meant no
completed work was ever lost across those interruptions). Every number below is a completed
cell's output or a completed checkpoint unit; nothing is estimated or fabricated.

## S1. Split and main result — complete (5 seeds)

Split (from `00_Dataset_Integrity_Audit.ipynb`, same protocol as NEU-DET): train 1142, val 163,
test 326. Seeds `[42, 1337, 2024, 7, 20250101]`.

Accuracy **0.9902 ± 0.0040**, macro-F1 **0.9906 ± 0.0041** (range 0.9874-0.9975 across seeds).
Seed-averaged prediction: accuracy 0.9908, macro-F1 0.9912, **95% CI (0.9785, 1.0000)** — unlike
NEU-DET's degenerate point-CI, this one is a real interval (326 test images, more errors occur),
so it does not need the same small-sample caveat, though it should still be reported as-is.

Per-class F1 (5-seed mean): Crazing 1.000, Patches 1.000, Pitted surface 0.995, Inclusion 0.990,
Rolled in scale 0.988, **Scratches 0.970 (worst class)**. This directly answers Reviewer 1 Q11 /
R3 Q11 for the SteelDefectX side — replace whatever Table 5 (SteelDefectX half) currently shows
with these arithmetically-verified values (all computed via `sklearn.metrics`, never hand-typed).

## S2. SOTA comparison table — complete, and the same finding as NEU-DET

| Model | Representation | Accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| ShuffleNetV2-x0.5 | image (pretrained) | 1.0000 | **1.0000** | 347,942 |
| XGBoost | raw descriptor vector | 0.9969 | 0.9975 | 17,222 |
| MobileNetV3-Small | image (pretrained) | 0.9939 | 0.9935 | 1,524,006 |
| RandomForest | raw descriptor vector | 0.9908 | 0.9924 | 182,928 |
| **SteelSense-BiLSTM** | **binned descriptors as text prompt** | **0.9902 ± 0.0040** | **0.9906 ± 0.0041** | **1,916,839** |
| MLP | raw descriptor vector | 0.9877 | 0.9874 | 57,478 |
| SVM-RBF | raw descriptor vector | 0.9877 | 0.9874 | 91,471 |
| LogisticRegression | raw descriptor vector | 0.9816 | 0.9823 | 558 |

Strongest baseline is **ShuffleNetV2-x0.5 at a perfect 1.0000** — better than SteelSense-BiLSTM,
same as the NEU-DET finding. McNemar exact test vs. ShuffleNetV2: BiLSTM wrong where ShuffleNet
was right on 3 images, the reverse on 0 images, **p = 0.25, not significant at 0.05** — direction
favors ShuffleNet but the sample is too small to call it conclusive either way. State this
plainly rather than the "significantly outperforms" framing currently in the manuscript; this is
the second dataset in a row where the claim doesn't hold up.

Full efficiency table (same protocol, params/MACs/latency/Macro-F1 together, answering Reviewer
3 Q6 the same way §13 does for NEU-DET):

| Model | Params (M) | Size (MB) | MMACs | Latency @ bs=1 (ms) | Macro-F1 |
|---|---|---|---|---|---|
| SteelSense-BiLSTM | 1.917 | 7.312 | 258.048 | 26.00 (classifier only) | 0.9906 |
| MobileNetV3-Small | 1.524 | 5.814 | 61.46 | 22.26 | 0.9935 |
| ShuffleNetV2-x0.5 | 0.348 | 1.327 | 43.554 | 29.38 | 1.0000 |

## S3. Order ablation — complete, and a genuinely different (more favorable) finding than NEU-DET

| Condition | Macro-F1 | vs. canonical (paired test) |
|---|---|---|
| A. Canonical (deployed) | 0.9911 | — |
| B. Fixed-shuffle | 0.9929 | p = 0.36 (n.s.) |
| C. Per-sample shuffle | 0.9801 | **p = 0.038 — significant** |
| D. DeepSets (order-free) | 0.9819 | **p = 0.047 — significant** |

Unlike NEU-DET (where nothing was significant), **on SteelDefectX the canonical BiLSTM
significantly beats both destroying the order (condition C) and the permutation-invariant
DeepSets control (condition D)**. This is real, worth featuring prominently in §4.6/Discussion —
it's the one piece of evidence across both datasets that actually supports keeping the recurrent
encoder over a cheaper order-free alternative. State both datasets' results side by side rather
than only the favorable one, since NEU-DET's null result is equally real and already in §8.

## S4. Component ablation — COMPLETE (18/18 units, 3 seeds each)

| Condition | Macro-F1 (mean ± SD) | vs. deployed |
|---|---|---|
| pool_max_only | **0.9929 ± 0.0042** | best of the six |
| full_5snapshot_ensemble (deployed) | 0.9911 ± 0.0001 | — |
| pool_mean_only | 0.9878 ± 0.0031 | slightly below |
| no_discretization_numeric | 0.9874 ± 0.0000 | slightly below |
| no_ensemble_best_checkpoint | 0.9855 ± 0.0026 | below |
| pool_attention_only | 0.9854 ± 0.0038 | below |

Same qualitative story as the earlier partial read, now confirmed with the full grid:
**pooling choice matters more on SteelDefectX than on NEU-DET** — max-pooling-only is actually
the single best condition (0.9929, edging out the deployed 3-way pooling), while
attention-only is the worst (0.9854), a ~0.75-point spread. Ensembling still shows a real,
consistent gap over the single-checkpoint variant (0.9911 vs 0.9855). The no-discretization
numeric control (0.9874) sits in the middle of the pack — unlike NEU-DET, where it was the
single best condition, here discretization neither clearly helps nor clearly hurts. **Report
both datasets' ablation tables side by side in the manuscript** (§4.6/Discussion) rather than
generalizing from either alone — the two datasets support different conclusions about which
component matters most, which is itself worth stating explicitly as a limitation of drawing
architecture conclusions from a single dataset.

## S5. Bin-count sweep — COMPLETE (8 combos, 2 seeds each)

Quantile and uniform strategies at 3/5/7/10 bins, epochs=30/patience=6 (lighter budget, same as
NEU-DET's sweep). **Best configuration: uniform, 7 bins** (macro-F1 0.9962 ± 0.0018, vocab 831)
— note this differs from NEU-DET, where **quantile, 5 bins** was best. Full table:

| Strategy | Bins | Macro-F1 (mean ± SD) |
|---|---|---|
| uniform | 7 | **0.9962 ± 0.0018** |
| quantile | 5 | 0.9917 ± 0.0046 |
| uniform | 10 | 0.9917 ± 0.0009 |
| quantile | 3 | 0.9916 ± 0.0009 |
| quantile | 10 | 0.9904 ± 0.0029 |
| uniform | 5 | 0.9899 ± 0.0018 |
| quantile | 7 | 0.9854 ± 0.0012 |
| uniform | 3 | 0.9792 ± 0.0091 |

**This is a real, reportable finding, not noise to paper over**: the optimal discretization
strategy and bin count are dataset-dependent (quantile/5 for NEU-DET, uniform/7 for
SteelDefectX). State this directly — it argues for sweeping per-dataset rather than asserting
one universal bin configuration, which is itself a more defensible position than the original
submission's unexplained fixed choice.

## S6. Interpretability — COMPLETE

Attention-by-class (most class-distinctive descriptors, mirroring NEU-DET's analysis):
Crazing → GLCM ASM/contrast + FFT band 3; Inclusion → GLCM dissimilarity/correlation + edge
uniformity; Patches → FFT bands 2/3 + GLCM ASM; Pitted surface → FFT bands 0/1 + spectral
anisotropy; Rolled in scale → LBP histogram bins 2-5; Scratches → intensity statistics
(Michelson contrast, mean, p10) + GLCM correlation. Different descriptor families dominate
per class, consistent with the qualitative defect descriptions in §3.1.2 of the manuscript
(e.g., Scratches driven by intensity/contrast statistics matches its long, high-contrast
linear appearance; Rolled-in-scale driven by LBP matches its textured, granular appearance).

## S7. Per-stage latency and thread-scaling — COMPLETE (CPU-only machine, see §4)

End-to-end latency at batch size 1: single-checkpoint 93.4 ms, 5-member ensemble 180.0 ms
(inference is 59.5% of end-to-end cost). Throughput 10.7 img/s (single checkpoint). BiLSTM
forward pass is 5.56× the cost of the order-free DeepSets control — read this next to §S3's
order ablation, where (unlike NEU-DET) the BiLSTM *does* significantly beat DeepSets on this
dataset, i.e. here the extra latency cost is buying a real, measured accuracy gain, not "paying
for nothing" as the equivalent NEU-DET sentence has to say.

Thread scaling (1/2/4/8 threads, real feature-extraction throughput): 1→12.4 img/s (baseline),
2→23.3 img/s (1.88×, 94% efficient), 4→26.7 img/s (2.15×, 54% efficient), 8→25.7 img/s (2.07×,
26% efficient). Scaling here is actually *better* at 2 threads than the NEU-DET machine's
equivalent measurement (94% vs 76% efficient) — both numbers are real measurements from the
same physical machine on different data, small differences are expected from system load
variance and should be reported as-is, not smoothed.

## S8. NEU-DET-overlap section — COMPLETE (answers Reviewer 2 Q7 / §14 in full)

| | Images | Classes | Test images | Accuracy | Macro-F1 |
|---|---|---|---|---|---|
| 6-class subset AS REPORTED | 1,631 | 6 | 326 | 0.9902 ± 0.0040 | 0.9906 ± 0.0041 |
| Same subset, NEU-DET duplicates removed | 584 | **2** | 117 | 0.9801 ± 0.0099 | 0.9794 ± 0.0101 |

Per-class overlap (confirms the audit numbers already cited in §14 and README.md): Patches,
Pitted surface, and Rolled in scale are **100% duplicates** of NEU-DET; Crazing is 99.5%;
Inclusion is 37.2%; only Scratches is 0% (entirely NEU-disjoint). **Overall: 1046 of 1631 images
(64.1%) overlap with NEU-DET.** After removing duplicates, only Inclusion and Scratches retain
enough images to be usable (≥20) — the six-class task genuinely cannot be reconstituted from
NEU-disjoint data, exactly as §14 already states. The 2-class NEU-disjoint result (macro-F1
0.9794) is reported here as the finding itself, not as a substitute 6-class benchmark.

## S9. Confusion matrix — COMPLETE (see §12 above for the full numeric table and PNG figure)
