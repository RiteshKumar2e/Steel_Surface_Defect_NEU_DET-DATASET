# Paper revision notes — NEU-DET, mapped to real measured results

Source of every number below: `SteelSenseV2/01_NEU_DET_SteelSense_BiLSTM.ipynb`, fully executed.
Each row: where in the current manuscript (`SteelSense_BiLSTM___Ritesh___24_07.pdf`) to edit,
what is there now, what to replace it with, and which decision-letter point it answers.

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
8GB of memory... GPU acceleration through the CUDA framework."

**Change to (measured, from the actual run that produced these numbers):**
`AMD64 Family 23 Model 24 (AuthenticAMD), 2 physical cores / 4 logical cores, 10.6 GB RAM,
cuda_available: False (CPU only), PyTorch 2.12.0+cpu, torch_threads=2, Windows 11`.

This is a direct contradiction the reviewers (and any careful reader) will catch if the i7/GPU
description stays next to CPU-only-measured latency numbers. Either re-run everything on the
literal hardware described, or — simpler and honest — rewrite this section to describe the
CPU-only machine that actually produced the reported numbers.

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

## 12. Confusion matrix figure (Reviewer 1 Q2, Reviewer 3 Q4)

**Now:** no confusion matrix is included for either dataset.

**Change to:** add the seed-averaged-prediction confusion matrix as a figure (NEU-DET first;
SteelDefectX once that run finishes — see note at the end of this file). For NEU-DET: the
seed-averaged prediction is a perfect diagonal matrix (0 off-diagonal entries across all 360
test images) — report it, but pair it with the small-test-set caveat from §7 above so it doesn't
read as an overclaim. The crazing↔rolled-in-scale pair Reviewer 1 specifically asked about
(Q3): 0 confusions either direction, 0.0% pairwise error rate, both classes at 60/60 support.

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
  check — **numbers pending, see note at the end of this file**; when available, report that
  only 2 of the 6 classes retain enough NEU-disjoint images to be usable at all, which is itself
  the finding to report rather than a clean transfer number.

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
  significance→§3/§5/§7, Q11 Macro-F1 + per-class both datasets→§7 (NEU-DET done, SteelDefectX
  pending — see below).

## What's still pending (not a gap in the plan, just not finished computing yet)

The SteelDefectX-side numbers (its own Table 5, its own efficiency/ablation/localization-N/A
notes, and the `steeldefectx_paper_clean` cross-dataset-adjacent result referenced in §14) are
not in this file yet because that notebook run is still executing in the background as of this
writing — it has been retraining the main 5-seed model for over 40 minutes on this machine's
CPU (2 physical cores, no GPU), which is far slower per epoch than expected; the earlier
NEU-DET bin-sweep on this same hardware took 66-110 minutes *per single training run*, so the
full SteelDefectX notebook likely needs several more hours, not the ~2 hours originally hoped
for. I'll add a SteelDefectX section to this file, following the exact same structure as §1-§14,
once it finishes.
