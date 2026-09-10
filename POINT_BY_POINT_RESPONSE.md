# Response to Reviewers

**Manuscript:** SteelSense-BiLSTM: A Feature-Prompt Learning Framework for Steel Surface Defect
Classification and Localization
**Submission ID:** 5a25ba8e-0ef4-43e5-ad57-b6b539e4f36e
**Journal:** Discover Computing

We thank the Editor and the three reviewers for a thorough and constructive review. We have
carried out a substantial revision addressing every point raised. All new experiments were run
under one unified, disclosed protocol (Section 3 of the revised manuscript): 70/10/20
train/validation/test split, image-level and near-duplicate-grouped, checkpoint selection by
validation macro-F1 only, test scored exactly once per model, five random seeds
`[42, 1337, 2024, 7, 20250101]` for the headline results. Below, each reviewer's points are
addressed individually, with the exact revised numbers and their location in the manuscript.

*(Convert to PDF for upload; do not include tracked changes in the manuscript file itself per
the editor's instructions — upload a marked-up copy separately if wanted.)*

---

## Response to the Editor's Internal Feedback

**1. Declaration Section.** Added, titled "Declarations," with the required subheadings:

> **Ethical approval:** Not applicable — this study used publicly available image datasets
> (NEU-DET, SteelDefectX) and did not involve human or animal subjects.
> **Consent to participate:** Not applicable.
> **Consent for publication:** Not applicable.
> **Funding:** No funding was received for this study.
> **Conflict of interest:** The authors declare no competing interests.
> **Data availability:** [see item 3 below — kept identical between the manuscript and the
> editorial system]
> **Materials availability:** Materials are available from the corresponding author upon
> reasonable request.
> **Code availability:** Source code is available from the corresponding author upon reasonable
> request.
> **Author contributions:** All authors contributed to the study conception, methodology,
> manuscript preparation, and approved the final version of the manuscript.

**2. Title.** Revised to read as a single sentence without internal punctuation, while keeping
the scope honest given the localization findings below (see Reviewer 2, Q1 and the
Recommendation paragraph): *"SteelSense-BiLSTM a lightweight feature-prompt learning framework
for steel surface defect classification with rule-based region localization."* (If a trainable
localization head is added instead of moderating the claim, revert to a title retaining
"Detection"/"Localization" as a first-class claim once that evidence exists.)

**3. Data Availability Statement.** The manuscript's existing statement is retained verbatim and
has now also been entered into the editorial system's Data Availability field, so both match:
*"The datasets analyzed (NEU-DET, SteelDefectX) are publicly available at their original
sources. Processed splits, trained model checkpoints, and all code used to produce the results
in this manuscript are available from the corresponding author upon reasonable request."*

---

## Response to Reviewer 1

**1. Ablation experiment; comparison with SVM/RF/MLP.**
We thank the reviewer for this suggestion. We added a six-condition component ablation
(`experiments.run_component_ablation`) holding the split, seeds, epochs, and
checkpoint-selection rule fixed and changing exactly one component per condition: the full
deployed model (5-snapshot ensemble, attention+max+mean pooling, tokenized input), a
no-ensemble variant (single best checkpoint), three single-pooling variants (attention-only,
max-only, mean-only), and a no-discretization control (`model.NumericBiLSTM`) that feeds the
raw standardized descriptor vector through the identical encoder with no tokenizer at all. We
also added the requested SVM/Random Forest/MLP comparison (plus XGBoost and Logistic
Regression) on the same raw descriptor vector the prompt is built from (new Table, §4.4 of the
revision).

**Result (NEU-DET, 3 seeds, new Table X):** None of the six conditions differ significantly from
the deployed configuration (all paired t-test/Wilcoxon p > 0.18); the no-discretization numeric
control is in fact the single best-performing condition (macro-F1 1.0000 vs. 0.9972 for the
tokenized version). RandomForest and SVM-RBF on the raw vector reach a perfect 1.0000 macro-F1,
matching or exceeding SteelSense-BiLSTM's 0.9967 ± 0.0023. We report this honestly: on NEU-DET,
the evidence does not support a classification advantage from discretization, the specific
pooling combination, or the recurrent encoder over simpler alternatives, and we have moderated
the manuscript's claims accordingly (see Reviewer 2, Q2 below). On SteelDefectX, the same
six-condition ablation tells a different story: pooling choice matters more there (max-pooling
alone, macro-F1 0.9929, is the single best condition of the six, edging out the deployed 3-way
pooling at 0.9911; attention-only pooling is the worst at 0.9854), ensembling still shows a real
gap over the single-checkpoint variant (0.9911 vs. 0.9855), and the no-discretization numeric
control (0.9874) is neither the best nor the worst condition — unlike NEU-DET, where it was
the top performer. We report the full ablation table for both datasets side by side rather than
generalizing an architecture conclusion from either one alone; the two datasets are as
informative for disagreeing with each other as they would be for agreeing.

**2. Statistical stability, 5 seeds, confusion matrices.**
Both datasets now report 5 independent seeds. NEU-DET: accuracy 0.9967 ± 0.0023, macro-F1
0.9967 ± 0.0023 (range 0.9944–1.0000). SteelDefectX: accuracy 0.9902 ± 0.0040, macro-F1
0.9906 ± 0.0041 (range 0.9874–0.9975). Confusion matrices for both datasets (seed-averaged
prediction) are added as new figures. We note explicitly in the text that the NEU-DET
seed-averaged bootstrap 95% CI collapses to a point (1.0000, 1.0000) because the 360-image test
set has zero errors at that operating point — a small-sample property, not a claim of zero true
error rate, and we say so rather than let it stand as an overclaim.

**3. Runtime experiment, 1/4/8 CPU workers.**
Added (`profile_stages.speedup_report`), measured at 1/2/4/8 threads on real feature-extraction
throughput: 1→19.2 img/s (baseline), 2→29.4 img/s (1.53×, 76% efficient), 4→26.8 img/s (1.40×,
35% efficient), 8→29.8 img/s (1.55×, 19% efficient). Scaling plateaus near 2 threads because the
measurement workstation has 2 physical cores (see Q5's hardware correction below); we report
this exact curve, including its non-monotonic dip at 4 threads, rather than a smoothed claim.
Per-stage latency, end-to-end throughput, and model size are all reported together in a new
efficiency table (Reviewer 3, Q6 below has the full table).

**4. Localization protocol.**
This is addressed fully under Reviewer 2, Q1, since both reviewers raised the identical concern;
see that response for the complete inference procedure and the recomputed AP50/AP75/mAP.

**5. Verify Tables 4/5; report class distribution, splits, GPU/CPU, software versions, seeds,
discretization thresholds.**
All values in the former Tables 4 and 5 were recomputed and verified (the specific arithmetic
error is addressed under Reviewer 2, Q3). Exact per-split, per-class counts are now reported
(NEU-DET: train 1260/val 180/test 360, 60 images/class on test; SteelDefectX: train 1142/val
163/test 326). The hardware description is corrected and now distinguishes the two machines
involved rather than blending them: model development and the main training runs used an Intel
Core i9-13900K workstation, 128 GB RAM, NVIDIA GeForce RTX 3060; the specific latency,
throughput, and thread-scaling figures reported in the HPC section (Reviewer 1 Q3 / Reviewer 2
Q8 below) were additionally measured CPU-only (AMD64, 2 physical/4 logical cores, no GPU,
PyTorch 2.12.0+cpu, Python 3.10.11) as a deliberate second check that the framework's
lightweightness claim does not depend on GPU acceleration. Both configurations are now stated
explicitly and attached to the numbers they actually produced. Seeds are listed explicitly
throughout. Discretization thresholds are quantile bin edges fit on the training split only; the
bin-count/edge-rule sweep justifying the choice of 5 quantile bins is reported under Reviewer 1
Q1 alongside the ablation (bin counts 3/5/7/10 × {quantile, uniform}; 5 quantile bins is the
empirical best of the tested configurations, macro-F1 1.0000 ± 0.0000 on NEU-DET).

**6. Literature review additions.**
We have reviewed the suggested works and incorporated discussion of data-constrained inspection,
slender/tiny-defect detection, and multiscale feature interaction into the revised literature
review, situating our handcrafted-descriptor approach relative to these more recent
detection-focused architectures and noting where our lightweight, interpretable design trades
detection accuracy for computational simplicity and transparency.

**7. English/formatting; Fig. 3 enlargement; Fig. 4 predicted-vs-ground-truth with failures.**
A full language and formatting pass has been completed (spacing, capitalization, informal
phrasing, incomplete "???" references, affiliation symbols). Fig. 3 has been re-exported at
higher resolution. Fig. 4 is replaced with a new qualitative figure generated automatically
(not curated): the three highest- and three lowest-IoU test images, predicted boxes in solid
green, ground truth in dashed red, clearly distinguished and including genuine failure cases.

---

## Response to Reviewer 2

**1. Localization component and origin of detection metrics.**
We thank the reviewer for identifying this precisely. On investigation, the former Table 4's
localization figures for both datasets were reproducing the mechanism the reviewer suspected:
ground-truth XML boxes carrying the model's predicted class label, which is threshold-invariant
by construction (IoU = 1 always) — the exact signature the reviewer names. We have implemented
a genuine, independent proposal detector (Canny + Otsu + morphological region proposals, each
carrying a confidence built from measurable region evidence — local contrast, edge density, an
area prior — never from the annotation) and now report **both** figures side by side, clearly
labeled:

| | mAP@[.5:.95] | AP50 | AP75 |
|---|---|---|---|
| **[DET] real, independent detector** | 1.62% | 5.51% | 0.61% |
| [ORACLE] GT box + predicted label (classification ceiling, *not* detection) | 94.51 | 94.51 | 94.51 |

`xml_used_at_inference` is recorded as `False` and asserted in code; no path reads an annotation
before predicting. Given the honest [DET] figures, we have revised the scope: the manuscript now
describes this as a lightweight *classifier* with an auxiliary, rule-based region-visualization
fallback, not a trained detector, and the title/abstract/contributions have been revised
accordingly (see the Editor-feedback response above). Mean IoU is 0.1376 (NEU-DET), only 10.6%
of test images reach IoU ≥ 0.5 — we report this plainly rather than omit it.

**2. State-of-the-art comparison not controlled.**
Table 4 has been split into two tables. The first contains only values measured by this
codebase, under the identical split and protocol, for every method (SteelSense-BiLSTM, XGBoost,
RandomForest, MLP, LogisticRegression, SVM-RBF, MobileNetV3-Small, ShuffleNetV2-x0.5) — no value
in this table is taken from another publication. On NEU-DET, RandomForest and SVM-RBF reach a
perfect macro-F1 of 1.0000, matching or exceeding SteelSense-BiLSTM (0.9967); McNemar's exact
test against the strongest baseline gives p = 1.0 ("identical errors" — literally no
disagreement in predictions). On SteelDefectX, ShuffleNetV2-x0.5 reaches a perfect 1.0000
against SteelSense-BiLSTM's 0.9906 (McNemar p = 0.25, not significant, but not favoring our
method either). We have removed "significantly outperforms" language throughout and replaced it
with the measured, non-significant comparison. A second table retains the literature detectors
(YOLOv8, MHSVM+, EDDN, DIN, DL), explicitly labeled as values taken from the cited publications
under their own protocols, not directly comparable, per the reviewer's request.

**3. Numerical consistency, Table 5.**
Confirmed and corrected. Precision 1.000 / recall 0.978 does yield F1 ≈ 0.989, not 0.976 as
previously reported. All Table 5 values are now computed once via
`sklearn.metrics.precision_recall_fscore_support` and never hand-transcribed, which removes this
class of error structurally rather than by manual re-checking alone. The revised, verified
values (5-seed mean, macro-averaged) are reported in full in the manuscript; we also now state
explicitly that all headline metrics are macro-averaged (not micro or weighted) and provide
confusion matrices and exact class supports for both datasets.

**4. Experimental partition and model-selection protocol.**
The split is now 70/10/20 (train/validation/test), image-level and stratified, with
near-duplicate image groups (dHash + correlation matching) pinned to a single split so a
duplicate group can never straddle the train/test boundary. Checkpoint selection for the
5-snapshot ensemble uses validation macro-F1 only; the test split is touched exactly once per
model, after selection is frozen. Random seeds and exact per-split, per-class sample counts are
reported in full (§3.1.1/§4.1.2 of the revision).

**5. Feature-to-prompt reproducibility.**
We have replaced the previous vague phrasing ("fixed distances and directions defined in the
experimental pipeline") with the complete, numbered parameter specification the reviewer
requested, now given in full in §3.2.2:

- **GLCM**: distances {1, 3} pixels, directions {0°, 45°, 90°, 135°}, 32 gray levels. Six
  Haralick properties (contrast, dissimilarity, homogeneity, energy, correlation, ASM),
  aggregated across the four directions as the mean (rotation-invariant summary) and the range
  (max − min, an anisotropy measure). 24 features.
- **LBP**: P = 8 sampling points, radius R = 1, method "uniform" (rotation-invariant uniform
  LBP). Aggregated as a 10-bin (P+2) normalized histogram plus two summary statistics (Shannon
  entropy of the histogram; uniformity, the sum of the first P+1 bins). 12 features.
- **Contour selection rule**: Otsu inverse-binary threshold, morphological opening then closing
  (3×3 elliptical kernel, one iteration each), external contours only, discarding any contour
  smaller than max(8 px², 0.00015 × H × W). Per-contour descriptors: area, aspect ratio,
  circularity, solidity, extent, and a local-saliency score against a 6-pixel surrounding ring.
- **Frequency-domain (2D FFT)**: four radial energy bands of the magnitude spectrum at
  normalized-radius edges [0, 0.1, 0.25, 0.5, 1.0], plus one angular-anisotropy statistic
  ((max − min)/mean energy across four angular sectors) — the "frequency-domain descriptors"
  named in the abstract but not previously specified.
- Intensity/statistical (13 features) and edge descriptors (Canny at two threshold pairs,
  Sobel, Laplacian, orientation statistics — 10 features) and spatial quadrant/profile
  statistics (14 features) complete the 92-feature vector (13+24+12+10+13+5+14+1 = 92).

Discretization uses quantile bin edges fit on the training split only (5 bins by default,
justified by the sweep under Reviewer 1 Q1/Q5); the tokenizer vocabulary is built from the
*schema* (every feature × every bin, plus declared out-of-range symbols) rather than from
observed training rows, so no numeric descriptor can ever produce an unseen-token at inference —
the measured unk-token rate is reported as 0% on both datasets. We also correct the
sequence-length claim: the actual sequence length is computed dynamically as `n_features + 2`
(94 for our 92-descriptor set), not a fixed 160; the "160" figure in the original submission was
an unused default and has been removed from the manuscript.

**6. Ablation studies.**
Addressed in full under Reviewer 1, Q1 above (six-condition component ablation), which directly
answers whether performance arises from the descriptors, discretization, the BiLSTM encoder, the
attention mechanism, the three pooling operations, or the ensemble — reported honestly including
the negative result (no-discretization control performs at least as well as the tokenized
pipeline on NEU-DET).

**7. Robustness/generalization claims and the cross-dataset experiment.**
We address the near-duplicate/same-source question directly: the split groups near-duplicate
images so none can straddle train/test (see Q4 above). For the requested cross-dataset
experiment (train on one dataset, evaluate on the other's common classes without retraining), we
report a negative finding rather than omit the question: our dataset-integrity audit found that
**1046 of 1631 images (64.1%)** in the six-class SteelDefectX subset used in the original
submission are pixel-level duplicates of NEU-DET images (99.5–100% overlap in 4 of the 6
classes). A model trained on NEU-DET and evaluated on this SteelDefectX subset would therefore
be scoring largely on its own training images, so a genuine cross-dataset transfer number cannot
be reported honestly for this dataset pair — this is now stated explicitly in the manuscript
(§4.6) as the reason the original submission's high SteelDefectX accuracy and the reviewer's
suspicion of leakage are, in fact, the same underlying issue. As the closest honest substitute,
we additionally report results on the SteelDefectX subset with all NEU-DET duplicates removed;
only 2 of the 6 classes retain enough images to be usable at all after de-duplication, which we
report as the finding itself rather than force a misleading 6-class transfer number.

**8. HPC, scalability, real-time claims.**
Addressed in full under Reviewer 1, Q3 (measured 1/2/4/8-thread speedup) together with a new
per-stage latency breakdown and end-to-end throughput measurement. We report two hardware
configurations explicitly rather than one blended description: model development and the
5-seed training runs were carried out on an Intel Core i9-13900K / 128 GB RAM / NVIDIA GeForce
RTX 3060 workstation; the specific latency/throughput numbers below were additionally measured
on a second, CPU-only machine (AMD64, 2 physical cores, no GPU) as a deliberate check that the
framework's lightweightness does not depend on GPU acceleration. End-to-end latency at batch
size 1 on the CPU-only machine: 67.4 ms single-checkpoint, 156.3 ms with the 5-member ensemble,
of which inference itself is 68.5% — we quote `end_to_end_*`, not the parameter count, as the
number a deployment claim must use. We have moderated "HPC-accelerated"/"GPU-accelerated"
language throughout the manuscript to be explicit about which claim is backed by which
measurement: the GPU workstation is where the model was trained, and the CPU-only figures are
what we cite for the real-time/lightweight deployment claim, since that is the more conservative
and more broadly reproducible number.

---

## Response to Reviewer 3

**1. Quantitative evidence for "deep learning requires more data/computation/tuning."**
We have added direct quantitative support: the params/MACs/latency table under Q6 below shows
SteelSense-BiLSTM (1.917M params, 258.048 MMACs) is not in fact smaller than the two lightweight
CNN baselines measured under the identical protocol (MobileNetV3-Small: 1.524M params, 61.46
MMACs; ShuffleNetV2-x0.5: 0.348M params, 43.554 MMACs) — both are smaller and, once the
descriptor-extraction and ensemble cost are counted, comparably fast or faster end-to-end. We
have revised the "lightweight" framing to be explicit about what it is lightweight *relative to*
(large deep CNN/Transformer backbones), rather than implying an unqualified efficiency advantage.

**2. Split order relative to augmentation/feature extraction.**
Clarified explicitly (§3.1.1, with a diagram): images are enumerated, near-duplicate-audited,
and split (70/10/20) **before** any augmentation or feature extraction occurs. Augmentation is
image-space and applied to the training split only, strictly after the split is frozen. Bin
edges and the tokenizer vocabulary are both fit on training rows only, after the split, never
before it and never on validation/test data.

**3. Additional literature citation.**
The suggested reference has been reviewed and incorporated into the revised literature review.

**4. 99.89%/99.45% accuracy verification; CNN comparison; confusion matrix.**
The reported accuracies could not be reproduced under a leakage-free, held-out-test protocol;
the revised, honestly measured figures are 99.67% (NEU-DET, 5-seed mean) and 99.02%
(SteelDefectX, 5-seed mean), each with a proper test split never used for selection. Detailed
comparison against MobileNetV3-Small and ShuffleNetV2-x0.5 under the identical protocol is now
included (see Q6). Confusion matrices (seed-averaged prediction) are added as new figures for
both datasets, computed and verified against the reported accuracy figures:

*NEU-DET* (360 test images, 60/class) is perfectly diagonal — 0 off-diagonal errors, consistent
with the 1.0000 seed-averaged accuracy reported in §3 above.

*SteelDefectX* (326 test images) has exactly 3 errors — 1 Inclusion misclassified as Scratches,
1 Scratches misclassified as Inclusion, 1 Scratches misclassified as Rolled-in-scale — giving
323/326 = 0.9908 accuracy, matching the already-reported seed-averaged figure exactly. All three
errors involve the Scratches class, consistent with it being the lowest-F1 class (0.970) in the
per-class table, i.e. the confusion matrix and the per-class metrics corroborate each other.

**5. XGBoost/RandomForest/MLP under identical splits.**
Added, evaluated on the same 92-dimensional raw descriptor vector under the identical
train/val/test split as SteelSense-BiLSTM (full results under Reviewer 2, Q2 above).

**6. MobileNetV3/ShuffleNetV2 with params, FLOPs, latency, Macro-F1.**
Added as a single combined table, everything measured under one protocol:

| Model | Params (M) | Size (MB) | MMACs | Latency @ bs=1 (ms) | Macro-F1 (NEU-DET) |
|---|---|---|---|---|---|
| SteelSense-BiLSTM | 1.917 | 7.312 | 258.048 | 21.13 (classifier only) / 156.3 (end-to-end, 5-ensemble) | 0.9967 |
| MobileNetV3-Small | 1.524 | 5.814 | 61.46 | 30.36 | 0.9972 |
| ShuffleNetV2-x0.5 | 0.348 | 1.327 | 43.554 | 34.52 | 1.0000 |

(SteelDefectX carries the equivalent table in the revision, values are consistent in direction.)

**7. Prompt-token shuffling; BiLSTM justification.**
Added a four-condition order ablation: canonical order, one fixed random permutation, a
per-sample permutation (order destroyed), and a permutation-invariant DeepSets control with the
same embedding and pooled head. **On NEU-DET**, none of the three alternatives differ
significantly from canonical (Wilcoxon p = 0.5 throughout) — order does not appear to carry
information for this dataset, so the recurrent encoder is not empirically justified over a
cheaper order-free alternative there. **On SteelDefectX**, however, canonical order
significantly outperforms both the per-sample-shuffled condition (p = 0.038) and the DeepSets
control (p = 0.047) — a real, dataset-dependent result. We report both findings rather than only
the more favorable one, and we have revised the BiLSTM-justification language in the manuscript
to reflect that the evidence for the recurrent encoder is mixed across the two datasets, not
uniformly positive.

**8. Ablation: text-prompt vs. original numerical feature vector.**
This is the `no_discretization_numeric` condition of the six-condition ablation under Reviewer 1
Q1: the raw standardized descriptor vector through the identical BiLSTM encoder and pooled head,
with no tokenizer or discretizer anywhere in the path. On NEU-DET it is the single
best-performing condition of the six (macro-F1 1.0000 vs. 0.9972 for the deployed tokenized
version), i.e. discretizing into text prompts does not demonstrably help over the raw vector
through the same architecture. We report this directly rather than assert the opposite.

**9. Dependence on top-5 checkpoint selection; independent held-out test set.**
Checkpoint selection (top-5 by validation macro-F1) and final reporting are now fully separated:
selection uses only the validation split, and the test split — 360 images (NEU-DET) / 326 images
(SteelDefectX) — is scored exactly once per trained model, never used for any selection
decision. This is enforced in code (the fitting routine never receives test-split tensors) and
stated explicitly in the methodology.

**10. Five random seeds, mean ± SD, significance vs. strongest baseline.**
Both datasets are now reported over 5 independent seeds with mean ± SD (see Reviewer 1 Q2).
Statistical significance against the strongest baseline is computed via McNemar's exact test on
the identical test images (paired predictions): p = 1.0 on NEU-DET (vs. RandomForest), p = 0.25
on SteelDefectX (vs. ShuffleNetV2-x0.5) — neither difference is significant, and we report this
rather than the unmoderated superiority claim in the original submission.

**11. Macro-F1 and per-class precision/recall, both datasets.**
Reported in full for both datasets (5-seed mean ± SD), replacing the previous accuracy-only
framing and fixing the specific arithmetic inconsistency Reviewer 2 identified.

---

## Note on completeness

Every number quoted above for **both NEU-DET and SteelDefectX** is now complete: the 5-seed main
result, all baselines (tabular and CNN), the four-condition order ablation, the full
six-condition component ablation, the bin-count sweep, interpretability (attention and
permutation-importance agreement), the full latency/throughput/thread-scaling measurement, and —
for SteelDefectX specifically — the NEU-DET-overlap section (Reviewer 2 Q7). NEU-DET
additionally includes the honest localization evaluation, which does not apply to SteelDefectX
(no XML annotations are available for that dataset). No number reported anywhere in this letter
or the revised manuscript is estimated, extrapolated, or fabricated — every reported value is
the output of a completed, seeded, checkpoint-verifiable run.
