# Manuscript Revision Guide

**Paper:** Steel surface defect recognition on NEU-DET
**Status:** rejected by *Discover Artificial Intelligence* (18 Aug 2026); rebuilding for resubmission
**Code:** [`new_model_code.ipynb`](new_model_code.ipynb) · **Tables:** [`paper_tables.tex`](paper_tables.tex) (regenerate with `python make_tables.py`)

---

## 0. How to read this document

| Marker | Meaning |
|---|---|
| plain number | Measured and verified. Safe to publish. |
| `[RUN]` | Not yet measured. Do not guess it. |
| ⚠️ | A finding that changes what the paper can claim. |

> **The one rule for this revision:** every accuracy and mAP number in the previous
> submission came from an evaluation protocol that was defective. Treat all of them
> as void. Do not carry a single one forward.

---

## 1. What was actually wrong

Three of the reviewers' objections were caused by defects in the code, not by
unclear writing. The response letter is far stronger saying *"this was a bug, here
is the fix"* than *"we have clarified the text."*

### 1.1 The whole-image ground-truth fallback — the fatal one

```python
gts = pd.get('gt_regions', [])
if not gts and ('true_label' in pd):
    gts = [{'bbox': [0, 0, img_size, img_size], 'label': pd['true_label']}]
```

When boxes were missing, the code invented a ground-truth box covering **the entire
image**. Almost any prediction overlaps the whole image at IoU ≥ 0.5, so AP drifted
towards classification accuracy.

⚠️ **This explains the previously reported AP50 of 0.9734 alongside an accuracy of
about 0.97.** Those were not localization results. `load_boxes()` now raises rather
than falling back, and a regression test fails if the behaviour returns.

### 1.2 A dead classification head

```python
x = layers.Conv2D(512, 3, ...)(x)             # x is still the INPUT tensor
x = layers.Dense(1024, ...)(x)
x = layers.Dense(num_classes, 'softmax')(x)   # overwritten on the next line
x = layers.Dense(256, 'relu')(p2_pooled)      # the head that actually trained
```

Two heads written, one trained. **The architecture described in the paper is the one
that was discarded.**

### 1.3 The pyramid skipped stride 16

Taps were labelled C2–C5 but sat at strides 2/4/8/32 — `block_1_expand_relu` is
stride 2. The four-level pyramid in the paper was never built. Corrected:

| Level | MobileNetV2 layer | Stride | Size @200px |
|---|---|---|---|
| C2 | `block_3_expand_relu` | 4 | 50×50 |
| C3 | `block_6_expand_relu` | 8 | 25×25 |
| C4 | `block_13_expand_relu` | 16 | 13×13 |
| C5 | `out_relu` | 32 | 7×7 |

---

## 2. Verified facts for the manuscript

### 2.1 Task definition

> The task is six-class steel surface defect **detection**, trained on the
> human-annotated bounding boxes distributed with NEU-DET. A weak-localization
> variant, trained from image-level labels only, is reported as a controlled
> comparison.

The paper now has a genuine detection head (§5), so "detection" is finally accurate
for the main model — but only for that model. The Grad-CAM branch must still be
called **weak localization** wherever it appears.

### 2.2 Input specification

> All experiments use 200×200×3 inputs. NEU-DET images are natively 200×200
> single-channel 8-bit grayscale; the native resolution is retained without
> resampling and the single channel is replicated three times for the
> ImageNet-pretrained MobileNetV2 stem.

Pipeline: read 200×200 grayscale → replicate to 3 channels → [0,1] → `preprocess_input` → [−1,1] → **training split only:** random H/V flip, ±10 % brightness.

Geometric augmentation is limited to flips deliberately, because boxes must remain
axis-aligned in the original frame. **State this** — it is a design justification of
the kind Reviewer 2 asked for.

### 2.3 Ground-truth protocol

| Property | Value |
|---|---|
| Images | 1800 (300 per class) |
| Native frame | 200×200, depth 1, all 1800 identical |
| Images with verified VOC XML | **1800 / 1800 (100 %)** |
| Total ground-truth boxes | **4189** |
| Boxes flagged `difficult` | 81 |
| Boxes per image | 1–9 |

> Ground truth for all IoU/AP/mAP computation is the set of human-annotated
> PASCAL-VOC boxes distributed with NEU-DET (4189 boxes over 1800 images).
> Contour-derived regions, detector-generated proposals and whole-image boxes are
> never used as ground truth. Coverage is 100 %, so the evaluation subset is the
> complete held-out test split. Boxes flagged `difficult` are excluded from
> matching, following PASCAL-VOC practice.

### 2.4 Split

Frozen, stratified, seed 42, written once to `paper_results/splits/split_v1.csv`.

| Split | Images | Boxes |
|---|---|---|
| Train | 1260 (210/class) | 2899 |
| Validation | 180 (30/class) | 423 |
| Test | 360 (60/class) | 867 |

Model selection uses validation only; the test split is scored **once**.

---

## 3. ⚠️ Two findings that change the paper

These are the most valuable results of the revision, and both are publishable in
their own right. They are also what makes the resubmission credible: they show the
authors interrogated their own method rather than defending it.

### 3.1 Classification accuracy is saturated — the ablation must not use it

Measured on the **test** split, 3+2 epochs:

| Variant | Params | Accuracy | AP50 |
|---|---|---|---|
| MobileNetV2 baseline | 2,423,110 | **100.00** | 8.34 |
| + FPN | 3,196,102 | 99.72 | 7.83 |
| + AMFF | 3,358,111 | 99.17 | 9.12 |
| + CSAF | 3,358,687 | 98.61 | 9.14 |
| Full model | 3,364,064 | 99.72 | 9.64 |

**The bare backbone already reaches 100 % test accuracy.** Every added module makes
accuracy slightly *worse*. A pilot run over 12 architectural variants put 10 of them
at exactly 100 % validation accuracy.

Consequences for the paper:

- The ablation **cannot** be reported on accuracy. There is no headroom, so no
  module can demonstrate a contribution. Report ablations on AP50.
- Any claim that the architecture "improves classification over the baseline" is
  unsupportable and must be removed.
- ⚠️ **CSAF contributes +0.02 AP50 over AMFF.** The central novelty claim has no
  experimental support at this operating point. Either find a setting where it does,
  or withdraw the claim. Do not defend it — that is precisely what Reviewer 2
  objected to.

Suggested sentence: *"Classification accuracy saturates on NEU-DET: a MobileNetV2
baseline already reaches 100 % on the held-out test split (Table 6). Ablations are
therefore reported on AP₅₀, which remains far from ceiling."*

### 3.2 Weak localization fails for a structural reason, and we can show why

AP50 per class from the Grad-CAM branch, against ground-truth box statistics:

| Class | Median box (% of image) | Boxes/image | Weak-loc AP50 |
|---|---|---|---|
| **Pitted surface** | **55.5** | 1.63 | **57.6** |
| Crazing | 21.1 | 2.31 | 0.1 |
| Rolled-in scale | 12.3 | 2.09 | 0.0 |
| Patches | 9.3 | 2.91 | 0.1 |
| Scratches | 7.5 | 2.07 | 0.3 |
| Inclusion | 4.1 | 2.95 | 0.0 |

**AP tracks ground-truth box size almost perfectly.** The single class that works is
the one where 58 % of boxes cover more than half the image — where "highlight
everything" is the correct answer. For the other five, defects are small and
multiple (2–3 per image, 4–12 % of the frame), and a class activation map cannot
separate instances.

This was tested, not assumed: a 72-configuration parameter sweep on validation
(thresholds, minimum area, morphology kernel, Otsu) moved AP50 by **−0.01**. The
limitation is the supervision signal, not the post-processing.

Suggested framing: *"Weak localization succeeds only where the annotation approaches
full-frame extent. Its per-class AP correlates with median ground-truth box area
(Table 4), indicating that class activation maps cannot resolve the small,
multi-instance defects that dominate NEU-DET."*

---

## 4. The detection head (new)

This is the substantive answer to Reviewer 4's central criticism, and the route the
original revision analysis recommended:

> *"If reliable bounding-box annotations are available, Option B provides the
> stronger response to the reviewers' central methodological criticism."*

The dataset has 100 % box coverage. The classification pipeline never used it.

### 4.1 Architecture

```
Input 200×200×3
  → MobileNetV2 → C2 C3 C4 C5
  → FPN with AMFF
  → CSAF cross-scale context, broadcast to each detection level
  → SEAM per level
  → shared anchor-free head: 6 class logits + box regression (l,t,r,b) + centerness
```

All three named modules survive and stay ablatable. The difference is that the
network now has a localization objective, so the pyramid has work the image-level
loss never asked of it.

### 4.2 Head design — FCOS-style, anchor-free

Objects are assigned to a pyramid level by size, which matters here because NEU-DET
boxes span 40 px to 150 px on a 200 px image:

| Level | Stride | Grid | Object size |
|---|---|---|---|
| P2 | 4 | 50×50 | 0–64 px |
| P3 | 8 | 25×25 | 64–128 px |
| P4 | 16 | 13×13 | ≥128 px |

3294 prediction locations in total. A location is positive when it falls inside a
ground-truth box **and** the box's largest side-distance lies in that level's range;
ambiguous locations go to the smallest box. Loss = focal (classification) + GIoU
(regression, centerness-weighted) + BCE (centerness). Score at inference is
√(class × centerness), followed by per-class NMS at IoU 0.5.

**Methods-section detail worth stating:** the classification bias is initialised so
the initial foreground probability is ≈0.01. Without it the focal loss is swamped by
background and training stalls — a reproducibility detail reviewers appreciate.

### 4.3 Results (first run, 12+8 epochs, constant LR)

| Method | Supervision | Params | AP50 | AP75 | mAP50:95 |
|---|---|---|---|---|---|
| Weak localization (Grad-CAM) | image labels only | 3,364,064 | 9.71 | 3.62 | 4.30 |
| **Detection head** | bounding boxes | 3,966,821 | **63.88** | 17.02 | 27.01 |

Per-class AP50, with the weak-localization value for contrast:

| Class | Detection | Weak loc. |
|---|---|---|
| Pitted surface | 83.3 | 57.6 |
| Scratches | 82.8 | 0.3 |
| Patches | 82.6 | 0.1 |
| Inclusion | 67.2 | 0.0 |
| Rolled-in scale | 39.0 | 0.0 |
| Crazing | 28.4 | 0.1 |

**Three of the five classes that scored zero under weak localization now exceed 80.**
That is the cleanest possible confirmation of §3.2: the bottleneck was supervision.

### 4.4 ⚠️ What the numbers still say

- **AP75 (17.0) is far below AP50 (63.9).** Boxes are being *found* but their edges
  are imprecise. This is the signature of under-trained box regression, and it drags
  mAP50:95 down to 27.0.
- **Validation loss plateaued** at a constant learning rate from roughly epoch 4,
  which is why a cosine-decay schedule and a deeper backbone unfreeze were added.
- **Precision at the reported operating point is low (2.25 %) with recall 95.4 %,**
  because the score threshold (0.02) was selected on validation to maximise AP. That
  is correct for AP, but if the paper quotes precision/recall it must pick and state
  a sensible operating point instead.

Report AP50, AP75 and mAP50:95 together. Quoting AP50 alone would hide the box-quality
problem, and a reviewer who asks for AP75 will find it.

---

## 5. Structure — six sections

| § | Title | Contents |
|---|---|---|
| 1 | Introduction | Industrial motivation (short), prior limitations, research gap, contributions |
| 2 | Related Work | Steel defect recognition; lightweight CNNs; FPN; attention; anchor-free detectors |
| 3 | Proposed Method | Architecture; AMFF; CSAF; SEAM; detection head; weak-localization variant |
| 4 | Experimental Setup | Dataset; split; preprocessing; implementation; metrics; hardware |
| 5 | Results and Discussion | Detection; ablation; saturation and box-size analyses; efficiency; errors |
| 6 | Conclusion | Findings, limitations, future work |

Nomenclature table goes in the front matter (Reviewer 1, c4).

---

## 6. Research gap, in three parts

Reviewer 2 asked for this structure explicitly.

**(a) What existing work does.** Feature pyramids merge scales with a *fixed*
arithmetic rule — FPN adds, PANet concatenates. Attention modules (SE, CBAM, ECA)
are then bolted on to recalibrate the merged tensor.

**(b) The limitation.** All of these act *within a single tensor*. None decides **how
much each scale contributes**. That proportion is fixed at design time and identical
for every image. Steel defects violate this: crazing is fine texture living at C2,
pitted surface is a large low-frequency region living at C5 — and our own box
statistics (§3.2) quantify the spread: median box area ranges from 4.1 % to 55.5 % of
the frame across classes.

**(c) The contribution.** CSAF makes the proportion learned, per-sample and
competitive:

```
w = softmax_ℓ( MLP( [ GAP(P2) ; GAP(P3) ; GAP(P4) ; GAP(P5) ] ) )
F = Σ_ℓ  w_ℓ ⊙ P̃_ℓ
```

Each level's gate depends on **all** levels, and the softmax over the level axis
makes them compete.

⚠️ **This claim is currently unsupported** (§3.1: +0.02 AP50). Before writing it as a
contribution, check the `+ CSAF` ablation row on the detection model and
`table_csaf_weights.csv`. If both are flat, the honest contributions list is the
evaluation protocol and the two analyses in §3 — which is still a publishable paper.

### Contributions, as currently supported by evidence

1. A reproducible evaluation protocol for NEU-DET: frozen stratified split, ground
   truth restricted to verified VOC boxes, classification and localization reported
   separately.
2. An analysis showing classification accuracy is saturated on this dataset, so
   architectural ablations must be judged on localization metrics.
3. An analysis explaining when weak localization from image-level labels works and
   when it cannot, quantified against ground-truth box statistics.
4. A lightweight anchor-free detector reaching AP50 `[RUN]` at 3.97 M parameters.
5. *(Conditional)* CSAF, if the ablation supports it.

---

## 7. Tables

All generated by `python make_tables.py` into `paper_tables.tex`. Requires
`booktabs`, `amssymb`, `multirow`.

| Label | Contents | Status |
|---|---|---|
| `tab:split_images` / `tab:split_boxes` | Partition counts | filled |
| `tab:gt_protocol` | Ground-truth audit | filled |
| `tab:boxsize` | **Box-size analysis (§3.2)** | filled |
| `tab:architecture` | Backbone taps | filled |
| `tab:saturation` | **Accuracy saturation (§3.1)** | filled |
| `tab:main_result` | Classifier, multi-seed | 1 seed — needs 5 |
| `tab:ablation_main` | Module ladder | filled |
| `tab:ablation_attention` | SE / ECA / CBAM / AMFF | `[RUN]` |
| `tab:ablation_dilation` | Dilation isolated from capacity | `[RUN]` |
| `tab:localization` | **Detection vs weak localization** | detection filled |
| `tab:detector_per_class` | Per-class detection AP50 | filled |
| `tab:cls_comparison` | Backbone comparison | `[RUN]` |
| `tab:literature` | Published results | `[fill]` from citations |
| `tab:efficiency` | Params / MACs / latency | partly filled |
| `tab:nomenclature`, `tab:setup` | Front matter | filled |

The generator writes a **PROVISIONAL — NOT FOR SUBMISSION** banner into the `.tex`
whenever results come from fewer than 5 seeds, and marks single-run values with a
dagger (standard deviation is *undefined* with one run, not zero). The banner
disappears by itself once the full protocol has been run.

---

## 8. Response to reviewers

| Comment | Response |
|---|---|
| **R1.1** terminology | Task defined once; "detection" now accurate for the main model, "weak localization" used consistently for the CAM branch |
| **R1.2** input/preprocessing | Resolved to 200×200×3, verified from the data (all 1800 images are 200×200×1) |
| **R1.3 / R2.6** repetition | Terminology stated once and cross-referenced |
| **R1.4** nomenclature | Added |
| **R1.5 / R4.6** structure | Six sections; conclusion restored to the end |
| **R2.1–2.2** design justification | Each component justified against a stated limitation; augmentation and BatchNorm choices justified |
| **R2.3** novelty | Narrowed and made falsifiable; AMFF described as a placement choice and tested against SE/CBAM/ECA |
| **R2.4** positioning | Detection with box supervision; weak-localization variant reported as a controlled comparison |
| **R2.5** research gap | Introduction restructured as existing work → limitation → contribution |
| **R2.7** ablation | Module ladder, controlled attention comparison, dilation isolated from capacity, AMFF composition |
| **R2.8** register | Full language pass |
| **R3.1** keywords | Abbreviated terms removed |
| **R3.2** inference speed | Latency, p95 and FPS on stated hardware |
| **R3.3** Fig. 2 | Regenerated from the code; typo fixed |
| **R3.4** data amounts | Exact per-class counts, images and boxes |
| **R3.5** YOLO-11/26 | YOLOv8n, YOLO11n **and YOLO12n** on our split. YOLO26 is not in the Ultralytics release used (8.3.174 ships v3–v12 and RT-DETR), so it could not be run under an identical protocol |
| **R4.1** task | Detection head with box supervision; training objective stated explicitly |
| **R4.2** ground truth | Dedicated protocol subsection; verified VOC only; the whole-image fallback removed and regression-tested |
| **R4.3** unfair comparison | Reproduced baselines on our protocol kept separate from published results, which carry a protocol column |
| **R4.4** inconsistencies | Single source of truth in the config |
| **R4.5** architecture | Modules defined once; CEAM retired in favour of CSAF; two implementation defects corrected |
| **R4.7** efficiency | Full cost table with environment reporting |

**Suggested opening.** Thank the reviewers, then state plainly that re-examination
found genuine defects in the evaluation code — a whole-image ground-truth
substitution, a discarded classification head, a mislabelled pyramid — that all
results have been regenerated under a corrected protocol, and that the claims have
been narrowed to what the new evidence supports. Reviewers respond well to that.

---

## 9. Red lines

- ❌ Any accuracy or mAP number from the previous submission, including 0.8517 /
  0.9734 / 0.7300 — those are artifacts of the whole-image ground-truth fallback.
- ❌ "Weak localization" results described as detection, or vice versa.
- ❌ "Real-time" unless measured FPS on named hardware supports it.
- ❌ A published number ranked in the same column as one of ours.
- ❌ CSAF claimed as the contribution while the ablation shows +0.02 AP50.
- ❌ AP50 quoted without AP75 and mAP50:95 alongside.

---

## 10. Order of work

1. `pip install "numpy==1.26.4"` — TensorFlow 2.15 cannot import under NumPy 2.x.
2. Detector retraining with cosine decay is running; fill §4.3 from
   `paper_results/tables/detector_results.json`.
3. Run the remaining ablations (attention, dilation) **on the detector**, where AP50
   has headroom, not on the saturated classifier.
4. Run the YOLO baselines for the comparison table.
5. Multi-seed the final model (5 seeds) and regenerate — the provisional banner
   disappears when that is done.
6. Decide the CSAF claim from evidence.
7. Write §§1–2 last, once the results are settled.

**Commit after every run.** `make_tables.py`, `paper_tables.tex` and this file were
lost three times to filesystem/sync deletion before being tracked in git.

### Getting AP above 75

Currently 63.88. In order of expected value:

1. **Cosine LR decay** — added; the first run plateaued at a constant rate. Should
   help AP75 and mAP most, since those measure box quality.
2. **Longer schedule on a GPU.** At 11 min/epoch on 2 CPU cores this is the binding
   constraint. The notebook is portable; on a GPU 60–80 epochs is routine.
3. **Deeper backbone unfreeze** — added (60 layers instead of 30). Detection depends
   on spatial detail the ImageNet stem was not trained to preserve.
4. **Higher input resolution.** 200×200 is native, but upsampling to 320 or 400 gives
   the P2 level more to work with for the small classes (inclusion at 4.1 % of the
   frame is ~40 px). State any change of resolution explicitly.
