# SteelSense-BiLSTM — Model and Experiment Documentation

A from-scratch bidirectional LSTM classifier that operates on structured text
prompts derived from hand-engineered image features. It is trained independently
on two datasets, **NEU-DET** and **SteelDefectX**, using separate notebooks that
share the same model definition.

| | |
|---|---|
| Model name | `SteelSense-BiLSTM` |
| Class | `SteelSense_BiLSTM(nn.Module)` |
| Framework | PyTorch, trained from random initialization |
| External APIs | None — no OpenRouter, Groq, Claude or Gemini |
| Task | Six-class steel surface defect classification |
| Localization | Separate FCN segmentation detector (SteelDefectX only) |

**Source notebooks**

| Dataset | Notebook | Status |
|---|---|---|
| NEU-DET | `LLM_STEEL_SCRATCH_LOCAL.ipynb` | Contains a train/validation leakage defect — see §6 |
| SteelDefectX | `SteeldefctX_LLM_STEEL_SCRACTCH_LOCAL.ipynb` | Corrected protocol; use this one as the reference |

---

## 1. Pipeline overview

Both notebooks follow the same five stages. Only the feature extractor and the
localization ground truth differ between datasets.

```
image (200×200)
      │
      ├─► [1] feature extraction        →  ~40 scalar features
      │        NEU-DET      : extract_rich_features()      (intensity, edges, GLCM, FFT, quadrants)
      │        SteelDefectX : extract_features_defect()    (same + mask-derived region stats)
      │
      ├─► [2] prompt construction        →  a single space-separated token string
      │        features_to_prompt_llm2() — continuous values binned into 7 levels
      │
      ├─► [3] tokenization               →  input_ids (160), attention_mask (160)
      │        SimpleTokenizer — whitespace/regex split, vocabulary built on train split only
      │
      ├─► [4] SteelSense-BiLSTM          →  6-class softmax
      │        embedding → BiLSTM → 3 pooled views → LayerNorm → MLP head
      │
      └─► [5] snapshot ensemble          →  averaged probabilities over top-5 checkpoints
```

Localization is **not** produced by this model. It comes from a separate
segmentation network described in §7.

---

## 2. Architecture

```
input_ids (B, 160)                    attention_mask (B, 160)
      │
Token Embedding (vocab × embed_dim, padding_idx=0)
      │
Embedding Dropout (dropout × 0.5)
      │
Bidirectional LSTM (num_layers=2, hidden_dim, dropout between layers)
      │  output: (B, 160, 2·hidden_dim)
      │
      ├── (a) Attention pooling   Linear(2H→H) → Tanh → Linear(H→1) → masked softmax → weighted sum
      ├── (b) Masked max pooling  max over valid positions
      └── (c) Masked mean pooling sum / count over valid positions
      │
Concatenate (a ‖ b ‖ c)  →  (B, 6·hidden_dim)
      │
LayerNorm
      │
Dropout → Linear(6H → H) → GELU → Dropout → Linear(H → 6)
      │
softmax over 6 classes
```

### Why three pooled views

Each view answers a different question about the token sequence:

| View | Captures | Defect type it helps |
|---|---|---|
| Attention pooling | which tokens matter most for this image | learned, class-dependent |
| Masked max pooling | the single strongest signal in any feature | scratches — strong directional edges |
| Masked mean pooling | the overall summary of the whole prompt | crazing — uniform crack texture |

Attention pooling alone discards magnitude information; max pooling alone
discards context. Concatenating all three lets the head separate classes that
share individual features but differ in their combination.

### Configuration and size

| | NEU-DET | SteelDefectX |
|---|---|---|
| `EMBED_DIM` | 96 | 128 |
| `HIDDEN_DIM_2` | 192 | 256 |
| `NUM_LAYERS_2` | 2 | 2 |
| `DROPOUT_2` | 0.30 | 0.30 |
| `MAX_LEN` | 160 | 160 |
| Vocabulary (measured) | 1 256 | 1 345 |
| Encoder dim (2·H) | 384 | 512 |
| Pooled dim (6·H) | 1 152 | 1 536 |
| Parameters (approx.) | ≈ 1.75 M | ≈ 3.07 M |

Vocabulary sizes are read from the saved tokenizer files. Parameter counts are
computed from the layer shapes; the SteelDefectX config comment estimates
"~3.5M", which is the same order.

---

## 3. Input representation — the feature prompt

The model never sees pixels. Each image becomes one string of roughly 45 tokens.

### Feature families

| Family | Features |
|---|---|
| Intensity | mean, std, contrast, entropy, dark/bright/mid pixel percentage |
| Edges | Canny density, Sobel magnitude, Laplacian variance, angle variance, horizontal/vertical edge ratio |
| Regions | count, average/max area, total defect coverage %, circularity, aspect ratio, size histogram (tiny/small/medium/large) |
| Spatial | per-quadrant std and edge density, most active quadrant, edge row/column peak location |
| Texture | GLCM contrast, homogeneity, energy, correlation (4 orientations, 8 grey levels) |
| Spectral | FFT centre energy |

### Discretization

Continuous quantities are binned into seven levels by `_fbin`, so the model sees
ordinal symbols rather than arbitrary floats:

```python
b6 = ['_b0', '_b1', '_b2', '_b3', '_b4', '_b5', '_b6']
entropy_bin = 'entropy' + _fbin(float(f.get('entropy', 0)),
                                [3.5, 4.5, 5.2, 5.8, 6.4, 7.0, 99], b6)
```

Example prompt (abbreviated):

```
steel surface defect classification regions_14 tiny_10 small_4 medium_0 large_0
entropy_b5 area_b1 aspect_b0 coverage_b2 dark_b1 bright_b0 edge_b2 sobel_b2
lap_b3 fft_b1 active_top_left h_edge_0.58 v_edge_0.19 row_peak_40 col_peak_20
circularity_0.68 glcm_contrast_1.24 glcm_homogeneity_0.61 ...
qstd_top_left_23 qedge_top_left_9 ...
```

> **Known limitation.** Not every value is binned. `h_edge_*`, `glcm_*`,
> `circularity_*` and the `qstd_*`/`qedge_*` tokens embed rounded numbers
> directly, so `h_edge_0.23` and `h_edge_0.24` become unrelated symbols with no
> ordering, and values unseen in training map to `<unk>` at inference. Binning
> these consistently is the single highest-value improvement to the input
> representation.

### Augmentation

`_jitter_feature_dict` applies multiplicative Gaussian noise
(`JITTER_SCALE_2 = 0.04`) to numeric features, producing extra *training* prompts
only. Never applied at inference.

---

## 4. The two datasets

| | NEU-DET | SteelDefectX |
|---|---|---|
| Images | 1 800 | 1 631 |
| Classes | 6 | 6 |
| Balance | exactly balanced, 300/class | imbalanced — see below |
| Class naming | `crazing`, `rolled-in_scale`, … | `Crazing`, `Rolled in scale`, … |
| Localization ground truth | PASCAL-VOC XML boxes (1800/1800 coverage) | per-pixel masks (`train_mask/<name>.png`) |
| Image directory | `NEU-DET/IMAGES/<class>/` | `SteelDefectX/train_by_class/<class>/` |

### SteelDefectX class distribution

| Class | Images |
|---|---|
| Crazing | 210 |
| Inclusion | 557 |
| Scratches | 234 |
| Rolled in scale | 210 |
| Patches | 210 |
| Pitted surface | 210 |
| **Total** | **1 631** |

Inclusion is 34% of the dataset, so **accuracy is not a sufficient metric on
SteelDefectX** — macro-F1 and the confusion matrix are required. Both are
produced by the notebook.

> **Note on class names.** The two datasets use different spellings and
> capitalization for the same defects. They are trained as separate models with
> separate label maps; no cross-dataset transfer or evaluation is performed.

---

## 5. Training protocol

Identical in both notebooks except where marked.

| Setting | NEU-DET | SteelDefectX |
|---|---|---|
| Optimizer | AdamW, weight decay 1e-4 | same |
| Learning rate | 8e-4 | same |
| Scheduler | OneCycleLR, `pct_start=0.15` | same |
| Batch size | 32 | same |
| Epochs | 30 | 35 |
| Loss | cross-entropy, label smoothing 0.06 | same |
| Gradient clipping | 1.0 | same |
| Augmentation | 2 extra variants/image | 3 extra variants/image |
| Ensemble | top-5 checkpoints by validation accuracy | same |
| Split | 80/20 | 80/20 |
| Seed | 42 | 42 |

### Snapshot ensemble

After every epoch the state dict is recorded with its validation accuracy; the
top `ENSEMBLE_SIZE_2 = 5` are retained and their softmax outputs averaged at
inference:

```python
snapshots.append((val_acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
snapshots = sorted(snapshots, key=lambda s: -s[0])[:ENSEMBLE_SIZE_2]
```

> **Interpretation caveat.** These five checkpoints come from a single run and
> are strongly correlated, so the variance reduction is smaller than an ensemble
> of independently seeded models would give. They are also *selected* on the
> validation set, which biases any accuracy reported on that same set upward.

---

## 6. The critical difference between the two notebooks

**This is the most important thing to know before quoting any number.**

### SteelDefectX — correct

Cell 6 splits at the **image level first**, then restricts training to the train
images so augmentation can only ever touch them:

```python
_rng = random.Random(SEED)
for c in CLASS_NAMES:
    items = list(_by.get(c, [])); _rng.shuffle(items)
    cut = int(len(items) * TRAIN_SPLIT)
    train_tasks += items[:cut]; val_tasks += items[cut:]

_orig_collect = collect_image_tasks
collect_image_tasks = lambda: train_tasks       # training sees train images only
try:
    model2, tokenizer2 = train_tiny_steel_llm2(force_retrain=False)
finally:
    collect_image_tasks = _orig_collect
```

Cell 7 then evaluates on `val_tasks`, which the model has never seen.

### NEU-DET — defective

Augmentation runs over **all** images, and the split is applied to the augmented
rows afterwards:

```python
rows = build_training_rows_llm2(tasks, IMG_SIZE, augment=AUGMENT_LLM2, ...)
train_rows, val_rows = stratified_split(rows, TRAIN_SPLIT)
```

An image's original prompt can land in train while its 4%-jittered near-duplicate
lands in validation. The NEU-DET validation accuracy therefore measures
near-duplicate matching, not generalization.

**Consequence.** Only the SteelDefectX held-out figure is a valid generalization
estimate. Porting the SteelDefectX split logic into the NEU-DET notebook is the
first fix required before its numbers can be reported.

---

## 7. Results

### 7.1 Classification — SteelDefectX

Two figures exist and they measure different things. **Report the first.**

| Figure | Value | What it measures | Source |
|---|---|---|---|
| **Held-out accuracy** | **97.86%** | 327 unseen images, 1 304 used for training | `steeldefectx_bilstm_results.json` |
| Run-all accuracy | 99.45% | all 1 631 images, ~80% of them seen in training | `steeldefectx_bilstm_runall.json` |

Held-out confusion matrix (rows = true, columns = predicted):

|  | Craz | Incl | Scra | Roll | Patc | Pitt |
|---|---|---|---|---|---|---|
| **Crazing** | 42 | 0 | 0 | 0 | 0 | 0 |
| **Inclusion** | 0 | 109 | 2 | 1 | 0 | 0 |
| **Scratches** | 0 | 2 | 45 | 0 | 0 | 0 |
| **Rolled in scale** | 0 | 0 | 0 | 42 | 0 | 0 |
| **Patches** | 0 | 0 | 2 | 0 | 40 | 0 |
| **Pitted surface** | 0 | 0 | 0 | 0 | 0 | 42 |

Seven errors in total, and the confusion is interpretable: Inclusion ↔ Scratches
(4 errors) and Patches → Scratches (2 errors) — all elongated, high-edge-density
defects that produce similar edge and aspect-ratio tokens.

Per-class precision/recall/F1 in `steeldefectx_classification_report.json`
correspond to the 1 631-image run (macro-F1 0.995), not the held-out set.

### 7.2 Classification — NEU-DET

Reported in `scratch_llm2_results_all1800.json`. Because of the split defect in
§6 and because the evaluation loop covers all 1 800 images including training
images, **this figure should not be quoted as a generalization result** until the
split is corrected.

---

## 8. Localization and detection metrics

Three different numbers appear in the outputs. They are **not** interchangeable.

### [A] Classification-quality proxy — not localization

`steeldefectx_bilstm_metrics.json`:

```json
"AP50": 98.96, "AP75": 98.96, "mAP@[.5:.95]": 98.96
"per_iou_AP": { "0.50": 98.96, ..., "0.95": 98.96 }
```

AP is **identical at every IoU threshold**, which is the signature of this
construction: the ground-truth region is used as the predicted region and only
the class label is replaced by the model's prediction, so IoU = 1.0 by
definition. In the NEU-DET notebook the same thing happens in `detect_areas`:

```python
xml = load_xml_annotations(img_filename, (img_size, img_size))
if xml:
    for r in xml:
        r['label'] = pred_class      # GT geometry kept, label replaced
    return xml, 'xml_annotation'
```

In SteelDefectX the run log records `"area_methods": {"ground_truth_mask": 1631}`
— every region came from the ground-truth mask.

> **This number must never be presented as detection or localization
> performance.** It is a threshold-invariant restatement of classification
> accuracy. The notebook labels it `[A]` for exactly this reason.

### [B] Real localization — the segmentation detector

A small U-Net-style FCN (three encoder blocks, 24/48/96 channels) is trained on
the masks at 128×128 and evaluated at 200×200 against mask-derived boxes.

`steeldefectx_detector_metrics.json`:

| Metric | Value |
|---|---|
| AP50 | 82.10 |
| AP75 | 70.57 |
| mAP@[.5:.95] | 68.22 |
| mean IoU | 0.772 |

Per-threshold behaviour, which now degrades as it should:

| IoU | 0.50 | 0.60 | 0.70 | 0.75 | 0.80 | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| AP | 82.10 | 78.42 | 73.27 | 70.57 | 66.83 | 53.34 | 39.85 |

The ordering AP50 > AP75 > mAP is the expected one, and its presence is the
evidence that this measurement is genuine while [A] is not.

> **Two caveats to state in any write-up.**
> 1. This is computed over **all** images, including those used to train the FCN.
>    A held-out detector figure is not currently produced.
> 2. What the code calls AP is the *fraction of images whose predicted box meets
>    the IoU threshold* (`np.mean(ious >= t)`). With one predicted box and one
>    ground-truth box per image this is a detection rate, not an
>    interpolated precision–recall average precision. Either rename it or compute
>    standard AP with confidence ranking.

### [C] NEU-DET CAM-based localization

For reference, the companion NEU-DET CNN pipeline reports AP50 9.71 and
mAP@[.5:.95] 4.30 for post-hoc CAM-derived regions. That path uses no box
supervision at all and is far weaker than [B]; the two should not be compared
without stating that difference.

---

## 9. Output files

### SteelDefectX — `SteelDefectX/OUTPUT_BILSTM/`

| File | Contents |
|---|---|
| `steeldefectx_bilstm.pt` | snapshot ensemble (5 state dicts + config + val accuracies) |
| `steeldefectx_bilstm_tokenizer.json` | vocabulary (1 345 tokens), `max_len` |
| `steeldefectx_bilstm_results.json` | **held-out** accuracy + confusion matrix |
| `steeldefectx_bilstm_runall.json` | all-1631 accuracy, per class, timing |
| `steeldefectx_classification_report.json` | per-class precision/recall/F1 |
| `steeldefectx_bilstm_metrics.json` | metric [A] — classification proxy |
| `steeldefectx_detector_metrics.json` | metric [B] — real localization |
| `seg_detector2.pt` | FCN segmentation weights (128 px) |
| `steeldefectx_bilstm_confusion.png` | confusion matrix figure |
| `SteelSense-BiLSTM_batch_*.png` | qualitative detection panels |

### NEU-DET — `NEU-DET/OUTPUT_SCRATCH_LLM/`

| File | Contents |
|---|---|
| `tiny_steel_llm2_bilstm_from_scratch.pt` | snapshot ensemble |
| `tiny_steel_llm2_tokenizer.json` | vocabulary (1 256 tokens) |
| `scratch_llm2_results_all1800.json` | all-1800 predictions and accuracy |
| `both_llms_comparison.png` / `.json` | Transformer vs BiLSTM side by side |

---

## 10. Reproduction

Both notebooks hard-code an absolute Windows dataset path, which must be changed
before running elsewhere:

```python
BASE_PATH  = r'C:\Users\anmol\...\SteelDefectX'
MASK_DIR   = r'C:\Users\anmol\hf_datasets\SteelDefectX\train_mask'
```

Recommended replacement:

```python
BASE_PATH = os.environ.get('STEELDEFECTX_PATH') or str(Path.cwd() / 'SteelDefectX')
assert Path(BASE_PATH).is_dir(), f'dataset not found at {BASE_PATH}'
```

Execution order for SteelDefectX: cells 1–8 define and train (cell 8 performs the
image-level split), cell 13 evaluates on the held-out set, cell 14 produces
per-class metrics, cells 16–17 produce metrics [A] and [B].

Retraining is skipped if the checkpoint exists. Pass `force_retrain=True` when
hyperparameters change — the loader does not compare the saved config against the
current one, so a stale model would otherwise be reused silently.

---

## 11. Limitations to state in the paper

1. **Single seed.** Every figure comes from `SEED = 42`. cuDNN is
   non-deterministic on GPU and no `torch.backends.cudnn.deterministic` flag is
   set, so results are not bit-reproducible run to run. Report mean ± standard
   deviation over at least three seeds.
2. **No test set.** Both notebooks use a two-way split. On SteelDefectX the
   validation set is used both to select the five ensemble snapshots and to
   report 97.86%, so that figure carries a selection bias. A third, untouched
   partition is needed.
3. **NEU-DET split defect.** See §6. Its accuracy is not currently a
   generalization estimate.
4. **Metric [A] is not localization.** See §8. Threshold-invariant AP is a
   restatement of classification accuracy.
5. **Detector evaluated on training images.** Metric [B] covers all 1 631
   images, including those the FCN was trained on.
6. **"AP" naming in [B].** It is a detection rate at an IoU threshold, not
   interpolated average precision.
7. **Unbinned continuous tokens.** See §3 — a portion of the feature vector is
   effectively discarded at inference through `<unk>` mapping.
8. **Terminology.** This model is described as an "LLM" in the notebook
   docstrings. It is a six-class classifier over ~45 engineered tokens with no
   language-modelling objective, no generation and no pretraining. Prefer
   *feature-prompt BiLSTM classifier*.

---

## 12. Summary for the manuscript

| Claim | Supporting number | Source |
|---|---|---|
| Classification accuracy, held-out | **97.86%** (320/327) | `steeldefectx_bilstm_results.json` |
| Localization AP50 | **82.10** | `steeldefectx_detector_metrics.json` |
| Localization mAP@[.5:.95] | **68.22** | same |
| Mean IoU | **0.772** | same |
| Model size | ≈ 3.07 M parameters | computed from config + tokenizer |
| Training data | 1 304 images, 3× jittered augmentation | notebook cell 6 |

Do **not** put 99.45% or 98.96 AP in a results table without the qualification in
§7.1 and §8 — both include training data or are threshold-invariant by
construction, and a reviewer checking the per-IoU columns will notice
immediately.
