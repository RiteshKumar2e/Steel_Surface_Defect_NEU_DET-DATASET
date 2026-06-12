# Steel Surface Defect Detection using Two From-Scratch Tiny LLMs

This folder contains a complete, runnable local project for steel surface defect
**classification and localization** on the NEU-DET dataset.

It does **not** use OpenRouter, Groq, Claude, Gemini, or any external API. Both
models are lightweight, domain-specific neural classifiers **trained from scratch**
(random initialization) on your dataset using structured visual descriptors.

The project trains and compares **two independent from-scratch models**:

| | **LLM 1** | **LLM 2** |
|---|---|---|
| Name | `TinySteelLLM_FromScratch` | `SteelSense-BiLSTM` |
| Architecture | Transformer encoder (self-attention) | BiLSTM + multi-view attention pooling |
| Defined in | Cell 5 | Cell 5B |
| Train / load | Cell 7 | Cell 7B |
| Per-image pipeline | `process_image()` (Cell 8) | `process_image_llm2()` (Cell 8B) |
| Run all 1800 images | Cell 9 → `predictions` | Cell 11B → `predictions_llm2` |
| Results file | `scratch_llm_results_all1800.json` | `scratch_llm2_results_all1800.json` |
| Saved weights | `tiny_steel_llm_from_scratch.pt` | `tiny_steel_llm2_bilstm_from_scratch.pt` |

Both share the same handcrafted feature extractor and the same localization /
metrics code; only the prompt-builder and the neural model differ.

---

## The Two Models

### LLM 1 — `TinySteelLLM_FromScratch` (Transformer)
- **Encoder:** `nn.TransformerEncoderLayer` stack (multi-head self-attention).
- **Config:** embed 96, 2 layers, 4 attention heads, 18 epochs, lr 3e-4.
- **Prompt:** coarse structured-text prompt (`features_to_prompt`, ~18 tokens, 4-level bins).

### LLM 2 — `SteelSense-BiLSTM` (recurrent + attention pooling)
- **Encoder:** 2-layer **Bidirectional LSTM**, hidden 192/direction → 384 per time-step.
- **Pooling:** three parallel views — learned **attention pooling**, masked **max pooling**,
  masked **mean pooling** — concatenated (1152-d) → LayerNorm.
- **Head:** `Linear 1152→192 → GELU → Linear 192→6` → softmax.
- **Config:** embed 96, hidden 192, 2 layers, dropout 0.30, 30 epochs, lr 8e-4 (OneCycle).
- **Prompt:** richer prompt (`features_to_prompt_llm2`, ~40 tokens, finer 7-level bins,
  GLCM texture stats, per-quadrant tokens).
- **Training tricks:** label smoothing 0.06, feature-space augmentation (+2 jittered copies
  per image, ~4% Gaussian noise), AdamW + gradient clipping.
- **Inference:** **snapshot ensemble** — the softmax probabilities of the top-5
  validation checkpoints are averaged.

> Full LLM 2 design notes: see [`LLM2_BiLSTM_DETAILS.md`](LLM2_BiLSTM_DETAILS.md).
> Architecture figures: `SteelSense_BiLSTM_architecture.png/.pdf` (model) and
> `SteelSense_BiLSTM_pipeline.png/.pdf` (full 8-stage pipeline).

---

## Folder Structure

```text
SteelScratchLLM/
├── LLM_STEEL_SCRATCH_LOCAL.ipynb        # main notebook (both models)
├── LLM2_BiLSTM_DETAILS.md               # LLM 2 design write-up
├── SteelSense_BiLSTM_architecture.png/.pdf   # LLM 2 model diagram
├── SteelSense_BiLSTM_pipeline.png/.pdf       # LLM 2 8-stage pipeline diagram
├── make_architecture_diagram.py         # regenerates the model diagram
├── make_pipeline_diagram_llm2.py        # regenerates the pipeline diagram
├── src/                                 # optional CLI version (Transformer)
│   ├── config.py  features.py  model.py
│   ├── train.py   predict.py   evaluate.py
├── models/  results/  outputs/  sample_data/
├── requirements.txt
├── run_train.bat  run_predict.bat  run_evaluate.bat
└── NEU-DET/                             # your dataset (see below)
```

## Dataset Setup

Put your NEU-DET dataset inside this folder like this:

```text
SteelScratchLLM/
├── NEU-DET/
│   ├── IMAGES/
│   │   ├── crazing/
│   │   ├── inclusion/
│   │   ├── patches/
│   │   ├── pitted_surface/
│   │   ├── rolled-in_scale/
│   │   └── scratches/
│   └── ANNOTATIONS/
│       ├── image1.xml
│       └── ...
```

---

## How to Run (Notebook — recommended)

Open `LLM_STEEL_SCRATCH_LOCAL.ipynb` and **Run All**. Cell order:

1. **Cells 1–4** — setup, config, `extract_rich_features`, LLM 1 prompt builder.
2. **Cell 5 / 5B** — define LLM 1 (Transformer) and LLM 2 (BiLSTM).
3. **Cell 6** — localization (XML boxes, else class-aware contour detector).
4. **Cell 7 / 7B** — train (or load) LLM 1 and LLM 2. First run trains; afterwards
   set `force_retrain=False` to reload the saved weights instead of retraining.
5. **Cell 9 / 11B** — run every image → `predictions` / `predictions_llm2`.
6. **Cells 10–13** — detection visualizations and area/accuracy charts (per model).
7. **Cell 16 / 16B** — detection metrics (mAP / AP50 / AP75) for LLM 1 / LLM 2.
8. **Cell 17** — side-by-side comparison (`both_llms_comparison.png/.json`).

> Tip: if you edit the notebook outside the IDE, use **Revert File + Restart Kernel
> + Run All** so the kernel picks up the new code.

## How to Run (CLI — optional, Transformer only)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\train.py    --dataset NEU-DET --epochs 35
python src\predict.py  --image sample_data\test.jpg --dataset NEU-DET --save results\prediction.jpg
python src\evaluate.py --dataset NEU-DET --draw
```

(or double-click `run_train.bat` / `run_predict.bat` / `run_evaluate.bat`)

---

## How It Works (shared pipeline)

```text
Steel image (200×200)
        ↓
OpenCV preprocessing  (BGR→RGB, resize, grayscale)
        ↓
Handcrafted visual feature extraction  (~30 descriptors)
        ↓
Feature → structured text prompt        (LLM 1: coarse | LLM 2: rich)
        ↓
From-scratch model  (LLM 1: Transformer | LLM 2: BiLSTM)
        ↓
Defect class prediction + confidence
        ↓
Localization: XML bounding box if available, else class-aware contour detection
        ↓
Evaluation: classification metrics + localization metrics
```

### Handcrafted features (`extract_rich_features`, Cell 3)
- **Intensity:** mean, std, contrast, entropy, dark/bright pixel %.
- **Edges/gradient:** Canny edge density, Sobel magnitude, Laplacian variance,
  horizontal/vertical edge ratio.
- **Regions (morphological):** count, area, % coverage, circularity, aspect ratio, size bins.
- **Spatial:** 4-quadrant std & edge density, edge row/column peak location.
- **Frequency:** FFT center energy.
- **Texture:** GLCM (contrast, homogeneity, energy, correlation).

> Note: the pipeline does **not** apply median filtering, contrast enhancement, or
> normalization in feature extraction, and does **not** use LBP — only the operations
> listed above. (Contour-based localization separately applies CLAHE + blur.)

---

## Localization & Metrics (read this carefully)

Each metrics cell (16 / 16B) reports **two clearly-labelled scores**, because the
two measure very different things:

- **[A] Classification-quality score** — regions are the **XML ground-truth boxes
  relabelled** with the predicted class. IoU is therefore 1.0 for every matched box,
  so this score is threshold-invariant and effectively measures **classification
  accuracy** (both models reach ~99.9% on the 1800-image set).

- **[B] Localization score** — regions are produced by the **class-aware contour
  detector** (no ground-truth leakage), ranked by a genuine per-box confidence, and
  scored with real IoU against the XML boxes. This is an **honest detection metric**
  (mAP/AP50/AP75) and is far lower than [A] — classical contour localization on
  NEU-DET genuinely lands in the low tens of percent AP50, not 90%+.

This separation is intentional: it keeps the high classification number and the
modest-but-honest localization number from being confused with each other.

---

## Defect Classes

```text
crazing   inclusion   patches   pitted_surface   rolled-in_scale   scratches
```

## Output Files (in `NEU-DET/OUTPUT_SCRATCH_LLM/`)

| File | Description |
|---|---|
| `scratch_llm_results_all1800.json` | LLM 1 accuracy / per-class results |
| `scratch_llm2_results_all1800.json` | LLM 2 accuracy / per-class results |
| `tiny_steel_llm_from_scratch.pt` | LLM 1 trained weights |
| `tiny_steel_llm2_bilstm_from_scratch.pt` | LLM 2 snapshot ensemble (list of state-dicts) |
| `TinySteelLLM_FromScratch_*` / `SteelSense-BiLSTM_*` | per-model detection & area charts |
| `both_llms_comparison.png/.json` | side-by-side accuracy comparison |
| `final_summary.json` | mAP/AP50/AP75 (classification-quality + localization) |

---

## Important Note (for research / project writing)

These are **not** GPT-level large language models. They are local, domain-specific
classifiers trained from scratch on structured visual descriptors. Suggested wording:

> Two lightweight from-scratch models — an LLM-inspired Transformer encoder and a
> Bidirectional-LSTM classifier with multi-view attention pooling — were trained on
> structured visual descriptors extracted from steel surface images. No pretrained
> weights or external LLM APIs were used. Classification is evaluated by accuracy,
> and localization is evaluated separately with genuine IoU-based mAP/AP50/AP75.
