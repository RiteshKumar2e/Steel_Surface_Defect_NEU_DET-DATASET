# Steel Surface Defect Detection using From-Scratch Tiny LLM

This folder contains a complete runnable local project for steel surface defect classification and localization.

It does **not** use OpenRouter, Groq, Claude, Gemini, or any external API. The classifier is a lightweight Transformer / LLM-style model trained from scratch on your dataset.

## Folder Structure

```text
SteelScratchLLM/
├── src/
│   ├── config.py
│   ├── features.py
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── models/
├── results/
├── outputs/
├── sample_data/
├── requirements.txt
├── run_train.bat
├── run_predict.bat
├── run_evaluate.bat
└── LLM_STEEL_SCRATCH_LOCAL.ipynb
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

If your dataset has a different folder name, pass it using `--dataset`.

## Windows Run

Open CMD or PowerShell inside the `SteelScratchLLM` folder.

### 1. Install and Train

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\train.py --dataset NEU-DET --epochs 35
```

Or double-click:

```text
run_train.bat
```

### 2. Predict One Image

Put one test image as:

```text
sample_data/test.jpg
```

Then run:

```bash
python src\predict.py --image sample_data\test.jpg --dataset NEU-DET --save results\prediction.jpg
```

Or double-click:

```text
run_predict.bat
```

### 3. Evaluate Full Dataset

```bash
python src\evaluate.py --dataset NEU-DET --draw
```

This creates:

```text
results/evaluation_results.csv
results/classification_report.txt
results/*.jpg visual outputs
```

## How It Works

```text
Steel Image
↓
OpenCV preprocessing
↓
Handcrafted visual feature extraction
↓
Feature-to-text conversion
↓
From-scratch Tiny Transformer model
↓
Defect class prediction
↓
XML bounding boxes or contour fallback localization
```

## Defect Classes

```text
crazing
inclusion
patches
pitted_surface
rolled-in_scale
scratches
```

## Important Note

This is not a GPT-level large language model. It is a local domain-specific Transformer model trained from scratch. For research/project writing, use this wording:

> A lightweight LLM-inspired Transformer classifier was trained from scratch using structured visual descriptors extracted from steel surface images.

