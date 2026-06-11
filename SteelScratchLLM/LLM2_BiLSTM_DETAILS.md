# 2nd LLM (SteelSense-BiLSTM) — Kaise Bana Hai aur Kya-Kya Use Hua Hai

Ye file batati hai ki notebook (`LLM_STEEL_SCRATCH_LOCAL.ipynb`) me jo **doosra from-scratch LLM** banaya gaya hai, woh kaise design kiya gaya hai, kaun-kaun se techniques use hue hain, aur woh pehle wale LLM (Transformer) se kaise alag hai.

> Dono LLMs **local hain, from scratch train hote hain** — koi OpenRouter / Groq / Claude / Gemini jaisi external API use nahi hoti. Sirf PyTorch model jo aapke NEU-DET dataset par khud train hota hai.

---

## 1. Dono LLMs ka basic farak

| | **LLM 1** (pehle wala) | **LLM 2** (naya wala) |
|---|---|---|
| Naam | `TinySteelLLM_FromScratch` | `SteelSense-BiLSTM` |
| Architecture family | Transformer (self-attention) | BiLSTM (recurrent + attention pooling) |
| Defined kaha hai | Cell 5 | Cell 5B |
| Train/Load | Cell 7 | Cell 7B |
| Per-image pipeline | `process_image()` (Cell 8) | `process_image_llm2()` (Cell 8B) |
| Sabhi 1800 images run | Cell 11 → `predictions` | Cell 11B → `predictions_llm2` |
| Result file | `scratch_llm_results_all1800.json` | `scratch_llm2_results_all1800.json` |
| Saved model | `tiny_steel_llm_from_scratch.pt` | `tiny_steel_llm2_bilstm_from_scratch.pt` |

Dono ka result alag-alag JSON me save hota hai aur Cell 17 me side-by-side comparison chart bhi banta hai (`both_llms_comparison.png` / `.json`).

---

## 2. LLM 2 ka Architecture — `SteelSense-BiLSTM` (Cell 5B)

Layer-by-layer flow:

```
input tokens (feature-prompt se bane)
        │
   Token Embedding (size = EMBED_DIM = 96)
        │
   Embedding Dropout
        │
   Bidirectional LSTM  (hidden = HIDDEN_DIM_2 = 192, layers = 2)
        │
        ├── (a) Attention Pooling   → "kaunse tokens sabse zaroori hain"
        ├── (b) Masked Max Pooling  → "har feature ka sabse strong signal"
        └── (c) Masked Mean Pooling → "poore prompt ka overall summary"
        │
   Concatenate (a + b + c)  →  LayerNorm
        │
   Classifier head (Linear → GELU → Dropout → Linear → 6 classes)
```

**Multi-view pooling kyu use kiya?**
Sirf attention-pooling se model ko sirf "important tokens" dikhte hain. Max-pooling se "sabse strong signal" pakad me aata hai (jaise scratches me strong directional edges), aur mean-pooling se "overall texture summary" milta hai (jaise crazing ka uniform crack-pattern). Teeno ko milake dene se model confusable classes (scratches vs patches vs crazing) ko better separate kar pata hai.

---

## 3. Input — LLM 2 ka apna "rich" feature-prompt (`features_to_prompt_llm2`)

LLM 1 chhote, coarse prompt (~18 tokens, sirf 4-level bins) use karta hai. LLM 2 ke liye ek **alag aur zyada detailed prompt-builder** banaya gaya hai jo same image-features (jo `extract_rich_features` Cell 3 me nikalta hai) ko zyada baareeki se text-tokens me todta hai:

- **Finer bins (7 levels instead of 4)**: entropy, avg_area, aspect-ratio, defect coverage %, dark/bright pixel %, edge density, sobel magnitude, laplacian variance, FFT spectral energy
- **GLCM texture stats** (contrast, homogeneity, energy, correlation) — 2-decimal precision tokens
- **Per-quadrant stats** — har quadrant (top-left/top-right/bottom-left/bottom-right) ka std-deviation aur edge-density alag token banta hai
- **Edge-peak location tokens** — row/col me sabse zyada edges kaha concentrate hain
- **Circularity, aspect-ratio averages**, region counts (tiny/small/medium/large), most-active quadrant

Iska fayda: model ko har image ka ek bahut zyada **distinctive "fingerprint"** milta hai, jisse visually-similar classes (scratches/patches/crazing) ke beech better separation hota hai.

---

## 4. Extra Data — Training-time Augmentation (`build_training_rows_llm2` + `_jitter_feature_dict`)

From-scratch model ko zyada data se fayda hota hai, isliye:

- Har training image ke liye **`AUGMENT_LLM2 = 2` extra "jittered" copies** banayi jaati hain
- `_jitter_feature_dict()` numeric features (entropy, area, edges, etc.) me chhota sa **random Gaussian noise** (`JITTER_SCALE_2 = 0.04`, yaani ~4%) daal deta hai, fir us perturbed feature-set se naya prompt banta hai
- Isse training set ka size **3x ho jaata hai** (1 original + 2 jittered) bina dataset me naye images add kiye — model zyada robust banta hai aur overfit kam karta hai

---

## 5. Training Process — kaun se tricks use hue (Cell 5B → `train_tiny_steel_llm2`)

| Technique | Kya karta hai | Kyu use kiya |
|---|---|---|
| **Label Smoothing** (`LABEL_SMOOTH_2 = 0.06`) | Model ko 100% confident hone se rokta hai | Overconfidence kam, generalization better |
| **OneCycle LR Scheduler** | Learning-rate ko warm-up → peak → cool-down pattern me chalata hai | Standard flat-LR se zyada stable aur fast convergence |
| **AdamW optimizer** + gradient clipping | Weight-updates stable rakhta hai | Training crash/explode nahi hoti |
| **Dropout = 0.30** + embedding-dropout | Random neurons "band" karta hai training me | Bada model (192-dim) overfit na kare |
| **EPOCHS_2 = 30** | LLM 2 ka apna training-budget (LLM 1 se zyada lamba) | Bada/recurrent model ko converge hone me zyada steps chahiye |

---

## 6. Snapshot Ensemble — Final accuracy-boost trick

Training ke dauraan sirf last/best model save karne ke bajaye:

- Har epoch ke baad model ka validation-accuracy check hota hai
- **Top `ENSEMBLE_SIZE_2 = 5` checkpoints** (sabse achi val-accuracy waale) memory me rakhe jaate hain
- Inference ke time (`classify_defect_llm2`) **paanchon models ko run karke unki probabilities ko average** kiya jaata hai, aur final prediction us average se nikalta hai

Ye ek well-known technique hai jisse ek single training-run ke "luck/variance" ka asar kam ho jaata hai aur final accuracy thodi aur badh jaati hai — bina koi naya data ya bada model banaye.

---

## 7. Config / Hyperparameters (Cell 2 me defined)

```python
MODEL_NAME_2     = 'SteelSense-BiLSTM'
HIDDEN_DIM_2     = 192     # BiLSTM hidden size
NUM_LAYERS_2     = 2       # LSTM layers
DROPOUT_2        = 0.30    # regularization
EPOCHS_2         = 30      # apna training budget
LEARNING_RATE_2  = 8e-4    # OneCycle ka peak LR
LABEL_SMOOTH_2   = 0.06    # label smoothing factor
AUGMENT_LLM2     = 2       # har image ke 2 extra jittered variants
JITTER_SCALE_2   = 0.04    # ~4% noise
ENSEMBLE_SIZE_2  = 5       # top-5 snapshot models average honge
```

---

## 8. Files jo LLM 2 generate/use karta hai

| File | Kya hai |
|---|---|
| `tiny_steel_llm2_bilstm_from_scratch.pt` | Trained model — isme **5 snapshot state-dicts ki list** save hoti hai (single model nahi) |
| `tiny_steel_llm2_tokenizer.json` | LLM 2 ka apna vocabulary/tokenizer (rich-prompt ke tokens se bana) |
| `scratch_llm2_results_all1800.json` | Saare 1800 images ka final accuracy/per-class result |
| `SteelSense-BiLSTM_batch_*.png` | Detection visualizations (50 images/class) |
| `SteelSense-BiLSTM_area_analysis.png` | Area-wise analysis chart |
| `both_llms_comparison.png` / `.json` | Dono LLMs ka side-by-side accuracy comparison |

---

## 9. Pehli baar run karne ka note

Architecture/checkpoint-format pehle wale single-model version se badal gaya hai (ab list-of-snapshots save hoti hai), isliye **Cell 7B me `force_retrain=True`** rakha gaya hai — pehli baar fresh training hogi. Uske baad agar dobara run karna ho to `force_retrain=False` kar sakte ho taaki saved ensemble seedha load ho jaye (training dobara na karni pade).




