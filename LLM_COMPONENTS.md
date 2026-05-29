# Steel Surface Defect Detection - Local Scratch LLM Components

## Overview
This document details all the components, libraries, and techniques used to build the **TinySteelLLM** - a from-scratch Transformer-based classifier for steel surface defect detection. **No external LLM API (OpenRouter, Groq, Claude, Gemini) is used.**

---

## 1. Core Libraries & Dependencies

### Deep Learning Framework
- **PyTorch** (`torch`, `torch.nn`, `torch.nn.functional`)
  - Neural network modules (Embedding, Linear, LayerNorm, Dropout)
  - Transformer encoder implementation
  - Optimization (AdamW)
  - CUDA support for GPU acceleration

### Computer Vision
- **OpenCV** (`cv2`)
  - Image reading and preprocessing (BGR to RGB conversion)
  - Image resizing to 200x200 standard size
  - Edge detection (Canny algorithm)
  - Sobel and Laplacian operators for feature extraction
  - Morphological operations (closing, opening)
  - Contour detection and analysis
  - Gaussian blur for preprocessing

- **scikit-image** (`skimage.feature`)
  - GLCM (Gray-Level Co-occurrence Matrix) computation
  - Texture analysis properties (contrast, homogeneity, energy, correlation)

### Numerical & Data Processing
- **NumPy** (`numpy`)
  - Array operations and statistical calculations
  - FFT (Fast Fourier Transform) for frequency domain analysis
  - Histogram computation for entropy calculation
  - Mathematical operations for feature engineering

- **Matplotlib** (`matplotlib.pyplot`, `matplotlib.patches`)
  - Visualization of detection results
  - Batch image plotting with bounding boxes
  - Statistical charts and analysis plots

- **Collections** (`Counter`)
  - Token frequency counting for tokenizer vocabulary building

### File & Data Handling
- **json** - Configuration and results serialization
- **xml.etree.ElementTree** - XML annotation parsing
- **os** - File system operations
- **random** - Seed management for reproducibility
- **re** - Regular expression for text tokenization
- **hashlib** - (imported but not actively used)
- **time** - Performance timing

---

## 2. Image Feature Extraction (Cell 3)

### Rich Feature Set Extracted per Image

#### **Basic Statistical Features**
- `mean_intensity` - Average pixel intensity
- `std_intensity` - Standard deviation of intensity
- `contrast` - Ratio of standard deviation to mean
- `entropy` - Shannon entropy of pixel distribution

#### **Edge & Gradient Features**
- `edge_density` - Percentage of edge pixels detected by Canny
- `sobel_magnitude` - Average magnitude from Sobel operators
- `laplacian_var` - Variance of Laplacian operator
- `angle_variance` - Variance of gradient directions
- `horizontal_edge_ratio` - Proportion of vertical gradients
- `vertical_edge_ratio` - Proportion of horizontal gradients

#### **Morphological Region Analysis**
- `num_regions` - Count of connected defect regions
- `avg_area`, `max_area` - Average and max region area
- `total_defect_pct` - Percentage of image covered by defects
- `avg_circularity`, `min_circularity` - Shape roundness measure
- `avg_aspect_ratio`, `max_aspect_ratio` - Width-to-height ratios
- `regions_tiny`, `regions_small`, `regions_medium`, `regions_large` - Histogram of region sizes

#### **Quadrant Analysis**
- `quadrant_std` - Standard deviation per quadrant (top-left, top-right, bottom-left, bottom-right)
- `quadrant_edge_density` - Edge density per quadrant
- `most_active_quadrant` - Quadrant with highest variation
- `edge_row_peak_pct`, `edge_col_peak_pct` - Location of peak edge activity

#### **Brightness Distribution**
- `dark_pixel_pct` - Pixels with intensity < 80
- `bright_pixel_pct` - Pixels with intensity > 180
- `mid_pixel_pct` - Pixels in mid-range (80-180)

#### **Frequency Domain**
- `fft_center_energy` - Energy concentration in central frequency band

#### **Texture Features (GLCM)**
- `glcm_contrast` - Contrast measure from co-occurrence matrix
- `glcm_homogeneity` - Local homogeneity
- `glcm_energy` - Angular second moment
- `glcm_correlation` - Correlation in texture

**Total: ~35 numerical features per image**

---

## 3. Feature-to-Prompt Conversion (Cell 4)

### Tokenization Strategy
The extracted numerical features are converted into **discrete categorical tokens** for the Transformer:

```
steel surface defect classification
regions_14 tiny_10 small_4 medium_0 large_0
entropy_high area_small aspect_medium coverage_medium
dark_medium edge_medium active_top_left
h_edge_0.6 v_edge_0.2 circularity_0.7
glcm_contrast_2.1 glcm_energy_3.5
```

### Feature Binning
Continuous features are discretized into categorical bins:
- **Entropy**: very_low (0-4) → low (4-5.5) → medium (5.5-6.5) → high (6.5-10)
- **Avg Area**: tiny (0-100) → small (100-500) → medium (500-2500) → large (2500+)
- **Max Aspect Ratio**: low (0-1.8) → medium (1.8-4) → high (4-8) → very_high (8+)
- **Coverage %**: low (0-5) → medium (5-15) → high (15-35) → very_high (35-100)
- **Dark Pixels %**: low (0-8) → medium (8-20) → high (20-40) → very_high (40+)
- **Edge Density**: low (0-4) → medium (4-10) → high (10-20) → very_high (20+)

---

## 4. Tokenizer Implementation (Cell 5)

### SimpleTokenizer Class
**Purpose**: Convert natural language feature prompts into token ID sequences

**Key Methods**:
- `tokenize(text)` - Splits text by alphanumeric patterns (regex: `[a-zA-Z0-9_.\-]+`)
- `build(texts, min_freq)` - Builds vocabulary from training texts (min frequency threshold)
- `encode(text)` - Converts text to fixed-length token ID sequences
- `save(path)` / `load(path)` - Persistence for trained tokenizers

**Special Tokens**:
- `<pad>` (ID: 0) - Padding token
- `<unk>` (ID: 1) - Unknown token
- `<cls>` (ID: 2) - Classification token (prepended to all sequences)

**Encoding Process**:
1. Prepend `<cls>` token
2. Tokenize text into words
3. Map each word to vocabulary ID (or UNK if not in vocab)
4. Pad/truncate to `MAX_LEN=160` tokens
5. Create attention mask (1 for real tokens, 0 for padding)

---

## 5. TinySteelLLM Architecture (Cell 5)

### Model Design
A lightweight Transformer encoder for text classification on feature prompts.

```
Input (Token IDs)
    ↓
Token Embedding (vocab_size → 96 dims)
    ↓
Positional Embedding (max_len → 96 dims)
    ↓
Element-wise Addition
    ↓
Transformer Encoder (2 layers, 4 heads, 96 dims)
    ↓
Layer Normalization
    ↓
Extract [CLS] token representation
    ↓
Classification Head (96 → 96 → 6 classes)
    ↓
Logits → Softmax → Class Probability
```

### Model Configuration
| Parameter | Value | Purpose |
|-----------|-------|---------|
| **vocab_size** | ~500-600 | Number of unique tokens |
| **embed_dim** | 96 | Token embedding dimension |
| **num_heads** | 4 | Multi-head attention heads |
| **num_layers** | 2 | Transformer encoder layers |
| **max_len** | 160 | Maximum sequence length |
| **dropout** | 0.15 | Regularization |
| **hidden_dim** | 384 (4×embed_dim) | FFN intermediate dim |

### Components
1. **Token Embedding** - Learns dense representations for each token
2. **Positional Embedding** - Encodes token position information
3. **Transformer Encoder** - Multi-head self-attention + feed-forward layers
4. **Layer Normalization** - Stabilizes training
5. **Classification Head** - 3-layer MLP with GELU activation

---

## 6. Training Pipeline (Cell 5)

### Dataset Preparation
- **SteelPromptDataset** - PyTorch Dataset class
  - Wraps feature prompts + labels
  - Tokenizes on-the-fly during loading
  - Returns: `input_ids`, `attention_mask`, `label`

### Data Splitting
- **Stratified Train-Val Split** (80-20 ratio)
  - Maintains class distribution in both splits
  - Prevents data leakage

### Training Configuration
| Hyperparameter | Value |
|---|---|
| **Batch Size** | 32 |
| **Epochs** | 18 |
| **Learning Rate** | 3e-4 |
| **Optimizer** | AdamW (weight_decay=1e-4) |
| **Loss Function** | Cross-Entropy |
| **Gradient Clip** | 1.0 |
| **Device** | GPU (CUDA) or CPU |

### Training Loop
1. Forward pass through model
2. Compute cross-entropy loss
3. Backward pass with gradient clipping
4. Adam optimizer step
5. Validation every epoch
6. Save best model based on validation accuracy

### Early Stopping
- Tracks best validation accuracy
- Saves best model state during training
- Loads best state after training completes

---

## 7. Local Defect Area Detection (Cell 6)

### Two-Tier Area Detection Strategy

#### **Tier 1: XML Annotations (Preferred)**
- Parse bounding boxes from NEU-DET XML annotation files
- Map coordinates to resized 200×200 image space
- Extract region area, label, source metadata

#### **Tier 2: Contour Detection (Fallback)**
- Applies morphological operations:
  - Gaussian blur
  - Canny edge detection
  - Binary thresholding (Otsu's method)
  - Morphological close & open operations
- Finds connected contours/regions
- Filters by minimum area threshold (60 pixels)
- Limits to top 8 regions by area
- If no regions found, uses full image as fallback

### Detection Pipeline
```
Image → Extract Features → LLM Classification → Area Detection
                                                   ├─ XML? → Use XML boxes
                                                   └─ No XML? → Contour detection
```

---

## 8. Evaluation Metrics (Cell 11-12)

### Per-Class Metrics
- **True Positives (TP)** - Correct classifications
- **Accuracy per class** - TP / Total for each defect type
- **Confusion** - Tracked implicitly through per-class accuracy

### Overall Metrics
- **Overall Accuracy** - Total correct / Total images
- **Processing Time** - Total elapsed time + ETA estimation

### Class Distribution
Targets 95%+ accuracy on all 6 defect classes:
- crazing
- inclusion
- patches
- pitted_surface
- rolled-in_scale
- scratches

---

## 9. Visualization Components (Cell 8, 10, 13)

### Batch Visualization
- **Grid Display** - 6×N image grid with bounding boxes
- **Color Coding** - Defect types use distinct colors (DEFECT_COLORS dict)
- **Source Indicators** - Box colors indicate detection source (XML, LLM, contour, fallback)
- **Metadata** - Per-image: predicted label, confidence, region count, detection method

### Statistical Plots
1. **Per-Class Accuracy Chart** - Bar plot comparing class-wise accuracy
2. **Method Distribution Pie** - Classification method usage (should be "scratch_tiny_llm" 100%)
3. **Area Analysis Chart**:
   - Distribution of images per class
   - Average confidence per class
   - Average regions per image
   - Average defect area percentage
   - Area detection method breakdown

---

## 10. Results & Persistence (Cell 11)

### JSON Results Structure
```json
{
  "model": "TinySteelLLM_FromScratch",
  "total": 1800,
  "correct": 1710,
  "accuracy": "95.00%",
  "time_sec": 245.3,
  "note": "Local from-scratch tiny Transformer classifier...",
  "per_class": {
    "crazing": {
      "correct": 285,
      "total": 300,
      "accuracy": "95.00%"
    },
    ...
  },
  "cls_methods": {
    "scratch_tiny_llm": 1800
  },
  "area_methods": {
    "xml_annotation": 900,
    "contour_detection": 900
  }
}
```

### Output Directory
- **MODEL_PATH**: Trained PyTorch model checkpoint (.pt)
- **TOKENIZER_PATH**: Serialized tokenizer vocabulary (.json)
- **RESULTS_JSON**: Final accuracy report (.json)
- **Visualization PNGs**: Batch detection images & analysis charts

---

## 11. Key Design Decisions

### Why From-Scratch LLM?
✅ No external API dependency (privacy, cost, latency)  
✅ Reproducible, deterministic results  
✅ Full control over model architecture  
✅ Learning on domain-specific feature prompts  

### Why Feature-Based Prompts?
✅ Compact representation (35 features → token sequence)  
✅ Interpretable (bins have semantic meaning)  
✅ Fast inference (no image encoding needed)  
✅ Leverages domain knowledge of steel defects  

### Why Stratified Split?
✅ Ensures balanced class distribution  
✅ Prevents overfitting on rare classes  
✅ More reliable validation metrics  

### Why Tiny Transformer?
✅ Lightweight (96 embedding dims, 2 layers)  
✅ Fast training (~18 epochs)  
✅ Sufficient for token classification task  
✅ Scales to full 1800 images efficiently  

---

## 12. Reproducibility Settings

**Seed Management**:
```python
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if GPU available
```

**Deterministic Operations**:
- Stratified split uses `random.Random(SEED)`
- All randomness controlled for reproducible results

---

## Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Image I/O** | OpenCV | Load, resize, convert color spaces |
| **Feature Extraction** | NumPy, scikit-image, OpenCV | Extract 35+ numerical features |
| **Feature→Prompt** | Python regex, binning | Convert features to token sequences |
| **Tokenization** | Custom SimpleTokenizer | Build vocabulary, encode sequences |
| **Deep Learning** | PyTorch Transformer | Train classifier on feature prompts |
| **Training** | PyTorch DataLoader, AdamW | Stratified split, cross-entropy loss |
| **Evaluation** | NumPy, collections | Per-class accuracy, confusion tracking |
| **Visualization** | Matplotlib | Batch plots, analysis charts |
| **Persistence** | JSON, PyTorch .pt | Save/load model, tokenizer, results |
| **Reproducibility** | Random seeds, stratification | Deterministic results |

---

**Total from-scratch LLM components: 1 Transformer + 1 Tokenizer trained locally on steel defect feature prompts. No external APIs used.**
