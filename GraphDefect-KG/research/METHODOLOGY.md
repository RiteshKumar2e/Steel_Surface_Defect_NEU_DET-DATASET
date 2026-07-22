# Methodology — GraphDefect-KG

This document specifies the full pipeline precisely enough to reproduce it. Implementation
lives in `backend/` and is reused by the notebook.

## 1. Preprocessing (`backend/utils/preprocessing.py`)
Input BGR image → resize to `IMAGE_SIZE` (default 224) → grayscale → Gaussian denoise →
optional CLAHE (clip 2.0, 8×8 grid) → Canny edges (50/150). Additional derived structures:
- **Patches:** an `N×N` grid (default 4×4 → 16 patches). Each patch keeps its bounding box.
- **Regions:** Otsu threshold + connected components (unsupervised region proxy).
- **Contours:** external contours for shape descriptors.

## 2. Feature Extraction
### 2.1 MobileNetV2 (`models/mobilenet_feature_extractor.py`)
Torchvision MobileNetV2 (ImageNet weights, classifier removed). Global average pooling →
1280-d embedding. Applied to the whole image and to each patch. A projection head
(`Linear → LayerNorm → GELU`) reduces 1280→`projected_dim` (128) with optional multi-head
self-attention refinement across patch nodes. Backbone frozen by default; fine-tuning
supported.

### 2.2 Handcrafted descriptor (`utils/feature_extraction.py`)
A fixed 32-d vector per image/patch:
mean, std, entropy, edge density, Sobel response, Laplacian variance, GLCM
(contrast/correlation/energy/homogeneity), 10-bin uniform LBP histogram, contour area,
perimeter, circularity, aspect ratio, extent, solidity, orientation, compactness, roughness,
bbox width/height, contour count. Values are scaled to roughly `[0, 1]` for stable graphs.

## 3. Region Graph Construction (`models/knn_graph_builder.py`)
Node feature = `[projected CNN patch embedding (128) | handcrafted (32)]` → 160-d.
- **Distance:** cosine (default), Euclidean, or a `combined` feature+spatial distance.
- **KNN:** each node connects to its `K` (default 5) nearest neighbours; weight
  `w_ij = exp(-d_ij) ∈ (0,1]`.
- **Connectivity guarantees:** symmetrisation, nearest-neighbour rescue for isolated nodes,
  optional self-loops. Symmetric normalisation `D^{-1/2} A D^{-1/2}` available.

## 4. Graph Learning
### 4.1 GCN (`models/gcn_model.py`)
Input projection → `L` GCN layers (`GCNConv → BatchNorm → GELU → Dropout`, residual) →
global mean+max pooling → MLP → 6 logits. Message passing is implemented with native
`scatter_add` so it runs without the optional `torch-scatter` extension.

### 4.2 GraphSAGE / GAT (`models/gnn_model.py`)
- **GraphSAGE:** mean-aggregation with separate self/neighbour transforms.
- **GAT:** multi-head attention; per-edge attention weights are stored and surfaced as **edge
  importance** for explainability.

### 4.3 Hybrid model (`models/hybrid_model.py`)
Six components — CNN-global, CNN-patch (mean), handcrafted, GCN embedding, GNN embedding, KG
affinity — are each projected to a shared width, weighted by a **softmax gate**, concatenated
and passed to an MLP head. Gate weights are exposed as component importance.

## 5. Knowledge Graph (`graph/knowledge_graph.py`)
A curated directed graph: `DefectClass --has_property--> VisualProperty` and
`DefectClass --associated_with--> Cause`. From an image's interpretable features we compute
soft **property activations** (heuristic, expert-set thresholds) and a per-class **affinity
vector** (mean activation of that class's properties). This is an auxiliary prior and
explanation aid — **not** the classifier. Stored as JSON (Neo4j-ready).

## 6. Prediction Graph & Reasoning (`graph/graph_reasoner.py`, `reasoning/`)
The prediction graph stitches image → patches → feature-summary/evidence → prediction →
class → knowledge/cause nodes, with real importance signals (GAT attention aggregated to
nodes; KG activations). The reasoner assembles supporting/contradicting evidence, important
patches, KG paths and the alternative class. A template generator produces a fluent,
evidence-grounded explanation (optional local Transformer hook; never required).

## 7. Honest Prediction Policy (`models/model_loader.py`, `services/prediction_service.py`)
- If a trained **hybrid** checkpoint exists → it is authoritative.
- Else if the fitted **MobileNetV2+KNN** baseline exists → it is authoritative (genuinely
  trained, non-parametric).
- Else → the untrained hybrid's output is returned **flagged as invalid** (`model_trained =
  false`). Every response carries `prediction_source` and `untrained_notice`.

## 8. Serving (`backend/main.py`, `api/`)
FastAPI exposes `/api/predict`, `/api/graph/{id}`, `/api/prediction/{id}`,
`/api/download-report/{id}`, `/api/classes`, `/api/model-info`, `/api/health`. The graph is
serialised to Cytoscape.js elements; the frontend renders it interactively.
