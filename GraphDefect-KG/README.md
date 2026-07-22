# GraphDefect-KG

**Knowledge-Guided Graph Neural Learning for Explainable Industrial Surface Defect Detection**

GraphDefect-KG turns a steel surface image into a **region graph**, learns over it with
**GCN / GNN** models, fuses deep + handcrafted + knowledge-graph signals in a hybrid
classifier, and explains every prediction through an **interactive, clickable graph** — all
served by a light, academic FastAPI web app and reproduced in a research notebook.

Six NEU-DET classes: *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale,
Scratches*.

> **This is a research prototype.** The deep hybrid graph model ships **untrained**. Out of
> the box, the authoritative class comes from a genuinely fitted **MobileNetV2 + KNN**
> baseline (fits in seconds from cached backbone features). Untrained model outputs are
> always flagged (`model_trained`, `prediction_source`, `untrained_notice`) and never
> presented as valid research results. Train the models via the notebook for
> publication-grade numbers.

---

## 1. Pipeline

```
Input Image → Preprocess → MobileNetV2 + Handcrafted features
           → Patch/Region nodes → KNN graph → GCN + GAT
           → Gated hybrid fusion → 6-class prediction
           → Knowledge graph reasoning → grounded explanation
           → Interactive web graph
```

See `research/ARCHITECTURE.md` for full diagrams and schemas.

## 2. Project layout
```
GraphDefect-KG/
├── backend/          # FastAPI app, models, graph, reasoning, services, utils
├── frontend/         # index.html, results.html, css/, js/ (+ vendored cytoscape)
├── notebooks/        # model_code.ipynb (33 sections, runs top-to-bottom)
├── research/         # proposal, methodology, novelty, experiment/ablation plans, paper outline
├── data/             # raw / processed / annotations / graph_data
├── tests/            # pytest suite
└── run.py            # server entry point
```

## 3. Setup

Requires **Python 3.10+**.

```bash
# from the GraphDefect-KG/ directory
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/mac: source .venv/bin/activate

# 1) install PyTorch for your platform (CPU example):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# 2) install the rest:
pip install -r backend/requirements.txt
```

`torch-geometric` is optional — the graph models use a built-in native message-passing
implementation and run without it.

### Dataset
Point the app at NEU-DET (auto-discovered if it sits at `../NEU-DET/IMAGES` relative to this
folder). Otherwise set in `.env` (copy from `.env.example`):
```
NEU_DET_IMAGES=/path/to/NEU-DET/IMAGES
NEU_DET_ANNOTATIONS=/path/to/NEU-DET/ANNOTATIONS
```
Expected structure: `IMAGES/<class_folder>/*.jpg` (crazing, inclusion, patches,
pitted_surface, rolled-in_scale, scratches).

## 4. Run the web app
```bash
python run.py                 # http://127.0.0.1:8000
python run.py --port 8080 --reload
```
On the first prediction the MobileNetV2+KNN baseline is fitted and cached to
`backend/saved_models/knn_baseline.joblib` (needs the dataset). Then:
1. Open `http://127.0.0.1:8000`, drag in a steel surface image, click **Analyse defect**.
2. The results page shows the prediction, probabilities, evidence, KG reasoning path, model
   comparison, and the **interactive prediction graph** (click nodes/edges for details;
   filter, search, highlight the prediction path).
3. Download the JSON report.

Interactive API docs: `http://127.0.0.1:8000/docs`.

## 5. Train the models (for real results)
Open the notebook and run it top to bottom:
```bash
jupyter notebook notebooks/model_code.ipynb
```
It builds the region-graph dataset, trains GCN/GAT (real gradient descent), evaluates on a
held-out test set, runs baselines/ablations/explainability, and saves checkpoints. To make
the **web API** use a trained deep model, train a `HybridDefectModel` and save its
`state_dict` under key `"model"` to `backend/saved_models/hybrid_model.pt` — the loader picks
it up automatically and switches `prediction_source` to `hybrid`.

Increase `SAMPLES_PER_CLASS` and `EPOCHS` in the notebook for meaningful numbers (defaults
are a fast smoke test). Follow `research/EXPERIMENT_PLAN.md` and `research/ABLATION_STUDY.md`
for the full protocol.

## 6. API endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | landing page |
| POST | `/api/predict` | analyse an image → full result JSON |
| POST | `/api/build-graph` | analyse and return only the graph |
| GET | `/api/graph/{id}` | Cytoscape graph elements |
| GET | `/api/prediction/{id}` | stored prediction result |
| GET | `/api/download-report/{id}` | downloadable JSON report |
| GET | `/api/classes` | class list + KG properties/causes |
| GET | `/api/model-info` | backbone + model status |
| GET | `/api/health` | health + training status |

## 7. Tests
```bash
pip install pytest
pytest tests -q          # run from GraphDefect-KG/
```
Tests are dataset-free (synthetic images) except where noted.

## 8. Honesty & limitations
- No fabricated metrics anywhere. Every number must come from executing the notebook.
- The knowledge-graph affinity is a **rule-based domain prior / explanation aid**, not a
  learned classifier.
- Per-image graphs are small; the KG thresholds are heuristic; experiments in the notebook
  run at CPU scale by default. See `research/NOVELTY_AND_RESEARCH_GAP.md` (all novelty is
  marked *proposed* pending literature review and controlled experiments).

## 9. Configuration
All knobs live in `backend/config.py` and can be overridden via environment variables /
`.env` (image size, patch grid, KNN `K`/metric, pretrained weights, device, upload limits).

---
GraphDefect-KG · research prototype for explainable industrial inspection.
Predictions are model outputs, not certified inspection results.
