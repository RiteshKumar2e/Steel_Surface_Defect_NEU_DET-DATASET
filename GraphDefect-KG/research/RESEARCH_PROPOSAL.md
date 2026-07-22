# Research Proposal — GraphDefect-KG

**Full title:** GraphDefect-KG: Knowledge-Guided Graph Neural Learning with MobileNetV2 for Explainable Steel Surface Defect Detection

## 1. Background and Motivation
Automated visual inspection of steel surfaces is a core task in manufacturing quality
control. Convolutional neural networks (CNNs) achieve strong accuracy on benchmarks such as
NEU-DET, but they treat an image as a regular pixel grid and expose little about *why* a
defect was predicted. In safety- and cost-sensitive industrial settings, an inspector needs
not only a label but **evidence**: which regions matter, which visual properties were
detected, and how they relate to known defect mechanisms.

## 2. Problem Statement
Given a single-channel steel surface image, classify it into one of six NEU-DET defect
categories — *Crazing, Inclusion, Patches, Pitted Surface, Rolled-in Scale, Scratches* — and
produce a **graph-grounded, human-readable explanation** that ties the decision to visual
evidence and domain knowledge.

## 3. Proposed Approach (overview)
1. **Preprocessing** — resize, CLAHE, denoising, edge and region extraction, patch tiling.
2. **Feature extraction** — MobileNetV2 deep embeddings + a 32-dimensional handcrafted
   descriptor (statistical, GLCM, LBP, shape).
3. **Region graph** — patch nodes carry fused features; edges are built by KNN over feature
   (and optionally spatial) similarity.
4. **Graph learning** — GCN and GAT/GraphSAGE learn graph-level representations.
5. **Hybrid fusion** — a gated fusion of CNN-global, CNN-patch, handcrafted, GCN, GNN and a
   knowledge-graph affinity vector feeds the six-class head.
6. **Knowledge graph + reasoning** — a curated domain KG links defects to visual properties
   and manufacturing causes, producing evidence-grounded explanations and an interactive,
   clickable graph.

## 4. Objectives
- **O1.** Build a reusable, modular framework shared by a research notebook and a web API.
- **O2.** Represent images as region graphs fusing deep and handcrafted features.
- **O3.** Compare classical ML, CNN, graph and hybrid models under one protocol.
- **O4.** Ground explanations in real model evidence (attention, KG paths).
- **O5.** Deliver an interactive graph interface for inspecting the decision path.

## 5. Datasets
- **NEU-DET** (1,800 images, 6 classes, 300 each) with PASCAL-VOC bounding-box annotations.
- Splits are created per-experiment with stratification; see `EXPERIMENT_PLAN.md`.

## 6. Evaluation
Accuracy, macro/weighted F1, precision, recall, ROC-AUC, PR-AUC, confusion matrices, plus
efficiency metrics (parameters, inference/graph-construction time). Localisation metrics
(IoU / mAP) where annotations are used.

## 7. Expected Contributions
See `NOVELTY_AND_RESEARCH_GAP.md`. Each contribution is stated as a **proposed** claim to be
validated against a literature review and controlled experiments — not asserted as settled.

## 8. Risks and Mitigations
- *Small graphs per image* → limited message-passing depth. Mitigation: patch-grid tuning,
  region nodes, residual connections.
- *Rule-based KG affinity is a heuristic prior.* Mitigation: it is used only as an auxiliary
  signal and explanation aid; the learned models drive classification.
- *Compute limits.* Mitigation: MobileNetV2 backbone, CPU-friendly native message passing.

## 9. Reproducibility
All feature/graph/model code is shared between `backend/` and `notebooks/model_code.ipynb`.
No metric in this repository is hard-coded; every reported number must be produced by
executing the notebook or training scripts on the user's machine.
