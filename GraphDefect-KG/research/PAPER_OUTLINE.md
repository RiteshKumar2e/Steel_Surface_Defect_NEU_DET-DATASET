# Paper Outline — GraphDefect-KG

**Working title:** GraphDefect-KG: Knowledge-Guided Graph Neural Learning with MobileNetV2
for Explainable Steel Surface Defect Detection

**Alternative title:** A Knowledge Graph-Enhanced GNN Framework for Explainable Six-Class
Industrial Surface Defect Recognition

## 1. Title
## 2. Abstract
150–250 words: problem, gap, approach (region graph + GCN/GNN + KG), one-line results
placeholder (fill after experiments), explainability contribution.

## 3. Keywords
Surface defect detection; graph neural networks; knowledge graph; explainable AI;
MobileNetV2; NEU-DET; industrial inspection.

## 4. Introduction
Industrial inspection context; limitations of pixel-grid CNNs; need for grounded explanation;
summary of contributions (link each to `NOVELTY_AND_RESEARCH_GAP.md`, marked proposed).

## 5. Research Motivation
Why relationships between regions and manufacturing knowledge matter for trust and diagnosis.

## 6. Research Gap
Condensed from `NOVELTY_AND_RESEARCH_GAP.md`; supported by the related-work survey.

## 7. Related Work
CNNs for defect detection; graph neural networks in vision; knowledge graphs in
manufacturing; explainable AI / GNN explainers; LLM explanations and their grounding problem.

## 8. Proposed Methodology (overview + figure)
System diagram (see `ARCHITECTURE.md`).

## 9. Image Preprocessing
## 10. Feature Extraction (MobileNetV2 + handcrafted)
## 11. Graph Construction (KNN region graph)
## 12. GCN and GNN Learning
## 13. Knowledge Graph Reasoning
## 14. Hybrid Classification (gated fusion)
## 15. Explainability (image- and graph-level)
## 16. Experimental Setup
Hardware, hyperparameters, splits, seeds (link `EXPERIMENT_PLAN.md`).
## 17. Dataset (NEU-DET)
## 18. Evaluation Metrics
## 19. Results
Main comparison table; ROC/PR; confusion matrix. **Numbers filled from execution only.**
## 20. Comparison with Baselines
Classical ML vs CNN vs graph vs hybrid.
## 21. Ablation Study
From `ABLATION_STUDY.md`.
## 22. Error Analysis
Confusable class pairs; failure cases with graph explanations.
## 23. Discussion
When graphs help; role of the KG; interpretability findings; efficiency trade-offs.
## 24. Limitations
Small graphs, heuristic KG thresholds, single dataset, compute scale.
## 25. Conclusion
## 26. Future Work
Neo4j-backed KG; learned property detectors; region-level detection/localisation; grounded
local LLM; cross-dataset generalisation; user study with inspectors.

### Suggested figures/tables
- F1: system architecture. F2: region graph + KG example. F3: attention heatmap.
- F4: interactive graph screenshot. T1: main results. T2: ablation. T3: efficiency.
