# Ablation Study Plan — GraphDefect-KG

Goal: isolate the contribution of each component. **All cells are to be filled by executing
the notebook / training scripts.** Numbers must never be invented.

## 1. Configurations
| # | Configuration | CNN | Handcrafted | KNN graph | GCN | GNN | KG | Explanation |
|---|---------------|-----|-------------|-----------|-----|-----|----|-------------|
| 1 | MobileNetV2 only (KNN clf) | ✓ | | | | | | |
| 2 | Handcrafted only (KNN clf) | | ✓ | | | | | |
| 3 | KNN graph + GCN | ✓ | ✓ | ✓ | ✓ | | | |
| 4 | GCN only | ✓ | ✓ | ✓ | ✓ | | | |
| 5 | GNN (GAT) only | ✓ | ✓ | ✓ | | ✓ | | ✓ |
| 6 | MobileNetV2 + GCN | ✓ | | ✓ | ✓ | | | |
| 7 | MobileNetV2 + GNN | ✓ | | ✓ | | ✓ | | ✓ |
| 8 | MobileNetV2 + handcrafted + GNN | ✓ | ✓ | ✓ | | ✓ | | ✓ |
| 9 | Full model **without** KG | ✓ | ✓ | ✓ | ✓ | ✓ | | ✓ |
|10 | Full model **without** LLM explanation | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | |
|11 | Full model **without** KNN graph (fully connected) | ✓ | ✓ | | ✓ | ✓ | ✓ | ✓ |
|12 | **Full proposed hybrid** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## 2. Metrics recorded per configuration
Accuracy · macro-F1 · weighted-F1 · ROC-AUC · parameters · inference time (ms) ·
graph-construction time (ms) · qualitative interpretability note.

## 3. Results table (to be filled by execution)
| # | Config | Acc | macroF1 | ROC-AUC | Params | Infer ms | Notes |
|---|--------|-----|---------|---------|--------|----------|-------|
| 1 | | | | | | | |
| … | | | | | | | |
|12 | | | | | | | |

## 4. Analysis questions
- Does the KNN region graph help over a flat classifier at **matched** features (1/2 vs 4/5)?
- Marginal value of handcrafted features on top of CNN (6/7 vs 8)?
- Effect of the KG component (9 vs 12) on accuracy and calibration?
- Cost of full connectivity vs KNN sparsity (11 vs 12): accuracy and time.
- Does removing the explanation module (10) change accuracy? (It should not — it is
  post-hoc; use this as a sanity check.)

## 5. Explainability ablations (faithfulness)
- **Node deletion:** remove top-k important patches → expect confidence drop.
- **Edge perturbation:** shuffle high-attention edges → expect accuracy drop.
- **KG-path consistency:** fraction of predictions whose top evidence lies on a valid KG path.
