# Novelty and Research Gap — GraphDefect-KG

> **Status of claims.** Everything below is a **proposed research contribution**. None of it
> should be presented as an established result until (a) a systematic literature review
> confirms the gap, and (b) controlled experiments on this repository demonstrate the effect.
> Do not cite unverified novelty in a paper.

## 1. Identified Research Gaps (to be confirmed by literature review)
1. **Grid assumption.** Most steel-defect classifiers treat the image as a regular pixel grid
   (CNNs) and do not explicitly model relationships *between* defect regions.
2. **Limited semantic reasoning.** CNN classifiers output a label with little grounded
   explanation of *why* a defect was predicted.
3. **Disconnected knowledge.** Few systems connect visual evidence to manufacturing-process
   knowledge (causes, mechanisms).
4. **Partial hybridisation.** Existing graph approaches often use either deep or handcrafted
   features, rarely fusing both *and* a domain knowledge graph on graph nodes.
5. **Non-interactive explanations.** Research prototypes seldom expose an interactive,
   node/edge-level view of the decision path.
6. **Ungrounded LLM explanations.** LLM-generated rationales can be plausible but
   disconnected from the model's actual evidence.
7. **Evidence-grounded GNN reasoning is under-explored** for surface-defect inspection.

## 2. Proposed Contributions (each marked PROPOSED)
- **C1 (PROPOSED).** A region-aware graph representation of steel surface images with patch
  nodes carrying *fused* MobileNetV2 + handcrafted descriptors.
- **C2 (PROPOSED).** KNN-based adaptive graph construction linking visually similar regions
  (cosine / Euclidean / combined spatial-feature distance).
- **C3 (PROPOSED).** A unified comparison of classical ML, CNN, graph and hybrid models for
  six-class NEU-DET classification within one reproducible framework.
- **C4 (PROPOSED).** A domain knowledge graph connecting defects ↔ visual properties ↔
  manufacturing causes, used as an auxiliary signal and reasoning substrate.
- **C5 (PROPOSED).** Graph-grounded natural-language explanations derived from real evidence
  (GAT attention, KG activations) rather than free-form generation.
- **C6 (PROPOSED).** An interactive, clickable web graph exposing node/edge-level
  contributions to the prediction.
- **C7 (PROPOSED).** Dual explainability — image-level (region importance) and graph-level
  (node/edge importance, KG paths).

## 3. How Each Claim Will Be Tested
| Claim | Test | Success criterion (to be set a priori) |
|-------|------|----------------------------------------|
| C1/C2 | Ablation: node features and graph metrics | graph model ≥ flat baseline under matched features |
| C3    | Unified benchmark table (seeded, CV) | statistically compared accuracy/F1 |
| C4    | With/without KG affinity component | measurable effect on accuracy and/or calibration |
| C5    | Faithfulness checks (perturbation, deletion) | explanation attributions track model behaviour |
| C6/C7 | User-facing evaluation / expert study | qualitative usefulness, not accuracy |

## 4. Threats to Validity
- Small per-image graphs; heuristic KG thresholds; single dataset; CPU-scale experiments in
  the notebook. Address via larger runs, multiple seeds, cross-dataset tests, and ablations
  documented in `ABLATION_STUDY.md`.
