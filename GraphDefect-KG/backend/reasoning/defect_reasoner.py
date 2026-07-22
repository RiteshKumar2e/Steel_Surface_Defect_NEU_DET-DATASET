"""Rule-based, evidence-grounded defect reasoner.

Combines the model prediction, graph evidence and knowledge-graph relations into
a structured reasoning object. Deliberately transparent: every conclusion cites
the evidence it came from, so explanations stay grounded in real model signals
rather than free-form generation.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..config import CLASS_NAMES
from ..graph.knowledge_graph import KnowledgeGraph


def _alternative_class(probabilities: Dict[str, float], predicted: str) -> str:
    ranked = sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
    for cls, _ in ranked:
        if cls != predicted:
            return cls
    return predicted


def reason(
    *,
    predicted_class: str,
    probabilities: Dict[str, float],
    confidence: float,
    global_features: Dict[str, float],
    patch_features: List[Dict[str, float]],
    node_importance: np.ndarray,
    kg: KnowledgeGraph,
    prediction_source: str,
) -> Dict:
    """Produce the structured reasoning payload."""
    top_props = kg.top_properties(global_features, k=5)
    pred_props = set(kg.class_properties(predicted_class))

    supporting: List[Dict] = []
    contradicting: List[Dict] = []
    for prop, score in top_props:
        entry = {"property": prop, "activation": round(float(score), 3)}
        if prop in pred_props:
            supporting.append(entry)
        else:
            contradicting.append(entry)

    # Important patches (indices) by node importance.
    if node_importance is not None and node_importance.size:
        order = np.argsort(node_importance)[::-1]
        important_patches = [int(i) for i in order[:min(4, len(order))]]
    else:
        important_patches = []

    kg_path = kg.reasoning_path(predicted_class, global_features)
    alternative = _alternative_class(probabilities, predicted_class)

    # Confidence-margin sanity flag (transparency, not fabricated certainty).
    ranked = sorted(probabilities.values(), reverse=True)
    margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float(ranked[0])

    return {
        "predicted_class": predicted_class,
        "confidence": round(float(confidence), 4),
        "confidence_margin": round(margin, 4),
        "prediction_source": prediction_source,
        "supporting_evidence": supporting,
        "contradicting_evidence": contradicting,
        "important_patches": important_patches,
        "visual_properties": [p for p, _ in top_props],
        "knowledge_graph_path": kg_path,
        "class_causes": kg.class_causes(predicted_class),
        "alternative_prediction": alternative,
    }
