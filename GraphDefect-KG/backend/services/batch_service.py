"""Fast compact prediction for batch / whole-folder analysis.

Unlike :func:`prediction_service.run_prediction` (which builds the full
heterogeneous explanation graph and runs the GNN), this path is optimised for
throughput: preprocess -> MobileNetV2 global embedding -> KNN baseline class ->
knowledge-graph reason + short explanation -> important-region boxes from
handcrafted property matching. Suitable for scoring a folder of images at once.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import CLASS_NAMES
from ..graph.graph_reasoner import compute_node_importance_from_patches
from ..graph.knowledge_graph import KnowledgeGraph
from ..models.model_loader import ModelBundle, load_bundle
from ..reasoning.defect_reasoner import reason
from ..reasoning.explanation_generator import generate_explanation
from ..utils.feature_extraction import compute_named_features
from ..utils.helpers import get_logger, timer
from ..utils.preprocessing import preprocess_image

logger = get_logger(__name__)

_KG: Optional[KnowledgeGraph] = None


def _kg() -> KnowledgeGraph:
    global _KG
    if _KG is None:
        _KG = KnowledgeGraph.load_or_build()
    return _KG


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def predict_compact(
    image_bgr: np.ndarray,
    filename: str,
    bundle: Optional[ModelBundle] = None,
) -> Dict:
    """Score a single image quickly and return a compact result with reason."""
    bundle = bundle or load_bundle()
    kg = _kg()
    with timer() as t:
        pre = preprocess_image(image_bgr)
        global_named = compute_named_features(pre.gray)
        global_cnn = bundle.extractor.global_embedding(pre.resized_bgr)

        # class probabilities (honest source policy). Batch mode uses the fitted
        # MobileNetV2+KNN baseline; if it is unavailable we flag the result.
        if bundle.knn_baseline is not None:
            prob_vec = bundle.knn_baseline.predict_proba(global_cnn)
            source = "mobilenet_knn_baseline"
            model_trained = True
        else:
            prob_vec = _softmax(global_cnn[:len(CLASS_NAMES)])  # degenerate fallback
            source = "untrained"
            model_trained = False

    pred_idx = int(np.argmax(prob_vec))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(prob_vec[pred_idx])
    probabilities = {CLASS_NAMES[i]: float(prob_vec[i]) for i in range(len(CLASS_NAMES))}

    # per-patch handcrafted features -> important regions
    patch_named = [compute_named_features(p.image) for p in pre.patches]
    node_importance = compute_node_importance_from_patches(patch_named, predicted_class, kg)
    imp_norm = node_importance / (node_importance.max() + 1e-9)
    order = np.argsort(imp_norm)[::-1][:3]
    important_regions = [
        {"patch": int(i), "bbox": pre.patches[int(i)].bbox(),
         "importance": round(float(imp_norm[int(i)]), 3)}
        for i in order
    ]

    reasoning = reason(
        predicted_class=predicted_class, probabilities=probabilities,
        confidence=confidence, global_features=global_named,
        patch_features=patch_named, node_importance=node_importance,
        kg=kg, prediction_source=source,
    )
    explanation = generate_explanation(
        reasoning, global_named, untrained_notice=not bundle.hybrid_trained)

    return {
        "filename": filename,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "prediction_source": source,
        "model_trained": model_trained,
        "reason": explanation["explanation"],
        "visual_properties": reasoning["visual_properties"][:4],
        "causes": reasoning["class_causes"],
        "knowledge_graph_path": reasoning["knowledge_graph_path"],
        "important_regions": important_regions,
        "image_size": pre.size,
    }


def predict_many(images: List[Dict], bundle: Optional[ModelBundle] = None) -> Dict:
    """Score a list of ``{filename, image_bgr}`` dicts. Returns summary + results."""
    bundle = bundle or load_bundle()
    results: List[Dict] = []
    with timer() as t:
        for item in images:
            try:
                results.append(predict_compact(item["image_bgr"], item["filename"], bundle))
            except Exception as exc:  # keep the batch resilient
                logger.warning("Batch item %s failed: %s", item.get("filename"), exc)
                results.append({"filename": item.get("filename", "?"), "error": str(exc)})

    ok = [r for r in results if "error" not in r]
    counts: Dict[str, int] = {c: 0 for c in CLASS_NAMES}
    for r in ok:
        counts[r["predicted_class"]] += 1
    return {
        "count": len(results),
        "succeeded": len(ok),
        "failed": len(results) - len(ok),
        "class_distribution": counts,
        "elapsed_ms": round(t["ms"], 1),
        "results": results,
    }
