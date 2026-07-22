"""End-to-end prediction orchestration.

Ties together preprocessing, feature extraction, KNN graph construction, the
GCN/GNN/hybrid models, the knowledge graph, reasoning and explanation into a
single frontend-ready result. Honesty is enforced here: when the hybrid network
is untrained, the authoritative class comes from the fitted MobileNetV2+KNN
baseline and every response carries a ``model_trained`` flag and notice.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from ..config import CLASS_NAMES, settings
from ..graph.graph_reasoner import (
    build_prediction_graph, compute_node_importance_from_patches,
)
from ..graph.graph_serializer import build_graph_payload, save_graph
from ..graph.knowledge_graph import KnowledgeGraph, property_activations
from ..models.hybrid_model import HybridInputs
from ..models.model_loader import ModelBundle, load_bundle
from ..reasoning.defect_reasoner import reason
from ..reasoning.explanation_generator import generate_explanation
from ..utils.feature_extraction import (
    FEATURE_NAMES, compute_named_features, dominant_visual_properties,
)
from ..utils.helpers import get_logger, timer
from ..utils.preprocessing import preprocess_image
from .graph_service import construct_graph

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


def _probs_dict(vec: np.ndarray) -> Dict[str, float]:
    return {CLASS_NAMES[i]: float(vec[i]) for i in range(len(CLASS_NAMES))}


def _aggregate_edge_attention_to_nodes(
    edge_index: np.ndarray, attention: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Sum incoming edge attention onto target nodes -> node importance."""
    imp = np.zeros(num_nodes, dtype=np.float32)
    for e in range(edge_index.shape[1]):
        t = int(edge_index[0, e])
        if e < len(attention):
            imp[t] += float(attention[e])
    return imp


def run_prediction(
    image_bgr: np.ndarray,
    prediction_id: str,
    filename: str = "upload.png",
    bundle: Optional[ModelBundle] = None,
) -> Dict:
    """Run the full pipeline and return the assembled result dict."""
    bundle = bundle or load_bundle()
    kg = _kg()
    timings: Dict[str, float] = {}

    # 1. Preprocess ------------------------------------------------------- #
    with timer() as t:
        pre = preprocess_image(image_bgr)
    timings["preprocess_ms"] = t["ms"]

    # 2. Global features -------------------------------------------------- #
    with timer() as t:
        global_named = compute_named_features(pre.gray)
        global_cnn = bundle.extractor.global_embedding(pre.resized_bgr)  # (1280,)
    timings["feature_extraction_ms"] = t["ms"]

    # 3. Region graph ----------------------------------------------------- #
    with timer() as t:
        region_graph, patch_features, patch_cnn = construct_graph(pre, bundle.extractor)
    timings["graph_construction_ms"] = t["ms"]

    # 4. Graph model inference ------------------------------------------- #
    device = bundle.device
    x = torch.from_numpy(region_graph.node_features).to(device)
    ei = torch.from_numpy(region_graph.edge_index).to(device)
    ew = torch.from_numpy(region_graph.edge_weight).to(device)

    kg_affinity = kg.affinity_vector(global_named)                 # (6,)
    hybrid_inp = HybridInputs(
        node_x=x, edge_index=ei, edge_weight=ew,
        cnn_global=torch.from_numpy(global_cnn.astype(np.float32)).to(device),
        cnn_patch=torch.from_numpy(patch_cnn.mean(0).astype(np.float32)).to(device),
        handcrafted=torch.from_numpy(
            np.array([global_named[n] for n in FEATURE_NAMES],
                     dtype=np.float32)).to(device),
        kg_embed=torch.from_numpy(kg_affinity).to(device),
    )
    with timer() as t, torch.no_grad():
        gcn_logits, _ = bundle.gcn(x, ei, ew)
        gnn_logits, _ = bundle.gnn(x, ei, ew)
        edge_att = bundle.gnn.edge_importance()
        hybrid_logits, hybrid_aux = bundle.hybrid(hybrid_inp)
    timings["graph_inference_ms"] = t["ms"]

    edge_att_np = edge_att.cpu().numpy() if edge_att is not None else None
    gates = bundle.hybrid.component_importance()

    # 5. Authoritative prediction (honesty policy) ----------------------- #
    source = bundle.primary_source()
    if source == "hybrid":
        prob_vec = _softmax(hybrid_logits.cpu().numpy()[0])
        model_trained = True
    elif source == "mobilenet_knn_baseline":
        prob_vec = bundle.knn_baseline.predict_proba(global_cnn)
        model_trained = True   # baseline is genuinely fitted
    else:  # untrained hybrid, no baseline
        prob_vec = _softmax(hybrid_logits.cpu().numpy()[0])
        model_trained = False

    pred_idx = int(np.argmax(prob_vec))
    predicted_class = CLASS_NAMES[pred_idx]
    confidence = float(prob_vec[pred_idx])
    probabilities = _probs_dict(prob_vec)
    untrained_notice = not bundle.hybrid_trained

    # 6. Node / edge importance ------------------------------------------ #
    if edge_att_np is not None and edge_att_np.size:
        node_importance = _aggregate_edge_attention_to_nodes(
            region_graph.edge_index, edge_att_np, region_graph.num_nodes)
        if node_importance.sum() == 0:
            node_importance = compute_node_importance_from_patches(
                patch_features, predicted_class, kg)
    else:
        node_importance = compute_node_importance_from_patches(
            patch_features, predicted_class, kg)

    # 7. Reasoning + explanation ----------------------------------------- #
    reasoning = reason(
        predicted_class=predicted_class, probabilities=probabilities,
        confidence=confidence, global_features=global_named,
        patch_features=patch_features, node_importance=node_importance,
        kg=kg, prediction_source=source,
    )
    explanation = generate_explanation(
        reasoning, global_named, untrained_notice=untrained_notice)

    # 8. Prediction graph + serialise ------------------------------------ #
    with timer() as t:
        pgraph = build_prediction_graph(
            region_graph=region_graph, patch_features=patch_features,
            global_features=global_named, predicted_class=predicted_class,
            probabilities=probabilities, confidence=confidence, kg=kg,
            node_importance=node_importance, edge_attention=edge_att_np,
            alternative_class=reasoning["alternative_prediction"],
        )
        graph_payload = build_graph_payload(pgraph)
        save_graph(prediction_id, graph_payload)
    timings["graph_build_ms"] = t["ms"]

    # 9. Model comparison (transparent trained/untrained labelling) ------ #
    comparison = _model_comparison(bundle, gcn_logits, gnn_logits, hybrid_logits, global_cnn)

    # 10. Important regions ---------------------------------------------- #
    imp_norm = node_importance / (node_importance.max() + 1e-9)
    order = np.argsort(imp_norm)[::-1][:4]
    important_regions = [
        {
            "patch": int(i),
            "bbox": pre.patches[int(i)].bbox(),
            "importance": round(float(imp_norm[int(i)]), 4),
            "dominant_property": max(
                property_activations(patch_features[int(i)]).items(),
                key=lambda kv: kv[1])[0],
        }
        for i in order
    ]

    result = {
        "prediction_id": prediction_id,
        "filename": filename,
        "image_url": f"/uploads/{Path(filename).name}",
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
        "prediction_source": source,
        "model_trained": model_trained,
        "hybrid_trained": bundle.hybrid_trained,
        "untrained_notice": untrained_notice,
        "reasoning": reasoning,
        "explanation": explanation["explanation"],
        "explanation_prompt": explanation["prompt"],
        "feature_statements": explanation["feature_statements"],
        "kg_statements": explanation["kg_statements"],
        "visual_features": dominant_visual_properties(global_named),
        "kg_affinity": _probs_dict(kg_affinity),
        "component_gates": gates,
        "important_regions": important_regions,
        "graph_summary": graph_payload["counts"],
        "model_comparison": comparison,
        "classes": CLASS_NAMES,
        "timings_ms": {k: round(v, 3) for k, v in timings.items()},
    }
    return result


def _model_comparison(bundle, gcn_logits, gnn_logits, hybrid_logits, global_cnn) -> List[Dict]:
    """Show each sub-model's top prediction with an explicit trained flag."""
    rows: List[Dict] = []

    if bundle.knn_baseline is not None:
        pv = bundle.knn_baseline.predict_proba(global_cnn)
        rows.append({
            "model": "MobileNetV2 + KNN (baseline)",
            "predicted_class": CLASS_NAMES[int(np.argmax(pv))],
            "confidence": round(float(pv.max()), 4),
            "trained": True,
        })

    for name, logits, trained in [
        ("GCN", gcn_logits, False),
        ("GAT (GNN)", gnn_logits, False),
        ("Hybrid (proposed)", hybrid_logits, bundle.hybrid_trained),
    ]:
        pv = _softmax(logits.cpu().numpy()[0])
        rows.append({
            "model": name,
            "predicted_class": CLASS_NAMES[int(np.argmax(pv))],
            "confidence": round(float(pv.max()), 4),
            "trained": bool(trained),
        })
    return rows
