"""Builds the heterogeneous *prediction graph* that explains a single inference.

This is distinct from the KNN region graph (which feeds the GNN) and the domain
knowledge graph (static expertise). The prediction graph stitches them together:

    Image -> Patches -> {Texture / Shape / Edge feature nodes}
          -> Defect-evidence nodes -> Prediction node -> Defect-class nodes
          -> Knowledge / Cause nodes -> Explanation node

Node and edge importance are derived from real signals (GNN attention, patch
property activations, class probabilities) so the interactive graph reflects how
the model actually reached its decision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..config import CLASS_NAMES
from .graph_schema import EdgeType, NodeType, make_edge, make_node
from .knowledge_graph import KnowledgeGraph, property_activations


@dataclass
class PredictionGraph:
    nodes: List[Dict] = field(default_factory=list)
    edges: List[Dict] = field(default_factory=list)

    def add_node(self, node: Dict) -> str:
        self.nodes.append(node)
        return node["id"]

    def add_edge(self, edge: Dict) -> None:
        self.edges.append(edge)

    def node_ids(self) -> List[str]:
        return [n["id"] for n in self.nodes]


def build_prediction_graph(
    *,
    region_graph,                       # RegionGraph
    patch_features: List[Dict[str, float]],
    global_features: Dict[str, float],
    predicted_class: str,
    probabilities: Dict[str, float],
    confidence: float,
    kg: KnowledgeGraph,
    node_importance: Optional[np.ndarray] = None,
    edge_attention: Optional[np.ndarray] = None,
    alternative_class: Optional[str] = None,
) -> PredictionGraph:
    """Assemble the full explanatory graph for one prediction."""
    pg = PredictionGraph()

    # --- Image root node ------------------------------------------------- #
    img_id = pg.add_node(make_node(
        "image", NodeType.IMAGE, "Steel Surface Image",
        semantic="Root node representing the analysed image.",
        supports=True,
    ))

    # --- Patch nodes (from region graph) --------------------------------- #
    n_patches = region_graph.num_nodes
    if node_importance is None:
        node_importance = np.ones(n_patches, dtype=np.float32)
    imp_norm = node_importance / (node_importance.max() + 1e-9)

    patch_ids: List[str] = []
    for i in range(n_patches):
        meta = region_graph.node_meta[i]
        feats = patch_features[i] if i < len(patch_features) else global_features
        acts = property_activations(feats)
        top_prop = max(acts.items(), key=lambda kv: kv[1])
        pid = pg.add_node(make_node(
            f"patch_{i}", NodeType.PATCH, f"Patch {i}",
            bbox=meta.get("bbox"),
            position=meta.get("position"),
            importance=round(float(imp_norm[i]), 4),
            features={k: round(v, 4) for k, v in feats.items()
                      if k in ("edge_density", "std_intensity", "entropy",
                               "glcm_contrast", "aspect_ratio", "circularity")},
            dominant_property=top_prop[0],
            dominant_property_score=round(float(top_prop[1]), 4),
            semantic=f"Image patch with dominant visual cue: {top_prop[0]}.",
            supports=bool(top_prop[0] in kg.class_properties(predicted_class)),
        ))
        patch_ids.append(pid)
        pg.add_edge(make_edge(img_id, pid, EdgeType.CONTAINS, weight=1.0))

    # --- KNN edges between patches (visual similarity) ------------------- #
    ei = region_graph.edge_index
    ew = region_graph.edge_weight
    seen = set()
    for e in range(ei.shape[1]):
        s, t = int(ei[0, e]), int(ei[1, e])
        if s == t:
            continue
        key = tuple(sorted((s, t)))
        if key in seen:
            continue
        seen.add(key)
        att = float(edge_attention[e]) if edge_attention is not None and e < len(edge_attention) else None
        pg.add_edge(make_edge(
            f"patch_{s}", f"patch_{t}", EdgeType.KNN_OF, weight=float(ew[e]),
            similarity=round(float(ew[e]), 4),
            attention=round(att, 4) if att is not None else None,
            contribution=round(att, 4) if att is not None else round(float(ew[e]), 4),
        ))

    # --- Feature-summary nodes (texture / shape / edge) ------------------ #
    feature_groups = {
        NodeType.TEXTURE_FEATURE: ("Texture", ["glcm_contrast", "glcm_homogeneity",
                                                "glcm_energy", "entropy", "roughness"]),
        NodeType.SHAPE_FEATURE: ("Shape", ["circularity", "aspect_ratio", "solidity",
                                            "extent", "orientation"]),
        NodeType.EDGE_FEATURE: ("Edge", ["edge_density", "sobel_response",
                                         "laplacian_variance"]),
    }
    for ntype, (name, keys) in feature_groups.items():
        vals = {k: round(global_features.get(k, 0.0), 4) for k in keys}
        fid = pg.add_node(make_node(
            f"feat_{name.lower()}", ntype, f"{name} Features",
            features=vals,
            semantic=f"Aggregated {name.lower()} descriptors extracted from the image.",
            supports=True,
        ))
        pg.add_edge(make_edge(fid, img_id, EdgeType.EXTRACTED_FROM, weight=1.0))

    # --- Defect-evidence nodes (top visual properties) ------------------- #
    top_props = kg.top_properties(global_features, k=4)
    pred_props = set(kg.class_properties(predicted_class))
    evidence_ids: List[str] = []
    for j, (prop, score) in enumerate(top_props):
        supports = prop in pred_props
        eid = pg.add_node(make_node(
            f"evidence_{j}", NodeType.DEFECT_EVIDENCE, prop,
            activation=round(float(score), 4),
            semantic=f"Detected visual property '{prop}' (activation {score:.2f}).",
            supports=bool(supports),
        ))
        evidence_ids.append(eid)
        rel = EdgeType.SUPPORTS if supports else EdgeType.CONTRADICTS
        pg.add_edge(make_edge(img_id, eid, EdgeType.HAS_PROPERTY, weight=float(score)))
        # link evidence to knowledge concept node (added below via KG)

    # --- Prediction node ------------------------------------------------- #
    pred_id = pg.add_node(make_node(
        "prediction", NodeType.PREDICTION, f"Prediction: {predicted_class}",
        confidence=round(float(confidence), 4),
        probabilities={k: round(float(v), 4) for k, v in probabilities.items()},
        semantic=f"Hybrid model prediction with confidence {confidence:.2f}.",
        supports=True,
    ))
    for eid in evidence_ids:
        ev_node = next(n for n in pg.nodes if n["id"] == eid)
        rel = EdgeType.SUPPORTS if ev_node.get("supports") else EdgeType.CONTRADICTS
        pg.add_edge(make_edge(eid, pred_id, rel,
                              weight=float(ev_node.get("activation", 0.5))))

    # link most important patches to the prediction
    top_patch_idx = np.argsort(imp_norm)[::-1][:min(3, n_patches)]
    for i in top_patch_idx:
        pg.add_edge(make_edge(f"patch_{int(i)}", pred_id, EdgeType.SUPPORTS,
                              weight=float(imp_norm[int(i)]),
                              contribution=round(float(imp_norm[int(i)]), 4)))

    # --- Defect-class nodes (all six, sized by probability) -------------- #
    for cls in CLASS_NAMES:
        p = float(probabilities.get(cls, 0.0))
        cid = pg.add_node(make_node(
            f"class_{cls.replace(' ', '_')}", NodeType.DEFECT_CLASS, cls,
            probability=round(p, 4),
            is_prediction=bool(cls == predicted_class),
            semantic=f"Defect class '{cls}' — model probability {p:.2f}.",
            supports=bool(cls == predicted_class),
        ))
        rel = EdgeType.PREDICTS if cls == predicted_class else EdgeType.BELONGS_TO
        pg.add_edge(make_edge(pred_id, cid, rel, weight=p))

    # --- Knowledge & cause nodes for the predicted class ----------------- #
    pred_class_id = f"class_{predicted_class.replace(' ', '_')}"
    for prop in kg.class_properties(predicted_class):
        kid = f"kg_{prop.replace(' ', '_')}"
        if kid not in pg.node_ids():
            act = property_activations(global_features).get(prop, 0.0)
            pg.add_node(make_node(
                kid, NodeType.KNOWLEDGE, prop,
                activation=round(float(act), 4),
                semantic=f"Knowledge concept: {predicted_class} typically shows '{prop}'.",
                supports=True,
            ))
        pg.add_edge(make_edge(pred_class_id, kid, EdgeType.HAS_PATTERN, weight=1.0))
        # connect matching evidence node to this knowledge concept
        for ev in pg.nodes:
            if ev["type"] == NodeType.DEFECT_EVIDENCE.value and ev["label"] == prop:
                pg.add_edge(make_edge(ev["id"], kid, EdgeType.SEMANTICALLY_RELATED,
                                      weight=1.0))

    for cause in kg.class_causes(predicted_class):
        cid = f"cause_{cause.replace(' ', '_')}"
        if cid not in pg.node_ids():
            pg.add_node(make_node(
                cid, NodeType.CAUSE, cause,
                semantic=f"Likely manufacturing cause of {predicted_class}: {cause}.",
                supports=True,
            ))
        pg.add_edge(make_edge(pred_class_id, cid, EdgeType.ASSOCIATED_WITH, weight=1.0))

    return pg


def compute_node_importance_from_patches(
    patch_features: List[Dict[str, float]], predicted_class: str, kg: KnowledgeGraph
) -> np.ndarray:
    """Fallback node-importance when no GNN attention is available.

    Scores each patch by how strongly its dominant visual property matches the
    predicted class's known properties.
    """
    pred_props = set(kg.class_properties(predicted_class))
    scores = []
    for feats in patch_features:
        acts = property_activations(feats)
        match = sum(acts[p] for p in pred_props if p in acts)
        scores.append(match)
    arr = np.array(scores, dtype=np.float32)
    return arr if arr.size else np.ones(1, dtype=np.float32)
