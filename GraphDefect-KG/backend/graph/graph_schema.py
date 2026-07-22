"""Node/edge type vocabulary and light schema helpers for the heterogeneous
prediction graph and the domain knowledge graph.

Centralising the vocabulary here keeps the builder, serialiser, reasoner and the
frontend legend in agreement. Every node/edge carries a ``type`` drawn from these
enums so the UI can style them consistently.
"""
from __future__ import annotations

from enum import Enum
from typing import Dict, List


class NodeType(str, Enum):
    IMAGE = "image"
    PATCH = "patch"
    REGION = "region"
    CNN_FEATURE = "cnn_feature"
    TEXTURE_FEATURE = "texture_feature"
    SHAPE_FEATURE = "shape_feature"
    EDGE_FEATURE = "edge_feature"
    DEFECT_EVIDENCE = "defect_evidence"
    DEFECT_CLASS = "defect_class"
    PREDICTION = "prediction"
    KNOWLEDGE = "knowledge_concept"
    CAUSE = "cause_concept"
    EXPLANATION = "explanation"


class EdgeType(str, Enum):
    SPATIALLY_ADJACENT = "spatially_adjacent_to"
    VISUALLY_SIMILAR = "visually_similar_to"
    KNN_OF = "knn_of"
    CONTAINS = "contains"
    EXTRACTED_FROM = "extracted_from"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    ASSOCIATED_WITH = "associated_with"
    PREDICTS = "predicts"
    BELONGS_TO = "belongs_to"
    HAS_TEXTURE = "has_texture"
    HAS_SHAPE = "has_shape"
    HAS_ORIENTATION = "has_orientation"
    HAS_CONFIDENCE = "has_confidence"
    HAS_PATTERN = "has_pattern"
    HAS_PROPERTY = "has_property"
    SEMANTICALLY_RELATED = "semantically_related_to"


# Visual styling hints consumed by the frontend legend (shape + colour family).
NODE_STYLE: Dict[str, Dict[str, str]] = {
    NodeType.IMAGE.value:          {"shape": "rectangle",     "color": "#2b6cb0", "label": "Image"},
    NodeType.PATCH.value:          {"shape": "round-rectangle","color": "#4299e1", "label": "Patch"},
    NodeType.REGION.value:         {"shape": "hexagon",       "color": "#3182ce", "label": "Region"},
    NodeType.CNN_FEATURE.value:    {"shape": "ellipse",       "color": "#805ad5", "label": "CNN feature"},
    NodeType.TEXTURE_FEATURE.value:{"shape": "ellipse",       "color": "#d69e2e", "label": "Texture"},
    NodeType.SHAPE_FEATURE.value:  {"shape": "ellipse",       "color": "#dd6b20", "label": "Shape"},
    NodeType.EDGE_FEATURE.value:   {"shape": "ellipse",       "color": "#38a169", "label": "Edge feature"},
    NodeType.DEFECT_EVIDENCE.value:{"shape": "diamond",       "color": "#e53e3e", "label": "Detected evidence"},
    NodeType.DEFECT_CLASS.value:   {"shape": "pentagon",      "color": "#c53030", "label": "Defect class"},
    NodeType.PREDICTION.value:     {"shape": "star",          "color": "#276749", "label": "Prediction"},
    NodeType.KNOWLEDGE.value:      {"shape": "triangle",      "color": "#0d9488", "label": "Knowledge property"},
    NodeType.CAUSE.value:          {"shape": "octagon",       "color": "#b7791f", "label": "Manufacturing cause"},
    NodeType.EXPLANATION.value:    {"shape": "round-tag",     "color": "#5a67d8", "label": "Explanation"},
}

EDGE_STYLE: Dict[str, Dict[str, str]] = {
    EdgeType.KNN_OF.value:            {"color": "#a0aec0", "style": "solid",  "label": "kNN of"},
    EdgeType.VISUALLY_SIMILAR.value:  {"color": "#90cdf4", "style": "solid",  "label": "similar to"},
    EdgeType.SPATIALLY_ADJACENT.value:{"color": "#cbd5e0", "style": "dashed", "label": "adjacent to"},
    EdgeType.CONTAINS.value:          {"color": "#4a5568", "style": "solid",  "label": "contains"},
    EdgeType.EXTRACTED_FROM.value:    {"color": "#9f7aea", "style": "dotted", "label": "extracted from"},
    EdgeType.SUPPORTS.value:          {"color": "#38a169", "style": "solid",  "label": "supports"},
    EdgeType.CONTRADICTS.value:       {"color": "#e53e3e", "style": "dashed", "label": "contradicts"},
    EdgeType.ASSOCIATED_WITH.value:   {"color": "#319795", "style": "solid",  "label": "associated with"},
    EdgeType.PREDICTS.value:          {"color": "#276749", "style": "solid",  "label": "predicts"},
    EdgeType.BELONGS_TO.value:        {"color": "#2c5282", "style": "solid",  "label": "belongs to"},
    EdgeType.HAS_PATTERN.value:       {"color": "#00897b", "style": "solid",  "label": "has pattern"},
    EdgeType.HAS_PROPERTY.value:      {"color": "#00897b", "style": "solid",  "label": "has property"},
    EdgeType.HAS_TEXTURE.value:       {"color": "#d69e2e", "style": "solid",  "label": "has texture"},
    EdgeType.HAS_SHAPE.value:         {"color": "#dd6b20", "style": "solid",  "label": "has shape"},
    EdgeType.HAS_ORIENTATION.value:   {"color": "#dd6b20", "style": "solid",  "label": "has orientation"},
    EdgeType.HAS_CONFIDENCE.value:    {"color": "#718096", "style": "dotted", "label": "has confidence"},
    EdgeType.SEMANTICALLY_RELATED.value:{"color": "#5a67d8", "style": "dashed","label": "related to"},
}


def make_node(node_id: str, ntype: NodeType, label: str, **attrs) -> Dict:
    """Construct a standardised node dict."""
    style = NODE_STYLE.get(ntype.value, {})
    return {
        "id": node_id,
        "type": ntype.value,
        "label": label,
        "shape": style.get("shape", "ellipse"),
        "color": style.get("color", "#718096"),
        **attrs,
    }


def make_edge(source: str, target: str, etype: EdgeType, weight: float = 1.0, **attrs) -> Dict:
    """Construct a standardised edge dict."""
    style = EDGE_STYLE.get(etype.value, {})
    return {
        "id": f"{source}__{etype.value}__{target}",
        "source": source,
        "target": target,
        "relation": etype.value,
        "weight": round(float(weight), 4),
        "color": style.get("color", "#a0aec0"),
        "line_style": style.get("style", "solid"),
        **attrs,
    }


def legend() -> Dict[str, List[Dict]]:
    """Legend payload for the frontend (node + edge families)."""
    return {
        "nodes": [{"type": k, **v} for k, v in NODE_STYLE.items()],
        "edges": [{"relation": k, **v} for k, v in EDGE_STYLE.items()],
    }
