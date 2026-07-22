"""Domain knowledge graph for steel-surface defects.

Encodes expert domain knowledge (NOT learned from data) linking the six NEU-DET
defect classes to their characteristic visual properties and likely
manufacturing causes. The graph is stored as JSON/NetworkX so it is portable and
Neo4j-ready (each entity/relation maps cleanly to nodes/edges).

The knowledge graph is used for two things:
  1. Grounding explanations — providing defect -> property -> cause paths.
  2. Producing a *soft* KG affinity vector (one score per class) from an image's
     interpretable features. This is a rule-based prior used as an auxiliary
     signal / explanation aid, **not** the final classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from ..config import CLASS_NAMES, settings
from ..utils.helpers import get_logger, load_json, save_json
from .graph_schema import EdgeType, NodeType

logger = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Domain knowledge (curated)
# --------------------------------------------------------------------------- #
VISUAL_PROPERTIES = [
    "Linear pattern", "Network-like cracks", "Dark embedded material",
    "Irregular patch", "Small cavities", "Scale-like texture",
    "Long narrow marks", "High edge density", "Rough texture",
    "Repetitive texture", "Directional pattern", "Local intensity variation",
]

CAUSES = [
    "Rolling pressure", "Cooling stress", "Surface contamination",
    "Material impurity", "Oxidation", "Mechanical abrasion",
    "Improper descaling", "Thermal stress", "Foreign particle inclusion",
]

# class -> (visual properties, causes)
CLASS_KNOWLEDGE: Dict[str, Dict[str, List[str]]] = {
    "Crazing": {
        "properties": ["Network-like cracks", "High edge density", "Repetitive texture"],
        "causes": ["Thermal stress", "Cooling stress"],
    },
    "Inclusion": {
        "properties": ["Dark embedded material", "Local intensity variation"],
        "causes": ["Material impurity", "Foreign particle inclusion"],
    },
    "Patches": {
        "properties": ["Irregular patch", "Local intensity variation"],
        "causes": ["Surface contamination", "Oxidation"],
    },
    "Pitted Surface": {
        "properties": ["Small cavities", "Rough texture", "High edge density"],
        "causes": ["Oxidation", "Surface contamination"],
    },
    "Rolled-in Scale": {
        "properties": ["Scale-like texture", "Rough texture", "Repetitive texture"],
        "causes": ["Improper descaling", "Rolling pressure"],
    },
    "Scratches": {
        "properties": ["Long narrow marks", "Linear pattern", "Directional pattern",
                       "High edge density"],
        "causes": ["Mechanical abrasion", "Rolling pressure"],
    },
}

# Interpretable-feature rules that *detect* the presence of each visual property
# from the handcrafted descriptor. Each rule returns a soft activation in [0, 1].
# NOTE: heuristic thresholds, expert-set, not fitted to labels.
def _clamp(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def property_activations(feats: Dict[str, float]) -> Dict[str, float]:
    """Soft activation (0..1) for each visual property given named features."""
    ed = feats.get("edge_density", 0.0)
    std = feats.get("std_intensity", 0.0)
    ent = feats.get("entropy", 0.0)
    contrast = feats.get("glcm_contrast", 0.0)
    homog = feats.get("glcm_homogeneity", 0.0)
    energy = feats.get("glcm_energy", 0.0)
    circ = feats.get("circularity", 0.0)
    ar = feats.get("aspect_ratio", 1.0)
    solidity = feats.get("solidity", 0.0)
    roughness = feats.get("roughness", 0.0)
    ncont = feats.get("num_contours", 0.0)
    elongation = abs(np.log((ar + 1e-3)))          # 0 when square, grows when elongated

    return {
        "Linear pattern": _clamp(0.6 * elongation + 0.6 * ed),
        "Network-like cracks": _clamp(1.4 * ed + 0.8 * ent - 0.6 * solidity),
        "Dark embedded material": _clamp(1.2 * std * (1 - feats.get("mean_intensity", 0.5))),
        "Irregular patch": _clamp(1.2 * (1 - circ) * (0.5 + std)),
        "Small cavities": _clamp(0.5 * ncont + 0.8 * (1 - circ) * ed),
        "Scale-like texture": _clamp(1.3 * homog * (0.5 + roughness)),
        "Long narrow marks": _clamp(0.9 * elongation + 0.7 * ed),
        "High edge density": _clamp(3.0 * ed),
        "Rough texture": _clamp(4.0 * roughness),
        "Repetitive texture": _clamp(1.5 * energy + 0.6 * homog),
        "Directional pattern": _clamp(0.9 * elongation),
        "Local intensity variation": _clamp(3.0 * std),
    }


class KnowledgeGraph:
    """Wraps a NetworkX knowledge graph with build / query / affinity helpers."""

    def __init__(self, graph: Optional[nx.DiGraph] = None):
        self.g: nx.DiGraph = graph if graph is not None else self.build()

    # ------------------------------------------------------------------ #
    @staticmethod
    def build() -> nx.DiGraph:
        """Construct the curated domain knowledge graph."""
        g = nx.DiGraph()
        for prop in VISUAL_PROPERTIES:
            g.add_node(prop, type=NodeType.KNOWLEDGE.value, label=prop)
        for cause in CAUSES:
            g.add_node(cause, type=NodeType.CAUSE.value, label=cause)
        for cls in CLASS_NAMES:
            g.add_node(cls, type=NodeType.DEFECT_CLASS.value, label=cls)
            info = CLASS_KNOWLEDGE[cls]
            for prop in info["properties"]:
                g.add_edge(cls, prop, relation=EdgeType.HAS_PROPERTY.value)
            for cause in info["causes"]:
                g.add_edge(cls, cause, relation=EdgeType.ASSOCIATED_WITH.value)
        return g

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or settings.knowledge_graph_file
        data = {
            "directed": True,
            "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": u, "target": v, **d} for u, v, d in self.g.edges(data=True)],
        }
        save_json(path, data)
        return path

    @classmethod
    def load_or_build(cls, path: Optional[Path] = None) -> "KnowledgeGraph":
        path = path or settings.knowledge_graph_file
        if Path(path).exists():
            try:
                data = load_json(path)
                g = nx.DiGraph()
                for node in data["nodes"]:
                    nid = node.pop("id")
                    g.add_node(nid, **node)
                for edge in data["edges"]:
                    g.add_edge(edge["source"], edge["target"],
                               relation=edge.get("relation", "related_to"))
                return cls(g)
            except Exception as exc:
                logger.warning("Failed to load KG (%s); rebuilding", exc)
        kg = cls(cls.build())
        kg.save(path)
        return kg

    # ------------------------------------------------------------------ #
    def class_properties(self, cls: str) -> List[str]:
        return CLASS_KNOWLEDGE.get(cls, {}).get("properties", [])

    def class_causes(self, cls: str) -> List[str]:
        return CLASS_KNOWLEDGE.get(cls, {}).get("causes", [])

    def affinity_vector(self, feats: Dict[str, float]) -> np.ndarray:
        """Rule-based soft affinity to each class (length == NUM_CLASSES).

        For each class, average the detected activation of its characteristic
        visual properties. Returned in ``CLASS_NAMES`` order. This is a domain
        prior / explanation aid, not a learned classifier output.
        """
        acts = property_activations(feats)
        scores = []
        for cls in CLASS_NAMES:
            props = self.class_properties(cls)
            score = float(np.mean([acts.get(p, 0.0) for p in props])) if props else 0.0
            scores.append(score)
        vec = np.array(scores, dtype=np.float32)
        return vec

    def reasoning_path(self, cls: str, feats: Dict[str, float]) -> List[Dict]:
        """Return defect -> property -> cause triples ranked by feature support."""
        acts = property_activations(feats)
        path: List[Dict] = []
        for prop in self.class_properties(cls):
            path.append({
                "subject": cls, "relation": EdgeType.HAS_PROPERTY.value,
                "object": prop, "support": round(acts.get(prop, 0.0), 3),
            })
        for cause in self.class_causes(cls):
            path.append({
                "subject": cls, "relation": EdgeType.ASSOCIATED_WITH.value,
                "object": cause, "support": None,
            })
        path.sort(key=lambda t: (t["support"] is not None, t["support"] or 0), reverse=True)
        return path

    def top_properties(self, feats: Dict[str, float], k: int = 4) -> List[Tuple[str, float]]:
        acts = property_activations(feats)
        return sorted(acts.items(), key=lambda kv: kv[1], reverse=True)[:k]
