"""Serialise a :class:`PredictionGraph` into frontend-ready formats.

Primary target is Cytoscape.js ``elements`` (a flat list of ``{data: {...}}``
node/edge objects). Also persists the graph to ``data/graph_data`` so it can be
re-fetched by prediction id and exported from the notebook.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..config import settings
from ..utils.helpers import load_json, save_json
from .graph_reasoner import PredictionGraph
from .graph_schema import legend


def to_cytoscape(pg: PredictionGraph) -> Dict[str, List[Dict]]:
    """Convert a prediction graph into Cytoscape.js element lists."""
    node_ids = set(pg.node_ids())
    elements: List[Dict] = []
    for n in pg.nodes:
        importance = n.get("importance")
        if importance is None:
            importance = n.get("activation", n.get("probability", 0.5))
        elements.append({
            "group": "nodes",
            "data": {
                "id": n["id"],
                "label": n["label"],
                "ntype": n["type"],
                "shape": n.get("shape", "ellipse"),
                "color": n.get("color", "#718096"),
                "importance": float(importance or 0.0),
                "supports": bool(n.get("supports", False)),
                "meta": {k: v for k, v in n.items()
                         if k not in ("id", "label", "type", "shape", "color")},
            },
        })
    edge_elems: List[Dict] = []
    for e in pg.edges:
        if e["source"] not in node_ids or e["target"] not in node_ids:
            continue
        edge_elems.append({
            "group": "edges",
            "data": {
                "id": e["id"],
                "source": e["source"],
                "target": e["target"],
                "relation": e["relation"],
                "weight": float(e.get("weight", 1.0)),
                "color": e.get("color", "#a0aec0"),
                "line_style": e.get("line_style", "solid"),
                "meta": {k: v for k, v in e.items()
                         if k not in ("id", "source", "target", "relation",
                                      "color", "line_style")},
            },
        })
    return {"nodes": elements, "edges": edge_elems}


def build_graph_payload(pg: PredictionGraph) -> Dict:
    """Full payload consumed by the frontend graph view."""
    cy = to_cytoscape(pg)
    return {
        "elements": cy["nodes"] + cy["edges"],
        "counts": {"nodes": len(cy["nodes"]), "edges": len(cy["edges"])},
        "legend": legend(),
        "node_types": sorted({n["data"]["ntype"] for n in cy["nodes"]}),
        "edge_types": sorted({e["data"]["relation"] for e in cy["edges"]}),
    }


def save_graph(prediction_id: str, payload: Dict, out_dir: Optional[Path] = None) -> Path:
    out_dir = out_dir or settings.graph_data_dir
    path = out_dir / f"graph_{prediction_id}.json"
    save_json(path, payload)
    return path


def load_graph(prediction_id: str, out_dir: Optional[Path] = None) -> Optional[Dict]:
    out_dir = out_dir or settings.graph_data_dir
    path = out_dir / f"graph_{prediction_id}.json"
    if not path.exists():
        return None
    return load_json(path)
