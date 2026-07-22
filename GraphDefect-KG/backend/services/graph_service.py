"""Builds graph node features and the KNN region graph from a preprocessed image.

Node feature = [projected MobileNetV2 patch embedding | handcrafted descriptor].
The same routine is reused by the notebook so backend and experiments share one
feature definition.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..models.knn_graph_builder import RegionGraph, build_knn_graph
from ..models.mobilenet_feature_extractor import MobileNetV2FeatureExtractor
from ..utils.feature_extraction import compute_named_features
from ..utils.preprocessing import PreprocessResult


def build_region_nodes(
    pre: PreprocessResult, extractor: MobileNetV2FeatureExtractor,
) -> Tuple[np.ndarray, List[Dict], np.ndarray, List[Dict[str, float]], np.ndarray]:
    """Return (node_features, node_meta, positions, patch_named_features, patch_cnn).

    node_features : (N, projected_dim + handcrafted_dim)
    patch_cnn     : (N, 1280) raw patch embeddings (for mean pooling in hybrid)
    """
    patches = pre.patches
    patch_imgs = [p.image for p in patches]

    # CNN patch embeddings -> projected.
    cnn_raw = extractor.patch_embeddings(patch_imgs)          # (N, 1280)
    cnn_proj = extractor.project(cnn_raw)                     # (N, projected_dim)

    # Handcrafted per-patch features.
    named_feats: List[Dict[str, float]] = []
    handcrafted = []
    from ..utils.feature_extraction import FEATURE_NAMES
    for p in patches:
        feats = compute_named_features(p.image)
        named_feats.append(feats)
        handcrafted.append([feats[n] for n in FEATURE_NAMES])
    handcrafted = np.asarray(handcrafted, dtype=np.float32)

    node_features = np.concatenate([cnn_proj, handcrafted], axis=1).astype(np.float32)
    node_features = np.nan_to_num(node_features)

    size = float(pre.size)
    positions = np.array(
        [[(p.x0 + p.x1) / 2 / size, (p.y0 + p.y1) / 2 / size] for p in patches],
        dtype=np.float32,
    )
    node_meta: List[Dict] = [
        {
            "index": p.index, "type": "patch",
            "bbox": p.bbox(),
            "position": {"row": p.row, "col": p.col},
        }
        for p in patches
    ]
    return node_features, node_meta, positions, named_feats, cnn_raw


def construct_graph(
    pre: PreprocessResult, extractor: MobileNetV2FeatureExtractor,
) -> Tuple[RegionGraph, List[Dict[str, float]], np.ndarray]:
    """Full region-graph construction. Returns (graph, patch_features, patch_cnn)."""
    node_features, node_meta, positions, named_feats, cnn_raw = build_region_nodes(
        pre, extractor
    )
    graph = build_knn_graph(node_features, node_meta, positions)
    return graph, named_feats, cnn_raw
