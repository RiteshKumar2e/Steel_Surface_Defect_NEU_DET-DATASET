"""Tests for KNN graph construction and the knowledge graph."""
import numpy as np

from backend.models.knn_graph_builder import build_knn_graph, normalized_adjacency
from backend.graph.knowledge_graph import KnowledgeGraph, property_activations
from backend.config import CLASS_NAMES


def test_knn_graph_no_isolated_nodes(node_features):
    feats, meta, positions = node_features
    g = build_knn_graph(feats, meta, positions)
    assert g.num_nodes == 12
    # every node has at least one edge
    assert (g.adjacency.sum(axis=1) > 0).all()
    # edge weights are in (0, 1]
    assert g.edge_weight.min() > 0 and g.edge_weight.max() <= 1.0 + 1e-6


def test_knn_graph_symmetric_and_selfloops(node_features):
    feats, meta, positions = node_features
    g = build_knn_graph(feats, meta, positions)
    assert np.allclose(g.adjacency, g.adjacency.T, atol=1e-6)
    assert np.all(np.diag(g.adjacency) > 0)  # self loops added


def test_normalized_adjacency_bounded(node_features):
    feats, meta, positions = node_features
    g = build_knn_graph(feats, meta, positions)
    norm = normalized_adjacency(g.adjacency)
    assert np.isfinite(norm).all()


def test_single_node_graph():
    feats = np.zeros((1, 160), dtype="float32")
    g = build_knn_graph(feats, [{"index": 0}], np.zeros((1, 2), dtype="float32"))
    assert g.num_nodes == 1 and g.num_edges >= 1


def test_knowledge_graph_structure():
    kg = KnowledgeGraph.load_or_build()
    for cls in CLASS_NAMES:
        assert kg.class_properties(cls), f"{cls} has no properties"
        assert kg.class_causes(cls), f"{cls} has no causes"


def test_affinity_vector_shape_and_range():
    kg = KnowledgeGraph.load_or_build()
    feats = {n: 0.5 for n in
             ["edge_density", "std_intensity", "entropy", "glcm_contrast",
              "glcm_homogeneity", "glcm_energy", "circularity", "aspect_ratio",
              "solidity", "roughness", "num_contours", "mean_intensity"]}
    vec = kg.affinity_vector(feats)
    assert vec.shape == (len(CLASS_NAMES),)
    assert (vec >= 0).all() and (vec <= 1).all()


def test_property_activations_bounded():
    acts = property_activations({"edge_density": 1.0, "std_intensity": 1.0})
    assert all(0.0 <= v <= 1.0 for v in acts.values())
