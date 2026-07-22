"""Tests for the graph/hybrid model definitions (shapes and gradients)."""
import numpy as np
import torch

from backend.models.gcn_model import GCN
from backend.models.gnn_model import GNN
from backend.models.hybrid_model import HybridDefectModel, HybridInputs
from backend.models.knn_graph_builder import build_knn_graph
from backend.config import CLASS_NAMES

NODE_DIM = 160
HAND_DIM = 32
KG_DIM = len(CLASS_NAMES)


def _graph():
    rng = np.random.default_rng(2)
    feats = rng.standard_normal((10, NODE_DIM)).astype("float32")
    pos = rng.random((10, 2)).astype("float32")
    g = build_knn_graph(feats, [{"index": i} for i in range(10)], pos)
    x = torch.from_numpy(g.node_features)
    ei = torch.from_numpy(g.edge_index)
    ew = torch.from_numpy(g.edge_weight)
    return x, ei, ew


def test_gcn_output_shape():
    x, ei, ew = _graph()
    model = GCN(NODE_DIM, num_classes=len(CLASS_NAMES))
    logits, embed = model(x, ei, ew)
    assert logits.shape == (1, len(CLASS_NAMES))
    assert embed.shape[0] == 1


def test_gat_attention_available():
    x, ei, ew = _graph()
    model = GNN(NODE_DIM, arch="gat", num_classes=len(CLASS_NAMES))
    logits, _ = model(x, ei, ew)
    assert logits.shape == (1, len(CLASS_NAMES))
    att = model.edge_importance()
    assert att is not None and att.shape[0] == ei.shape[1]


def test_graphsage_runs():
    x, ei, ew = _graph()
    model = GNN(NODE_DIM, arch="graphsage", num_classes=len(CLASS_NAMES))
    logits, _ = model(x, ei, ew)
    assert logits.shape == (1, len(CLASS_NAMES))
    assert model.edge_importance() is None  # SAGE has no attention


def test_hybrid_forward_and_gates():
    x, ei, ew = _graph()
    model = HybridDefectModel(NODE_DIM, HAND_DIM, KG_DIM, num_classes=len(CLASS_NAMES))
    inp = HybridInputs(
        node_x=x, edge_index=ei, edge_weight=ew,
        cnn_global=torch.randn(1280), cnn_patch=torch.randn(1280),
        handcrafted=torch.randn(HAND_DIM), kg_embed=torch.randn(KG_DIM),
    )
    logits, aux = model(inp)
    assert logits.shape == (1, len(CLASS_NAMES))
    gates = model.component_importance()
    assert abs(sum(gates.values()) - 1.0) < 1e-3  # softmax gate sums to 1


def test_gcn_backward_pass():
    x, ei, ew = _graph()
    model = GCN(NODE_DIM, num_classes=len(CLASS_NAMES))
    logits, _ = model(x, ei, ew)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
