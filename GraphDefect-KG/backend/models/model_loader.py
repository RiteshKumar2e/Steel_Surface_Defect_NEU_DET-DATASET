"""Model lifecycle: construction, checkpoint loading, and the KNN baseline.

Honesty policy (per project spec):
  * The GCN / GNN / hybrid networks are always constructed with valid
    architectures. If no trained checkpoint exists they are flagged
    ``trained=False`` and their raw logits are NOT presented as authoritative.
  * A MobileNetV2 + KNN baseline is a genuinely fitted, non-parametric model
    (fits in seconds from cached backbone features). When available it provides
    the demo's authoritative class prediction, clearly labelled as such.
  * Nothing here fabricates accuracies or predictions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import CLASS_NAMES, FOLDER_TO_CLASS, settings
from ..utils.feature_extraction import HANDCRAFTED_DIM
from ..utils.helpers import get_logger
from .gcn_model import GCN
from .gnn_model import GNN
from .hybrid_model import HybridDefectModel
from .mobilenet_feature_extractor import MobileNetV2FeatureExtractor

logger = get_logger(__name__)

KNN_BASELINE_FILE = "knn_baseline.joblib"
HYBRID_CKPT_FILE = "hybrid_model.pt"

# Node feature dim fed to graph models: projected CNN patch features + handcrafted.
NODE_FEATURE_DIM = settings.model.projected_dim + HANDCRAFTED_DIM


@dataclass
class KNNBaseline:
    """MobileNetV2 (global embedding) + KNN classifier baseline."""
    knn: object
    scaler: object
    classes: List[str] = field(default_factory=lambda: list(CLASS_NAMES))
    n_train: int = 0

    def predict_proba(self, embedding: np.ndarray) -> np.ndarray:
        x = self.scaler.transform(embedding.reshape(1, -1))
        proba = self.knn.predict_proba(x)[0]
        # align to canonical class order
        full = np.zeros(len(self.classes), dtype=np.float32)
        for cls_idx, cls_label in enumerate(self.knn.classes_):
            full[int(cls_label)] = proba[cls_idx]
        return full


@dataclass
class ModelBundle:
    extractor: MobileNetV2FeatureExtractor
    gcn: GCN
    gnn: GNN
    hybrid: HybridDefectModel
    knn_baseline: Optional[KNNBaseline]
    hybrid_trained: bool
    device: str

    def primary_source(self) -> str:
        """Which model produces the authoritative class label."""
        if self.hybrid_trained:
            return "hybrid"
        if self.knn_baseline is not None:
            return "mobilenet_knn_baseline"
        return "untrained_hybrid"


# --------------------------------------------------------------------------- #
# KNN baseline fitting
# --------------------------------------------------------------------------- #
def _iter_dataset(max_per_class: Optional[int] = None) -> List[Tuple[Path, int]]:
    """List (image_path, class_index) pairs from the NEU-DET folder."""
    root = settings.dataset_images
    items: List[Tuple[Path, int]] = []
    if root is None:
        return items
    for folder, cls_label in FOLDER_TO_CLASS.items():
        cdir = root / folder
        if not cdir.exists():
            continue
        cls_idx = CLASS_NAMES.index(cls_label)
        files = sorted(cdir.glob("*.jpg")) + sorted(cdir.glob("*.png")) + \
            sorted(cdir.glob("*.bmp"))
        if max_per_class:
            files = files[:max_per_class]
        items += [(f, cls_idx) for f in files]
    return items


def fit_knn_baseline(
    extractor: MobileNetV2FeatureExtractor,
    max_per_class: int = 120,
    k: int = 5,
    save: bool = True,
) -> Optional[KNNBaseline]:
    """Fit the MobileNetV2 + KNN baseline from cached backbone embeddings."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    import cv2

    items = _iter_dataset(max_per_class)
    if not items:
        logger.warning("Dataset not found; KNN baseline unavailable.")
        return None

    logger.info("Fitting KNN baseline on %d images ...", len(items))
    X, y = [], []
    for path, cls_idx in items:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.resize(img, (settings.preprocess.image_size,) * 2)
        emb = extractor.global_embedding(img)
        X.append(emb)
        y.append(cls_idx)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    scaler = StandardScaler().fit(X)
    knn = KNeighborsClassifier(n_neighbors=min(k, len(X) - 1), weights="distance")
    knn.fit(scaler.transform(X), y)
    baseline = KNNBaseline(knn=knn, scaler=scaler, n_train=len(X))

    if save:
        import joblib
        out = settings.saved_models_dir / KNN_BASELINE_FILE
        joblib.dump({"knn": knn, "scaler": scaler, "n_train": len(X)}, out)
        logger.info("Saved KNN baseline -> %s", out)
    return baseline


def load_knn_baseline(
    extractor: MobileNetV2FeatureExtractor, auto_fit: bool = True
) -> Optional[KNNBaseline]:
    path = settings.saved_models_dir / KNN_BASELINE_FILE
    if path.exists():
        try:
            import joblib
            blob = joblib.load(path)
            logger.info("Loaded KNN baseline from %s", path)
            return KNNBaseline(knn=blob["knn"], scaler=blob["scaler"],
                               n_train=blob.get("n_train", 0))
        except Exception as exc:
            logger.warning("Failed to load KNN baseline (%s)", exc)
    if auto_fit:
        return fit_knn_baseline(extractor)
    return None


# --------------------------------------------------------------------------- #
# Bundle loading
# --------------------------------------------------------------------------- #
_BUNDLE: Optional[ModelBundle] = None


def load_bundle(auto_fit_knn: bool = True) -> ModelBundle:
    """Construct (and cache) the full model bundle. Idempotent."""
    global _BUNDLE
    if _BUNDLE is not None:
        return _BUNDLE

    device = settings.resolve_device()
    extractor = MobileNetV2FeatureExtractor(device=device)

    gcn = GCN(NODE_FEATURE_DIM, num_classes=len(CLASS_NAMES)).to(device).eval()
    gnn = GNN(NODE_FEATURE_DIM, arch="gat", num_classes=len(CLASS_NAMES)).to(device).eval()
    hybrid = HybridDefectModel(
        node_dim=NODE_FEATURE_DIM, handcrafted_dim=HANDCRAFTED_DIM,
        kg_dim=len(CLASS_NAMES), num_classes=len(CLASS_NAMES),
    ).to(device).eval()

    hybrid_trained = False
    ckpt = settings.saved_models_dir / HYBRID_CKPT_FILE
    if ckpt.exists():
        try:
            import torch
            state = torch.load(ckpt, map_location=device)
            hybrid.load_state_dict(state["model"] if "model" in state else state)
            hybrid_trained = True
            logger.info("Loaded trained hybrid checkpoint from %s", ckpt)
        except Exception as exc:
            logger.warning("Could not load hybrid checkpoint (%s); using untrained.", exc)

    knn_baseline = load_knn_baseline(extractor, auto_fit=auto_fit_knn)

    _BUNDLE = ModelBundle(
        extractor=extractor, gcn=gcn, gnn=gnn, hybrid=hybrid,
        knn_baseline=knn_baseline, hybrid_trained=hybrid_trained, device=device,
    )
    logger.info("Model bundle ready (primary source: %s)", _BUNDLE.primary_source())
    return _BUNDLE


def reset_bundle() -> None:
    global _BUNDLE
    _BUNDLE = None
