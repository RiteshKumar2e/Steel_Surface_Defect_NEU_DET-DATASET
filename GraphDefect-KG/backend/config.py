"""Central configuration for GraphDefect-KG.

All paths are resolved relative to the project root so the code stays portable
across machines (Windows / Linux) with no hard-coded absolute paths. Every value
can be overridden through an environment variable (see ``.env.example``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
BACKEND_DIR = Path(__file__).resolve().parent           # .../GraphDefect-KG/backend
PROJECT_ROOT = BACKEND_DIR.parent                        # .../GraphDefect-KG
REPO_ROOT = PROJECT_ROOT.parent                          # .../Steel_Surface_Defect_...


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


def _find_dataset_root() -> Optional[Path]:
    """Best-effort discovery of the NEU-DET image folder.

    Looks in a handful of conventional locations. Returns ``None`` if the
    dataset is not present (the app still runs; training is simply disabled).
    """
    override = os.environ.get("NEU_DET_IMAGES")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [
        PROJECT_ROOT / "data" / "raw" / "NEU-DET" / "IMAGES",
        REPO_ROOT / "NEU-DET" / "IMAGES",
        REPO_ROOT / "NEU-DET" / "train" / "images",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


def _find_annotation_root() -> Optional[Path]:
    override = os.environ.get("NEU_DET_ANNOTATIONS")
    candidates: List[Path] = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [
        REPO_ROOT / "NEU-DET" / "ANNOTATIONS",
        REPO_ROOT / "NEU-DET" / "train" / "annotations",
    ]
    for c in candidates:
        if c.exists() and c.is_dir():
            return c.resolve()
    return None


# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #
# Canonical class order used everywhere (model output index -> label).
CLASS_NAMES: List[str] = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches",
]

# Maps the on-disk folder names (NEU-DET) to canonical class labels.
FOLDER_TO_CLASS = {
    "crazing": "Crazing",
    "inclusion": "Inclusion",
    "patches": "Patches",
    "pitted_surface": "Pitted Surface",
    "rolled-in_scale": "Rolled-in Scale",
    "scratches": "Scratches",
}

NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class PreprocessConfig:
    image_size: int = int(os.environ.get("IMAGE_SIZE", 224))
    apply_clahe: bool = os.environ.get("APPLY_CLAHE", "1") == "1"
    clahe_clip: float = 2.0
    clahe_grid: int = 8
    gaussian_ksize: int = 3
    canny_low: int = 50
    canny_high: int = 150
    patch_grid: int = int(os.environ.get("PATCH_GRID", 4))  # NxN patch grid
    # ImageNet normalisation stats for the MobileNetV2 backbone.
    mean: tuple = (0.485, 0.456, 0.406)
    std: tuple = (0.229, 0.224, 0.225)


@dataclass
class GraphConfig:
    knn_k: int = int(os.environ.get("KNN_K", 5))
    metric: str = os.environ.get("KNN_METRIC", "cosine")  # cosine | euclidean | combined
    spatial_weight: float = 0.3          # weight of spatial distance in "combined"
    add_self_loops: bool = True
    symmetric: bool = True


@dataclass
class ModelConfig:
    backbone: str = "mobilenet_v2"
    pretrained: bool = os.environ.get("PRETRAINED", "1") == "1"
    freeze_backbone: bool = True
    cnn_embed_dim: int = 1280            # MobileNetV2 penultimate width
    projected_dim: int = 128            # projection applied to node features
    handcrafted_dim: int = 32           # length of the handcrafted descriptor
    gcn_hidden: int = 128
    gcn_layers: int = 3
    gnn_hidden: int = 128
    gat_heads: int = 4
    dropout: float = 0.3
    fusion_hidden: int = 256


@dataclass
class Settings:
    app_name: str = "GraphDefect-KG"
    app_version: str = "0.1.0"
    host: str = os.environ.get("HOST", "127.0.0.1")
    port: int = int(os.environ.get("PORT", 8000))
    device: str = os.environ.get("DEVICE", "auto")  # auto | cpu | cuda
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", 10))
    allowed_extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp")

    # Directories
    backend_dir: Path = BACKEND_DIR
    project_root: Path = PROJECT_ROOT
    saved_models_dir: Path = _env_path("SAVED_MODELS_DIR", BACKEND_DIR / "saved_models")
    uploads_dir: Path = _env_path("UPLOADS_DIR", BACKEND_DIR / "uploads")
    graph_data_dir: Path = _env_path("GRAPH_DATA_DIR", PROJECT_ROOT / "data" / "graph_data")
    frontend_dir: Path = PROJECT_ROOT / "frontend"

    # Dataset (may be None if not present)
    dataset_images: Optional[Path] = field(default_factory=_find_dataset_root)
    dataset_annotations: Optional[Path] = field(default_factory=_find_annotation_root)

    # Knowledge graph file (generated on first run if missing)
    knowledge_graph_file: Path = _env_path(
        "KG_FILE", PROJECT_ROOT / "data" / "graph_data" / "knowledge_graph.json"
    )

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def ensure_dirs(self) -> None:
        for d in (self.saved_models_dir, self.uploads_dir, self.graph_data_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
