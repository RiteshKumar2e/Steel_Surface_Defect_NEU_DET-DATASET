"""Configurable image preprocessing pipeline for steel surface images.

Everything operates on grayscale/BGR numpy arrays (OpenCV convention) except the
final tensor produced for the CNN backbone. The pipeline is deliberately
composable: each step is a small pure function and :func:`preprocess_image`
wires them together according to :class:`~backend.config.PreprocessConfig`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..config import PreprocessConfig, settings
from .helpers import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Basic ops
# --------------------------------------------------------------------------- #
def load_image(path: str) -> np.ndarray:
    """Read an image from disk as BGR uint8. Raises on failure."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img


def decode_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (from an upload) into a BGR uint8 array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Uploaded data could not be decoded as an image")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize(img: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def gaussian(img: np.ndarray, ksize: int) -> np.ndarray:
    if ksize <= 1:
        return img
    k = ksize if ksize % 2 == 1 else ksize + 1
    return cv2.GaussianBlur(img, (k, k), 0)


def canny_edges(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, low, high)


def normalize_unit(img: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] float32."""
    return img.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Patches / regions / contours
# --------------------------------------------------------------------------- #
@dataclass
class Patch:
    index: int
    row: int
    col: int
    x0: int
    y0: int
    x1: int
    y1: int
    image: np.ndarray  # grayscale patch

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    def bbox(self) -> Dict[str, int]:
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1}


def generate_patches(gray: np.ndarray, grid: int) -> List[Patch]:
    """Split an image into a ``grid x grid`` set of non-overlapping patches."""
    h, w = gray.shape[:2]
    ph, pw = h // grid, w // grid
    patches: List[Patch] = []
    idx = 0
    for r in range(grid):
        for c in range(grid):
            y0, x0 = r * ph, c * pw
            y1 = (r + 1) * ph if r < grid - 1 else h
            x1 = (c + 1) * pw if c < grid - 1 else w
            patches.append(
                Patch(idx, r, c, x0, y0, x1, y1, gray[y0:y1, x0:x1].copy())
            )
            idx += 1
    return patches


def segment_regions(gray: np.ndarray, n_regions: int = 6) -> np.ndarray:
    """Simple intensity-threshold region map (Otsu + connected components).

    Returns an int label map. Used to derive region-level nodes. This is an
    unsupervised proxy for defect regions, not a trained segmenter.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n_labels, labels = cv2.connectedComponents(binary)
    return labels


def extract_contours(gray: np.ndarray) -> List[np.ndarray]:
    """Return external contours found on an Otsu-thresholded image."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    original_bgr: np.ndarray
    resized_bgr: np.ndarray
    gray: np.ndarray            # enhanced grayscale (uint8)
    edges: np.ndarray           # Canny edge map
    patches: List[Patch]
    region_map: np.ndarray
    size: int


def preprocess_image(
    img_bgr: np.ndarray, cfg: Optional[PreprocessConfig] = None
) -> PreprocessResult:
    """Run the full configurable preprocessing pipeline on a BGR image."""
    cfg = cfg or settings.preprocess
    resized = resize(img_bgr, cfg.image_size)
    gray = to_gray(resized)
    gray = gaussian(gray, cfg.gaussian_ksize)
    if cfg.apply_clahe:
        gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_grid)
    edges = canny_edges(gray, cfg.canny_low, cfg.canny_high)
    patches = generate_patches(gray, cfg.patch_grid)
    region_map = segment_regions(gray)
    return PreprocessResult(
        original_bgr=img_bgr,
        resized_bgr=resized,
        gray=gray,
        edges=edges,
        patches=patches,
        region_map=region_map,
        size=cfg.image_size,
    )


def to_cnn_tensor(resized_bgr: np.ndarray, cfg: Optional[PreprocessConfig] = None):
    """Convert a resized BGR image into a normalised ``1x3xHxW`` torch tensor."""
    import torch

    cfg = cfg or settings.preprocess
    rgb = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
    x = normalize_unit(rgb)                       # HxWx3 in [0,1]
    mean = np.array(cfg.mean, dtype=np.float32)
    std = np.array(cfg.std, dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))                # 3xHxW
    return torch.from_numpy(x).unsqueeze(0).float()
