"""Handcrafted (interpretable) visual descriptors for steel surface regions.

Produces a fixed-length descriptor (``HANDCRAFTED_DIM`` values) plus a dictionary
of named, human-readable properties that are surfaced in the explanation and the
knowledge-graph reasoning. The same routine is applied to whole images, patches
and regions so every graph node carries comparable features.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# Descriptor layout (order matters — kept in sync with FEATURE_NAMES).
LBP_BINS = 10          # uniform LBP with P=8 -> 10 bins
HANDCRAFTED_DIM = 32

FEATURE_NAMES = [
    "mean_intensity", "std_intensity", "entropy", "edge_density",
    "sobel_response", "laplacian_variance",
    "glcm_contrast", "glcm_correlation", "glcm_energy", "glcm_homogeneity",
    *[f"lbp_{i}" for i in range(LBP_BINS)],
    "contour_area", "perimeter", "circularity", "aspect_ratio",
    "extent", "solidity", "orientation", "compactness",
    "roughness", "bbox_width", "bbox_height", "num_contours",
]
assert len(FEATURE_NAMES) == HANDCRAFTED_DIM, len(FEATURE_NAMES)


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def _glcm_props(gray: np.ndarray) -> Tuple[float, float, float, float]:
    # Quantise to 32 levels for a compact, stable GLCM.
    q = (gray.astype(np.float32) / 255.0 * 31).astype(np.uint8)
    glcm = graycomatrix(
        q, distances=[1], angles=[0, np.pi / 2], levels=32,
        symmetric=True, normed=True,
    )
    contrast = float(graycoprops(glcm, "contrast").mean())
    corr = float(np.nan_to_num(graycoprops(glcm, "correlation").mean()))
    energy = float(graycoprops(glcm, "energy").mean())
    homog = float(graycoprops(glcm, "homogeneity").mean())
    return contrast, corr, energy, homog


def _lbp_hist(gray: np.ndarray) -> np.ndarray:
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS), density=True)
    return hist.astype(np.float32)


def _shape_stats(gray: np.ndarray) -> Dict[str, float]:
    """Contour/shape descriptors from the largest external contour."""
    h, w = gray.shape[:2]
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stats = {
        "contour_area": 0.0, "perimeter": 0.0, "circularity": 0.0,
        "aspect_ratio": 1.0, "extent": 0.0, "solidity": 0.0,
        "orientation": 0.0, "compactness": 0.0, "bbox_width": 0.0,
        "bbox_height": 0.0, "num_contours": float(len(contours)),
    }
    if not contours:
        return stats
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perim = float(cv2.arcLength(c, True))
    x, y, bw, bh = cv2.boundingRect(c)
    hull = cv2.convexHull(c)
    hull_area = float(cv2.contourArea(hull)) + 1e-6
    img_area = float(h * w)
    stats.update(
        contour_area=area / img_area,
        perimeter=perim / (2 * (h + w)),
        circularity=float(4 * np.pi * area / (perim ** 2 + 1e-6)),
        aspect_ratio=float(bw / (bh + 1e-6)),
        extent=float(area / (bw * bh + 1e-6)),
        solidity=float(area / hull_area),
        compactness=float(perim ** 2 / (area + 1e-6)) / 1000.0,
        bbox_width=float(bw / w),
        bbox_height=float(bh / h),
    )
    if len(c) >= 5:
        (_, _), (_, _), angle = cv2.fitEllipse(c)
        stats["orientation"] = float(angle / 180.0)
    return stats


def compute_named_features(gray: np.ndarray) -> Dict[str, float]:
    """Return a dict of all named handcrafted features for a grayscale image."""
    gray = gray if gray.dtype == np.uint8 else gray.astype(np.uint8)
    mean = float(gray.mean() / 255.0)
    std = float(gray.std() / 255.0)
    ent = _entropy(gray) / 8.0  # max entropy for 8-bit is 8
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float((edges > 0).mean())
    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=3)
    sobel_resp = float(np.abs(sobel).mean() / 255.0)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var() / 10000.0)
    contrast, corr, energy, homog = _glcm_props(gray)
    lbp = _lbp_hist(gray)
    shape = _shape_stats(gray)
    roughness = float(std * edge_density)

    feats: Dict[str, float] = {
        "mean_intensity": mean,
        "std_intensity": std,
        "entropy": ent,
        "edge_density": edge_density,
        "sobel_response": sobel_resp,
        "laplacian_variance": lap_var,
        "glcm_contrast": contrast,
        "glcm_correlation": corr,
        "glcm_energy": energy,
        "glcm_homogeneity": homog,
        **{f"lbp_{i}": float(lbp[i]) for i in range(LBP_BINS)},
        "contour_area": shape["contour_area"],
        "perimeter": shape["perimeter"],
        "circularity": shape["circularity"],
        "aspect_ratio": shape["aspect_ratio"],
        "extent": shape["extent"],
        "solidity": shape["solidity"],
        "orientation": shape["orientation"],
        "compactness": shape["compactness"],
        "roughness": roughness,
        "bbox_width": shape["bbox_width"],
        "bbox_height": shape["bbox_height"],
        "num_contours": shape["num_contours"] / 50.0,  # scaled
    }
    return feats


def compute_handcrafted_vector(gray: np.ndarray) -> np.ndarray:
    """Fixed-length ``HANDCRAFTED_DIM`` descriptor (numpy float32)."""
    feats = compute_named_features(gray)
    vec = np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def dominant_visual_properties(feats: Dict[str, float]) -> Dict[str, float]:
    """Return a compact, interpretable subset used by the reasoning module."""
    keys = [
        "edge_density", "std_intensity", "entropy", "glcm_contrast",
        "glcm_homogeneity", "circularity", "aspect_ratio", "orientation",
        "roughness", "solidity",
    ]
    return {k: round(float(feats.get(k, 0.0)), 4) for k in keys}
