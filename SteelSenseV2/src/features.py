"""Handcrafted descriptor extraction.

Produces a fixed-length, fixed-order vector of named scalars. Every downstream
consumer -- the prompt builder, the tabular baselines, permutation importance --
reads the SAME vector, so the comparison between "text prompt + BiLSTM" and
"raw vector + XGBoost" differs only in the representation, never in the
information available (Reviewer 1, Q4).

Extraction is deterministic and depends on nothing but the image and
FeatureConfig, so results are cached on disk keyed by config + file content.

Augmentation (`augment_image`) is image-space, not feature-space, and is only
ever applied to images the split manifest marks as `train`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from scipy import stats as sstats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

from config import FeatureConfig

FEATURE_VERSION = "v2.1"

# Stage names used by the per-stage latency profiler (Reviewer 2, Q6).
STAGES = [
    "io_resize",
    "preprocess",
    "intensity",
    "glcm",
    "lbp",
    "edges",
    "contours",
    "spectral",
    "spatial",
]


# ---------------------------------------------------------------------------
# Feature families
# ---------------------------------------------------------------------------


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    p = hist / (hist.sum() + 1e-9)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def f_intensity(gray: np.ndarray) -> Dict[str, float]:
    g = gray.astype(np.float32)
    flat = g.ravel()
    p10, p25, p50, p75, p90 = np.percentile(flat, [10, 25, 50, 75, 90])
    return {
        "int_mean": float(g.mean()),
        "int_std": float(g.std()),
        "int_skew": float(sstats.skew(flat)),
        "int_kurtosis": float(sstats.kurtosis(flat)),
        "int_entropy": _entropy(gray),
        "int_p10": float(p10),
        "int_p50": float(p50),
        "int_p90": float(p90),
        "int_iqr": float(p75 - p25),
        "int_range_ratio": float((p90 - p10) / (g.std() + 1e-6)),
        "int_dark_frac": float((flat < p50 - 1.0 * g.std()).mean()),
        "int_bright_frac": float((flat > p50 + 1.0 * g.std()).mean()),
        "int_michelson": float((flat.max() - flat.min()) / (flat.max() + flat.min() + 1e-6)),
    }


def f_glcm(gray: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    """GLCM at four orientations. Reported as the orientation MEAN (a rotation
    invariant summary) and the orientation RANGE (an anisotropy measure, which
    is what separates scratches from crazing)."""
    lv = cfg.glcm_levels
    q = (gray.astype(np.uint16) * lv // 256).astype(np.uint8)
    angles = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    out: Dict[str, float] = {}
    glcm = graycomatrix(
        q, distances=list(cfg.glcm_distances), angles=angles,
        levels=lv, symmetric=True, normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        v = graycoprops(glcm, prop)  # (n_dist, n_angle)
        for di, d in enumerate(cfg.glcm_distances):
            row = v[di]
            out[f"glcm_d{d}_{prop}_mean"] = float(np.mean(row))
            out[f"glcm_d{d}_{prop}_range"] = float(np.max(row) - np.min(row))
    return out


def f_lbp(gray: np.ndarray, cfg: FeatureConfig) -> Dict[str, float]:
    P, R = cfg.lbp_points, cfg.lbp_radius
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    nb = P + 2
    hist, _ = np.histogram(lbp, bins=nb, range=(0, nb))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-9)
    out = {f"lbp_h{i}": float(hist[i]) for i in range(nb)}
    nz = hist[hist > 0]
    out["lbp_entropy"] = float(-(nz * np.log2(nz)).sum())
    out["lbp_uniformity"] = float((hist[: P + 1]).sum())
    return out


def f_edges(gray: np.ndarray, blur: np.ndarray) -> Dict[str, float]:
    e_lo = cv2.Canny(blur, 30, 90)
    e_hi = cv2.Canny(blur, 80, 200)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    ang = np.arctan2(gy, gx)
    strong = mag > (mag.mean() + mag.std())
    if strong.sum() > 10:
        a = ang[strong] * 2.0  # orientation is pi-periodic
        C, S = float(np.cos(a).mean()), float(np.sin(a).mean())
        r = float(np.hypot(C, S))
        dom = float(0.5 * np.arctan2(S, C))
    else:
        r, dom = 0.0, 0.0
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    h_e = float(np.abs(gx).mean())
    v_e = float(np.abs(gy).mean())
    return {
        "edge_canny_lo": float((e_lo > 0).mean()),
        "edge_canny_hi": float((e_hi > 0).mean()),
        "edge_sobel_mean": float(mag.mean()),
        "edge_sobel_std": float(mag.std()),
        "edge_sobel_p95": float(np.percentile(mag, 95)),
        "edge_lap_var": float(lap.var()),
        "edge_orient_coherence": r,          # 0 = isotropic, 1 = one direction
        "edge_orient_dominant": dom,
        "edge_hv_ratio": float(h_e / (v_e + 1e-6)),
        "edge_hv_asym": float(abs(h_e - v_e) / (h_e + v_e + 1e-6)),
    }


def f_contours(gray: np.ndarray, blur: np.ndarray) -> Tuple[Dict[str, float], List[List[int]], List[float]]:
    h, w = gray.shape
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=1)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(8.0, 0.00015 * h * w)

    areas, aspects, circ, solidity, extent, boxes, scores = [], [], [], [], [], [], []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        per = float(cv2.arcLength(c, True))
        hull = cv2.convexHull(c)
        ha = float(cv2.contourArea(hull)) + 1e-6
        areas.append(a)
        aspects.append(float(max(bw, bh) / (min(bw, bh) + 1e-6)))
        circ.append(float(4.0 * np.pi * a / (per * per + 1e-6)))
        solidity.append(float(a / ha))
        extent.append(float(a / (bw * bh + 1e-6)))
        boxes.append([int(x), int(y), int(x + bw), int(y + bh)])
        # Region saliency: local contrast against the surrounding ring.
        pad = 6
        y0, y1 = max(0, y - pad), min(h, y + bh + pad)
        x0, x1 = max(0, x - pad), min(w, x + bw + pad)
        inner = gray[y : y + bh, x : x + bw].astype(np.float32)
        outer = gray[y0:y1, x0:x1].astype(np.float32)
        scores.append(float(abs(inner.mean() - outer.mean()) / (outer.std() + 1e-3)))

    n = len(areas)
    area_arr = np.array(areas) if n else np.zeros(1)
    total = float(area_arr.sum()) if n else 0.0
    feats = {
        "cnt_n": float(n),
        "cnt_n_tiny": float(sum(1 for a in areas if a < 0.001 * h * w)),
        "cnt_n_small": float(sum(1 for a in areas if 0.001 * h * w <= a < 0.01 * h * w)),
        "cnt_n_large": float(sum(1 for a in areas if a >= 0.01 * h * w)),
        "cnt_coverage": float(total / (h * w)),
        "cnt_area_mean": float(area_arr.mean() / (h * w)) if n else 0.0,
        "cnt_area_max": float(area_arr.max() / (h * w)) if n else 0.0,
        "cnt_area_cv": float(area_arr.std() / (area_arr.mean() + 1e-6)) if n else 0.0,
        "cnt_aspect_mean": float(np.mean(aspects)) if n else 0.0,
        "cnt_aspect_max": float(np.max(aspects)) if n else 0.0,
        "cnt_circularity_mean": float(np.mean(circ)) if n else 0.0,
        "cnt_solidity_mean": float(np.mean(solidity)) if n else 0.0,
        "cnt_extent_mean": float(np.mean(extent)) if n else 0.0,
        "cnt_fg_frac": float((th > 0).mean()),
    }
    return feats, boxes, scores


def f_spectral(gray: np.ndarray) -> Dict[str, float]:
    g = gray.astype(np.float32)
    g = g - g.mean()
    F = np.fft.fftshift(np.abs(np.fft.fft2(g)))
    h, w = F.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    rad = np.hypot(yy - cy, xx - cx) / (min(h, w) / 2.0)
    tot = float(F.sum()) + 1e-9
    out = {}
    edges_r = [0.0, 0.1, 0.25, 0.5, 1.0]
    for i in range(4):
        m = (rad >= edges_r[i]) & (rad < edges_r[i + 1])
        out[f"fft_band{i}"] = float(F[m].sum() / tot)
    # Angular anisotropy of the spectrum: high for directional textures.
    ang = np.arctan2(yy - cy, xx - cx)
    sectors = []
    for i in range(4):
        m = (ang >= -np.pi / 2 + i * np.pi / 4) & (ang < -np.pi / 2 + (i + 1) * np.pi / 4)
        sectors.append(float(F[m].sum() / tot))
    s = np.array(sectors)
    out["fft_aniso"] = float((s.max() - s.min()) / (s.mean() + 1e-9))
    return out


def f_spatial(gray: np.ndarray, blur: np.ndarray) -> Dict[str, float]:
    h, w = gray.shape
    e = cv2.Canny(blur, 50, 150)
    out: Dict[str, float] = {}
    quads = []
    for qi, (ys, xs) in enumerate(
        [(slice(0, h // 2), slice(0, w // 2)), (slice(0, h // 2), slice(w // 2, w)),
         (slice(h // 2, h), slice(0, w // 2)), (slice(h // 2, h), slice(w // 2, w))]
    ):
        sub = gray[ys, xs]
        ed = float((e[ys, xs] > 0).mean())
        out[f"q{qi}_std"] = float(sub.std())
        out[f"q{qi}_edge"] = ed
        quads.append(ed)
    q = np.array(quads)
    out["quad_edge_spread"] = float(q.max() - q.min())
    out["quad_edge_cv"] = float(q.std() / (q.mean() + 1e-6))
    rows = (e > 0).mean(axis=1)
    cols = (e > 0).mean(axis=0)
    out["row_peak_pos"] = float(np.argmax(rows) / max(1, h - 1))
    out["col_peak_pos"] = float(np.argmax(cols) / max(1, w - 1))
    out["row_profile_cv"] = float(rows.std() / (rows.mean() + 1e-6))
    out["col_profile_cv"] = float(cols.std() / (cols.mean() + 1e-6))
    return out


# ---------------------------------------------------------------------------
# Top-level extraction
# ---------------------------------------------------------------------------


def extract(image: np.ndarray, cfg: FeatureConfig, with_regions: bool = False):
    """image: uint8 grayscale, already resized to cfg.img_size."""
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    feats: Dict[str, float] = {}
    feats.update(f_intensity(image))
    feats.update(f_glcm(image, cfg))
    feats.update(f_lbp(image, cfg))
    feats.update(f_edges(image, blur))
    cf, boxes, scores = f_contours(image, blur)
    feats.update(cf)
    feats.update(f_spectral(image))
    feats.update(f_spatial(image, blur))
    feats = {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in feats.items()}
    if with_regions:
        return feats, boxes, scores
    return feats


def load_gray(path: str, size: int) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"unreadable image: {path}")
    if img.shape[:2] != (size, size):
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


# ---------------------------------------------------------------------------
# Augmentation -- TRAIN SPLIT ONLY
# ---------------------------------------------------------------------------


def augment_image(img: np.ndarray, variant: int, seed: int = 0) -> np.ndarray:
    """Deterministic image-space augmentation.

    variant 0 is the identity. Variants are geometric/photometric transforms of
    the ORIGINAL IMAGE, applied before feature extraction, so an augmented
    sample is a genuinely different descriptor vector rather than a jittered
    copy of one. This is the change that removes the near-duplicate coupling
    the original feature-space jitter created.
    """
    if variant <= 0:
        return img
    rng = np.random.RandomState((seed * 977 + variant * 7919) % (2**31 - 1))
    out = img
    op = variant % 6
    if op == 1:
        out = cv2.flip(out, 1)
    elif op == 2:
        out = cv2.flip(out, 0)
    elif op == 3:
        out = cv2.rotate(out, cv2.ROTATE_90_CLOCKWISE)
    elif op == 4:
        out = cv2.rotate(out, cv2.ROTATE_180)
    elif op == 5:
        out = cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gamma = float(rng.uniform(0.82, 1.22))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    out = cv2.LUT(out, lut)
    if rng.rand() < 0.5:
        noise = rng.normal(0.0, 3.0, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(out)


# ---------------------------------------------------------------------------
# Cached batch extraction
# ---------------------------------------------------------------------------


def _cache_key(paths: Sequence[str], cfg: FeatureConfig, variants: int, tag: str) -> str:
    h = hashlib.sha256()
    h.update(FEATURE_VERSION.encode())
    h.update(repr((cfg.img_size, cfg.glcm_levels, cfg.glcm_distances,
                   cfg.lbp_points, cfg.lbp_radius)).encode())
    h.update(f"{variants}|{tag}".encode())
    for p in paths:
        h.update(Path(p).name.encode())
    return h.hexdigest()[:20]


def extract_many(
    paths: Sequence[str],
    labels: np.ndarray,
    cfg: FeatureConfig,
    variants: int = 0,
    cache_dir: Path = None,
    tag: str = "",
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Returns (X, y, feature_names, source_index).

    `source_index[i]` is the index into `paths` that row i came from, so
    augmented rows can always be traced back to their source image.
    """
    key = _cache_key(paths, cfg, variants, tag)
    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = cache_dir / f"feat_{key}.npz"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if verbose:
                print(f"[features] cache hit {cache.name}  X={z['X'].shape}")
            return z["X"], z["y"], list(z["names"]), z["src"]

    rows, ys, src = [], [], []
    names: List[str] = []
    for i, p in enumerate(paths):
        base = load_gray(p, cfg.img_size)
        for v in range(variants + 1):
            img = augment_image(base, v, seed=i)
            f = extract(img, cfg)
            if not names:
                names = sorted(f.keys())
            rows.append([f[k] for k in names])
            ys.append(labels[i])
            src.append(i)
        if verbose and (i + 1) % 250 == 0:
            print(f"[features] {i + 1}/{len(paths)} images")

    X = np.asarray(rows, dtype=np.float32)
    y = np.asarray(ys, dtype=np.int64)
    s = np.asarray(src, dtype=np.int64)
    if cache is not None:
        np.savez_compressed(cache, X=X, y=y, names=np.array(names), src=s)
        if verbose:
            print(f"[features] cached -> {cache.name}  X={X.shape}")
    return X, y, names, s
