"""End-to-end, per-stage latency at batch size 1 (Reviewer 2, Q6).

The deployment claim in the paper rests on the BiLSTM being small. Small in
parameters is not the same as fast at batch size 1, and the model is only one
stage of a pipeline whose inputs must first be computed from the image by
classical CV operators. A breakdown is the only honest way to present the
claim, so this module measures every stage separately on the same hardware,
batch size 1, after warm-up:

    io_resize      imread + resize to the working resolution
    preprocess     grayscale conversion + Gaussian blur
    intensity      histogram statistics and entropy
    glcm           gray-level co-occurrence at 2 distances x 4 angles
    lbp            local binary pattern histogram
    edges          Canny, Sobel, Laplacian, orientation statistics
    contours       Otsu + morphology + contour geometry
    spectral       2D FFT band energies
    spatial        quadrant and profile statistics
    discretize     bin lookup for all features
    prompt         token string construction
    tokenize       token -> id encoding
    bilstm_x1      one BiLSTM forward pass
    bilstm_ens     the k-member snapshot ensemble
    deepsets_x1    the order-free control, for the cost side of R1 Q6
    localize       proposal generation + NMS

`speedup_report` additionally measures what thread-count parallelism actually
buys, so the "HPC-accelerated" claim can be replaced by a measured number or
dropped.
"""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch

import features as FT
from complexity import hardware, latency
from discretize import Discretizer
from prompt import Tokenizer, token_matrix


def _time(fn: Callable, repeats: int, warmup: int = 10) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ts)
    return {
        "mean_ms": round(statistics.mean(ts), 4),
        "median_ms": round(statistics.median(ts), 4),
        "sd_ms": round(statistics.pstdev(ts), 4),
        "p95_ms": round(s[int(0.95 * (len(s) - 1))], 4),
    }


def profile_pipeline(
    image_path: str,
    fcfg,
    disc: Discretizer,
    tok: Tokenizer,
    model,
    snapshots: Sequence[Dict],
    feature_names: Sequence[str],
    repeats: int = 100,
    include_localization: bool = True,
    deepsets_model=None,
) -> Dict:
    size = fcfg.img_size
    stages: Dict[str, Dict] = {}

    stages["io_resize"] = _time(lambda: FT.load_gray(image_path, size), repeats)
    gray = FT.load_gray(image_path, size)

    stages["preprocess"] = _time(lambda: cv2.GaussianBlur(gray, (5, 5), 0), repeats)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    stages["intensity"] = _time(lambda: FT.f_intensity(gray), repeats)
    stages["glcm"] = _time(lambda: FT.f_glcm(gray, fcfg), repeats)
    stages["lbp"] = _time(lambda: FT.f_lbp(gray, fcfg), repeats)
    stages["edges"] = _time(lambda: FT.f_edges(gray, blur), repeats)
    stages["contours"] = _time(lambda: FT.f_contours(gray, blur), repeats)
    stages["spectral"] = _time(lambda: FT.f_spectral(gray), repeats)
    stages["spatial"] = _time(lambda: FT.f_spatial(gray, blur), repeats)

    feats = FT.extract(gray, fcfg)
    X = np.asarray([[feats[k] for k in feature_names]], dtype=np.float32)
    stages["discretize"] = _time(lambda: disc.transform(X, count_oor=False), repeats)
    B, O = disc.transform(X, count_oor=False)

    stages["prompt_build"] = _time(
        lambda: token_matrix(B, O, feature_names, disc.oor_policy), repeats
    )
    toks = token_matrix(B, O, feature_names, disc.oor_policy)
    stages["tokenize"] = _time(lambda: tok.encode_batch(toks), repeats)
    ids = torch.from_numpy(tok.encode_batch(toks))

    model.load_state_dict(snapshots[0])
    model.eval()
    with torch.no_grad():
        stages["bilstm_forward_x1"] = _time(lambda: model(ids), repeats)

        def _ens():
            acc = None
            for sd in snapshots:
                model.load_state_dict(sd)
                p = torch.softmax(model(ids), dim=-1)
                acc = p if acc is None else acc + p
            return acc / len(snapshots)

        stages[f"bilstm_ensemble_x{len(snapshots)}"] = _time(_ens, max(20, repeats // 4))

    if include_localization:
        from localize import propose

        stages["localization_proposals"] = _time(lambda: propose(gray), max(20, repeats // 4))

    # The order-free control from src/model.py, timed on the identical input.
    # Reviewer 1 Q6 is a question about accuracy; this is the cost side of the
    # same trade-off, and the two have to be read together.
    if deepsets_model is not None:
        deepsets_model.eval()
        with torch.no_grad():
            stages["deepsets_forward_x1"] = _time(lambda: deepsets_model(ids), repeats)

    feat_stages = ["intensity", "glcm", "lbp", "edges", "contours", "spectral", "spatial"]
    ens_key = f"bilstm_ensemble_x{len(snapshots)}"
    totals = {
        "feature_extraction_ms": round(sum(stages[k]["median_ms"] for k in feat_stages), 3),
        "representation_ms": round(
            sum(stages[k]["median_ms"] for k in ("discretize", "prompt_build", "tokenize")), 3
        ),
        "inference_single_ms": round(stages["bilstm_forward_x1"]["median_ms"], 3),
        "inference_ensemble_ms": round(stages[ens_key]["median_ms"], 3),
    }
    totals["end_to_end_single_ms"] = round(
        stages["io_resize"]["median_ms"]
        + stages["preprocess"]["median_ms"]
        + totals["feature_extraction_ms"]
        + totals["representation_ms"]
        + totals["inference_single_ms"],
        3,
    )
    totals["end_to_end_ensemble_ms"] = round(
        totals["end_to_end_single_ms"]
        - totals["inference_single_ms"]
        + totals["inference_ensemble_ms"],
        3,
    )
    if include_localization:
        totals["end_to_end_with_localization_ms"] = round(
            totals["end_to_end_ensemble_ms"] + stages["localization_proposals"]["median_ms"], 3
        )
    totals["throughput_img_per_s_single"] = round(1000.0 / totals["end_to_end_single_ms"], 2)
    totals["inference_share_of_end_to_end_pct"] = round(
        100.0 * totals["inference_ensemble_ms"] / totals["end_to_end_ensemble_ms"], 2
    )
    if "deepsets_forward_x1" in stages:
        ds = stages["deepsets_forward_x1"]["median_ms"]
        totals["deepsets_forward_ms"] = round(ds, 3)
        totals["bilstm_vs_deepsets_speed_ratio"] = round(
            stages["bilstm_forward_x1"]["median_ms"] / max(ds, 1e-6), 2
        )

    return {
        "hardware": hardware(),
        "batch_size": 1,
        "repeats": repeats,
        "image": Path(image_path).name,
        "image_size": size,
        "stages_ms": stages,
        "totals": totals,
        "note": (
            "Read `totals` before making any deployment claim, and quote "
            "end_to_end_*, never the parameter count. On CPU at batch size 1 "
            "the BiLSTM forward pass is usually NOT negligible relative to "
            "descriptor extraction: an LSTM is sequential over the token "
            "sequence, so its cost is dominated by per-timestep kernel "
            "overhead rather than by its MAC count. A 1.75M-parameter model is "
            "not automatically a fast one at batch size 1, and "
            "`inference_share_of_end_to_end_pct` is the number that settles it "
            "for this hardware. The snapshot ensemble multiplies that share by "
            "`ensemble_size`."
        ),
    }


def speedup_report(
    image_paths: Sequence[str], fcfg, thread_counts: Sequence[int] = (1, 2, 4, 8)
) -> Dict:
    """Measured parallel speed-up over a batch of images (Reviewer 2, Q6).

    Replaces the unquantified "HPC-accelerated" claim with a curve. Feature
    extraction is per-image independent, so this is the part that parallelises;
    the measurement uses a thread pool over `image_paths`.
    """
    from concurrent.futures import ThreadPoolExecutor

    def work(p):
        return FT.extract(FT.load_gray(p, fcfg.img_size), fcfg)

    prev = torch.get_num_threads()
    out: Dict[str, object] = {"n_images": len(image_paths), "hardware": hardware()}
    timings = {}
    base = None
    for nt in thread_counts:
        cv2.setNumThreads(1)          # avoid double-counting OpenCV's own pool
        torch.set_num_threads(1)
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=nt) as ex:
            list(ex.map(work, image_paths))
        el = time.perf_counter() - t0
        base = base or el
        timings[str(nt)] = {
            "seconds": round(el, 3),
            "img_per_s": round(len(image_paths) / el, 2),
            "speedup_vs_1_thread": round(base / el, 2),
            "parallel_efficiency": round(base / el / nt, 3),
        }
    cv2.setNumThreads(-1)
    torch.set_num_threads(prev)
    out["by_thread_count"] = timings
    return out
