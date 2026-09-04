"""Near-duplicate auditing (Reviewer 1 Q2, Reviewer 1/2 leakage concerns).

Two audits are provided and both are run before any model is trained.

1. INTRA-dataset. Groups near-duplicate images inside one dataset so that a
   duplicate group can never straddle the train/test boundary.

2. CROSS-dataset. Compares SteelDefectX against NEU-DET. This audit is the
   reason the revision exists: four SteelDefectX classes are pixel copies of
   NEU-DET images re-encoded at 256x256, so the "two independent datasets"
   claim in the original submission does not hold.

Matching is a two-stage test, deliberately conservative:
    stage 1  difference hash (dHash, 64 bit), Hamming distance <= dup_hamming_max
    stage 2  Pearson correlation of the 64x64 z-scored grayscale image >= dup_corr_min

Stage 1 is a cheap recall filter; stage 2 rejects the false positives that a
64-bit hash produces on low-texture surfaces such as pitted_surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def dhash(gray: np.ndarray, bits: int = 8) -> np.ndarray:
    r = cv2.resize(gray, (bits + 1, bits), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).ravel()


def thumb(gray: np.ndarray, size: int = 64) -> np.ndarray:
    v = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
    return (v - v.mean()) / (v.std() + 1e-6)


@dataclass
class Signatures:
    paths: List[str]
    hashes: np.ndarray  # (N, bits*bits) bool
    thumbs: np.ndarray  # (N, size*size) float32, z-scored

    def __len__(self) -> int:
        return len(self.paths)


def build_signatures(paths: Sequence[str], bits: int = 8, size: int = 64) -> Signatures:
    hs, ts, keep = [], [], []
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        hs.append(dhash(g, bits))
        ts.append(thumb(g, size))
        keep.append(str(p))
    return Signatures(keep, np.array(hs, dtype=bool), np.array(ts, dtype=np.float32))


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation of z-scored vectors."""
    return (a @ b) / float(b.shape[0])


# ---------------------------------------------------------------------------
# Cross-dataset audit
# ---------------------------------------------------------------------------


def cross_dataset_duplicates(
    query: Signatures,
    reference: Signatures,
    hamming_max: int = 4,
    corr_min: float = 0.95,
) -> Dict[str, dict]:
    """For every query image, its best matching reference image if it is a
    near-duplicate. Returns {query_path: {ref, corr, hamming}}."""
    out: Dict[str, dict] = {}
    if len(query) == 0 or len(reference) == 0:
        return out
    ref_h = reference.hashes
    for i, qp in enumerate(query.paths):
        hd = (ref_h != query.hashes[i]).sum(axis=1)
        cand = np.flatnonzero(hd <= hamming_max)
        if cand.size == 0:
            continue
        c = _corr(reference.thumbs[cand], query.thumbs[i])
        j = int(np.argmax(c))
        if float(c[j]) >= corr_min:
            out[qp] = {
                "ref": reference.paths[int(cand[j])],
                "corr": round(float(c[j]), 4),
                "hamming": int(hd[cand[j]]),
            }
    return out


# ---------------------------------------------------------------------------
# Intra-dataset audit
# ---------------------------------------------------------------------------


def duplicate_groups(
    sig: Signatures,
    hamming_max: int = 2,
    corr_min: float = 0.98,
    labels: Optional[Sequence[str]] = None,
    same_class_only: bool = True,
) -> Tuple[Dict[str, int], List[List[str]], List[dict]]:
    """Union-find over near-duplicate pairs.

    Returns (path -> group id, groups, cross_class_pairs).

    Every image gets a group id; a singleton is its own group. Splitting on the
    group id instead of the image id is what makes the train/test boundary
    duplicate-free.

    When `same_class_only` is set, a near-duplicate pair whose two images carry
    different labels is NOT merged. Merging those would chain unrelated
    low-texture images into one giant component through transitive closure, and
    a cross-class pair is a labelling question rather than a leakage one, so it
    is returned separately for reporting.
    """
    n = len(sig)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    H = sig.hashes.astype(np.uint8)
    cross_class: List[dict] = []
    for i in range(n):
        if i + 1 >= n:
            break
        hd = (H[i + 1 :] != H[i]).sum(axis=1)
        cand = np.flatnonzero(hd <= hamming_max) + i + 1
        if cand.size == 0:
            continue
        c = _corr(sig.thumbs[cand], sig.thumbs[i])
        for k, j in enumerate(cand):
            if float(c[k]) < corr_min:
                continue
            j = int(j)
            if labels is not None and labels[i] != labels[j]:
                cross_class.append(
                    {
                        "a": sig.paths[i],
                        "b": sig.paths[j],
                        "label_a": labels[i],
                        "label_b": labels[j],
                        "corr": round(float(c[k]), 4),
                        "hamming": int(hd[j - i - 1]),
                    }
                )
                if same_class_only:
                    continue
            union(i, j)

    roots: Dict[int, int] = {}
    assign: Dict[str, int] = {}
    for i, p in enumerate(sig.paths):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        assign[p] = roots[r]
    groups: List[List[str]] = [[] for _ in range(len(roots))]
    for p, g in assign.items():
        groups[g].append(p)
    return assign, groups, cross_class


def summarize(report: Dict[str, dict], class_of) -> dict:
    per_class: Dict[str, int] = {}
    for q in report:
        c = class_of(q)
        per_class[c] = per_class.get(c, 0) + 1
    return {"n_duplicates": len(report), "per_class": per_class}


def save_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
