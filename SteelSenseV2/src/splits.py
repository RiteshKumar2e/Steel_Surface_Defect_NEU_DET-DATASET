"""Image-level, stratified, three-way splitting (Reviewer 1, Q1 and Q2).

Protocol, in the order the operations actually happen:

    1. enumerate images and labels                (no pixels read yet)
    2. near-duplicate audit -> duplicate groups   (raw images)
    3. split the GROUPS, stratified by class      <-- the split happens HERE
    4. only then: augment (train groups only)
    5. only then: extract features
    6. only then: fit the discretizer on train
    7. train, select on val
    8. touch test exactly once

Steps 4-7 are in other modules; this module owns steps 1-3 and writes a
manifest that every downstream script reads, so no script can accidentally
re-split. The manifest is hashed and the hash is written next to it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import DATASETS, IMG_EXTS, DatasetSpec, SplitConfig
from duplicates import (
    build_signatures,
    cross_dataset_duplicates,
    duplicate_groups,
    save_report,
)


@dataclass
class Item:
    path: str
    label: str
    stem: str
    split: str = ""
    group: int = -1


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_images(spec: DatasetSpec) -> List[Item]:
    items: List[Item] = []
    for cls in spec.classes:
        d = spec.image_root / cls
        if not d.is_dir():
            raise FileNotFoundError(f"missing class directory: {d}")
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                items.append(Item(path=str(p), label=cls, stem=p.stem))
    if not items:
        raise RuntimeError(f"no images found under {spec.image_root}")
    return items


# ---------------------------------------------------------------------------
# Build the split
# ---------------------------------------------------------------------------


def _stratified_group_split(
    items: Sequence[Item], cfg: SplitConfig
) -> Dict[str, str]:
    """Assign each duplicate GROUP to one split, stratified by class.

    A group inherits the class of its majority member; groups are shuffled
    within a class and dealt out so that the realised proportions track the
    requested ones as closely as integer group sizes allow.
    """
    by_group: Dict[int, List[Item]] = defaultdict(list)
    for it in items:
        by_group[it.group].append(it)

    group_class: Dict[int, str] = {}
    for g, members in by_group.items():
        counts: Dict[str, int] = defaultdict(int)
        for m in members:
            counts[m.label] += 1
        group_class[g] = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

    by_class: Dict[str, List[int]] = defaultdict(list)
    for g, c in group_class.items():
        by_class[c].append(g)

    rng = random.Random(cfg.seed)
    assign: Dict[int, str] = {}
    for c in sorted(by_class):
        groups = sorted(by_class[c], key=lambda g: (-len(by_group[g]), g))
        rng.shuffle(groups)
        n_img = sum(len(by_group[g]) for g in groups)
        want = {
            "train": cfg.train * n_img,
            "val": cfg.val * n_img,
            "test": cfg.test * n_img,
        }
        have = {"train": 0.0, "val": 0.0, "test": 0.0}
        # Largest-remainder greedy: give each group to whichever split is
        # furthest below its quota. Exact for singleton groups, and stable.
        for g in groups:
            deficit = {k: want[k] - have[k] for k in want}
            target = max(deficit.items(), key=lambda kv: (kv[1], kv[0]))[0]
            assign[g] = target
            have[target] += len(by_group[g])

    return {it.path: assign[it.group] for it in items}


def build_split(
    dataset: str,
    cfg: SplitConfig,
    out_dir: Path,
    verbose: bool = True,
) -> pd.DataFrame:
    """Produce (and persist) the frozen manifest for one dataset."""
    spec = DATASETS[dataset]
    items = enumerate_images(spec)
    if verbose:
        print(f"[split] {dataset}: {len(items)} images, {len(spec.classes)} classes")

    audit: Dict[str, object] = {"dataset": dataset, "n_images_raw": len(items)}

    # ---- cross-dataset duplicate removal (SteelDefectX vs NEU-DET) ---------
    dropped: Dict[str, dict] = {}
    if spec.drop_duplicates_of:
        ref_spec = DATASETS[spec.drop_duplicates_of]
        ref_items = enumerate_images(ref_spec)
        if verbose:
            print(
                f"[split] cross-dataset audit against {spec.drop_duplicates_of} "
                f"({len(ref_items)} reference images) ..."
            )
        ref_sig = build_signatures([i.path for i in ref_items], cfg.dhash_bits)
        qry_sig = build_signatures([i.path for i in items], cfg.dhash_bits)
        dropped = cross_dataset_duplicates(
            qry_sig, ref_sig, cfg.xdup_hamming_max, cfg.xdup_corr_min
        )
        label_of = {i.path: i.label for i in items}
        per_class: Dict[str, int] = defaultdict(int)
        for q in dropped:
            per_class[label_of[q]] += 1
        n_by_class: Dict[str, int] = defaultdict(int)
        for i in items:
            n_by_class[i.label] += 1
        audit["cross_dataset"] = {
            "reference": spec.drop_duplicates_of,
            "thresholds": {
                "hamming_max": cfg.xdup_hamming_max,
                "corr_min": cfg.xdup_corr_min,
            },
            "n_dropped": len(dropped),
            "pct_dropped": round(100.0 * len(dropped) / len(items), 2),
            "per_class": {
                c: {
                    "n": n_by_class[c],
                    "dropped": per_class.get(c, 0),
                    "pct": round(100.0 * per_class.get(c, 0) / max(1, n_by_class[c]), 1),
                }
                for c in sorted(n_by_class)
            },
            "examples": [
                {"query": Path(k).name, "ref": Path(v["ref"]).name,
                 "corr": v["corr"], "hamming": v["hamming"]}
                for k, v in list(dropped.items())[:15]
            ],
        }
        items = [i for i in items if i.path not in dropped]
        if verbose:
            print(f"[split] dropped {len(dropped)} cross-dataset duplicates")

        # A class can be wiped out by de-duplication; drop it from the task.
        counts: Dict[str, int] = defaultdict(int)
        for i in items:
            counts[i.label] += 1
        small = [c for c in spec.classes if counts[c] < spec.min_class_size]
        if small:
            audit["classes_removed_too_small"] = {c: counts[c] for c in small}
            items = [i for i in items if i.label not in small]
            if verbose:
                print(f"[split] removed classes below min_class_size: {small}")

    # ---- intra-dataset duplicate grouping ---------------------------------
    sig = build_signatures([i.path for i in items], cfg.dhash_bits)
    label_by_path = {i.path: i.label for i in items}
    sig_labels = [label_by_path[p] for p in sig.paths]
    assign, groups, cross_class = duplicate_groups(
        sig, cfg.dup_hamming_max, cfg.dup_corr_min,
        labels=sig_labels, same_class_only=cfg.group_same_class_only,
    )
    multi = [g for g in groups if len(g) > 1]
    audit["intra_dataset"] = {
        "n_images": len(items),
        "n_groups": len(groups),
        "n_multi_image_groups": len(multi),
        "n_images_in_multi_groups": int(sum(len(g) for g in multi)),
        "largest_group": max((len(g) for g in groups), default=0),
        "policy": cfg.dedupe,
        "thresholds": {
            "hamming_max": cfg.dup_hamming_max,
            "corr_min": cfg.dup_corr_min,
        },
        # Near-identical images carrying DIFFERENT labels. Not merged; reported
        # because they bound how well any model can possibly do.
        "n_cross_class_near_duplicate_pairs": len(cross_class),
        "cross_class_examples": [
            {"a": Path(d["a"]).name, "b": Path(d["b"]).name,
             "label_a": d["label_a"], "label_b": d["label_b"], "corr": d["corr"]}
            for d in cross_class[:10]
        ],
        "duplicate_group_members": [
            [Path(p).name for p in g] for g in multi[:20]
        ],
    }
    if verbose:
        print(
            f"[split] intra-dataset: {len(groups)} groups, "
            f"{len(multi)} contain >1 image ({sum(len(g) for g in multi)} images), "
            f"{len(cross_class)} cross-class near-duplicate pairs"
        )

    if cfg.dedupe == "drop":
        keep = {g[0] for g in groups}
        items = [i for i in items if i.path in keep]
        sig = build_signatures([i.path for i in items], cfg.dhash_bits)
        label_by_path = {i.path: i.label for i in items}
        assign, groups, _ = duplicate_groups(
            sig, cfg.dup_hamming_max, cfg.dup_corr_min,
            labels=[label_by_path[p] for p in sig.paths],
            same_class_only=cfg.group_same_class_only,
        )

    for it in items:
        it.group = assign.get(it.path, -1)

    # ---- assign splits ----------------------------------------------------
    frozen = cfg.reuse_frozen_split
    used_frozen = False
    if dataset == "neu_det" and frozen and Path(frozen).exists():
        prev = pd.read_csv(frozen)
        by_stem = dict(zip(prev["stem"].astype(str), prev["split"].astype(str)))
        if all(it.stem in by_stem for it in items):
            for it in items:
                it.split = by_stem[it.stem]
            used_frozen = True
            if verbose:
                print(f"[split] reusing frozen split {Path(frozen).name} "
                      "(identical to the CNN baselines' split)")
            # A frozen split predates the duplicate audit; verify it anyway.
            straddle = _groups_straddling(items)
            audit["frozen_split_group_violations"] = straddle
            if straddle and cfg.dedupe == "group":
                if verbose:
                    print(f"[split] frozen split puts {straddle} duplicate groups "
                          "on both sides; re-splitting instead")
                used_frozen = False
    if not used_frozen:
        mapping = _stratified_group_split(items, cfg)
        for it in items:
            it.split = mapping[it.path]

    audit["reused_frozen_split"] = used_frozen
    audit["group_violations_final"] = _groups_straddling(items)

    df = pd.DataFrame(
        [
            {
                "stem": it.stem,
                "class": it.label,
                "split": it.split,
                "group": it.group,
                "path": it.path,
            }
            for it in items
        ]
    ).sort_values(["class", "split", "stem"]).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    man = out_dir / f"split_{dataset}.csv"
    df.to_csv(man, index=False)
    h = hashlib.sha256(man.read_bytes()).hexdigest()
    (out_dir / f"split_{dataset}.sha256").write_text(h + "\n", encoding="utf-8")

    audit["counts"] = {
        k: int(v) for k, v in df["split"].value_counts().to_dict().items()
    }
    audit["per_class_counts"] = (
        pd.crosstab(df["class"], df["split"]).to_dict(orient="index")
    )
    audit["manifest_sha256"] = h
    save_report(out_dir / f"split_audit_{dataset}.json", audit)

    if verbose:
        print(f"[split] {audit['counts']}")
        print(f"[split] manifest -> {man}  sha256={h[:16]}...")
    return df


def _groups_straddling(items: Sequence[Item]) -> int:
    seen: Dict[int, set] = defaultdict(set)
    for it in items:
        if it.group >= 0:
            seen[it.group].add(it.split)
    return sum(1 for g, s in seen.items() if len(s) > 1)


def load_split(dataset: str, out_dir: Path) -> pd.DataFrame:
    man = out_dir / f"split_{dataset}.csv"
    if not man.exists():
        raise FileNotFoundError(
            f"{man} not found -- run `python run_all.py --stage split` first"
        )
    expected = (out_dir / f"split_{dataset}.sha256").read_text().strip()
    actual = hashlib.sha256(man.read_bytes()).hexdigest()
    if expected != actual:
        raise RuntimeError(f"split manifest {man} was modified after freezing")
    return pd.read_csv(man)


def get_split_arrays(df: pd.DataFrame) -> Tuple[List[str], np.ndarray, List[str]]:
    classes = sorted(df["class"].unique().tolist())
    idx = {c: i for i, c in enumerate(classes)}
    return (
        df["path"].tolist(),
        np.array([idx[c] for c in df["class"]], dtype=np.int64),
        classes,
    )
