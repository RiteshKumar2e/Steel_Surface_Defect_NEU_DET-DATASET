"""Assembles the frozen split into tensors, in the legal order.

    split manifest (images)
        -> augment TRAIN ONLY
        -> extract features (train / val / test separately)
        -> fit discretizer on TRAIN ONLY
        -> transform val and test with the train-fitted edges, counting OOR
        -> build prompts, fit tokenizer from the SCHEMA
        -> encode

Nothing in this module can see the test split before the discretizer is fitted,
because the fit happens on `Bundle.train.X` which is built from train paths
only. That ordering is the answer to Reviewer 1 Q1/Q2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from config import CACHE_DIR, ExperimentConfig
from discretize import Discretizer
from features import extract_many
from prompt import Tokenizer, apply_order, token_matrix


@dataclass
class SplitData:
    name: str
    paths: List[str]
    X: np.ndarray            # raw un-discretized descriptors (baselines use this)
    y: np.ndarray
    src: np.ndarray          # index into `paths` of the source image
    ids: Optional[np.ndarray] = None   # token ids
    tokens: Optional[List[List[str]]] = None

    def __len__(self) -> int:
        return len(self.y)


@dataclass
class Bundle:
    train: SplitData
    val: SplitData
    test: SplitData
    classes: List[str]
    feature_names: List[str]
    tokenizer: Tokenizer
    discretizer: Discretizer
    oor_report: Dict


def _subset(df: pd.DataFrame, split: str, classes: Sequence[str]):
    d = df[df["split"] == split]
    idx = {c: i for i, c in enumerate(classes)}
    return d["path"].tolist(), np.array([idx[c] for c in d["class"]], dtype=np.int64)


def build_bundle(
    df: pd.DataFrame,
    cfg: ExperimentConfig,
    order: str = "canonical",
    order_seed: int = 0,
    n_bins: Optional[int] = None,
    strategy: Optional[str] = None,
    aug_variants: Optional[int] = None,
    verbose: bool = True,
) -> Bundle:
    classes = sorted(df["class"].unique().tolist())
    fcfg = cfg.features
    fcfg.img_size = cfg.spec().img_size
    bcfg = cfg.bins
    if n_bins is not None:
        bcfg.n_bins = n_bins
    if strategy is not None:
        bcfg.strategy = strategy
    n_aug = cfg.train.aug_variants if aug_variants is None else aug_variants

    parts: Dict[str, SplitData] = {}
    names: List[str] = []
    for split in ("train", "val", "test"):
        paths, y = _subset(df, split, classes)
        variants = n_aug if split == "train" else 0   # <-- train only
        X, yy, names, src = extract_many(
            paths, y, fcfg,
            variants=variants,
            cache_dir=CACHE_DIR,
            tag=f"{cfg.dataset}:{split}",
            verbose=verbose,
        )
        parts[split] = SplitData(split, paths, X, yy, src)
        if verbose:
            print(f"[pipeline] {split}: {len(paths)} images -> {X.shape[0]} rows, "
                  f"{X.shape[1]} features")

    # ---- discretizer fitted on TRAIN ONLY ---------------------------------
    disc = Discretizer.fit(parts["train"].X, names, bcfg)
    disc.reset_counters()

    oor: Dict[str, dict] = {}
    for split in ("train", "val", "test"):
        disc.reset_counters()
        B, O = disc.transform(parts[split].X, count_oor=True)
        oor[split] = disc.oor_report()
        toks = token_matrix(B, O, names, bcfg.oor_policy)
        parts[split].tokens = apply_order(toks, order, seed=order_seed)

    max_len = len(names) + 2
    tok = Tokenizer.fit(names, bcfg.n_bins, max_len=max_len, include_oor=True)
    for split in ("train", "val", "test"):
        parts[split].ids = tok.encode_batch(parts[split].tokens)
        rate = tok.unk_rate(parts[split].tokens)
        oor[split]["unk_token_rate"] = round(float(rate), 6)

    if verbose:
        print(f"[pipeline] vocab={len(tok)} tokens, seq_len={max_len}, "
              f"bins={bcfg.n_bins}/{bcfg.strategy}")
        print(f"[pipeline] test out-of-range rate: "
              f"{oor['test']['overall_rate_pct']}% of values "
              f"({oor['test']['mean_oor_features_per_image']} features/image)")

    return Bundle(
        train=parts["train"],
        val=parts["val"],
        test=parts["test"],
        classes=classes,
        feature_names=names,
        tokenizer=tok,
        discretizer=disc,
        oor_report=oor,
    )
