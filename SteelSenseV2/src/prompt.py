"""Prompt construction and tokenization.

The prompt is a sequence of one token per descriptor, in a fixed canonical
order:

    ss_cls glcm_d1_contrast_mean=b3 glcm_d1_contrast_range=b0 ... cnt_n=b4

Reviewer 1 Q6 asks whether the order matters at all. The answer must be
measured, not asserted, so `order` is a first-class argument:

    canonical   : the fixed order above (the deployed configuration)
    shuffled    : one FIXED random permutation, applied to every sample --
                  tests whether the specific order carries information
    per_sample  : a NEW random permutation per sample -- destroys any order
                  information; a recurrent encoder that still matches its
                  canonical score is not using order

The vocabulary is fitted on the TRAIN split only. Because every numeric feature
is mapped to one of `n_bins` in-vocabulary tokens (plus the declared
out-of-range tokens), a val/test descriptor can never emit <unk>; the <unk>
entry is retained only so the tokenizer stays well defined for foreign input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

PAD, UNK, CLS = "<pad>", "<unk>", "<cls>"


def token_matrix(
    bins: np.ndarray,
    oor: np.ndarray,
    names: Sequence[str],
    oor_policy: str = "clamp",
) -> List[List[str]]:
    """(N, F) bins -> N token lists, canonical order."""
    out: List[List[str]] = []
    nm = list(names)
    for i in range(bins.shape[0]):
        toks = [CLS]
        row = bins[i]
        orow = oor[i]
        for j, n in enumerate(nm):
            if oor_policy == "oor_token" and orow[j] != 0:
                toks.append(f"{n}=oor{'lo' if orow[j] < 0 else 'hi'}")
            else:
                toks.append(f"{n}=b{int(row[j])}")
        out.append(toks)
    return out


def apply_order(
    tokens: List[List[str]],
    order: str,
    seed: int = 0,
    keep_cls: bool = True,
) -> List[List[str]]:
    if order == "canonical":
        return tokens
    rng = np.random.RandomState(seed)
    out = []
    if order == "shuffled":
        n = len(tokens[0]) - (1 if keep_cls else 0)
        perm = rng.permutation(n)
        for t in tokens:
            head = t[:1] if keep_cls else []
            body = t[1:] if keep_cls else t
            out.append(head + [body[k] for k in perm])
        return out
    if order == "per_sample":
        for t in tokens:
            head = t[:1] if keep_cls else []
            body = t[1:] if keep_cls else list(t)
            p = rng.permutation(len(body))
            out.append(head + [body[k] for k in p])
        return out
    raise ValueError(f"unknown order mode: {order}")


class Tokenizer:
    """Whitespace-free tokenizer over a closed symbol set."""

    def __init__(self, vocab: Optional[Dict[str, int]] = None, max_len: int = 160):
        self.vocab = vocab or {PAD: 0, UNK: 1, CLS: 2}
        self.max_len = max_len

    @classmethod
    def fit(
        cls,
        names: Sequence[str],
        n_bins: int,
        max_len: int = 160,
        include_oor: bool = True,
    ) -> "Tokenizer":
        """Fitted from the SCHEMA, not from observed training rows.

        Every (feature, bin) pair is registered whether or not the training
        split happened to produce it, which removes the "unseen symbol at
        inference" failure mode entirely (Reviewer 1, Q8).
        """
        t = cls(max_len=max_len)
        for n in names:
            for b in range(n_bins):
                t.vocab.setdefault(f"{n}=b{b}", len(t.vocab))
            if include_oor:
                t.vocab.setdefault(f"{n}=oorlo", len(t.vocab))
                t.vocab.setdefault(f"{n}=oorhi", len(t.vocab))
        return t

    def encode(self, tokens: Sequence[str]) -> List[int]:
        ids = [self.vocab.get(t, self.vocab[UNK]) for t in tokens][: self.max_len]
        return ids + [0] * (self.max_len - len(ids))

    def encode_batch(self, batch: Sequence[Sequence[str]]) -> np.ndarray:
        return np.asarray([self.encode(t) for t in batch], dtype=np.int64)

    def unk_rate(self, batch: Sequence[Sequence[str]]) -> float:
        tot = unk = 0
        for t in batch:
            for x in t:
                tot += 1
                unk += int(x not in self.vocab)
        return unk / max(1, tot)

    def __len__(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab, "max_len": self.max_len}, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        p = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(p["vocab"], p["max_len"])


def readable_prompt(tokens: Sequence[str], limit: int = 12) -> str:
    """A human-facing rendering used in the paper figure and the README."""
    body = [t for t in tokens if t != CLS][:limit]
    return " ".join(body) + (" ..." if len(tokens) - 1 > limit else "")
