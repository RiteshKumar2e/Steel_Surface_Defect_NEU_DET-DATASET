"""Lightweight CNN baselines (Reviewer 1, Q5).

MobileNetV3-Small and ShuffleNetV2-x0.5 are the two the reviewer named; MobileNetV2
and ResNet18 are included as familiar reference points. Every one of them:

  * reads the SAME frozen split manifest -- identical train/val/test images;
  * selects its checkpoint on VALIDATION macro-F1 only;
  * is scored once on the test split;
  * is profiled by complexity.py, so parameters, MACs and batch-size-1 latency
    are measured the same way as for SteelSense-BiLSTM.

ImageNet initialization is used when the weights can be fetched, and the record
says which was used (`pretrained: true|false`), because a from-scratch CNN on
1260 images is a much weaker baseline and the distinction has to be visible.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import cv2

from complexity import profile_model
from metrics import evaluate

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class ImageFolderList(Dataset):
    """Loads from an explicit (path, label) list so the split manifest -- not a
    directory layout -- is the single source of truth."""

    def __init__(self, paths: Sequence[str], labels: np.ndarray, size: int = 224, train: bool = False):
        self.paths = list(paths)
        self.labels = np.asarray(labels)
        self.size = size
        self.train = train

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = cv2.imread(self.paths[i], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train:
            rng = np.random
            if rng.rand() < 0.5:
                img = cv2.flip(img, 1)
            if rng.rand() < 0.5:
                img = cv2.flip(img, 0)
            k = rng.randint(0, 4)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k))
            gamma = float(rng.uniform(0.85, 1.18))
            lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
            img = cv2.LUT(img, lut)
        x = img.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(x.transpose(2, 0, 1)), int(self.labels[i])


def build_backbone(name: str, num_classes: int, pretrained: bool = True) -> Tuple[nn.Module, bool]:
    import torchvision.models as tvm

    def _try(fn, weights_enum):
        if pretrained:
            try:
                return fn(weights=weights_enum.IMAGENET1K_V1), True
            except Exception:
                pass
        return fn(weights=None), False

    if name == "MobileNetV3-Small":
        m, pt = _try(tvm.mobilenet_v3_small, tvm.MobileNet_V3_Small_Weights)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    elif name == "MobileNetV2":
        m, pt = _try(tvm.mobilenet_v2, tvm.MobileNet_V2_Weights)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "ShuffleNetV2-x0.5":
        m, pt = _try(tvm.shufflenet_v2_x0_5, tvm.ShuffleNet_V2_X0_5_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "ShuffleNetV2-x1.0":
        m, pt = _try(tvm.shufflenet_v2_x1_0, tvm.ShuffleNet_V2_X1_0_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "ResNet18":
        m, pt = _try(tvm.resnet18, tvm.ResNet18_Weights)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    else:
        raise ValueError(f"unknown backbone {name}")
    return m, pt


DEFAULT_BACKBONES = ["MobileNetV3-Small", "ShuffleNetV2-x0.5", "MobileNetV2", "ResNet18"]


@torch.no_grad()
def _probs(model: nn.Module, dl: DataLoader) -> np.ndarray:
    model.eval()
    out = []
    for xb, _ in dl:
        out.append(torch.softmax(model(xb), dim=-1).cpu().numpy())
    return np.concatenate(out, 0)


def train_backbone(
    name: str,
    tr_paths: Sequence[str],
    tr_y: np.ndarray,
    va_paths: Sequence[str],
    va_y: np.ndarray,
    te_paths: Sequence[str],
    te_y: np.ndarray,
    classes: Sequence[str],
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 3e-4,
    img_size: int = 224,
    pretrained: bool = True,
    num_workers: int = 0,
    verbose: bool = True,
) -> Tuple[Dict, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, was_pretrained = build_backbone(name, len(classes), pretrained)

    dl_tr = DataLoader(
        ImageFolderList(tr_paths, tr_y, img_size, train=True),
        batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    dl_va = DataLoader(ImageFolderList(va_paths, va_y, img_size),
                       batch_size=64, num_workers=num_workers)
    dl_te = DataLoader(ImageFolderList(te_paths, te_y, img_size),
                       batch_size=64, num_workers=num_workers)

    counts = np.bincount(tr_y, minlength=len(classes)).astype(np.float64)
    w = torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1)), dtype=torch.float32)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=0.05)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_state, best_score, hist = None, -1.0, []
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for xb, yb in dl_tr:
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss.item()) * len(yb)
            n += len(yb)
        sched.step()
        vp = _probs(model, dl_va)
        vm = evaluate(va_y, vp.argmax(1), classes)
        hist.append({"epoch": ep + 1, "train_loss": tot / max(1, n),
                     "val_macro_f1": vm["macro_f1"]})
        if vm["macro_f1"] > best_score:
            best_score = vm["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  [{name}] ep {ep+1:2d}/{epochs} loss {tot/max(1,n):.4f} "
                  f"val_macroF1 {vm['macro_f1']:.4f}")
    train_s = time.perf_counter() - t0

    model.load_state_dict(best_state)          # selected on validation only
    prob = _probs(model, dl_te)                # test touched once
    m = evaluate(te_y, prob.argmax(1), classes, prob)
    m.update(
        {
            "model": name,
            "seed": seed,
            "pretrained": bool(was_pretrained),
            "img_size": img_size,
            "epochs": epochs,
            "val_macro_f1": float(best_score),
            "train_seconds": round(train_s, 1),
            "representation": "raw image pixels",
        }
    )
    m["complexity"] = profile_model(
        name, model, torch.randn(1, 3, img_size, img_size), repeats=50
    )
    if verbose:
        print(f"  [{name}] test acc {m['accuracy']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"params {m['complexity']['n_params']:,} "
              f"MMACs {m['complexity']['mmacs']} "
              f"lat {m['complexity']['latency_bs1']['median_ms']} ms")
    return m, prob
