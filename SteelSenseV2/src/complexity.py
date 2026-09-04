"""Parameters, FLOPs and latency, measured the same way for every model.

Reviewer 1 Q5 asks for parameters, FLOPs, latency and Macro-F1 side by side.
For the comparison to mean anything, all four must be measured under one
protocol, so this module is the single place any of them is produced:

  * parameters : trainable tensors only.
  * FLOPs      : multiply-accumulates counted by `thop` for the CNNs. `thop`
                 does not model nn.LSTM, so LSTM cost is counted analytically
                 (see `lstm_flops`) and added. One forward pass, batch size 1.
                 Reported as MACs; multiply by 2 for "FLOPs" if the manuscript
                 uses that convention -- both are printed to avoid the usual
                 factor-of-two ambiguity between papers.
  * latency    : wall clock, batch size 1, single process, after warm-up,
                 median and p95 over `repeats` runs, threads pinned.

The BiLSTM's cost is NOT the whole story for a deployed system, because it
consumes descriptors that must first be computed from the image. That
end-to-end cost is in profile_stages.py, and the two must be quoted together.
"""

from __future__ import annotations

import platform
import statistics
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


def count_params(model: nn.Module) -> Dict[str, float]:
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_ = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"n_params": int(n), "size_mb": round(bytes_ / (1024**2), 3)}


def lstm_flops(input_size: int, hidden: int, layers: int, seq_len: int, bidir: bool = True) -> int:
    """MACs for an nn.LSTM forward pass.

    Per direction, per layer, per timestep the four gates cost
    4 * hidden * (input_size + hidden) MACs; element-wise gate activations are
    ignored, as is conventional.
    """
    dirs = 2 if bidir else 1
    total = 0
    in_sz = input_size
    for _ in range(layers):
        total += dirs * seq_len * 4 * hidden * (in_sz + hidden)
        in_sz = hidden * dirs
    return int(total)


def model_macs(model: nn.Module, example: torch.Tensor, is_token_model: bool = False) -> Dict:
    """MACs for one forward pass at the given input."""
    out: Dict[str, object] = {}
    try:
        from thop import profile as thop_profile

        m = _CountableWrapper(model) if is_token_model else model
        macs, _ = thop_profile(m, inputs=(example,), verbose=False)
        out["macs_thop"] = int(macs)
    except Exception as e:
        out["macs_thop"] = None
        out["thop_error"] = repr(e)

    extra = 0
    for mod in model.modules():
        if isinstance(mod, nn.LSTM):
            seq_len = int(example.shape[1])
            extra += lstm_flops(
                mod.input_size, mod.hidden_size, mod.num_layers, seq_len, mod.bidirectional
            )
    out["macs_lstm_analytic"] = int(extra)
    base = out.get("macs_thop") or 0
    total = int(base + extra)
    out["macs_total"] = total
    out["mflops_2x_macs"] = round(2 * total / 1e6, 3)
    out["mmacs"] = round(total / 1e6, 3)
    return out


class _CountableWrapper(nn.Module):
    """thop cannot trace nn.Embedding lookups from int input; wrap so the
    embedding is skipped and only the downstream Linear layers are counted."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


@torch.no_grad()
def latency(
    fn: Callable[[], object],
    repeats: int = 200,
    warmup: int = 20,
) -> Dict[str, float]:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts_sorted = sorted(ts)
    return {
        "mean_ms": round(statistics.mean(ts), 4),
        "median_ms": round(statistics.median(ts), 4),
        "sd_ms": round(statistics.pstdev(ts), 4),
        "p95_ms": round(ts_sorted[int(0.95 * (len(ts) - 1))], 4),
        "min_ms": round(ts_sorted[0], 4),
        "repeats": repeats,
    }


def torch_model_latency(
    model: nn.Module, example: torch.Tensor, repeats: int = 200, threads: Optional[int] = None
) -> Dict[str, float]:
    prev = torch.get_num_threads()
    if threads:
        torch.set_num_threads(threads)
    model.eval()
    with torch.no_grad():
        out = latency(lambda: model(example), repeats=repeats)
    out["torch_threads"] = torch.get_num_threads()
    torch.set_num_threads(prev)
    return out


def hardware() -> Dict[str, str]:
    """Recorded next to every latency number so the numbers are attributable."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": str(torch.get_num_threads()),
        "cuda_available": str(torch.cuda.is_available()),
    }
    try:
        import psutil

        info["physical_cores"] = str(psutil.cpu_count(logical=False))
        info["logical_cores"] = str(psutil.cpu_count(logical=True))
        info["ram_gb"] = str(round(psutil.virtual_memory().total / 1e9, 1))
    except Exception:
        import os

        info["logical_cores"] = str(os.cpu_count())
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


def profile_model(
    name: str,
    model: nn.Module,
    example: torch.Tensor,
    is_token_model: bool = False,
    repeats: int = 200,
) -> Dict:
    rec = {"model": name, "input_shape": list(example.shape)}
    rec.update(count_params(model))
    rec.update(model_macs(model, example, is_token_model))
    rec["latency_bs1"] = torch_model_latency(model, example, repeats)
    return rec
