"""Pre-computes the split manifests and the feature cache for both datasets.

Run once; every notebook then loads from cache in seconds. Safe to re-run --
the manifests are deterministic and the feature cache is keyed by config.
"""
import sys, time
sys.path.insert(0, 'src')
from config import ExperimentConfig, SplitConfig, RESULTS_DIR
from splits import build_split
from pipeline import build_bundle

for ds in ("neu_det", "steeldefectx", "steeldefectx_paper"):
    print(f"\n{'='*70}\n{ds}\n{'='*70}", flush=True)
    t = time.perf_counter()
    df = build_split(ds, SplitConfig(), RESULTS_DIR / "splits")
    cfg = ExperimentConfig(dataset=ds)
    b = build_bundle(df, cfg, verbose=True)
    print(f"[{ds}] done in {time.perf_counter()-t:.1f}s  "
          f"train={len(b.train.y)} rows / {len(b.train.paths)} images, "
          f"val={len(b.val.y)}, test={len(b.test.y)}, "
          f"classes={len(b.classes)}, vocab={len(b.tokenizer)}", flush=True)
