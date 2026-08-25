"""Feasibility probe: can the proposed model train on this CPU, and how fast?

Measures per-step time on a few real batches and extrapolates, and separately
checks that the checkpoint path used by train_one() works under Keras 2.15
(the .weights.h5 extension is a Keras 3 convention and may not be supported).
"""
import contextlib
import io
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

print("loading pipeline ...", flush=True)
nb = json.load(open("new_model_code.ipynb", encoding="utf-8"))
NS = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    if src.lstrip().startswith("QUICK"):
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(src, f"cell_{i}", "exec"), NS)

CFG = NS["CFG"]
TRAIN_DF = NS["TRAIN_DF"]
print(f"train images: {len(TRAIN_DF)}  batch: {CFG.batch_size}")

# ---- checkpoint-format check -------------------------------------------------
model, _ = NS["build_model"](**NS["ARCH_VARIANTS"]["fpn_amff"], name="probe")
n_params = model.count_params()
print(f"fpn_amff params: {n_params:,}")

tmp = "_probe.weights.h5"
try:
    model.save_weights(tmp)
    model.load_weights(tmp)
    print("checkpoint format OK (.weights.h5 works under this Keras)")
    os.remove(tmp)
except Exception as e:
    print(f"CHECKPOINT FORMAT FAILS: {type(e).__name__}: {str(e)[:200]}")

# ---- step timing -------------------------------------------------------------
ds = NS["make_dataset"](TRAIN_DF, training=True, seed=42)
steps_per_epoch = int(np.ceil(len(TRAIN_DF) / CFG.batch_size))

n_warm, n_time = 2, 6
times = []
for k, (x, y) in enumerate(ds.take(n_warm + n_time)):
    t0 = time.time()
    model.train_on_batch(x, y)
    dt = time.time() - t0
    if k >= n_warm:
        times.append(dt)
    print(f"  step {k}: {dt:.2f}s" + ("  (warmup)" if k < n_warm else ""), flush=True)

per_step = float(np.median(times))
epoch_s = per_step * steps_per_epoch
print()
print(f"median step : {per_step:.2f}s   steps/epoch: {steps_per_epoch}")
print(f"~epoch      : {epoch_s/60:.1f} min")
for ef, et, tag in [(15, 15, "15+15 (Colab protocol)"), (8, 8, "8+8 (reduced)"), (5, 5, "5+5 (smoke)")]:
    print(f"~1 model {tag:<24}: {epoch_s*(ef+et)/3600:.1f} h")
