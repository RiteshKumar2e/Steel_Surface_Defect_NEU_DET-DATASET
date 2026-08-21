# Running the experiments on a GPU (Colab)

The 15-model backbone comparison takes roughly **16–20 hours** on this laptop
(2 CPU cores, no GPU) — measured, not estimated: ResNet18 alone took 76 minutes.
On a free Colab T4 the same suite is about **45–75 minutes**.

Nothing about the experiment changes. Same frozen split, same preprocessing, same
schedule, same evaluation code. Only the hardware moves.

---

## 1. What to upload

| Item | Why |
|---|---|
| `NEU-DET/` (IMAGES + ANNOTATIONS) | the dataset — 1800 JPGs + 1800 XMLs |
| `paper_results/splits/split_v1.csv` | **the frozen split. Do not regenerate it.** |
| `new_model_code.ipynb` | the pipeline |
| `run_backbone_comparison.py` | the comparison runner |

Zip them on Windows. **Include only IMAGES and ANNOTATIONS** — the
`OUTPUT_SCRATCH_LLM` folder inside `NEU-DET/` is 79 MB of unrelated output and
would nearly quadruple the upload:

```powershell
Compress-Archive -Path NEU-DET\IMAGES, NEU-DET\ANNOTATIONS `
  -DestinationPath neu_data.zip
Compress-Archive -Path new_model_code.ipynb, run_backbone_comparison.py, `
  make_tables.py, paper_results\splits -DestinationPath neu_code.zip
```

Measured sizes:

| Folder | Size | Needed? |
|---|---|---|
| `NEU-DET/IMAGES` | 30 MB | yes |
| `NEU-DET/ANNOTATIONS` | 6.5 MB | yes |
| `NEU-DET/OUTPUT_SCRATCH_LLM` | 79 MB | **no** |

So the upload is ~37 MB, not 115 MB. On Colab, unpack into the same layout:

```
/content/steel/NEU-DET/IMAGES/...
/content/steel/NEU-DET/ANNOTATIONS/...
/content/steel/paper_results/splits/split_v1.csv
/content/steel/new_model_code.ipynb
/content/steel/run_backbone_comparison.py
```

> **The split CSV is the important file.** It records which image belongs to
> train/val/test. If it is missing, the pipeline generates a *new* split and every
> number you already have becomes incomparable. The code now rebuilds file paths
> from `(class, stem)` on whatever machine it runs on, so the same CSV works
> everywhere — and it asserts loudly if any image from the split is absent.

---

## 2. Colab notebook — cell by cell

### Cell 1: GPU check

```python
!nvidia-smi
import tensorflow as tf
print("TF:", tf.__version__, "| GPUs:", tf.config.list_physical_devices('GPU'))
```

If `GPUs: []`, go to **Runtime → Change runtime type → T4 GPU** and rerun.

### Cell 2: Keras 2 compatibility

The code is written against the Keras 2 API. Current Colab ships Keras 3, where
several things behave differently.

```python
!pip install -q tf-keras
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"      # must be set BEFORE importing TF
```

Then **Runtime → Restart session** and continue from Cell 3.

### Cell 3: Upload and unpack

```python
from google.colab import files
up = files.upload()      # pick BOTH neu_data.zip and neu_code.zip
!mkdir -p /content/steel/NEU-DET
!unzip -q neu_data.zip -d /content/steel/NEU-DET
!unzip -q neu_code.zip -d /content/steel
%cd /content/steel
!ls && ls NEU-DET
```

### Cell 4: Sanity-check the split before training anything

```python
import pandas as pd
df = pd.read_csv("paper_results/splits/split_v1.csv")
print(df.groupby("split").size())
print("unique images:", df.stem.nunique(), "| expected 1800")
```

Expected output: `train 1260, val 180, test 360`, 1800 unique images. If this does
not match, stop — the wrong CSV was uploaded.

### Cell 5: Run the comparison

```python
import os
os.environ["BB_EF"] = "15"      # frozen epochs
os.environ["BB_ET"] = "15"      # fine-tune epochs
os.environ["BB_SEED"] = "42"
!python -u run_backbone_comparison.py
```

On GPU you can afford a longer schedule than the 4+4 this laptop was limited to.
**Use the same numbers for every model** — that is what makes the comparison fair.
Whatever you choose is recorded in `results/config.json`.

### Cell 6: Download the results

```python
!zip -qr results.zip results
from google.colab import files
files.download("results.zip")
```

---

## 3. Two things that will bite you

**Colab disconnects after ~90 minutes idle.** The runner writes
`results/_model_<name>.json` after every model and skips completed ones on
restart, so a disconnect costs one model, not the whole run. Just re-run Cell 5.

**The efficiency numbers will be GPU numbers.** Latency and FPS measured on a T4
are not comparable to the CPU numbers already in `paper_tables.tex`. Either:

- report GPU timings for everything and re-measure the proposed model on GPU too, or
- keep accuracy from Colab and re-measure *all* latency/FPS on one CPU afterwards.

Do not mix. A table with some rows timed on T4 and some on a laptop CPU is exactly
the kind of uncontrolled comparison that drew Reviewer 4's objection.

---

## 4. What else is worth running there

Once the environment is up, the expensive items still outstanding all become cheap:

```python
# in the notebook, after Section 13
ABL = run_ablation(seed=42, with_localization=True,
                   studies=("A","B","C","D","E"))   # includes the CSAF novelty test
YOL = run_yolo_baselines()                          # YOLOv8n / 11n / 12n
det, recs = evaluate_detector(train_detector(epochs_frozen=60,
                                             epochs_finetune=40)[0])
```

The detector is the one most worth redoing on GPU. It currently sits at **AP50
64.77** after 40 epochs, and validation loss was still falling when the budget ran
out — 100+ epochs is routine on a GPU and cannot be reached here.

Then locally: `python make_tables.py` to regenerate `paper_tables.tex`.
