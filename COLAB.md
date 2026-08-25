# Running the NEU-DET experiments on Colab

Everything here runs the project's own pipeline on the project's own frozen
split. No number is copied, estimated or hand-entered — each script writes the
CSV it derived its numbers from, so every value in the paper can be traced back
to a file on disk.

## Which steps do I actually need?

**Minimum path to a finished table: 0 → 1 → 2 → 3 → 5 → 6.**

Step 3 measures the numbers but does not build anything; step 5 is what turns
them into `corrected_tables.tex`. Stopping after step 3 leaves you with CSVs and
no table, and stopping before step 6 loses everything when the session ends.

| Step | Required | Time on a T4 |
|---|---|---|
| 0 — GPU runtime | yes | — |
| 1 — upload and unpack | yes | ~2 min |
| 2 — install `timm`, `thop` | yes | ~1 min |
| 3 — backbone comparison | yes | ~40–60 min |
| 4 — multi-seed | no, but it is what makes the ranking defensible | ~2–4 h |
| 5 — rebuild tables and figure | yes | ~10 s |
| 6 — download | yes | ~1 min |

Step 4 can be skipped for now and run later; nothing in step 5 depends on it.

**The one line people miss** is in step 5. `make_corrected_tables.py` reads from
`paper_results/colab_results/results/`, not from the `results/` directory the run
just wrote, so copy first or the tables will be rebuilt from the *old* numbers
and your new run will not appear:

```bash
cp -r results/* paper_results/colab_results/results/
python make_corrected_tables.py
```

## What you need to upload

| File | What it is |
|---|---|
| `neu_colab.zip` | code, the frozen split, and this file |
| `neu_data.zip` | the NEU-DET dataset (`IMAGES/` + `ANNOTATIONS/`) |

Both live in the project root on your machine.

---

## 0. Set the runtime to GPU

**Runtime → Change runtime type → Hardware accelerator: GPU (T4 is fine).**

Do this *before* running anything. On CPU a single model takes hours; on a T4 the
whole backbone comparison is roughly 40–60 minutes.

---

## 1. Upload and unpack

```python
from google.colab import files
up = files.upload()          # pick neu_colab.zip AND neu_data.zip together
```

```bash
%%bash
set -e
rm -rf /content/steel
mkdir -p /content/steel/NEU-DET

unzip -q /content/neu_colab.zip -d /content/steel
unzip -q /content/neu_data.zip  -d /content/steel/NEU-DET

ls /content/steel
echo "--- images: $(find /content/steel/NEU-DET/IMAGES -name '*.jpg' | wc -l) (expect 1800)"
echo "--- annots: $(find /content/steel/NEU-DET/ANNOTATIONS -name '*.xml' | wc -l) (expect 1800)"
echo "--- split : $(wc -l < /content/steel/paper_results/splits/split_v1.csv) (expect 1801)"
```

The layout the scripts expect:

```
/content/steel/
├── new_model_code.ipynb          the pipeline every script imports from
├── run_backbone_comparison.py
├── run_multiseed.py
├── make_corrected_tables.py
├── make_module_figure.py
├── NEU-DET/
│   ├── IMAGES/<class>/*.jpg
│   └── ANNOTATIONS/*.xml
└── paper_results/splits/split_v1.csv
```

The split CSV stores *which image is in which split* — that part is frozen. The
absolute paths inside it were written on a Windows machine and are rebuilt from
`(class, stem)` at load time, so the same CSV is valid here. Never pass
`force=True` to `build_split()`: regenerating the split invalidates every number
already reported.

---

## 2. Install the two extra dependencies

```bash
%%bash
pip -q install timm thop
python -c "import tensorflow as tf; print('TF', tf.__version__, '| GPU', tf.config.list_physical_devices('GPU'))"
```

`timm` supplies RepViT / EdgeNeXt / FastViT, which have no Keras port. `thop`
is used for the MAC counts in the efficiency table. TensorFlow is preinstalled.

---

## 3. The run that matters: proposed = MobileNetV2 + FPN + AMFF

This is the one to do first. The previous backbone table trained
`train_one("full")` — FPN + AMFF + CSAF + SEAM — while labelling the row
*MobileNetV2 + FPN + AMFF*, so the row reported a different network from the one
it named. `BB_PROPOSED` now selects the configuration, defaulting to `fpn_amff`.

```bash
%%bash
cd /content/steel
BB_PROPOSED=fpn_amff BB_EF=15 BB_ET=15 python run_backbone_comparison.py
```

`BB_EF` / `BB_ET` are the frozen and fine-tuning epoch counts. **15 + 15 matches
the schedule the existing baseline numbers were produced under** — keep them
there unless you re-run every model, because mixing schedules is what made the
current tables disagree with each other (the same model scored 99.72 at 15+15
and 93.33 at 30+30).

Useful flags:

```bash
# just a couple of models, to check the plumbing before committing an hour
BB_ONLY=MobileNetV2,ResNet50 python run_backbone_comparison.py

# a different configuration claimed as proposed
BB_PROPOSED=full python run_backbone_comparison.py
```

Output lands in `/content/steel/results/`: `backbone_comparison.csv`,
`efficiency_results.csv`, `per_class_results.csv`, per-model confusion matrices,
and `predictions/` — keep that last one, it is what makes McNemar tests possible
later. The run writes after every model, so an interrupted session resumes
instead of starting over.

---

## 4. Multi-seed — what makes the ranking defensible

Everything in the manuscript is currently one seed on a 360-image test set,
where **one image is 0.28 accuracy points**. Runs that should be near-identical
have differed by far more than that:

```
"full", seed 42, 15+15 epochs  ->  99.72
"full", seed 42, 30+30 epochs  ->  93.33

AMFF channel-only -> 100.00 | spatial-only -> 82.78 | both -> 93.33
```

Those variants differ by a handful of parameters, so a 17-point spread is run
variance, not architecture. `CFG.model_seeds` already declares five seeds
(42, 1337, 2026, 7, 99); only one was ever run.

```bash
%%bash
cd /content/steel
MS_VARIANTS=baseline,fpn,fpn_amff,fpn_amff_csaf,full MS_EF=15 MS_ET=15 python run_multiseed.py
```

Faster version — the three configurations the argument actually turns on:

```bash
%%bash
cd /content/steel
MS_VARIANTS=baseline,fpn_amff,full MS_BACKBONES=MobileNetV2,ResNet50 \
MS_EF=15 MS_ET=15 python run_multiseed.py
```

Writes to `/content/steel/results_multiseed/`:

- `summary_mean_sd.csv` — mean ± sd per configuration
- `paired_ttests.csv` — paired across seeds, which removes the seed effect and
  is far more sensitive than comparing two independent means
- `mcnemar_pooled.csv` — exact binomial McNemar on pooled predictions
- `runs.csv`, `predictions/` — the per-run record

Also resume-safe. Budget roughly `runs × 4–8 min` on a T4.

**This can go either way.** The module ladder currently has `fpn_amff` ahead of
`full`, but on one seed. Five seeds may confirm that, narrow it to nothing, or
reverse it — that is the point of running it.

---

## 5. Rebuild the tables and the figure

```bash
%%bash
cd /content/steel
python make_corrected_tables.py
python make_module_figure.py
```

`make_corrected_tables.py` reads only the measured CSVs and emits
`paper_results/tables/corrected_tables.tex` with four tables: the module ladder,
overall metrics (with an error count and a Wilson 95% interval per row), accuracy
against cost, and per-class F1. It ends with a consistency audit — the test split
is exactly balanced at 60 images per class, so **macro recall must equal accuracy**,
and the audit fails loudly if any row violates that.

By default it reads the Colab results already committed under
`paper_results/colab_results/results/`. To build the tables from a fresh run,
point it at the new directory:

```bash
%%bash
cd /content/steel
cp -r results/* paper_results/colab_results/results/
python make_corrected_tables.py
```

`make_module_figure.py` regenerates the AMFF / CSAF / SEAM figure
(`paper_results/figures/figure3_modules.pdf` + `.png`) from the layer definitions
in the notebook, so the figure cannot drift from the code the way the old
CEAM/SEAM diagram did.

---

## 6. Download the results

```python
import shutil
from google.colab import files
shutil.make_archive('/content/results_out', 'zip', '/content/steel/results')
files.download('/content/results_out.zip')

shutil.make_archive('/content/paper_out', 'zip', '/content/steel/paper_results')
files.download('/content/paper_out.zip')
```

Keep `predictions/` in whatever you download. Without the per-image prediction
vectors, no significance test can be recomputed later and the numbers cannot be
audited.

---

## Notes and gotchas

**Do not mix schedules in one table.** The backbone table and the ablation ladder
were run at 15+15 and 30+30 epochs respectively, which is why plain MobileNetV2
reads 100.00 in one and 98.61 in the other. Pick one schedule, report everything
under it, and state it in the caption.

**Do not run this locally on Windows/CPU and merge the output into the Colab
table.** Different TensorFlow version, different device, different kernels — the
comparison stops being controlled. Run the whole table in one environment.

**ShuffleNetV2 collapses** onto a single class under the shared schedule and
predicts *pitted surface* for every test image (16.67% = 1/6). That is a
non-converged run, not a measurement of the architecture; the table marks it with
a dagger and excludes it from the best-value comparison.

**Session timeouts.** Colab disconnects idle sessions. Both scripts resume from
what they already wrote, so re-running the same command after a drop continues
rather than restarting.
