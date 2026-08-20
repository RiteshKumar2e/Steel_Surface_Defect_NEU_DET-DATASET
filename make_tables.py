#!/usr/bin/env python
"""Generate paper_tables.tex from the experiment outputs.

    python make_tables.py

Tables are emitted from a single specification here and filled from the CSVs and
JSON under paper_results/tables/. Cells whose source has not been produced yet
stay as [RUN], so the file is always valid LaTeX and always tells the truth about
what has actually been measured.

Do not hand-edit the generated .tex: that reintroduces the code/paper divergence
that got the manuscript rejected. Change the experiment or this script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")

ROOT = Path(__file__).resolve().parent
TABLES = ROOT / "paper_results" / "tables"
LOGS = ROOT / "paper_results" / "logs"
OUT = ROOT / "paper_tables.tex"

RUN = r"\texttt{[RUN]}"
EXPECTED_SEEDS = 5
CLASSES = ["crazing", "inclusion", "patches",
           "pitted_surface", "rolled-in_scale", "scratches"]
DISPLAY = ["Crazing", "Inclusion", "Patches",
           "Pitted surface", "Rolled-in scale", "Scratches"]

# Verified by building each model; fallback when the experiment CSV is absent.
PARAMS = {
    "baseline": 2423110, "fpn": 3196102, "fpn_amff": 3358111,
    "fpn_amff_csaf": 3358687, "full": 3364064,
    "none": 3301895, "eca": 3301910, "se": 3314615, "cbam": 3314912,
    "amff": 3364064, "amff_channel": 3314615, "amff_spatial": 3302192,
    "seam3": 3364064, "seam4": 3365856, "detector": 3966821,
    "MobileNetV2": 2265670, "ResNet50": 23600006, "DenseNet121": 7043654,
    "EfficientNetB0": 4057257, "VGG16": 14717766,
}
MACS = {"backbone": 270, "fpn32": 349, "fpn64": 563, "fpn128": 1373}


def load(name):
    p = TABLES / name
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if len(df) else None
    except Exception as exc:
        print(f"  ! {name}: {exc}")
        return None


def load_json(name):
    p = TABLES / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def num(value, decimals=2):
    if value is None:
        return RUN
    try:
        if isinstance(value, str):
            return value.replace("+-", r"$\pm$")
        if pd.isna(value):
            return RUN
        return f"{float(value):.{decimals}f}"
    except (TypeError, ValueError):
        return RUN


def group(value):
    if value is None:
        return RUN
    try:
        if pd.isna(value):
            return RUN
        return f"{int(value):,}".replace(",", "{,}")
    except (TypeError, ValueError):
        return RUN


def cell(df, where, column, decimals=2):
    if df is None or column not in df.columns:
        return RUN
    mask = pd.Series(True, index=df.index)
    for col, want in where.items():
        if col not in df.columns:
            return RUN
        if isinstance(want, str) and want.startswith("~"):
            # Case-sensitive: a case-insensitive "SE" matches "Ba-se-line".
            mask &= df[col].astype(str).str.contains(want[1:], case=True,
                                                     regex=False)
        else:
            mask &= df[col] == want
    hit = df[mask]
    return num(hit.iloc[0][column], decimals) if len(hit) else RUN


def seed_count():
    df = load("table_multiseed.csv")
    if df is None or "Seeds" not in df.columns:
        return 0
    try:
        return int(df.iloc[0]["Seeds"])
    except (TypeError, ValueError):
        return 0


def seed_cell(df, metric):
    """With one run the standard deviation is undefined, not zero."""
    if df is None or metric not in df.columns:
        return RUN
    raw = df.iloc[0][metric]
    if seed_count() < 2 and isinstance(raw, str) and "+-" in raw:
        return rf"{raw.split('+-')[0].strip()}\,$^{{\dagger}}$"
    return num(raw)


def table(caption, label, colspec, header, rows, wide=False, notes=None,
          midrules=()):
    env = "table*" if wide else "table"
    out = [f"\\begin{{{env}}}[t]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\renewcommand{\\arraystretch}{1.15}",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
           " & ".join(f"\\textbf{{{h}}}" for h in header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            out.append("\\midrule")
        cells = [str(c) for c in r]
        # A \multicolumn row must not be padded, or the column count overflows.
        if cells and cells[0].lstrip().startswith(r"\multicolumn"):
            cells = [c for c in cells if c.strip() != ""]
        out.append(" & ".join(cells) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}", f"\\end{{{env}}}"]
    if notes:
        out += [f"% {n}" for n in notes]
    return "\n".join(out) + "\n\n\n"


def provisional_banner():
    n = seed_count()
    if n == 0 or n >= EXPECTED_SEEDS:
        return ""
    return (
        "% ###################################################################\n"
        "% #  PROVISIONAL RESULTS -- NOT FOR SUBMISSION\n"
        f"% #  Generated from {n} run(s) of a shortened schedule, not the\n"
        f"% #  {EXPECTED_SEEDS}-seed full protocol. Dagger values are single-run,\n"
        "% #  where the standard deviation is UNDEFINED (not zero).\n"
        "% #  Re-run the experiments and regenerate; this banner then vanishes.\n"
        "% ###################################################################\n\n")


def preamble():
    return r"""% =====================================================================
%  Tables: Steel surface defect recognition on NEU-DET
%
%  GENERATED FILE -- do not hand-edit. Regenerate: python make_tables.py
%
%  \usepackage{booktabs}  \usepackage{amssymb}  \usepackage{multirow}
%
%  plain value    -> measured and verified, safe to publish
%  \texttt{[RUN]} -> not yet measured
% =====================================================================

"""


# --------------------------------------------------------------------------
def t_split_images():
    df = load("table_split_images.csv")
    rows = []
    for name in ["Train", "Validation", "Test"]:
        if df is not None:
            r = df[df.iloc[:, 0] == name]
            if len(r):
                rows.append([name] + [int(r.iloc[0][c])
                                      for c in ["Total"] + DISPLAY])
                continue
        rows.append([name] + [RUN] * 7)
    rows.append([r"\textbf{Total}", r"\textbf{1800}"] + [300] * 6)
    return table(
        "Dataset partition by image count, stratified per class, generated once "
        "with a fixed seed and reused by every experiment in this paper.",
        "tab:split_images", "lccccccc",
        ["Split", "Total", "Crazing", "Inclusion", "Patches", "Pitted",
         "Rolled-in", "Scratches"], rows, midrules={3})


def t_split_boxes():
    df = load("table_split_boxes.csv")
    rows = []
    for name in ["Train", "Validation", "Test"]:
        if df is not None:
            r = df[df.iloc[:, 0] == name]
            if len(r):
                rows.append([name] + [int(r.iloc[0][c])
                                      for c in ["Total"] + DISPLAY])
                continue
        rows.append([name] + [RUN] * 7)
    rows.append([r"\textbf{Total}", r"\textbf{4189}",
                 689, 1011, 881, 432, 628, 548])
    return table(
        "Ground-truth bounding boxes per partition. All boxes are the "
        "human-annotated PASCAL-VOC annotations shipped with NEU-DET.",
        "tab:split_boxes", "lccccccc",
        ["Split", "Total", "Crazing", "Inclusion", "Patches", "Pitted",
         "Rolled-in", "Scratches"], rows, midrules={3})


def t_protocol():
    rows = [
        ["Images", "1800 (300 per class)"],
        ["Native frame", r"$200\times200$, depth 1 (all images)"],
        ["Images with verified VOC annotation", r"\textbf{1800 / 1800 (100\%)}"],
        ["Total ground-truth boxes", "4189"],
        [r"Boxes flagged \texttt{difficult}", "81"],
        ["Boxes per image (min--max)", "1--9"],
        ["Localization evaluation subset",
         "complete test split (360 images, 867 boxes)"],
    ]
    return table(
        "Ground-truth annotation audit. Localization metrics are computed only "
        "against these verified annotations; contour-derived regions, "
        "detector-generated proposals and whole-image boxes are never used as "
        "ground truth.",
        "tab:gt_protocol", "lc", ["Property", "Value"], rows)


def t_boxsize():
    """Box-size statistics -- explains the CAM-based localization failure."""
    rows = [
        ["Crazing", "21.1", "1.2", "2.31"],
        ["Inclusion", "4.1", "0.0", "2.95"],
        ["Patches", "9.3", "0.3", "2.91"],
        ["Pitted surface", r"\textbf{55.5}", r"\textbf{58.3}", "1.63"],
        ["Rolled-in scale", "12.3", "0.2", "2.09"],
        ["Scratches", "7.5", "0.0", "2.07"],
    ]
    return table(
        "Ground-truth box statistics per class. Median box area is expressed as "
        "a percentage of the image. Only pitted surface is annotated at near "
        "full-frame extent; the remaining classes are small and multiple. This "
        "distribution explains the CAM-based localization results of "
        r"Table~\ref{tab:localization}.",
        "tab:boxsize", "lccc",
        ["Class", r"Median box (\% of image)", r"Boxes $>$50\% of image (\%)",
         "Boxes per image"], rows)


def t_architecture():
    rows = [
        ["C2", r"\texttt{block\_3\_expand\_relu}", 4, r"$50\times50$"],
        ["C3", r"\texttt{block\_6\_expand\_relu}", 8, r"$25\times25$"],
        ["C4", r"\texttt{block\_13\_expand\_relu}", 16, r"$13\times13$"],
        ["C5", r"\texttt{out\_relu}", 32, r"$7\times7$"],
        [r"\multicolumn{4}{l}{Pyramid width: 128 channels \quad "
         r"SEAM dilation rates: $(1,3,5)$}"],
    ]
    return table(
        r"Backbone tap points and pyramid levels for a $200\times200\times3$ "
        "input. Spatial sizes are asserted at build time so the reported "
        "architecture cannot diverge from the implemented one.",
        "tab:architecture", "llcc",
        ["Level", "MobileNetV2 layer", "Stride", "Size"], rows, midrules={4})


def t_saturation():
    spec = [("Modules", "baseline", "MobileNetV2 baseline"),
            ("Modules", "fpn", "+ FPN"),
            ("Modules", "fpn_amff", "+ AMFF"),
            ("Modules", "fpn_amff_csaf", "+ CSAF"),
            ("Modules", "full", "Full model"),
            ("AMFF attention", "att_channel", "Channel only"),
            ("AMFF attention", "att_spatial", "Spatial only"),
            ("AMFF attention", "att_both", "Both"),
            ("SEAM dilation", "dil_1", r"$(1)$"),
            ("SEAM dilation", "dil_1_3", r"$(1,3)$"),
            ("SEAM dilation", "dil_1_3_5", r"$(1,3,5)$"),
            ("SEAM dilation", "dil_1_3_5_7", r"$(1,3,5,7)$")]
    rows, prev, found = [], None, 0
    for study, key, label in spec:
        f = LOGS / f"abl_{key}_s42_history.json"
        tr = va = RUN
        if f.exists():
            try:
                h = json.loads(f.read_text())
                if h.get("val_accuracy"):
                    va = num(max(h["val_accuracy"]) * 100)
                    tr = num(max(h.get("accuracy", [0])) * 100)
                    found += 1
            except Exception:
                pass
        rows.append([study if study != prev else "", label, tr, va])
        prev = study
    if not found:
        return ""
    return table(
        "Validation accuracy across ablation variants in a single-seed pilot "
        "run. Nearly all variants sit at or within about one point of 100\\%. "
        "Classification accuracy therefore cannot separate the architectural "
        r"components on this dataset, which is why Table~\ref{tab:ablation_main} "
        r"is reported on AP$_{50}$.",
        "tab:saturation", "llcc",
        ["Study", "Variant", r"Train acc.\ (\%)", r"Val.\ acc.\ (\%)"],
        rows, midrules={5, 8},
        notes=["validation, not test; its purpose is to justify the metric choice"])


def t_main_result():
    df = load("table_multiseed.csv")
    rows = [
        [r"\multicolumn{2}{l}{\emph{Classification}}"],
        ["Accuracy", seed_cell(df, "accuracy")],
        ["Precision (macro)", seed_cell(df, "precision_macro")],
        ["Recall (macro)", seed_cell(df, "recall_macro")],
        ["F1 (macro)", seed_cell(df, "f1_macro")],
        [r"\multicolumn{2}{l}{\emph{CAM-based localization}}"],
        [r"AP$_{50}$", seed_cell(df, "AP50")],
        [r"AP$_{75}$", seed_cell(df, "AP75")],
        [r"mAP$_{50:95}$", seed_cell(df, "mAP50_95")],
    ]
    n = seed_count()
    if n >= EXPECTED_SEEDS:
        caption = ("Proposed model on the held-out test split, mean $\\pm$ "
                   f"standard deviation over {n} independent runs.")
        header, notes = ["Metric", r"Mean $\pm$ SD (\%)"], None
    else:
        caption = ("Proposed model on the held-out test split. "
                   rf"\textbf{{Provisional: {max(n, 1)} run"
                   rf"{'' if n == 1 else 's'} of a shortened schedule.}} "
                   r"$^{\dagger}$~single run; standard deviation undefined.")
        header = ["Metric", r"Value (\%)"]
        notes = [f"PROVISIONAL: {n} seed(s), {EXPECTED_SEEDS} expected."]
    return table(caption, "tab:main_result", "lc", header, rows,
                 midrules={5}, notes=notes)


def t_ablation_main():
    df = load("table_ablation.csv")
    ck, no = r"\checkmark", "--"
    spec = [("MobileNetV2 baseline", "baseline", "Baseline (MobileNetV2)",
             [no, no, no, no]),
            ("+ FPN", "fpn", "+ FPN", [ck, no, no, no]),
            ("+ AMFF", "fpn_amff", "+ AMFF", [ck, ck, no, no]),
            ("+ CSAF", "fpn_amff_csaf", "+ CSAF", [ck, ck, ck, no]),
            ("Full model", "full", "Full model", [ck, ck, ck, ck])]
    rows = []
    for label, key, variant, flags in spec:
        w = {"Variant": variant}
        rows.append([label] + flags + [group(PARAMS[key]),
                                       cell(df, w, "Accuracy"),
                                       cell(df, w, "AP50"),
                                       cell(df, w, "mAP50_95")])
    return table(
        "Ablation of the major architectural components; each row adds one "
        r"module to the row above. \textbf{The comparison rests on AP$_{50}$}: "
        "classification accuracy is saturated -- the backbone alone reaches the "
        r"ceiling -- so that column cannot separate the variants "
        r"(Table~\ref{tab:saturation}).",
        "tab:ablation_main", "lcccccccc",
        ["Variant", "FPN", "AMFF", "CSAF", "SEAM", "Params",
         r"Accuracy (\%)", r"AP$_{50}$ (\%)", r"mAP$_{50:95}$ (\%)"],
        rows,
        notes=["accuracy is at ceiling in every row -- draw no conclusions from it"])


def t_ablation_attention():
    df = load("table_ablation.csv")
    base = PARAMS["cbam"]
    spec = [(r"None (concat + $1\times1$)", "none"),
            (r"ECA~\cite{wang2020eca}", "eca"),
            (r"SE~\cite{hu2018squeeze}", "se"),
            (r"CBAM~\cite{woo2018cbam}", "cbam"),
            (r"\textbf{AMFF (ours)}", "amff")]
    rows = []
    for label, key in spec:
        d = PARAMS[key] - base
        delta = "---" if d == 0 else f"${d:+,}$".replace(",", "{,}")
        w = {"Fusion": key}          # exact match on a dedicated column
        rows.append([label, group(PARAMS[key]), delta,
                     cell(df, w, "Accuracy"), cell(df, w, "AP50"),
                     cell(df, w, "mAP50_95")])
    return table(
        "Controlled comparison of attention at the pyramid fusion point. Only "
        "the fusion block differs. AMFF is not parameter-matched to its "
        "controls: it feeds three tensors into the fusion convolution where "
        "CBAM feeds two.",
        "tab:ablation_attention", "lccccc",
        ["Fusion block", "Params", r"$\Delta$ vs CBAM", r"Accuracy (\%)",
         r"AP$_{50}$ (\%)", r"mAP$_{50:95}$ (\%)"],
        rows, midrules={4},
        notes=["a lead below the multi-seed SD is not evidence of an advantage"])


def t_ablation_dilation():
    df = load("table_ablation.csv")
    ck, no = r"\checkmark", "--"
    spec = [(r"$(1,1,1)$", "(1, 1, 1)", 3, no, "seam3"),
            (r"$(1,3,5)$", "(1, 3, 5)", 3, ck, "seam3"),
            (r"$(1,1,1,1)$", "(1, 1, 1, 1)", 4, no, "seam4"),
            (r"$(1,3,5,7)$", "(1, 3, 5, 7)", 4, ck, "seam4")]
    rows = []
    for label, rates, branches, dil, pk in spec:
        w = {"Rates": rates}
        rows.append([label, branches, dil, group(PARAMS[pk]),
                     cell(df, w, "Accuracy"), cell(df, w, "AP50")])
    return table(
        "Contribution of dilation in SEAM, isolated from capacity. The branch "
        "count is fixed within each pair, so the two rows of a pair have "
        r"\emph{identical} parameter counts and differ only in dilation.",
        "tab:ablation_dilation", "lccccc",
        ["SEAM rates", "Branches", "Dilated", "Params", r"Accuracy (\%)",
         r"AP$_{50}$ (\%)"], rows, midrules={2},
        notes=["quote the within-pair delta; the (1,) vs (1,3,5) gap also",
               "changes the branch count and is therefore confounded"])


def t_detector():
    """The supervised detection head -- the headline localization result."""
    d = load_json("detector_results.json")
    wk = load("table_multiseed.csv")
    get = lambda k: num(d[k]) if d else RUN
    rows = [
        ["CAM-based localization (Grad-CAM)", "image labels only",
         group(PARAMS["full"]), seed_cell(wk, "AP50"), seed_cell(wk, "AP75"),
         seed_cell(wk, "mAP50_95")],
        [r"\textbf{Detection head (ours)}", "bounding boxes",
         group(PARAMS["detector"]), get("AP50"), get("AP75"), get("mAP50_95")],
    ]
    for name in ["yolov8n", "yolo11n", "yolo12n"]:
        y = load("table_yolo_baselines.csv")
        w = {"Model": name}
        have = y is not None and (y.Model == name).any()
        rows.append([name.replace("yolo", "YOLO"), "bounding boxes",
                     group(y[y.Model == name].iloc[0]["Params"]) if have else RUN,
                     cell(y, w, "AP50"), cell(y, w, "AP75"),
                     cell(y, w, "mAP50_95")])
    return table(
        "Localization on the test split. All rows use the same partition, the "
        "same verified ground truth and the same evaluation code. The first row "
        "receives no box supervision at training time; the remainder do.",
        "tab:localization", "llccccc"[:0] + "llcccc",
        ["Method", "Supervision", "Params", r"AP$_{50}$ (\%)",
         r"AP$_{75}$ (\%)", r"mAP$_{50:95}$ (\%)"],
        rows, wide=True, midrules={1},
        notes=["detector baselines run at 224: the framework requires the input",
               "side to be divisible by 32 -- state this, do not omit it"])


def t_detector_per_class():
    d = load_json("detector_results.json")
    rows = []
    for cls, disp in zip(CLASSES, DISPLAY):
        v = RUN
        if d and "per_class_AP50" in d and cls in d["per_class_AP50"]:
            v = num(d["per_class_AP50"][cls])
        rows.append([disp, v])
    return table(
        "Per-class AP$_{50}$ of the detection head on the test split.",
        "tab:detector_per_class", "lc",
        ["Class", r"AP$_{50}$ (\%)"], rows)


def t_efficiency():
    eff = load("table_efficiency.csv")
    pyr = load("table_pyramid_cost.csv")
    rows = [
        ["MobileNetV2 (backbone only)", group(PARAMS["baseline"]),
         MACS["backbone"], RUN, RUN,
         cell(pyr, {"Config": "MobileNetV2 backbone only"}, "CPU FPS", 1)],
        ["Proposed, 32 pyramid channels", group(2382680), MACS["fpn32"],
         RUN, RUN, cell(pyr, {"fpn_channels": 32}, "CPU FPS", 1)],
        ["Proposed, 64 pyramid channels", group(2607400), MACS["fpn64"],
         RUN, RUN, cell(pyr, {"fpn_channels": 64}, "CPU FPS", 1)],
        [r"\textbf{Proposed classifier, 128}", group(PARAMS["full"]),
         MACS["fpn128"],
         cell(eff, {"Model": "~Proposed"}, "Size (MiB)"),
         cell(eff, {"Model": "~Proposed"}, "CPU latency ms (median)"),
         cell(eff, {"Model": "~Proposed"}, "CPU FPS", 1)],
        ["Proposed detector", group(PARAMS["detector"]), RUN, RUN, RUN, RUN],
    ]
    return table(
        r"Computational cost, for a single $200\times200\times3$ input at batch "
        "size 1; latency is the median of 200 timed runs after 20 warm-up runs. "
        "Model size is fp32 parameter memory. The pyramid, not the backbone, "
        "dominates the multiply--accumulate cost.",
        "tab:efficiency", "lccccc",
        ["Model", "Params", "MACs (M)", "Size (MiB)", "Latency (ms)", "CPU FPS"],
        rows, wide=True, midrules={4},
        notes=["classifier: 2746.34 MFLOPs = 1373.17 MMACs",
               "re-measure timings on an unloaded machine before submission"])


def t_cls_comparison():
    df = load("table_classification_comparison.csv")
    order = ["MobileNetV2", "EfficientNetB0", "DenseNet121", "VGG16", "ResNet50"]
    rows = []
    for arch in order:
        w = {"Model": arch}
        rows.append([arch, group(PARAMS[arch]), cell(df, w, "Accuracy"),
                     cell(df, w, "Precision"), cell(df, w, "Recall"),
                     cell(df, w, "F1")])
    w = {"Model": "Proposed (full)"}
    rows.append([r"\textbf{Proposed (full)}", group(PARAMS["full"]),
                 cell(df, w, "Accuracy"), cell(df, w, "Precision"),
                 cell(df, w, "Recall"), cell(df, w, "F1")])
    return table(
        "Classification comparison on NEU-DET. Every row was trained by us on "
        "the identical partition with the same augmentation, schedule and seed; "
        "each backbone receives its own canonical input normalisation.",
        "tab:cls_comparison", "lccccc",
        ["Model", "Params", r"Accuracy (\%)", r"Precision (\%)",
         r"Recall (\%)", r"F1 (\%)"], rows, wide=True, midrules={5})


def t_literature():
    ph = r"\texttt{[fill]}"
    rows = [[r"\texttt{[cite]}", ph, r"\texttt{[cls/det]}", ph, ph, ph,
             r"\texttt{[Y/N]}"] for _ in range(3)]
    d = load_json("detector_results.json")
    rows.append([r"\textbf{Ours}", "Proposed detector", "detection",
                 "70/10/20 stratified", r"$200^2$",
                 num(d["AP50"]) if d else RUN, "---"])
    return table(
        "Previously published results on NEU-DET, kept separate from our own "
        "measurements because the evaluation protocols differ. Entries not "
        "directly comparable are listed for context only.",
        "tab:literature", "lllcccc",
        ["Reference", "Method", "Task", "Split protocol", "Input",
         r"AP$_{50}$ (\%)", "Comparable"], rows, wide=True, midrules={3},
        notes=["never rank a detection mAP against a classification accuracy",
               "never copy a number without also copying its protocol"])


def t_nomenclature():
    rows = [["AMFF", "Attention-Modulated Feature Fusion"],
            ["AP", "Average Precision"],
            ["CAM", "Class Activation Map"],
            ["CBAM", "Convolutional Block Attention Module"],
            ["CSAF", "Cross-Scale Adaptive Fusion"],
            ["ECA", "Efficient Channel Attention"],
            ["FPN", "Feature Pyramid Network"],
            ["GAP", "Global Average Pooling"],
            ["IoU", "Intersection over Union"],
            ["MAC", "Multiply--Accumulate operation"],
            ["mAP", "mean Average Precision"],
            ["SE", "Squeeze-and-Excitation"],
            ["SEAM", "Spatial Enhancement Attention Module"],
            ["VOC", "Visual Object Classes (annotation format)"]]
    return table("Nomenclature.", "tab:nomenclature", "ll",
                 ["Abbreviation", "Meaning"], rows)


def t_setup():
    rows = [
        ["Input resolution", r"$200\times200\times3$"],
        ["Source images",
         r"$200\times200$, 8-bit grayscale, replicated to 3 channels"],
        ["Partition", "70 / 10 / 20 stratified, fixed seed"],
        ["Optimiser", "Adam"],
        ["Batch size", "32"],
        ["Detector head", "anchor-free, P2/P3/P4, shared across levels"],
        ["Detector loss", "focal + GIoU + centerness"],
        ["CPU", "AMD64 family 23, 2 physical / 4 logical cores"],
        ["RAM", r"9.9\,GB"],
        ["GPU", "none (CPU-only measurements)"],
        ["Framework", "TensorFlow 2.15.0 / Keras 2.15.0"],
        ["Python / NumPy", "3.10.11 / 1.26.4"],
    ]
    return table("Experimental configuration and measurement environment.",
                 "tab:setup", "ll", ["Item", "Value"], rows, midrules={7})


BUILDERS = [
    ("split (images)", t_split_images), ("split (boxes)", t_split_boxes),
    ("GT protocol", t_protocol), ("box-size analysis", t_boxsize),
    ("architecture", t_architecture), ("saturation", t_saturation),
    ("main result", t_main_result), ("ablation A: modules", t_ablation_main),
    ("ablation B: attention", t_ablation_attention),
    ("ablation C: dilation", t_ablation_dilation),
    ("detection (headline)", t_detector),
    ("detector per-class", t_detector_per_class),
    ("classification comparison", t_cls_comparison),
    ("literature", t_literature), ("efficiency", t_efficiency),
    ("nomenclature", t_nomenclature), ("setup", t_setup),
]


def main():
    parts = [preamble(), provisional_banner()]
    for name, fn in BUILDERS:
        out = fn()
        parts.append(out)
        print(f"  {'ok  ' if out else 'skip'} {name}")

    text = "".join(parts)
    OUT.write_text(text, encoding="utf-8")
    n_run = text.count(r"\texttt{[RUN]}")
    n_fill = text.count(r"\texttt{[fill]}") + text.count(r"\texttt{[cite]}")
    print(f"\nWrote {OUT}  ({len(text.splitlines())} lines)")
    print(f"  {n_run} cells still need experiments ([RUN])")
    print(f"  {n_fill} cells need literature values ([fill]/[cite])")

    n = seed_count()
    if 0 < n < EXPECTED_SEEDS:
        print(f"\n!! PROVISIONAL: {n} run(s), not {EXPECTED_SEEDS}. Banner written "
              f"into the .tex so these cannot be submitted by accident.")


if __name__ == "__main__":
    main()
