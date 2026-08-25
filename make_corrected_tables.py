#!/usr/bin/env python
"""Regenerate the backbone-comparison tables from the measured result files.

Reads ONLY paper_results/colab_results/results/*.csv -- every number printed here
is one that run_backbone_comparison.py actually recorded. Nothing is edited by
hand, which is the property the previous version of Table 1 lost.

Emits three tables:

  tab:overall_metrics   accuracy / macro P / R / F1, plus the error count and a
                        Wilson 95% CI, so the reader can see that the top group
                        is separated by one or two test images and is therefore
                        statistically tied.
  tab:accuracy_cost     accuracy against parameters, size, MACs and throughput --
                        the axis on which the proposed model actually leads.
  tab:perclass          per-class F1.

Run:  python make_corrected_tables.py
"""
import math
from pathlib import Path

import pandas as pd

RES = Path("paper_results/colab_results/results")
OUT = Path("paper_results/tables")
ABL = OUT / "table_ablation.csv"
N_TEST = 360          # config.json: n_test
N_CLASSES = 6         # 60 images per class, verified from the confusion matrices
PROPOSED = "Proposed (AMFF-CNN)"

# The proposed model is MobileNetV2 + FPN + AMFF. In the module ladder that is
# the "+ FPN + AMFF" row; CSAF and SEAM are later additions that are NOT part of
# it. The backbone-comparison run scored "full" (FPN+AMFF+CSAF+SEAM) under this
# name, so that row is labelled for what it actually trained until it is re-run.
LADDER_PROPOSED = "+ FPN + AMFF"
BACKBONE_ROW_IS_FULL = True

def collapsed_runs():
    """Models whose run collapsed onto a single class -- every test image given
    the same label. That is a non-converged run, not a measurement of the
    architecture, so it is flagged rather than silently ranked with the rest.

    Detected from the confusion matrices rather than hardcoded by name: the same
    architecture can collapse in one session and train normally in the next, and
    a hardcoded list would then libel a run that actually worked.
    """
    bad = set()
    cm_dir = RES / "confusion_matrices"
    if not cm_dir.is_dir():
        return bad
    for p in cm_dir.glob("*.csv"):
        try:
            cm = pd.read_csv(p, index_col=0)
        except Exception:
            continue
        if (cm.sum(axis=0) > 0).sum() <= 1:      # predictions in one column only
            bad.add(p.stem.replace("_", " "))
    return bad


FAILED = collapsed_runs()


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion -- correct at p near 1,
    where the normal approximation is not."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def esc(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def fmt(v, nd=2):
    return "--" if pd.isna(v) else f"{v:.{nd}f}"


def load():
    comp = pd.read_csv(RES / "backbone_comparison.csv")
    eff = pd.read_csv(RES / "efficiency_results.csv")
    per = pd.read_csv(RES / "per_class_results.csv")
    df = comp.merge(eff.drop(columns=["Init"]), on="Model", how="left")
    df["Errors"] = (round((100 - df["Accuracy"]) / 100 * N_TEST)).astype(int)
    lo, hi = zip(*[wilson(N_TEST - e, N_TEST) for e in df["Errors"]])
    df["CI_lo"], df["CI_hi"] = lo, hi
    return df, per


def order(df):
    """Converged models by accuracy, then failed runs, then the proposed model."""
    ok = df[~df.Model.isin(FAILED) & (df.Model != PROPOSED)]
    ok = ok.sort_values("Accuracy", ascending=False)
    bad = df[df.Model.isin(FAILED)]
    prop = df[df.Model == PROPOSED]
    return pd.concat([ok, bad, prop])


def row_name(m):
    if m == PROPOSED:
        # Name the network that was actually trained. run_backbone_comparison.py
        # called train_one("full"), so this row is FPN+AMFF+CSAF+SEAM, not the
        # proposed MobileNetV2+FPN+AMFF. Re-run with BB_PROPOSED=fpn_amff to
        # replace it, then this label can become "Proposed (AMFF-CNN)".
        return (r"MobileNetV2 + FPN + AMFF + CSAF + SEAM$^{\ddagger}$"
                if BACKBONE_ROW_IS_FULL else r"\textbf{Proposed (AMFF-CNN)}")
    if m in FAILED:
        return esc(m) + r"$^{\dagger}$"
    return esc(m)


def t_overall(df):
    d = order(df)
    best = d[~d.Model.isin(FAILED)]["Accuracy"].max()
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Classification performance on the NEU-DET test split "
         r"($n=360$; 60 images per class) under the shared backbone-comparison "
         r"protocol: identical frozen split, preprocessing, schedule and seed for "
         r"every model. Because the test set is exactly class-balanced, macro "
         r"recall is identical to accuracy by construction. The error column and "
         r"the Wilson 95\% interval are given because the leading models are "
         r"separated by one or two images and are therefore not statistically "
         r"distinguishable on this split.}",
         r"\label{tab:overall_metrics}",
         r"\renewcommand{\arraystretch}{1.12}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Accuracy (\%)} & "
         r"\textbf{Macro Precision (\%)} & \textbf{Macro Recall (\%)} & "
         r"\textbf{Macro F1 (\%)} & \textbf{Errors} & "
         r"\textbf{95\% CI (accuracy)} \\", r"\midrule"]
    for _, r in d.iterrows():
        acc = (r"\textbf{%s}" % fmt(r.Accuracy)
               if r.Accuracy >= best - 1e-9 and r.Model not in FAILED
               else fmt(r.Accuracy))
        L.append(f"{row_name(r.Model)} & {acc} & {fmt(r.MacroPrecision)} & "
                 f"{fmt(r.MacroRecall)} & {fmt(r.MacroF1)} & "
                 f"{int(r.Errors)}/{N_TEST} & "
                 f"[{fmt(r.CI_lo, 1)}, {fmt(r.CI_hi, 1)}] \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", "}",
          r"\smallskip", r"\footnotesize"]
    if FAILED:
        L.append(
            r"$^{\dagger}$ " + ", ".join(sorted(esc(m) for m in FAILED))
            + (" did" if len(FAILED) > 1 else " did")
            + r" not converge under the shared schedule: the run collapsed onto a "
              r"single class, giving every test image the same label. Reported for "
              r"completeness and excluded from the best-value comparison.\\")
    L += [
          r"$^{\ddagger}$ This row was produced by training the full model "
          r"(FPN + AMFF + CSAF + SEAM), not the proposed MobileNetV2 + FPN + AMFF. "
          r"It is named for the network that was trained. Re-running this table "
          r"with the proposed configuration is required before it can carry the "
          r"proposed label; note also that this table uses a 15+15 epoch schedule "
          r"while Table~\ref{tab:module_ladder} uses 30+30, so the two are not "
          r"directly comparable.",
          r"\end{table*}", ""]
    return "\n".join(L)


def t_cost(df):
    d = order(df)
    d = d[~d.Model.isin(FAILED)]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Accuracy against computational cost on the same split. "
         r"Accuracy saturates across the pretrained backbones, so cost is the axis "
         r"on which the configurations actually separate. Latency and throughput "
         r"were measured on the same machine at batch size 1; MACs are unavailable "
         r"for the \texttt{timm} models, which were evaluated through PyTorch.}",
         r"\label{tab:accuracy_cost}",
         r"\renewcommand{\arraystretch}{1.12}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r"\textbf{Model} & \textbf{Accuracy (\%)} & \textbf{Params (M)} & "
         r"\textbf{Size (MiB)} & \textbf{MACs (M)} & \textbf{Latency (ms)} & "
         r"\textbf{FPS} \\", r"\midrule"]
    for _, r in d.iterrows():
        L.append(f"{row_name(r.Model)} & {fmt(r.Accuracy)} & "
                 f"{fmt(r.Params / 1e6)} & {fmt(r['Size (MiB)'])} & "
                 f"{fmt(r['MACs (M)'], 1)} & {fmt(r['Latency (ms)'], 1)} & "
                 f"{fmt(r.FPS, 1)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""]
    return "\n".join(L)


def t_perclass(per, df):
    d = order(df)[["Model"]].merge(per, on="Model", how="left")
    cols = [c for c in per.columns if c.endswith("F1") and c != "Macro F1"]
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Per-class $F_1$ (\%) on the NEU-DET test split.}",
         r"\label{tab:perclass}",
         r"\renewcommand{\arraystretch}{1.12}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "r" * (len(cols) + 1) + "}", r"\toprule",
         r"\textbf{Model} & "
         + " & ".join(r"\textbf{%s}" % esc(c.replace(" F1", "")) for c in cols)
         + r" & \textbf{Macro} \\", r"\midrule"]
    for _, r in d.iterrows():
        vals = " & ".join(fmt(r[c]) for c in cols)
        L.append(f"{row_name(r.Model)} & {vals} & {fmt(r['Macro F1'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}", ""]
    return "\n".join(L)


def t_ladder():
    """Module ladder. Every row here comes from ONE run (30+30 epochs, seed 42,
    same frozen split), so within this table the comparison is like-for-like --
    which is exactly what the backbone table cannot offer, since it was run under
    a different schedule."""
    if not ABL.exists():
        return ""
    a = pd.read_csv(ABL)
    a = a[a.Study == "A: modules"].copy()
    if a.empty:
        return ""
    best = a["Accuracy"].max()
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Module ladder on the NEU-DET test split ($n=360$). Every row "
         r"is trained with the identical frozen split, schedule (30 frozen + 30 "
         r"fine-tuning epochs) and seed, so the rows are directly comparable to "
         r"one another. The proposed configuration, MobileNetV2 + FPN + AMFF, is "
         r"the strongest point of the ladder; adding CSAF and SEAM on top of it "
         r"reduces accuracy. Single seed -- see the note below the table.}",
         r"\label{tab:module_ladder}",
         r"\renewcommand{\arraystretch}{1.12}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"\textbf{Configuration} & \textbf{Accuracy (\%)} & "
         r"\textbf{Macro F1 (\%)} & \textbf{Errors} & \textbf{Params (M)} & "
         r"\textbf{$\Delta$ vs baseline} \\", r"\midrule"]
    base = a.iloc[0]["Accuracy"]
    for _, r in a.iterrows():
        name = esc(r.Variant)
        is_prop = r.Variant.strip() == LADDER_PROPOSED
        if is_prop:
            name = r"\textbf{%s  (proposed)}" % esc(r.Variant)
        acc = fmt(r.Accuracy)
        if r.Accuracy >= best - 1e-9:
            acc = r"\textbf{%s}" % acc
        err = int(round((100 - r.Accuracy) / 100 * N_TEST))
        delta = r.Accuracy - base
        L.append(f"{name} & {acc} & {fmt(r.F1_macro)} & {err}/{N_TEST} & "
                 f"{fmt(r.Params / 1e6)} & {delta:+.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}%", "}",
          r"\smallskip", r"\footnotesize",
          r"These rows are a single seed. On a 360-image test set one image is "
          r"0.28 accuracy points, and runs of this pipeline that differ only in "
          r"schedule length have differed by more than six points, so the "
          r"ordering within this ladder should be confirmed over the five seeds "
          r"declared in \texttt{CFG.model\_seeds} before it is relied upon.",
          r"\end{table*}", ""]
    return "\n".join(L)


def main():
    df, per = load()
    OUT.mkdir(parents=True, exist_ok=True)
    tex = ("% Generated by make_corrected_tables.py from the measured result CSVs.\n"
           "% Requires: \\usepackage{booktabs,graphicx}\n\n"
           + t_ladder() + "\n" + t_overall(df) + "\n" + t_cost(df)
           + "\n" + t_perclass(per, df))
    p = OUT / "corrected_tables.tex"
    p.write_text(tex, encoding="utf-8")
    print(f"saved -> {p}\n")

    # --- consistency audit, printed so the numbers can be checked by eye -----
    print("Consistency audit (balanced test set => macro recall must equal accuracy):")
    bad = df[(df.Accuracy - df.MacroRecall).abs() > 0.01]
    print("  rows violating that identity:", len(bad) if len(bad) else "none")
    top = df[~df.Model.isin(FAILED)].nlargest(6, "Accuracy")
    print("\nTop of the table, by test errors out of 360:")
    for _, r in top.iterrows():
        print(f"  {r.Model:<24} {r.Accuracy:6.2f}%  {int(r.Errors)} error(s)"
              f"  CI [{r.CI_lo:.1f}, {r.CI_hi:.1f}]")
    pr = df[df.Model == PROPOSED].iloc[0]
    print(f"\nProposed: {pr.Accuracy:.2f}% at {pr.Params/1e6:.2f} M params, "
          f"{pr['Size (MiB)']:.2f} MiB")
    for m in ["ResNet50", "DenseNet121", "VGG16"]:
        o = df[df.Model == m].iloc[0]
        print(f"  vs {m:<12} {o.Accuracy:6.2f}% at {o.Params/1e6:6.2f} M "
              f"({o.Params/pr.Params:.1f}x the parameters)")


if __name__ == "__main__":
    main()
