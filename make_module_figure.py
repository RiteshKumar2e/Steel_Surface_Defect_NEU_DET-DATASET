#!/usr/bin/env python
"""Module-detail figure for the AMFF-CNN manuscript.

Replaces the old CEAM/SEAM block diagram. Every block, symbol and tensor shape
here is transcribed from the layers that `build_model()` in new_model_code.ipynb
actually instantiates, so the figure cannot disagree with the network:

  AMFF  -> class AMFF(layers.Layer)   -- pairwise, 3x, inside the FPN top-down path
  CSAF  -> class CSAF(layers.Layer)   -- once, after the pyramid (replaces CEAM)
  SEAM  -> class SEAM(layers.Layer)   -- once, on the fused map

Outputs figure3_modules.{pdf,png} (vector PDF for LaTeX, 400 dpi PNG for preview).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ---------------------------------------------------------------- config echo
C_FPN = 128                      # CFG.fpn_channels
SEAM_RATES = (1, 3, 5)           # CFG.seam_dilation_rates
CSAF_RED = 8                     # CFG.csaf_reduction
LEVELS = [("P2", 50), ("P3", 25), ("P4", 13), ("P5", 7)]
N_LEV = len(LEVELS)
CSAF_HID = max(C_FPN * N_LEV // CSAF_RED, 16)     # = 64

# ---------------------------------------------------------------- palette
INK      = "#22303F"
EDGE     = "#3A4A5C"
C_TENSOR = "#DCE7F2"   # feature tensors
C_CONV   = "#CFE6D6"   # conv / BN / ReLU
C_ATTN   = "#F7DCC8"   # attention-producing layers
C_NOVEL  = "#E7D6F2"   # the cross-scale softmax -- the contribution
C_OP     = "#FFFFFF"   # operator nodes
C_OUT    = "#FBEFC0"   # module outputs
ARR      = "#3A4A5C"
ARR_ATT  = "#C06A3E"   # attention (gate) signal
ARR_SKIP = "#5B7FA6"   # identity / lateral / skip


def box(ax, x, y, w, h, text, fc, fs=8.4, weight="bold"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.055",
                                facecolor=fc, edgecolor=EDGE, linewidth=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, weight=weight, linespacing=1.45)
    return (x, y, w, h)


def opnode(ax, x, y, sym, r=0.23, fc=C_OP, fs=12):
    ax.add_patch(Circle((x, y), r, facecolor=fc, edgecolor=EDGE, linewidth=1.3, zorder=3))
    ax.text(x, y, sym, ha="center", va="center", fontsize=fs, color=INK,
            weight="bold", zorder=4)


def arrow(ax, p1, p2, color=ARR, ls="-", conn=None, lw=1.25, scale=11):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=scale,
                                 color=color, linewidth=lw, linestyle=ls,
                                 connectionstyle=conn or "arc3,rad=0",
                                 shrinkA=1.5, shrinkB=1.5, zorder=2))


def note(ax, x, y, text, fs=7.3, color="#55606B", ha="center", style="italic"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fs, color=color,
            style=style, linespacing=1.35)


def panel_title(ax, text, sub):
    ax.text(0.02, 0.985, text, transform=ax.transAxes, ha="left", va="top",
            fontsize=11.5, weight="bold", color=INK)
    ax.text(0.02, 0.912, sub, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.2, color="#55606B", style="italic")


# =========================================================== (a) AMFF
def draw_amff(ax):
    ax.set_xlim(0, 16); ax.set_ylim(-0.45, 5.6); ax.axis("off")
    panel_title(ax, "(a)  AMFF - Attention-Modulated Feature Fusion",
                "one merge step of the FPN top-down path; applied 3x "
                "(P5 to P4, P4 to P3, P3 to P2)")

    yc, yh, ys, yl = 4.05, 2.75, 1.30, 0.42

    box(ax, 0.15, yh - 0.45, 1.45, 0.9,
        "$P_{i+1}$\nhigh level\n$H^{\\prime}{\\times}W^{\\prime}{\\times}C$",
        C_TENSOR, 8.0)
    box(ax, 1.95, yh - 0.45, 1.5, 0.9,
        "Resize\n(nearest)\n$\\rightarrow H{\\times}W$", C_CONV, 8.0)
    arrow(ax, (1.60, yh), (1.95, yh))

    # split point on the high-level trunk
    ax.plot([3.45, 3.95], [yh, yh], color=ARR, lw=1.25, zorder=2)
    ax.plot([3.95], [yh], marker="o", ms=4.2, color=ARR, zorder=3)

    # -- channel branch -----------------------------------------------------
    box(ax, 4.25, yc - 0.40, 1.15, 0.8, "GAP\n$1{\\times}1{\\times}C$", C_ATTN, 7.9)
    box(ax, 5.60, yc - 0.40, 1.25, 0.8, "FC $C/8$\nReLU", C_ATTN, 7.9)
    box(ax, 7.05, yc - 0.40, 1.15, 0.8, "FC $C$\n$\\sigma$", C_ATTN, 7.9)
    arrow(ax, (3.95, yh), (4.25, yc), color=ARR, conn="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (5.40, yc), (5.60, yc))
    arrow(ax, (6.85, yc), (7.05, yc))
    opnode(ax, 8.90, yc, "$\\odot$")
    arrow(ax, (8.20, yc), (8.67, yc), color=ARR_ATT)
    note(ax, 6.85, yc + 0.62,
         "$M_c\\in\\mathbb{R}^{1\\times1\\times C}$, broadcast over $H,W$", 7.1)

    # -- spatial branch -----------------------------------------------------
    box(ax, 4.25, ys - 0.40, 1.55, 0.8,
        "[ max ; avg ]\nover channels\n$H{\\times}W{\\times}2$", C_ATTN, 7.6)
    box(ax, 6.00, ys - 0.40, 1.35, 0.8, "Conv $7{\\times}7$\n$\\sigma$", C_ATTN, 7.9)
    arrow(ax, (3.95, yh), (4.25, ys), color=ARR, conn="angle,angleA=0,angleB=90,rad=0")
    arrow(ax, (5.80, ys), (6.00, ys))
    opnode(ax, 8.90, ys, "$\\odot$")
    arrow(ax, (7.35, ys), (8.67, ys), color=ARR_ATT)
    note(ax, 6.85, ys - 0.64,
         "$M_s\\in\\mathbb{R}^{H\\times W\\times1}$, broadcast over $C$", 7.1)

    # The high-level tensor is the other operand of both gates. It travels along
    # the empty middle lane, then splits vertically into the two product nodes,
    # so it never crosses an attention block.
    ax.plot([3.95, 8.90], [yh, yh], color=ARR, lw=1.25, ls=(0, (5, 2)), zorder=2)
    ax.plot([8.90], [yh], marker="o", ms=4.2, color=ARR, zorder=3)
    arrow(ax, (8.90, yh), (8.90, yc - 0.23), color=ARR, ls=(0, (5, 2)))
    arrow(ax, (8.90, yh), (8.90, ys + 0.23), color=ARR, ls=(0, (5, 2)))
    note(ax, 6.30, yh + 0.26, "resized tensor $\\tilde{P}_{i+1}$", 7.1)

    # -- lateral ------------------------------------------------------------
    box(ax, 0.15, yl - 0.35, 1.45, 0.7, "$C_i$\nlateral", C_TENSOR, 8.0)
    box(ax, 1.95, yl - 0.35, 1.5, 0.7,
        "$1{\\times}1$ conv\n$C{=}%d$" % C_FPN, C_CONV, 8.0)
    arrow(ax, (1.60, yl), (1.95, yl))
    arrow(ax, (3.45, yl), (9.62, yl), color=ARR_SKIP)

    # -- concat / fuse ------------------------------------------------------
    box(ax, 9.70, 0.10, 0.95, 4.55, "C\no\nn\nc\na\nt\n\n$3C$", C_TENSOR, 8.0)
    arrow(ax, (9.13, yc), (9.70, yc))
    arrow(ax, (9.13, ys), (9.70, ys))
    arrow(ax, (9.62, yl), (9.70, yl), color=ARR_SKIP)

    box(ax, 11.05, yh - 0.50, 1.75, 1.0,
        "$1{\\times}1$ Conv, $C{=}%d$\nBN $\\rightarrow$ ReLU" % C_FPN, C_CONV, 8.0)
    arrow(ax, (10.65, yh), (11.05, yh))
    box(ax, 13.20, yh - 0.45, 1.35, 0.9, "$P_i$\n$H{\\times}W{\\times}C$", C_OUT, 8.0)
    arrow(ax, (12.80, yh), (13.20, yh))

    note(ax, 14.95, 4.15,
         "ablation knob\nmode in {channel,\nspatial, both}", 7.0, color="#7A6250")
    note(ax, 7.3, -0.28,
         "$P_i=\\mathrm{ReLU}\\left(\\mathrm{BN}\\left(\\mathrm{Conv}_{1\\times1}"
         "\\left[\\,\\tilde{P}_{i+1}\\odot M_c\\,;\\,\\tilde{P}_{i+1}\\odot M_s\\,;\\,"
         "\\mathrm{Conv}_{1\\times1}(C_i)\\,\\right]\\right)\\right)$",
         8.2, color=INK, style="normal")


# =========================================================== (b) CSAF
def draw_csaf(ax):
    ax.set_xlim(0, 16); ax.set_ylim(-0.45, 5.6); ax.axis("off")
    panel_title(ax, "(b)  CSAF - Cross-Scale Adaptive Fusion   (replaces CEAM)",
                "applied once, after the pyramid; the gate of every level is computed "
                "from a descriptor of all levels and normalised across the scale axis")

    ys_ = [3.85, 2.90, 1.95, 1.00]
    y_desc, y_main = 4.15, 1.55

    for (name, sz), y in zip(LEVELS, ys_):
        box(ax, 0.15, y - 0.33, 1.30, 0.66,
            "$%s$\n$%d{\\times}%d{\\times}%d$" % (name, sz, sz, C_FPN), C_TENSOR, 7.7)
        box(ax, 1.75, y - 0.33, 1.45, 0.66,
            "identity" if name == "P2" else "bilinear\n$\\rightarrow 50{\\times}50$",
            C_CONV, 7.5)
        arrow(ax, (1.45, y), (1.75, y))
        arrow(ax, (3.20, y), (3.62, y))

    # Every resized level enters one stack; both downstream paths read from the
    # stack, so no level's forward path has to cross the descriptor path.
    box(ax, 3.62, 0.55, 1.00, 3.75,
        "S\nt\na\nc\nk\n\n$L{\\times}50{\\times}50{\\times}C$", C_TENSOR, 7.6)

    # -- descriptor path (top row) -----------------------------------------
    box(ax, 5.15, y_desc - 0.40, 1.50, 0.80, "GAP over $H,W$\nper level", C_ATTN, 7.8)
    box(ax, 6.90, y_desc - 0.40, 1.25, 0.80, "Concat\n$L\\!\\cdot\\!C$", C_TENSOR, 7.8)
    box(ax, 8.40, y_desc - 0.40, 1.15, 0.80, "FC $%d$\nReLU" % CSAF_HID, C_ATTN, 7.8)
    box(ax, 9.80, y_desc - 0.40, 1.15, 0.80, "FC $L\\!\\cdot\\!C$", C_ATTN, 7.8)
    box(ax, 11.25, y_desc - 0.48, 2.05, 0.96,
        "reshape $(L,C)$\nsoftmax over $L$", C_NOVEL, 8.3)
    arrow(ax, (4.62, 4.05), (5.15, y_desc - 0.12), color=ARR_ATT)
    for x1, x2 in [(6.65, 6.90), (8.15, 8.40), (9.55, 9.80), (10.95, 11.25)]:
        arrow(ax, (x1, y_desc), (x2, y_desc), color=ARR_ATT)
    note(ax, 12.28, y_desc - 0.86,
         "$w\\in\\mathbb{R}^{L\\times C}$,   $\\sum_{l=1}^{L} w_{l,c}=1$\n"
         "per-sample, per-channel, competitive across scales", 7.4, color="#6B4E86")

    # -- main path (bottom row) --------------------------------------------
    opnode(ax, 14.20, y_main, "$\\Sigma$", r=0.29, fs=11)
    arrow(ax, (4.62, y_main), (13.85, y_main), color=ARR_SKIP)
    note(ax, 8.6, y_main + 0.27,
         "the $L$ resized levels, unmodified", 7.2, color="#41648A")
    arrow(ax, (13.30, y_desc), (14.20, y_main + 0.29), color=ARR_ATT,
          conn="angle,angleA=0,angleB=90,rad=0")

    box(ax, 11.75, -0.32, 4.10, 0.82,
        "$3{\\times}3$ Conv, $C{=}%d$ $\\rightarrow$ BN $\\rightarrow$ ReLU"
        "  $\\Rightarrow$  $F$ ($50{\\times}50{\\times}C$)" % C_FPN, C_OUT, 8.0)
    arrow(ax, (14.20, y_main - 0.29), (14.20, 0.50), color=ARR)

    note(ax, 5.7, -0.05,
         "$F=\\mathrm{ReLU}\\left(\\mathrm{BN}\\left(\\mathrm{Conv}_{3\\times3}"
         "\\left(\\sum_{l} w_{l}\\odot\\tilde{P}_{l}\\right)\\right)\\right),"
         "\\quad w=\\mathrm{softmax}_{l}\\left(g\\left(\\left["
         "\\mathrm{GAP}(\\tilde{P}_2);\\ldots;\\mathrm{GAP}(\\tilde{P}_5)"
         "\\right]\\right)\\right)$", 8.2, color=INK, style="normal")


# =========================================================== (c) SEAM
def draw_seam(ax):
    ax.set_xlim(0, 16); ax.set_ylim(-0.40, 4.45); ax.axis("off")
    panel_title(ax, "(c)  SEAM - Spatial Enhancement Attention Module",
                "applied once, on the fused map; parallel dilated depthwise "
                "convolutions produce a single residual spatial gate")

    yb = [2.85, 1.85, 0.85]
    ymid = 1.85
    y_id = 3.45

    box(ax, 0.15, ymid - 0.45, 1.35, 0.9, "$F$\n$H{\\times}W{\\times}C$", C_TENSOR, 8.0)
    ax.plot([1.50, 2.05], [ymid, ymid], color=ARR, lw=1.25)
    ax.plot([2.05], [ymid], marker="o", ms=4.2, color=ARR, zorder=3)

    for r, y in zip(SEAM_RATES, yb):
        box(ax, 2.40, y - 0.38, 2.15, 0.76,
            "DWConv $3{\\times}3$, $r{=}%d$\nBN $\\rightarrow$ ReLU" % r, C_CONV, 7.9)
        arrow(ax, (2.05, ymid), (2.40, y), conn="angle,angleA=0,angleB=90,rad=0")
        arrow(ax, (4.55, y), (5.00, y))

    box(ax, 5.00, 0.42, 0.90, 2.90, "C\no\nn\nc\na\nt\n\n$3C$", C_TENSOR, 7.8)
    box(ax, 6.35, ymid - 0.45, 1.55, 0.9,
        "$1{\\times}1$ Conv, 1 ch\n$\\sigma$", C_ATTN, 8.0)
    arrow(ax, (5.90, ymid), (6.35, ymid))

    box(ax, 8.35, ymid - 0.40, 1.35, 0.8, "$1+A$", C_ATTN, 8.6)
    arrow(ax, (7.90, ymid), (8.35, ymid), color=ARR_ATT)
    note(ax, 8.15, ymid + 0.72, "$A\\in\\mathbb{R}^{H\\times W\\times1}$", 7.2)

    opnode(ax, 10.55, ymid, "$\\odot$")
    arrow(ax, (9.70, ymid), (10.32, ymid), color=ARR_ATT)

    # identity path carrying F into the gate, routed above the branches
    ax.plot([2.05, 2.05], [ymid, y_id], color=ARR_SKIP, lw=1.25, zorder=2)
    ax.plot([2.05, 10.55], [y_id, y_id], color=ARR_SKIP, lw=1.25, zorder=2)
    arrow(ax, (10.55, y_id), (10.55, ymid + 0.23), color=ARR_SKIP)
    note(ax, 6.3, y_id + 0.22, "identity path ($F$)", 7.2, color="#41648A")

    box(ax, 11.30, ymid - 0.45, 1.55, 0.9,
        "$F_{out}$\n$H{\\times}W{\\times}C$", C_OUT, 8.0)
    arrow(ax, (10.78, ymid), (11.30, ymid))

    box(ax, 13.30, ymid - 0.55, 2.45, 1.1,
        "GAP $\\rightarrow$ Dropout 0.5\n$\\rightarrow$ Dense(6, softmax)", C_OUT, 8.0)
    arrow(ax, (12.85, ymid), (13.30, ymid))

    note(ax, 6.6, -0.22,
         "$F_{out}=F\\odot\\left(1+\\sigma\\left(\\mathrm{Conv}_{1\\times1}"
         "\\left[\\,\\phi_{r=1}(F);\\phi_{r=3}(F);\\phi_{r=5}(F)\\,\\right]\\right)\\right)$"
         "$,\\quad \\phi_r=\\mathrm{ReLU}(\\mathrm{BN}(\\mathrm{DWConv}_{3\\times3,r}))$",
         8.2, color=INK, style="normal")


# =========================================================== legend
def draw_legend(ax):
    ax.set_xlim(0, 16); ax.set_ylim(0, 1.0); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.1, 0.08), 15.8, 0.84, boxstyle="round,pad=0.05",
                                facecolor="#FAFBFC", edgecolor="#C6CFD8", linewidth=1.0))
    items = [
        ("$\\odot$", "element-wise product; the gate is\nbroadcast along its singleton axes"),
        ("$\\Sigma$", "weighted sum over\nthe scale axis $l$"),
        ("$\\sigma$", "sigmoid"),
    ]
    xs = [0.72, 4.95, 7.95]
    for (sym, txt), x in zip(items, xs):
        opnode(ax, x, 0.50, sym, r=0.17, fs=9)
        ax.text(x + 0.30, 0.50, txt, ha="left", va="center", fontsize=7.4,
                color=INK, linespacing=1.35)

    ax.text(9.35, 0.50, "[ ; ]  concatenation along\n         the channel axis",
            ha="left", va="center", fontsize=7.4, color=INK, linespacing=1.35)

    arrow(ax, (12.55, 0.66), (13.05, 0.66), color=ARR_ATT)
    ax.text(13.15, 0.66, "attention (gate) signal", fontsize=7.4, va="center", color=INK)
    arrow(ax, (12.55, 0.34), (13.05, 0.34), color=ARR_SKIP)
    ax.text(13.15, 0.34, "identity / lateral path", fontsize=7.4, va="center", color=INK)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="paper_results/figures")
    ap.add_argument("--stem", default="figure3_modules")
    a = ap.parse_args()

    fig = plt.figure(figsize=(15.2, 12.8))
    gs = fig.add_gridspec(4, 1, height_ratios=[6.05, 6.05, 4.85, 0.95],
                          hspace=0.14, left=0.012, right=0.988,
                          top=0.968, bottom=0.010)
    draw_amff(fig.add_subplot(gs[0]))
    draw_csaf(fig.add_subplot(gs[1]))
    draw_seam(fig.add_subplot(gs[2]))
    draw_legend(fig.add_subplot(gs[3]))

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        p = out / f"{a.stem}.{ext}"
        fig.savefig(p, dpi=400, bbox_inches="tight", facecolor="white")
        print(f"saved -> {p}")


if __name__ == "__main__":
    main()
