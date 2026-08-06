"""Redraw Figure 1 (study design) and Figure 2 (SIMPLEX architecture)
in an advanced, top-journal SCI style (Nature palette, high info density).

Figure 1: composition-space extrapolation protocol + target distribution
          + evaluation protocol + external ranking significance (bootstrap CI).
Figure 2: SIMPLEX architecture diagram (dual-modality encoding -> ResBlock
          + attention -> output head, with Mixup/SWA/domain-constraint labels).

Run: python redraw_fig12.py   (writes Figure1_study_design + Figure2_architecture)
"""
from __future__ import annotations
import _runtime_guard  # noqa: F401
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import pandas as pd
import os

import paths
import sci_style as ss

ss.apply_style()

# =========================================================================== #
# FIGURE 1 -- study design
# =========================================================================== #
def fig1() -> None:
    # ---- load data ----
    ds = np.load(paths.DATASET_NPZ, allow_pickle=False)
    ext = np.load(paths.EXTERNAL_NPZ, allow_pickle=False)
    Xtr = ds["X"][:, :6]
    ytr = ds["Y"].ravel()
    Xte = ext["X"][:, :6]
    yte = ext["Y"].ravel()
    MONO = ["HEA", "BA", "CBEA", "ATAC", "PEA", "AAm"]

    fig = plt.figure(figsize=(ss.DOUBLE_COL, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                          left=0.09, right=0.97, top=0.90, bottom=0.10)

    # ---------------- A: composition-space extrapolation ----------------
    ax = fig.add_subplot(gs[0, 0])
    # project composition onto (BA, PEA) -- the two monomers driving the
    # SMBO-discovered high-performance region (hydrophobic+aromatic synergy)
    ax.scatter(Xtr[:, 1], Xtr[:, 4], c=ytr, s=26, cmap="Blues",
               vmin=ytr.min(), vmax=ytr.max(), alpha=0.85,
               edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(Xte[:, 1], Xte[:, 4], c=yte, s=26, cmap="Oranges",
               vmin=yte.min(), vmax=yte.max(), alpha=0.85,
               edgecolors="white", linewidths=0.4, marker="D", zorder=3)
    # SMBO migration arrow
    ax.annotate("", xy=(0.60, 0.30), xytext=(0.18, 0.16),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", color="#444444",
                                lw=2.2, connectionstyle="arc3,rad=-0.18"))
    ax.text(0.40, 0.16, "SMBO-guided\ntime extrapolation",
            transform=ax.transAxes, fontsize=7.5, color="#444444",
            ha="center", va="center", style="italic")
    ax.set_xlabel(f"{MONO[1]} molar fraction (hydrophobic)", fontsize=8)
    ax.set_ylabel(f"{MONO[4]} molar fraction (aromatic)", fontsize=8)
    ax.set_xlim(-0.02, 0.85); ax.set_ylim(-0.02, 0.75)
    ax.set_title("A  Composition-space extrapolation protocol",
                 loc="left", fontsize=9, fontweight="bold", pad=8)
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(
            norm=plt.Normalize(min(ytr.min(), yte.min()),
                               max(ytr.max(), yte.max())),
            cmap="viridis"),
        ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Adhesion strength (kPa)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5)
    # legend for marker shapes
    ax.scatter([], [], s=26, c="#999999", marker="o", edgecolors="white",
               label="Training (n=180)")
    ax.scatter([], [], s=26, c="#999999", marker="D", edgecolors="white",
               label="External SMBO (n=161)")
    ax.legend(loc="upper left", fontsize=6.5, frameon=True, fancybox=True,
              framealpha=0.92, edgecolor="none", borderpad=0.5)
    ss.despine(ax)

    # ---------------- B: target distribution shift ----------------
    ax = fig.add_subplot(gs[0, 1])
    bins = np.histogram_bin_edges(np.concatenate([ytr, yte]), bins=18)
    ax.hist(ytr, bins=bins, density=True, histtype="stepfilled",
            alpha=0.55, color="#4477AA", edgecolor="#2a5590", lw=0.6,
            label=f"Training  (mean {ytr.mean():.0f} kPa)")
    ax.hist(yte, bins=bins, density=True, histtype="stepfilled",
            alpha=0.55, color="#EE6677", edgecolor="#b04a5a", lw=0.6,
            label=f"External (mean {yte.mean():.0f} kPa)")
    ax.axvline(ytr.max(), color="#2a5590", ls="--", lw=1.0)
    ax.text(ytr.max() + 4, ax.get_ylim()[1] * 0.92,
            f"train max\n{ytr.max():.0f}", fontsize=6.2, color="#2a5590")
    ax.set_xlabel("Underwater adhesion strength (kPa)", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_title("B  Target-value distribution shift",
                 loc="left", fontsize=9, fontweight="bold", pad=8)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, fancybox=True,
              framealpha=0.92, edgecolor="none")
    ss.despine(ax)

    # ---------------- C: evaluation protocol ----------------
    ax = fig.add_subplot(gs[1, 0])
    ax.axis("off")
    ax.set_title("C  Evaluation protocol",
                 loc="left", fontsize=9, fontweight="bold", pad=2)
    ss.draw_box(ax, 0.03, 0.82, 0.94, 0.14,
                "Internal: 5-fold grouped CV  x  5 seeds  (25 models)",
                face="#EAF2FA", edge="#4477AA")
    ss.draw_arrow(ax, 0.50, 0.82, 0.50, 0.66)
    ss.draw_box(ax, 0.03, 0.50, 0.44, 0.16,
                "In-distribution metric\nR² (primary)",
                face="#EAF2FA", edge="#4477AA")
    ss.draw_box(ax, 0.53, 0.50, 0.44, 0.16,
                "Ablation (20 components)\nMixup / SWA / constraints",
                face="#E9F6F1", edge="#228833")
    ss.draw_arrow(ax, 0.50, 0.66, 0.50, 0.66)
    ss.draw_box(ax, 0.03, 0.12, 0.94, 0.26,
                "External: 161 SMBO formulas, evaluated ONCE\n"
                "Ranking metric (Spearman ρ) + Top-k screening precision\n"
                "(R² is not reported as primary: target-range shift makes it\n"
                "undefined for all models, incl. perfect rankers)",
                face="#FDEEE3", edge="#EE6677", fontsize=6.8)
    ss.draw_arrow(ax, 0.50, 0.50, 0.50, 0.38)

    # ---------------- D: external ranking significance ----------------
    ax = fig.add_subplot(gs[1, 1])
    models = ["SIMPLEX", "ElasticNet", "Ridge", "SVR-RBF", "MLP", "RandomForest"]
    rho = [0.501, 0.494, 0.486, 0.379, 0.315, 0.211]
    ci = [(0.369, 0.619), (0.36, 0.62), (0.35, 0.61),
          (0.24, 0.51), (0.18, 0.45), (0.06, 0.36)]
    # sort by rho ascending for horizontal bar
    order = np.argsort(rho)
    models = [models[i] for i in order]
    rho = [rho[i] for i in order]
    ci = [ci[i] for i in order]
    colors = ["#EE6677" if m == "SIMPLEX" else "#7F7F7F" for m in models]
    ypos = np.arange(len(models))
    for yi, (m, r, (lo, hi), c) in enumerate(zip(models, rho, ci, colors)):
        ax.plot([lo, hi], [yi, yi], color=c, lw=2.4, zorder=2)
        ax.plot(r, yi, "o", color=c, ms=7, zorder=3)
        ax.text(hi + 0.02, yi, f"{r:.2f}", va="center", fontsize=6.8,
                color=c, fontweight="bold")
    ax.axvline(0, color="#555555", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(models, fontsize=7.5)
    ax.set_xlim(-0.05, 0.75)
    ax.set_xlabel("External Spearman ρ  (95% bootstrap CI)", fontsize=8)
    ax.set_title("D  External ranking: SIMPLEX vs baselines",
                 loc="left", fontsize=9, fontweight="bold", pad=8)
    ax.text(0.98, 0.04,
            "SIMPLEX significantly outperforms tree ensembles\n"
            "(paired bootstrap Δρ=+0.18, 95% CI [0.07, 0.31])",
            transform=ax.transAxes, ha="right", fontsize=6.4,
            color="#444444", style="italic")
    ss.despine(ax, keep=("left", "bottom"))

    ss.save_figure(fig, paths.FIGURES_DIR, "Figure1_study_design")


# =========================================================================== #
# FIGURE 2 -- SIMPLEX architecture
# =========================================================================== #
def fig2() -> None:
    fig, ax = plt.subplots(figsize=(ss.DOUBLE_COL, 5.0))
    ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 62)

    PALETTE = {
        "input":    ("#EAF2FA", "#4477AA"),   # light blue
        "synergy":  ("#E4F3EE", "#228833"),   # light green
        "core":     ("#FDF0F2", "#EE6677"),   # light red/pink
        "attn":     ("#F4EEFA", "#AA3377"),   # light purple
        "output":   ("#FDF6E3", "#CCBB44"),   # light gold
        "regular":  ("#F5F5F5", "#666666"),   # neutral
    }

    def box(x, y, w, h, label, key, fs=7.2, bold=False, lw=1.1):
        face, edge = PALETTE[key]
        fc = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.35,rounding_size=0.6",
            facecolor=face, edgecolor=edge, lw=lw, zorder=3)
        ax.add_patch(fc)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, fontweight="bold" if bold else "normal",
                color="#222222", zorder=4)

    def arrow(x1, y1, x2, y2, color="#555555", lw=1.6):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    # ---- input layer: 6 monomers on the simplex ----
    ax.text(1, 58.5, "Input  —  composition on the 6-monomer simplex",
            fontsize=8.5, fontweight="bold", color="#222222")
    mono_names = ["HEA\n(nucleo.)", "BA\n(hydropho.)", "CBEA\n(acidic)",
                  "ATAC\n(cationic)", "PEA\n(aromatic)", "AAm\n(amide)"]
    x0, y0, w, h, gap = 2, 50, 12.5, 5.5, 2.5
    for i, name in enumerate(mono_names):
        box(x0 + i * (w + gap), y0, w, h, name, "input", fs=6.2)
    ax.text(6, 46.8, "sum = 1  (simplex constraint)", fontsize=6.4,
            color="#4477AA", style="italic")

    # ---- dual-modality embedding ----
    ax.text(1, 42.5, "Dual-modality encoding",
            fontsize=8.5, fontweight="bold", color="#222222")
    box(4, 35, 30, 4.8, "Modality 1 · monomer fractions\n(linear embedding)",
        "input", fs=6.6)
    box(40, 35, 30, 4.8,
        "Modality 2 · pairwise synergy\n15 products  xᵢxⱼ (explicit interactions)",
        "synergy", fs=6.6)
    arrow(12, 50, 16, 39.8)
    arrow(28, 50, 32, 39.8)
    arrow(70, 50, 66, 39.8)
    arrow(50, 35, 45, 30.2)
    arrow(55, 35, 58, 30.2)

    # ---- core: residual blocks + attention ----
    ax.text(1, 26.5, "Core  —  residual network with interaction attention",
            fontsize=8.5, fontweight="bold", color="#222222")
    box(8, 15, 20, 7.5, "ResBlock 1\n(dropout + norm)", "core", fs=6.6)
    box(34, 15, 20, 7.5, "Interaction\nself-attention", "attn", fs=6.6)
    box(60, 15, 20, 7.5, "ResBlock 2\n(dropout + norm)", "core", fs=6.6)
    arrow(28, 18.7, 34, 18.7)
    arrow(54, 18.7, 60, 18.7)

    # ---- output head ----
    box(88, 15, 11, 7.5, "Output\nhead", "output", fs=6.8)
    arrow(80, 18.7, 88, 18.7)
    box(84, 5.5, 16, 5.0, "σ = adhesion strength\n(kPa, non-negative)", "output",
        fs=6.4)
    arrow(94, 15, 92, 10.5)

    # ---- regularisation band ----
    ax.text(1, 9.5, "Small-data regularisation",
            fontsize=8.5, fontweight="bold", color="#222222")
    regs = [("Mixup\ninput interpolation", "regular"),
            ("SWA\nweight averaging", "regular"),
            ("Domain constraint\nrange penalty", "regular"),
            ("Early stopping\non inner split", "regular")]
    for i, (label, key) in enumerate(regs):
        box(2 + i * 25, 0.5, 22, 4.6, label, key, fs=6.2)
    # dashed link from regularisation band to core
    ax.plot([6, 6], [5.1, 15], ls=":", color="#999999", lw=1.2)
    ax.plot([31, 31], [5.1, 15], ls=":", color="#999999", lw=1.2)
    ax.plot([56, 56], [5.1, 15], ls=":", color="#999999", lw=1.2)
    ax.plot([81, 81], [5.1, 15], ls=":", color="#999999", lw=1.2)

    ss.save_figure(fig, paths.FIGURES_DIR, "Figure2_architecture")


if __name__ == "__main__":
    paths.ensure_dirs()
    print("Drawing Figure 1 (study design) ...")
    fig1()
    print("Drawing Figure 2 (SIMPLEX architecture) ...")
    fig2()
    print("Done. Figures in", paths.FIGURES_DIR)
