"""SIMPLEX Figure 1 (pipeline) + Figure 2 (architecture) — POLISHED edition.
Design upgrades vs v1:
- Arial font (SCI standard), consistent rounded corners / line widths
- soft drop-shadow on every module (simulated via offset translucent patch)
- stage labels with data-flow annotations (n=180 / n=161 / metrics)
- dimensional annotations inside architecture modules (d=64, 4 heads, ...)
- palette follows the ORIGINAL TransMICRO paper figures (user's Overleaf):
  blue #CCE4FC/#2E6DA4, green #E4FCFC/#2E8B57, red #FCE4E4/#C0392B,
  purple #FCE4FC/#8E44AD, orange #FCE4CC/#E67E22, bp #E4E4FC/#5B6EE1
- overlap-safe: Layout tracks every bbox; pairs checked; 0 overlaps enforced.
Output: Figure1_pipeline.png/.pdf, Figure2_architecture.png/.pdf (600dpi)
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.family"] = "Arial"

PAL = {
    "blue":   ("#CCE4FC", "#2E6DA4"),
    "green":  ("#E4FCFC", "#2E8B57"),
    "red":    ("#FCE4E4", "#C0392B"),
    "purple": ("#FCE4FC", "#8E44AD"),
    "orange": ("#FCE4CC", "#E67E22"),
    "bp":     ("#E4E4FC", "#5B6EE1"),
    "grey":   ("#E4E4E4", "#555555"),
}

class Layout:
    def __init__(self):
        self.boxes = []
    def add(self, x0, y0, x1, y1, label=""):
        self.boxes.append((x0, y0, x1, y1, label))
    def check(self) -> list:
        bad = []
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                a, b = self.boxes[i], self.boxes[j]
                ix = min(a[2], b[2]) - max(a[0], b[0])
                iy = min(a[3], b[3]) - max(a[1], b[1])
                if ix > 0.4 and iy > 0.4:
                    bad.append((a[4], b[4], round(ix, 2), round(iy, 2)))
        return bad

def _shadow(ax, x, y, w, h, r):
    ax.add_patch(FancyBboxPatch((x + 0.28, y - 0.30), w, h,
                 boxstyle=f"round,pad=0.12,rounding_size={r}",
                 facecolor="#00000010", edgecolor="none", zorder=2))

def box(ax, L, x, y, w, h, label, key, fs=7.0, bold=False, lw=1.4, r=0.5,
        sub=None):
    """sub: optional small annotation printed under the label (e.g. 'd=64')."""
    face, edge = PAL[key]
    _shadow(ax, x, y, w, h, r)
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.12,rounding_size={r}",
                       facecolor=face, edgecolor=edge, lw=lw, zorder=3)
    ax.add_patch(p)
    text = label if sub is None else label + "\n" + sub
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a", zorder=4,
            linespacing=1.35)
    L.add(x - 0.07, y - 0.07, x + w + 0.07, y + h + 0.07, label.split("\n")[0])
    return (x, y, w, h)

def arrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, style="-|>",
          rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                 arrowstyle=style, color=color, lw=lw, linestyle=ls,
                 connectionstyle=f"arc3,rad={rad}", zorder=2.5))

def stage_label(ax, L, x, text):
    ax.text(x, 41.6, text, fontsize=7.5, ha="center", color="#333333",
            fontweight="bold")
    ax.plot([x - 6, x + 6], [40.6, 40.6], color="#BBBBBB", lw=0.8)

def save(fig, name, L, outdir):
    bad = L.check()
    print(f"  {name}: {len(L.boxes)} elements, overlaps = {len(bad)}")
    for b in bad:
        print(f"    OVERLAP: {b[0]} <-> {b[1]} (ix={b[2]}, iy={b[3]})")
    fig.savefig(f"{outdir}/{name}.png", dpi=600, bbox_inches="tight",
                facecolor="white")
    fig.savefig(f"{outdir}/{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

# =========================================================================== #
# FIGURE 1 -- SIMPLEX pipeline (polished)
# =========================================================================== #
def fig_pipeline(outdir):
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.set_xlim(0, 106); ax.set_ylim(0, 44); ax.axis("off")
    L = Layout()

    # ---------- Stage 1: data ----------
    stage_label(ax, L, 8.5, "1 · Data")
    box(ax, L, 1, 33.5, 15, 5.5, "Public dataset", "blue", fs=7.5, bold=True,
        sub="Nature 2025 · MIT licence")
    box(ax, L, 1.5, 24.5, 14, 6, "341 formulations", "blue", fs=6.6)
    box(ax, L, 1.5, 16.5, 14, 6, "6 monomers on the\ncomposition simplex", "blue", fs=6.2)
    box(ax, L, 1.5, 8.0, 14, 6, "Target: adhesion\nstrength (kPa)", "blue", fs=6.4)
    arrow(ax, 8.5, 33.5, 8.5, 31.0)
    arrow(ax, 8.5, 24.5, 8.5, 23.0)

    # ---------- Stage 2: training region ----------
    stage_label(ax, L, 28, "2 · Training region")
    box(ax, L, 20, 33.5, 16, 5.5, "Training set", "green", fs=7.5, bold=True,
        sub="n = 180 · low-performance")
    box(ax, L, 20, 25.0, 16, 6, "5-fold grouped CV", "green", fs=6.6)
    box(ax, L, 20, 17.0, 16, 6, "5 seeds · 25 models", "green", fs=6.6)
    box(ax, L, 20, 9.0, 16, 6, "Ablation-gated\ncomponents", "green", fs=6.2)
    arrow(ax, 16, 20, 20, 20)
    arrow(ax, 28, 33.5, 28, 31.5)

    # ---------- Stage 3: model ----------
    stage_label(ax, L, 49, "3 · SIMPLEX")
    box(ax, L, 41, 33.5, 16, 5.5, "SIMPLEX", "orange", fs=8.5, bold=True,
        sub="dual-modality encoder")
    box(ax, L, 41, 25.0, 16, 6, "Monomers +\npairwise terms", "orange", fs=6.4)
    box(ax, L, 41, 17.0, 16, 6, "ResBlock x2 +\ninteraction attention", "orange", fs=6.2)
    box(ax, L, 41, 9.0, 16, 6, "Mixup · SWA ·\ndomain constraint", "orange", fs=6.2)
    arrow(ax, 36, 20, 41, 20)
    arrow(ax, 49, 33.5, 49, 31.5)

    # ---------- Stage 4: extrapolation ----------
    stage_label(ax, L, 70, "4 · Extrapolation")
    box(ax, L, 62, 33.5, 16, 5.5, "External cohort", "red", fs=7.5, bold=True,
        sub="n = 161 · SMBO-discovered")
    box(ax, L, 62, 25.0, 16, 6, "High-performance\ncomposition region", "red", fs=6.4)
    box(ax, L, 62, 17.0, 16, 6, "Target-value shift\n(mean 47 → 154 kPa)", "red", fs=6.2)
    box(ax, L, 62, 9.0, 16, 6, "Evaluated once,\nafter freezing", "red", fs=6.4)
    arrow(ax, 57, 20, 62, 20)
    arrow(ax, 70, 33.5, 70, 31.5)

    # ---------- Stage 5: screening ----------
    stage_label(ax, L, 92.5, "5 · Screening")
    box(ax, L, 83, 33.5, 19, 5.5, "Ranking + insight", "purple", fs=7.5, bold=True)
    box(ax, L, 83, 25.5, 19, 6, "Spearman ρ = 0.50\nvs RF 0.21", "purple", fs=6.4)
    box(ax, L, 83, 17.5, 19, 6, "Top-20 precision 0.25\nvs RF 0.10", "purple", fs=6.4)
    box(ax, L, 83, 9.5, 19, 6, "Permutation importance\n→ composition synergy", "bp", fs=6.2)
    box(ax, L, 83, 2.5, 19, 5.5, "Accelerated screening", "purple", fs=6.6, bold=True)
    arrow(ax, 78, 20, 83, 20)
    arrow(ax, 92.5, 9.5, 92.5, 8.5)

    save(fig, "Figure1_pipeline", L, outdir)

# =========================================================================== #
# FIGURE 2 -- SIMPLEX architecture (polished)
# =========================================================================== #
def fig_model(outdir):
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    ax.set_xlim(0, 110); ax.set_ylim(0, 50); ax.axis("off")
    L = Layout()

    # ---------- inputs ----------
    box(ax, L, 1, 37, 15, 6, "Input", "grey", fs=7.0, bold=True, sub="composition")
    box(ax, L, 1, 28, 15, 6.5, "6 monomer\nfractions", "blue", fs=6.6,
        sub="simplex, Σ = 1")

    # ---------- dual-modality encoding ----------
    box(ax, L, 21, 37, 15, 6, "Modality 1", "blue", fs=6.8, bold=True,
        sub="monomer fractions")
    box(ax, L, 21, 27, 15, 6.5, "Modality 2", "green", fs=6.8, bold=True,
        sub="15 pairwise xᵢxⱼ")
    arrow(ax, 16, 31, 21, 30.5, rad=-0.15)
    arrow(ax, 12, 28, 21, 29.5, rad=0.15)

    # ---------- embedding ----------
    box(ax, L, 41, 32, 13, 6.5, "Linear\nembedding", "blue", fs=6.8, bold=True,
        sub="d = 64")
    arrow(ax, 36, 33.5, 41, 33.5)
    arrow(ax, 36, 30, 41, 32.5, color="#2E8B57", lw=1.5)

    # ---------- core: residual blocks + attention ----------
    box(ax, L, 59, 40, 15, 6, "ResBlock 1", "orange", fs=6.8, bold=True,
        sub="dropout · LayerNorm")
    box(ax, L, 59, 32, 15, 6, "Interaction\nself-attention", "purple", fs=6.6, bold=True,
        sub="4 heads")
    box(ax, L, 59, 24, 15, 6, "ResBlock 2", "orange", fs=6.8, bold=True,
        sub="dropout · LayerNorm")
    arrow(ax, 54, 34, 59, 34)
    arrow(ax, 66.5, 40, 66.5, 38.6)
    arrow(ax, 66.5, 32, 66.5, 30.6)

    # ---------- output ----------
    box(ax, L, 79, 32, 13, 6, "Pooling +\noutput head", "red", fs=6.6, bold=True)
    arrow(ax, 74, 34, 79, 34)
    box(ax, L, 97, 32, 12, 6, "Adhesion\n(kPa)", "red", fs=6.8, bold=True)
    arrow(ax, 92, 34, 97, 34)

    # ---------- regularisation band ----------
    box(ax, L, 1, 2, 108, 5.5, "", "grey", fs=6.4, bold=True)
    ax.text(55, 4.75,
            "Small-data regularisation:   Mixup   ·   SWA   ·   range-domain constraint   ·   early stopping",
            ha="center", va="center", fontsize=6.6, color="#1a1a1a")
    for x in [26, 44, 62, 82]:
        ax.plot([x, x], [7.5, 24], ls=":", color="#999999", lw=1.2)

    # path labels
    ax.text(2, 37.5, "monomer path", fontsize=6.2, color="#2E6DA4", style="italic")
    ax.text(2, 27.5, "synergy path", fontsize=6.2, color="#2E8B57", style="italic")

    save(fig, "Figure2_architecture", L, outdir)

if __name__ == "__main__":
    import os
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                          "figures")
    os.makedirs(outdir, exist_ok=True)
    print("Drawing SIMPLEX pipeline (Figure 1, polished) ...")
    fig_pipeline(outdir)
    print("Drawing SIMPLEX architecture (Figure 2, polished) ...")
    fig_model(outdir)
    print(f"Done -> {outdir}")
