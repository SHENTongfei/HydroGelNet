"""Publication-grade matplotlib styling for high-end SCI figures (Nature/Science/Cell).

Upgraded from okabe-ito-only to a full Nature-style palette with:
- Bold Arial panel letters OUTSIDE axes top-left (no overlap with titles)
- Per-figure oklch perceptual palette + warm accent for ours vs cool for baselines
- Anti-overlap layout helpers (wspace/hspace tuned)
- Chart-type library: lollipop, slope, dumbbell, ridge, heatmap, dot-strip
- 600dpi export PNG + PDF vector
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import os
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

# ---------------------------------------------------------------------------
# PALETTES  (Nature / Science / Cell aesthetic)
# ---------------------------------------------------------------------------
# Okabe-Ito retained as the colourblind-safe base.
OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# Nature-style three-colour system: ours (warm) + baseline (cool) + neutral.
NATURE = {
    "ours":    "#D55E00",   # warm vermillion (Nature uses this for our-method emphasis)
    "ours_d":  "#8C2D04",
    "ours_l":  "#F2C9B5",
    "base":    "#0072B2",   # cool blue (baselines)
    "base_d":  "#003F66",
    "base_l":  "#B4D2E6",
    "neutral": "#6E6E6E",   # grey for context
    "neut_l":  "#D9D9D9",
    "neut_d":  "#3F3F3F",
    "accent":  "#E69F00",   # gold for highlights
    "good":    "#2E7D32",   # positive (gain)
    "bad":     "#B71C1C",   # negative (loss)
}

# Sequential / diverging colormaps (Nature-friendly, not rainbow).
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP  = "RdBu_r"

MODEL_COLOR    = NATURE["ours"]     # SIMPLEX / proposed system
BASELINE_COLOR = NATURE["base"]
NEUTRAL_GREY   = NATURE["neutral"]
LIGHT_GREY     = NATURE["neut_l"]

# Journal column widths in inches.
SINGLE_COL   = 3.5
ONE_HALF_COL = 5.0
DOUBLE_COL   = 7.2


def apply_style() -> None:
    """Apply the global rcParams used by every figure in the project."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
        "grid.color": "#BFBFBF",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    })


# ---------------------------------------------------------------------------
# PANEL LETTERS -- positioned OUTSIDE axes top-left, NEVER overlap titles
# ---------------------------------------------------------------------------
def panel_label(ax: Axes, letter: str, dx: float = -0.22, dy: float = 1.18,
                fontsize: float = 13, weight: str = "bold") -> None:
    """Place a bold Arial panel letter OUTSIDE the axes (top-left margin).

    Defaults are tuned for a 3x3 grid in DOUBLE_COL width: dx=-0.22, dy=1.18.
    Letter sits in figure coords (transform=ax.transAxes so it scales with the axes),
    positioned above and to the left of the axes box - never overlapping tick
    labels, axis labels, or panel titles.
    """
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes,
            fontsize=fontsize, fontweight=weight,
            family="sans-serif", va="bottom", ha="left",
            color="#000000", zorder=10)


def label_panels(axes: Iterable[Axes], start: str = "A", **kwargs) -> None:
    """Label a sequence of axes with consecutive letters (A, B, C, ...)."""
    code = ord(start.upper())
    for i, ax in enumerate(axes):
        panel_label(ax, chr(code + i), **kwargs)


def stars(p_value: float) -> str:
    if p_value is None:
        return "n/a"
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def annotate_significance(ax: Axes, x1: float, x2: float, y: float,
                          text: str, height: float = 0.02,
                          color: str = "black", fontsize: int = 8) -> None:
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = height * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=color)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


def despine(ax: Axes, keep: Sequence[str] = ("left", "bottom")) -> None:
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save_figure(fig: Figure, out_dir: str, name: str,
                formats: Sequence[str] = ("png", "pdf"), dpi: int = 600) -> list:
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fmt in formats:
        path = os.path.join(out_dir, "{0}.{1}".format(name, fmt))
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        written.append(path)
        print("[figure] wrote", path, flush=True)
    plt.close(fig)
    return written


# ---------------------------------------------------------------------------
# ANTI-OVERLAP LAYOUT HELPER
# ---------------------------------------------------------------------------
def tight_layout_with_external_labels(fig: Figure, rect=(0, 0, 1, 0.94),
                                      wspace=0.55, hspace=0.55):
    """Like tight_layout but reserves room for external panel labels."""
    fig.tight_layout(rect=rect)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)


# ---------------------------------------------------------------------------
# ADVANCED CHART-TYPE LIBRARY  (Nature-style, no plain bar spam)
# ---------------------------------------------------------------------------
def lollipop(ax: Axes, categories: Sequence, values: Sequence,
             color: str = NATURE["ours"], neutral: str = NATURE["neutral"],
             value_fmt: str = "{:.2f}", s: float = 40, label_top: bool = True):
    """Lollipop chart - cleaner than bar chart for ranked/ordered categories."""
    y_pos = np.arange(len(categories))
    ax.hlines(y=y_pos, xmin=0, xmax=values, color=color, lw=1.4, alpha=0.85)
    ax.scatter(values, y_pos, s=s, color=color, zorder=3,
               edgecolor="white", linewidth=0.8)
    if label_top:
        for x, y in zip(values, y_pos):
            ax.text(x, y + 0.18, value_fmt.format(x),
                    ha="left", va="bottom", fontsize=7, color=color)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
    ax.set_axisbelow(True)


def slope_chart(ax: Axes, left_vals: Sequence, right_vals: Sequence,
                left_label: str = "Internal", right_label: str = "External",
                labels: Sequence = None,
                highlight: int = 0,
                color_high: str = NATURE["ours"],
                color_rest: str = NATURE["neutral"]):
    """Slope chart (paired left/right) - shows per-model change between two regimes."""
    n = len(left_vals)
    for i in range(n):
        col = color_high if i == highlight else color_rest
        lw = 2.0 if i == highlight else 0.9
        ax.plot([0, 1], [left_vals[i], right_vals[i]],
                "-o", color=col, lw=lw, markersize=5,
                markeredgecolor="white", markeredgewidth=0.6, zorder=3)
    ax.set_xlim(-0.25, 1.25)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([left_label, right_label], fontsize=8)
    if labels:
        for i, lab in enumerate(labels):
            y = (left_vals[i] + right_vals[i]) / 2.0
            ax.text(0.5, y, lab, fontsize=6.5, ha="center", va="center",
                    color=NATURE["neutral"] if i != highlight else NATURE["ours_d"],
                    fontweight="bold" if i == highlight else "normal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25, color="#BFBFBF")
    ax.set_axisbelow(True)


def dumbbell(ax: Axes, left_vals: Sequence, right_vals: Sequence,
             labels: Sequence, highlight: int = 0,
             color_a: str = NATURE["base"], color_b: str = NATURE["ours"]):
    """Dumbbell chart - connects paired values with two endpoints highlighted."""
    n = len(left_vals)
    y_pos = np.arange(n)[::-1]  # top-to-bottom
    for i in range(n):
        col_a = color_a if i != highlight else color_b
        col_b = color_b if i != highlight else color_a
        ax.hlines(y=y_pos[i], xmin=left_vals[i], xmax=right_vals[i],
                  color=NATURE["neutral"], lw=1.0, zorder=2)
        ax.scatter(left_vals[i], y_pos[i], s=55, color=col_a,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.scatter(right_vals[i], y_pos[i], s=55, color=col_b,
                   edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
    ax.set_axisbelow(True)


def ridge_strip(ax: Axes, data: dict, palette=None,
                title: str = "", xlab: str = ""):
    """Ridge-style strip plot - per-group distributions overlaid with mean line."""
    if palette is None:
        palette = OKABE_ITO
    keys = list(data.keys())
    ys = np.arange(len(keys))[::-1]
    for i, k in enumerate(keys):
        v = np.asarray(data[k])
        col = palette[i % len(palette)]
        ax.scatter(v, np.full_like(v, ys[i], dtype=float), s=12,
                   color=col, alpha=0.6, edgecolor="white",
                   linewidth=0.4, zorder=2)
        ax.hlines(ys[i], v.min(), v.max(),
                  color=col, lw=0.6, alpha=0.4, zorder=1)
        ax.scatter(v.mean(), ys[i], s=60, marker="D",
                   color=col, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels(keys, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
    ax.set_axisbelow(True)
    if xlab:
        ax.set_xlabel(xlab)


# ---------------------------------------------------------------------------
# Schematic drawing primitives (used by Figure 1 / Figure 2)
# ---------------------------------------------------------------------------
def draw_box(ax: Axes, x: float, y: float, w: float, h: float, text: str,
             face: str = "#EAF2FA", edge: str = "#0072B2",
             fontsize: int = 7, lw: float = 0.9, radius: float = 0.02,
             text_color: str = "black", bold: bool = False) -> None:
    from matplotlib.patches import FancyBboxPatch
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.004,rounding_size={0}".format(radius),
        linewidth=lw, edgecolor=edge, facecolor=face, mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2.0, y + h / 2.0, text, ha="center", va="center",
            fontsize=fontsize, color=text_color,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def draw_arrow(ax: Axes, x1: float, y1: float, x2: float, y2: float,
               color: str = "#4D4D4D", lw: float = 0.9,
               style: str = "-|>", curve: float = 0.0) -> None:
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8,
        linewidth=lw, color=color,
        connectionstyle="arc3,rad={0}".format(curve),
    )
    ax.add_patch(arrow)


def blank_canvas(ax: Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


if __name__ == "__main__":
    apply_style()
    print("[sci_style v2] style applied; Nature palette:", NATURE)