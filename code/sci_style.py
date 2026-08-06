"""Publication-grade matplotlib styling helpers.

Everything a journal figure needs: colourblind-safe palette, consistent fonts,
panel letters, significance annotations and dual-format export at 600 dpi.
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

# Okabe-Ito colourblind-safe qualitative palette.
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

MODEL_COLOR = "#D55E00"
BASELINE_COLOR = "#0072B2"
NEUTRAL_GREY = "#7F7F7F"
LIGHT_GREY = "#D9D9D9"

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# Journal column widths in inches.
SINGLE_COL = 3.5
ONE_HALF_COL = 5.0
DOUBLE_COL = 7.2


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
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,   # editable text in Illustrator
        "ps.fonttype": 42,
        "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
    })


def panel_label(ax: Axes, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """Place a bold uppercase panel letter at the top-left of an axes."""
    ax.text(dx, dy, letter.upper(), transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left")


def label_panels(axes: Iterable[Axes], start: str = "A", **kwargs) -> None:
    """Label a sequence of axes with consecutive letters."""
    code = ord(start.upper())
    for i, ax in enumerate(axes):
        panel_label(ax, chr(code + i), **kwargs)


def stars(p_value: float) -> str:
    """Convert a p-value into the conventional significance annotation."""
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
    """Draw a significance bracket between two x positions."""
    span = ax.get_ylim()[1] - ax.get_ylim()[0]
    h = height * span
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, c=color)
    ax.text((x1 + x2) / 2.0, y + h, text, ha="center", va="bottom",
            fontsize=fontsize, color=color)


def despine(ax: Axes, keep: Sequence[str] = ("left", "bottom")) -> None:
    """Hide every spine except those listed in ``keep``."""
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save_figure(fig: Figure, out_dir: str, name: str,
                formats: Sequence[str] = ("png", "pdf"), dpi: int = 600) -> list:
    """Save a figure in several formats and return the written paths."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for fmt in formats:
        path = os.path.join(out_dir, "{0}.{1}".format(name, fmt))
        fig.savefig(path, format=fmt, dpi=dpi, bbox_inches="tight")
        written.append(path)
        print("[figure] wrote", path, flush=True)
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #
# Schematic drawing primitives (used by the pipeline / architecture figures)
# --------------------------------------------------------------------------- #
def draw_box(ax: Axes, x: float, y: float, w: float, h: float, text: str,
             face: str = "#EAF2FA", edge: str = "#0072B2",
             fontsize: int = 7, lw: float = 0.9, radius: float = 0.02,
             text_color: str = "black", bold: bool = False) -> None:
    """Draw a rounded schematic box with centred multi-line text."""
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
    """Draw a schematic arrow between two points."""
    from matplotlib.patches import FancyArrowPatch
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=8,
        linewidth=lw, color=color,
        connectionstyle="arc3,rad={0}".format(curve),
    )
    ax.add_patch(arrow)


def blank_canvas(ax: Axes) -> None:
    """Turn an axes into a blank schematic canvas with 0-1 coordinates."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


if __name__ == "__main__":
    apply_style()
    print("[sci_style] style applied; palette:", OKABE_ITO)
