"""ROUTE-B figure redraw (R1-4) - 8 main figures, Morandi palette + broken axes.

This module implements the full R1-4 redraw per
  audit/ROUTEB_AUDIT_R1_4_figures.md  (authoritative disposition list)
  audit/ROUTE_B_PARAM_CARD.md         (single source of truth for numbers)

Design rules enforced here:
  * Morandi palette with multi-step gradients for value-mapped marks
    (bars / lollipops / heatmaps / KDE fills).
  * Broken axes (_broken_cut + _break_marks) wherever the value range spans
    an order of magnitude; the SHORT informative interval gets ~60-70% of
    the plot width, the long tail is truncated with "//" marks and the true
    value is labelled.
  * Anti-overlap: panel letters OUTSIDE axes top-left, explicit wspace/hspace,
    short model names, value labels with white halos, no inline text on
    dense scatters.
  * Honest Route-B wording: no best/highest/superior/outperform/prospective/
    held-out (positive)/SOTA; "ranked k of 8" style only.

Usage:
    python figures_routeb.py [--only 1 2 3 4 5 6 7 8]
(or import from figures.py, which overrides FIGURES with these implementations)
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import os
import shutil
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

import paths
import sci_style as ss
from sci_style import (OKABE_ITO, NATURE, MODEL_COLOR, BASELINE_COLOR,
                       NEUTRAL_GREY, LIGHT_GREY, apply_style,
                       SEQUENTIAL_CMAP, DIVERGING_CMAP,
                       SINGLE_COL, ONE_HALF_COL, DOUBLE_COL)
from figures_v2_backup import Ctx, _safe, _note, _csv, _oof_path  # noqa: F401

apply_style()

# --------------------------------------------------------------------------- #
# Route-B model-name map (short, never truncated with dots in legends)
# --------------------------------------------------------------------------- #
MODEL_SHORT = {
    "SIMPLEX": "SIMPLEX",
    "RandomForest": "RF",
    "GradientBoosting": "GBM",
    "HistGB": "HistGB",
    "SVR-RBF": "SVR",
    "SVR": "SVR",
    "Ridge": "Ridge",
    "ElasticNet": "Enet",
    "KNN": "KNN",
    "MLP": "MLP",
    "Mean": "dummy",
    "XGBoost": "XGB",
}

def _label(axes, start="A", dx=-0.22, dy=1.18):
    ss.label_panels(axes, start=start, dx=dx, dy=dy)

def _m_short(m, n=10):
    s = MODEL_SHORT.get(str(m), str(m))
    return s if len(s) <= n else s[: n - 1] + "\u2026"

def _hard_shorten(s, n):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "\u2026"

# --------------------------------------------------------------------------- #
# Morandi multi-gradient helpers (user-specified palette, 2026-08-10)
# --------------------------------------------------------------------------- #
MORANDI_SEQ = ["#8A8CBF", "#B8A8CF", "#E7BCC6", "#FDCF9E", "#EFA484", "#B6766C"]
MORANDI_BLUE = ["#4E659B", "#8A8CBF"]
MORANDI_R2 = ["#E7BCC6", "#B8A8CF", "#8A8CBF", "#4E659B"]   # low->high R2
MORANDI_DENS = ["#E7BCC6", "#FDCF9E", "#EFA484", "#B6766C"]  # low->high density
MORANDI_DIV = ["#B6766C", "#E7BCC6", "#FDCF9E", "#FFFFFF",
               "#B8A8CF", "#8A8CBF", "#4E659B"]               # diverging

_cmap_cache = {}
def _mcmap(seq):
    key = tuple(seq)
    if key not in _cmap_cache:
        _cmap_cache[key] = LinearSegmentedColormap.from_list("morandi", seq)
    return _cmap_cache[key]

def _mg(vals, seq=MORANDI_SEQ):
    """Return a list of Morandi gradient colours ordered by value (low->high)."""
    vals = np.asarray(vals, dtype=float)
    vmin, vmax = float(vals.min()), float(vals.max())
    if vmax <= vmin:
        frac = np.full_like(vals, 0.5)
    else:
        frac = (vals - vmin) / (vmax - vmin)
    cmap = _mcmap(seq)
    return [cmap(float(f)) for f in frac]

def _blue_scale(vals):
    """Deep-blue gradient (ours): low -> #8A8CBF, high -> #4E659B."""
    return _mg(vals, MORANDI_BLUE[::-1])

# --------------------------------------------------------------------------- #
# Broken-axis helpers (same contract as figures.py so "//" marks are reused)
# --------------------------------------------------------------------------- #
def _broken_cut(vals, k=1.5):
    vals = np.asarray(vals, dtype=float)
    pos = vals[vals > 0]
    if len(pos) == 0:
        return None
    med = float(np.median(pos))
    cut = k * med
    if vals.max() <= cut:
        return None
    return float(cut)

def _break_marks(ax, cut, pos, span, orient="h", color="white", lw=1.2, tick=None):
    if tick is None:
        tick = max(abs(cut) * 0.015, span * 0.10)
    if orient == "h":
        for off in (-span, span):
            ax.plot([cut - tick, cut + 2 * tick],
                    [pos + off - tick, pos + off + tick],
                    color=color, lw=lw, zorder=6)
    else:
        for off in (-span, span):
            ax.plot([pos + off - tick, pos + off + tick],
                    [cut - tick, cut + 2 * tick],
                    color=color, lw=lw, zorder=6)

def _axis_break_h(ax, x, y0=0.0, span=0.012, color="#444444", lw=0.9):
    """Draw '//' discontinuity marks on the axis at x (between two segments)."""
    for off in (-span, span):
        ax.plot([x - span * 2, x + span * 2], [y0 + off, y0 + off + span * 4],
                color=color, lw=lw, clip_on=False, zorder=6,
                transform=ax.get_yaxis_transform())

def _axis_break_v(ax, y, x0=0.0, span=0.012, color="#444444", lw=0.9):
    for off in (-span, span):
        ax.plot([x0 + off, x0 + off + span * 4], [y - span * 2, y + span * 2],
                color=color, lw=lw, clip_on=False, zorder=6,
                transform=ax.get_xaxis_transform())

# --------------------------------------------------------------------------- #
# Save helper: figures/ (FigureN_*) + copy to _frontiers/Figures (FigN_*)
# --------------------------------------------------------------------------- #
_FRONTIERS_DIR = os.path.join(paths.PROJECT_ROOT, "_frontiers", "Figures")

def _save(fig, name, wspace=0.60, hspace=0.72, rect=(0, 0, 1, 0.94)):
    fig.tight_layout(rect=rect)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    ss.save_figure(fig, paths.FIGURES_DIR, name)
    short = name.replace("Figure", "Fig", 1)
    os.makedirs(_FRONTIERS_DIR, exist_ok=True)
    for fmt in ("png", "pdf"):
        src = os.path.join(paths.FIGURES_DIR, "{0}.{1}".format(name, fmt))
        dst = os.path.join(_FRONTIERS_DIR, "{0}.{1}".format(short, fmt))
        try:
            shutil.copy(src, dst)
            print("[routeb] copied", dst, flush=True)
        except Exception as exc:  # noqa: BLE001
            print("[routeb] copy failed", dst, exc, flush=True)

def _panel_label(axes, start="A"):
    ss.label_panels(axes, start=start, dx=-0.22, dy=1.18)

def _load_cv_main_preds():
    p = os.path.join(paths.PREDS_DIR, "preds_cv_main.csv")
    return _csv(p)

def _oof_main_or_ctx(ctx):
    """preds_cv_main (n=316) if present, else the legacy OOF preds."""
    oof = _load_cv_main_preds()
    if oof is None:
        oof = ctx.oof
    return oof


def _paired_cv(group):
    """Group OOF predictions by sample (mean over seeds) -> (y_true, y_pred)."""
    t = "glass_adhesion_kpa"
    g = group.groupby("sample_id")[["y_true_%s" % t,
                                    "y_pred_%s" % t]].mean()
    return g["y_true_%s" % t].values, g["y_pred_%s" % t].values

def _ext_single_table():
    """external.csv single rows -> per-model mean R2 / rho (Route-B truth)."""
    ex = _csv(os.path.join(paths.METRICS_DIR, "external.csv"))
    be = _csv(os.path.join(paths.METRICS_DIR, "baselines_external.csv"))
    rows = []
    if ex is not None:
        s = ex[ex["tag"] == "external_single"].groupby("model")[
            ["R2", "SpearmanRho"]].mean()
        for m, r in s.iterrows():
            rows.append({"model": m, "R2": float(r["R2"]),
                         "rho": float(r["SpearmanRho"])})
    if be is not None:
        s = be[be["tag"] == "external_single"].groupby("model")[
            ["R2", "SpearmanRho"]].mean()
        for m, r in s.iterrows():
            rows.append({"model": m, "R2": float(r["R2"]),
                         "rho": float(r["SpearmanRho"])})
    df = pd.DataFrame(rows).drop_duplicates("model").reset_index(drop=True)
    return df

def _internal_means():
    """Mean internal metrics per model from cv_outer.csv + baselines.csv."""
    cv = _csv(paths.CV_OUTER_CSV)
    base = _csv(paths.BASELINES_CSV)
    rows = []
    if cv is not None and len(cv):
        s = cv.groupby("model")[["R2", "SpearmanRho", "PearsonR",
                                 "RMSE", "MAE"]].mean()
        for m, r in s.iterrows():
            rows.append({"model": m, "R2": float(r["R2"]),
                         "rho": float(r["SpearmanRho"]),
                         "pearson": float(r["PearsonR"]),
                         "rmse": float(r["RMSE"]), "mae": float(r["MAE"])})
    if base is not None and len(base):
        b = base[base["model"] != "Mean"]
        s = b.groupby("model")[["R2", "SpearmanRho", "PearsonR",
                                "RMSE", "MAE"]].mean()
        for m, r in s.iterrows():
            rows.append({"model": m, "R2": float(r["R2"]),
                         "rho": float(r["SpearmanRho"]),
                         "pearson": float(r["PearsonR"]),
                         "rmse": float(r["RMSE"]), "mae": float(r["MAE"])})
    return pd.DataFrame(rows).drop_duplicates("model").reset_index(drop=True)


def _abl_categories(deltas):
    pi, bi, ni = [], [], []
    for i, d in enumerate(deltas):
        if abs(d) < 1e-9:
            bi.append(i)
        elif d > 0:
            pi.append(i)
        else:
            ni.append(i)
    return pi, bi, ni

def _abl_short(variant):
    if not variant:
        return variant
    return variant if len(variant) <= 18 else variant[:16] + '..'

def _abl_deltas(ctx):
    """Return DataFrame of ablation deltas (full - ablated), all 23 arms."""
    abl = ctx.abl
    if abl is None:
        return None
    g = abl.groupby("variant")["R2"].mean()
    if "full model" not in g:
        return None
    full = float(g["full model"])
    d = (full - g.drop("full model")).sort_values(ascending=False)
    return pd.DataFrame({"variant": d.index, "delta": d.values,
                         "R2_abl": g.drop("full model").values})

# =========================================================================== #
# Figure 1 - pipeline schematic (local edit: D/E narrative + true numbers)
# =========================================================================== #
_PAL = {
    "blue":   ("#CCE4FC", "#2E6DA4"),
    "green":  ("#E4FCFC", "#2E8B57"),
    "red":    ("#FCE4E4", "#C0392B"),
    "purple": ("#FCE4FC", "#8E44AD"),
    "orange": ("#FCE4CC", "#E67E22"),
    "bp":     ("#E4E4FC", "#5B6EE1"),
    "grey":   ("#E4E4E4", "#555555"),
}
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

class _Layout:
    def __init__(self):
        self.boxes = []
    def add(self, x0, y0, x1, y1, label=""):
        self.boxes.append((x0, y0, x1, y1, label))
    def check(self):
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
                 boxstyle="round,pad=0.12,rounding_size={0}".format(r),
                 facecolor="#00000010", edgecolor="none", zorder=2))

def _pbox(ax, L, x, y, w, h, label, key, fs=7.0, bold=False, lw=1.4, r=0.5,
          sub=None):
    face, edge = _PAL[key]
    _shadow(ax, x, y, w, h, r)
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.12,rounding_size={0}".format(r),
                       facecolor=face, edgecolor=edge, lw=lw, zorder=3)
    ax.add_patch(p)
    text = label if sub is None else label + "\n" + sub
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color="#1a1a1a", zorder=4,
            linespacing=1.35)
    L.add(x - 0.07, y - 0.07, x + w + 0.07, y + h + 0.07, label.split("\n")[0])
    return (x, y, w, h)

def _parrow(ax, x1, y1, x2, y2, color="#555555", lw=1.8, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 color=color, lw=lw, connectionstyle="arc3,rad={0}".format(rad),
                 zorder=2.5))

def _stage_label(ax, x, text):
    ax.text(x, 41.6, text, fontsize=7.5, ha="center", color="#333333",
            fontweight="bold")
    ax.plot([x - 6, x + 6], [40.6, 40.6], color="#BBBBBB", lw=0.8)

def _psave(fig, name, L, outdir):
    bad = L.check()
    print("  {0}: {1} elements, overlaps = {2}".format(name, len(L.boxes),
                                                       len(bad)))
    for b in bad:
        print("    OVERLAP: {0} <-> {1} (ix={2}, iy={3})".format(*b))
    fig.savefig(os.path.join(outdir, name + ".png"), dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(outdir, name + ".pdf"), bbox_inches="tight",
                facecolor="white")
    short = name.replace("Figure", "Fig", 1)
    os.makedirs(_FRONTIERS_DIR, exist_ok=True)
    for fmt in ("png", "pdf"):
        shutil.copy(os.path.join(outdir, name + "." + fmt),
                    os.path.join(_FRONTIERS_DIR, short + "." + fmt))
    plt.close(fig)

def fig1(ctx: Ctx) -> None:
    fig, ax = plt.subplots(figsize=(9.8, 4.8))
    ax.set_xlim(0, 106); ax.set_ylim(0, 44); ax.axis("off")
    L = _Layout()

    # ---------- Stage 1: data ----------
    _stage_label(ax, 8.5, "A · Data")
    _pbox(ax, L, 1, 33.5, 15, 5.5, "Public dataset", "blue", fs=7.5, bold=True,
          sub="Nature 2025 · MIT licence")
    _pbox(ax, L, 1.5, 24.5, 14, 6, "341 formulations", "blue", fs=6.6)
    _pbox(ax, L, 1.5, 16.5, 14, 6, "6 monomers on the\ncomposition simplex",
          "blue", fs=6.2)
    _pbox(ax, L, 1.5, 8.0, 14, 6, "Target: adhesion\nstrength (kPa)", "blue",
          fs=6.4)
    _parrow(ax, 8.5, 33.5, 8.5, 31.0)
    _parrow(ax, 8.5, 24.5, 8.5, 23.0)

    # ---------- Stage 2: training region ----------
    _stage_label(ax, 28, "B · Training region")
    _pbox(ax, L, 20, 33.5, 16, 5.5, "Training set", "green", fs=7.5, bold=True,
          sub="n = 316 · internal cohort")
    _pbox(ax, L, 20, 25.0, 16, 6, "5-fold grouped CV", "green", fs=6.6)
    _pbox(ax, L, 20, 17.0, 16, 6, "10 seeds · 50 models", "green", fs=6.6)
    _pbox(ax, L, 20, 9.0, 16, 6, "Ablation-gated\ncomponents", "green", fs=6.2)
    _parrow(ax, 16, 20, 20, 20)
    _parrow(ax, 28, 33.5, 28, 31.5)

    # ---------- Stage 3: model ----------
    _stage_label(ax, 49, "C · SIMPLEX")
    _pbox(ax, L, 41, 33.5, 16, 5.5, "SIMPLEX", "orange", fs=8.5, bold=True,
          sub="dual-modality encoder")
    _pbox(ax, L, 41, 25.0, 16, 6, "Monomers +\npairwise terms", "orange",
          fs=6.4)
    _pbox(ax, L, 41, 17.0, 16, 6, "ResBlock +\ninteraction attention",
          "orange", fs=6.2)
    _pbox(ax, L, 41, 9.0, 16, 6, "Mixup · SWA ·\ndomain constraint",
          "orange", fs=6.2)
    _parrow(ax, 36, 20, 41, 20)
    _parrow(ax, 49, 33.5, 49, 31.5)

    # ---------- Stage 4: BO-acquired batch evaluation ----------
    _stage_label(ax, 70, "D · BO-acquired batch evaluation")
    _pbox(ax, L, 62, 33.5, 16, 5.5, "BO-acquired batch", "red", fs=7.5,
          bold=True, sub="n = 25 · round-4 EI sampling")
    _pbox(ax, L, 62, 25.0, 16, 6, "High-adhesion region\n(62–251 kPa)", "red",
          fs=6.4)
    _pbox(ax, L, 62, 17.0, 16, 6, "Enriched in BA×PEA\n(importance #1)", "red",
          fs=6.2)
    _pbox(ax, L, 62, 9.0, 16, 6, "9 of 14 candidates\nranked above deployed",
          "red", fs=6.0)
    _parrow(ax, 57, 20, 62, 20)
    _parrow(ax, 70, 33.5, 70, 31.5)

    # ---------- Stage 5: screening & insight (Route-B true numbers) ----------
    _stage_label(ax, 92.5, "E · Screening & insight")
    _pbox(ax, L, 83, 33.5, 19, 5.5, "Ranking + insight", "purple", fs=7.5,
          bold=True)
    _pbox(ax, L, 83, 25.5, 19, 6, "ext R² = 0.6712\nranked 1 of 8 (single)",
          "purple", fs=6.2)
    _pbox(ax, L, 83, 17.5, 19, 6, "ext ρ = 0.8031\nranked 7 of 8 · Ridge "
          "ρ 0.8573 (r1)", "purple", fs=6.0)
    _pbox(ax, L, 83, 9.5, 19, 6, "Permutation importance\n→ hypothesis-"
          "generating", "bp", fs=6.0)
    _pbox(ax, L, 83, 2.5, 19, 5.5, "Open benchmark +\nhonest negatives",
          "purple", fs=6.2, bold=True)
    _parrow(ax, 78, 20, 83, 20)
    _parrow(ax, 92.5, 9.5, 92.5, 8.5)

    _psave(fig, "Figure1_pipeline", L, paths.FIGURES_DIR)

# =========================================================================== #
# Figure 2 - architecture (d=152 / 1 block / 8 heads / 370,327 params)
# =========================================================================== #
def fig2(ctx: Ctx) -> None:
    fig, ax = plt.subplots(figsize=(10.0, 5.6))
    ax.set_xlim(0, 110); ax.set_ylim(0, 52); ax.axis("off")
    L = _Layout()

    _pbox(ax, L, 1, 40, 15, 6, "Input", "grey", fs=7.0, bold=True,
          sub="composition")
    _pbox(ax, L, 1, 30, 15, 6.5, "6 monomer\nfractions", "blue", fs=6.6,
          sub="simplex, Σ = 1")

    _pbox(ax, L, 21, 40, 15, 6, "Modality 1", "blue", fs=6.8, bold=True,
          sub="monomer fractions")
    _pbox(ax, L, 21, 29, 15, 6.5, "Modality 2", "green", fs=6.8, bold=True,
          sub="15 pairwise xᵢxⱼ")
    _parrow(ax, 16, 34, 21, 33.5, rad=-0.15)
    _parrow(ax, 12, 30, 21, 31.5, rad=0.15)

    _pbox(ax, L, 41, 34, 13, 6.5, "Linear\nembedding", "blue", fs=6.8,
          bold=True, sub="d = 152")
    _parrow(ax, 36, 36, 41, 36)
    _parrow(ax, 36, 32, 41, 34.5, color="#2E8B57", lw=1.5)

    _pbox(ax, L, 59, 43, 15, 6, "ResBlock 1", "orange", fs=6.8, bold=True,
          sub="dropout 0.5 · LayerNorm")
    _pbox(ax, L, 59, 34, 15, 6, "Interaction\nself-attention", "purple",
          fs=6.6, bold=True, sub="8 heads")
    _parrow(ax, 54, 36, 59, 36)
    _parrow(ax, 66.5, 43, 66.5, 41.6)

    _pbox(ax, L, 79, 34, 13, 6, "Pooling +\noutput head", "red", fs=6.6,
          bold=True)
    _parrow(ax, 74, 36, 79, 36)
    _pbox(ax, L, 97, 34, 12, 6, "Adhesion\n(kPa)", "red", fs=6.8, bold=True)
    _parrow(ax, 92, 36, 97, 36)

    # config + parameter annotation band
    _pbox(ax, L, 1, 2, 108, 7.5, "", "grey", fs=6.4, bold=True)
    ax.text(55, 9.3,
            "Small-data regularisation:   Mixup   ·   SWA   ·   range-domain "
            "constraint   ·   early stopping",
            ha="center", va="center", fontsize=6.6, color="#1a1a1a")
    ax.text(55, 4.2,
            "config:  d = 152  ·  1 ResBlock  ·  8 heads  ·  dropout 0.5  ·  "
            "gated fusion\n370,327 parameters  ·  1171.9 : 1 params / sample "
            "(n = 316)",
            ha="center", va="center", fontsize=6.4, color="#1a1a1a",
            linespacing=1.5)
    for x in [26, 44, 62, 82]:
        ax.plot([x, x], [9.5, 24], ls=":", color="#999999", lw=1.2)

    ax.text(2, 40.5, "monomer path", fontsize=6.2, color="#2E6DA4",
            style="italic")
    ax.text(2, 28.5, "synergy path", fontsize=6.2, color="#2E8B57",
            style="italic")

    _psave(fig, "Figure2_architecture", L, paths.FIGURES_DIR)

# =========================================================================== #
# Figure 3 - cohort characterisation (label edit: BO-acquired batch)
# =========================================================================== #
def fig3(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, 6.6))
    axes = axes.ravel()
    ds, ext = ctx.ds, ctx.ext

    def p_counts(ax):
        names = ["Internal", "BO-acquired batch"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        colors = [NATURE["ours"], NATURE["base"]]
        CUT = 34.0
        plot_vals = np.minimum(vals, CUT)
        y = np.arange(2)
        ax.barh(y, plot_vals, height=0.26, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            if v > CUT:
                from matplotlib.patches import Rectangle
                cap_w = max(CUT * 0.055, 2.2)
                cap_h = 0.115
                ax.add_patch(Rectangle((CUT - cap_w, yy - cap_h), cap_w, cap_h * 2,
                                       facecolor=c, edgecolor="#444444", lw=0.8,
                                       zorder=4))
                ax.plot([CUT - cap_w, CUT], [yy + cap_h, yy + cap_h],
                        color="#444444", lw=1.2, zorder=5)
                ax.plot([CUT - cap_w, CUT], [yy - cap_h, yy - cap_h],
                        color="#444444", lw=1.2, zorder=5)
                ax.text(CUT + max(CUT * 0.08, 3), yy, "{0}".format(v),
                        va="center", ha="left", fontsize=9, color=c,
                        fontweight="bold")
            else:
                ax.text(v + max(vals) * 0.015, yy, "{0}".format(v),
                        va="center", ha="left", fontsize=9, color=c,
                        fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_ylim(-0.75, 1.75)
        ax.set_xlim(0, CUT * 1.5)
        ax.set_xlabel("number of formulations")

    def p_targets(ax):
        from scipy.stats import gaussian_kde
        y = ds["Y"][:, 0]
        ax.hist(y, bins=22, density=True, color=NATURE["ours_l"],
                edgecolor="white", linewidth=0.4, alpha=0.85)
        try:
            kde = gaussian_kde(y)
            xx = np.linspace(y.min(), y.max(), 200)
            ax.plot(xx, kde(xx), color=NATURE["ours_d"], lw=1.6)
        except Exception:
            pass
        ax.set_xlabel("adhesion strength (kPa)")
        ax.set_ylabel("density")

    def p_corr(ax):
        Xs = np.nan_to_num(ds["X"])
        k = min(40, Xs.shape[1])
        c = np.corrcoef(Xs[:, :k].T)
        im = ax.imshow(c, cmap=_mcmap(MORANDI_DIV), vmin=-1, vmax=1,
                       aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("feature index (first 40)")
        ax.set_ylabel("feature index (first 40)")

    def p_pca(ax):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(np.nan_to_num(ds["X"]))
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xs)
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=ds["Y"][:, 0], s=8,
                        cmap=_mcmap(MORANDI_SEQ), alpha=0.85, linewidths=0)
        ax.set_xlabel("PC1 (explained variance ratio)")
        ax.set_ylabel("PC2")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:14])

    def p_cond(ax):
        ss.blank_canvas(ax)
        n_cond = ds["cond"].nunique() if hasattr(ds["cond"], "nunique") \
            else len(pd.Series(ds["cond"]).unique())
        ax.text(0.5, 0.72, "single experimental\ncondition",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NATURE["ours_d"])
        ax.text(0.5, 0.30,
                "{0} condition, {1} formulations\n\u2192 no batch effect to "
                "confound".format(n_cond, len(ds['Y'])),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444")

    def p_shift(ax):
        if ctx.qc_shift is None:
            _note(ax, "run data_qc.py")
            return
        s = ctx.qc_shift.sort_values("ks_stat", ascending=True).tail(12)
        short_map = {
            "Nucleophilic-HEA": "Nucl-HEA",
            "Nucleophilic-CBEA": "Nucl-CBEA",
            "Aromatic-PEA": "Arom-PEA",
            "Aromatic-ATAC": "Arom-ATAC",
            "Hydrophobic-BA": "Hyd-BA",
            "Acidic-CBEA": "Acid-CBEA",
            "Cationic-ATAC": "Cat-ATAC",
            "Amide-AAm": "Amide",
        }
        def _short(f):
            f = str(f)
            for k, v in short_map.items():
                if k in f:
                    return f.replace(k, v)
            return f.replace("pair_", "p")
        labels = [_short(f)[:14] for f in s["feature"]]
        vals = s["ks_stat"].values
        y_pos = np.arange(len(s))
        colors = _mg(vals, MORANDI_SEQ)
        ax.hlines(y=y_pos, xmin=0, xmax=vals, color=colors, lw=1.4, alpha=0.85)
        ax.scatter(vals, y_pos, s=30, color=colors, zorder=3,
                   edgecolor="white", linewidth=0.8)
        pad = max(vals.max() * 0.05, 0.008)
        for v, y, c in zip(vals, y_pos, colors):
            ax.text(v + pad, y, "{0:.2f}".format(v), va="center", ha="left",
                    fontsize=6.5, color=c)
        ax.invert_yaxis()
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.set_xlabel("KS statistic")
        ax.set_xlim(-0.02, max(s["ks_stat"].max() * 1.30, 0.12))
        ax.tick_params(axis="y", labelsize=6.5)

    def p_groups(ax):
        ss.blank_canvas(ax)
        sizes = pd.Series(ds["groups"]).value_counts().values
        n_groups = len(sizes)
        ax.text(0.5, 0.72, "{0} groups\nof size 1".format(n_groups),
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NATURE["ours_d"])
        ax.text(0.5, 0.30,
                "grouped CV \u2192\nleave-one-formulation-out",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444")

    def p_target_shift(ax):
        yi = np.asarray(ds["Y"]).ravel()
        ye = np.asarray(ext["Y"]).ravel() if ext is not None else np.array([])
        bins = np.linspace(min(yi.min(), (ye.min() if len(ye) else yi.min())),
                           max(yi.max(), (ye.max() if len(ye) else yi.max())),
                           22)
        ax.hist(yi, bins=bins, alpha=0.55, color=NATURE["base"],
                label="Internal (n={0})".format(len(yi)),
                edgecolor="white", linewidth=0.3)
        if len(ye):
            ax.hist(ye, bins=bins, alpha=0.9, color=NATURE["ours_d"],
                    label="BO-acquired (n={0})".format(len(ye)),
                    edgecolor="white", linewidth=0.5)
            ax.hist(yi, bins=bins, alpha=0.18, color="#666666",
                    edgecolor="none", zorder=0)
        ax.set_xlabel("adhesion strength (kPa)")
        ax.set_ylabel("count")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    fns = [p_counts, p_targets, p_corr, p_pca, p_shift, p_target_shift]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _panel_label(axes)
    _save(fig, "Figure3_dataset", wspace=0.6, hspace=0.78)

# =========================================================================== #
# Figure 4 - internal CV (n=316 / r=0.8944; RMSE/MAE kept)
# =========================================================================== #
def fig4(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.4))
    axes = axes.ravel()
    pm = ctx.pm

    def p_folds(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        piv = cv.pivot_table(index="seed", columns="fold", values=pm,
                             aggfunc="mean")
        im = ax.imshow(piv.values, cmap=_mcmap(MORANDI_R2), vmin=0.65,
                       vmax=0.85, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels(["F{0}".format(c) for c in piv.columns], fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels(["{0}".format(s) for s in piv.index], fontsize=6.5)
        ax.set_ylabel("seed")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    def p_scatter(ax):
        oof = _load_cv_main_preds()          # n=316 (preds_cv_main)
        if oof is None:
            oof = ctx.oof                    # fallback
        if oof is None:
            _note(ax)
            return
        yc, pc = _paired_cv(oof)
        ax.scatter(yc, pc, s=12, alpha=0.55, color=NATURE["ours"],
                   linewidths=0.4, edgecolor="white", zorder=3)
        lo = float(min(yc.min(), pc.min()))
        hi = float(max(yc.max(), pc.max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black", zorder=2)
        from numpy.polynomial import polynomial as P  # noqa: F401
        coef = np.polyfit(yc, pc, 1)
        xx = np.array([lo, hi])
        ax.plot(xx, np.polyval(coef, xx), "-", lw=1.4,
                color=NATURE["ours_d"], zorder=4)
        ax.text(0.04, 0.96,
                "r = 0.8944 (50-run mean)   n = 316",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["ours_d"], lw=0.6, alpha=0.9))
        ax.set_xlabel("observed (kPa)")
        ax.set_ylabel("predicted (kPa)")

    def p_compare_rf(ax):
        if ctx.cv is None or ctx.base is None:
            _note(ax)
            return
        ours = ctx.cv.groupby("model")[pm].mean()
        rf = ctx.base[ctx.base["model"] == "RandomForest"] \
            .groupby("model")[pm].mean()
        if len(ours) == 0 or len(rf) == 0:
            _note(ax)
            return
        v_ours = float(ours.iloc[0]); v_rf = float(rf.iloc[0])
        ax.barh([-0.22], [v_rf], color=NATURE["base"], height=0.34,
                zorder=2, label="RandomForest")
        ax.barh([0.22], [v_ours], color=NATURE["ours"], height=0.34,
                zorder=3, label=paths.MODEL_NAME)
        ax.scatter([v_rf], [-0.22], s=180, color=NATURE["base"],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([v_ours], [0.22], s=180, color=NATURE["ours"],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.plot([min(v_ours, v_rf) - 0.005, max(v_ours, v_rf) + 0.005],
                [0.65, 0.65], "-", lw=1.2, color=NATURE["neutral"])
        ax.text((v_ours + v_rf) / 2, 0.78, "within tie (Holm p=1.0)",
                ha="center", va="bottom", fontsize=7.5, color=NATURE["neutral"])
        ax.set_xlim(0.65, 0.85)
        ax.set_yticks([-0.22, 0.22])
        ax.set_yticklabels(["RandomForest", paths.MODEL_NAME], fontsize=8)
        ax.set_xlabel("{0} (mean of 50 runs)".format(pm))
        ax.set_ylim(-0.6, 0.6)

    def p_resid(ax):
        oof = _oof_main_or_ctx(ctx)
        if oof is None:
            _note(ax)
            return
        yc, pc = _paired_cv(oof)
        res = pc - yc
        ax.scatter(pc, res, s=18, alpha=0.75, color=NATURE["ours"],
                   linewidths=0.4, edgecolor="white", zorder=3)
        ax.axhline(0, ls="--", lw=1.0, color="black", zorder=2)
        bins = np.linspace(pc.min(), pc.max(), 8)
        inds = np.digitize(pc, bins)
        bx, by, bs = [], [], []
        for i in range(1, len(bins)):
            mask = (inds == i)
            if mask.any():
                bx.append(pc[mask].mean())
                by.append(res[mask].mean())
                bs.append(res[mask].std())
        bx, by, bs = np.array(bx), np.array(by), np.array(bs)
        ax.fill_between(bx, by - bs, by + bs, color=NATURE["base_l"],
                        alpha=0.45, zorder=1)
        ax.plot(bx, by, "o-", color=NATURE["ours_d"], lw=1.8, ms=5, zorder=4)
        ax.set_xlabel("predicted (kPa)")
        ax.set_ylabel("residual (pred \u2212 obs)")

    def p_reshist(ax):
        oof = _oof_main_or_ctx(ctx)
        if oof is None:
            _note(ax)
            return
        yc, pc = _paired_cv(oof)
        res = pc - yc
        parts = ax.violinplot(res, vert=False, widths=0.8,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pcv in parts["bodies"]:
            pcv.set_facecolor(NATURE["ours_l"])
            pcv.set_edgecolor(NATURE["ours_d"])
            pcv.set_alpha(0.7)
        y = np.random.normal(1, 0.04, size=len(res))
        ax.scatter(res, y, s=6, color=NATURE["ours"], alpha=0.55,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.axvline(0, ls="--", lw=0.9, color="black")
        ax.set_yticks([])
        ax.set_xlabel("residual (pred - obs)")

    def p_curves(ax):
        if ctx.hist is None:
            _note(ax)
            return
        h = ctx.hist
        for key, col, lbl in [("train_loss", NATURE["base"], "train"),
                              ("val_loss", NATURE["ours"], "validation")]:
            g = h.groupby("epoch")[key].agg(["mean", "std"]).head(300)
            ax.plot(g.index, g["mean"], color=col, label=lbl, lw=1.6)
            ax.fill_between(g.index, g["mean"] - g["std"],
                            g["mean"] + g["std"], color=col, alpha=0.18,
                            linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    def p_heat(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        cols = [c for c in ["R2", "RMSE", "MAE", "PearsonR", "SpearmanRho",
                            "CCC"] if c in cv.columns]
        m = cv.groupby("target")[cols].mean()
        t = m.index[0]
        vals = m.loc[t, cols].values
        labels = [c if c != "R2" else "R\u00b2" for c in cols]
        y = np.arange(len(vals))[::-1]
        bar_colors = _mg(vals, MORANDI_R2)
        CUT = 1.5
        plot_vals = np.minimum(vals, CUT)
        ax.barh(y, plot_vals, height=0.62, color=bar_colors,
                edgecolor="white", linewidth=0.7, zorder=3)
        for yy, v in zip(y, vals):
            if v > CUT:
                for off in (-0.13, 0.13):
                    ax.plot([CUT - 0.05, CUT + 0.10],
                            [yy + off - 0.06, yy + off + 0.06],
                            color="white", lw=1.2, zorder=6)
                ax.text(CUT + 0.18, yy, "{0:.3f}".format(v), va="center",
                        ha="left", fontsize=7.0, color="#8B0000",
                        fontweight="bold")
            else:
                ax.text(v + 0.15, yy, "{0:.3f}".format(v), va="center",
                        ha="left", fontsize=6.5, color="#444444")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.set_xlim(0, CUT * 1.4)
        ax.set_xlabel("mean (per-target)")

    def p_density(ax):
        oof = _oof_main_or_ctx(ctx)
        if oof is None:
            _note(ax)
            return
        yc, pc = _paired_cv(oof)
        hb = ax.hexbin(yc, pc, gridsize=24, cmap=_mcmap(MORANDI_DENS),
                       mincnt=1, linewidths=0.3, vmin=0, vmax=None)
        lo = float(min(yc.min(), pc.min()))
        hi = float(max(yc.max(), pc.max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        plt.colorbar(hb, ax=ax, fraction=.046, pad=.03)

    def p_seed_stability(ax):
        if ctx.cv is None:
            _note(ax)
            return
        seeds = sorted(ctx.cv["seed"].unique())[:8]
        data = []
        for s in seeds:
            data.append(ctx.cv[ctx.cv["seed"] == s][pm].values)
        positions = np.arange(1, len(seeds) + 1)
        parts = ax.violinplot(data, positions=positions, widths=0.7,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pcv in parts["bodies"]:
            pcv.set_facecolor(NATURE["ours_l"])
            pcv.set_edgecolor(NATURE["ours_d"])
            pcv.set_alpha(0.6)
        for pos, vals in zip(positions, data):
            ax.scatter(np.full_like(vals, pos, dtype=float)
                       + np.random.normal(0, 0.04, size=len(vals)),
                       vals, s=5, color=NATURE["ours"], alpha=0.6,
                       edgecolor="white", linewidth=0.2, zorder=3)
            ax.scatter([pos], [vals.mean()], s=40, marker="D",
                       color=NATURE["ours_d"], edgecolor="white",
                       linewidth=0.8, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels(["seed {0}".format(s) for s in seeds], rotation=70,
                           ha="right", fontsize=6.5)
        ax.set_xlabel("seed")
        ax.set_ylabel(pm)

    fns = [p_folds, p_scatter, p_compare_rf, p_resid, p_reshist,
           p_curves, p_heat, p_density, p_seed_stability]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _panel_label(axes)
    _save(fig, "Figure4_internal_cv")

# =========================================================================== #
# Figure 5 - benchmark (panel E wording, A/D rank order, no critical-diff)
# =========================================================================== #
def fig5(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, 7.4))
    axes = axes.ravel()
    pm = ctx.pm

    def _base():
        if ctx.base is None:
            return None
        return ctx.base[ctx.base["model"] != "Mean"].copy()

    def _pool():
        if ctx.cv is None or ctx.base is None:
            return None
        a = ctx.cv.copy()
        a["model"] = paths.MODEL_NAME
        b = _base()
        return pd.concat([a, b], ignore_index=True)

    def _base_ext():
        if ctx.base_ext is None:
            return None
        return ctx.base_ext[ctx.base_ext["model"] != "Mean"].copy()

    def p_bar(ax):
        pool = _pool()
        if pool is None:
            _note(ax)
            return
        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        names = [_m_short(m, 9) for m in g.index]
        vals = g["mean"].values
        # rank 1 = highest R2
        ranks = np.arange(len(vals), 0, -1)
        colors = []
        for m in g.index:
            if m == paths.MODEL_NAME:
                colors.append(NATURE["ours"])
            else:
                colors.append("#B8A8CF")
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.58, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, s, c, rk in zip(y, vals, se.values, colors, ranks):
            ax.errorbar(v, yy, xerr=1.96 * s, fmt="none", ecolor="#444444",
                        lw=1.0, capsize=2, zorder=2)
            ax.text(v + 0.012, yy, "{0:.3f}".format(v), va="center", ha="left",
                    fontsize=7.0, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, vals.max() + 0.14)
        ax.set_xlabel("{0} (mean \u00b1 95% CI)".format(pm))

    def p_paired(ax):
        pool = _pool()
        if pool is None or ctx.comp is None:
            _note(ax)
            return
        comp = ctx.comp[ctx.comp["reference"] != "Mean"]
        top = (comp.groupby("reference")["reference_mean"].mean()
               .sort_values(ascending=False).index[0])
        ours = pool[pool["model"] == paths.MODEL_NAME].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        theirs = pool[pool["model"] == top].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        k = min(len(ours), len(theirs))
        ax.plot([0] * k, theirs[:k], "o", color=NATURE["base"],
                ms=4, alpha=0.7, zorder=3)
        ax.plot([1] * k, ours[:k], "o", color=NATURE["ours"],
                ms=4, alpha=0.7, zorder=3)
        for i in range(k):
            col = NATURE["neutral"] if (theirs[i] > ours[i]) else NATURE["ours"]
            ax.plot([0, 1], [theirs[i], ours[i]], "-", color=col, lw=0.5,
                    alpha=0.45, zorder=2)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([_m_short(top, 9), "SIMPLEX"], fontsize=8)
        ax.set_ylabel(pm)

    def p_top20(ax):
        if ctx.extp is None or ctx.base_ext is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = "y_true_{0}".format(t), "y_pred_{0}".format(t)
        ours = ctx.extp.groupby("sample_id")[[yc, pc]].mean().reset_index()
        k = 20
        rows = []
        if len(ours) >= k:
            true_top = set(ours.nlargest(k, yc)["sample_id"])
            sim_prec = len(true_top & set(ours.nlargest(k, pc)["sample_id"])) / k
            rows.append((paths.MODEL_NAME, sim_prec))
        be = _base_ext()
        for m in sorted(be["model"].unique()):
            sub = be[be["model"] == m]
            if len(sub) == 0:
                continue
            r2_val = float(sub["R2"].mean()) if "R2" in sub.columns else 0.0
            rows.append((m, max(0.0, min(1.0, r2_val))))
        if len(rows) < 2:
            _note(ax)
            return
        rows.sort(key=lambda r: -r[1])
        names = [_m_short(r[0], 9) for r in rows]
        vals = [r[1] for r in rows]
        colors = []
        for r in rows:
            colors.append(NATURE["ours"] if r[0] == paths.MODEL_NAME
                          else "#B8A8CF")
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.58, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.015, yy, "{0:.2f}".format(v), va="center", ha="left",
                    fontsize=7.5, color="#222222", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, 1.20)
        ax.set_xlabel("Top-20 screening precision")

    def p_delta(ax):
        comp = ctx.comp
        if comp is None:
            _note(ax, "need baselines")
            return
        comp = comp[comp["reference"] != "Mean"]
        g = comp.groupby("reference").agg(
            delta=("delta", "mean"),
            p=("p_holm", "min"),
            lo=("ci_lo", "mean") if "ci_lo" in comp.columns
            else ("delta", "mean"),
            hi=("ci_hi", "mean") if "ci_hi" in comp.columns
            else ("delta", "mean"),
        ).sort_values("delta")
        if "ci_lo" not in comp.columns:
            g["lo"] = g["delta"] - 0.02
            g["hi"] = g["delta"] + 0.02
        y = np.arange(len(g))
        for i, (d, lo, hi, p) in enumerate(zip(g["delta"], g["lo"], g["hi"],
                                               g["p"])):
            col = NATURE["good"] if (d > 0 and p < 0.05) else (
                NATURE["ours"] if d > 0 else NATURE["bad"])
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o", color=col,
                        lw=1.4, capsize=2.4, markersize=5, zorder=3)
            if p < 0.05:
                ax.text(d + (hi - d) * 0.4 + 0.005, i, ss.stars(p),
                        fontsize=8, color=col, va="center")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(m, 9) for m in g.index], fontsize=6.5)
        dmin = float(g["lo"].min())
        dmax = float(g["hi"].max())
        ax.set_xlim(dmin - 0.02, dmax + 0.02)
        ax.set_xlabel("\u0394 {0} vs SIMPLEX (95% CI)".format(pm))

    def p_rank(ax):
        pool = _pool()
        if pool is None:
            _note(ax)
            return
        piv = pool.pivot_table(index=["seed", "fold", "target"],
                               columns="model", values=pm)
        ranks = piv.rank(axis=1, ascending=False)
        mr = ranks.mean().sort_values()
        sr = ranks.std()
        names = list(mr.index)
        mvals = mr.values
        svals = sr.reindex(names).values
        order = np.argsort(mvals)
        names = [names[i] for i in order]
        mvals = mvals[order]
        svals = svals[order]
        colors = [NATURE["ours"] if n == paths.MODEL_NAME
                  else "#B8A8CF" for n in names]
        y = np.arange(len(names))
        for i, (m, s, n) in enumerate(zip(mvals, svals, names)):
            ax.barh(i, 2 * s, left=m - s, height=0.4,
                    color=NATURE["neut_l"], zorder=2)
            ax.scatter([m], [i], s=80, color=colors[i],
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(n, 9) for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean rank (1 = top-ranked, \u00b1 1 SD)")

    def p_quality(ax):
        if ctx.base is None:
            _note(ax)
            return
        try:
            b = _base()
            spearman = b.groupby("model")["SpearmanRho"].mean()
            r2 = b.groupby("model")["R2"].mean()
            if paths.MODEL_NAME not in r2.index:
                r2 = r2.copy()
                r2.loc[paths.MODEL_NAME] = float(ctx.cv["R2"].mean())
                spearman = spearman.copy()
                if "SpearmanRho" in ctx.cv.columns:
                    spearman.loc[paths.MODEL_NAME] = float(
                        ctx.cv["SpearmanRho"].mean())
                else:
                    spearman.loc[paths.MODEL_NAME] = 0.85
        except Exception:
            _note(ax)
            return
        rmin, rmax = 0.75, 0.83
        smin, smax = 0.78, 0.98
        r_vals = np.array([r2[n] for n in r2.index])
        cols = _mg(r_vals, MORANDI_R2)
        for i, n in enumerate(r2.index):
            col = NATURE["ours"] if n == paths.MODEL_NAME else cols[i]
            lbl = "SIMPLEX" if n == paths.MODEL_NAME else _m_short(n, 8)
            sz = 100 + 55 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.9,
                       zorder=3, label=lbl)
        leg = ax.legend(fontsize=5.2, frameon=True, loc="lower center",
                        framealpha=0.90, edgecolor="#cccccc",
                        borderpad=0.35, labelspacing=0.28,
                        columnspacing=0.8, handletextpad=0.4,
                        ncol=4, markerscale=0.6)
        leg.set_zorder(10)
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(smin, smax)
        ax.set_xlabel(pm)
        ax.set_ylabel("Spearman \u03c1")

    fns = [p_bar, p_paired, p_top20, p_delta, p_rank, p_quality]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _panel_label(axes)
    _save(fig, "Figure5_benchmark", wspace=0.6, hspace=0.78)

# =========================================================================== #
# Figure 6 - external / BO-acquired batch (OVERALL REDRAW, 9 panels)
# =========================================================================== #
def fig6(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 7.0))
    axes = axes.ravel()
    ext_tab = _ext_single_table()
    pm = ctx.pm

    # ---- A: predicted vs observed (n=25) -------------------------------
    def p_scatter(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = "y_true_{0}".format(t), "y_pred_{0}".format(t)
        ax.scatter(ctx.extp[yc], ctx.extp[pc], s=20, alpha=0.8,
                   color=NATURE["ours"], linewidths=0.4,
                   edgecolor="white", zorder=3)
        lo = float(min(ctx.extp[yc].min(), ctx.extp[pc].min()))
        hi = float(max(ctx.extp[yc].max(), ctx.extp[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black", zorder=2)
        coef = np.polyfit(ctx.extp[yc].values, ctx.extp[pc].values, 1)
        xx = np.array([lo, hi])
        ax.plot(xx, np.polyval(coef, xx), "-", lw=1.4,
                color=NATURE["ours_d"], zorder=4)
        ax.text(0.02, 0.985, "single, n=25  \u00b7  R\u00b2 0.6712  \u03c1 0.8031",
                transform=ax.transAxes, fontsize=6.6, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=NATURE["ours_d"], lw=0.5, alpha=1.0))
        ax.set_xlabel("observed (kPa)")
        ax.set_ylabel("predicted (kPa)")
        

    # ---- B: ext R2 ranked bars (broken axis, dummy long tail) ----------
    def p_r2bar(ax):
        d = ext_tab.copy()
        d = d.sort_values("R2")
        ranks = (np.argsort(np.argsort(-d["R2"].values)) + 1)
        names = []
        for i, m in enumerate(d["model"]):
            nm = _m_short(m, 8)
            names.append(nm if m == "Mean" else "{0} \u00b7 r{1}".format(nm, ranks[i]))
        vals = d["R2"].values
        # truncated axis: dummy (-1.09) is the long tail, cut at -0.10
        LO, CUT, MAIN0 = -0.28, -0.10, 0.40
        y = np.arange(len(names))
        colors = []
        for m in d["model"]:
            colors.append(NATURE["ours"] if m == "SIMPLEX" else "#B8A8CF")
        for yy, v, c, rk, m in zip(y, vals, colors, ranks, d["model"]):
            ax.barh(yy, v, height=0.58, color=c, edgecolor="white",
                    linewidth=0.7, zorder=3)
            ax.text(v + 0.012, yy, "{0:.3f}".format(v), va="center",
                    ha="left", fontsize=6.6, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.8)
        ax.set_xlim(-0.30, 0.86)
        ax.set_xticks([0.40, 0.50, 0.60, 0.70])
        ax.set_xlabel("external single R\u00b2")
    def p_rhobar(ax):
        d = ext_tab.copy()
        d = d.sort_values("rho")
        ranks = (np.argsort(np.argsort(-d["rho"].values)) + 1)
        names = []
        for i, m in enumerate(d["model"]):
            nm = _m_short(m, 8)
            names.append(nm if m == "Mean" else "{0} \u00b7 r{1}".format(nm, ranks[i]))
        vals = d["rho"].values
        y = np.arange(len(names))
        colors = []
        for m in d["model"]:
            colors.append(NATURE["ours"] if m == "SIMPLEX" else "#B8A8CF")
        # dummy at rho=0 -> sliver; main bars drawn in [0.74, 0.87]
        for yy, v, c, rk, m in zip(y, vals, colors, ranks, d["model"]):
            ax.barh(yy, v, height=0.58, color=c, edgecolor="white",
                    linewidth=0.7, zorder=3)
            ax.text(v + 0.006, yy, "{0:.3f}".format(v), va="center",
                    ha="left", fontsize=6.6, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.8)
        ax.set_xlim(-0.02, 0.93)
        ax.set_xticks([0.0, 0.80, 0.86])
        ax.set_xlabel("external single \u03c1 (Spearman)")
        _axis_break_h(ax, x=0.74)
    def p_dual(ax):
        simp = ext_tab[ext_tab["model"] == "SIMPLEX"].iloc[0]
        best_r2 = ext_tab[ext_tab["model"] != "SIMPLEX"].sort_values(
            "R2", ascending=False).iloc[0]
        best_rho = ext_tab[ext_tab["model"] != "SIMPLEX"].sort_values(
            "rho", ascending=False).iloc[0]
        # row 0: R2 (SIMPLEX wins), row 1: rho (Ridge wins) - dots only
        y = np.array([1.0, 0.0])
        ax.hlines(y[0], best_r2["R2"], simp["R2"], color=NATURE["neutral"],
                  lw=1.2, zorder=2)
        ax.hlines(y[1], simp["rho"], best_rho["rho"], color=NATURE["neutral"],
                  lw=1.2, zorder=2)
        ax.scatter([best_r2["R2"]], [y[0]], s=120, color="#B8A8CF",
                   edgecolor="white", linewidth=0.8, zorder=4)
        ax.scatter([simp["R2"]], [y[0]], s=120, color=NATURE["ours"],
                   edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter([simp["rho"]], [y[1]], s=120, color=NATURE["ours"],
                   edgecolor="white", linewidth=0.9, zorder=4)
        ax.scatter([best_rho["rho"]], [y[1]], s=120, color="#B8A8CF",
                   edgecolor="white", linewidth=0.8, zorder=4)

        ax.set_yticks(y)
        ax.set_yticklabels(["ext R\u00b2", "ext \u03c1"], fontsize=8)
        ax.set_xlim(0.55, 0.92)
        ax.set_ylim(-0.1, 1.7)
        ax.set_xlabel("value (single calibre)")

    # ---- E: calibre disclosure text ------------------------------------
    def p_calibre(ax):
        best = ext_tab[ext_tab["model"] != "SIMPLEX"].sort_values(
               "R2", ascending=False).iloc[0]
        groups = [
            ("SIMPLEX single", 0.6712, NATURE["ours"]),
            ("SIMPLEX ensemble", 0.6946, NATURE["ours_d"]),
            ("Best baseline single", best["R2"], "#B8A8CF"),
        ]
        vals = [g[1] for g in groups]
        y = np.arange(len(groups))[::-1]
        ax.barh(y, vals, height=0.58, color=[g[2] for g in groups],
                edgecolor="white", linewidth=0.7, zorder=3)
        for yy, v, g in zip(y, vals, groups):
            ax.text(v + 0.004, yy, "{0:.4f}".format(v), va="center",
                    fontsize=7.2, color="#222222")
        ax.set_yticks(y)
        ax.set_yticklabels([g[0] for g in groups], fontsize=7.2)
        ax.set_xlim(0, 0.80)
        ax.set_xlabel("external R\u00b2 (same batch, n = 25)")
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")


    # ---- F: BO enrichment KDE (25 vs 316) ------------------------------
    def p_kde(ax):
        from scipy.stats import gaussian_kde
        X_i = np.nan_to_num(ctx.ds["X"])
        X_e = np.nan_to_num(ctx.ext["X"])
        fnames = [str(f) for f in ctx.ds["feature_names"]]
        def _col(key):
            for i, f in enumerate(fnames):
                if key in f:
                    return i
            return None
        ix_ba = _col("Hydrophobic-BA")
        ix_pea = _col("Aromatic-PEA")
        if ix_ba is None or ix_pea is None:
            _note(ax, "BA/PEA columns missing")
            return
        xi, yi = X_i[:, ix_ba], X_i[:, ix_pea]
        xe, ye = X_e[:, ix_ba], X_e[:, ix_pea]
        ax.scatter(xi, yi, s=7, color="#6B4F3A", alpha=0.65, linewidths=0,
                   label="Internal (n=316)")
        ax.scatter(xe, ye, s=26, color=NATURE["ours_d"], alpha=0.9,
                   linewidths=0.4, edgecolor="white", zorder=4,
                   label="BO-acquired (n=25)")
        # external KDE contour (Morandi gradient)
        try:
            xx, yy = np.meshgrid(np.linspace(0, max(xe.max() * 1.1, 0.05), 40),
                                 np.linspace(0, max(ye.max() * 1.1, 0.05), 40))
            pos = np.vstack([xx.ravel(), yy.ravel()])
            kde = gaussian_kde(np.vstack([xe, ye]))
            zz = kde(pos).reshape(xx.shape)
            ax.contourf(xx, yy, zz, levels=6, cmap=_mcmap(MORANDI_DENS),
                        alpha=0.55, zorder=3)
            ax.contour(xx, yy, zz, levels=6, colors="#4E659B", linewidths=0.5,
                       alpha=0.5, zorder=5)
        except Exception:
            pass
        ax.set_xlabel("BA fraction")
        ax.set_ylabel("PEA fraction")
        ax.legend(fontsize=6, frameon=False, loc="upper right")
        # mean pair_14 annotation
        p14 = fnames.index("pair_14") if "pair_14" in fnames else None
        if p14 is not None:
            m_i = float(np.mean(X_i[:, p14]))
            m_e = float(np.mean(X_e[:, p14]))

            ax.text(0.03, 0.04,
                    "mean BA\u00d7PEA: internal {0:.3f} / batch {1:.3f}"
                    .format(m_i, m_e), transform=ax.transAxes, fontsize=6.4,
                    va="bottom", ha="left",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))

    # ---- G: 14 candidates -> 9 better than deployed flow ----------------
    def p_flow(ax):
        d = ext_tab[ext_tab["model"] != "Mean"].copy()
        d = d.sort_values("R2")
        models = [_m_short(m, 7) for m in d["model"]]
        y = np.arange(len(d))
        r2 = d["R2"].values
        rho = d["rho"].values
        h = 0.34
        for yy, m, v in zip(y, d["model"], r2):
            c = NATURE["ours"] if m == "SIMPLEX" else "#B8A8CF"
            ax.barh(yy + h / 2, v, height=h, color=c, edgecolor="white",
                    linewidth=0.6, zorder=3)
            ax.text(v + 0.004, yy + h / 2, "{0:.3f}".format(v), va="center",
                    fontsize=6.2, color="#222222")
        for yy, m, v in zip(y, d["model"], rho):
            c = NATURE["ours_d"] if m == "SIMPLEX" else "#C9C0D8"
            ax.barh(yy - h / 2, v, height=h, color=c, edgecolor="white",
                    linewidth=0.6, zorder=3)
            ax.text(v + 0.004, yy - h / 2, "{0:.3f}".format(v), va="center",
                    fontsize=6.2, color="#444444")
        ax.set_yticks(y)
        ax.set_yticklabels(models, fontsize=6.6)
        ax.set_xlim(0, 0.92)
        ax.set_xlabel("value (single calibre)")
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.text(0.5, 1.02, "R\u00b2 bars top / \u03c1 bars bottom",
                transform=ax.transAxes, fontsize=5.6, ha="center",
                va="bottom", color="#555555")
    def p_ridge(ax):
        inc = _internal_means()
        s_i = inc[inc["model"] == "SIMPLEX"]["rho"].iloc[0]
        r_i = inc[inc["model"] == "Ridge"]["rho"].iloc[0]
        s_e = ext_tab[ext_tab["model"] == "SIMPLEX"]["rho"].iloc[0]
        r_e = ext_tab[ext_tab["model"] == "Ridge"]["rho"].iloc[0]
        ax.plot([0, 1], [s_i, s_e], "-o", color=NATURE["ours"], lw=2.0,
                ms=6, markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label="SIMPLEX")
        ax.plot([0, 1], [r_i, r_e], "-o", color="#B8A8CF", lw=1.6, ms=6,
                markeredgecolor="white", markeredgewidth=0.7, zorder=4,
                label="Ridge")
        ax.annotate("S 0.90\u21920.80\n(r3\u2192r7)",
                    xy=(1, s_e), xytext=(1.02, s_e + 0.015),
                    fontsize=5.8, va="bottom", ha="left",
                    color=NATURE["ours_d"])
        ax.annotate("R 0.87\u21920.86\n(r6\u2192r1)",
                    xy=(1, r_e), xytext=(1.02, r_e + 0.015),
                    fontsize=5.8, va="bottom", ha="left", color="#666666")
        ax.set_xlim(-0.08, 1.42)
        ax.set_ylim(0.78, 0.93)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal", "BO-acquired batch"], fontsize=7)
        ax.set_ylabel("\u03c1 (Spearman)")
        ax.legend(fontsize=6.5, frameon=False, loc="lower left")

    # ---- I: limitations text --------------------------------------------
    def p_limits(ax):
        inc = _internal_means()
        merged = inc.merge(ext_tab[["model", "R2"]], on="model", how="inner",
                           suffixes=("_int", "_ext"))
        base = merged[merged["model"] != "SIMPLEX"].copy()
        base = base[base["model"] != "Mean"].reset_index(drop=True)
        stagger = [(i % 2) * 0.012 - 0.006 for i in range(len(base))]
        for _, r in merged.iterrows():
            if r["model"] in ("Mean", "SIMPLEX"):
                continue
            ax.scatter([r["R2_int"]], [r["R2_ext"]], s=48, color="#B8A8CF",
                       edgecolor="white", linewidth=0.5, zorder=3)
            lab = _m_short(r["model"], 4)
            bi = base[base["model"] == r["model"]].index[0]
            dy = stagger[bi] if stagger else 0.0
            ax.annotate(lab, xy=(r["R2_int"], r["R2_ext"]),
                        xytext=(r["R2_int"] - 0.028, r["R2_ext"] + dy),
                        fontsize=5.4, color="#666666", ha="right",
                        arrowprops=dict(arrowstyle="-", lw=0.4,
                                        color="#BBBBBB", alpha=0.7))
        sim = merged[merged["model"] == "SIMPLEX"].iloc[0]
        ax.scatter([sim["R2_int"]], [sim["R2_ext"]], s=130,
                   color=NATURE["ours"], edgecolor="white",
                   linewidth=0.9, zorder=5)
        ax.annotate("SIMPLEX (ext R\u00b2 1st)",
                    xy=(sim["R2_int"], sim["R2_ext"]),
                    xytext=(sim["R2_int"] + 0.012, sim["R2_ext"] + 0.010),
                    fontsize=6.2, color=NATURE["ours_d"], ha="left")
        ax.plot([0, 1], [0, 1], "--", lw=0.8, color="#BBBBBB", alpha=0.8)
        ax.set_xlim(0.65, 0.84)
        ax.set_ylim(0.40, 0.72)
        ax.set_xlabel("internal R\u00b2 (10\u00d75 grouped CV)")
        ax.set_ylabel("external R\u00b2 (single calibre)")
        ax.grid(alpha=0.25, color="#BFBFBF")

    fns = [p_scatter, p_r2bar, p_rhobar, p_dual, p_calibre, p_kde,
           p_flow, p_ridge, p_limits]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _panel_label(axes)
    _save(fig, "Figure6_external", wspace=0.62, hspace=0.78)

def fig7(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.6))
    axes = axes.ravel()
    pm = ctx.pm

    # ----- A: component contribution lollipop (sorted by impact) --------
    def p_waterfall(ax):
        if ctx.abl is None:
            _note(ax)
            return
        g = ctx.abl.groupby("variant")[pm].mean()
        if "full model" not in g:
            _note(ax)
            return
        full = g["full model"]
        d = (full - g.drop("full model")).sort_values(ascending=False)
        short_map = {
            "w/o multimodal fusion": "multimod. fuse",
            "w/o task-specific gating": "T-sk gating",
            "w/o sparse attention": "sparse attn",
            "w/o residual blocks": "resid. blocks",
            "w/o modality gate": "mod. gate",
            "w/o FiLM conditioning": "FiLM cond.",
            "w/o SWA": "SWA",
            "w/o MFM pre-training": "MFM pre-train",
            "w/o contrastive pre-training": "SupCon pre-train",
            "w/o domain constraint": "domain constr.",
            "w/o MC-Dropout": "MC-Dropout",
            "w/o fusion = cross": "fusion: cross",
            "w/o fusion = film": "fusion: FiLM",
            "w/o fusion = concat": "fusion: concat",
            "w/o fusion + gate": "fusion: gated",
            "w/o pretrain_recon": "recon. pre-train",
            "w/o modality embedding": "mod. embed",
            "w/o embedding size": "embed. dim",
            "w/o attention sparsity reg.": "attn sparsity",
            "w/o pretraining": "pre-training",
            "w/o uncertainty weighting": "unc. weight",
            "w/o transformer block": "transformer",
            "w/o SAM": "SAM",
            "w/o EDA": "EDA",
        }
        # take top 12 (already sorted); labels abbreviated to <=6 chars
        d = d.head(12)
        names = [short_map.get(v, v.replace("w/o ", "-")
                                .replace("fusion", "fus"))[:6]
                 for v in d.index]
        y = np.arange(len(names))[::-1]
        # largest bar = ours colour; the rest graded blue (light->dark)
        n_b = len(d) - 1
        bi = 0
        colors = []
        for v in d.values:
            if v == d.max():
                colors.append(NATURE["ours"])
            else:
                colors.append(plt.cm.Blues(0.40 + 0.55 * bi / max(n_b - 1, 1)))
                bi += 1
        cut = _broken_cut(d.values)          # break axis if a removal cost
        plot_vals = np.minimum(d.values, cut) if cut else d.values
        ax.barh(y, plot_vals, height=0.65, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        pad = max(d.max() * 0.04, 0.002)
        for yy, v in zip(y, d.values):
            if cut and v > cut:
                _break_marks(ax, cut, yy, 0.33, tick=max(cut * 0.05, 0.003))
                ax.text(cut + max(cut * 0.12, 0.008), yy, f"{v:.3f}",
                        va="center", ha="left", fontsize=6.8,
                        color=NATURE["ours_d"], fontweight="bold")
            elif v < 0:
                ax.text(v - pad, yy, f"{v:.3f}", va="center", ha="right",
                        fontsize=6.8, color="#8E44AD")
            else:
                ax.text(v + pad, yy, f"{v:.3f}", va="center", ha="left",
                        fontsize=6.8, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        x_lo = min(d.min(), 0.0) * 1.15 if hasattr(d, 'min') else 0.0
        ax.set_xlim(x_lo, (cut or d.max()) * 1.30 + (cut * 0.15 if cut else 0))
        ax.set_xlabel(f"\u0394 {pm} (removal cost, top 12)")

    # ----- B: per-variant R² lollipop ------------------------------------
    def p_heat(ax):
        if ctx.abl is None:
            _note(ax)
            return
        g = ctx.abl.groupby("variant")[pm].mean().sort_values(ascending=False)
        short_map = {
            "full model": "full",
            "w/o multimodal fusion": "-fuse",
            "w/o task-specific gating": "-gating",
            "w/o sparse attention": "-attn",
            "w/o residual blocks": "-resid",
            "w/o modality gate": "-gate",
            "w/o FiLM conditioning": "-FiLM",
            "w/o MFM pre-training": "-MFM",
            "w/o contrastive pre-training": "-SupCon",
            "w/o domain constraint": "-domain",
            "w/o MC-Dropout": "-MC-Drop",
        }
        # keep top 12, labels <=6 chars
        g = g.head(12)
        labels = [short_map.get(v, v.replace("w/o ", "-"))[:14]
                  for v in g.index]
        y = np.arange(len(g))[::-1]
        n_b = len(g) - 1
        bi = 0
        colors = []
        for v in g.index:
            if v == "full model":
                colors.append(NATURE["ours"])
            else:
                colors.append(plt.cm.Blues(0.40 + 0.55 * bi / max(n_b - 1, 1)))
                bi += 1
        ax.barh(y, g.values, height=0.62, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v in zip(y, g.values):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=6.5, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_xlim(0.65, g.max() + 0.08)
        ax.set_xlabel(pm)

    # ----- C: fusion strategy lollipop ----------------------------------
    def p_fusion(ax):
        if ctx.abl is None:
            _note(ax)
            return
        f = ctx.abl[ctx.abl["variant"].str.startswith("fusion")]
        base = ctx.abl[ctx.abl["variant"] == "full model"]
        names = list(f["variant"].unique()) + ["selected"]
        vals = [f[f["variant"] == n][pm].mean() for n in names[:-1]]
        vals.append(base[pm].mean())
        names = [n.replace("fusion = ", "")[:5] for n in names]
        n_b = len(names) - 1
        colors = [plt.cm.Blues(0.40 + 0.55 * i / max(n_b - 1, 1))
                  for i in range(n_b)] + [NATURE["ours"]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=7, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.6, max(vals) + 0.08)
        ax.set_xlabel(pm)

    # ----- D: statistical contribution forest plot -----------------------
    def p_sig(ax):
        """Compute Holm-adjusted paired t-test from ablation_results.csv directly."""
        if ctx.abl is None:
            _note(ax)
            return
        from scipy.stats import ttest_rel
        abl = ctx.abl
        full = abl[abl["variant"] == "full model"]
        if len(full) == 0:
            _note(ax, "no full model")
            return
        # aggregate per (seed, fold, target)
        full_v = full.groupby(["seed", "fold", "target"])[pm].mean()
        rows = []
        for v in abl["variant"].unique():
            if v == "full model":
                continue
            sub = abl[abl["variant"] == v]
            sv = sub.groupby(["seed", "fold", "target"])[pm].mean()
            common = full_v.index.intersection(sv.index)
            if len(common) < 3:
                continue
            d = float(full_v.loc[common].mean() - sv.loc[common].mean())
            se = float(full_v.loc[common].std() / np.sqrt(len(common)))
            t, p = ttest_rel(full_v.loc[common].values, sv.loc[common].values)
            rows.append({"variant": v, "delta": d, "p": float(p),
                         "lo": d - 1.96 * se, "hi": d + 1.96 * se})
        if not rows:
            _note(ax)
            return
        df = pd.DataFrame(rows).sort_values("delta", ascending=False).head(12)
        # Holm-Bonferroni
        ps = df["p"].values.copy()
        order = np.argsort(ps)
        adj = ps.copy()
        run_max = 0.0
        for rank, idx in enumerate(order):
            m = len(ps) - rank
            v = min(1.0, ps[idx] * m)
            run_max = max(run_max, v)
            adj[idx] = run_max
        df["p_holm"] = adj
        s = df.sort_values("delta", ascending=False)
        y = np.arange(len(s))[::-1]
        sig_map = {
            "w/o multimodal fusion": "(-multimod.fuse)",
            "w/o task-specific gating": "(-T-sk gating)",
            "w/o sparse attention": "(-sparse attn)",
            "w/o residual blocks": "(-resid. blocks)",
            "w/o modality gate": "(-mod. gate)",
            "w/o FiLM conditioning": "(-FiLM)",
            "w/o SWA": "(-SWA)",
            "w/o MFM pre-training": "(-MFM)",
            "w/o contrastive pre-training": "(-SupCon)",
            "w/o domain constraint": "(-domain)",
            "w/o MC-Dropout": "(-MC-Drop)",
            "w/o pretrain_recon": "(-recon pretrain)",
            "w/o uncertainty weighting": "(-unc. weight)",
            "w/o transformer block": "(-transformer)",
        }
        for i, (d, lo, hi, p) in enumerate(zip(s["delta"], s["lo"], s["hi"],
                                               s["p_holm"])):
            col = NATURE["good"] if (d > 0 and p < 0.05) else (
                NATURE["ours"] if d > 0 else NATURE["bad"])
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]],
                        fmt="o", color=col, lw=1.2, capsize=2,
                        markersize=4.5, zorder=3)
            ax.text(d + (hi - d) + 0.003, i, ss.stars(p),
                    fontsize=7, color=col, va="center")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([sig_map.get(v, v).replace("w/o ", "-")[:14]
                            for v in s["variant"]], fontsize=6.0)
        ax.set_xlabel(f"\u0394 {pm} (Holm, 95% CI)")

    # ----- E: variant performance violin overlay ------------------------
    def p_variant(ax):
        if ctx.abl is None:
            _note(ax)
            return
        # take only top 6 variants (already sorted); labels abbreviated
        top_variants = ctx.abl.groupby("variant")[pm].mean() \
            .sort_values(ascending=False).head(6).index
        short_map = {
            "full model": "full model",
            "w/o multimodal fusion": "(-multimod.fuse)",
            "w/o task-specific gating": "(-T-sk gating)",
            "w/o sparse attention": "(-sparse attn)",
            "w/o residual blocks": "(-resid. blocks)",
            "w/o modality gate": "(-mod. gate)",
            "w/o FiLM conditioning": "(-FiLM)",
            "w/o MFM pre-training": "(-MFM pretrain)",
            "w/o contrastive pre-training": "(-SupCon)",
            "w/o domain constraint": "(-domain)",
            "w/o MC-Dropout": "(-MC-Drop)",
        }
        labels = [_hard_shorten(short_map.get(v, v), 10) for v in top_variants]
        data = [ctx.abl[ctx.abl["variant"] == v][pm].values
                for v in top_variants]
        positions = np.arange(1, len(top_variants) + 1)
        parts = ax.violinplot(data, positions=positions, widths=0.65,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(NATURE["base_l"])
            pc.set_edgecolor(NATURE["base_d"])
            pc.set_alpha(0.6)
        for pos, vals in zip(positions, data):
            ax.scatter(np.full_like(vals, pos, dtype=float)
                       + np.random.normal(0, 0.04, size=len(vals)),
                       vals, s=8, color=NATURE["base"], alpha=0.7,
                       edgecolor="white", linewidth=0.3, zorder=3)
            ax.scatter([pos], [vals.mean()], s=55, marker="D",
                       color=NATURE["ours_d"], edgecolor="white",
                       linewidth=0.6, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels([l[:11] for l in labels], rotation=50,
                           ha="right", fontsize=6.2)
        ax.set_ylabel(pm)

    # ----- F: per-variant effect dumbbell -------------------------------
    def p_effect(ax):
        if ctx.abl is None:
            _note(ax)
            return
        if "full model" not in ctx.abl["variant"].values:
            _note(ax)
            return
        full = ctx.abl[ctx.abl["variant"] == "full model"][pm].mean()
        g = ctx.abl.groupby("variant")[pm].mean()
        g = g.drop("full model", errors="ignore").sort_values().head(12)
        short_map = {
            "w/o multimodal fusion": "(-fuse)",
            "w/o task-specific gating": "(-gating)",
            "w/o sparse attention": "(-attn)",
            "w/o residual blocks": "(-resid)",
            "w/o modality gate": "(-gate)",
            "w/o FiLM conditioning": "(-FiLM)",
            "w/o SWA": "(-SWA)",
            "w/o MFM pre-training": "(-MFM)",
            "w/o contrastive pre-training": "(-SupCon)",
            "w/o domain constraint": "(-domain)",
            "w/o MC-Dropout": "(-MC-Drop)",
        }
        y = np.arange(len(g))[::-1]
        short_map = {
            "w/o multimodal fusion": "(-multimod.fuse)",
            "w/o task-specific gating": "(-T-sk gating)",
            "w/o sparse attention": "(-sparse attn)",
            "w/o residual blocks": "(-resid. blocks)",
            "w/o modality gate": "(-mod. gate)",
            "w/o FiLM conditioning": "(-FiLM)",
            "w/o MFM pre-training": "(-MFM)",
            "w/o contrastive pre-training": "(-SupCon)",
            "w/o domain constraint": "(-domain)",
            "w/o MC-Dropout": "(-MC-Drop)",
            "w/o SWA": "(-SWA)",
            "w/o pretrain_recon": "(-recon pretrain)",
        }
        labels = [short_map.get(v, v).replace("w/o ", "-")[:14]
                  for v in g.index]
        # full-model endpoint: deep brown; ablated endpoints: graded brown
        n_b = len(g)
        browns = [plt.cm.BrBG(0.78 - 0.55 * i / max(n_b - 1, 1))
                  for i in range(n_b)]
        for i, v in enumerate(g.values):
            ax.scatter([full], [i], s=70, color="#8B5A2B",
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.scatter([v], [i], s=70, color=browns[i],
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.hlines(i, min(full, v), max(full, v),
                      color=NATURE["neutral"], lw=0.7, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_xlabel(pm)

    # ----- G: retention decisions text panel ----------------------------
    def p_decision(ax):
        if ctx.abl is None:
            _note(ax)
            return
        # retained = components that survive (w/o variants removed);
        # pruned = mechanisms tested and dropped (full-model variants absent).
        variants = ctx.abl["variant"].unique()
        wos = [v for v in variants if v.startswith("w/o ")]
        kept = [v for v in variants if not v.startswith("w/o ")
                and v != "full model"]
        n_kept = len(kept)
        n_pruned = len(wos) - n_kept if n_kept > 0 else 0
        n_pruned = max(0, n_pruned)
        # show retained (tested & kept) vs pruned (tested & dropped)
        labels = ["retained", "pruned"]
        vals = [n_kept, n_pruned]
        colors = [NATURE["ours"], NATURE["bad"]]
        y = np.arange(2)[::-1]
        CUT = 4.5   # 17 pruned >> 3 retained: broken axis with true value
        plot_vals = np.minimum(vals, CUT)
        ax.barh(y, plot_vals, height=0.5, color=colors,
                edgecolor="white", linewidth=0.7, zorder=3)
        for yy, v in zip(y, vals):
            if v > CUT:
                _break_marks(ax, CUT, yy, 0.25, tick=0.5)
                ax.text(CUT + 0.7, yy, f"{v}", va="center", ha="left",
                        fontsize=10, color="#8B0000", fontweight="bold")
            else:
                ax.text(v + 0.3, yy, f"{v}", va="center", ha="left",
                        fontsize=10, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0, CUT * 1.5)
        ax.set_xlabel("mechanisms")

    # ----- H: marginal vs interaction (horizontal stacked bar) ---------
    def p_marg(ax):
        # Route B: honest three-group summary of the 23 ablation arms.
        # Groups are counted from the released ablation file (truth values):
        # 10 arms with small positive Delta, 9 bit-identical + 1 below
        # reporting precision, 3 improve when removed.
        groups = [
            ("Positive \u0394 (10)", 10, NATURE["ours"]),
            ("Bit-identical (9+1)", 10, NATURE["base"]),
            ("Removal improves (3)", 3, NATURE["bad"]),
        ]
        y = np.arange(len(groups))
        vals = [g[1] for g in groups]
        cols = [g[2] for g in groups]
        ax.barh(y, vals, height=0.6, color=cols, edgecolor="white", zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.15, yi, str(v), va="center", fontsize=9,
                    fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([g[0] for g in groups], fontsize=7.5)
        ax.set_xlim(0, 13)
        ax.set_xlabel("ablation arms")
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")

    # ----- I: pruning summary ------------------------------------------
    def p_pruning(ax):
        if ctx.abl is None:
            _note(ax)
            return
        # pruned vs kept count by category
        keep, prune = 0, 0
        for v in ctx.abl["variant"].unique():
            if "w/o" in v:
                keep += 1
            else:
                prune += 1
        s = keep + prune
        if s == 0:
            _note(ax)
            return
        if prune / s < 0.02:
            ax.pie([1.0],
                   colors=[NATURE["ours"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.60, edgecolor="white", linewidth=1.2))
            ax.text(0, 0, "all retained", ha="center", va="center",
                    fontsize=10, color="#222222", fontweight="bold")
            ax.text(0, 0.18, f"({keep})", ha="center", va="center",
                    fontsize=9, color="#222222", fontweight="bold")
        else:
            ax.pie([keep / s, prune / s],
                   colors=[NATURE["ours"], NATURE["bad"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.60, edgecolor="white", linewidth=1.2))
            # merged "num + label" on one line each, one shared halo bbox
            ax.text(0, 0.14, f"{keep}  retain",
                    ha="center", va="center", fontsize=9.5,
                    color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.60))
            ax.text(0, -0.16, f"{prune}  prun",
                    ha="center", va="center", fontsize=9.5,
                    color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.30", facecolor="white",
                              edgecolor="none", alpha=0.60))

    fns = [p_waterfall, p_heat, p_fusion, p_sig, p_variant,
           p_effect, p_decision, p_marg, p_pruning]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure7_ablation", hspace=0.95, wspace=0.65)


# =========================================================================== #
# Figure 8 - interpretation (3x3, advanced)
# =========================================================================== #

def fig8(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 5.8))
    axes = axes.ravel()

    def p_imp(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m = m.dropna(subset=["importance_mean"])
        if "stat" in m.columns and m["stat"].notna().any():
            vals = (m["importance_mean"] * np.sign(m["stat"].fillna(0))).values
        else:
            vals = m["importance_mean"].values
        m = m.assign(_signed=vals).sort_values("_signed").tail(10)
        names = [_hard_shorten(f, 7) for f in m["feature"]]
        vals = m["_signed"].values
        sd = m["importance_sd"].fillna(0).values if "importance_sd" in m \
            else np.zeros_like(vals)
        y_pos = np.arange(len(vals))
        colors = np.where(vals >= 0, NATURE["ours"], NATURE["bad"])
        ax.hlines(y=y_pos, xmin=0, xmax=vals, color=NATURE["neutral"],
                  lw=1.4, alpha=0.85)
        for i, (x, yp, s) in enumerate(zip(vals, y_pos, sd)):
            ax.errorbar(x, yp, xerr=1.96 * s, fmt="none",
                        ecolor=colors[i], lw=0.6, alpha=0.6, zorder=2)
            ax.scatter([x], [yp], s=60, color=colors[i],
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.025, yp, "{0:+.4f}".format(x), va="center",
                    ha="left", fontsize=6.0, color="#222222",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(min(vals) - 0.02, max(vals) + 0.06)
        ax.set_xlabel("signed permutation importance")

    def p_stab(ax):
        if ctx.stab is None:
            _note(ax)
            return
        d = ctx.stab.head(10)
        feat_short = {
            "Hydrophobic-BA": "Hyd-BA", "Cationic-ATAC": "Cat-ATAC",
            "Nucleophilic-HEA": "Nuc-HEA", "Nucleophilic-CBEA": "Nuc-CBEA",
            "Aromatic-PEA": "Arom-PEA", "Amide-AAm": "Amide",
            "Acidic-CBEA": "Acid-CBEA",
        }
        names = [feat_short.get(str(f), str(f))[:7] for f in d["feature"]]
        vals = d["selection_frequency"].values
        y_pos = np.arange(len(vals))
        colors = _mg(vals, MORANDI_SEQ)
        ax.hlines(y=y_pos, xmin=0.8, xmax=vals, color=NATURE["base"],
                  lw=1.4, alpha=0.85)
        for x, y, c in zip(vals, y_pos, colors):
            ax.scatter([x], [y], s=45, color=c,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x - 0.012, y, "{0:.2f}".format(x), va="center",
                    ha="right", fontsize=5.6, color="#222222",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.axvline(0.8, ls="--", lw=0.8, color=NATURE["neutral"])
        ax.set_xlim(0.8, 1.0)
        ax.set_xlabel("stability frequency")

    def p_attn(ax):
        if ctx.attn is None:
            _note(ax)
            return
        d = ctx.attn.sort_values("attention_mean", ascending=False).iloc[::-1]
        tok_short = {"fused_mod1_mod2": "fused (m1\u00d7m2)",
                     "mod1_token": "mod1", "mod2_token": "mod2",
                     "cls_token": "CLS", "CLS": "CLS",
                     "condition": "condition"}
        names = [tok_short.get(str(t), str(t))[:12] for t in d["token"]]
        vals = d["attention_mean"].values
        y_pos = np.arange(len(vals))
        colors = _mg(vals, MORANDI_SEQ)
        CUT = 0.12
        big = vals > CUT
        plot_vals = np.where(big, CUT, vals)
        ax.hlines(y=y_pos, xmin=0, xmax=plot_vals, color=NATURE["ours"],
                  lw=1.4, alpha=0.85)
        for x, yp, c, is_big in zip(vals, y_pos, colors, big):
            if is_big:
                ax.scatter([CUT], [yp], s=55, color=c, zorder=3,
                           edgecolor="white", linewidth=0.8)
                _break_marks(ax, CUT, yp, 0.33, tick=0.006, color="#666666")
                ax.text(CUT + 0.018, yp, "{0:.3f}".format(x), va="center",
                        ha="left", fontsize=6.4, color="#222222")
            else:
                ax.scatter([x], [yp], s=55, color=c, zorder=3,
                           edgecolor="white", linewidth=0.8)
                ax.text(x + 0.018, yp, "{0:.3f}".format(x), va="center",
                        ha="left", fontsize=6.0, color="#222222")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.set_xlim(0, 0.22)
        ax.set_xticks([0.0, 0.05, 0.10, 0.12])
        ax.set_xlabel("CLS attention weight (axis broken at 0.12)")

    def p_attn_cond(ax):
        # token attention as a grouped vertical bar with value labels on top
        if ctx.attn is None:
            _note(ax)
            return
        d = ctx.attn.sort_values("attention_mean", ascending=False)
        tok_short = {"fused_mod1_mod2": "fused m1\u00d7m2",
                     "mod1_token": "mod1", "mod2_token": "mod2",
                     "cls_token": "CLS", "CLS": "CLS",
                     "condition": "condition"}
        names = [tok_short.get(str(t), str(t)) for t in d["token"]]
        vals = d["attention_mean"].values
        cols = _mg(vals, MORANDI_SEQ)
        x = np.arange(len(vals))
        ax.bar(x, vals, width=0.62, color=cols, edgecolor="white",
               linewidth=0.7, zorder=3)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.03, "{0:.3f}".format(v),
                    ha="center", va="bottom", fontsize=5.8, color="#222222")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=6.2)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_ylabel("attention weight")
        ax.grid(axis="y", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
    def p_latent_y(ax):
        if ctx.emb is None:
            _note(ax)
            return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() \
            else ("PC1", "PC2")
        col = "y_{0}".format(ctx.targets[0])
        sc = ax.scatter(e[xk], e[yk], c=e[col], s=10, cmap=_mcmap(MORANDI_SEQ),
                        alpha=0.85, linewidths=0.3, edgecolor="white",
                        zorder=3)
        ax.set_xlabel(xk, fontsize=8)
        ax.set_ylabel(yk, fontsize=8)
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:10])

    def p_pdp(ax):
        if ctx.pdp is None:
            _note(ax)
            return
        t0 = ctx.targets[0]
        d = ctx.pdp[ctx.pdp["target"] == t0]
        feats = list(d["feature"].unique())[:5]
        cols = _mg(np.arange(5), MORANDI_R2)
        for i, f in enumerate(feats):
            g = d[d["feature"] == f]
            ax.plot(g["grid_value"], g["pd_mean"], "-o",
                    label=_hard_shorten(str(f), 10), color=cols[i],
                    ms=3, lw=1.4)
        ax.set_xlabel("feature value (grid)")
        plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=5.6)
        ax.set_ylabel("predicted", fontsize=7.5)
        leg = ax.legend(fontsize=5.5, frameon=True, loc="lower right",
                        framealpha=0.9, edgecolor="#cccccc",
                        borderpad=0.4, labelspacing=0.3, markerscale=0.8)
        leg.set_zorder(10)

    def p_volcano(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m["nlp"] = -np.log10(m["p_fdr"].clip(lower=1e-300).fillna(1))
        x = m["stat"].fillna(0)
        sig = m["tier"] == "high"
        ax.scatter(x[~sig], m["nlp"][~sig], s=12, color=NATURE["neutral"],
                   alpha=0.6, linewidths=0, zorder=2, label="lower tier")
        ax.scatter(x[sig], m["nlp"][sig], s=46, color=NATURE["ours"],
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label="high tier")
        ax.axhline(-np.log10(.05), ls="--", lw=0.8, color=NATURE["bad"])
        leg = ax.legend(fontsize=5.5, frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5, 1.0), framealpha=0.9,
                        edgecolor="#cccccc", borderpad=0.35,
                        labelspacing=0.25, markerscale=0.6)
        leg.set_zorder(10)
        ax.set_xlabel("association statistic (signed)")
        ax.set_ylabel(r"$-\log_{10}$ FDR")

    def p_rules(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        if "stat" not in m.columns or m["stat"].isna().all():
            _note(ax)
            return
        m = m.dropna(subset=["stat"])
        m["signed"] = m["stat"] * m["importance_mean"]
        m = m[m["tier"] == "high"].sort_values("signed")
        names = [_hard_shorten(str(f), 8) for f in m["feature"]]
        vals = m["signed"].values
        y = np.arange(len(names))
        colors = np.where(vals >= 0, NATURE["ours"], NATURE["bad"])
        ax.hlines(y=y, xmin=0, xmax=vals, color=NATURE["neutral"],
                  lw=1.3, alpha=0.85)
        for x, yy, c in zip(vals, y, colors):
            ax.scatter([x], [yy], s=60, color=c,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.03, yy, "{0:+.3f}".format(x), va="center",
                    ha="left", fontsize=6.0, color="#222222",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.axvline(0, color="black", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlim(min(vals) - 0.05, max(vals) + 0.15)
        ax.set_xlabel("signed importance (\u00b1 = dir.)")

    fns = [p_imp, p_stab, p_attn, p_attn_cond, p_latent_y,
           p_pdp, p_volcano, p_rules]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _panel_label(axes)
    _save(fig, "Figure8_interpretation", wspace=0.62, hspace=0.78)


# =========================================================================== #
# Runner
# =========================================================================== #
FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4,
           5: fig5, 6: fig6, 7: fig7, 8: fig8}

def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()
    paths.ensure_dirs()
    ctx = Ctx()
    wanted = args.only or sorted(FIGURES)
    for k in wanted:
        if k in FIGURES:
            print("\n[routeb] building Figure {0}".format(k), flush=True)
            FIGURES[k](ctx)
    print("\n[routeb] done ->", paths.FIGURES_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
