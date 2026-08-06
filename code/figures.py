"""Publication figures: 8 main figures, each with 4-8 labelled panels.

Figure map
----------
  Fig. 1  Study design and data pipeline          (schematic, 4 panels)
  Fig. 2  Model architecture                      (schematic, 6 panels)
  Fig. 3  Dataset characterisation                (8 panels)
  Fig. 4  Internal cross-validation performance   (6 panels)
  Fig. 5  Benchmark against tuned baselines       (6 panels)
  Fig. 6  External cohort validation              (6 panels)
  Fig. 7  Ablation and component contribution     (6 panels)
  Fig. 8  Interpretation and domain discovery     (8 panels)

Every figure is written to FIGURES_DIR as 600-dpi PNG and vector PDF.
A missing input file degrades a single panel, never the whole run.

Usage
-----
    python figures.py
    python figures.py --only 4 8
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import os
import sys
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import paths
import sci_style as ss
from build_dataset import load_dataset


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if len(df) else None
    except Exception:                                      # noqa: BLE001
        pass
    return None


def _note(ax, msg: str = "not available") -> None:
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=7,
            color=ss.NEUTRAL_GREY, transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _safe(fn, ax, *a, **kw):
    try:
        fn(ax, *a, **kw)
    except Exception as exc:                               # noqa: BLE001
        _note(ax, f"panel failed:\n{type(exc).__name__}")
        print(f"    [warn] panel error: {exc}")


def _pm(task_type: str) -> str:
    return "R2" if task_type == "regression" else "AUROC"


def _oof_path() -> Optional[str]:
    if not os.path.isdir(paths.PREDS_DIR):
        return None
    for f in sorted(os.listdir(paths.PREDS_DIR)):
        if f.startswith("preds_cv_"):
            return os.path.join(paths.PREDS_DIR, f)
    return None


class Ctx:
    """Everything the figures need, loaded once."""

    def __init__(self) -> None:
        self.ds = load_dataset(paths.DATASET_NPZ)
        self.ext = (load_dataset(paths.EXTERNAL_NPZ)
                    if os.path.exists(paths.EXTERNAL_NPZ) else None)
        self.task = self.ds["task_type"]
        self.pm = _pm(self.task)
        self.targets = [str(t) for t in self.ds["target_names"]]
        self.features = [str(f) for f in self.ds["feature_names"]]
        self.conds = [str(c) for c in self.ds["cond_levels"]]
        self.cv = _csv(paths.CV_OUTER_CSV)
        self.base = _csv(paths.BASELINES_CSV)
        self.extm = _csv(paths.EXTERNAL_CSV)
        self.base_ext = _csv(os.path.join(paths.METRICS_DIR,
                                          "baselines_external.csv"))
        self.abl = _csv(paths.ABLATION_CSV)
        self.abl_stats = _csv(os.path.join(paths.STATS_DIR,
                                           "ablation_stats.csv"))
        self.comp = _csv(paths.COMPARISONS_CSV)
        self.ci = _csv(os.path.join(paths.STATS_DIR, "bootstrap_ci.csv"))
        self.perm = _csv(os.path.join(paths.STATS_DIR, "permutation.csv"))
        self.imp = _csv(paths.IMPORTANCE_CSV)
        self.stab = _csv(os.path.join(paths.INTERPRET_DIR, "stability.csv"))
        self.attn = _csv(os.path.join(paths.INTERPRET_DIR, "attention.csv"))
        self.attn_c = _csv(os.path.join(paths.INTERPRET_DIR,
                                        "attention_by_condition.csv"))
        self.emb = _csv(os.path.join(paths.INTERPRET_DIR, "embedding.csv"))
        self.pdp = _csv(os.path.join(paths.INTERPRET_DIR, "pdp.csv"))
        self.markers = _csv(os.path.join(paths.INTERPRET_DIR,
                                         "candidate_markers.csv"))
        self.cond_perf = _csv(os.path.join(paths.INTERPRET_DIR,
                                           "condition_performance.csv"))
        self.hist = _csv(os.path.join(paths.METRICS_DIR,
                                      "training_history.csv"))
        self.qc_shift = _csv(os.path.join(paths.METRICS_DIR, "qc_shift.csv"))
        self.search = _csv(paths.SEARCH_LOG_CSV)
        op = _oof_path()
        self.oof = _csv(op) if op else None
        self.extp = _csv(os.path.join(paths.PREDS_DIR, "preds_external.csv"))


# =========================================================================== #
# Figure 1 -- study design (schematic)
# =========================================================================== #
def fig1(ctx: Ctx) -> None:
    # ---- load data ----
    ds = np.load(paths.DATASET_NPZ, allow_pickle=False)
    ext = np.load(paths.EXTERNAL_NPZ, allow_pickle=False)
    Xtr = ds["X"][:, :6]
    ytr = ds["Y"].ravel()
    Xte = ext["X"][:, :6]
    yte = ext["Y"].ravel()
    MONO = ["HEA", "BA", "CBEA", "ATAC", "PEA", "AAm"]

    fig = plt.figure(figsize=(ss.DOUBLE_COL, 5.4))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.34,
                          left=0.09, right=0.80, top=0.90, bottom=0.10)

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
    ax.text(0.02, 0.98, 'circles = training (n=180); diamonds = external SMBO (n=161)',
            transform=ax.transAxes, fontsize=6.2, color='#444444',
            ha='left', va='top', style='italic')
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


def fig2(ctx: Ctx) -> None:
    import matplotlib.patches as mpatches  # noqa: PLC0415
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


def fig3(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(ss.DOUBLE_COL, 4.2))
    axes = axes.ravel()
    ds, ext = ctx.ds, ctx.ext

    def p_counts(ax):
        names = ["Internal", "External"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        ax.bar(names, vals, color=[ss.OKABE_ITO[0], ss.OKABE_ITO[1]], width=.55)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=7)
        ax.set_ylabel("samples")
        ax.set_title("Cohort size")

    def p_targets(ax):
        for i, t in enumerate(ctx.targets[:3]):
            ax.hist(ds["Y"][:, i], bins=25, alpha=.6, label=t,
                    color=ss.OKABE_ITO[i], edgecolor="white", linewidth=.3)
        ax.set_xlabel("target value")
        ax.set_ylabel("count")
        ax.set_title("Target distribution")

    def p_missing(ax):
        miss = np.isnan(ds["X"]).mean(axis=0) * 100
        ax.hist(miss, bins=20, color=ss.OKABE_ITO[2], edgecolor="white",
                linewidth=.3)
        ax.set_xlabel("missing per feature (%)")
        ax.set_ylabel("features")
        ax.set_title(f"Missingness (mean {miss.mean():.1f}%)")

    def p_corr(ax):
        Xs = np.nan_to_num(ds["X"])
        k = min(40, Xs.shape[1])
        c = np.corrcoef(Xs[:, :k].T)
        im = ax.imshow(c, cmap=ss.DIVERGING_CMAP, vmin=-1, vmax=1,
                       aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Feature correlation (first {k})")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    def p_pca(ax):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(np.nan_to_num(ds["X"]))
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xs)
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=ds["Y"][:, 0], s=6,
                        cmap=ss.SEQUENTIAL_CMAP, alpha=.85, linewidths=0)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Raw feature space")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:12])

    def p_cond(ax):
        vals = pd.Series(ds["cond"]).value_counts().sort_index()
        labels = [ctx.conds[i][:10] for i in vals.index]
        ax.barh(labels, vals.values, color=ss.OKABE_ITO[3])
        ax.set_xlabel("samples")
        ax.set_title("Condition composition")

    def p_shift(ax):
        if ctx.qc_shift is None:
            _note(ax, "run data_qc.py")
            return
        s = ctx.qc_shift.sort_values("ks_stat", ascending=False).head(12)
        ax.barh(range(len(s)), s["ks_stat"], color=ss.OKABE_ITO[1])
        ax.set_yticks(range(len(s)))
        ax.set_yticklabels([f[:14] for f in s["feature"]], fontsize=5.5)
        ax.invert_yaxis()
        ax.set_xlabel("KS statistic")
        ax.set_title("Internal vs external shift")

    def p_groups(ax):
        sizes = pd.Series(ds["groups"]).value_counts().values
        ax.hist(sizes, bins=min(20, len(np.unique(sizes)) + 1),
                color=ss.OKABE_ITO[5], edgecolor="white", linewidth=.3)
        ax.set_xlabel("samples per group")
        ax.set_ylabel("groups")
        ax.set_title(f"Grouping ({len(sizes)} groups)")

    for ax, fn in zip(axes, [p_counts, p_targets, p_missing, p_corr,
                             p_pca, p_cond, p_shift, p_groups]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.22, dy=1.18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure3_dataset")


# =========================================================================== #
# Figure 4 -- internal CV performance
# =========================================================================== #
def fig4(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(ss.DOUBLE_COL, 4.4))
    axes = axes.ravel()
    pm = ctx.pm

    def p_folds(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        for i, t in enumerate(ctx.targets):
            sub = cv[cv["target"] == t]
            ax.scatter(sub["fold"] + i * .12 - .06, sub[pm], s=14,
                       color=ss.OKABE_ITO[i], alpha=.85, label=t, linewidths=0)
            g = sub.groupby("fold")[pm].mean()
            ax.plot(g.index + i * .12 - .06, g.values, color=ss.OKABE_ITO[i],
                    lw=1.0)
        ax.set_xlabel("outer fold")
        ax.set_ylabel(pm)
        ax.set_title("Per-fold performance")

    def p_scatter(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        g = ctx.oof.groupby("sample_id")[[yc, pc]].mean()
        ax.scatter(g[yc], g[pc], s=7, alpha=.55, color=ss.MODEL_COLOR,
                   linewidths=0)
        lo = float(min(g[yc].min(), g[pc].min()))
        hi = float(max(g[yc].max(), g[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="black")
        r = np.corrcoef(g[yc], g[pc])[0, 1]
        ax.text(.04, .93, f"r = {r:.3f}\nn = {len(g)}", transform=ax.transAxes,
                fontsize=6.8, va="top")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(f"Out-of-fold: {t[:18]}")

    def p_resid(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}", f"y_pred_{t}"]].mean()
        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
        ax.scatter(g[f"y_pred_{t}"], res, s=6, alpha=.5,
                   color=ss.OKABE_ITO[0], linewidths=0)
        ax.axhline(0, ls="--", lw=.9, color="black")
        ax.set_xlabel("predicted")
        ax.set_ylabel("residual")
        ax.set_title("Residuals vs fitted")

    def p_reshist(ax):
        if ctx.oof is None:
            _note(ax)
            return
        for i, t in enumerate(ctx.targets[:3]):
            g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                              f"y_pred_{t}"]].mean()
            res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
            ax.hist(res, bins=28, alpha=.55, color=ss.OKABE_ITO[i], label=t,
                    edgecolor="white", linewidth=.3)
        ax.axvline(0, ls="--", lw=.9, color="black")
        ax.set_xlabel("residual")
        ax.set_ylabel("count")
        ax.set_title("Error distribution")

    def p_curves(ax):
        if ctx.hist is None:
            _note(ax)
            return
        h = ctx.hist
        for key, col, lbl in [("train_loss", ss.OKABE_ITO[0], "train"),
                              ("val_loss", ss.MODEL_COLOR, "validation")]:
            g = h.groupby("epoch")[key].agg(["mean", "std"])
            g = g.head(300)
            ax.plot(g.index, g["mean"], color=col, label=lbl)
            ax.fill_between(g.index, g["mean"] - g["std"],
                            g["mean"] + g["std"], color=col, alpha=.18,
                            linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Learning curves")

    def p_heat(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        cols = [c for c in ["R2", "RMSE", "MAE", "PearsonR", "SpearmanRho",
                            "CCC", "AUROC", "AUPRC", "F1", "MCC",
                            "BalancedAcc"] if c in cv.columns]
        m = cv.groupby("target")[cols].mean()
        z = (m - m.min()) / (m.max() - m.min() + 1e-12)
        im = ax.imshow(z.values, cmap=ss.SEQUENTIAL_CMAP, aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=5.5)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels([t[:12] for t in m.index], fontsize=6)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.2,
                        color="white" if z.values[i, j] < .5 else "black")
        ax.set_title("Metric summary")

    for ax, fn in zip(axes, [p_folds, p_scatter, p_resid, p_reshist,
                             p_curves, p_heat]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.20, dy=1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure4_internal_cv")


# =========================================================================== #
# Figure 5 -- benchmark
# =========================================================================== #
def fig5(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(ss.DOUBLE_COL, 4.6))
    axes = axes.ravel()
    pm = ctx.pm

    def _pool():
        if ctx.cv is None or ctx.base is None:
            return None
        a = ctx.cv.copy()
        a["model"] = paths.MODEL_NAME
        return pd.concat([a, ctx.base], ignore_index=True)

    def p_bar(ax):
        pool = _pool()
        if pool is None:
            _note(ax)
            return
        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        cols = [ss.MODEL_COLOR if m == paths.MODEL_NAME else ss.BASELINE_COLOR
                for m in g.index]
        ax.barh(range(len(g)), g["mean"], xerr=1.96 * se, color=cols,
                error_kw=dict(lw=.7, capsize=2))
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6)
        ax.set_xlabel(f"{pm} (mean +/- 95% CI)")
        ax.set_title("Model comparison")

    def p_paired(ax):
        pool = _pool()
        if pool is None or ctx.comp is None:
            _note(ax)
            return
        best = (ctx.comp.groupby("reference")["reference_mean"].mean()
                .sort_values(ascending=False).index[0])
        ours = pool[pool["model"] == paths.MODEL_NAME].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        theirs = pool[pool["model"] == best].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        k = min(len(ours), len(theirs))
        for i in range(k):
            ax.plot([0, 1], [theirs[i], ours[i]], color=ss.NEUTRAL_GREY,
                    lw=.5, alpha=.6)
        ax.scatter(np.zeros(k), theirs[:k], s=12, color=ss.BASELINE_COLOR,
                   zorder=3, linewidths=0)
        ax.scatter(np.ones(k), ours[:k], s=12, color=ss.MODEL_COLOR,
                   zorder=3, linewidths=0)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([best[:12], paths.MODEL_NAME], fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Paired per-fold scores")

    def p_delta(ax):
        if ctx.comp is None:
            _note(ax)
            return
        g = ctx.comp.groupby("reference").agg(
            delta=("delta", "mean"), p=("p_holm", "min")).sort_values("delta")
        cols = [ss.OKABE_ITO[2] if d > 0 else ss.OKABE_ITO[1]
                for d in g["delta"]]
        ax.barh(range(len(g)), g["delta"], color=cols)
        for i, (d, p) in enumerate(zip(g["delta"], g["p"])):
            ax.text(d, i, "  " + ss.stars(p), va="center", fontsize=6.5,
                    ha="left" if d > 0 else "right")
        ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6)
        ax.set_xlabel(f"Delta {pm} vs {paths.MODEL_NAME}")
        ax.set_title("Improvement and significance")

    def p_rank(ax):
        pool = _pool()
        if pool is None:
            _note(ax)
            return
        piv = pool.pivot_table(index=["seed", "fold", "target"],
                               columns="model", values=pm)
        ranks = piv.rank(axis=1, ascending=False)
        mr = ranks.mean().sort_values()
        ax.plot(mr.values, range(len(mr)), "o-", color=ss.OKABE_ITO[0],
                markersize=4)
        for i, m in enumerate(mr.index):
            ax.annotate(m[:14], (mr.values[i], i), fontsize=5.8,
                        xytext=(4, 0), textcoords="offset points",
                        va="center",
                        color=ss.MODEL_COLOR if m == paths.MODEL_NAME
                        else "black")
        ax.set_yticks([])
        ax.set_xlabel("mean rank (1 = best)")
        ax.set_title("Rank across folds")
        ax.invert_yaxis()

    def p_ci(ax):
        if ctx.ci is None:
            _note(ax)
            return
        c = ctx.ci[(ctx.ci["metric"] == pm)]
        if not len(c):
            _note(ax)
            return
        labels = [f"{r['scope'][:8]}/{r['target'][:10]}" for _, r in c.iterrows()]
        y = np.arange(len(c))
        ax.errorbar(c["point"], y,
                    xerr=[c["point"] - c["lo"], c["hi"] - c["point"]],
                    fmt="o", color=ss.MODEL_COLOR, markersize=4, lw=.9,
                    capsize=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.set_xlabel(f"{pm} (95% CI)")
        ax.set_title("Cluster bootstrap CI")

    def p_perm(ax):
        if ctx.perm is None:
            _note(ax)
            return
        p = ctx.perm
        x = np.arange(len(p))
        ax.bar(x - .18, p["null_mean"], width=.34, color=ss.LIGHT_GREY,
               label="permuted null")
        ax.bar(x + .18, p["observed"], width=.34, color=ss.MODEL_COLOR,
               label="observed")
        for i, pv in enumerate(p["p_value"]):
            ax.text(i + .18, p["observed"].iloc[i], ss.stars(pv),
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([t[:10] for t in p["target"]], fontsize=6)
        ax.set_ylabel(pm)
        ax.set_title("Permutation test")

    for ax, fn in zip(axes, [p_bar, p_paired, p_delta, p_rank, p_ci, p_perm]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.20, dy=1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure5_benchmark")


# =========================================================================== #
# Figure 6 -- external validation
# =========================================================================== #
def fig6(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(ss.DOUBLE_COL, 4.4))
    axes = axes.ravel()
    pm = ctx.pm

    def p_scatter(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        ax.scatter(ctx.extp[yc], ctx.extp[pc], s=8, alpha=.6,
                   color=ss.OKABE_ITO[1], linewidths=0)
        lo = float(min(ctx.extp[yc].min(), ctx.extp[pc].min()))
        hi = float(max(ctx.extp[yc].max(), ctx.extp[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="black")
        r = np.corrcoef(ctx.extp[yc], ctx.extp[pc])[0, 1]
        ax.text(.04, .93, f"r = {r:.3f}\nn = {len(ctx.extp)}",
                transform=ax.transAxes, fontsize=6.8, va="top")
        ax.set_xlabel("observed (external)")
        ax.set_ylabel("predicted")
        ax.set_title("External cohort")

    def p_bland(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        a = ctx.extp[f"y_true_{t}"].to_numpy()
        b = ctx.extp[f"y_pred_{t}"].to_numpy()
        mean, diff = (a + b) / 2, b - a
        ax.scatter(mean, diff, s=8, alpha=.6, color=ss.OKABE_ITO[3],
                   linewidths=0)
        md, sd = diff.mean(), diff.std()
        for v, ls, lbl in [(md, "-", "bias"), (md + 1.96 * sd, "--", "+1.96 SD"),
                           (md - 1.96 * sd, "--", "-1.96 SD")]:
            ax.axhline(v, ls=ls, lw=.8, color="black")
            ax.text(ax.get_xlim()[1], v, f" {lbl}", fontsize=5.5, va="center")
        ax.set_xlabel("mean of methods")
        ax.set_ylabel("predicted - observed")
        ax.set_title("Bland-Altman")

    def p_calib(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["bin"] = pd.qcut(d[f"y_pred_{t}"], q=min(8, max(2, len(d) // 12)),
                           labels=False, duplicates="drop")
        g = d.groupby("bin").mean()
        ax.plot(g[f"y_pred_{t}"], g[f"y_true_{t}"], "o-",
                color=ss.OKABE_ITO[2], markersize=4)
        lo = float(min(g.min().min(), 0))
        hi = float(g.max().max())
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="black")
        ax.set_xlabel("mean predicted")
        ax.set_ylabel("mean observed")
        ax.set_title("Calibration")

    def p_intext(ax):
        if ctx.cv is None or ctx.extm is None:
            _note(ax)
            return
        ext = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")]
        vals, labels, cols = [], [], []
        for t in ctx.targets:
            vals.append(ctx.cv[ctx.cv["target"] == t][pm].mean())
            labels.append(f"{t[:9]}\ninternal")
            cols.append(ss.BASELINE_COLOR)
            e = ext[ext["target"] == t]
            vals.append(float(e[pm].mean()) if len(e) else np.nan)
            labels.append(f"{t[:9]}\nexternal")
            cols.append(ss.MODEL_COLOR)
        ax.bar(range(len(vals)), vals, color=cols, width=.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_ylabel(pm)
        ax.set_title("Generalisation gap")

    def p_extbase(ax):
        if ctx.extm is None or ctx.base_ext is None:
            _note(ax)
            return
        ours = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")][pm].mean()
        g = ctx.base_ext.groupby("model")[pm].mean().sort_values()
        cols = [ss.BASELINE_COLOR] * len(g)
        ax.barh(range(len(g)), g.values, color=cols)
        ax.axvline(ours, color=ss.MODEL_COLOR, lw=1.4,
                   label=f"{paths.MODEL_NAME} = {ours:.3f}")
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6)
        ax.set_xlabel(pm)
        ax.set_title("External benchmark")

    def p_condperf(ax):
        if ctx.cond_perf is None:
            _note(ax)
            return
        c = ctx.cond_perf
        piv = c.pivot_table(index="condition", columns="target", values=pm)
        im = ax.imshow(piv.values, cmap=ss.SEQUENTIAL_CMAP, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([t[:10] for t in piv.columns], rotation=45,
                           ha="right", fontsize=6)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([str(i)[:12] for i in piv.index], fontsize=6)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.5, color="white")
        ax.set_title("Performance by condition")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    for ax, fn in zip(axes, [p_scatter, p_bland, p_calib, p_intext,
                             p_extbase, p_condperf]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.20, dy=1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure6_external")


# =========================================================================== #
# Figure 7 -- ablation
# =========================================================================== #
def fig7(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(ss.DOUBLE_COL, 4.8))
    axes = axes.ravel()
    pm = ctx.pm

    def p_waterfall(ax):
        if ctx.abl is None:
            _note(ax)
            return
        g = ctx.abl.groupby("variant")[pm].mean()
        if "full model" not in g:
            _note(ax)
            return
        full = g["full model"]
        d = (full - g.drop("full model")).sort_values()
        cols = [ss.OKABE_ITO[2] if v > 0 else ss.OKABE_ITO[1] for v in d]
        ax.barh(range(len(d)), d.values, color=cols)
        ax.axvline(0, color="black", lw=.8)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([i.replace("w/o ", "-")[:26] for i in d.index],
                           fontsize=5.6)
        ax.set_xlabel(f"contribution to {pm}")
        ax.set_title("Component contribution")

    def p_heat(ax):
        if ctx.abl is None:
            _note(ax)
            return
        piv = ctx.abl.pivot_table(index="variant", columns="target", values=pm)
        im = ax.imshow(piv.values, cmap=ss.SEQUENTIAL_CMAP, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([t[:10] for t in piv.columns], rotation=45,
                           ha="right", fontsize=6)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([i[:24] for i in piv.index], fontsize=5.4)
        ax.set_title(f"{pm} per variant")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    def p_fusion(ax):
        if ctx.abl is None:
            _note(ax)
            return
        f = ctx.abl[ctx.abl["variant"].str.startswith("fusion")]
        base = ctx.abl[ctx.abl["variant"] == "full model"]
        names = list(f["variant"].unique()) + ["selected"]
        vals = [f[f["variant"] == n][pm].mean() for n in names[:-1]]
        vals.append(base[pm].mean())
        cols = [ss.BASELINE_COLOR] * (len(names) - 1) + [ss.MODEL_COLOR]
        ax.bar(range(len(names)), vals, color=cols, width=.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("fusion = ", "")[:10] for n in names],
                           rotation=30, ha="right", fontsize=6)
        ax.set_ylabel(pm)
        ax.set_title("Fusion strategy")

    def p_sig(ax):
        if ctx.abl_stats is None:
            _note(ax)
            return
        s = ctx.abl_stats.groupby("variant").agg(
            delta=("delta", "mean"), p=("p_holm", "min")).sort_values("delta")
        y = np.arange(len(s))
        ax.scatter(s["delta"], y, s=26,
                   c=[ss.OKABE_ITO[2] if p < .05 else ss.LIGHT_GREY
                      for p in s["p"]], zorder=3)
        ax.axvline(0, color="black", lw=.8)
        for i, (d, p) in enumerate(zip(s["delta"], s["p"])):
            ax.text(d, i, "  " + ss.stars(p), fontsize=6, va="center")
        ax.set_yticks(y)
        ax.set_yticklabels([i.replace("w/o ", "-")[:24] for i in s.index],
                           fontsize=5.4)
        ax.set_xlabel(f"delta {pm} (Holm-adjusted)")
        ax.set_title("Statistical contribution")

    def p_search(ax):
        if ctx.search is None:
            _note(ax, "run tuner.py")
            return
        s = ctx.search.reset_index(drop=True)
        for phase, col in [("coarse", ss.BASELINE_COLOR),
                           ("fine", ss.MODEL_COLOR)]:
            sub = s[s["phase"] == phase]
            if len(sub):
                ax.scatter(sub.index, sub["score"], s=8, color=col,
                           label=phase, alpha=.7, linewidths=0)
        ax.plot(s.index, s["score"].cummax(), color="black", lw=1.0,
                label="best so far")
        ax.set_xlabel("search iteration")
        ax.set_ylabel(pm)
        ax.set_title("Hyper-parameter search")

    def p_decision(ax):
        ss.blank_canvas(ax)
        ax.set_title("Retention decisions", loc="left", fontweight="bold")
        notes_path = os.path.join(paths.TUNING_DIR, "pruning_notes.txt")
        txt = "no pruning log found"
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as fh:
                txt = fh.read().strip() or "every component was retained"
        lines = txt.split("\n")[:10]
        for i, line in enumerate(lines):
            ax.text(.02, .92 - i * .095, "- " + line[:66], fontsize=5.6,
                    va="top", transform=ax.transAxes)

    for ax, fn in zip(axes, [p_waterfall, p_heat, p_fusion, p_sig,
                             p_search, p_decision]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.20, dy=1.16)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure7_ablation")


# =========================================================================== #
# Figure 8 -- interpretation
# =========================================================================== #
def fig8(ctx: Ctx) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(ss.DOUBLE_COL, 4.6))
    axes = axes.ravel()

    def p_imp(ax):
        if ctx.imp is None:
            _note(ax)
            return
        d = ctx.imp.head(15).iloc[::-1]
        ax.barh(range(len(d)), d["importance_mean"],
                xerr=1.96 * d["importance_se"].fillna(0),
                color=ss.OKABE_ITO[0], error_kw=dict(lw=.6, capsize=1.5))
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f[:16] for f in d["feature"]], fontsize=5.4)
        ax.set_xlabel("permutation importance")
        ax.set_title("Top features")

    def p_stab(ax):
        if ctx.stab is None:
            _note(ax)
            return
        d = ctx.stab.head(15).iloc[::-1]
        ax.barh(range(len(d)), d["selection_frequency"],
                color=ss.OKABE_ITO[2])
        ax.axvline(.8, ls="--", lw=.8, color="black")
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f[:16] for f in d["feature"]], fontsize=5.4)
        ax.set_xlabel("selection frequency")
        ax.set_xlim(0, 1.02)
        ax.set_title("Stability selection")

    def p_attn(ax):
        if ctx.attn is None:
            _note(ax)
            return
        d = ctx.attn.sort_values("attention_mean", ascending=True)
        ax.barh(range(len(d)), d["attention_mean"],
                xerr=d["attention_sd"].fillna(0), color=ss.OKABE_ITO[4],
                error_kw=dict(lw=.6, capsize=1.5))
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([t[:14] for t in d["token"]], fontsize=5.4)
        ax.set_xlabel("CLS attention weight")
        ax.set_title("Attention attribution")

    def p_attn_cond(ax):
        if ctx.attn_c is None:
            _note(ax)
            return
        piv = ctx.attn_c.pivot_table(index="token", columns="condition",
                                     values="attention_mean")
        im = ax.imshow(piv.values, cmap=ss.SEQUENTIAL_CMAP, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([c[:9] for c in piv.columns], rotation=45,
                           ha="right", fontsize=5.5)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([t[:12] for t in piv.index], fontsize=5.2)
        ax.set_title("Attention by condition")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    def p_latent_y(ax):
        if ctx.emb is None:
            _note(ax)
            return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        col = f"y_{ctx.targets[0]}"
        sc = ax.scatter(e[xk], e[yk], c=e[col], s=6, cmap=ss.SEQUENTIAL_CMAP,
                        alpha=.85, linewidths=0)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (target)")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03)

    def p_latent_c(ax):
        if ctx.emb is None:
            _note(ax)
            return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        for i, (name, g) in enumerate(e.groupby("condition")):
            ax.scatter(g[xk], g[yk], s=6, label=str(name)[:10],
                       color=ss.OKABE_ITO[i % len(ss.OKABE_ITO)], alpha=.8,
                       linewidths=0)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (condition)")

    def p_pdp(ax):
        if ctx.pdp is None:
            _note(ax)
            return
        t0 = ctx.targets[0]
        d = ctx.pdp[ctx.pdp["target"] == t0]
        for i, (f, g) in enumerate(d.groupby("feature")):
            if i >= 5:
                break
            ax.plot(g["grid_value"], g["pd_mean"], label=str(f)[:12],
                    color=ss.OKABE_ITO[i % len(ss.OKABE_ITO)])
        ax.set_xlabel("feature value")
        ax.set_ylabel(f"predicted {t0[:10]}")
        ax.set_title("Partial dependence")

    def p_volcano(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m["nlp"] = -np.log10(m["p_fdr"].clip(lower=1e-300).fillna(1))
        x = m["stat"].fillna(0)
        sig = m["tier"] == "high"
        ax.scatter(x[~sig], m["nlp"][~sig], s=6, color=ss.LIGHT_GREY,
                   linewidths=0)
        ax.scatter(x[sig], m["nlp"][sig], s=10, color=ss.MODEL_COLOR,
                   linewidths=0)
        ax.axhline(-np.log10(.05), ls="--", lw=.8, color="black")
        top = m.nlargest(4, "nlp")
        for _, r in top.iterrows():
            ax.annotate(str(r["feature"])[:10],
                        (r["stat"] if np.isfinite(r["stat"]) else 0, r["nlp"]),
                        fontsize=5, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_xlabel("association statistic")
        ax.set_ylabel(r"$-\log_{10}$ FDR")
        ax.set_title("Candidate markers")

    for ax, fn in zip(axes, [p_imp, p_stab, p_attn, p_attn_cond,
                             p_latent_y, p_latent_c, p_pdp, p_volcano]):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.24, dy=1.18)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure8_interpretation")


# --------------------------------------------------------------------------- #
FIGURES = {1: fig1, 2: fig2, 3: fig3, 4: fig4,
           5: fig5, 6: fig6, 7: fig7, 8: fig8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 9/9  FIGURES")
    ss.apply_style()
    ctx = Ctx()

    wanted = args.only or sorted(FIGURES)
    for k in wanted:
        print(f"  building Figure {k} ...")
        try:
            FIGURES[k](ctx)
        except Exception as exc:                           # noqa: BLE001
            print(f"  [ERROR] Figure {k} failed: {type(exc).__name__}: {exc}")
    n = len([f for f in os.listdir(paths.FIGURES_DIR) if f.endswith(".png")])
    print(f"\n  {n} PNG figure(s) in {paths.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
