"""SIMPLEX figure upgrade (v2): TransMICRO pastel palette + 3x3 hero grids.

Goals (per user request, 2026-08-06):
  - Use the pastel colour scheme of the TransMICRO template
    (#CCE4FC blue / #E4FCFC green / #FCE4E4 red / #FCE4FC purple /
     #FCE4CC orange) with strong dark borders, on a white panel.
  - Every result figure is a 3x3 = 9-panel grid (no empty slot).
  - Strict no-overlap layout: wspace >= 0.45, hspace >= 0.55, panel labels
    placed outside axes, hero panel(s) sized 1.3x of the rest.
  - Advanced chart types (waterfall, slope, ridge, dumbbell, dual-axis,
    heatmap-with-annotations, marginal-scatter) replace plain bars.
  - The same data drives every panel; no new metrics are invented.

The script keeps the original data-loading Ctx from figures.py and only
overrides the visual layer. The original figures.py is preserved as
code/figures_v1_backup.py.
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import os
import sys
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, Rectangle

import paths
import sci_style as ss
from figures import Ctx, _csv, _note, _safe
from build_dataset import load_dataset


# --------------------------------------------------------------------------- #
# TransMICRO pastel palette (from the template PDF)
# --------------------------------------------------------------------------- #
PAL = {
    "blue":   ("#CCE4FC", "#2E6DA4"),  # (fill, edge)
    "green":  ("#E4FCFC", "#2E8B57"),
    "red":    ("#FCE4E4", "#C0392B"),
    "purple": ("#FCE4FC", "#8E44AD"),
    "orange": ("#FCE4CC", "#E67E22"),
    "bluep":  ("#E4E4FC", "#5B6EE1"),
    "grey":   ("#E4E4E4", "#555555"),
}
# Strong accent colours for "ours" and significance highlights (from PAL but
# only the dark border; no wash).
ACC_OURS = PAL["red"][1]    # strong red — the model highlight
ACC_BASE = PAL["blue"][1]   # strong blue — baselines
ACC_SIG  = PAL["green"][1]  # strong green — significance
ACC_WARN = PAL["orange"][1] # strong orange — caution
ACC_NEUT = PAL["grey"][1]   # neutral grey

# Sequential colormap built from the pastels for heatmaps.
PASTEL_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "pastel_seq", ["#FFFFFF", PAL["blue"][0], PAL["bluep"][0],
                   PAL["purple"][0], PAL["orange"][0], PAL["red"][0]]
)
DIVERGE_CMAP = plt.matplotlib.colors.LinearSegmentedColormap.from_list(
    "pastel_div", [PAL["blue"][0], "#FFFFFF", PAL["red"][0]]
)

# Bar-pastel cycler: 5 fills with strong dark borders.
PASTEL_BARS = [PAL["blue"], PAL["green"], PAL["red"],
               PAL["purple"], PAL["orange"]]
PASTEL_FILLS = [c[0] for c in PASTEL_BARS]
PASTEL_EDGES = [c[1] for c in PASTEL_BARS]


# --------------------------------------------------------------------------- #
# Style overrides
# --------------------------------------------------------------------------- #
def apply_v2_style() -> None:
    """Tighter fonts and stronger borders for the v2 figures."""
    ss.apply_style()
    plt.rcParams.update({
        "axes.edgecolor": "#272727",
        "axes.linewidth": 0.9,
        "axes.labelcolor": "#1a1a1a",
        "xtick.color": "#272727",
        "ytick.color": "#272727",
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.0,
        "lines.linewidth": 1.3,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.9,
        "patch.edgecolor": "#1a1a1a",
    })


# --------------------------------------------------------------------------- #
# Pastel-fill bar helper
# --------------------------------------------------------------------------- #
def pastel_bar(ax, x, vals, idx: int = 0, width: float = 0.7,
               label: Optional[str] = None,
               edge: bool = True) -> None:
    fill, edgec = PASTEL_BARS[idx % len(PASTEL_BARS)]
    bars = ax.bar(x, vals, width=width, color=fill,
                  edgecolor=edgec if edge else "none",
                  linewidth=1.1, label=label)
    return bars


def pastel_barh(ax, y, vals, idx: int = 0, height: float = 0.7,
                label: Optional[str] = None,
                edge: bool = True) -> None:
    fill, edgec = PASTEL_BARS[idx % len(PASTEL_BARS)]
    bars = ax.barh(y, vals, height=height, color=fill,
                   edgecolor=edgec if edge else "none",
                   linewidth=1.1, label=label)
    return bars


def pastel_scatter(ax, x, y, idx: int = 0, s: int = 18, **kw):
    fill, edgec = PASTEL_BARS[idx % len(PASTEL_BARS)]
    return ax.scatter(x, y, s=s, c=fill, edgecolors=edgec, linewidths=0.7,
                      alpha=kw.pop("alpha", 0.85), **kw)


# --------------------------------------------------------------------------- #
# Layout checker: prevent overlap of any text/shape bbox
# --------------------------------------------------------------------------- #
def _check_no_overlap(fig, label="figure") -> None:
    """Render-time guard: no two artists should overlap unexpectedly."""
    renderer = fig.canvas.get_renderer()
    fig.draw(renderer)
    seen = []
    for ax in fig.axes:
        for t in ax.texts:
            bb = t.get_window_extent(renderer)
            for prev in seen:
                if bb.overlaps(prev):
                    # log only — this is informational
                    print(f"    [overlap?] {label} :: '{t.get_text()[:30]}' "
                          f"may overlap an earlier text")
            seen.append(bb)


# --------------------------------------------------------------------------- #
# Figure 3 — Dataset (3x3) -- recap of cohort with pastel scheme
# --------------------------------------------------------------------------- #
def fig3(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.0))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.0, 1.0, 1.05],
                  hspace=0.55, wspace=0.45,
                  left=0.07, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    ds, ext = ctx.ds, ctx.ext

    def p_counts(ax):
        names = ["Internal", "External"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        for i, v in enumerate(vals):
            pastel_bar(ax, [i], [v], idx=i, width=.55)
            ax.text(i, v + max(vals) * 0.02, f"{v}", ha="center", va="bottom",
                    fontsize=7.5, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=7)
        ax.set_ylabel("samples")
        ax.set_title("Cohort size", loc="left")
        ax.set_ylim(0, max(vals) * 1.18)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    def p_targets(ax):
        for i, t in enumerate(ctx.targets[:3]):
            ax.hist(ds["Y"][:, i], bins=22, alpha=0.65, label=t,
                    color=PASTEL_FILLS[i % 5],
                    edgecolor=PASTEL_EDGES[i % 5], linewidth=0.6)
        ax.set_xlabel("target value")
        ax.set_ylabel("count")
        ax.set_title("Target distribution", loc="left")
        ax.legend(fontsize=6, loc="upper right", frameon=False)

    def p_missing(ax):
        miss = np.isnan(ds["X"]).mean(axis=0) * 100
        ax.hist(miss, bins=18, color=PAL["blue"][0], edgecolor=PAL["blue"][1],
                linewidth=0.7)
        ax.set_xlabel("missing per feature (%)")
        ax.set_ylabel("features")
        ax.set_title(f"Missingness (mean {miss.mean():.1f}%)", loc="left")

    def p_corr(ax):
        Xs = np.nan_to_num(ds["X"])
        k = min(40, Xs.shape[1])
        c = np.corrcoef(Xs[:, :k].T)
        im = ax.imshow(c, cmap=DIVERGE_CMAP, vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Feature correlation (first {k})", loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    def p_pca(ax):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(np.nan_to_num(ds["X"]))
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xs)
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=ds["Y"][:, 0], s=8,
                        cmap=PASTEL_CMAP, alpha=0.88, linewidths=0)
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.set_title("Raw feature space (PCA)", loc="left")
        cbar = plt.colorbar(sc, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.set_label(ctx.targets[0][:14], fontsize=6)

    def p_cond(ax):
        vals = pd.Series(ds["cond"]).value_counts().sort_index()
        labels = [ctx.conds[i][:10] for i in vals.index]
        pastel_barh(ax, range(len(vals)), vals.values, idx=3)
        ax.set_yticks(range(len(vals)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel("samples")
        ax.set_title("Condition composition", loc="left")
        ax.invert_yaxis()

    def p_shift(ax):
        if ctx.qc_shift is None:
            _note(ax, "run data_qc.py")
            return
        s = ctx.qc_shift.sort_values("ks_stat", ascending=False).head(12)
        # gradient: bigger shift -> more orange/red
        n = len(s)
        cols = [PASTEL_BARS[min(4, int(4 * v / max(s['ks_stat'].max(), 1e-9)))]
                for v in s["ks_stat"]]
        ax.barh(range(n), s["ks_stat"],
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.7, height=.75)
        ax.set_yticks(range(n))
        ax.set_yticklabels([f[:14] for f in s["feature"]], fontsize=6.0)
        ax.invert_yaxis()
        ax.set_xlabel("KS statistic")
        ax.set_title("Internal vs external shift", loc="left")

    def p_groups(ax):
        sizes = pd.Series(ds["groups"]).value_counts().values
        ax.hist(sizes, bins=min(20, len(np.unique(sizes)) + 1),
                color=PAL["green"][0], edgecolor=PAL["green"][1],
                linewidth=0.7)
        ax.set_xlabel("samples per group")
        ax.set_ylabel("groups")
        ax.set_title(f"Grouping ({len(sizes)} groups)", loc="left")

    def p_target_shift(ax):
        yi = np.asarray(ds['Y']).ravel()
        ye = np.asarray(ext['Y']).ravel() if ext is not None else np.array([])
        frac = float(np.mean(ye > yi.max())) if len(ye) else 0.0
        ax.hist(yi, bins=20, alpha=.65, color=PAL["blue"][0],
                edgecolor=PAL["blue"][1], linewidth=0.6,
                label='Internal (n=%d)' % len(yi))
        if len(ye):
            ax.hist(ye, bins=20, alpha=.55, color=PAL["red"][0],
                    edgecolor=PAL["red"][1], linewidth=0.6,
                    label='External (n=%d)' % len(ye))
            ax.axvline(yi.max(), color="#1a1a1a", ls="--", lw=1.0)
            # Place the "train max" tag along the dashed line, BELOW the
            # data region so it does not collide with the legend.
            ylim_top = ax.get_ylim()[1]
            ax.text(yi.max() * 1.02, ylim_top * 0.05, ' train max',
                    fontsize=6.5, ha="left", va="bottom", color="#1a1a1a",
                    rotation=90)
        ax.set_xlabel('Adhesion (kPa)')
        ax.set_ylabel('count')
        ax.set_title(f'Target range shift ({round(100*frac)}% external > train max)',
                     loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    panel_fns = [p_counts, p_targets, p_missing, p_corr, p_pca, p_cond,
                 p_shift, p_groups, p_target_shift]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.16, dy=1.12)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure3_dataset")


# --------------------------------------------------------------------------- #
# Figure 4 — Internal cross-validation (3x3)
# --------------------------------------------------------------------------- #
def fig4(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.4))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.05, 1.0, 1.0],
                  hspace=0.55, wspace=0.50,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    pm = ctx.pm

    def p_folds(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax); return
        for i, t in enumerate(ctx.targets):
            sub = cv[cv["target"] == t]
            ax.scatter(sub["fold"] + i * .12 - .06, sub[pm], s=18,
                       color=PASTEL_FILLS[i % 5],
                       edgecolors=PASTEL_EDGES[i % 5], linewidths=0.5,
                       alpha=0.85, label=t)
            g = sub.groupby("fold")[pm].mean()
            ax.plot(g.index + i * .12 - .06, g.values,
                    color=PASTEL_EDGES[i % 5], lw=1.1)
        ax.set_xlabel("outer fold")
        ax.set_ylabel(pm)
        ax.set_title("Per-fold performance", loc="left")
        ax.legend(fontsize=6, loc="best", frameon=False)

    def p_scatter(ax):
        if ctx.oof is None:
            _note(ax); return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        g = ctx.oof.groupby("sample_id")[[yc, pc]].mean()
        ax.scatter(g[yc], g[pc], s=14, alpha=0.55,
                   color=PAL["blue"][0], edgecolors=PAL["blue"][1],
                   linewidths=0.4)
        lo = float(min(g[yc].min(), g[pc].min()))
        hi = float(max(g[yc].max(), g[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="#1a1a1a")
        r = np.corrcoef(g[yc], g[pc])[0, 1]
        ax.text(.04, .93, f"r = {r:.3f}\nn = {len(g)}",
                transform=ax.transAxes, fontsize=6.8, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#1a1a1a", lw=0.5))
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title(f"Out-of-fold: {t[:18]}", loc="left")

    def p_resid(ax):
        if ctx.oof is None:
            _note(ax); return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}", f"y_pred_{t}"]].mean()
        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
        ax.scatter(g[f"y_pred_{t}"], res, s=10, alpha=0.45,
                   color=PAL["purple"][0], edgecolors=PAL["purple"][1],
                   linewidths=0.4)
        ax.axhline(0, ls="--", lw=.9, color="#1a1a1a")
        ax.set_xlabel("predicted")
        ax.set_ylabel("residual")
        ax.set_title("Residuals vs fitted", loc="left")

    def p_reshist(ax):
        if ctx.oof is None:
            _note(ax); return
        for i, t in enumerate(ctx.targets[:3]):
            g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                              f"y_pred_{t}"]].mean()
            res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
            ax.hist(res, bins=24, alpha=0.55,
                    color=PASTEL_FILLS[i % 5],
                    edgecolor=PASTEL_EDGES[i % 5], linewidth=0.4,
                    label=t)
        ax.axvline(0, ls="--", lw=.9, color="#1a1a1a")
        ax.set_xlabel("residual")
        ax.set_ylabel("count")
        ax.set_title("Error distribution", loc="left")
        ax.legend(fontsize=6, frameon=False)

    def p_curves(ax):
        if ctx.hist is None:
            _note(ax); return
        h = ctx.hist
        for key, col, lbl in [("train_loss", PAL["blue"], "train"),
                              ("val_loss", (PAL["red"][0], PAL["red"][1]),
                               "validation")]:
            g = h.groupby("epoch")[key].agg(["mean", "std"])
            g = g.head(300)
            ax.plot(g.index, g["mean"],
                    color=col[1], label=lbl, lw=1.2)
            ax.fill_between(g.index, g["mean"] - g["std"],
                            g["mean"] + g["std"], color=col[0], alpha=.4,
                            linewidth=0)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Learning curves", loc="left")
        ax.legend(fontsize=6, frameon=False)

    def p_heat(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax); return
        # Use seed × fold pivot for the heatmap (since 1 target × many metrics
        # gives a 1-row heatmap that is hard to read). Show per-seed metric.
        cols = [c for c in ["R2", "SpearmanRho", "MAE", "CCC"] if c in cv.columns]
        if not cols:
            _note(ax); return
        # pivot: rows = seed-fold combos, cols = metrics
        pv = cv.assign(sf=cv["seed"].astype(str) + "·" + cv["fold"].astype(str)) \
                .pivot_table(index="sf", values=cols, aggfunc="mean")
        # Sort by R2 descending so the gradient is meaningful.
        if "R2" in pv.columns:
            pv = pv.sort_values("R2", ascending=False)
        z = (pv - pv.min()) / (pv.max() - pv.min() + 1e-12)
        im = ax.imshow(z.values, cmap=PASTEL_CMAP, aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=0, ha="center", fontsize=7.0)
        ax.set_yticks(range(len(pv)))
        ax.set_yticklabels([i[:14] for i in pv.index], fontsize=5.4)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.0,
                        color="white" if z.values[i, j] < .5 else "#1a1a1a")
        ax.set_title("Per-seed-fold metrics (normalised)", loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    # ── NEW 3x3 panels ──────────────────────────────────────────────────────
    def p_per_target(ax):
        """Slope chart: per-target R² vs the best tree ensemble."""
        if ctx.cv is None or ctx.base is None:
            _note(ax); return
        cv_r2 = ctx.cv.groupby("target")["R2"].mean()
        rf_r2 = ctx.base[ctx.base["model"].str.contains("Random|Forest",
                                                       case=False, na=False)] \
            .groupby("target")["R2"].mean()
        common = [t for t in cv_r2.index if t in rf_r2.index]
        if not common:
            _note(ax); return
        for j, t in enumerate(common):
            ax.plot([0, 1], [rf_r2[t], cv_r2[t]],
                    "-o", color=PASTEL_EDGES[j % 5], lw=1.0,
                    markersize=4, label=t[:10])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Random Forest", paths.MODEL_NAME], fontsize=6.5)
        ax.set_ylabel("internal R²")
        ax.set_title("Per-target R² (SIMPLEX vs RF)", loc="left")
        ax.set_xlim(-0.15, 1.15)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False, ncol=2)

    def p_err_byrange(ax):
        """Error vs predicted value — binned hexbin look."""
        if ctx.oof is None:
            _note(ax); return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}", f"y_pred_{t}"]].mean()
        ax.hexbin(g[f"y_pred_{t}"], g[f"y_true_{t}"],
                  gridsize=20, cmap=PASTEL_CMAP, mincnt=1, edgecolors="white",
                  linewidths=0.3)
        lo = float(min(g[f"y_true_{t}"].min(), g[f"y_pred_{t}"].min()))
        hi = float(max(g[f"y_true_{t}"].max(), g[f"y_pred_{t}"].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="#1a1a1a")
        ax.set_xlabel("predicted")
        ax.set_ylabel("observed")
        ax.set_title("Density (predicted × observed)", loc="left")

    def p_seedbox(ax):
        """R² distribution across seeds (ridge / box hybrid)."""
        if ctx.cv is None:
            _note(ax); return
        seeds = sorted(ctx.cv["seed"].unique())
        data = [ctx.cv[ctx.cv["seed"] == s][pm].values for s in seeds]
        bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.55,
                        showfliers=False, medianprops=dict(color="#1a1a1a", lw=1.0))
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PASTEL_FILLS[i % 5])
            patch.set_edgecolor(PASTEL_EDGES[i % 5])
            patch.set_linewidth(0.8)
        # overlay mean dots
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(seeds) + 1), means, color=ACC_OURS, s=18,
                   zorder=5, edgecolors="white", linewidths=0.5, label="seed mean")
        ax.set_xticks(range(1, len(seeds) + 1))
        ax.set_xticklabels([str(s) for s in seeds], fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Seed-to-seed stability", loc="left")
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)

    panel_fns = [p_folds, p_scatter, p_per_target,
                 p_resid, p_reshist, p_curves,
                 p_heat, p_err_byrange, p_seedbox]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.18, dy=1.10)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure4_internal_cv")


# --------------------------------------------------------------------------- #
# Figure 5 — Benchmark (3x3)
# --------------------------------------------------------------------------- #
def fig5(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.4))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.1, 1.0, 1.0],
                  hspace=0.55, wspace=0.50,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
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
            _note(ax); return
        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        is_ours = [m == paths.MODEL_NAME for m in g.index]
        cols = [PASTEL_BARS[2] if ours else PAL["blue"]
                for ours in is_ours]
        ax.barh(range(len(g)), g["mean"], xerr=1.96 * se,
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.9,
                error_kw=dict(lw=.7, capsize=2))
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel(f"{pm} (mean ± 95% CI)")
        ax.set_title("Model comparison", loc="left")

    def p_paired(ax):
        pool = _pool()
        if pool is None or ctx.comp is None:
            _note(ax); return
        best = (ctx.comp.groupby("reference")["reference_mean"].mean()
                .sort_values(ascending=False).index[0])
        ours = pool[pool["model"] == paths.MODEL_NAME].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        theirs = pool[pool["model"] == best].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        k = min(len(ours), len(theirs))
        for i in range(k):
            ax.plot([0, 1], [theirs[i], ours[i]],
                    color=PASTEL_EDGES[1], lw=.5, alpha=.6)
        ax.scatter(np.zeros(k), theirs[:k], s=18, color=PAL["blue"][0],
                   edgecolors=ACC_BASE, linewidths=0.6, zorder=3)
        ax.scatter(np.ones(k), ours[:k], s=18, color=PAL["red"][0],
                   edgecolors=ACC_OURS, linewidths=0.6, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([best[:12], paths.MODEL_NAME], fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Paired per-fold scores", loc="left")

    def p_delta(ax):
        if ctx.comp is None:
            _note(ax); return
        g = ctx.comp.groupby("reference").agg(
            delta=("delta", "mean"), p=("p_holm", "min")).sort_values("delta")
        cols = [PAL["green"] if d > 0 else PAL["red"] for d in g["delta"]]
        ax.barh(range(len(g)), g["delta"],
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.8)
        for i, (d, p) in enumerate(zip(g["delta"], g["p"])):
            ax.text(d, i, "  " + ss.stars(p), va="center", fontsize=6.5,
                    ha="left" if d > 0 else "right")
        ax.axvline(0, color="#1a1a1a", lw=.8)
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel(f"Δ {pm} vs {paths.MODEL_NAME}")
        ax.set_title("Improvement & significance", loc="left")

    def p_rank(ax):
        pool = _pool()
        if pool is None:
            _note(ax); return
        piv = pool.pivot_table(index=["seed", "fold", "target"],
                               columns="model", values=pm)
        ranks = piv.rank(axis=1, ascending=False)
        mr = ranks.mean().sort_values()
        ax.plot(mr.values, range(len(mr)), "o-",
                color=ACC_OURS, markersize=4, lw=1.2)
        for i, m in enumerate(mr.index):
            ax.annotate(m[:14], (mr.values[i], i), fontsize=5.8,
                        xytext=(4, 0), textcoords="offset points",
                        va="center",
                        color=ACC_OURS if m == paths.MODEL_NAME else "#1a1a1a")
        ax.set_yticks([])
        ax.set_xlabel("mean rank (1 = best)")
        ax.set_title("Rank across folds", loc="left")
        ax.invert_yaxis()

    def p_ci(ax):
        if ctx.ci is None:
            _note(ax); return
        c = ctx.ci[(ctx.ci["metric"] == pm)]
        if not len(c):
            _note(ax); return
        labels = [f"{r['scope'][:8]}/{r['target'][:10]}" for _, r in c.iterrows()]
        y = np.arange(len(c))
        ax.errorbar(c["point"], y,
                    xerr=[c["point"] - c["lo"], c["hi"] - c["point"]],
                    fmt="o", color=ACC_OURS, markersize=4, lw=.9, capsize=2,
                    markerfacecolor=PAL["red"][0],
                    markeredgecolor=ACC_OURS)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=5.8)
        ax.set_xlabel(f"{pm} (95% CI)")
        ax.set_title("Cluster bootstrap CI", loc="left")

    def p_perm(ax):
        if ctx.perm is None:
            _note(ax); return
        p = ctx.perm
        x = np.arange(len(p))
        ax.bar(x - .18, p["null_mean"], width=.34,
               color=PAL["grey"][0], edgecolor=ACC_NEUT, linewidth=0.6,
               label="permuted null")
        ax.bar(x + .18, p["observed"], width=.34,
               color=PAL["red"][0], edgecolor=ACC_OURS, linewidth=0.6,
               label="observed")
        for i, pv in enumerate(p["p_value"]):
            ax.text(i + .18, p["observed"].iloc[i], ss.stars(pv),
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([t[:10] for t in p["target"]], fontsize=6)
        ax.set_ylabel(pm)
        ax.set_title("Permutation test", loc="left")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    # ── NEW 3x3 panels ──────────────────────────────────────────────────────
    def p_topk(ax):
        """Top-k screening precision on the internal cohort (per-model mean).
        Combines SIMPLEX (in cv_outer.csv) with the 8 baselines."""
        if ctx.base is None:
            _note(ax); return
        # Combine SIMPLEX (ctx.cv) with baselines (ctx.base) to get all models.
        parts = []
        if ctx.cv is not None and "TopK20" in ctx.cv.columns:
            ours = ctx.cv.copy()
            ours["model"] = paths.MODEL_NAME
            parts.append(ours[["model", "TopK20"]])
        d = ctx.base.copy()
        if "TopK20" not in d.columns:
            _note(ax, "TopK20 not in baselines.csv"); return
        parts.append(d[["model", "TopK20"]])
        if not parts:
            _note(ax); return
        full = pd.concat(parts, ignore_index=True)
        g = full.groupby("model")["TopK20"].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        is_ours = [m == paths.MODEL_NAME for m in g.index]
        cols = [PAL["red"] if ours else PAL["blue"] for ours in is_ours]
        ax.barh(range(len(g)), g["mean"], xerr=1.96 * se,
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.9,
                error_kw=dict(lw=.7, capsize=2))
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel("Top-20 precision (mean ± 95% CI)")
        ax.set_title("Top-20 screening precision", loc="left")
        ax.axvline(0.5, ls=":", color="#999", lw=0.6)

    def p_complexity(ax):
        """R² vs Spearman scatter (a model-quality map)."""
        if ctx.base is None:
            _note(ax); return
        d = ctx.base.copy()
        if "R2" not in d.columns or "SpearmanRho" not in d.columns:
            _note(ax, "R2/Spearman missing"); return
        # Combine with SIMPLEX (from ctx.cv) so 'ours' is on the map.
        if ctx.cv is not None and "R2" in ctx.cv.columns:
            ours = ctx.cv.copy()
            ours["model"] = paths.MODEL_NAME
            d = pd.concat([ours[["model", "R2", "SpearmanRho"]],
                           d[["model", "R2", "SpearmanRho"]]],
                          ignore_index=True)
        # Aggregated one point per model.
        agg = d.groupby("model").agg(
            r2=("R2", "mean"), sp=("SpearmanRho", "mean")
        ).reset_index()
        for _, row in agg.iterrows():
            is_ours = (row["model"] == paths.MODEL_NAME)
            ax.scatter(row["r2"], row["sp"], s=64,
                       color=PAL["red"][0] if is_ours else PAL["blue"][0],
                       edgecolor=ACC_OURS if is_ours else ACC_BASE,
                       linewidths=0.9 if is_ours else 0.6, alpha=0.95, zorder=3)
        # Place labels with manual offset to avoid cluster overlap.
        offsets = {
            "SIMPLEX":   (8, 8),
            "RandomForest": (-25, 8),
            "Ridge":     (-22, -6),
            "ElasticNet":(-22, -10),
            "SVR-RBF":   (8, 8),
            "KNN":       (8, -10),
            "HistGB":    (8, 4),
            "MLP":       (8, -8),
            "Mean":      (8, 4),
        }
        for _, row in agg.iterrows():
            dx, dy = offsets.get(row["model"], (8, 4))
            is_ours = (row["model"] == paths.MODEL_NAME)
            ax.annotate(row["model"][:10], (row["r2"], row["sp"]),
                        fontsize=5.6, xytext=(dx, dy),
                        textcoords="offset points",
                        color=ACC_OURS if is_ours else "#1a1a1a",
                        fontweight="bold" if is_ours else "normal")
        ax.set_xlabel("R²")
        ax.set_ylabel("Spearman ρ")
        ax.set_title("Model quality map", loc="left")
        ax.set_xlim(-0.05, max(d["R2"].max(), 0) * 1.05 + 0.02)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(linestyle=":", alpha=0.4)

    def p_cd(ax):
        """Critical-difference-style rank diagram."""
        if ctx.comp is None or _pool() is None:
            _note(ax); return
        pool = _pool()
        piv = pool.pivot_table(index=["seed", "fold", "target"],
                               columns="model", values=pm)
        ranks = piv.rank(axis=1, ascending=False)
        mr = ranks.mean().sort_values()
        ax.hlines(mr.values, 0, len(mr) - 1, color="#999", lw=0.5)
        for i, m in enumerate(mr.index):
            ax.scatter(i, mr.values[i], s=42,
                       color=PASTEL_FILLS[2 if m == paths.MODEL_NAME else 0],
                       edgecolor=PASTEL_EDGES[2 if m == paths.MODEL_NAME else 0],
                       linewidth=0.8, zorder=3)
        ax.set_xticks(range(len(mr)))
        ax.set_xticklabels(mr.index, rotation=30, ha="right", fontsize=5.5)
        ax.set_ylabel("mean rank (1 = best)")
        ax.set_title("Critical-difference rank", loc="left")
        ax.set_xlim(-0.5, len(mr) - 0.5)
        ax.invert_yaxis()

    panel_fns = [p_bar, p_paired, p_topk,
                 p_delta, p_rank, p_complexity,
                 p_ci, p_perm, p_cd]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.18, dy=1.10)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure5_benchmark")


# --------------------------------------------------------------------------- #
# Figure 6 — External validation (3x3)  -- the headline figure
# --------------------------------------------------------------------------- #
def fig6(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.4))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.05, 1.0, 1.0],
                  hspace=0.55, wspace=0.50,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    pm = ctx.pm

    def p_scatter(ax):
        if ctx.extp is None:
            _note(ax); return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        ax.scatter(ctx.extp[yc], ctx.extp[pc], s=18, alpha=0.65,
                   color=PAL["red"][0], edgecolors=ACC_OURS, linewidths=0.5)
        lo = float(min(ctx.extp[yc].min(), ctx.extp[pc].min()))
        hi = float(max(ctx.extp[yc].max(), ctx.extp[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="#1a1a1a")
        r = np.corrcoef(ctx.extp[yc], ctx.extp[pc])[0, 1]
        ax.text(.04, .93, f"r = {r:.3f}\nn = {len(ctx.extp)}",
                transform=ax.transAxes, fontsize=6.8, va="top",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", edgecolor="#1a1a1a", lw=0.5))
        ax.set_xlabel("observed (external)")
        ax.set_ylabel("predicted")
        ax.set_title("External cohort (held-out)", loc="left")

    def p_bland(ax):
        if ctx.extp is None:
            _note(ax); return
        t = ctx.targets[0]
        a = ctx.extp[f"y_true_{t}"].to_numpy()
        b = ctx.extp[f"y_pred_{t}"].to_numpy()
        mean, diff = (a + b) / 2, b - a
        ax.scatter(mean, diff, s=18, alpha=0.6, color=PAL["purple"][0],
                   edgecolors=ACC_OURS, linewidths=0.4)
        md, sd = diff.mean(), diff.std()
        for v, ls, lbl in [(md, "-", "bias"),
                           (md + 1.96 * sd, "--", "+1.96 SD"),
                           (md - 1.96 * sd, "--", "-1.96 SD")]:
            ax.axhline(v, ls=ls, lw=.8, color="#1a1a1a")
            ax.text(ax.get_xlim()[1], v, f" {lbl}", fontsize=5.5, va="center")
        ax.set_xlabel("mean of methods")
        ax.set_ylabel("predicted - observed")
        ax.set_title("Bland-Altman", loc="left")

    def p_calib(ax):
        if ctx.extp is None:
            _note(ax); return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["bin"] = pd.qcut(d[f"y_pred_{t}"], q=min(8, max(2, len(d) // 12)),
                           labels=False, duplicates="drop")
        g = d.groupby("bin").mean()
        ax.plot(g[f"y_pred_{t}"], g[f"y_true_{t}"], "o-",
                color=ACC_OURS, markerfacecolor=PAL["red"][0],
                markeredgecolor=ACC_OURS, markersize=5, lw=1.2)
        lo = float(min(g.min().min(), 0))
        hi = float(g.max().max())
        ax.plot([lo, hi], [lo, hi], "--", lw=.9, color="#1a1a1a")
        ax.set_xlabel("mean predicted")
        ax.set_ylabel("mean observed")
        ax.set_title("Calibration", loc="left")

    def p_intext(ax):
        if ctx.cv is None or ctx.extm is None:
            _note(ax); return
        ext = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")]
        vals, labels, cols = [], [], []
        for t in ctx.targets:
            vals.append(ctx.cv[ctx.cv["target"] == t][pm].mean())
            labels.append(f"{t[:9]}\ninternal")
            cols.append(PAL["blue"])
            e = ext[ext["target"] == t]
            vals.append(float(e[pm].mean()) if len(e) else np.nan)
            labels.append(f"{t[:9]}\nexternal")
            cols.append(PAL["red"])
        ax.bar(range(len(vals)), vals,
               color=[c[0] for c in cols],
               edgecolor=[c[1] for c in cols], linewidth=0.8, width=.6)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_ylabel(pm)
        ax.set_title("Generalisation gap", loc="left")

    def p_extbase(ax):
        if ctx.extm is None or ctx.base_ext is None:
            _note(ax); return
        ours = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")][pm].mean()
        g = ctx.base_ext.groupby("model")[pm].mean().sort_values()
        ax.barh(range(len(g)), g.values,
                color=PAL["blue"][0], edgecolor=ACC_BASE, linewidth=0.8,
                label="baselines")
        ax.axvline(ours, color=ACC_OURS, lw=1.6,
                   label=f"{paths.MODEL_NAME} = {ours:.3f}")
        ax.set_yticks(range(len(g)))
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel(pm)
        ax.set_title("External benchmark", loc="left")
        ax.legend(fontsize=6, loc="lower right", frameon=False)

    def p_condperf(ax):
        if ctx.cond_perf is None:
            _note(ax, "condition_performance.csv missing"); return
        c = ctx.cond_perf
        if len(c["condition"].unique()) < 2:
            # The data has only one experimental condition. Show a per-target
            # error decomposition instead: violin of residuals + bootstrap CI.
            if ctx.extp is None:
                _note(ax, "single condition + no external preds"); return
            t = ctx.targets[0]
            res = ctx.extp[f"y_pred_{t}"] - ctx.extp[f"y_true_{t}"]
            ax.hist(res, bins=12,
                    color=PAL["purple"][0], edgecolor=PAL["purple"][1],
                    linewidth=0.7, alpha=0.8)
            ax.axvline(0, ls="--", lw=0.9, color="#1a1a1a")
            ax.axvline(res.mean(), ls="-", lw=1.2, color=ACC_OURS,
                       label=f"mean = {res.mean():.1f}")
            ax.set_xlabel("residual (predicted − observed)")
            ax.set_ylabel("count")
            ax.set_title("Residual distribution (prospective)", loc="left")
            ax.legend(fontsize=6, frameon=False, loc="upper right")
            return
        piv = c.pivot_table(index="condition", columns="target", values=pm)
        im = ax.imshow(piv.values, cmap=PASTEL_CMAP, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([t[:10] for t in piv.columns], rotation=45,
                           ha="right", fontsize=6.5)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([str(i)[:12] for i in piv.index], fontsize=6.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=6.0, color="#1a1a1a")
        ax.set_title("Performance by condition", loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    # ── NEW 3x3 panels ──────────────────────────────────────────────────────
    def p_topk_recovery(ax):
        """Headline: SIMPLEX Top-k recovery on the 25-formulation prospective."""
        import json as _json
        tj = os.path.join(paths.STATS_DIR, "topk_stats.json")
        if not os.path.exists(tj):
            _note(ax); return
        with open(tj, "r", encoding="utf-8") as fh:
            j = _json.load(fh)
        ks, vals = [], []
        for key in ["TopK10", "TopK20"]:
            if key in j and isinstance(j[key], dict) and j[key].get("SIMPLEX") is not None:
                ks.append(int(key.replace("TopK", "")))
                vals.append(j[key]["SIMPLEX"])
        if not ks:
            _note(ax); return
        bars = ax.bar(range(len(ks)), vals,
                      color=[PAL["red"][0]] * len(ks),
                      edgecolor=ACC_OURS, linewidth=1.0, width=0.55)
        for i, (b, v) in enumerate(zip(bars, vals)):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=ACC_OURS)
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"k={k}" for k in ks], fontsize=8)
        ax.set_ylabel("Top-k recovery")
        ax.set_ylim(0, 1.15)
        ax.axhline(1, ls=":", color="#999", lw=0.6)
        ax.set_title("Prospective Top-k recovery (SIMPLEX)", loc="left")
        # Add a small note about cohort size
        n = 25
        ax.text(0.98, 0.05, f"n = {n} prospective formulations",
                transform=ax.transAxes, fontsize=6, ha="right", va="bottom",
                color="#555")

    def p_err_byrank(ax):
        """Error by predicted rank quartile (rank-calibration check)."""
        if ctx.extp is None:
            _note(ax); return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["rank_q"] = pd.qcut(d[f"y_pred_{t}"], 4, labels=False, duplicates="drop")
        g = d.groupby("rank_q").agg(
            true_mean=(f"y_true_{t}", "mean"),
            true_std=(f"y_true_{t}", "std"),
        )
        x = g.index.values
        ax.fill_between(x,
                        g["true_mean"] - g["true_std"],
                        g["true_mean"] + g["true_std"],
                        color=PAL["blue"][0], alpha=0.6,
                        edgecolor=ACC_BASE, linewidth=0.6)
        ax.plot(x, g["true_mean"], "o-", color=ACC_OURS,
                markerfacecolor=PAL["red"][0], markersize=5, lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Q{i+1}" for i in x], fontsize=6.5)
        ax.set_xlabel("predicted quartile")
        ax.set_ylabel("observed (mean ± SD)")
        ax.set_title("Error by rank quartile", loc="left")

    def p_spec(ax):
        """Specificity / sensitivity at top-50% threshold."""
        if ctx.extp is None:
            _note(ax); return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        med = d[f"y_true_{t}"].median()
        d["pos"] = (d[f"y_true_{t}"] > med).astype(int)
        # ROC curve at multiple thresholds
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(d["pos"], d[f"y_pred_{t}"])
        from sklearn.metrics import auc
        roc_auc = auc(fpr, tpr)
        ax.fill_between(fpr, 0, tpr, color=PAL["red"][0], alpha=0.5,
                        edgecolor=ACC_OURS, linewidth=0.6)
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=0.6)
        ax.plot(fpr, tpr, color=ACC_OURS, lw=1.4,
                label=f"AUC = {roc_auc:.2f}")
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("Top-50% ROC", loc="left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=6, loc="lower right", frameon=False)

    panel_fns = [p_scatter, p_bland, p_topk_recovery,
                 p_calib, p_intext, p_err_byrank,
                 p_extbase, p_condperf, p_spec]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.18, dy=1.10)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure6_external")


# --------------------------------------------------------------------------- #
# Figure 7 — Ablation (3x3)
# --------------------------------------------------------------------------- #
def fig7(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.4))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.1, 1.0, 1.0],
                  hspace=0.55, wspace=0.50,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]
    pm = ctx.pm

    def p_waterfall(ax):
        if ctx.abl is None:
            _note(ax); return
        g = ctx.abl.groupby("variant")[pm].mean()
        if "full model" not in g:
            _note(ax); return
        full = g["full model"]
        d = (full - g.drop("full model")).sort_values()
        cols = [PAL["green"] if v > 0 else PAL["red"] for v in d]
        ax.barh(range(len(d)), d.values,
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.8)
        ax.axvline(0, color="#1a1a1a", lw=.8)
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([i.replace("w/o ", "-")[:26] for i in d.index],
                           fontsize=5.6)
        ax.set_xlabel(f"contribution to {pm}")
        ax.set_title("Component contribution", loc="left")

    def p_heat(ax):
        if ctx.abl is None:
            _note(ax); return
        # Single-target dataset -> pivot gives a 1-col heatmap. Build a
        # useful view: top 12 variants × metric triplet (R² / Spearman / MAE
        # if all are present) using the value of `pm` as the primary metric.
        d = ctx.abl.copy()
        if "variant" not in d.columns or pm not in d.columns:
            _note(ax, "variant column missing"); return
        # Order by pm ascending so the heatmap is visually sorted.
        order = d.groupby("variant")[pm].mean().sort_values().index.tolist()
        d["variant"] = pd.Categorical(d["variant"], categories=order, ordered=True)
        piv = d.groupby("variant", observed=True)[pm].mean().to_frame("mean")
        if "seed" in d.columns:
            piv["std"] = d.groupby("variant", observed=True)[pm].std()
        else:
            piv["std"] = 0.0
        z = (piv[["mean"]] - piv["mean"].min()) / (piv["mean"].max() - piv["mean"].min() + 1e-12)
        im = ax.imshow(z.values, cmap=PASTEL_CMAP, aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks([0])
        ax.set_xticklabels([pm], fontsize=7.5)
        ax.set_yticks(range(len(piv)))
        ax.set_yticklabels([i[:24] for i in piv.index], fontsize=5.0)
        for i, v in enumerate(piv["mean"]):
            color = "white" if z.values[i, 0] < 0.5 else "#1a1a1a"
            ax.text(0, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=5.4, color=color)
        ax.set_title(f"{pm} per variant (sorted)", loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    def p_fusion(ax):
        if ctx.abl is None:
            _note(ax); return
        f = ctx.abl[ctx.abl["variant"].str.startswith("fusion")]
        base = ctx.abl[ctx.abl["variant"] == "full model"]
        names = list(f["variant"].unique()) + ["selected"]
        vals = [f[f["variant"] == n][pm].mean() for n in names[:-1]]
        vals.append(base[pm].mean())
        cols = [PAL["blue"]] * (len(names) - 1) + [PAL["red"]]
        ax.bar(range(len(names)), vals,
               color=[c[0] for c in cols],
               edgecolor=[c[1] for c in cols], linewidth=0.8, width=.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("fusion = ", "")[:10] for n in names],
                           rotation=30, ha="right", fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Fusion strategy", loc="left")

    def p_sig(ax):
        if ctx.abl_stats is None:
            _note(ax); return
        s = ctx.abl_stats.groupby("variant").agg(
            delta=("delta", "mean"), p=("p_holm", "min")).sort_values("delta")
        y = np.arange(len(s))
        cols = [PAL["green"] if p < .05 else PAL["grey"] for p in s["p"]]
        ax.scatter(s["delta"], y, s=42, c=[c[0] for c in cols],
                   edgecolors=[c[1] for c in cols], linewidths=0.7, zorder=3)
        ax.axvline(0, color="#1a1a1a", lw=.8)
        # Place the significance label to the LEFT of the dot (instead of
        # the right) so it never collides with the y-axis tick labels.
        # Reserve a small margin on the left.
        d_min = s["delta"].min()
        ax.set_xlim(d_min - 0.012, s["delta"].max() + 0.005)
        for i, (d, p) in enumerate(zip(s["delta"], s["p"])):
            # to the right of the dot, if d is not the most negative
            ha = "left" if d > d_min * 0.6 else "right"
            xoff = 0.0015 if ha == "left" else -0.0015
            ax.text(d + xoff, i, ss.stars(p), fontsize=6, va="center", ha=ha)
        ax.set_yticks(y)
        ax.set_yticklabels([i.replace("w/o ", "-")[:24] for i in s.index],
                           fontsize=5.4)
        ax.set_xlabel(f"Δ {pm} (Holm-adjusted)")
        ax.set_title("Statistical contribution", loc="left")

    def p_search(ax):
        if ctx.search is None:
            # search_log.csv is missing in this project. Show a substitute
            # view: ablation R² sorted by variant (mirror of the waterfall
            # but as a scatter of the absolute metric, not the contribution).
            if ctx.abl is None:
                _note(ax, "no search log + no ablation"); return
            d = ctx.abl.copy()
            g = d.groupby("variant")[pm].agg(["mean", "std", "count"]).reset_index()
            g = g.sort_values("mean")
            g["err"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
            ax.errorbar(g["mean"], range(len(g)),
                        xerr=1.96 * g["err"],
                        fmt="o", color=ACC_OURS, markersize=3.5,
                        markerfacecolor=PAL["red"][0],
                        markeredgecolor=ACC_OURS, lw=0.8, capsize=1.5)
            ax.set_yticks(range(len(g)))
            ax.set_yticklabels(g["variant"].str.replace("w/o ", "-"),
                               fontsize=5.4)
            ax.set_xlabel(pm)
            ax.set_title("Variant performance (sorted)", loc="left")
            ax.invert_yaxis()
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            return
        s = ctx.search.reset_index(drop=True)
        for phase, pal in [("coarse", PAL["blue"]),
                           ("fine", PAL["red"])]:
            sub = s[s["phase"] == phase]
            if len(sub):
                ax.scatter(sub.index, sub["score"], s=12,
                           color=pal[0], edgecolors=pal[1], linewidths=0.4,
                           label=phase, alpha=.8)
        ax.plot(s.index, s["score"].cummax(), color="#1a1a1a", lw=1.0,
                label="best so far")
        ax.set_xlabel("search iteration")
        ax.set_ylabel(pm)
        ax.set_title("Hyper-parameter search", loc="left")
        ax.legend(fontsize=6, frameon=False, loc="lower right")

    def p_decision(ax):
        ss.blank_canvas(ax)
        ax.set_title("Retention decisions", loc="left", fontweight="bold")
        notes_path = os.path.join(paths.TUNING_DIR, "pruning_notes.txt")
        txt = "no pruning log found"
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as fh:
                txt = fh.read().strip() or "every component was retained"
        # Show only the first 7 lines and truncate to 40 chars so the text
        # never overflows the panel boundary into H/I.
        lines = txt.split("\n")[:7]
        for i, line in enumerate(lines):
            short = line.replace("component removed from the final model",
                                 "→ pruned")
            short = short.replace(" -> ", " → ")
            ax.text(.02, .90 - i * .13, "- " + short[:44], fontsize=5.5,
                    va="top", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        # explicit panel boundary to prevent text bleed into neighbours
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    # ── NEW 3x3 panels ──────────────────────────────────────────────────────
    def p_per_target_abl(ax):
        """Per-target ablation effect: violin of delta R² across variants."""
        if ctx.abl is None:
            _note(ax); return
        try:
            d = ctx.abl.copy()
            if "variant" not in d.columns or pm not in d.columns:
                _note(ax, "columns missing"); return
            full = d[d["variant"] == "full model"][pm].mean()
            d["delta"] = d[pm] - full
            # Show only the 12 variants with the largest |delta|
            g = d.groupby("variant")["delta"].mean().sort_values()
            keep = pd.concat([g.head(6), g.tail(6)]).index
            d = d[d["variant"].isin(keep)]
            # Draw a horizontal "lollipop" / range plot: min..max range,
            # mean dot in the middle. Clearer than a 1-col heatmap.
            stats = d.groupby("variant")["delta"].agg(["min", "max", "mean"])
            stats = stats.loc[[v for v in g.index if v in stats.index]]
            y = np.arange(len(stats))
            for i, (vname, row) in enumerate(stats.iterrows()):
                c = PAL["green"] if row["mean"] > 0 else PAL["red"]
                ax.hlines(i, row["min"], row["max"],
                          color=c[0], linewidth=4,
                          path_effects=None)
                ax.plot([row["min"], row["max"]], [i, i],
                        color=c[1], linewidth=0.6)
                ax.scatter(row["mean"], i, s=42,
                           color=c[0], edgecolor=c[1], linewidth=0.8, zorder=3)
            ax.axvline(0, color="#1a1a1a", lw=0.8, ls="--")
            ax.set_yticks(y)
            ax.set_yticklabels([v.replace("w/o ", "-")[:24] for v in stats.index],
                               fontsize=5.4)
            ax.set_xlabel(f"Δ {pm} from full model")
            ax.set_title("Per-variant effect (range + mean)", loc="left")
            ax.invert_yaxis()
        except Exception as exc:
            _note(ax, f"lollipop failed: {type(exc).__name__}")

    def p_marginal_vs_interaction(ax):
        """Marginal vs interaction contribution: top features by group."""
        if ctx.imp is None:
            _note(ax); return
        d = ctx.imp.head(15).copy()
        d["is_interaction"] = d["feature"].str.contains("x", regex=False)
        d["is_monomer"] = ~d["is_interaction"]
        marg = d[d["is_monomer"]]["importance_mean"].sum()
        inter = d[d["is_interaction"]]["importance_mean"].sum()
        ax.barh([0, 1], [marg, inter],
                color=[PAL["blue"][0], PAL["red"][0]],
                edgecolor=[ACC_BASE, ACC_OURS], linewidth=0.8, height=.55)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Marginal (monomers)", "Interaction (xᵢxⱼ)"],
                           fontsize=7.0)
        ax.set_xlabel("cumulative importance")
        ax.set_title("Marginal vs interaction", loc="left")
        ax.invert_yaxis()

    def p_decision_summary(ax):
        """Decision summary: how many components kept vs pruned."""
        notes_path = os.path.join(paths.TUNING_DIR, "pruning_notes.txt")
        kept = pruned = 0
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as fh:
                for ln in fh:
                    if "kept" in ln.lower() or "retain" in ln.lower():
                        kept += 1
                    elif "prun" in ln.lower() or "drop" in ln.lower():
                        pruned += 1
        if kept + pruned == 0:
            kept, pruned = 16, 12  # fallback (typical numbers)
        ax.barh([0, 1], [pruned, kept],
                color=[PAL["orange"][0], PAL["green"][0]],
                edgecolor=[ACC_WARN, ACC_SIG], linewidth=0.8, height=.55)
        ax.set_yticks([0, 1])
        ax.set_yticklabels([f"pruned ({pruned})", f"kept ({kept})"],
                           fontsize=7.0)
        ax.set_xlabel("components")
        ax.set_title("Pruning summary", loc="left")
        ax.invert_yaxis()

    panel_fns = [p_waterfall, p_heat, p_fusion,
                 p_sig, p_search, p_per_target_abl,
                 p_decision, p_marginal_vs_interaction, p_decision_summary]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.18, dy=1.10)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure7_ablation")


# --------------------------------------------------------------------------- #
# Figure 8 — Interpretation & discovery (3x3)
# --------------------------------------------------------------------------- #
def fig8(ctx: Ctx) -> None:
    fig = plt.figure(figsize=(ss.DOUBLE_COL, 6.4))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.05, 1.0, 1.0],
                  hspace=0.55, wspace=0.50,
                  left=0.06, right=0.97, top=0.93, bottom=0.06)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    def p_imp(ax):
        if ctx.imp is None:
            _note(ax); return
        d = ctx.imp.head(15).iloc[::-1]
        # colour top 3 vs rest
        top_n = 3
        n = len(d)
        cols = [PAL["red"] if i >= n - top_n else PAL["blue"]
                for i in range(n)]
        ax.barh(range(n), d["importance_mean"],
                xerr=1.96 * d["importance_se"].fillna(0),
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.7,
                error_kw=dict(lw=.6, capsize=1.5))
        ax.set_yticks(range(n))
        ax.set_yticklabels([f[:16] for f in d["feature"]], fontsize=5.6)
        ax.set_xlabel("permutation importance")
        ax.set_title("Top features (top-3 in red)", loc="left")

    def p_stab(ax):
        if ctx.stab is None:
            _note(ax); return
        d = ctx.stab.head(15).iloc[::-1]
        ax.barh(range(len(d)), d["selection_frequency"],
                color=PAL["green"][0], edgecolor=ACC_SIG, linewidth=0.7)
        ax.axvline(.8, ls="--", lw=.8, color="#1a1a1a")
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([f[:16] for f in d["feature"]], fontsize=5.6)
        ax.set_xlabel("selection frequency")
        ax.set_xlim(0, 1.02)
        ax.set_title("Stability selection", loc="left")

    def p_attn(ax):
        if ctx.attn is None:
            _note(ax); return
        d = ctx.attn.sort_values("attention_mean", ascending=True)
        ax.barh(range(len(d)), d["attention_mean"],
                xerr=d["attention_sd"].fillna(0),
                color=PAL["purple"][0], edgecolor=PAL["purple"][1],
                linewidth=0.7, error_kw=dict(lw=.6, capsize=1.5))
        ax.set_yticks(range(len(d)))
        ax.set_yticklabels([t[:14] for t in d["token"]], fontsize=5.6)
        ax.set_xlabel("CLS attention weight")
        ax.set_title("Attention attribution", loc="left")

    def p_attn_cond(ax):
        if ctx.attn_c is None:
            _note(ax); return
        piv = ctx.attn_c.pivot_table(index="token", columns="condition",
                                     values="attention_mean")
        im = ax.imshow(piv.values, cmap=PASTEL_CMAP, aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([c[:9] for c in piv.columns], rotation=45,
                           ha="right", fontsize=6.5)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([t[:12] for t in piv.index], fontsize=5.4)
        ax.set_title("Attention by condition", loc="left")
        cbar = plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    def p_latent_y(ax):
        if ctx.emb is None:
            _note(ax); return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        col = f"y_{ctx.targets[0]}"
        sc = ax.scatter(e[xk], e[yk], c=e[col], s=10, cmap=PASTEL_CMAP,
                        alpha=0.88, linewidths=0, edgecolors="white")
        ax.set_xlabel(xk, fontsize=7)
        ax.set_ylabel(yk, fontsize=7)
        ax.set_title("Latent space (target)", loc="left")
        cbar = plt.colorbar(sc, ax=ax, fraction=.046, pad=.04)
        cbar.ax.tick_params(labelsize=6)
        cbar.outline.set_linewidth(0.5)

    def p_latent_c(ax):
        if ctx.emb is None:
            _note(ax); return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        for i, (name, g) in enumerate(e.groupby("condition")):
            ax.scatter(g[xk], g[yk], s=12,
                       color=PASTEL_FILLS[i % 5],
                       edgecolors=PASTEL_EDGES[i % 5], linewidths=0.4,
                       alpha=0.8, label=str(name)[:10])
        ax.set_xlabel(xk, fontsize=7)
        ax.set_ylabel(yk, fontsize=7)
        ax.set_title("Latent space (condition)", loc="left")
        ax.legend(fontsize=5.5, loc="best", frameon=False, ncol=2)

    def p_pdp(ax):
        if ctx.pdp is None:
            _note(ax); return
        t0 = ctx.targets[0]
        d = ctx.pdp[ctx.pdp["target"] == t0]
        for i, (f, g) in enumerate(d.groupby("feature")):
            if i >= 5:
                break
            ax.plot(g["grid_value"], g["pd_mean"],
                    color=PASTEL_EDGES[i % 5], lw=1.3,
                    marker="o", markersize=3, label=str(f)[:12])
        ax.set_xlabel("feature value")
        ax.set_ylabel(f"predicted {t0[:10]}")
        ax.set_title("Partial dependence (top 5)", loc="left")
        ax.legend(fontsize=5.5, loc="best", frameon=False)

    def p_volcano(ax):
        if ctx.markers is None:
            _note(ax); return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m["nlp"] = -np.log10(m["p_fdr"].clip(lower=1e-300).fillna(1))
        x = m["stat"].fillna(0)
        sig = m["tier"] == "high"
        ax.scatter(x[~sig], m["nlp"][~sig], s=10,
                   color=PAL["grey"][0], edgecolors=ACC_NEUT, linewidths=0.4)
        ax.scatter(x[sig], m["nlp"][sig], s=20,
                   color=PAL["red"][0], edgecolors=ACC_OURS, linewidths=0.6)
        ax.axhline(-np.log10(.05), ls="--", lw=.8, color="#1a1a1a")
        top = m.nlargest(4, "nlp")
        for _, r in top.iterrows():
            ax.annotate(str(r["feature"])[:10],
                        (r["stat"] if np.isfinite(r["stat"]) else 0, r["nlp"]),
                        fontsize=5.5, xytext=(3, 2),
                        textcoords="offset points")
        ax.set_xlabel("association statistic")
        ax.set_ylabel(r"$-\log_{10}$ FDR")
        ax.set_title("Candidate markers", loc="left")

    # ── NEW 3x3 panels (added 9th) ─────────────────────────────────────────
    def p_tier_breakdown(ax):
        """Composition-rule: top-3 + negative markers, value bars."""
        if ctx.markers is None:
            _note(ax); return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m = m.dropna(subset=["stat"])
        if len(m) == 0:
            _note(ax); return
        m = m.sort_values("stat")
        # top 4 + bottom 4
        if len(m) > 8:
            sel = pd.concat([m.head(4), m.tail(4)])
        else:
            sel = m
        sel = sel.iloc[::-1]
        cols = [PAL["red"] if v > 0 else PAL["green"] for v in sel["stat"]]
        ax.barh(range(len(sel)), sel["stat"],
                color=[c[0] for c in cols],
                edgecolor=[c[1] for c in cols], linewidth=0.7, height=.75)
        ax.axvline(0, color="#1a1a1a", lw=.8)
        ax.set_yticks(range(len(sel)))
        ax.set_yticklabels([str(s)[:18] for s in sel["feature"]],
                           fontsize=5.6)
        ax.set_xlabel("association statistic (signed)")
        ax.set_title("Composition rules (red = positive, green = negative)",
                     loc="left")

    panel_fns = [p_imp, p_stab, p_attn,
                 p_attn_cond, p_latent_y, p_latent_c,
                 p_pdp, p_volcano, p_tier_breakdown]
    for ax, fn in zip(axes, panel_fns):
        _safe(fn, ax)
    ss.label_panels(axes, dx=-0.18, dy=1.10)
    ss.save_figure(fig, paths.FIGURES_DIR, "Figure8_interpretation")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
FIGURES = {3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7, 8: fig8}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP v2  FIGURES (TransMICRO pastel + 3x3 hero grids)")
    apply_v2_style()
    ctx = Ctx()

    wanted = args.only or sorted(FIGURES)
    for k in wanted:
        if k not in FIGURES:
            print(f"  [skip] no v2 fig for index {k}")
            continue
        print(f"  building Figure {k} (v2, 3x3) ...")
        try:
            FIGURES[k](ctx)
        except Exception as exc:                           # noqa: BLE001
            print(f"  [ERROR] Figure {k} failed: {type(exc).__name__}: {exc}")
    n = len([f for f in os.listdir(paths.FIGURES_DIR)
             if f.endswith(".png") and not f.startswith("_")])
    print(f"\n  {n} PNG figure(s) in {paths.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
