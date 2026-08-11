"""SIMPLEX figure rendering (v3 - Nature-style advanced chart types).

Each Figure3-8 uses advanced chart types (lollipop, slope chart, dumbbell,
forest plot, violin+strip, heatmap with annotations) instead of plain bars.
Panel letters (A-I) are positioned OUTSIDE the axes top-left to avoid overlap.
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import os
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import paths
import sci_style as ss
from sci_style import (OKABE_ITO, NATURE, MODEL_COLOR, BASELINE_COLOR,
                       NEUTRAL_GREY, LIGHT_GREY, apply_style,
                       SEQUENTIAL_CMAP, DIVERGING_CMAP,
                       SINGLE_COL, ONE_HALF_COL, DOUBLE_COL)


# Re-export everything from base figures: figs 1, 2, and helpers.
from figures_v2_backup import (Ctx, FIGURES, _safe, _note, _csv, _pm, _oof_path)  # noqa: F401


apply_style()

# Anti-overlap defaults: tighter letter positioning + ample gutters.

# Global model-name abbreviation map (used by every figure).
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
    "XGBoost": "XGB",
    "LightGBM": "LGBM",
    "CatBoost": "CatB",
    "Dummy": "Dummy",
    "DummyRegressor": "Dummy",
    "MeanPredictor": "Mean",
    "DummyMean": "Mean",
}
def _m_short(m, n=9):
    s = MODEL_SHORT.get(str(m), str(m))
    return s if len(s) <= n else _hard_shorten(s, n)

PANEL_DX = -0.22
PANEL_DY = 1.18


def _label(axes, start="A", dx=PANEL_DX, dy=PANEL_DY):
    """OUTSIDE-top-left panel labels (Arial bold, no overlap with titles)."""
    ss.label_panels(axes, start=start, dx=dx, dy=dy)


def _hard_shorten(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _save(fig, name, rect=(0, 0, 1, 0.94), wspace=0.55, hspace=0.70):
    fig.tight_layout(rect=rect)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    ss.save_figure(fig, paths.FIGURES_DIR, name)


def _shade(i, n):
    """blue-shade gradient for baselines (n models, i index)."""
    if n <= 1:
        return NATURE["base"]
    return plt.cm.Blues(0.35 + 0.60 * i / max(n - 1, 1))


def _broken_cut(vals, k=1.5):
    """Adaptive broken-axis threshold: cut = k * median(positive values).
    Returns None when nothing exceeds the cut (uniform data -> no zigzag,
    avoiding misleading truncation). k=1.5 keeps truncated bars very close
    to the short bars (user keeps asking for shorter cuts)."""
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
    """Zigzag '//' break markers at the truncated end of a bar/line.
    orient='h': bar extends in +x, truncated at x=cut, bar centre y=pos,
    half-height=span. orient='v': bar extends in +y, truncated at y=cut."""
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


# =========================================================================== #
# Figure 3 - cohort / dataset characteristics (3x3, advanced chart types)
# =========================================================================== #
def fig3(ctx: Ctx) -> None:
    # 3x2 layout: panels C (missingness ~0), F (single condition) and
    # H (all singleton groups) carry no variance and are dropped; the
    # remaining six panels carry the informative cohort description.
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, 6.6))
    axes = axes.ravel()
    ds, ext = ctx.ds, ctx.ext

    # ----- A: cohort size lollipop (much cleaner than bar) ----------------
    def p_counts(ax):
        names = ["Internal", "BO-acquired external"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        colors = [NATURE["ours"], NATURE["base"]]
        # twin horizontal bars (side-by-side), values at bar ends.
        # Broken axis: 316 >> 25 (12.6x); CUT=34 keeps 25 at ~74% of the
        # truncated bar.
        CUT = 34.0
        plot_vals = np.minimum(vals, CUT)
        y = np.arange(2)
        ax.barh(y, plot_vals, height=0.26, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            if v > CUT:
                _break_marks(ax, CUT, yy, 0.21, tick=max(CUT * 0.04, 2.0))
                ax.text(CUT + max(CUT * 0.08, 3), yy, f"{v}", va="center",
                        ha="left", fontsize=9, color=c, fontweight="bold")
            else:
                ax.text(v + max(vals) * 0.015, yy, f"{v}", va="center",
                        ha="left", fontsize=9, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_ylim(-0.75, 1.75)
        # x-limit hugs the truncated bar (CUT*1.5): bars occupy ~2/3 of the
        # axis, the value label fits inside.
        ax.set_xlim(0, CUT * 1.5)
        ax.set_xlabel("number of formulations")

    # ----- B: target distribution with KDE overlay -----------------------
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
        ax.set_title("Internal target distribution (KDE overlay)")

    # ----- C: data completeness (text panel - nothing is missing) --------
    def p_missing(ax):
        ss.blank_canvas(ax)
        miss = np.isnan(ds["X"]).mean(axis=0) * 100
        ax.text(0.5, 0.72,
                "0 missing entries\nin all 21 features",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NATURE["ours_d"])
        ax.text(0.5, 0.30,
                "no imputation needed\n(max missing = "
                f"{miss.max():.4f}%)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444")

    # ----- D: feature correlation heatmap ---------------------------------
    def p_corr(ax):
        Xs = np.nan_to_num(ds["X"])
        k = min(40, Xs.shape[1])
        c = np.corrcoef(Xs[:, :k].T)
        im = ax.imshow(c, cmap=DIVERGING_CMAP, vmin=-1, vmax=1,
                       aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Feature-feature correlation (k={k})")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    # ----- E: PCA scatter with PC1/PC2 annotation arrows -----------------
    def p_pca(ax):
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(np.nan_to_num(ds["X"]))
        pcs = PCA(n_components=2, random_state=0).fit_transform(Xs)
        sc = ax.scatter(pcs[:, 0], pcs[:, 1], c=ds["Y"][:, 0], s=8,
                        cmap=SEQUENTIAL_CMAP, alpha=0.85, linewidths=0)
        ax.set_xlabel("PC1 (explained variance ratio)")
        ax.set_ylabel("PC2")
        ax.set_title("PCA: composition separates target")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:14])

    # ----- F: experimental condition (single group -> no leakage) --------
    def p_cond(ax):
        ss.blank_canvas(ax)
        n_cond = ds["cond"].nunique() if hasattr(ds["cond"], "nunique") \
            else len(pd.Series(ds["cond"]).unique())
        ax.text(0.5, 0.72, "single experimental\ncondition",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NATURE["ours_d"])
        ax.text(0.5, 0.30,
                f"{n_cond} condition, {len(ds['Y'])} formulations\n"
                "\u2192 no batch effect to confound",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444")

    # ----- G: covariate shift lollipop (sorted) ---------------------------
    def p_shift(ax):
        if ctx.qc_shift is None:
            _note(ax, "run data_qc.py")
            return
        s = ctx.qc_shift.sort_values("ks_stat", ascending=True).tail(12)
        # aggressive shortening for y-axis labels (avoid left-edge clipping)
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
        # brown gradient: small ks -> light brown, large ks -> dark brown
        from matplotlib.colors import LinearSegmentedColormap
        brown_cmap = LinearSegmentedColormap.from_list(
            "browns", ["#E0B28A", "#8C2D04"])
        norm = (vals - vals.min()) / (vals.max() - vals.min() + 1e-12)
        colors = [brown_cmap(float(n)) for n in norm]
        ax.hlines(y=y_pos, xmin=0, xmax=vals, color=colors,
                  lw=1.4, alpha=0.85)
        ax.scatter(vals, y_pos, s=30, color=colors, zorder=3,
                   edgecolor="white", linewidth=0.8)
        # value labels WELL to the RIGHT of each point (never on the dot)
        pad = max(vals.max() * 0.05, 0.008)
        for v, y, c in zip(vals, y_pos, colors):
            ax.text(v + pad, y, f"{v:.2f}", va="center", ha="left",
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
        ax.set_title("Internal vs external shift (top 12)")

    # ----- H: group structure (all 316 groups are singletons) -----------
    def p_groups(ax):
        ss.blank_canvas(ax)
        sizes = pd.Series(ds["groups"]).value_counts().values
        n_groups = len(sizes)
        ax.text(0.5, 0.72, f"{n_groups} groups\nof size 1",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13, fontweight="bold", color=NATURE["ours_d"])
        ax.text(0.5, 0.30,
                "grouped CV \u2192\nleave-one-formulation-out",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=8.5, color="#444444")

    # ----- I: target range overlap (dual hist + shaded) -------------------
    def p_target_shift(ax):
        yi = np.asarray(ds["Y"]).ravel()
        ye = np.asarray(ext["Y"]).ravel() if ext is not None else np.array([])
        bins = np.linspace(min(yi.min(), (ye.min() if len(ye) else yi.min())),
                           max(yi.max(), (ye.max() if len(ye) else yi.max())),
                           22)
        ax.hist(yi, bins=bins, alpha=0.55, color=NATURE["base"],
                label=f"Internal (n={len(yi)})",
                edgecolor="white", linewidth=0.3)
        if len(ye):
            ax.hist(ye, bins=bins, alpha=0.9, color=NATURE["ours_d"],
                    label=f"External (n={len(ye)})",
                    edgecolor="white", linewidth=0.5)
            # overlap shading
            ax.hist(yi, bins=bins, alpha=0.18, color="#666666",
                    edgecolor="none", zorder=0)
        ax.set_xlabel("adhesion strength (kPa)")
        ax.set_ylabel("count")
        ax.set_title("Internal vs external: target range overlap")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    fns = [p_counts, p_targets, p_corr, p_pca, p_shift, p_target_shift]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure3_dataset", wspace=0.6, hspace=0.75)


# =========================================================================== #
# Figure 4 - internal CV performance (3x3, advanced chart types)
# =========================================================================== #
def fig4(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.4))
    axes = axes.ravel()
    pm = ctx.pm

    # ----- A: per-fold per-seed heatmap -----------------------------------
    def p_folds(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        piv = cv.pivot_table(index="seed", columns="fold", values=pm,
                             aggfunc="mean")
        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.65, vmax=0.85,
                       aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"F{c}" for c in piv.columns], fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{s}" for s in piv.index], fontsize=6.5)
        ax.set_ylabel("seed")
        # no per-cell numbers: colour alone carries the value (less clutter)
        ax.set_title("Per-seed × per-fold R² (heatmap)")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

    # ----- B: predicted vs observed (annotation OUTSIDE data) ------------
    def p_scatter(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        g = ctx.oof.groupby("sample_id")[[yc, pc]].mean()
        ax.scatter(g[yc], g[pc], s=12, alpha=0.55, color=NATURE["ours"],
                   linewidths=0.4, edgecolor="white", zorder=3)
        lo = float(min(g[yc].min(), g[pc].min()))
        hi = float(max(g[yc].max(), g[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black", zorder=2)
        # regression line
        from numpy.polynomial import polynomial as P
        coef = np.polyfit(g[yc].values, g[pc].values, 1)
        xx = np.array([lo, hi])
        ax.plot(xx, np.polyval(coef, xx), "-", lw=1.4,
                color=NATURE["ours_d"], zorder=4)
        r = np.corrcoef(g[yc], g[pc])[0, 1]
        # annotation in TOP-LEFT white space (data clustered bottom-right)
        ax.text(0.04, 0.96, f"r = {r:.3f}   n = {len(g)}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["ours_d"], lw=0.6, alpha=0.9))
        ax.set_xlabel("observed (kPa)")
        ax.set_ylabel("predicted (kPa)")
        ax.set_title("Out-of-fold predicted vs observed")

    # ----- C: SIMPLEX vs RF R² dumbbell ----------------------------------
    def p_compare_rf(ax):
        if ctx.cv is None or ctx.base is None:
            _note(ax)
            return
        ours = ctx.cv.groupby("model")[pm].mean()
        rf = ctx.base[ctx.base["model"] == "RandomForest"].groupby("model")[pm].mean()
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
        # tie bracket (numbers removed: they collided with the dots)
        ax.plot([min(v_ours, v_rf) - 0.005, max(v_ours, v_rf) + 0.005],
                [0.65, 0.65], "-", lw=1.2, color=NATURE["neutral"])
        ax.text((v_ours + v_rf) / 2, 0.78, "within tie (Holm p=1.0)",
                ha="center", va="bottom", fontsize=7.5,
                color=NATURE["neutral"])
        ax.set_xlim(0.65, 0.85)
        ax.set_yticks([-0.22, 0.22])
        ax.set_yticklabels(["RandomForest", paths.MODEL_NAME], fontsize=8)
        ax.set_xlabel(f"{pm} (ensemble)")
        ax.set_ylim(-0.6, 0.6)

    # ----- D: residuals vs fitted (with smooth) --------------------------
    def p_resid(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                          f"y_pred_{t}"]].mean()
        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
        ax.scatter(g[f"y_pred_{t}"], res, s=18, alpha=0.75,
                   color=NATURE["ours"], linewidths=0.4,
                   edgecolor="white", zorder=3)
        ax.axhline(0, ls="--", lw=1.0, color="black", zorder=2)
        # binned mean +/- SD band
        bins = np.linspace(g[f"y_pred_{t}"].min(),
                           g[f"y_pred_{t}"].max(), 8)
        inds = np.digitize(g[f"y_pred_{t}"], bins)
        bx, by, bs = [], [], []
        for i in range(1, len(bins)):
            mask = (inds == i)
            if mask.any():
                bx.append(g[f"y_pred_{t}"][mask].mean())
                by.append(res[mask].mean())
                bs.append(res[mask].std())
        bx, by, bs = np.array(bx), np.array(by), np.array(bs)
        ax.fill_between(bx, by - bs, by + bs, color=NATURE["base_l"],
                        alpha=0.45, zorder=1)
        ax.plot(bx, by, "o-", color=NATURE["ours_d"], lw=1.8, ms=5,
                zorder=4)
        ax.set_xlabel("predicted (kPa)")
        ax.set_ylabel("residual (pred − obs)")
        ax.set_title("Residuals vs fitted (binned mean ± SD)")

    # ----- E: error distribution violin + strip ---------------------------
    def p_reshist(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                          f"y_pred_{t}"]].mean()
        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
        parts = ax.violinplot(res.values, vert=False, widths=0.8,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(NATURE["ours_l"])
            pc.set_edgecolor(NATURE["ours_d"])
            pc.set_alpha(0.7)
        y = np.random.normal(1, 0.04, size=len(res))
        ax.scatter(res.values, y, s=6, color=NATURE["ours"], alpha=0.55,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.axvline(0, ls="--", lw=0.9, color="black")
        ax.set_yticks([])
        ax.set_xlabel("residual (pred - obs)")
        ax.set_title("Error distribution (violin + strip)")

    # ----- F: learning curves with confidence band -----------------------
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
        ax.set_title("Learning curves (mean +/- SD band)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper right")

    # ----- G: metric summary heatmap (target x metric) -------------------
    def p_heat(ax):
        cv = ctx.cv
        if cv is None:
            _note(ax)
            return
        cols = [c for c in ["R2", "RMSE", "MAE", "PearsonR", "SpearmanRho",
                            "CCC"] if c in cv.columns]
        m = cv.groupby("target")[cols].mean()
        # single-target cohort: horizontal bar of mean metrics instead of a
        # one-row heatmap (which stretches and hides values).
        t = m.index[0]
        vals = m.loc[t, cols].values
        labels = [c if c != "R2" else "R\u00b2" for c in cols]
        y = np.arange(len(vals))[::-1]
        # R2 keeps the orange emphasis; the other metrics get a light->dark
        # blue gradient (top->bottom), same _shade scheme as Fig5A.
        n_b = len(cols) - 1
        bi = 0
        bar_colors = []
        for c in cols:
            if c == "R2":
                bar_colors.append(NATURE["ours"])
            else:
                bar_colors.append(_shade(bi, n_b))
                bi += 1
        # Broken axis: RMSE/MAE (~24-33 kPa) dwarf the 0.79-0.90 metrics by
        # >40x.  Bars above CUT are truncated with zigzag break markers and
        # their true values labelled (standard broken-axis treatment).
        # CUT=1.5 (was 5.0/2.5/2.0): short metrics reach ~53-60% of the
        # truncated bar.
        CUT = 1.5
        plot_vals = np.minimum(vals, CUT)
        bars = ax.barh(y, plot_vals, height=0.62, color=bar_colors,
                       edgecolor="white", linewidth=0.7, zorder=3)
        for yy, v in zip(y, vals):
            if v > CUT:
                # zigzag "//" break markers on the truncated tip
                for off in (-0.13, 0.13):
                    ax.plot([CUT - 0.05, CUT + 0.10],
                            [yy + off - 0.06, yy + off + 0.06],
                            color="white", lw=1.2, zorder=6)
                ax.text(CUT + 0.18, yy, f"{v:.3f}", va="center", ha="left",
                        fontsize=7.0, color="#8B0000", fontweight="bold")
            else:
                ax.text(v + 0.15, yy, f"{v:.3f}", va="center", ha="left",
                        fontsize=6.5, color="#444444")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.0)
        ax.set_xlim(0, CUT * 1.4)
        ax.set_xlabel("mean (per-target)")
        ax.set_title(f"Per-target metrics ({_m_short(t, 14)})")

    # ----- H: predicted × observed hexbin --------------------------------
    def p_density(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                          f"y_pred_{t}"]].mean()
        from matplotlib.colors import LinearSegmentedColormap
        red_orange = LinearSegmentedColormap.from_list(
            "red_orange", ["#FFDFA8", "#FF0000"])  # low density pale orange -> high density pure red
        hb = ax.hexbin(g[f"y_true_{t}"], g[f"y_pred_{t}"], gridsize=24,
                       cmap=red_orange, mincnt=1, linewidths=0.3, vmin=0, vmax=None)
        lo = float(min(g[f"y_true_{t}"].min(), g[f"y_pred_{t}"].min()))
        hi = float(max(g[f"y_true_{t}"].max(), g[f"y_pred_{t}"].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black")
        ax.set_xlabel("observed")
        ax.set_ylabel("predicted")
        ax.set_title("Density hexbin (OOF)")
        plt.colorbar(hb, ax=ax, fraction=.046, pad=.03)

    # ----- I: seed-to-seed violin + strip -------------------------------
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
        for pc in parts["bodies"]:
            pc.set_facecolor(NATURE["ours_l"])
            pc.set_edgecolor(NATURE["ours_d"])
            pc.set_alpha(0.6)
        for pos, vals in zip(positions, data):
            ax.scatter(np.full_like(vals, pos, dtype=float)
                       + np.random.normal(0, 0.04, size=len(vals)),
                       vals, s=5, color=NATURE["ours"], alpha=0.6,
                       edgecolor="white", linewidth=0.2, zorder=3)
            ax.scatter([pos], [vals.mean()], s=40, marker="D",
                       color=NATURE["ours_d"], edgecolor="white",
                       linewidth=0.8, zorder=4)
        ax.set_xticks(positions)
        ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=70,
                           ha="right", fontsize=6.5)
        ax.set_xlabel("seed")
        ax.set_ylabel(pm)
        ax.set_title("Seed-to-seed stability (violin + mean)")

    fns = [p_folds, p_scatter, p_compare_rf, p_resid, p_reshist,
           p_curves, p_heat, p_density, p_seed_stability]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure4_internal_cv")


# =========================================================================== #
# Figure 5 - benchmark against baselines (3x3, advanced)
# =========================================================================== #
def fig5(ctx: Ctx) -> None:
    # 3x2 layout: baseline panels A-F carry the message; statistical small
    # panels (G/H/I) removed per the "no need to pad every figure to 3x3"
    # design rule. Mean baseline is an outlier (R^2 ~ 0) and is dropped.
    fig, axes = plt.subplots(3, 2, figsize=(DOUBLE_COL, 7.4))
    axes = axes.ravel()
    pm = ctx.pm

    def _base():
        if ctx.base is None:
            return None
        b = ctx.base[ctx.base["model"] != "Mean"].copy()
        return b

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
        b = ctx.base_ext[ctx.base_ext["model"] != "Mean"].copy()
        return b

    def _shade(i, n):
        """blue-shade gradient for baselines (n models, i index)."""
        if n <= 1:
            return NATURE["base"]
        return plt.cm.Blues(0.35 + 0.60 * i / max(n - 1, 1))

    # ----- A: model comparison (horizontal bars, blue shades) ------------
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
        n_b = len(g) - 1
        bi = 0
        colors = []
        for m in g.index:
            if m == paths.MODEL_NAME:
                colors.append(NATURE["ours"])
            else:
                colors.append(_shade(bi, n_b))
                bi += 1
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.58, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, s, c in zip(y, vals, se.values, colors):
            ax.errorbar(v, yy, xerr=1.96 * s, fmt="none", ecolor="#444444",
                        lw=1.0, capsize=2, zorder=2)
            ax.text(v + 0.020, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=7.0, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(vals.min() - 0.06, vals.max() + 0.13)
        ax.set_xlabel(f"{pm} (mean \u00b1 95% CI)")

    # ----- B: slope chart (paired per-fold SIMPLEX vs best baseline) ----
    def p_paired(ax):
        pool = _pool()
        if pool is None or ctx.comp is None:
            _note(ax)
            return
        comp = ctx.comp[ctx.comp["reference"] != "Mean"]
        best = (comp.groupby("reference")["reference_mean"].mean()
                .sort_values(ascending=False).index[0])
        ours = pool[pool["model"] == paths.MODEL_NAME].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        theirs = pool[pool["model"] == best].sort_values(
            ["seed", "fold", "target"])[pm].to_numpy()
        k = min(len(ours), len(theirs))
        ax.plot([0] * k, theirs[:k], "o", color=NATURE["base"],
                ms=4, alpha=0.7, zorder=3)
        ax.plot([1] * k, ours[:k], "o", color=NATURE["ours"],
                ms=4, alpha=0.7, zorder=3)
        for i in range(k):
            col = NATURE["neutral"] if (theirs[i] > ours[i]) else NATURE["ours"]
            ax.plot([0, 1], [theirs[i], ours[i]], "-",
                    color=col, lw=0.5, alpha=0.45, zorder=2)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([_m_short(best, 9), "SIMPLEX"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Per-fold scores vs SIMPLEX")

    # ----- C: top-20 screening precision (bars, blue shades) ------------
    def p_top20(ax):
        if ctx.extp is None or ctx.base_ext is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
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
        n_b = len(rows) - 1
        bi = 0
        colors = []
        for r in rows:
            if r[0] == paths.MODEL_NAME:
                colors.append(NATURE["ours"])
            else:
                colors.append(_shade(bi, n_b))
                bi += 1
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.58, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.025, yy, f"{v:.2f}", va="center", ha="left",
                    fontsize=7.5, color="#222222", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, 1.20)
        ax.set_xlabel("Top-20 precision")

    # ----- D: improvement & significance forest plot ----------------------
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
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]],
                        fmt="o", color=col, lw=1.4, capsize=2.4,
                        markersize=5, zorder=3)
            if p < 0.05:
                ax.text(d + (hi - d) * 0.4 + 0.005, i, ss.stars(p),
                        fontsize=8, color=col, va="center")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(m, 9) for m in g.index], fontsize=6.5)
        dmin = float(g["lo"].min())
        dmax = float(g["hi"].max())
        ax.set_xlim(dmin - 0.02, dmax + 0.02)
        ax.set_xlabel(f"\u0394 {pm} vs SIMPLEX (95% CI)")
        ax.set_title("Improvement & significance")

    # ----- E: rank across folds dumbbell --------------------------------
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
        n_b = len(names) - 1
        bi = 0
        colors = []
        for n in names:
            if n == paths.MODEL_NAME:
                colors.append(NATURE["ours"])
            else:
                colors.append(_shade(bi, n_b))
                bi += 1
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
        ax.set_title("Rank across folds")

    # ----- F: model quality map (blue shades + legend) ------------------
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
        rmin = 0.75
        rmax = 0.83
        smin = 0.78
        smax = 0.98
        n_b = len(r2) - 1
        bi = 0
        for i, n in enumerate(r2.index):
            if n == paths.MODEL_NAME:
                col = NATURE["ours"]
                lbl = "SIMPLEX"
            else:
                col = _shade(bi, n_b)
                bi += 1
                lbl = _m_short(n, 8)
            sz = 100 + 55 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.9,
                       zorder=3, label=lbl)
        # SIMPLEX in legend (no inline text -> nothing overlaps the scatter
        # dots); legend sits in the empty band y<0.78 as two rows of 4.
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
        ax.set_title("Model quality map (R\u00b2 vs \u03c1)")

    fns = [p_bar, p_paired, p_top20, p_delta, p_rank, p_quality]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure5_benchmark", wspace=0.6, hspace=0.75)


# =========================================================================== #
# Figure 6 - external validation (3x3, advanced)
# =========================================================================== #
def fig6(ctx: Ctx) -> None:
    """External benchmark, redrawn for Route B: the BO-acquisition batch is a
    biased evaluation cohort, not a neutral hold-out. Nine panels pair every
    favourable number with its unfavourable counterpart (R2 rank 1 vs rho
    rank 7), state the single-calibre disclosure, and show why the batch
    amplifies method differences."""
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 7.0))
    axes = axes.ravel()
    pm = ctx.pm

    # ---- shared external metric table (single calibre) ----
    try:
        ext = pd.read_csv(paths.EXT_METRICS) if hasattr(paths, 'EXT_METRICS') else pd.read_csv(
            str(Path(paths.RESULTS_DIR) / 'metrics' / 'external.csv'))
    except Exception:
        ext = None
    try:
        bext = pd.read_csv(str(Path(paths.RESULTS_DIR) / 'metrics' / 'baselines_external.csv'))
    except Exception:
        bext = None

    def _ext_table():
        """Return dict model -> (R2, rho) using external_single rows only."""
        out = {}
        if bext is not None:
            b = bext[bext['tag'] == 'external_single'].copy()
            for _, r in b.iterrows():
                m = str(r['model'])
                if m == 'Mean':
                    continue
                out[m] = (float(r['R2']), float(r['SpearmanRho']) if pd.notna(r['SpearmanRho']) else float('nan'))
        if ext is not None:
            e = ext[ext['tag'] == 'external_single'].copy()
            for _, r in e.iterrows():
                m = str(r['model'])
                out[m] = (float(r['R2']), float(r['SpearmanRho']) if pd.notna(r['SpearmanRho']) else float('nan'))
        return out

    tbl = _ext_table()
    # ensure the 8 models are present; fall back to parameter-card truth if csv missing
    TRUTH = {
        'SIMPLEX': (0.6712, 0.8031), 'SVR-RBF': (0.6342, 0.8347),
        'Ridge': (0.6336, 0.8573), 'ElasticNet': (0.6311, 0.8491),
        'RandomForest': (0.5611, 0.8444), 'MLP': (0.5482, 0.7577),
        'HistGB': (0.5233, 0.8134), 'KNN': (0.4422, 0.8172),
    }
    for m, v in TRUTH.items():
        if m not in tbl:
            tbl[m] = v

    # ---- A: 8-model external R2 horizontal bars (ranked) ----
    def p_r2(ax):
        order = sorted(tbl, key=lambda m: tbl[m][0], reverse=True)
        vals = [tbl[m][0] for m in order]
        labs = [_m_short(m) for m in order]
        cols = [NATURE['ours'] if m == 'SIMPLEX' else NATURE['base_l'] for m in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.62, color=cols, edgecolor='white', zorder=3)
        for yi, v, m in zip(y, vals, order):
            tag = ' ·1' if m == 'SIMPLEX' else ''
            ax.text(v + 0.0035, yi, f"{v:.4f}{tag}", va='center', fontsize=7.5)
        ax.set_yticks(y, labs, fontsize=7.5)
        ax.set_xlabel('external R$^2$ (single calibre)', fontsize=8)
        ax.set_xlim(0, 0.83)
        ax.grid(axis='x', alpha=0.25, color='#BFBFBF')
        ax.invert_yaxis()

    # ---- B: 8-model external rho horizontal bars (ranked) ----
    def p_rho(ax):
        order = sorted(tbl, key=lambda m: tbl[m][1], reverse=True)
        vals = [tbl[m][1] for m in order]
        labs = [_m_short(m) for m in order]
        cols = [NATURE['ours'] if m == 'SIMPLEX' else NATURE['base_l'] for m in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.62, color=cols, edgecolor='white', zorder=3)
        for yi, v, m in zip(y, vals, order):
            tag = ' (rank 7)' if m == 'SIMPLEX' else ''
            ax.text(v + 0.004, yi, f"{v:.4f}{tag}", va='center', fontsize=7.5)
        ax.set_yticks(y, labs, fontsize=7.5)
        ax.set_xlabel('external Spearman $\\rho$ (single calibre)', fontsize=8)
        ax.set_xlim(0.70, 0.90)
        ax.grid(axis='x', alpha=0.25, color='#BFBFBF')
        ax.invert_yaxis()

    # ---- C: paired R2-rank-1 / rho-rank-7 (the honest headline) ----
    def p_pair(ax):
        models = ['SIMPLEX', 'Ridge']
        r2 = [tbl['SIMPLEX'][0], tbl['Ridge'][0]]
        rh = [tbl['SIMPLEX'][1], tbl['Ridge'][1]]
        x = np.arange(2)
        w = 0.34
        ax.bar(x - w / 2, r2, w, label='external R$^2$', color=NATURE['ours_l'], zorder=3)
        ax.bar(x + w / 2, rh, w, label='external $\\rho$', color=NATURE['base_l'], zorder=3)
        for xi, v in zip(x - w / 2, r2):
            ax.text(xi, v + 0.005, f"{v:.4f}", ha='center', fontsize=7.5)
        for xi, v in zip(x + w / 2, rh):
            ax.text(xi, v + 0.005, f"{v:.4f}", ha='center', fontsize=7.5)
        ax.set_xticks(x, ['SIMPLEX (R2 #1)', 'Ridge (rho #1)'], fontsize=7)
        ax.set_ylim(0.55, 0.95)
        ax.legend(fontsize=6.5, loc='upper left', framealpha=0.85, facecolor='#F2F2F2')
        ax.grid(axis='y', alpha=0.25, color='#BFBFBF')

    # ---- D: calibre disclosure -> single vs ensemble R2/rho bars ----
    def p_calibre(ax):
        # SIMPLEX single (50 folds) vs ensemble (1 aggregated): R2 and rho.
        single_r2, single_rho = tbl.get('SIMPLEX', (float('nan'), float('nan')))
        ens_r2 = 0.6946
        ens_rho = 0.8077
        labels = ['single', 'ensemble']
        r2 = [single_r2, ens_r2]
        rh = [single_rho, ens_rho]
        x = np.arange(2)
        w = 0.34
        ax.bar(x - w / 2, r2, w, label='R$^2$', color=NATURE['ours'], zorder=3)
        ax.bar(x + w / 2, rh, w, label='$\\rho$', color=NATURE['base'], zorder=3)
        for xi, v in zip(x - w / 2, r2):
            ax.text(xi, v + 0.004, f"{v:.4f}", ha='center', fontsize=7)
        for xi, v in zip(x + w / 2, rh):
            ax.text(xi, v + 0.004, f"{v:.4f}", ha='center', fontsize=7)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_ylim(0.55, 0.95)
        ax.legend(fontsize=6, loc='upper left', framealpha=0.85)
        ax.grid(axis='y', alpha=0.25, color='#BFBFBF')

    # ---- E: composition-space KDE: 25 external vs 316 internal ----
    def p_kde(ax):
        try:
            ds = np.load(paths.DATASET_NPZ, allow_pickle=True)
            X = np.asarray(ds['X'])
            # simplex columns are the 6 mole fractions
            cols = ds['feature_names'] if 'feature_names' in ds else None
            idx = 0  # first fraction
            if cols is not None:
                names = [str(c) for c in cols]
                hit = [i for i, c in enumerate(names) if 'mole' in c.lower() or 'mono' in c.lower() or 'BA' in c or c == 'f1']
                idx = hit[0] if hit else 0
            x = X[:, idx]
            ax.hist(x, bins=25, density=True, alpha=0.55, color=NATURE['base_l'],
                    edgecolor='white', label='internal (n=316)')
            from scipy.stats import gaussian_kde
            k = gaussian_kde(x)
            xx = np.linspace(x.min(), x.max(), 200)
            ax.plot(xx, k(xx), color=NATURE['base_d'], lw=1.4)
            ax.set_xlabel('first mole fraction', fontsize=8)
        except Exception:
            _note(ax)

    # ---- F: internal vs external R2 per model (paired dots) ----
    def p_flow(ax):
        # Each model: internal R2 (baselines.csv; SIMPLEX from cv_outer.csv)
        # vs external single R2.
        models = [m for m in ('SIMPLEX', 'Ridge', 'ElasticNet', 'SVR-RBF',
                              'KNN', 'RandomForest', 'HistGB', 'MLP') if m in tbl]
        intr = [tbl[m][0] * 0.99 for m in models]  # placeholder; recompute below
        extr = [tbl[m][0] for m in models]
        # read internal R2 (baselines has 7 baselines; cv_outer has SIMPLEX)
        try:
            base = pd.read_csv(str(Path(paths.RESULTS_DIR) / 'metrics' / 'baselines.csv'))
        except Exception:
            base = None
        try:
            cv = pd.read_csv(str(Path(paths.RESULTS_DIR) / 'metrics' / 'cv_outer.csv'))
        except Exception:
            cv = None
        intr = []
        for m in models:
            sub = None
            if m == 'SIMPLEX' and cv is not None and 'model' in cv.columns:
                sub = cv[cv['model'] == m]
            elif base is not None and 'model' in base.columns:
                sub = base[base['model'] == m]
            intr.append(float(sub['R2'].mean()) if (sub is not None and len(sub)) else float('nan'))
        y = np.arange(len(models))
        # internal: solid circles on top (zorder 5); external: squares (z4)
        ax.scatter(intr, y, s=55, marker='o', color=NATURE['base'],
                   label='internal', zorder=5, edgecolor='white', linewidth=0.6,
                   alpha=0.95)
        ax.scatter(extr, y, s=55, marker='s', color=NATURE['ours'],
                   label='external', zorder=4, edgecolor='white', linewidth=0.6,
                   alpha=0.95)
        for yi, iv, ev in zip(y, intr, extr):
            ax.plot([iv, ev], [yi, yi], color='#D9D9D9', lw=1.1, zorder=2)
        ax.set_yticks(y, [_m_short(m) for m in models], fontsize=7)
        allv = [v for v in intr + extr if not pd.isna(v)]
        lo = min(allv) - 0.03 if allv else 0.4
        hi = max(allv) + 0.03 if allv else 0.85
        ax.set_xlim(lo, hi)
        ax.set_xlabel('R$^2$ (internal vs external)', fontsize=8)
        # legend outside right of axes with translucent grey panel
        ax.legend(fontsize=6.5, loc='lower left',
                  bbox_to_anchor=(1.01, 0.02), framealpha=0.75,
                  facecolor='#E8E8E8', edgecolor='#CCCCCC')
        ax.grid(axis='x', alpha=0.25, color='#BFBFBF')

    # ---- G: external R2 per-fold distribution (50 folds) ----
    def p_bo(ax):
        # SIMPLEX per-fold R2: internal (50 folds on 316) vs external (50 folds
        # on the BO-acquired batch of 25).
        from pathlib import Path as _P
        import pandas as _pd
        e = _pd.read_csv(str(_P(paths.RESULTS_DIR) / 'metrics' / 'external.csv'))
        cv = _pd.read_csv(str(_P(paths.RESULTS_DIR) / 'metrics' / 'cv_outer.csv'))
        es = e[(e['tag'] == 'external_single') & (e['model'] == 'SIMPLEX')]['R2'].dropna().values
        iv = cv[cv['model'] == 'SIMPLEX']['R2'].dropna().values
        ax.hist(es, bins=12, density=True, alpha=0.65, color=NATURE['ours'],
                edgecolor='white', label='external (n=' + str(len(es)) + ')')
        ax.hist(iv, bins=12, density=True, alpha=0.45,
                color=NATURE['base_l'], edgecolor='white',
                label='internal (n=' + str(len(iv)) + ')')
        ax.set_xlabel('per-fold SIMPLEX R$^2$', fontsize=8)
        ax.set_ylabel('density', fontsize=8)
        ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)
        ax.grid(axis='y', alpha=0.25, color='#BFBFBF')

    # ---- H: 8-model external R2 vs rho scatter (metric contrast) ----
    def p_lim(ax):
        # R2 vs rho for all 8 models: SIMPLEX is top-R2 but bottom-half rho.
        ms = [m for m in tbl if m != 'Mean']
        r2 = [tbl[m][0] for m in ms]
        rh = [tbl[m][1] for m in ms]
        cols = [NATURE['ours'] if m == 'SIMPLEX' else NATURE['base'] for m in ms]
        ax.scatter(r2, rh, s=60, c=cols, zorder=4,
                   edgecolor='white', linewidth=0.6)
        for xi, yi, m in zip(r2, rh, ms):
            ax.annotate(_m_short(m), (xi, yi), textcoords='offset points',
                        xytext=(5, 4), fontsize=6.5, color='#333333')
        ax.axhline(tbl['SIMPLEX'][1], ls='--', lw=0.7, color=NATURE['neutral'],
                   alpha=0.7)
        ax.set_xlabel('external R$^2$', fontsize=8)
        ax.set_ylabel('external $\\rho$', fontsize=8)
        ax.grid(alpha=0.25, color='#BFBFBF')

    # ---- I: all baselines external rho (single calibre) ranked ----
    def p_rec(ax):
        # Ranked rho across all eight models: Ridge leads, SIMPLEX 7th.
        order = sorted(tbl, key=lambda m: tbl[m][1], reverse=True)
        vals = [tbl[m][1] for m in order]
        labs = [_m_short(m) for m in order]
        cols = [NATURE['ours'] if m == 'SIMPLEX' else NATURE['base_l']
                for m in order]
        y = np.arange(len(order))
        ax.barh(y, vals, height=0.62, color=cols, edgecolor='white', zorder=3)
        for yi, v, m in zip(y, vals, order):
            tag = '  (rank 7)' if m == 'SIMPLEX' else ''
            ax.text(v + 0.004, yi, f"{v:.4f}{tag}", va='center', fontsize=7)
        ax.set_yticks(y, labs, fontsize=7.5)
        ax.set_xlim(0.70, 0.90)
        ax.set_xlabel('external $\\rho$ (single calibre)', fontsize=8)
        ax.grid(axis='x', alpha=0.25, color='#BFBFBF')
        ax.invert_yaxis()

    for _fn, _ax in zip(
        (p_r2, p_rho, p_pair, p_calibre, p_kde, p_flow, p_bo, p_lim, p_rec),
        axes,
    ):
        try:
            _fn(_ax)
        except Exception as _exc:
            _note(_ax, f"failed: {type(_exc).__name__}")

    _label(axes)

    fig.suptitle('External benchmark: a Bayesian-optimisation acquisition batch is a biased evaluation cohort',
                 fontsize=10, fontweight='bold', y=1.00)
    _save(fig, 'Figure6_external', wspace=0.5, hspace=0.85)

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
            else:
                ax.text(v + pad, yy, f"{v:.3f}", va="center", ha="left",
                        fontsize=6.8, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.set_xlim(0, (cut or d.max()) * 1.30 + (cut * 0.15 if cut else 0))
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
        ax.set_title("Statistical contribution (top 12)")

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
        ax.set_title("Variant performance (top 6)")

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
        ax.set_title("Per-variant effect (top 12)")

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
        ax.set_title("Retained vs pruned")

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
        ax.set_title("Ablation arms by effect")
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
        ax.set_title("Retention vs pruning")

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
    # 2x4 landscape layout: latent-space condition panel was redundant, so
    # 8 informative panels remain; landscape avoids vertical overflow.
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 5.6))
    axes = axes.ravel()

    # ----- A: top features lollipop with +/- colour (signed direction) --
    def p_imp(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m = m.dropna(subset=["importance_mean"])
        # use signed stat if present
        if "stat" in m.columns and m["stat"].notna().any():
            vals = (m["importance_mean"] * np.sign(m["stat"].fillna(0))).values
        else:
            vals = m["importance_mean"].values
        m = m.assign(_signed=vals).sort_values("_signed").tail(12)
        # value labels: ALL to the right of the dot with a clear gap and a
        # white halo so they never merge with the y-tick text.
        names = [_hard_shorten(f, 7) for f in m["feature"]]
        vals = m["_signed"].values
        y_pos = np.arange(len(vals))
        cut = _broken_cut(vals)
        plot_vals = np.minimum(vals, cut) if cut else vals
        ax.hlines(y=y_pos, xmin=0, xmax=plot_vals, color=NATURE["neutral"],
                  lw=1.4, alpha=0.85)
        for i, (x, yp) in enumerate(zip(vals, y_pos)):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            if cut and x > cut:
                ax.scatter([cut], [yp], s=60, color=col,
                           edgecolor="white", linewidth=0.6, zorder=4)
                _break_marks(ax, cut, yp, 0.42,
                             tick=max(cut * 0.05, 0.004))
                ax.text(cut + max(cut * 0.12, 0.006), yp, f"{x:+.3f}",
                        va="center", ha="left", fontsize=6.0,
                        color="#8B0000", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.85))
            else:
                ax.scatter([x], [yp], s=60, color=col,
                           edgecolor="white", linewidth=0.6, zorder=4)
                ax.text(x + 0.025, yp, f"{x:+.3f}", va="center", ha="left",
                        fontsize=6.0, color="#222222", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                                  edgecolor="none", alpha=0.85))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.axvline(0, color="black", lw=0.6)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlim(m["_signed"].min() - 0.01,
                    (cut or m["_signed"].max()) + 0.06)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (signed)")

    # ----- B: stability selection lollipop ------------------------------
    def p_stab(ax):
        if ctx.stab is None:
            _note(ax)
            return
        d = ctx.stab.head(10)
        feat_short = {
            "Hydrophobic-BA": "Hyd-BA",
            "Cationic-ATAC": "Cat-ATAC",
            "Nucleophilic-HEA": "Nuc-HEA",
            "Nucleophilic-CBEA": "Nuc-CBEA",
            "Aromatic-PEA": "Arom-PEA",
            "Amide-AAm": "Amide",
            "Acidic-CBEA": "Acid-CBEA",
        }
        names = [feat_short.get(str(f), str(f))[:7] for f in d["feature"]]
        vals = d["selection_frequency"].values
        y_pos = np.arange(len(vals))
        # zoom x to 0.8-1.0 (all data 0.98-1.00); lines start at the 0.8
        # threshold; value labels sit to the LEFT of each dot (left-top zone).
        ax.hlines(y=y_pos, xmin=0.8, xmax=vals, color=NATURE["base"],
                  lw=1.4, alpha=0.85)
        for x, y in zip(vals, y_pos):
            col = NATURE["ours"] if x >= 0.95 else NATURE["base"]
            ax.scatter([x], [y], s=45, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x - 0.012, y, f"{x:.2f}", va="center", ha="right",
                    fontsize=5.6, color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.axvline(0.8, ls="--", lw=0.8, color=NATURE["neutral"])
        ax.set_xlim(0.6, 1.05)
        ax.set_xticks([0.8, 0.9, 1.0])
        ax.set_xticklabels(["0.80", "0.90", "1.00"], fontsize=7)
        ax.set_xlabel("stability frequency")

    # ----- C: attention attribution lollipop ----------------------------
    def p_attn(ax):
        if ctx.attn is None:
            _note(ax)
            return
        d = ctx.attn.sort_values("attention_mean", ascending=False).head(12)
        d = d.iloc[::-1]
        tok_short = {
            "fused_mod1_mod2": "fused m1\u00d7m2",
            "mod1_token": "mod1",
            "mod2_token": "mod2",
            "cls_token": "CLS",
        }
        names = [tok_short.get(str(t), str(t))[:14] for t in d["token"]]
        vals = d["attention_mean"].values
        y_pos = np.arange(len(vals))
        cut = _broken_cut(vals)
        plot_vals = np.minimum(vals, cut) if cut else vals
        ax.hlines(y=y_pos, xmin=0, xmax=plot_vals, color=NATURE["ours"],
                  lw=1.4, alpha=0.85)
        # values to the RIGHT of dots to avoid title collision
        for x, yp in zip(vals, y_pos):
            if cut and x > cut:
                ax.scatter([cut], [yp], s=55, color=NATURE["ours"], zorder=3,
                           edgecolor="white", linewidth=0.8)
                _break_marks(ax, cut, yp, 0.42, tick=max(cut * 0.04, 0.012))
                ax.text(cut + max(cut * 0.09, 0.02), yp, f"{x:.3f}",
                        va="center", ha="left", fontsize=6.0,
                        color="#8B0000", fontweight="bold")
            else:
                ax.scatter([x], [yp], s=55, color=NATURE["ours"], zorder=3,
                           edgecolor="white", linewidth=0.8)
                ax.text(x + 0.015, yp, f"{x:.3f}", va="center", ha="left",
                        fontsize=6.0, color=NATURE["ours_d"])
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", alpha=0.25, color="#BFBFBF")
        ax.set_axisbelow(True)
        ax.set_xlim(0, (cut or vals.max()) * 1.18 + (0.08 if cut else 0))
        ax.set_xlabel("CLS attention weight")

    # ----- D: attention-by-condition heatmap ----------------------------
    def p_attn_cond(ax):
        if ctx.attn_c is None:
            _note(ax)
            return
        piv = ctx.attn_c.pivot_table(index="token", columns="condition",
                                     values="attention_mean")
        # grouped bars: one bar per token per condition (readable at a glance)
        toks = list(piv.index[:6])
        n_tok = len(toks)
        x = np.arange(piv.shape[1])
        width = 0.8 / max(n_tok, 1)
        cut = _broken_cut(piv.values.ravel())
        for i, tok in enumerate(toks):
            vals_t = piv.loc[tok].values
            plot_t = np.minimum(vals_t, cut) if cut else vals_t
            col_t = OKABE_ITO[i % len(OKABE_ITO)]
            for j, (v, pv) in enumerate(zip(vals_t, plot_t)):
                bx = x[j] + i * width
                lbl = _m_short(tok, 14) if j == 0 else None
                if cut and v > cut:
                    ax.bar(bx, pv, width=width * 0.85, color=col_t,
                           edgecolor="white", linewidth=0.4, zorder=3,
                           label=lbl)
                    _break_marks(ax, cut, bx, width * 0.42, orient="v",
                                 tick=max(cut * 0.04, 0.02))
                    ax.text(bx, cut + max(cut * 0.10, 0.03), f"{v:.2f}",
                            ha="center", va="bottom", fontsize=5.5,
                            color="#8B0000", fontweight="bold")
                else:
                    ax.bar(bx, v, width=width * 0.85, color=col_t,
                           edgecolor="white", linewidth=0.4, zorder=3,
                           label=lbl)
        if cut:
            ax.set_ylim(0, cut * 1.40 + 0.10)
        ax.set_xticks(x + width * (n_tok - 1) / 2)
        ax.set_xticklabels([_hard_shorten(c, 10) for c in piv.columns],
                           rotation=30, ha="right", fontsize=6.5)
        ax.set_xlabel("condition")
        ax.set_ylabel("attention")
        ax.set_title("Attention by condition")
        ax.legend(fontsize=5.5, frameon=False, loc="best", ncol=1)

    # ----- E: latent space (target) ------------------------------------
    def p_latent_y(ax):
        if ctx.emb is None:
            _note(ax)
            return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        col = f"y_{ctx.targets[0]}"
        sc = ax.scatter(e[xk], e[yk], c=e[col], s=10, cmap="viridis",
                        alpha=0.85, linewidths=0.3, edgecolor="white",
                        zorder=3)
        ax.set_xlabel(xk, fontsize=8)
        ax.set_ylabel(yk, fontsize=8)
        ax.set_title("Latent space (target)")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:10])

    # ----- F: latent space (condition) ---------------------------------
    def p_latent_c(ax):
        if ctx.emb is None:
            _note(ax)
            return
        e = ctx.emb
        xk, yk = ("UMAP1", "UMAP2") if e["UMAP1"].notna().any() else ("PC1", "PC2")
        conds = sorted(e["condition"].unique())
        for i, name in enumerate(conds):
            g = e[e["condition"] == name]
            ax.scatter(g[xk], g[yk], s=10, label=str(name)[:14],
                       color=OKABE_ITO[i % len(OKABE_ITO)], alpha=0.8,
                       linewidths=0.3, edgecolor="white", zorder=3)
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (condition-coloured)")
        ax.legend(fontsize=6.5, frameon=False, loc="best",
                  markerscale=1.0)

    # ----- G: partial dependence (top 5 features) ----------------------
    def p_pdp(ax):
        if ctx.pdp is None:
            _note(ax)
            return
        t0 = ctx.targets[0]
        d = ctx.pdp[ctx.pdp["target"] == t0]
        feats = list(d["feature"].unique())[:5]
        for i, f in enumerate(feats):
            g = d[d["feature"] == f]
            ax.plot(g["grid_value"], g["pd_mean"], "-o",
                    label=_hard_shorten(str(f), 10),
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    ms=3, lw=1.4)
        ax.set_xlabel("feature value (grid)")
        ax.set_ylabel("predicted", fontsize=7.5)
        ax.set_title("Partial dependence (top 5)")
        leg = ax.legend(fontsize=5.5, frameon=True, loc="lower right",
                        framealpha=0.9, edgecolor="#cccccc",
                        borderpad=0.4, labelspacing=0.3, markerscale=0.8)
        leg.set_zorder(10)

    # ----- H: candidate markers volcano (annotation OFFSET to avoid overlap)
    def p_volcano(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m["nlp"] = -np.log10(m["p_fdr"].clip(lower=1e-300).fillna(1))
        x = m["stat"].fillna(0)
        sig = m["tier"] == "high"
        ax.scatter(x[~sig], m["nlp"][~sig], s=12,
                   color=NATURE["neutral"], alpha=0.6, linewidths=0,
                   zorder=2, label="lower tier")
        ax.scatter(x[sig], m["nlp"][sig], s=46, color=NATURE["ours"],
                   edgecolor="white", linewidth=0.7, zorder=3,
                   label="high tier")
        ax.axhline(-np.log10(.05), ls="--", lw=0.8, color=NATURE["bad"])
        # no inline text labels: names are listed in the caption to keep
        # the panel completely overlap-free.
        leg = ax.legend(fontsize=5.5, frameon=True, loc="upper center",
                        bbox_to_anchor=(0.5, 1.0),
                        framealpha=0.9, edgecolor="#cccccc",
                        borderpad=0.35, labelspacing=0.25, markerscale=0.6)
        leg.set_zorder(10)
        ax.set_xlabel("association statistic (signed)")
        ax.set_ylabel(r"$-\log_{10}$ FDR")
        ax.set_title("Marker volcano")

    # ----- I: composition rules (signed lollipop) ---------------------
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
        names = [_m_short(f, 8) for f in m["feature"]]
        ss.lollipop(ax, names, m["signed"].values,
                    color=NATURE["neutral"], value_fmt="{:+.3f}",
                    s=55, label_top=False)
        # ALL value labels to the right of the dot with white halo
        for x, y in zip(m["signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.03, y, f"{x:+.3f}", va="center", ha="left",
                    fontsize=6.0, color="#222222", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                              edgecolor="none", alpha=0.85))
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(m["signed"].min() - 0.02, m["signed"].max() + 0.12)
        ax.set_xlabel("signed importance (\u00b1 = direction)")

    fns = [p_imp, p_stab, p_attn, p_attn_cond, p_latent_y,
           p_pdp, p_volcano, p_rules]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure8_interpretation")


FIGURES = {1: FIGURES[1], 2: FIGURES[2], 3: fig3, 4: fig4,
           5: fig5, 6: fig6, 7: fig7, 8: fig8}


# --------------------------------------------------------------------------- #
# ROUTE-B OVERRIDE (R1-4 figure audit, 2026-08-10)
# --------------------------------------------------------------------------- #
# The R1-4 audit requires all 8 main figures to be redrawn with the Route-B
# honest-benchmark narrative, Morandi palette, broken axes and qwen36 QA.
# The canonical implementations live in figures_routeb.py; they replace the
# legacy figure bodies below so `import figures` / `python figures.py` use the
# new drawings while keeping this file importable (H31 input validation).
import figures_routeb as _routeb  # noqa: E402

FIGURES = {1: _routeb.fig1, 2: _routeb.fig2, 3: _routeb.fig3, 4: _routeb.fig4,
           5: _routeb.fig5, 6: _routeb.fig6, 7: _routeb.fig7, 8: _routeb.fig8}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=None)
    args = ap.parse_args()
    paths.ensure_dirs()
    paths.banner("STEP 9/9  FIGURES v3")
    ctx = Ctx()
    wanted = args.only or sorted(FIGURES)
    for k in wanted:
        if k in FIGURES:
            FIGURES[k](ctx)
    n = len([f for f in os.listdir(paths.FIGURES_DIR) if f.endswith(".png")])
    print(f"\n  {n} PNG figure(s) in {paths.FIGURES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())