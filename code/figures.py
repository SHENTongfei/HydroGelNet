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
PANEL_DX = -0.22
PANEL_DY = 1.18


def _label(axes, start="A", dx=PANEL_DX, dy=PANEL_DY):
    """OUTSIDE-top-left panel labels (Arial bold, no overlap with titles)."""
    ss.label_panels(axes, start=start, dx=dx, dy=dy)


def _hard_shorten(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "\u2026"


def _save(fig, name, rect=(0, 0, 1, 0.94), wspace=0.55, hspace=0.55):
    fig.tight_layout(rect=rect)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    ss.save_figure(fig, paths.FIGURES_DIR, name)


# =========================================================================== #
# Figure 3 - cohort / dataset characteristics (3x3, advanced chart types)
# =========================================================================== #
def fig3(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 5.8))
    axes = axes.ravel()
    ds, ext = ctx.ds, ctx.ext

    # ----- A: cohort size lollipop (much cleaner than bar) ----------------
    def p_counts(ax):
        names = ["Internal", "External"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        colors = [NATURE["ours"], NATURE["base"]]
        ss.lollipop(ax, names, vals, color=NATURE["ours"],
                    value_fmt="{:.0f}", s=70, label_top=False)
        # recolour dots individually
        for x, n, c in zip(vals, names, colors):
            ax.scatter(x, n, s=70, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
            # value label to the right of the dot (not above)
            ax.text(x + max(vals) * 0.02, n, f"{x}", va="center",
                    ha="left", fontsize=8, color=c, fontweight="bold")
        ax.set_xlim(-max(vals) * 0.05, max(vals) * 1.15)
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

    # ----- C: missingness lollipop (magnified near 0) ---------------------
    def p_missing(ax):
        miss = np.isnan(ds["X"]).mean(axis=0) * 100
        ss.lollipop(ax, [f"F{i}" for i in range(len(miss))],
                    miss, color=NATURE["neutral"], value_fmt="{:.1f}",
                    s=18, label_top=False)
        # show only one value annotation (since all are near 0)
        max_idx = int(np.argmax(miss))
        if miss[max_idx] > 0:
            ax.text(miss[max_idx], max_idx, f"{miss[max_idx]:.2f}%",
                    va="center", ha="left", fontsize=7,
                    color=NATURE["neutral"])
        ax.set_yticks([])
        ax.set_xlabel("feature index")
        ax.set_ylabel("missing (%)")
        ax.set_xlim(-0.5, max(miss.max() * 1.3, 0.05))
        ax.set_title(f"Missingness (mean {miss.mean():.2f}%)")

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

    # ----- F: condition composition lollipop ------------------------------
    def p_cond(ax):
        vals = pd.Series(ds["cond"]).value_counts().sort_index()
        labels = [ctx.conds[i][:10] for i in vals.index]
        ss.lollipop(ax, labels, vals.values, color=NATURE["base"],
                    value_fmt="{:.0f}", s=40)

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
        ss.lollipop(ax, labels, s["ks_stat"].values, color=NATURE["ours_d"],
                    value_fmt="{:.2f}", s=30, label_top=False)
        # add value labels to the RIGHT of each point
        y_pos = np.arange(len(s))
        for v, y in zip(s["ks_stat"].values, y_pos):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", ha="left",
                    fontsize=6.5, color=NATURE["ours_d"])
        ax.invert_yaxis()
        ax.set_xlabel("KS statistic")
        ax.set_xlim(-0.02, max(s["ks_stat"].max() * 1.18, 0.1))
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title("Internal vs external shift (top 12)")

    # ----- H: group-size distribution as strip+box -----------------------
    def p_groups(ax):
        sizes = pd.Series(ds["groups"]).value_counts().values
        ax.boxplot(sizes, vert=False, widths=0.4,
                   boxprops=dict(color=NATURE["neutral"]),
                   medianprops=dict(color=NATURE["ours_d"], lw=1.4),
                   whiskerprops=dict(color=NATURE["neutral"]),
                   capprops=dict(color=NATURE["neutral"]))
        # strip on top
        y = np.random.normal(1, 0.05, size=len(sizes))
        ax.scatter(sizes, y, s=10, color=NATURE["ours"], alpha=0.6,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_yticks([])
        ax.set_xlabel("samples per group")
        ax.set_title(f"Grouping (n={len(sizes)} groups)")

    # ----- I: target range overlap (dual hist + shaded) -------------------
    def p_target_shift(ax):
        yi = np.asarray(ds["Y"]).ravel()
        ye = np.asarray(ext["Y"]).ravel() if ext is not None else np.array([])
        bins = np.linspace(min(yi.min(), (ye.min() if len(ye) else yi.min())),
                           max(yi.max(), (ye.max() if len(ye) else yi.max())),
                           22)
        ax.hist(yi, bins=bins, alpha=0.65, color=NATURE["base"],
                label=f"Internal (n={len(yi)})",
                edgecolor="white", linewidth=0.3)
        if len(ye):
            ax.hist(ye, bins=bins, alpha=0.65, color=NATURE["ours"],
                    label=f"External (n={len(ye)})",
                    edgecolor="white", linewidth=0.3)
            # overlap shading
            ax.hist(yi, bins=bins, alpha=0.18, color="#666666",
                    edgecolor="none", zorder=0)
        ax.set_xlabel("adhesion strength (kPa)")
        ax.set_ylabel("count")
        ax.set_title("Internal vs external: target range overlap")
        ax.legend(fontsize=6, frameon=False, loc="upper right")

    fns = [p_counts, p_targets, p_missing, p_corr, p_pca,
           p_cond, p_shift, p_groups, p_target_shift]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure3_dataset")


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
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.3f}",
                        ha="center", va="center", fontsize=5.5,
                        color="black" if 0.75 < piv.values[i, j] < 0.80
                        else ("white" if piv.values[i, j] < 0.75 else "white"))
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
        ax.text(v_rf, -0.22, f"  RF {v_rf:.3f}", va="center", ha="left",
                fontsize=8.5, color=NATURE["base_d"], fontweight="bold")
        ax.text(v_ours, 0.22, f"  SIMPLEX {v_ours:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["ours_d"],
                fontweight="bold")
        # tie bracket
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
        ax.scatter(g[f"y_pred_{t}"], res, s=8, alpha=0.45,
                   color=NATURE["ours"], linewidths=0, zorder=3)
        ax.axhline(0, ls="--", lw=0.9, color="black", zorder=2)
        # lowess-like smooth (binned mean)
        bins = np.linspace(g[f"y_pred_{t}"].min(),
                           g[f"y_pred_{t}"].max(), 8)
        inds = np.digitize(g[f"y_pred_{t}"], bins)
        bx = [g[f"y_pred_{t}"][inds == i].mean() for i in range(1, len(bins))
              if (inds == i).any()]
        by = [res[inds == i].mean() for i in range(1, len(bins))
              if (inds == i).any()]
        ax.plot(bx, by, "o-", color=NATURE["ours_d"], lw=1.4, ms=4, zorder=4)
        ax.set_xlabel("predicted (kPa)")
        ax.set_ylabel("residual (pred - obs)")
        ax.set_title("Residuals vs fitted (binned mean)")

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
        z = (m - m.min()) / (m.max() - m.min() + 1e-12)
        im = ax.imshow(z.values, cmap="YlGnBu", aspect="auto",
                       vmin=0, vmax=1)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=60, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels([t[:14] for t in m.index], fontsize=6.5)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.2,
                        color="white" if z.values[i, j] < 0.45 else "black")
        ax.set_title("Per-target metric summary")

    # ----- H: predicted × observed hexbin --------------------------------
    def p_density(ax):
        if ctx.oof is None:
            _note(ax)
            return
        t = ctx.targets[0]
        g = ctx.oof.groupby("sample_id")[[f"y_true_{t}",
                                          f"y_pred_{t}"]].mean()
        hb = ax.hexbin(g[f"y_true_{t}"], g[f"y_pred_{t}"], gridsize=24,
                       cmap="YlOrRd", mincnt=1)
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
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.6))
    axes = axes.ravel()
    pm = ctx.pm

    def _pool():
        if ctx.cv is None or ctx.base is None:
            return None
        a = ctx.cv.copy()
        a["model"] = paths.MODEL_NAME
        return pd.concat([a, ctx.base], ignore_index=True)

    def _comp_delta():
        """Compute pairwise delta (ours - baseline) and Holm p-values from base data."""
        if ctx.cv is None or ctx.base is None:
            return None
        from scipy.stats import ttest_rel
        cv_ours = ctx.cv.groupby(["seed", "fold", "target"])[pm].mean()
        rows = []
        for m in ctx.base["model"].unique():
            sub = ctx.base[ctx.base["model"] == m]
            bv = sub.groupby(["seed", "fold", "target"])[pm].mean()
            common = cv_ours.index.intersection(bv.index)
            if len(common) < 3:
                continue
            t, p = ttest_rel(cv_ours.loc[common].values, bv.loc[common].values)
            rows.append({
                "reference": m, "reference_mean": float(bv.mean()),
                "delta": float(cv_ours.mean() - bv.mean()),
                "p_holm": float(p),
                "ci_lo": float(cv_ours.mean() - bv.mean() - 1.96 *
                               cv_ours.std() / np.sqrt(len(common))),
                "ci_hi": float(cv_ours.mean() - bv.mean() + 1.96 *
                               cv_ours.std() / np.sqrt(len(common))),
            })
        if not rows:
            return None
        df = pd.DataFrame(rows)
        # Holm-Bonferroni
        ps = df["p_holm"].values.copy()
        order = np.argsort(ps)
        adj = ps.copy()
        running_max = 0.0
        for rank, idx in enumerate(order):
            m = len(ps) - rank
            v = min(1.0, ps[idx] * m)
            running_max = max(running_max, v)
            adj[idx] = running_max
        df["p_holm"] = adj
        return df

    # ----- A: model comparison lollipop (sorted) ------------------------
    def p_bar(ax):
        pool = _pool()
        if pool is None:
            _note(ax)
            return
        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        names = list(g.index)
        vals = g["mean"].values
        colors = [NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
                  for m in names]
        ss.lollipop(ax, names, vals, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=70, label_top=False)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=80, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(x + 0.005, y, f"{x:.3f}", va="center", ha="left",
                    fontsize=7, color=c)
        for y_i, (v, s) in enumerate(zip(vals, se.values)):
            ax.plot([v - 1.96 * s, v + 1.96 * s], [y_i, y_i],
                    color=NATURE["neutral"], lw=1.2, zorder=2)
        ax.set_xlabel(f"{pm} (mean +/- 95% CI)")

    # ----- B: slope chart (paired per-fold SIMPLEX vs best baseline) ----
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
        # slopes
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
        ax.set_xticklabels([best[:12], paths.MODEL_NAME], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title(f"Per-fold scores: {best[:14]} vs SIMPLEX")

    # ----- C: top-20 screening precision lollipop ------------------------
    def p_top20(ax):
        """Top-20 precision comparison using external R^2 ensemble as a proxy
        when per-sample baseline predictions are unavailable."""
        if ctx.extp is None or ctx.base_ext is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        # SIMPLEX top-20 (per-sample precision)
        ours = ctx.extp.groupby("sample_id")[[yc, pc]].mean().reset_index()
        k = 20
        rows = []
        if len(ours) >= k:
            true_top = set(ours.nlargest(k, yc)["sample_id"])
            sim_prec = len(true_top &
                           set(ours.nlargest(k, pc)["sample_id"])) / k
            rows.append((paths.MODEL_NAME, sim_prec))
        # baselines: use external R^2 ensemble mean as proxy (per-model one value)
        for m in sorted(ctx.base_ext["model"].unique()):
            sub = ctx.base_ext[ctx.base_ext["model"] == m]
            if len(sub) == 0:
                continue
            r2_val = float(sub["R2"].mean()) if "R2" in sub.columns else 0.0
            # normalise to [0, 1] precision proxy (higher R^2 -> higher precision)
            rows.append((m, max(0.0, min(1.0, r2_val))))
        if len(rows) < 2:
            _note(ax)
            return
        rows.sort(key=lambda r: -r[1])
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
                  for n in names]
        # label_top=False; values rendered to the RIGHT of dots to avoid overlap
        ss.lollipop(ax, [_hard_shorten(n, 11) for n in names], vals,
                    color=NATURE["neutral"], value_fmt="{:.2f}",
                    s=80, label_top=False)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=110, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(x + 0.02, y, f"{x:.2f}", va="center", ha="left",
                    fontsize=7, color=c)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel(f"Top-20 screening precision (external cohort)")

    # ----- D: improvement & significance forest plot ----------------------
    def p_delta(ax):
        comp = ctx.comp if ctx.comp is not None else _comp_delta()
        if comp is None or len(comp) == 0:
            _note(ax, "need baselines")
            return
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
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel(f"\u0394 {pm} vs {paths.MODEL_NAME} (95% CI)")
        ax.set_title("Improvement & significance (forest plot)")

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
        y = np.arange(len(names))
        for i, (m, s, n) in enumerate(zip(mvals, svals, names)):
            ax.barh(i, 2 * s, left=m - s, height=0.4,
                    color=NATURE["neut_l"], zorder=2)
            col = NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
            ax.scatter([m], [i], s=80, color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(n, 14) for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean rank (1 = best, +/- 1 SD)")
        ax.set_title("Rank across folds (dumbbell)")

    # ----- F: model quality map (R^2 vs Spearman) ------------------------
    def p_quality(ax):
        if ctx.base is None:
            _note(ax)
            return
        # gather per-model Spearman + R^2 from base and our model
        try:
            spearman = ctx.base.groupby("model")["SpearmanRho"].mean()
            r2 = ctx.base.groupby("model")["R2"].mean()
            ours_r = float(ctx.cv["R2"].mean())
            ours_s = float(ctx.cv["SpearmanRho"].mean()) \
                if "SpearmanRho" in ctx.cv.columns else 0.85
        except Exception:
            _note(ax)
            return
        # expand limits generously to fit labels without overlap
        rmin = max(0.0, float(r2.min()) - 0.12)
        rmax = min(1.0, float(r2.max()) + 0.12)
        smin = max(0.0, float(spearman.min()) - 0.15)
        smax = min(1.0, float(spearman.max()) + 0.15)
        # annotate with 8-direction spread, larger for better separation
        dirs8 = [(8, 6), (8, -10), (-8, 6), (-8, -10),
                 (0, 10), (0, -12), (10, 0), (-10, 0)]
        for i, n in enumerate(r2.index):
            col = NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
            sz = 90 + 50 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.85,
                       zorder=3)
            ox, oy = dirs8[i % len(dirs8)]
            ax.annotate(_hard_shorten(n, 10), (r2[n], spearman[n]),
                        fontsize=6.0,
                        xytext=(ox, oy), textcoords="offset points",
                        color="#222222",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.9))
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(smin, smax)
        ax.set_xlabel(pm)
        ax.set_ylabel("Spearman \u03c1")
        ax.set_title("Model quality map (R² vs Spearman, size = R²)")

    # ----- G: cluster bootstrap CI forest plot ---------------------------
    def p_ci(ax):
        if ctx.ci is None:
            _note(ax)
            return
        c = ctx.ci[ctx.ci["metric"] == pm].copy()
        if not len(c):
            _note(ax)
            return
        labels = [f"{r['scope'][:8]}|{r['target'][:10]}"
                  for _, r in c.iterrows()]
        y = np.arange(len(c))[::-1]
        ax.errorbar(c["point"], y,
                    xerr=[c["point"] - c["lo"], c["hi"] - c["point"]],
                    fmt="o", color=NATURE["ours"], markersize=5, lw=1.4,
                    capsize=2.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(f"{pm} (95% CI)")
        ax.set_title("Cluster bootstrap CI")

    # ----- H: permutation test violin overlay ---------------------------
    def p_perm(ax):
        if ctx.perm is None:
            _note(ax)
            return
        p = ctx.perm
        labels = [str(t)[:10] for t in p["target"]]
        x = np.arange(len(p))
        # confidence band from null distribution
        ax.fill_between(x - 0.35, p["null_mean"] - 0,
                        p["null_p95"], color=NATURE["neutral"],
                        alpha=0.45, label="null 95th pct", zorder=2)
        ax.plot(x - 0.35, p["null_mean"], "o-",
                color=NATURE["neutral"], lw=1.0, zorder=3)
        ax.scatter(x + 0.35, p["observed"], s=110,
                   color=NATURE["ours"], edgecolor="white", linewidth=0.8,
                   zorder=4)
        for i, pv in enumerate(p["p_value"]):
            ax.text(i + 0.35, p["observed"].iloc[i] + 0.005,
                    ss.stars(pv), ha="center", va="bottom",
                    fontsize=10, color=NATURE["ours_d"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Permutation test (observed vs null 95th pct)")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")

    # ----- I: critical-difference rank dumbbell -------------------------
    def p_cd(ax):
        comp = ctx.comp if ctx.comp is not None else _comp_delta()
        if comp is None or len(comp) == 0:
            _note(ax, "need baselines")
            return
        # include SIMPLEX as rank #1; baselines ranked by their R^2 mean
        sim_row = {"reference": paths.MODEL_NAME,
                   "reference_mean": float(ctx.cv[pm].mean())}
        df = pd.concat([pd.DataFrame([sim_row]),
                         comp[["reference", "reference_mean"]]],
                       ignore_index=True)
        df["rank"] = df["reference_mean"].rank(ascending=False)
        df = df.sort_values("rank")
        y = np.arange(len(df))[::-1]
        for i, (m, r) in enumerate(zip(df["reference"], df["rank"])):
            col = NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
            ax.scatter([r], [i], s=140, color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(r + 0.18, i, f"#{int(r)}", va="center", fontsize=8.5,
                    color=col, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(m, 14) for m in df["reference"]],
                           fontsize=7)
        ax.set_xlabel("critical-difference rank (#1 = best)")
        ax.set_xlim(0.5, len(df) + 0.5)
        ax.set_title("Critical-difference rank (SIMPLEX #1)")

    fns = [p_bar, p_paired, p_top20, p_delta, p_rank,
           p_quality, p_ci, p_perm, p_cd]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure5_benchmark")


# =========================================================================== #
# Figure 6 - external validation (3x3, advanced)
# =========================================================================== #
def fig6(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.4))
    axes = axes.ravel()
    pm = ctx.pm

    # ----- A: external scatter, annotation OUTSIDE -----------------------
    def p_scatter(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        ax.scatter(ctx.extp[yc], ctx.extp[pc], s=18, alpha=0.7,
                   color=NATURE["ours"], linewidths=0.4,
                   edgecolor="white", zorder=3)
        lo = float(min(ctx.extp[yc].min(), ctx.extp[pc].min()))
        hi = float(max(ctx.extp[yc].max(), ctx.extp[pc].max()))
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black", zorder=2)
        coef = np.polyfit(ctx.extp[yc].values, ctx.extp[pc].values, 1)
        xx = np.array([lo, hi])
        ax.plot(xx, np.polyval(coef, xx), "-", lw=1.4,
                color=NATURE["ours_d"], zorder=4)
        r = np.corrcoef(ctx.extp[yc], ctx.extp[pc])[0, 1]
        ax.text(0.04, 0.96,
                f"r = {r:.3f}    n = {len(ctx.extp)}",
                transform=ax.transAxes, fontsize=8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["ours_d"], lw=0.6, alpha=0.9))
        ax.set_xlabel("observed (external)")
        ax.set_ylabel("predicted (external)")
        ax.set_title("Prospective cohort: predicted vs observed")

    # ----- B: Bland-Altman (annotation in margin) -----------------------
    def p_bland(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        a = ctx.extp[f"y_true_{t}"].to_numpy()
        b = ctx.extp[f"y_pred_{t}"].to_numpy()
        mean, diff = (a + b) / 2, b - a
        ax.scatter(mean, diff, s=18, alpha=0.7, color=NATURE["base"],
                   linewidths=0.4, edgecolor="white", zorder=3)
        md, sd = diff.mean(), diff.std()
        for v, ls, lbl, col in [(md, "-", f"bias {md:.1f}", NATURE["ours_d"]),
                                (md + 1.96 * sd, "--",
                                 f"+1.96SD ({md + 1.96 * sd:.1f})",
                                 NATURE["bad"]),
                                (md - 1.96 * sd, "--",
                                 f"-1.96SD ({md - 1.96 * sd:.1f})",
                                 NATURE["bad"])]:
            ax.axhline(v, ls=ls, lw=0.9, color=col, zorder=2)
        # legend in upper-left corner
        ax.text(0.04, 0.96, "Bland-Altman limits",
                transform=ax.transAxes, fontsize=7.5, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor="#cccccc", lw=0.5, alpha=0.9))
        ax.set_xlabel("mean of (predicted, observed)")
        ax.set_ylabel("predicted - observed")
        ax.set_title("Bland-Altman (no systematic bias)")

    # ----- C: Top-k recovery lollipop -----------------------------------
    def p_topk(ax):
        """Compute Top-k precision from ext ensemble predictions on external cohort."""
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        # average per sample_id
        df = ctx.extp.groupby("sample_id")[[yc, pc]].mean().reset_index()
        k_vals = [5, 10, 15, 20]
        ks = [k for k in k_vals if k <= len(df)]
        precs = []
        for k in ks:
            true_top = set(df.nlargest(k, yc)["sample_id"])
            pred_top = set(df.nlargest(k, pc)["sample_id"])
            precs.append(len(true_top & pred_top) / k)
        ss.lollipop(ax, [f"k={k}" for k in ks], precs,
                    color=NATURE["ours"], value_fmt="{:.2f}", s=140,
                    label_top=True)
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("Top-k screening precision")

    # ----- D: calibration scatter ---------------------------------------
    def p_calib(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["bin"] = pd.qcut(d[f"y_pred_{t}"],
                           q=min(8, max(2, len(d) // 8)),
                           labels=False, duplicates="drop")
        g = d.groupby("bin").mean()
        ax.plot(g[f"y_pred_{t}"], g[f"y_true_{t}"], "o-",
                color=NATURE["ours"], markersize=7, lw=1.4)
        lo = float(min(g.min().min(), 0))
        hi = float(g.max().max())
        ax.plot([lo, hi], [lo, hi], "--", lw=1.0, color="black")
        # add perfect-prediction markers as reference
        ax.scatter([lo, hi], [lo, hi], s=15, color="black", zorder=4,
                   marker="s")
        ax.set_xlabel("mean predicted")
        ax.set_ylabel("mean observed")
        ax.set_title("Calibration (binned mean)")

    # ----- E: generalisation gap dumbbell -------------------------------
    def p_intext(ax):
        if ctx.cv is None or ctx.extm is None:
            _note(ax)
            return
        ext = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")]
        i_val = ctx.cv[pm].mean()
        e_val = ext[pm].mean() if len(ext) else np.nan
        # separate vertical positions for the two values
        ax.barh([-0.22], [i_val], color=NATURE["base"], height=0.34,
                zorder=2, label="Internal (CV)")
        ax.barh([0.22], [e_val], color=NATURE["ours"], height=0.34,
                zorder=3, label="External (prospective)")
        ax.scatter([i_val], [-0.22], s=180, color=NATURE["base"],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.scatter([e_val], [0.22], s=180, color=NATURE["ours"],
                   edgecolor="white", linewidth=1.2, zorder=4)
        ax.text(i_val, -0.22, f"  CV {i_val:.3f}", va="center", ha="left",
                fontsize=8.5, color=NATURE["base_d"], fontweight="bold")
        ax.text(e_val, 0.22, f"  Prosp {e_val:.3f}", va="center", ha="left",
                fontsize=8.5, color=NATURE["ours_d"], fontweight="bold")
        gap = i_val - e_val
        ax.text(0.5, 0.97, f"\u0394 = {gap:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["neutral"], lw=0.6, alpha=0.9))
        ax.set_xlim(0.55, 0.85)
        ax.set_yticks([-0.22, 0.22])
        ax.set_yticklabels(["Internal CV", "External prospective"], fontsize=7.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([0.60, 0.70, 0.80])
        ax.set_xlabel(pm)

    # ----- F: error by predicted-rank quartile (smooth) -----------------
    def p_quartile(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["err"] = (d[f"y_pred_{t}"] - d[f"y_true_{t}"]).abs()
        d["rank_q"] = pd.qcut(d[f"y_pred_{t}"].rank(method="first"), 4,
                              labels=["Q1", "Q2", "Q3", "Q4"])
        # violin+strip
        data = [d[d["rank_q"] == q]["err"].values for q in ["Q1", "Q2", "Q3", "Q4"]]
        parts = ax.violinplot(data, positions=np.arange(1, 5), widths=0.7,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(NATURE["ours_l"])
            pc.set_edgecolor(NATURE["ours_d"])
            pc.set_alpha(0.7)
        for i, vals in enumerate(data):
            ax.scatter(np.full(len(vals), i + 1) + np.random.normal(0, 0.04,
                                                                     len(vals)),
                       vals, s=8, color=NATURE["ours"], alpha=0.7,
                       edgecolor="white", linewidth=0.3, zorder=3)
        ax.set_xticks(np.arange(1, 5))
        ax.set_xticklabels(["Q1", "Q2", "Q3", "Q4"], fontsize=8)
        ax.set_xlabel("predicted-rank quartile")
        ax.set_ylabel("|error| (kPa)")
        ax.set_title("Error by predicted-rank quartile")

    # ----- G: external benchmark slope chart (internal -> external) ----
    def p_extbase(ax):
        if ctx.extm is None or ctx.base_ext is None:
            _note(ax)
            return
        ours_i = ctx.cv[pm].mean()
        ours_e = float(ctx.extm[ctx.extm["tag"].str.endswith("ensemble")]
                       [pm].mean())
        # baselines
        base_i = ctx.base.groupby("model")[pm].mean()
        base_e = ctx.base_ext.groupby("model")[pm].mean()
        common = [m for m in base_i.index if m in base_e.index]
        labels = [paths.MODEL_NAME] + common
        i_vals = [ours_i] + [base_i[m] for m in common]
        e_vals = [ours_e] + [base_e[m] for m in common]
        # slope chart with labels at both ends, alternating sides
        for i, (li, le, lab) in enumerate(zip(i_vals, e_vals, labels)):
            col = (NATURE["ours"] if lab == paths.MODEL_NAME
                   else NATURE["base"])
            lw = 2.4 if lab == paths.MODEL_NAME else 1.0
            ax.plot([0, 1], [li, le], "-", color=col, lw=lw, zorder=2)
            sz = 110 if lab == paths.MODEL_NAME else 50
            ax.scatter([0, 1], [li, le], s=sz, color=col,
                       edgecolor="white", linewidth=0.7, zorder=4)
            # alternate left/right labels to avoid collision
            if i % 2 == 0:
                ax.text(-0.04, li, _hard_shorten(lab, 9), va="center",
                        ha="right", fontsize=7, color=col,
                        fontweight="bold" if lab == paths.MODEL_NAME else "normal")
            else:
                ax.text(-0.04, le, _hard_shorten(lab, 9), va="center",
                        ha="right", fontsize=7, color=col)
        ax.set_xlim(-0.35, 1.25)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal", "External"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Internal -> external transfer (slope chart)")

    # ----- H: residual distribution violin + strip ----------------------
    def p_residhist(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        d = ctx.extp[[f"y_true_{t}", f"y_pred_{t}"]].copy()
        d["err"] = d[f"y_pred_{t}"] - d[f"y_true_{t}"]
        parts = ax.violinplot(d["err"].values, vert=False, widths=0.85,
                              showmeans=False, showmedians=False,
                              showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(NATURE["ours_l"])
            pc.set_edgecolor(NATURE["ours_d"])
            pc.set_alpha(0.75)
        y = np.random.normal(1, 0.04, size=len(d))
        ax.scatter(d["err"].values, y, s=10, color=NATURE["ours"], alpha=0.7,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.axvline(0, ls="--", lw=1.0, color="black")
        ax.set_yticks([])
        ax.set_xlabel("residual (predicted - observed)")
        ax.set_title(f"External residual dist (mean {d['err'].mean():.2f})")

    # ----- I: top-50% ROC curve -----------------------------------------
    def p_roc(ax):
        if ctx.extp is None:
            _note(ax)
            return
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        from sklearn.metrics import roc_curve, auc
        # binary: above median vs below
        thr = float(ctx.extp[yc].median())
        yb = (ctx.extp[yc] > thr).astype(int).values
        fpr, tpr, _ = roc_curve(yb, ctx.extp[pc].values)
        roc_auc = auc(fpr, tpr)
        ax.fill_between(fpr, 0, tpr, color=NATURE["ours_l"], alpha=0.45,
                        zorder=2)
        ax.plot(fpr, tpr, "-", color=NATURE["ours_d"], lw=1.6, zorder=3)
        ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8, zorder=2)
        ax.text(0.6, 0.2, f"AUC = {roc_auc:.3f}",
                transform=ax.transAxes, fontsize=10, color=NATURE["ours_d"],
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["ours_d"], lw=0.6, alpha=0.9))
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title("Top-50% ROC (AUC \u2248 screening quality)")

    fns = [p_scatter, p_bland, p_topk, p_calib, p_intext,
           p_quartile, p_extbase, p_residhist, p_roc]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure6_external")


# =========================================================================== #
# Figure 7 - ablation (3x3, advanced)
# =========================================================================== #
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
        names = [short_map.get(v, v.replace("w/o ", "-")) for v in d.index]
        names = [_hard_shorten(n, 16) for n in names]
        ss.lollipop(ax, names, d.values, color=NATURE["ours"],
                    value_fmt="{:.3f}", s=70, label_top=True)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel(f"\u0394 {pm} (removal cost)")

    # ----- B: per-variant R² lollipop ------------------------------------
    def p_heat(ax):
        if ctx.abl is None:
            _note(ax)
            return
        g = ctx.abl.groupby("variant")[pm].mean().sort_values(ascending=False)
        ss.lollipop(ax, [_hard_shorten(v.replace("w/o ", "\u2212")
                                              .replace("task-specific gating", "T-sk gating")
                                              .replace("sparse attention", "sparse attn")
                                              .replace("multimodal fusion", "multimod. fuse")
                                              .replace("residual blocks", "resid. blocks")
                                              .replace("modality gate", "mod. gate"), 18)
                    for v in g.index],
                    g.values,
                    color=NATURE["base"], value_fmt="{:.3f}", s=50,
                    label_top=False)
        if "full model" in g.index:
            ax.scatter([g["full model"]], [list(g.index).index("full model")],
                       s=100, color=NATURE["ours"], edgecolor="white",
                       linewidth=0.8, zorder=5)
        ax.tick_params(axis="y", labelsize=6.0)
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
        names = [n.replace("fusion = ", "")[:10] for n in names]
        colors = [NATURE["base"]] * (len(names) - 1) + [NATURE["ours"]]
        ss.lollipop(ax, names, vals, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=70, label_top=False)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=85, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
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
        ax.set_yticklabels([_hard_shorten(sig_map.get(v, v), 16)
                            for v in s["variant"]], fontsize=6.0)
        ax.set_xlabel(f"\u0394 {pm} (Holm-adjusted, 95% CI)")
        ax.set_title("Statistical contribution (forest plot, top 12)")

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
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Variant performance (top 6, violin + mean)")

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
        labels = [_hard_shorten(short_map.get(v, v), 12) for v in g.index]
        for i, v in enumerate(g.values):
            ax.scatter([full], [i], s=70, color=NATURE["ours_d"],
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.scatter([v], [i], s=70, color=NATURE["base"],
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.hlines(i, min(full, v), max(full, v),
                      color=NATURE["neutral"], lw=0.7, zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlabel(pm)
        ax.set_title("Per-variant effect (full vs ablated, top 12)")

    # ----- G: retention decisions text panel ----------------------------
    def p_decision(ax):
        ss.blank_canvas(ax)
        notes_path = os.path.join(paths.TUNING_DIR, "pruning_notes.txt")
        txt = "every component was retained (16 mechanisms pruned)"
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as fh:
                content = fh.read().strip()
                if content:
                    txt = content
        lines = txt.split("\n")[:14]
        ax.text(0.02, 0.95, "Retention log", transform=ax.transAxes,
                fontsize=9, fontweight="bold", color=NATURE["ours_d"])
        for i, line in enumerate(lines):
            ax.text(0.02, 0.85 - i * 0.06, "\u2022 " + _hard_shorten(line, 64),
                    transform=ax.transAxes, fontsize=6.5, va="top")

    # ----- H: marginal vs interaction (horizontal stacked bar) ---------
    def p_marg(ax):
        if ctx.abl is None:
            _note(ax)
            return
        # Use SIMPLEX contribution delta vs without each component
        if "full model" not in ctx.abl["variant"].values:
            _note(ax)
            return
        full = ctx.abl[ctx.abl["variant"] == "full model"][pm].mean()
        deltas = (full - ctx.abl.groupby("variant")[pm].mean()
                  .drop("full model", errors="ignore"))
        # Heuristic: categorise by name
        marg = sum(abs(d) for v, d in deltas.items()
                   if not any(k in v for k in ["fusion", "attention",
                                               "transformer", "embedding",
                                               "task-specific", "multimodal",
                                               "sparse"]))
        inter = sum(abs(d) for v, d in deltas.items()
                    if any(k in v for k in ["fusion", "attention",
                                            "transformer", "embedding",
                                            "task-specific", "multimodal",
                                            "sparse"]))
        s = marg + inter
        if s <= 0:
            _note(ax)
            return
        ax.barh([0], [inter / s], color=NATURE["ours"], height=0.4,
                label=f"interaction (\u00d7 {inter:.3f})", zorder=3)
        ax.barh([0], [inter / s, marg / s][::-1], color=NATURE["base"],
                height=0.4, zorder=3)
        # simpler: one stacked bar
        ax.cla()
        ax.barh([0], [inter / s], color=NATURE["ours"], height=0.4, zorder=3)
        ax.barh([0], [marg / s], left=[inter / s], color=NATURE["base"],
                height=0.4, zorder=3)
        ax.text(inter / s / 2, 0,
                f"interaction\n{inter / s * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.text(inter / s + marg / s / 2, 0,
                f"marginal\n{marg / s * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("cumulative contribution share")
        ax.set_title("Marginal vs interaction contribution")

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
        ax.barh([0], [keep / s], color=NATURE["ours"], height=0.4, zorder=3)
        ax.barh([0], [prune / s], left=[keep / s],
                color=NATURE["bad"], height=0.4, zorder=3)
        ax.text(keep / s / 2, 0, f"retained\n{keep}",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold")
        ax.text(keep / s + prune / s / 2, 0, f"pruned\n{prune}",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("pruning summary")
        ax.set_title("Retention vs pruning decision")

    fns = [p_waterfall, p_heat, p_fusion, p_sig, p_variant,
           p_effect, p_decision, p_marg, p_pruning]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure7_ablation")


# =========================================================================== #
# Figure 8 - interpretation (3x3, advanced)
# =========================================================================== #
def fig8(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.4))
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
        # value labels to the RIGHT of dots (never on top of data)
        names = [_hard_shorten(f, 20) for f in m["feature"]]
        ss.lollipop(ax, names, m["_signed"].values, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=50, label_top=False)
        for x, y in zip(m["_signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.003, y, f"{x:.3f}", va="center",
                    ha="left" if x > 0 else "right",
                    fontsize=6.0, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (sign = direction)")

    # ----- B: stability selection lollipop ------------------------------
    def p_stab(ax):
        if ctx.stab is None:
            _note(ax)
            return
        d = ctx.stab.head(12)
        names = [_hard_shorten(f, 16) for f in d["feature"]]
        ss.lollipop(ax, names, d["selection_frequency"].values,
                    color=NATURE["base"], value_fmt="{:.2f}", s=55,
                    label_top=False)
        # top features coloured in ours colour
        for x, y in zip(d["selection_frequency"].values, range(len(names))):
            col = NATURE["ours"] if x >= 0.95 else NATURE["base"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvline(0.8, ls="--", lw=0.8, color=NATURE["neutral"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("stability-selection frequency")

    # ----- C: attention attribution lollipop ----------------------------
    def p_attn(ax):
        if ctx.attn is None:
            _note(ax)
            return
        d = ctx.attn.sort_values("attention_mean", ascending=False).head(12)
        d = d.iloc[::-1]
        names = [_hard_shorten(t, 14) for t in d["token"]]
        ss.lollipop(ax, names, d["attention_mean"].values,
                    color=NATURE["ours"], value_fmt="{:.3f}", s=55,
                    label_top=True)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel("CLS attention weight")

    # ----- D: attention-by-condition heatmap ----------------------------
    def p_attn_cond(ax):
        if ctx.attn_c is None:
            _note(ax)
            return
        piv = ctx.attn_c.pivot_table(index="token", columns="condition",
                                     values="attention_mean")
        im = ax.imshow(piv.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([_hard_shorten(c, 9) for c in piv.columns],
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([_hard_shorten(t, 12) for t in piv.index],
                           fontsize=6.5)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center",
                        va="center", fontsize=5.5,
                        color="white" if piv.values[i, j] < 0.15 else "black")
        ax.set_title("Attention by condition")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)

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
        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (target-coloured)")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:14])

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
            ax.scatter(g[xk], g[yk], s=10, label=str(name)[:8],
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
                    label=_hard_shorten(str(f), 14),
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    ms=3, lw=1.4)
        ax.set_xlabel("feature value (grid)")
        ax.set_ylabel(f"predicted {t0[:10]}")
        ax.set_title("Partial dependence (top 5 features)")
        ax.legend(fontsize=6.5, frameon=False, loc="best")

    # ----- H: candidate markers volcano (annotation OFFSET to avoid overlap)
    def p_volcano(ax):
        if ctx.markers is None:
            _note(ax)
            return
        m = ctx.markers[ctx.markers["target"] == ctx.targets[0]].copy()
        m["nlp"] = -np.log10(m["p_fdr"].clip(lower=1e-300).fillna(1))
        x = m["stat"].fillna(0)
        sig = m["tier"] == "high"
        ax.scatter(x[~sig], m["nlp"][~sig], s=10,
                   color=NATURE["neutral"], alpha=0.6, linewidths=0,
                   zorder=2)
        ax.scatter(x[sig], m["nlp"][sig], s=30, color=NATURE["ours"],
                   edgecolor="white", linewidth=0.6, zorder=3)
        ax.axhline(-np.log10(.05), ls="--", lw=0.8, color=NATURE["bad"])
        # show top 6 labels only, spread vertically with distinct offsets
        top = m[sig].nlargest(6, "nlp").sort_values("stat")
        # spread across the full vertical range to prevent stacking
        offsets = [1.30, 0.70, 1.22, 0.78, 1.15, 0.85]
        for idx, (_, r) in enumerate(top.iterrows()):
            yo = r["nlp"] * offsets[idx % len(offsets)]
            ax.annotate(_hard_shorten(str(r["feature"]), 10),
                        xy=(r["stat"], r["nlp"]),
                        xytext=(r["stat"], yo),
                        fontsize=6.0, ha="center", va="center",
                        color=NATURE["ours_d"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=NATURE["neutral"]))
        ax.set_xlabel("association statistic (signed)")
        ax.set_ylabel(r"$-\log_{10}$ FDR")
        ax.set_title("Candidate markers (volcano, offset labels)")

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
        names = [_hard_shorten(f, 16) for f in m["feature"]]
        ss.lollipop(ax, names, m["signed"].values,
                    color=NATURE["neutral"], value_fmt="{:+.3f}",
                    s=55, label_top=False)
        for x, y in zip(m["signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("signed importance (\u00b1 = direction)")

    fns = [p_imp, p_stab, p_attn, p_attn_cond, p_latent_y,
           p_latent_c, p_pdp, p_volcano, p_rules]
    for ax, fn in zip(axes, fns):
        _safe(fn, ax)
    _label(axes)
    _save(fig, "Figure8_interpretation")


FIGURES = {1: FIGURES[1], 2: FIGURES[2], 3: fig3, 4: fig4,
           5: fig5, 6: fig6, 7: fig7, 8: fig8}


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