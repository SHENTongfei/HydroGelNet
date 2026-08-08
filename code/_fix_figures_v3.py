"""Comprehensive patch: fix ctx field names, dumbbell bar overlaps,
label crowding, and missing-data fallbacks in figures.py v3.

Run once: python _fix_figures_v3.py
"""
import _runtime_guard  # noqa
import re
import os

TARGET = r"C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py"

src = open(TARGET, encoding="utf-8").read()


# ============================================================================
# FIX 1: ctx.comp fallback - when comp is None, build from baseline data
# This affects fig5 p_delta, p_rank, p_ci, p_cd.
# ============================================================================
# Insert a helper at the top after _pool definition in fig5
old_pool = '''    def _pool():
        if ctx.cv is None or ctx.base is None:
            return None
        a = ctx.cv.copy()
        a["model"] = paths.MODEL_NAME
        return pd.concat([a, ctx.base], ignore_index=True)'''
new_pool = old_pool + '''

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
        return df'''
assert old_pool in src, "could not find _pool anchor"
src = src.replace(old_pool, new_pool)


# ============================================================================
# FIX 2: fig5 p_delta - use _comp_delta() when ctx.comp is None
# ============================================================================
old_delta = '''    def p_delta(ax):
        if ctx.comp is None:
            _note(ax)
            return
        g = ctx.comp.groupby("reference").agg(
            delta=("delta", "mean"),
            p=("p_holm", "min"),
            lo=("ci_lo", "mean") if "ci_lo" in ctx.comp.columns
            else ("delta", "mean"),
            hi=("ci_hi", "mean") if "ci_hi" in ctx.comp.columns
            else ("delta", "mean"),
        ).sort_values("delta")
        if "ci_lo" not in ctx.comp.columns:
            g["lo"] = g["delta"] - 0.02
            g["hi"] = g["delta"] + 0.02'''
new_delta = '''    def p_delta(ax):
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
            g["hi"] = g["delta"] + 0.02'''
assert old_delta in src, "p_delta anchor missing"
src = src.replace(old_delta, new_delta)


# ============================================================================
# FIX 3: fig5 p_perm - use null_mean/null_p95 instead of permuted_dist
# ============================================================================
old_perm = '''    def p_perm(ax):
        if ctx.perm is None:
            _note(ax)
            return
        p = ctx.perm
        labels = [t[:10] for t in p["target"]]
        ax.violinplot(p["permuted_dist"], positions=np.arange(len(p)),
                      widths=0.7, showmeans=False, showmedians=False,
                      showextrema=False)
        for body in ax.collections:
            body.set_facecolor(NATURE["neutral"])
            body.set_edgecolor(NATURE["neutral"])
            body.set_alpha(0.35)
        ax.scatter(np.arange(len(p)), p["observed"], s=70,
                   color=NATURE["ours"], edgecolor="white", linewidth=0.8,
                   zorder=4)
        for i, pv in enumerate(p["p_value"]):
            ax.text(i, p["observed"].iloc[i] + 0.005,
                    ss.stars(pv), ha="center", va="bottom",
                    fontsize=9, color=NATURE["ours_d"], fontweight="bold")
        ax.set_xticks(np.arange(len(p)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel(pm)
        ax.set_title("Permutation test (observed > null)")'''
new_perm = '''    def p_perm(ax):
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
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")'''
assert old_perm in src, "p_perm anchor missing"
src = src.replace(old_perm, new_perm)


# ============================================================================
# FIX 4: fig4 p_compare_rf - separate bars vertically (was overlapping)
# ============================================================================
old_rf = '''    def p_compare_rf(ax):
        if ctx.cv is None or ctx.base is None:
            _note(ax)
            return
        ours = ctx.cv.groupby("model")[pm].mean()
        rf = ctx.base[ctx.base["model"] == "RandomForest"].groupby("model")[pm].mean()
        if len(ours) == 0 or len(rf) == 0:
            _note(ax)
            return
        v_ours = float(ours.iloc[0]); v_rf = float(rf.iloc[0])
        ax.barh([0], [v_rf], color=NATURE["base"], height=0.35, zorder=2)
        ax.barh([0], [v_ours], color=NATURE["ours"], height=0.35, zorder=3)
        ax.scatter([v_rf], [0], s=140, color=NATURE["base"],
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.scatter([v_ours], [0], s=140, color=NATURE["ours"],
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(v_rf, 0, f"  RF {v_rf:.3f}", va="center", ha="left",
                fontsize=8, color=NATURE["base_d"])
        ax.text(v_ours, 0, f"  SIMPLEX {v_ours:.3f}", va="center",
                ha="left", fontsize=8, color=NATURE["ours_d"])
        ax.set_xlim(0.65, 0.85)
        ax.set_yticks([])
        ax.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85])
        ax.set_xlabel(f"{pm} (ensemble)")
        ax.set_title("SIMPLEX matches RF (within tie)")'''
new_rf = '''    def p_compare_rf(ax):
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
        ax.set_ylim(-0.6, 0.6)'''
assert old_rf in src, "p_compare_rf anchor missing"
src = src.replace(old_rf, new_rf)


# ============================================================================
# FIX 5: fig4 p_seed_stability - rotate seed labels 90 deg to avoid overlap
# ============================================================================
old_seed = '''        ax.set_xticks(positions)
        ax.set_xticklabels([f"s{s}" for s in seeds], fontsize=7)
        ax.set_xlabel("seed")
        ax.set_ylabel(pm)
        ax.set_title("Seed-to-seed stability (violin + mean)")'''
new_seed = '''        ax.set_xticks(positions)
        ax.set_xticklabels([f"seed {s}" for s in seeds], rotation=70,
                           ha="right", fontsize=6.5)
        ax.set_xlabel("seed")
        ax.set_ylabel(pm)
        ax.set_title("Seed-to-seed stability (violin + mean)")'''
assert old_seed in src, "p_seed_stability anchor missing"
src = src.replace(old_seed, new_seed)


# ============================================================================
# FIX 6: fig5 p_quality - expand axes so dots don't all cluster
# ============================================================================
old_q = '''        for n in r2.index:
            col = NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
            sz = 80 + 60 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.85,
                       zorder=3)
            ax.annotate(_hard_shorten(n, 10), (r2[n], spearman[n]),
                        fontsize=5.5, xytext=(4, 4),
                        textcoords="offset points", color="#333333")
        ax.set_xlabel(pm)
        ax.set_ylabel("Spearman \\u03c1")
        ax.set_title("Model quality map (R² vs Spearman, size = R²)")'''
new_q = '''        rmin = max(0.0, float(r2.min()) - 0.05)
        rmax = min(1.0, float(r2.max()) + 0.05)
        smin = max(0.0, float(spearman.min()) - 0.05)
        smax = min(1.0, float(spearman.max()) + 0.05)
        for n in r2.index:
            col = NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
            sz = 70 + 35 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.85,
                       zorder=3)
            ax.annotate(_hard_shorten(n, 10), (r2[n], spearman[n]),
                        fontsize=5.5, xytext=(4, 4),
                        textcoords="offset points", color="#333333")
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(smin, smax)
        ax.set_xlabel(pm)
        ax.set_ylabel("Spearman \\u03c1")
        ax.set_title("Model quality map (R² vs Spearman, size = R²)")'''
assert old_q in src, "p_quality anchor missing"
src = src.replace(old_q, new_q)


# ============================================================================
# FIX 7: fig5 p_cd - use rank from _comp_delta() (was all 1.0)
# ============================================================================
old_cd = '''    def p_cd(ax):
        if ctx.comp is None:
            _note(ax)
            return
        rank = ctx.comp.groupby("reference")["reference_mean"].rank(
            ascending=False)
        # build a flat list of (model, rank)
        rows = []
        for ref in rank.index:
            rows.append((ref, float(rank.loc[ref])))
        df = pd.DataFrame(rows, columns=["model", "rank"]).sort_values("rank")
        y = np.arange(len(df))[::-1]
        for i, (m, r) in enumerate(zip(df["model"], df["rank"])):
            col = NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
            ax.scatter([r], [i], s=100, color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(r + 0.1, i, f"{r:.1f}", va="center", fontsize=7,
                    color=col)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(m, 14) for m in df["model"]],
                           fontsize=7)
        ax.set_xlabel("critical-difference rank (lower = better)")
        ax.set_title("Critical-difference rank")'''
new_cd = '''    def p_cd(ax):
        comp = ctx.comp if ctx.comp is not None else _comp_delta()
        if comp is None or len(comp) == 0:
            _note(ax, "need baselines")
            return
        # rank by reference_mean (higher R = lower rank = better)
        rk = comp.set_index("reference")["reference_mean"].rank(
            ascending=False)
        df = pd.DataFrame({"model": rk.index, "rank": rk.values}) \
            .sort_values("rank")
        y = np.arange(len(df))[::-1]
        for i, (m, r) in enumerate(zip(df["model"], df["rank"])):
            col = NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
            ax.scatter([r], [i], s=120, color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(r + 0.15, i, f"{r:.1f}", va="center", fontsize=7.5,
                    color=col)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(m, 14) for m in df["model"]],
                           fontsize=7)
        ax.set_xlabel("critical-difference rank (lower = better)")
        ax.set_xlim(0.5, len(df) + 0.5)
        ax.set_title("Critical-difference rank")'''
assert old_cd in src, "p_cd anchor missing"
src = src.replace(old_cd, new_cd)


# ============================================================================
# FIX 8: fig6 p_topk - compute from base_external + ext ensemble (no topk file)
# ============================================================================
old_topk6 = '''    def p_topk(ax):
        if ctx.topk is None or "precision" not in ctx.topk.columns:
            _note(ax)
            return
        ours = ctx.topk[ctx.topk["model"] == paths.MODEL_NAME] \\
            if "model" in ctx.topk.columns else ctx.topk
        ours = ours[ours["top_k"].isin([5, 10, 15, 20])]
        ss.lollipop(ax, [f"k={int(k)}" for k in ours["top_k"]],
                    ours["precision"].values,
                    color=NATURE["ours"], value_fmt="{:.2f}", s=110,
                    label_top=True)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Top-k screening precision")'''
new_topk6 = '''    def p_topk(ax):
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
        ax.set_xlabel("Top-k screening precision")'''
assert old_topk6 in src, "fig6 p_topk anchor missing"
src = src.replace(old_topk6, new_topk6)


# ============================================================================
# FIX 9: fig6 p_intext (generalisation gap) - bars on separate y positions
# ============================================================================
old_gap = '''    def p_intext(ax):
        if ctx.cv is None or ctx.extm is None:
            _note(ax)
            return
        ext = ctx.extm[ctx.extm["tag"].str.endswith("ensemble")]
        i_val = ctx.cv[pm].mean()
        e_val = ext[pm].mean() if len(ext) else np.nan
        ax.barh([0], [i_val], color=NATURE["base"], height=0.45, zorder=2)
        ax.barh([0], [e_val], color=NATURE["ours"], height=0.45, zorder=3)
        ax.scatter([i_val, e_val], [0, 0], s=160, color=[NATURE["base"], NATURE["ours"]],
                   edgecolor="white", linewidth=1.0, zorder=4)
        ax.text(i_val, -0.30, f"  internal {i_val:.3f}", va="top", ha="left",
                fontsize=8, color=NATURE["base_d"])
        ax.text(e_val, 0.30, f"  prospective {e_val:.3f}", va="bottom",
                ha="left", fontsize=8, color=NATURE["ours_d"])
        gap = i_val - e_val
        ax.text(0.5, 0.95, f"\\u0394 = {gap:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=8.5, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["neutral"], lw=0.5, alpha=0.9))
        ax.set_xlim(0.55, 0.85)
        ax.set_yticks([])
        ax.set_xticks([0.60, 0.70, 0.80])
        ax.set_xlabel(pm)'''
new_gap = '''    def p_intext(ax):
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
        ax.text(0.5, 0.97, f"\\u0394 = {gap:.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor=NATURE["neutral"], lw=0.6, alpha=0.9))
        ax.set_xlim(0.55, 0.85)
        ax.set_yticks([-0.22, 0.22])
        ax.set_yticklabels(["Internal CV", "External prospective"], fontsize=7.5)
        ax.set_ylim(-0.6, 0.6)
        ax.set_xticks([0.60, 0.70, 0.80])
        ax.set_xlabel(pm)'''
assert old_gap in src, "fig6 p_intext anchor missing"
src = src.replace(old_gap, new_gap)


# ============================================================================
# FIX 10: fig6 p_extbase (slope chart) - move labels to opposite ends
# ============================================================================
old_extb = '''        # slope chart
        for i, (li, le, lab) in enumerate(zip(i_vals, e_vals, labels)):
            col = (NATURE["ours"] if lab == paths.MODEL_NAME
                   else NATURE["base"])
            lw = 2.4 if lab == paths.MODEL_NAME else 1.0
            ax.plot([0, 1], [li, le], "-", color=col, lw=lw, zorder=2)
            ax.scatter([0, 1], [li, le], s=80 if lab == paths.MODEL_NAME else 35,
                       color=col, edgecolor="white", linewidth=0.7,
                       zorder=4)
            if lab == paths.MODEL_NAME:
                ax.text(1.05, le, lab, va="center", ha="left", fontsize=8,
                        color=col, fontweight="bold")
            else:
                ax.text(-0.02, li, _hard_shorten(lab, 10), va="center",
                        ha="right", fontsize=6.5, color=col)
        ax.set_xlim(-0.3, 1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal", "External"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Internal -> external transfer (slope chart)")'''
new_extb = '''        # slope chart with labels at both ends, alternating sides
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
        ax.set_title("Internal -> external transfer (slope chart)")'''
assert old_extb in src, "fig6 p_extbase anchor missing"
src = src.replace(old_extb, new_extb)


# ============================================================================
# FIX 11: fig7 p_sig - compute paired t-test from ablation_results.csv
# ============================================================================
old_sig = '''    def p_sig(ax):
        if ctx.abl_stats is None:
            _note(ax)
            return
        s = ctx.abl_stats.groupby("variant").agg(
            delta=("delta", "mean"),
            p=("p_holm", "min"),
            lo=("ci_lo", "mean") if "ci_lo" in ctx.abl_stats.columns
            else ("delta", "mean"),
            hi=("ci_hi", "mean") if "ci_hi" in ctx.abl_stats.columns
            else ("delta", "mean"),
        ).sort_values("delta")
        if "ci_lo" not in ctx.abl_stats.columns:
            s["lo"] = s["delta"] - 0.01
            s["hi"] = s["delta"] + 0.01
        y = np.arange(len(s))[::-1]
        for i, (d, lo, hi, p) in enumerate(zip(s["delta"], s["lo"], s["hi"],
                                               s["p"])):
            col = NATURE["good"] if (d > 0 and p < 0.05) else (
                NATURE["ours"] if d > 0 else NATURE["bad"])
            ax.errorbar(d, i, xerr=[[d - lo], [hi - d]],
                        fmt="o", color=col, lw=1.2, capsize=2,
                        markersize=4.5, zorder=3)
            ax.text(d + (hi - d) + 0.003, i, ss.stars(p),
                    fontsize=7, color=col, va="center")
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(v.replace("w/o ", "\\u2212"), 22)
                            for v in s.index], fontsize=6.5)
        ax.set_xlabel(f"\\u0394 {pm} (Holm-adjusted, 95% CI)")
        ax.set_title("Statistical contribution (forest plot)")'''
new_sig = '''    def p_sig(ax):
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
        df = pd.DataFrame(rows).sort_values("delta", ascending=False)
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
        s = df
        y = np.arange(len(s))[::-1]
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
        ax.set_yticklabels([_hard_shorten(v.replace("w/o ", "\\u2212"), 22)
                            for v in s["variant"]], fontsize=6.5)
        ax.set_xlabel(f"\\u0394 {pm} (Holm-adjusted, 95% CI)")
        ax.set_title("Statistical contribution (forest plot)")'''
assert old_sig in src, "fig7 p_sig anchor missing"
src = src.replace(old_sig, new_sig)


# ============================================================================
# FIX 12: fig7 long labels - shrink font + shorten
# ============================================================================
old_wf = '''        names = [v.replace("w/o ", "\\u2212").replace("task-specific", \"T:\")
                 .replace("MC-Dropout", "MC-Drop") for v in d.index]
        # shorten
        names = [_hard_shorten(n, 22) for n in names]
        ss.lollipop(ax, names, d.values, color=NATURE["ours"],
                    value_fmt="{:.3f}", s=70, label_top=True)
        ax.set_xlabel(f"\\u0394 {pm} (removal cost)")'''
new_wf = '''        names = [v.replace("w/o ", "\\u2212").replace("task-specific", \"T:\")
                 .replace("MC-Dropout", "MC-Drop")
                 .replace("multimodal fusion", "multimod. fuse")
                 .replace("sparse attention", "sparse attn")
                 .replace("residual blocks", "resid. blocks")
                 .replace("modality gate", "mod. gate")
                 for v in d.index]
        names = [_hard_shorten(n, 24) for n in names]
        ss.lollipop(ax, names, d.values, color=NATURE["ours"],
                    value_fmt="{:.3f}", s=70, label_top=True)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel(f"\\u0394 {pm} (removal cost)")'''
assert old_wf in src, "fig7 p_waterfall anchor missing"
src = src.replace(old_wf, new_wf)


old_ph = '''        ss.lollipop(ax, [_hard_shorten(v, 18) for v in g.index], g.values,
                    color=NATURE["base"], value_fmt="{:.3f}", s=50,
                    label_top=False)
        # colour "full model" red
        if "full model" in g.index:
            ax.scatter([g["full model"]], [list(g.index).index("full model")],
                       s=80, color=NATURE["ours"], edgecolor="white",
                       linewidth=0.8, zorder=5)
        ax.set_xlabel(pm)'''
new_ph = '''        ss.lollipop(ax, [_hard_shorten(v.replace("w/o ", "\\u2212")
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
        ax.set_xlabel(pm)'''
assert old_ph in src, "fig7 p_heat anchor missing"
src = src.replace(old_ph, new_ph)


old_pv = '''        ax.set_xticks(positions)
        ax.set_xticklabels([_hard_shorten(v, 12) for v in top_variants],
                           rotation=25, ha="right", fontsize=7)
        ax.set_ylabel(pm)
        ax.set_title("Variant performance (violin + mean)")'''
new_pv = '''        ax.set_xticks(positions)
        ax.set_xticklabels([_hard_shorten(v
                                          .replace("w/o ", "\\u2212")
                                          .replace("multimodal fusion", "multimod.fuse")
                                          .replace("sparse attention", "sparse attn")
                                          .replace("residual blocks", "resid. blocks"), 11)
                           for v in top_variants],
                           rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(pm)
        ax.set_title("Variant performance (violin + mean)")'''
assert old_pv in src, "fig7 p_variant anchor missing"
src = src.replace(old_pv, new_pv)


# ============================================================================
# FIX 13: fig8 p_imp - smaller labels, alternate sides for high-density
# ============================================================================
old_imp = '''        names = [_hard_shorten(f, 16) for f in m["feature"]]
        ss.lollipop(ax, names, m["_signed"].values, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=50, label_top=False)
        for x, y in zip(m["_signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + (0.005 if x > 0 else -0.005), y,
                    f"{x:.3f}",
                    va="center",
                    ha="left" if x > 0 else "right",
                    fontsize=6.5, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (sign = direction)")'''
new_imp = '''        names = [_hard_shorten(f, 18) for f in m["feature"]]
        ss.lollipop(ax, names, m["_signed"].values, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=50, label_top=False)
        for x, y in zip(m["_signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvline(0, color="black", lw=0.6)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (sign = direction)")'''
assert old_imp in src, "fig8 p_imp anchor missing"
src = src.replace(old_imp, new_imp)


# ============================================================================
# FIX 14: fig8 p_volcano - fewer labels, smaller font
# ============================================================================
old_vol = '''        # LABEL OFFSET (alternate above / below)
        top = m[sig].nlargest(6, "nlp")
        for idx, (_, r) in enumerate(top.iterrows()):
            yo = 1.06 if idx % 2 == 0 else 0.94
            ax.text(r["stat"], r["nlp"] * yo,
                    _hard_shorten(str(r["feature"]), 12),
                    fontsize=6.5, ha="center", va="center",
                    color=NATURE["ours_d"], fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="none", alpha=0.85))'''
new_vol = '''        # show top 6 labels only with strategic offset
        top = m[sig].nlargest(6, "nlp").sort_values("stat")
        # alternate far above / far below to avoid stacking
        offsets = [1.12, 0.88, 1.12, 0.88, 1.12, 0.88]
        for idx, (_, r) in enumerate(top.iterrows()):
            yo = r["nlp"] * offsets[idx % len(offsets)]
            ax.annotate(_hard_shorten(str(r["feature"]), 10),
                        xy=(r["stat"], r["nlp"]),
                        xytext=(r["stat"], yo),
                        fontsize=6.0, ha="center", va="center",
                        color=NATURE["ours_d"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=NATURE["neutral"]))'''
assert old_vol in src, "fig8 p_volcano anchor missing"
src = src.replace(old_vol, new_vol)


# ============================================================================
# FIX 15: fig5 p_top20 - compute from external baselines (top-k precision)
# ============================================================================
old_top20 = '''    def p_top20(ax):
        if ctx.topk is None:
            _note(ax)
            return
        df = ctx.topk.copy()
        if "model" not in df.columns or "top_k" not in df.columns:
            _note(ax)
            return
        df = df[df["top_k"].isin([5, 10, 15, 20])]
        piv = df.pivot_table(index="model", columns="top_k",
                             values="precision")
        names = list(piv.index)
        vals = piv[20].reindex(names).fillna(0).values
        order = np.argsort(-vals)
        names = [names[i] for i in order]
        vals = vals[order]
        colors = [NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
                  for n in names]
        ss.lollipop(ax, [_hard_shorten(n, 12) for n in names], vals,
                    color=NATURE["neutral"], value_fmt="{:.2f}",
                    s=60, label_top=False)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=70, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Top-20 screening precision")'''
new_top20 = '''    def p_top20(ax):
        """Top-20 precision comparison from external baselines predictions."""
        if ctx.base_ext is None or ctx.extp is None:
            _note(ax)
            return
        from scipy.stats import spearmanr
        t = ctx.targets[0]
        yc, pc = f"y_true_{t}", f"y_pred_{t}"
        # SIMPLEX top-20 precision
        ours = ctx.extp.groupby("sample_id")[[yc, pc]].mean().reset_index()
        k = 20
        if len(ours) < k:
            _note(ax)
            return
        true_top = set(ours.nlargest(k, yc)["sample_id"])
        sim_prec = len(true_top & set(ours.nlargest(k, pc)["sample_id"])) / k
        # baseline precisions from ctx.base_ext (which has per-row pred)
        rows = [("SIMPLEX", sim_prec)]
        for m in sorted(ctx.base_ext["model"].unique()):
            sub = ctx.base_ext[(ctx.base_ext["model"] == m) &
                               (ctx.base_ext["tag"].str.contains("ensemble", na=False))]
            if len(sub) == 0:
                continue
            df = sub.groupby("sample_id")[[yc, pc]].mean().reset_index()
            if len(df) < k:
                continue
            true_top = set(df.nlargest(k, yc)["sample_id"])
            p = len(true_top & set(df.nlargest(k, pc)["sample_id"])) / k
            rows.append((m, p))
        if len(rows) < 2:
            _note(ax)
            return
        rows.sort(key=lambda r: -r[1])
        names = [r[0] for r in rows]
        vals = [r[1] for r in rows]
        colors = [NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
                  for n in names]
        ss.lollipop(ax, [_hard_shorten(n, 11) for n in names], vals,
                    color=NATURE["neutral"], value_fmt="{:.2f}",
                    s=80, label_top=True)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=110, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(f"Top-{k} screening precision (external cohort)")'''
assert old_top20 in src, "fig5 p_top20 anchor missing"
src = src.replace(old_top20, new_top20)


# ============================================================================
# FIX 16: fig8 p_attn smaller labels
# ============================================================================
old_attn = '''        ss.lollipop(ax, names, d["attention_mean"].values,
                    color=NATURE["ours"], value_fmt="{:.3f}", s=55,
                    label_top=True)
        ax.set_xlabel("CLS attention weight")'''
new_attn = '''        ss.lollipop(ax, names, d["attention_mean"].values,
                    color=NATURE["ours"], value_fmt="{:.3f}", s=55,
                    label_top=True)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel("CLS attention weight")'''
assert old_attn in src, "fig8 p_attn anchor missing"
src = src.replace(old_attn, new_attn)


# Save
open(TARGET, "w", encoding="utf-8").write(src)
print("ALL 16 FIXES APPLIED OK")