# -*- coding: utf-8 -*-
"""Figure 5 patches: model-name abbreviations everywhere, label overlaps,
D scale, F crowding, G plain lines, H null bug, I alignment, title length."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# ---------------- 5A: lollipop -> horizontal bars w/ labels right ---------
old_a = '''        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
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
        ax.set_xlabel(f"{pm} (mean +/- 95% CI)")'''
new_a = '''        g = pool.groupby("model")[pm].agg(["mean", "std", "count"])
        g = g.sort_values("mean")
        se = g["std"] / np.sqrt(g["count"])
        names = [_m_short(m, 9) for m in g.index]
        vals = g["mean"].values
        colors = [NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
                  for m in g.index]
        # horizontal bars with error bars; values at bar ends (no overlap)
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, s, c in zip(y, vals, se.values, colors):
            ax.errorbar(v, yy, xerr=1.96 * s, fmt="none", ecolor="#444444",
                        lw=1.0, capsize=2, zorder=2)
            ax.text(v + 0.008, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=7.5, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0.55, vals.max() + 0.07)
        ax.set_xlabel(f"{pm} (mean \u00b1 95% CI)")'''
assert old_a in src, "5A not found"
src = src.replace(old_a, new_a, 1)

# ---------------- 5B: abbreviate baseline name ----------------------------
old_b = '''        ax.set_xticklabels([best[:12], paths.MODEL_NAME], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title(f"Per-fold scores: {best[:14]} vs SIMPLEX")'''
new_b = '''        ax.set_xticklabels([_m_short(best, 9), "SIMPLEX"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Per-fold scores vs SIMPLEX")'''
assert old_b in src, "5B not found"
src = src.replace(old_b, new_b, 1)

# ---------------- 5C: top-20 - shorter xlabel, labels right ---------------
old_c = '''        rows.sort(key=lambda r: -r[1])
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
        ax.set_xlabel(f"Top-20 screening precision (external cohort)")'''
new_c = '''        rows.sort(key=lambda r: -r[1])
        names = [_m_short(r[0], 9) for r in rows]
        vals = [r[1] for r in rows]
        colors = [NATURE["ours"] if r[0] == paths.MODEL_NAME else NATURE["base"]
                  for r in rows]
        # horizontal bars; values to the RIGHT of bar end (never on dots)
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white",
                linewidth=0.7, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.015, yy, f"{v:.2f}", va="center", ha="left",
                    fontsize=7.5, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0, 1.18)
        ax.set_xlabel("Top-20 precision")'''
assert old_c in src, "5C not found"
src = src.replace(old_c, new_c, 1)

# ---------------- 5D: widen xlim so green dot never clips ---------------
old_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(g.index, fontsize=6.5)
        ax.set_xlabel(f"\\u0394 {pm} vs {paths.MODEL_NAME} (95% CI)")
        ax.set_title("Improvement & significance (forest plot)")'''
new_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(m, 10) for m in g.index], fontsize=6.5)
        dmin = float(g["lo"].min())
        dmax = float(g["hi"].max())
        ax.set_xlim(dmin - 0.02, dmax + 0.02)
        ax.set_xlabel(f"\\u0394 {pm} vs SIMPLEX (95% CI)")
        ax.set_title("Improvement & significance")'''
assert old_d in src, "5D not found"
src = src.replace(old_d, new_d, 1)

# ---------------- 5E: abbreviate model names -------------------------------
old_e = '''        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(n, 14) for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean rank (1 = best, +/- 1 SD)")
        ax.set_title("Rank across folds (dumbbell)")'''
new_e = '''        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(n, 9) for n in names], fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("mean rank (1 = best, \u00b1 1 SD)")
        ax.set_title("Rank across folds")'''
assert old_e in src, "5E not found"
src = src.replace(old_e, new_e, 1)

# ---------------- 5F: quality map - larger margins, shorter labels ------
old_f = '''        # expand limits generously to fit labels without overlap
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
        ax.set_ylabel("Spearman \\u03c1")
        ax.set_title("Model quality map (R² vs Spearman, size = R²)")'''
new_f = '''        # expand limits generously to fit labels without overlap
        rmin = max(0.0, float(r2.min()) - 0.15)
        rmax = min(1.0, float(r2.max()) + 0.15)
        smin = max(0.0, float(spearman.min()) - 0.18)
        smax = min(1.0, float(spearman.max()) + 0.18)
        # annotate with 8-direction spread, larger for better separation
        dirs8 = [(8, 6), (8, -12), (-8, 6), (-8, -12),
                 (0, 12), (0, -14), (12, 0), (-12, 0)]
        for i, n in enumerate(r2.index):
            col = NATURE["ours"] if n == paths.MODEL_NAME else NATURE["base"]
            sz = 100 + 55 * (r2[n] - r2.min())
            ax.scatter(r2[n], spearman[n], s=sz, color=col,
                       edgecolor="white", linewidth=0.8, alpha=0.85,
                       zorder=3)
            ox, oy = dirs8[i % len(dirs8)]
            ax.annotate(_m_short(n, 9), (r2[n], spearman[n]),
                        fontsize=6.5,
                        xytext=(ox, oy), textcoords="offset points",
                        color="#222222", fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.95))
        ax.set_xlim(rmin, rmax)
        ax.set_ylim(smin, smax)
        ax.set_xlabel(pm)
        ax.set_ylabel("Spearman \\u03c1")
        ax.set_title("Model quality map (R\u00b2 vs \u03c1)")'''
assert old_f in src, "5F not found"
src = src.replace(old_f, new_f, 1)

# ---------------- 5G: CI -> dumbbell with visible points -----------------
old_g = '''        labels = [f"{r['scope'][:8]}|{r['target'][:10]}"
                  for _, r in c.iterrows()]
        y = np.arange(len(c))[::-1]
        ax.errorbar(c["point"], y,
                    xerr=[c["point"] - c["lo"], c["hi"] - c["point"]],
                    fmt="o", color=NATURE["ours"], markersize=5, lw=1.4,
                    capsize=2.4, zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(f"{pm} (95% CI)")
        ax.set_title("Cluster bootstrap CI")'''
new_g = '''        labels = [f"{r['scope'][:6]}|{r['target'][:8]}"
                  for _, r in c.iterrows()]
        y = np.arange(len(c))[::-1]
        # dumbbell: line + end caps as filled dots (more visible than CI bars)
        for yy, (_, r) in zip(y, c.iterrows()):
            ax.plot([r["lo"], r["hi"]], [yy, yy], "-", color=NATURE["ours"],
                    lw=2.2, zorder=2)
            ax.scatter([r["lo"], r["hi"]], [yy, yy], s=30,
                       color="white", edgecolor=NATURE["ours_d"],
                       linewidth=0.9, zorder=4)
            ax.scatter([r["point"]], [yy], s=46, marker="D",
                       color=NATURE["ours_d"], edgecolor="white",
                       linewidth=0.6, zorder=5)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xlabel(f"{pm} (95% CI)")
        ax.set_title("Cluster bootstrap CI")'''
assert old_g in src, "5G not found"
src = src.replace(old_g, new_g, 1)

# ---------------- 5H: permutation - fix null text + clearer --------------
old_h = '''        p = ctx.perm
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
new_h = '''        p = ctx.perm
        labels = [str(t)[:8] for t in p["target"]]
        x = np.arange(len(p))
        # null band + observed markers; guard against NaN
        p = p.copy()
        p["null_p95"] = p["null_p95"].fillna(p["observed"])
        p["null_mean"] = p["null_mean"].fillna(0.0)
        ax.fill_between(x - 0.32, p["null_mean"], p["null_p95"],
                        color=NATURE["neutral"], alpha=0.5,
                        label="null 95th pct", zorder=2)
        ax.plot(x - 0.32, p["null_p95"], "o-", color=NATURE["neutral"],
                lw=1.0, ms=4, zorder=3)
        ax.scatter(x + 0.32, p["observed"], s=120,
                   color=NATURE["ours"], edgecolor="white", linewidth=0.8,
                   zorder=4)
        for i, pv in enumerate(p["p_value"]):
            if pd.isna(pv):
                continue
            ax.text(i + 0.32, p["observed"].iloc[i] + 0.005,
                    ss.stars(pv), ha="center", va="bottom",
                    fontsize=10, color=NATURE["ours_d"], fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Permutation test vs null")
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")'''
assert old_h in src, "5H not found"
src = src.replace(old_h, new_h, 1)

# ---------------- 5I: CD - labels right of dots, shorter ----------------
old_i = '''        y = np.arange(len(df))[::-1]
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
        ax.set_title("Critical-difference rank (SIMPLEX #1)")'''
new_i = '''        y = np.arange(len(df))[::-1]
        for i, (m, r) in enumerate(zip(df["reference"], df["rank"])):
            col = NATURE["ours"] if m == paths.MODEL_NAME else NATURE["base"]
            ax.scatter([r], [i], s=150, color=col,
                       edgecolor="white", linewidth=0.8, zorder=4)
            ax.text(r + 0.22, i, f"#{int(r)}", va="center", fontsize=9,
                    color=col, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels([_m_short(m, 9) for m in df["reference"]],
                           fontsize=7.5)
        ax.set_xlabel("rank (#1 = best)")
        ax.set_xlim(0.5, len(df) + 1.0)
        ax.set_title("Critical-difference rank")'''
assert old_i in src, "5I not found"
src = src.replace(old_i, new_i, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 5 patches applied (A/B/C/D/E/F/G/H/I)')
