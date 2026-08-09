# -*- coding: utf-8 -*-
"""MODEL_SHORT global abbreviation map + rewrite Figure 3 problem panels."""
# This script patches figures.py. Model names everywhere are shortened.

P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# ---------------------------------------------------------------- 1) helper
HELPER = '''
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
}
def _m_short(m, n=9):
    s = MODEL_SHORT.get(str(m), str(m))
    return s if len(s) <= n else _hard_shorten(s, n)
'''
src = src.replace('PANEL_DX = -0.22', HELPER + '\nPANEL_DX = -0.22', 1)

# ---------------------------------------------------------------- 2) fig3 A
old_a = '''    def p_counts(ax):
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
        ax.set_xlabel("number of formulations")'''
new_a = '''    def p_counts(ax):
        names = ["Internal", "Prospective"]
        vals = [len(ds["Y"]), len(ext["Y"]) if ext is not None else 0]
        colors = [NATURE["ours"], NATURE["base"]]
        # twin horizontal bars (side-by-side), values at bar ends
        y = np.arange(2)
        ax.barh(y, vals, height=0.42, color=colors, edgecolor="white",
                linewidth=0.8, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + max(vals) * 0.015, yy, f"{v}", va="center",
                    ha="left", fontsize=9, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.5)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_xlabel("number of formulations")'''
assert old_a in src, "fig3 A not found"
src = src.replace(old_a, new_a, 1)

# ---------------------------------------------------------------- 3) fig3 C
old_c = '''    def p_missing(ax):
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
        ax.set_title(f"Missingness (mean {miss.mean():.2f}%)")'''
new_c = '''    def p_missing(ax):
        miss = np.isnan(ds["X"]).mean(axis=0) * 100
        # horizontal mini-bars for all 21 features (all near 0)
        n = len(miss)
        y = np.arange(n)
        ax.barh(y, miss, height=0.62, color=NATURE["ours_l"],
                edgecolor=NATURE["ours_d"], linewidth=0.4, zorder=3)
        ax.set_yticks([])
        ax.set_xlabel("missing (%)")
        ax.set_xlim(0, max(miss.max() * 6.0, 0.05))
        ax.set_ylabel("21 features")
        ax.set_title(f"Missingness (max {miss.max():.3f}%)")'''
assert old_c in src, "fig3 C not found"
src = src.replace(old_c, new_c, 1)

# ---------------------------------------------------------------- 4) fig3 F
old_f = '''    def p_cond(ax):
        vals = pd.Series(ds["cond"]).value_counts().sort_index()
        labels = [ctx.conds[i][:10] for i in vals.index]
        ss.lollipop(ax, labels, vals.values, color=NATURE["base"],
                    value_fmt="{:.0f}", s=40)'''
new_f = '''    def p_cond(ax):
        vals = pd.Series(ds["cond"]).value_counts().sort_values(
            ascending=False)
        # top 4 conditions + aggregate "others"
        top = vals.head(4)
        other = vals.iloc[4:].sum()
        labels = [ctx.conds[i][:9] for i in top.index] + ["others"]
        counts = list(top.values) + [int(other)]
        colors = ([NATURE["base"]] * 4) + [NATURE["neutral"]]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, counts, height=0.5, color=colors,
                edgecolor="white", linewidth=0.6, zorder=3)
        for yy, v in zip(y, counts):
            ax.text(v + max(counts) * 0.01, yy, f"{v}", va="center",
                    ha="left", fontsize=8, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.set_xlim(0, max(counts) * 1.18)
        ax.set_xlabel("formulations")'''
assert old_f in src, "fig3 F not found"
src = src.replace(old_f, new_f, 1)

# ---------------------------------------------------------------- 5) fig3 G
old_g = '''        ss.lollipop(ax, labels, s["ks_stat"].values, color=NATURE["ours_d"],
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
        ax.set_title("Internal vs external shift (top 12)")'''
new_g = '''        ss.lollipop(ax, labels, s["ks_stat"].values, color=NATURE["ours_d"],
                    value_fmt="{:.2f}", s=30, label_top=False)
        # value labels WELL to the RIGHT of each point (never on the dot)
        y_pos = np.arange(len(s))
        pad = max(s["ks_stat"].max() * 0.05, 0.008)
        for v, y in zip(s["ks_stat"].values, y_pos):
            ax.text(v + pad, y, f"{v:.2f}", va="center", ha="left",
                    fontsize=6.5, color=NATURE["ours_d"])
        ax.invert_yaxis()
        ax.set_xlabel("KS statistic")
        ax.set_xlim(-0.02, max(s["ks_stat"].max() * 1.30, 0.12))
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_title("Internal vs external shift (top 12)")'''
assert old_g in src, "fig3 G not found"
src = src.replace(old_g, new_g, 1)

# ---------------------------------------------------------------- 6) fig3 H
old_h = '''    def p_groups(ax):
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
        ax.set_title(f"Grouping (n={len(sizes)} groups)")'''
new_h = '''    def p_groups(ax):
        sizes = pd.Series(ds["groups"]).value_counts().values
        # histogram of group sizes (one tall bar at size 1)
        bins = np.arange(sizes.min() - 0.5, sizes.max() + 1.5, 1.0)
        ax.hist(sizes, bins=bins, color=NATURE["ours"],
                edgecolor="white", linewidth=0.8, alpha=0.9, zorder=3)
        # annotate the dominant bar
        from collections import Counter
        cc = Counter(sizes)
        top_sz, top_n = cc.most_common(1)[0]
        ax.text(top_sz + 0.3, top_n, f"{top_n} groups of size {top_sz}",
                fontsize=7.5, color=NATURE["ours_d"], va="center")
        ax.set_xlabel("samples per group")
        ax.set_ylabel("groups")
        ax.set_title(f"Group sizes (n={len(sizes)} groups)")'''
assert old_h in src, "fig3 H not found"
src = src.replace(old_h, new_h, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 3 patches applied (A/C/F/G/H)')
