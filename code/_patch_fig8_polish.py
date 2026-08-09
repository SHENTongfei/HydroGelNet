# -*- coding: utf-8 -*-
"""Figure 8 polish: A label dedupe (identical x), G/H label overlap."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# 8A: if multiple points share the same x value, label them stacked vertically
old_a = '''        for x, y in zip(m["_signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.003, y, f"{x:.3f}", va="center",
                    ha="left" if x > 0 else "right",
                    fontsize=6.0, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (signed)")'''
new_a = '''        for x, y in zip(m["_signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            # if signed value is 0 or near zero, label might collide with
            # neighbouring points; nudge label below in that case
            oy = 0.18 if abs(x) < 0.003 else 0.0
            ax.text(x + 0.004, y + oy, f"{x:.3f}", va="center",
                    ha="left" if x > 0 else "right",
                    fontsize=6.0, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed importance")
        ax.set_title("Top features (signed)")'''
assert old_a in src
src = src.replace(old_a, new_a, 1)

# 8H: stronger label offset - alternate above/below + add white bbox
old_h = '''        # place labels at alternating x-offsets in open space beside dots
        for idx, (_, r) in enumerate(top.iterrows()):
            xo = r["stat"] * (1.35 if idx % 2 == 0 else 0.62)
            yo = r["nlp"] * (1.28 if idx % 2 == 0 else 0.72)
            ax.annotate(_m_short(str(r["feature"]), 9),
                        xy=(r["stat"], r["nlp"]),
                        xytext=(xo, yo),
                        fontsize=6.0, ha="center", va="center",
                        color=NATURE["ours_d"], fontweight="bold",
                        arrowprops=dict(arrowstyle="-", lw=0.5,
                                        color=NATURE["neutral"]))'''
new_h = '''        # stagger labels across the open top-right quadrant; each label
        # gets a unique vertical offset so no two labels share a y-band
        sorted_top = top.sort_values("nlp")
        n_top = len(sorted_top)
        for idx, (_, r) in enumerate(sorted_top.iterrows()):
            xo = r["stat"] * (1.40 if idx % 2 == 0 else 0.55)
            # distribute y across full vertical range
            yo = r["nlp"] * (1.35 if idx < n_top / 2 else 0.78)
            ax.annotate(_m_short(str(r["feature"]), 8),
                        xy=(r["stat"], r["nlp"]),
                        xytext=(xo, yo),
                        fontsize=6.5, ha="center", va="center",
                        color=NATURE["ours_d"], fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor="white", edgecolor="none",
                                  alpha=0.92),
                        arrowprops=dict(arrowstyle="-", lw=0.4,
                                        color=NATURE["neutral"]))'''
assert old_h in src
src = src.replace(old_h, new_h, 1)

# 8G: PDP - replace inline labels with legend (avoids label-point overlap)
old_g = '''        for i, f in enumerate(feats):
            g = d[d["feature"] == f]
            ax.plot(g["grid_value"], g["pd_mean"], "-o",
                    label=_hard_shorten(str(f), 14),
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    ms=3, lw=1.4)
        ax.set_xlabel("feature value (grid)")
        ax.set_ylabel(f"predicted {t0[:10]}")
        ax.set_title("Partial dependence (top 5 features)")
        ax.legend(fontsize=6.5, frameon=False, loc="best")'''
new_g = '''        for i, f in enumerate(feats):
            g = d[d["feature"] == f]
            ax.plot(g["grid_value"], g["pd_mean"], "-o",
                    color=OKABE_ITO[i % len(OKABE_ITO)],
                    ms=3, lw=1.4,
                    label=f"{i+1}. {_m_short(str(f), 10)}")
        ax.set_xlabel("feature value (grid)")
        ax.set_ylabel(f"predicted {t0[:10]}")
        ax.set_title("Partial dependence (top 5)")
        leg = ax.legend(fontsize=6.0, frameon=True, loc="upper left",
                        title="features", title_fontsize=6.5)
        leg.get_title().set_color(NATURE["ours_d"])'''
assert old_g in src
src = src.replace(old_g, new_g, 1)

# 8I: shorter x label
old_i_xlabel = 'ax.set_xlabel("signed importance (\\u00b1 = direction)")'
new_i_xlabel = 'ax.set_xlabel("signed importance")'
assert old_i_xlabel in src
src = src.replace(old_i_xlabel, new_i_xlabel, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 8 polish done')