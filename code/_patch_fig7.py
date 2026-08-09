# -*- coding: utf-8 -*-
"""Figure 7 patches: A waterfall->bars w/ labels, B compact, D/E/F titles,
E x-labels, H donut, I donut."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# ---------------- 7A: waterfall lollipop -> bars, labels right -----------
old_a = '''        names = [short_map.get(v, v.replace("w/o ", "-")) for v in d.index]
        names = [_hard_shorten(n, 16) for n in names]
        ss.lollipop(ax, names, d.values, color=NATURE["ours"],
                    value_fmt="{:.3f}", s=70, label_top=True)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.set_xlabel(f"\\u0394 {pm} (removal cost)")'''
new_a = '''        names = [short_map.get(v, v.replace("w/o ", "-")) for v in d.index]
        names = [_hard_shorten(n, 16) for n in names]
        y = np.arange(len(names))[::-1]
        colors = [NATURE["ours"] if v == d.max() else NATURE["base"]
                  for v in d.values]
        ax.barh(y, d.values, height=0.6, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        pad = max(d.max() * 0.03, 0.002)
        for yy, v in zip(y, d.values):
            ax.text(v + pad, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=6.5, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.0)
        ax.set_xlim(0, d.max() * 1.22)
        ax.set_xlabel(f"\\u0394 {pm} (removal cost)")'''
assert old_a in src, "7A not found"
src = src.replace(old_a, new_a, 1)

# ---------------- 7B: variant bars with compact labels -------------------
old_b = '''        g = ctx.abl.groupby("variant")[pm].mean().sort_values(ascending=False)
        ss.lollipop(ax, [_hard_shorten(v.replace("w/o ", "\\u2212")
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
new_b = '''        g = ctx.abl.groupby("variant")[pm].mean().sort_values(ascending=False)
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
        labels = [short_map.get(v, v.replace("w/o ", "-"))[:9]
                  for v in g.index]
        y = np.arange(len(g))[::-1]
        colors = [NATURE["ours"] if v == "full model" else NATURE["base"]
                  for v in g.index]
        ax.barh(y, g.values, height=0.6, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v in zip(y, g.values):
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=6.0, color="#333333")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlim(0.65, g.max() + 0.05)
        ax.set_xlabel(pm)'''
assert old_b in src, "7B not found"
src = src.replace(old_b, new_b, 1)

# ---------------- 7C: fusion labels shorter -------------------------------
old_c = '''        names = [n.replace("fusion = ", "")[:10] for n in names]
        colors = [NATURE["base"]] * (len(names) - 1) + [NATURE["ours"]]
        ss.lollipop(ax, names, vals, color=NATURE["neutral"],
                    value_fmt="{:.3f}", s=70, label_top=False)
        for x, y, c in zip(vals, range(len(names)), colors):
            ax.scatter([x], [y], s=85, color=c,
                       edgecolor="white", linewidth=0.8, zorder=4)
        ax.set_xlabel(pm)'''
new_c = '''        names = [n.replace("fusion = ", "")[:8] for n in names]
        colors = [NATURE["base"]] * (len(names) - 1) + [NATURE["ours"]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.004, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=7, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=7.5)
        ax.set_xlim(0.6, max(vals) + 0.05)
        ax.set_xlabel(pm)'''
assert old_c in src, "7C not found"
src = src.replace(old_c, new_c, 1)

# ---------------- 7D: forest - shorter title + abbreviated labels --------
old_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(sig_map.get(v, v), 16)
                            for v in s["variant"]], fontsize=6.0)
        ax.set_xlabel(f"\\u0394 {pm} (Holm-adjusted, 95% CI)")
        ax.set_title("Statistical contribution (forest plot, top 12)")'''
new_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(sig_map.get(v, v), 14)
                            for v in s["variant"]], fontsize=5.8)
        ax.set_xlabel(f"\\u0394 {pm} (Holm, 95% CI)")
        ax.set_title("Statistical contribution (top 12)")'''
assert old_d in src, "7D not found"
src = src.replace(old_d, new_d, 1)

# ---------------- 7E: variant violin - compact x labels ------------------
old_e = '''        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        ax.set_ylabel(pm)
        ax.set_title("Variant performance (top 6, violin + mean)")'''
new_e = '''        ax.set_xticks(positions)
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=5.8)
        ax.set_ylabel(pm)
        ax.set_title("Variant performance (top 6)")'''
assert old_e in src, "7E not found"
src = src.replace(old_e, new_e, 1)

# ---------------- 7F: dumbbell - shorter title ----------------------------
old_f = '''        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlabel(pm)
        ax.set_title("Per-variant effect (full vs ablated, top 12)")'''
new_f = '''        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlabel(pm)
        ax.set_title("Per-variant effect (top 12)")'''
assert old_f in src, "7F not found"
src = src.replace(old_f, new_f, 1)

# ---------------- 7H: stacked bar -> donut ---------------------------------
old_h = '''        ax.barh([0], [inter / s], color=NATURE["ours"], height=0.4, zorder=3)
        ax.barh([0], [marg / s], left=[inter / s], color=NATURE["base"],
                height=0.4, zorder=3)
        ax.text(inter / s / 2, 0,
                f"interaction\\n{inter / s * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.text(inter / s + marg / s / 2, 0,
                f"marginal\\n{marg / s * 100:.0f}%",
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("cumulative contribution share")
        ax.set_title("Marginal vs interaction contribution")'''
new_h = '''        wedges, _ = ax.pie(
            [inter / s, marg / s],
            colors=[NATURE["ours"], NATURE["base"]],
            startangle=90, counterclock=False,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
        ax.text(0, 0.12, f"interaction\\n{inter / s * 100:.0f}%",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")
        ax.text(0, -0.30, f"marginal\\n{marg / s * 100:.0f}%",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")
        ax.set_title("Marginal vs interaction")'''
assert old_h in src, "7H not found"
src = src.replace(old_h, new_h, 1)

# ---------------- 7I: pruning stacked bar -> donut -------------------------
old_i = '''        ax.barh([0], [keep / s], color=NATURE["ours"], height=0.4, zorder=3)
        ax.barh([0], [prune / s], left=[keep / s],
                color=NATURE["bad"], height=0.4, zorder=3)
        ax.text(keep / s / 2, 0, f"retained\\n{keep}",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold")
        ax.text(keep / s + prune / s / 2, 0, f"pruned\\n{prune}",
                ha="center", va="center", fontsize=8.5, color="white",
                fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_xlabel("pruning summary")
        ax.set_title("Retention vs pruning decision")'''
new_i = '''        ax.pie([keep / s, prune / s],
               colors=[NATURE["ours"], NATURE["bad"]],
               startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
        ax.text(0, 0.12, f"retained\\n{keep}",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")
        ax.text(0, -0.30, f"pruned\\n{prune}",
                ha="center", va="center", fontsize=9, color="white",
                fontweight="bold")
        ax.set_title("Retention vs pruning")'''
assert old_i in src, "7I not found"
src = src.replace(old_i, new_i, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 7 patches applied (A/B/C/D/E/F/H/I)')
