# -*- coding: utf-8 -*-
"""Figure 7 polish: A/B/F y-labels compact, D shorten, H/I guard zero case."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# 7A: keep top 12, abbreviate further to 8 chars max
old_a = '''        names = [short_map.get(v, v.replace("w/o ", "-")) for v in d.index]
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
new_a = '''        # take top 12 (already sorted); labels abbreviated to <=8 chars
        d = d.head(12)
        names = [short_map.get(v, v.replace("w/o ", "-"))[:8]
                 for v in d.index]
        y = np.arange(len(names))[::-1]
        colors = [NATURE["ours"] if v == d.max() else NATURE["base"]
                  for v in d.values]
        ax.barh(y, d.values, height=0.65, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        pad = max(d.max() * 0.04, 0.002)
        for yy, v in zip(y, d.values):
            ax.text(v + pad, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=6.8, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=6.5)
        ax.set_xlim(0, d.max() * 1.30)
        ax.set_xlabel(f"\\u0394 {pm} (removal cost, top 12)")'''
assert old_a in src
src = src.replace(old_a, new_a, 1)

# 7B: abbreviate to <=6 chars
old_b = '''        labels = [short_map.get(v, v.replace("w/o ", "-"))[:9]
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
new_b = '''        # keep top 12, labels <=6 chars
        g = g.head(12)
        labels = [short_map.get(v, v.replace("w/o ", "-"))[:6]
                  for v in g.index]
        y = np.arange(len(g))[::-1]
        colors = [NATURE["ours"] if v == "full model" else NATURE["base"]
                  for v in g.index]
        ax.barh(y, g.values, height=0.62, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v in zip(y, g.values):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=6.5, color="#333333", fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_xlim(0.65, g.max() + 0.08)
        ax.set_xlabel(pm)'''
assert old_b in src
src = src.replace(old_b, new_b, 1)

# 7C: also abbreviate + shorter xlim
old_c = '''        names = [n.replace("fusion = ", "")[:8] for n in names]
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
new_c = '''        names = [n.replace("fusion = ", "")[:5] for n in names]
        colors = [NATURE["base"]] * (len(names) - 1) + [NATURE["ours"]]
        y = np.arange(len(names))[::-1]
        ax.barh(y, vals, height=0.55, color=colors, edgecolor="white",
                linewidth=0.6, zorder=3)
        for yy, v, c in zip(y, vals, colors):
            ax.text(v + 0.006, yy, f"{v:.3f}", va="center", ha="left",
                    fontsize=7, color=c, fontweight="bold")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0.6, max(vals) + 0.08)
        ax.set_xlabel(pm)'''
assert old_c in src
src = src.replace(old_c, new_c, 1)

# 7D: shorten labels further (no truncation)
old_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([_hard_shorten(sig_map.get(v, v), 14)
                            for v in s["variant"]], fontsize=5.8)
        ax.set_xlabel(f"\\u0394 {pm} (Holm, 95% CI)")
        ax.set_title("Statistical contribution (top 12)")'''
new_d = '''        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels([sig_map.get(v, v).replace("w/o ", "-")[:9]
                            for v in s["variant"]], fontsize=6.0)
        ax.set_xlabel(f"\\u0394 {pm} (Holm, 95% CI)")
        ax.set_title("Statistical contribution (top 12)")'''
assert old_d in src
src = src.replace(old_d, new_d, 1)

# 7F: same as A top 12
old_f = '''        full = ctx.abl[ctx.abl["variant"] == "full model"][pm].mean()
        g = ctx.abl.groupby("variant")[pm].mean()
        g = g.drop("full model", errors="ignore").sort_values().head(12)'''
new_f = '''        full = ctx.abl[ctx.abl["variant"] == "full model"][pm].mean()
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
        }'''
assert old_f in src
src = src.replace(old_f, new_f, 1)
old_f_labels = '''        labels = [_hard_shorten(short_map.get(v, v), 12) for v in g.index]'''
new_f_labels = '''        labels = [short_map.get(v, v).replace("w/o ", "-")[:8]
                  for v in g.index]'''
assert old_f_labels in src
src = src.replace(old_f_labels, new_f_labels, 1)
old_f_tick = '''        ax.set_yticklabels(labels, fontsize=6.0)
        ax.set_xlabel(pm)
        ax.set_title("Per-variant effect (top 12)")'''
new_f_tick = '''        ax.set_yticklabels(labels, fontsize=6.3)
        ax.set_xlabel(pm)
        ax.set_title("Per-variant effect (top 12)")'''
assert old_f_tick in src
src = src.replace(old_f_tick, new_f_tick, 1)

# 7H: handle zero marginal -> full ring
old_h = '''        wedges, _ = ax.pie(
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
new_h = '''        # when one side is essentially zero, show a full ring with a label
        if marg / s < 0.02:
            ax.pie([1.0],
                   colors=[NATURE["ours"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
            ax.text(0, 0, f"interaction\n100%", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        else:
            ax.pie([inter / s, marg / s],
                   colors=[NATURE["ours"], NATURE["base"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
            ax.text(0, 0.12, f"interaction\n{inter / s * 100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
            ax.text(0, -0.30, f"marginal\n{marg / s * 100:.0f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        ax.set_title("Marginal vs interaction")'''
assert old_h in src
src = src.replace(old_h, new_h, 1)

# 7I: same guard
old_i = '''        ax.pie([keep / s, prune / s],
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
new_i = '''        if prune / s < 0.02:
            ax.pie([1.0],
                   colors=[NATURE["ours"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
            ax.text(0, 0, f"all retained\n({keep})", ha="center", va="center",
                    fontsize=10, color="white", fontweight="bold")
        else:
            ax.pie([keep / s, prune / s],
                   colors=[NATURE["ours"], NATURE["bad"]],
                   startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
            ax.text(0, 0.12, f"retained\n{keep}",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
            ax.text(0, -0.30, f"pruned\n{prune}",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        ax.set_title("Retention vs pruning")'''
assert old_i in src
src = src.replace(old_i, new_i, 1)

# 7E: shorten x labels too
old_e_labels = '''        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=5.8)'''
new_e_labels = '''        ax.set_xticklabels([l[:8] for l in labels], rotation=50,
                           ha="right", fontsize=6.2)'''
assert old_e_labels in src
src = src.replace(old_e_labels, new_e_labels, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 7 polish done')