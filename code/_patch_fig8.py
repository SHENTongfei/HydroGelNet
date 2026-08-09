# -*- coding: utf-8 -*-
"""Figure 8 patches: layout 3x3 -> 4x2 (drop redundant latent-condition panel),
B y-labels, D chart type, H volcano overlap, I y-labels."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# ---------------- layout: 3x3 -> 4x2 --------------------------------------
old_layout = '''def fig8(ctx: Ctx) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(DOUBLE_COL, 6.4))
    axes = axes.ravel()'''
new_layout = '''def fig8(ctx: Ctx) -> None:
    # 4x2 layout: latent-space condition panel was redundant (same cluster
    # structure as target colouring), so 8 informative panels remain.
    fig, axes = plt.subplots(4, 2, figsize=(DOUBLE_COL, 9.2))
    axes = axes.ravel()'''
assert old_layout in src, "8 layout not found"
src = src.replace(old_layout, new_layout, 1)

# ---------------- 8B: abbreviate y labels ---------------------------------
old_b = '''        d = ctx.stab.head(12)
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
        ax.set_xlabel("stability-selection frequency")'''
new_b = '''        d = ctx.stab.head(12)
        names = [_m_short(f, 12) for f in d["feature"]]
        ss.lollipop(ax, names, d["selection_frequency"].values,
                    color=NATURE["base"], value_fmt="{:.2f}", s=55,
                    label_top=False)
        # top features coloured in ours colour; values to the right
        for x, y in zip(d["selection_frequency"].values, range(len(names))):
            col = NATURE["ours"] if x >= 0.95 else NATURE["base"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + 0.03, y, f"{x:.2f}", va="center", ha="left",
                    fontsize=6.0, color=col)
        ax.axvline(0.8, ls="--", lw=0.8, color=NATURE["neutral"])
        ax.set_xlim(0, 1.15)
        ax.set_xlabel("stability-selection frequency")'''
assert old_b in src, "8B not found"
src = src.replace(old_b, new_b, 1)

# ---------------- 8D: heatmap -> grouped bars (attention by condition) ---
old_d = '''    def p_attn_cond(ax):
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
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)'''
new_d = '''    def p_attn_cond(ax):
        if ctx.attn_c is None:
            _note(ax)
            return
        piv = ctx.attn_c.pivot_table(index="token", columns="condition",
                                     values="attention_mean")
        # line plot per token across conditions (heatmap -> trends)
        for i, tok in enumerate(piv.index[:8]):
            ax.plot(piv.columns, piv.loc[tok].values, "o-",
                    color=OKABE_ITO[i % len(OKABE_ITO)], lw=1.3, ms=3.5,
                    label=_m_short(tok, 10), zorder=3)
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([_hard_shorten(c, 8) for c in piv.columns],
                           rotation=45, ha="right", fontsize=6.5)
        ax.set_xlabel("condition")
        ax.set_ylabel("attention")
        ax.set_title("Attention by condition (per-token trend)")
        ax.legend(fontsize=5.5, frameon=False, loc="best")'''
assert old_d in src, "8D not found"
src = src.replace(old_d, new_d, 1)

# ---------------- 8E: latent target - shorten title -----------------------
old_e = '''        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (target-coloured)")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:14])'''
new_e = '''        ax.set_xlabel(xk)
        ax.set_ylabel(yk)
        ax.set_title("Latent space (target-coloured)")
        plt.colorbar(sc, ax=ax, fraction=.046, pad=.03,
                     label=ctx.targets[0][:10])'''
assert old_e in src, "8E not found"
src = src.replace(old_e, new_e, 1)

# ---------------- 8F: removed (drop from fns) -----------------------------
# keep the def but it will not be wired; comment out wiring below
old_fn_list = '''    fns = [p_imp, p_stab, p_attn, p_attn_cond, p_latent_y,
           p_latent_c, p_pdp, p_volcano, p_rules]'''
new_fn_list = '''    fns = [p_imp, p_stab, p_attn, p_attn_cond, p_latent_y,
           p_pdp, p_volcano, p_rules]'''
assert old_fn_list in src, "8 fns not found"
src = src.replace(old_fn_list, new_fn_list, 1)

# ---------------- 8H: volcano - stronger label offsets --------------------
old_h = '''        # show top 6 labels only, spread vertically with distinct offsets
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
                                        color=NATURE["neutral"]))'''
new_h = '''        # show top 6 labels only; stagger text far from both axis and dots
        top = m[sig].nlargest(6, "nlp").sort_values("stat")
        # place labels at alternating x-offsets in open space beside dots
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
assert old_h in src, "8H not found"
src = src.replace(old_h, new_h, 1)

# ---------------- 8I: rules - abbreviate y labels -------------------------
old_i = '''        m = m[m["tier"] == "high"].sort_values("signed")
        names = [_hard_shorten(f, 16) for f in m["feature"]]
        ss.lollipop(ax, names, m["signed"].values,
                    color=NATURE["neutral"], value_fmt="{:+.3f}",
                    s=55, label_top=False)
        for x, y in zip(m["signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("signed importance (\\u00b1 = direction)")'''
new_i = '''        m = m[m["tier"] == "high"].sort_values("signed")
        names = [_m_short(f, 12) for f in m["feature"]]
        ss.lollipop(ax, names, m["signed"].values,
                    color=NATURE["neutral"], value_fmt="{:+.3f}",
                    s=55, label_top=False)
        for x, y in zip(m["signed"].values, range(len(names))):
            col = NATURE["good"] if x > 0 else NATURE["bad"]
            ax.scatter([x], [y], s=60, color=col,
                       edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x + (0.004 if x > 0 else -0.004), y, f"{x:+.3f}",
                    va="center", ha="left" if x > 0 else "right",
                    fontsize=5.8, color=col)
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlabel("signed importance (\u00b1 = direction)")'''
assert old_i in src, "8I not found"
src = src.replace(old_i, new_i, 1)

# also shorten 8A title
old_a_title = '''        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (sign = direction)")'''
new_a_title = '''        ax.tick_params(axis="y", labelsize=6.0)
        ax.set_xlabel("signed permutation importance")
        ax.set_title("Top features (signed)")'''
assert old_a_title in src, "8A title not found"
src = src.replace(old_a_title, new_a_title, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 8 patches applied (layout 4x2, B/D/E/H/I + A title)')
