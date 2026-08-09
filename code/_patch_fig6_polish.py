# -*- coding: utf-8 -*-
"""Figure 6 final polish: A title shorten, E label offset, G label spacing."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# 6A: shorten title
old_a = 'ax.set_title("Prospective cohort: predicted vs observed")'
new_a = 'ax.set_title("Prospective: predicted vs observed")'
assert old_a in src
src = src.replace(old_a, new_a, 1)

# 6E: pull labels further right so they don't kiss the bar end
old_e = '''        ax.text(i_val + 0.006, -0.22, f"CV {i_val:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["base_d"],
                fontweight="bold")
        ax.text(e_val + 0.006, 0.22, f"Prosp {e_val:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["ours_d"],
                fontweight="bold")'''
new_e = '''        pad = 0.018
        ax.text(i_val + pad, -0.22, f"CV {i_val:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["base_d"],
                fontweight="bold")
        ax.text(e_val + pad, 0.22, f"Prosp {e_val:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["ours_d"],
                fontweight="bold")'''
assert old_e in src
src = src.replace(old_e, new_e, 1)

# 6G: stagger labels further, abbreviate, increase left padding
old_g = '''        # slope chart: labels beside each end, staggered vertically
        used = []
        for i, (li, le, lab) in enumerate(zip(i_vals, e_vals, labels)):
            col = (NATURE["ours"] if lab == paths.MODEL_NAME
                   else NATURE["base"])
            lw = 2.6 if lab == paths.MODEL_NAME else 1.0
            ax.plot([0, 1], [li, le], "-", color=col, lw=lw, zorder=2)
            sz = 120 if lab == paths.MODEL_NAME else 55
            ax.scatter([0, 1], [li, le], s=sz, color=col,
                       edgecolor="white", linewidth=0.7, zorder=4)
            lab_s = _m_short(lab, 8)
            # left label at start value (stagger if too close)
            oy = 0.0
            for (uu, yy) in used:
                if abs(yy - li) < 0.012:
                    oy = 0.014 if li >= uu else -0.014
            ax.text(-0.30, li + oy, lab_s, va="center", ha="left",
                    fontsize=7, color=col,
                    fontweight="bold" if lab == paths.MODEL_NAME else "normal")
            used.append((li, li))
        ax.set_xlim(-0.42, 1.28)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal", "External"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Internal \u2192 external transfer")'''
new_g = '''        # slope chart: labels beside each end, staggered vertically
        used = []
        for i, (li, le, lab) in enumerate(zip(i_vals, e_vals, labels)):
            col = (NATURE["ours"] if lab == paths.MODEL_NAME
                   else NATURE["base"])
            lw = 2.6 if lab == paths.MODEL_NAME else 1.0
            ax.plot([0, 1], [li, le], "-", color=col, lw=lw, zorder=2)
            sz = 120 if lab == paths.MODEL_NAME else 55
            ax.scatter([0, 1], [li, le], s=sz, color=col,
                       edgecolor="white", linewidth=0.7, zorder=4)
            lab_s = _m_short(lab, 7)
            # left label at start value with stronger stagger
            oy = 0.0
            sign = 1
            for (uu, yy) in used:
                if abs(yy - li) < 0.04:
                    oy = 0.035 if sign > 0 else -0.035
                    sign *= -1
            ax.text(-0.50, li + oy, lab_s, va="center", ha="left",
                    fontsize=6.5, color=col,
                    fontweight="bold" if lab == paths.MODEL_NAME else "normal")
            used.append((li, li))
        ax.set_xlim(-0.62, 1.28)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Internal", "External"], fontsize=8)
        ax.set_ylabel(pm)
        ax.set_title("Internal \u2192 external transfer")'''
assert old_g in src
src = src.replace(old_g, new_g, 1)

# 6F: shorter xlabel
old_f = 'ax.set_xlabel("predicted-rank quartile")'
new_f = 'ax.set_xlabel("predicted-rank quartile (Q1\u2013Q4)")'
assert old_f in src
src = src.replace(old_f, new_f, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 6 final polish done')