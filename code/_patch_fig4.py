# -*- coding: utf-8 -*-
"""Figure 4 patches: A no cell numbers, C label spacing, D redraw, H colour."""
P = r'C:\Users\TS\WorkBuddy\HydroGelNet\code\figures.py'
src = open(P, encoding='utf-8').read()

# ---------------- 4A: remove per-cell numbers from heatmap ----------------
old_a = '''        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.65, vmax=0.85,
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
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)'''
new_a = '''        im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.65, vmax=0.85,
                       aspect="auto")
        ax.set_xticks(range(piv.shape[1]))
        ax.set_xticklabels([f"F{c}" for c in piv.columns], fontsize=7)
        ax.set_yticks(range(piv.shape[0]))
        ax.set_yticklabels([f"{s}" for s in piv.index], fontsize=6.5)
        ax.set_ylabel("seed")
        # no per-cell numbers: colour alone carries the value (less clutter)
        ax.set_title("Per-seed \u00d7 per-fold R\u00b2 (heatmap)")
        plt.colorbar(im, ax=ax, fraction=.046, pad=.03)'''
assert old_a in src, "4A not found"
src = src.replace(old_a, new_a, 1)

# ---------------- 4C: labels to the right of bars, no overlap ------------
old_c = '''        ax.text(v_rf, -0.22, f"  RF {v_rf:.3f}", va="center", ha="left",
                fontsize=8.5, color=NATURE["base_d"], fontweight="bold")
        ax.text(v_ours, 0.22, f"  SIMPLEX {v_ours:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["ours_d"],
                fontweight="bold")'''
new_c = '''        ax.text(v_rf + 0.008, -0.22, f"RF {v_rf:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["base_d"],
                fontweight="bold")
        ax.text(v_ours + 0.008, 0.22, f"SIMPLEX {v_ours:.3f}", va="center",
                ha="left", fontsize=8.5, color=NATURE["ours_d"],
                fontweight="bold")'''
assert old_c in src, "4C not found"
src = src.replace(old_c, new_c, 1)

# ---------------- 4D: residuals - bigger points, band + density ----------
old_d = '''        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
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
        ax.set_title("Residuals vs fitted (binned mean)")'''
new_d = '''        res = g[f"y_pred_{t}"] - g[f"y_true_{t}"]
        ax.scatter(g[f"y_pred_{t}"], res, s=14, alpha=0.6,
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
        ax.fill_between(bx, by - bs, by + bs, color=NATURE["ours_l"],
                        alpha=0.5, zorder=1)
        ax.plot(bx, by, "o-", color=NATURE["ours_d"], lw=1.6, ms=4.5,
                zorder=4)
        ax.set_xlabel("predicted (kPa)")
        ax.set_ylabel("residual (pred \u2212 obs)")
        ax.set_title("Residuals vs fitted (binned mean \u00b1 SD)")'''
assert old_d in src, "4D not found"
src = src.replace(old_d, new_d, 1)

# ---------------- 4H: hexbin colour deepened ------------------------------
old_h = '''        hb = ax.hexbin(g[f"y_true_{t}"], g[f"y_pred_{t}"], gridsize=24,
                       cmap="YlOrRd", mincnt=1)'''
new_h = '''        hb = ax.hexbin(g[f"y_true_{t}"], g[f"y_pred_{t}"], gridsize=24,
                       cmap="inferno", mincnt=1, linewidths=0.3)'''
assert old_h in src, "4H not found"
src = src.replace(old_h, new_h, 1)

open(P, 'w', encoding='utf-8').write(src)
print('Figure 4 patches applied (A/C/D/H)')
