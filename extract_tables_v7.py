"""Extract table numbers for SIMPLEX v7 rewrite (4 tables with SD in parens).

Table 1: Internal CV metrics per model (R2/RMSE/MAE mean +/- SD across folds)
Table 2: Prospective metrics on 25-formulation cohort (R2, Spearman, AUC,
         Top-10, Top-20) -- from baselines_external.csv + external.csv
Table 3: Generalisation gap (internal R2 -> external R2 per model)
Table 4: Ablation contributions (delta R2, Holm p) -- from ablation csv

All numbers come from real result files; no fabrication.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
METRICS = os.path.join(ROOT, "results", "metrics")
STATS = os.path.join(ROOT, "results", "stats")
ABL = os.path.join(ROOT, "results", "ablation")

TARGET = "glass_adhesion_kpa"


def mean_sd(vals, fmt=".3f"):
    v = np.asarray(vals, dtype=float)
    return f"{v.mean():{fmt}} ({v.std():{fmt}})" if len(v) > 1 else f"{v.mean():{fmt}}"


def main():
    lines = []

    # ---------- Table 1: internal CV per model ----------
    cv = pd.read_csv(os.path.join(METRICS, "cv_outer.csv"))
    base = pd.read_csv(os.path.join(METRICS, "baselines.csv"))
    base = base[base["target"] == TARGET]
    lines.append("=== Table 1: Internal 5-fold grouped CV (5 seeds), mean (SD) ===")
    lines.append(f"{'Model':<14}{'R2':>14}{'RMSE':>14}{'MAE':>14}")
    rows = []
    ours = cv[cv["model"] == "SIMPLEX"]
    rows.append(("SIMPLEX", ours["R2"], ours["RMSE"], ours["MAE"]))
    for m in sorted(base["model"].unique()):
        sub = base[base["model"] == m]
        rows.append((m, sub["R2"], sub["RMSE"], sub["MAE"]))
    rows.sort(key=lambda r: r[1].mean(), reverse=True)
    for name, r2, rmse, mae in rows:
        star = " *" if name == "SIMPLEX" else ""
        lines.append(f"{name:<14}{mean_sd(r2):>14}{mean_sd(rmse):>14}{mean_sd(mae):>14}{star}")
    lines.append("")

    # ---------- Table 2: prospective metrics ----------
    lines.append("=== Table 2: Prospective 25-formulation screening metrics ===")
    ext = pd.read_csv(os.path.join(METRICS, "external.csv"))
    extb = pd.read_csv(os.path.join(METRICS, "baselines_external.csv"))
    extb = extb[extb["target"] == TARGET]
    ours_ext = ext[ext["model"] == "SIMPLEX"]
    lines.append(f"SIMPLEX external R2={ours_ext['R2'].mean():.3f}  "
                 f"Spearman={ours_ext['SpearmanRho'].mean():.3f}")
    # bootstrap CI for external R2
    try:
        boot = pd.read_csv(os.path.join(STATS, "bootstrap_ci.csv"))
        sim = boot[(boot["model"] == "SIMPLEX") & (boot["scope"] == "external")
                   & (boot["metric"] == "R2")]
        if len(sim):
            lines.append(f"SIMPLEX external R2 95% CI [{sim['lo'].iloc[0]:.3f}, "
                         f"{sim['hi'].iloc[0]:.3f}]")
    except Exception as e:
        lines.append(f"(bootstrap CI unavailable: {e})")
    lines.append("")

    # ---------- Table 3: generalisation gap ----------
    lines.append("=== Table 3: Internal -> External generalisation gap ===")
    int_r2 = {name: r2.mean() for name, r2, _, _ in rows}
    ext_r2 = {}
    ext_r2["SIMPLEX"] = ours_ext["R2"].mean()
    for _, r in extb.iterrows():
        m = r["model"]
        ext_r2.setdefault(m, []).append(r["R2"])
    ext_r2 = {m: (np.mean(v) if isinstance(v, list) else v)
              for m, v in ext_r2.items()}
    for m in sorted(set(list(int_r2.keys()) + list(ext_r2.keys()))):
        if m in int_r2 and m in ext_r2:
            gap = int_r2[m] - ext_r2[m]
            lines.append(f"{m:<14} internal {int_r2[m]:.3f} -> external "
                         f"{ext_r2[m]:.3f}  gap {gap:+.3f}")
    lines.append("")

    # ---------- Table 4: ablation ----------
    for f in ["ablation_results.csv", "ablation.csv"]:
        p = os.path.join(ABL, f)
        if os.path.exists(p):
            abl = pd.read_csv(p)
            lines.append(f"=== Table 4: Ablation ({f}) ===")
            lines.append(abl.to_string(index=False))
            break
    else:
        lines.append("=== Table 4: ablation file not found (L5 running) ===")

    out = "\n".join(lines)
    print(out)
    with open(os.path.join(ROOT, "audit", "tables_v7.md"), "w",
              encoding="utf-8") as fh:
        fh.write(out + "\n")


if __name__ == "__main__":
    main()
