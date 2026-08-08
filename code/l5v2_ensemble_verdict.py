"""L5v2 完成后立即计算 ensemble R2 + 分支裁决（不等 gate，train 落盘即用）。

用法：python l5v2_ensemble_verdict.py
前置：results/preds/preds_cv_main.csv 已更新（L5v2 train 完成）
输出：audit/l5v2_verdict.json + 控制台
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
PREDS = os.path.join(ROOT, "results", "preds")
METRICS = os.path.join(ROOT, "results", "metrics")
TUNING = os.path.join(ROOT, "results", "tuning")
AUDIT = os.path.join(ROOT, "audit")

RF_SINGLE = 0.8093
RF_ENSEMBLE = 0.8164  # RF 25-model ensemble ref (computed from baselines preds)
TARGET = "glass_adhesion_kpa"


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def main():
    # ---- config validation (H31) ----
    cfg_p = os.path.join(TUNING, "best_config_final.json")
    with open(cfg_p, encoding="utf-8") as f:
        cfg = json.load(f)
    cu_p = os.path.join(TUNING, "config_used.json")
    with open(cu_p, encoding="utf-8") as f:
        cu = json.load(f)
    cfg_ok = (cfg.get("use_swa") is True and cu.get("use_swa") is True
              and cu.get("_inherited_v2") is True)
    print(f"[H31] file swa={cfg.get('use_swa')} config_used swa={cu.get('use_swa')} "
          f"marker={cu.get('_inherited_v2')} -> {'OK' if cfg_ok else 'CONFIG MISMATCH!'}")
    if not cfg_ok:
        print(">>> BLOCKED: config mismatch, do not trust these numbers <<<")
        # still continue to print but mark verdict invalid

    # ---- per-fold + ensemble R2 ----
    preds = pd.read_csv(os.path.join(PREDS, "preds_cv_main.csv"))
    preds = preds[preds["tag"] == "main"]
    yc = f"y_true_{TARGET}"
    pc = f"y_pred_{TARGET}"

    per_fold = r2(preds[yc], preds[pc])
    ens = preds.groupby("sample_id").agg(
        yt=(yc, "first"), yp=(pc, "mean"))
    ens_r2 = r2(ens["yt"].values, ens["yp"].values)

    # per-seed ensemble (leave-one-seed-out style stability)
    seeds = sorted(preds["seed"].unique())
    per_seed = {}
    for s in seeds:
        sub = preds[preds["seed"] == s]
        per_seed[int(s)] = round(r2(sub[yc], sub[pc]), 4)

    # ---- external ----
    ext_p = os.path.join(PREDS, "preds_external.csv")
    ext_r2 = None
    if os.path.exists(ext_p):
        ext = pd.read_csv(ext_p)
        eyc = [c for c in ext.columns if c.startswith("y_true")]
        epc = [c for c in ext.columns if c.startswith("y_pred")]
        if eyc and epc:
            ext_r2 = round(r2(ext[eyc[0]], ext[epc[0]]), 4)

    # ---- verdict ----
    beat_single = ens_r2 > RF_SINGLE
    beat_ens = ens_r2 > RF_ENSEMBLE
    if beat_ens:
        branch = "A_full_win"
        verdict = f"INTERNAL FULL WIN: ensemble R2 {ens_r2:.4f} > RF ensemble {RF_ENSEMBLE}"
    elif beat_single:
        branch = "A_half_win"
        verdict = (f"INTERNAL PARTIAL: ensemble {ens_r2:.4f} > RF single {RF_SINGLE} "
                   f"but < RF ensemble {RF_ENSEMBLE}")
    else:
        branch = "B_tie"
        verdict = f"INTERNAL TIE: ensemble {ens_r2:.4f} vs RF {RF_SINGLE}/{RF_ENSEMBLE}"

    out = {
        "per_fold_r2": round(per_fold, 4),
        "ensemble_r2": round(ens_r2, 4),
        "rf_single_ref": RF_SINGLE,
        "rf_ensemble_ref": RF_ENSEMBLE,
        "per_seed_r2": per_seed,
        "external_r2": ext_r2,
        "config_valid": cfg_ok,
        "branch": branch,
        "verdict": verdict,
        "n_seeds": len(seeds),
    }
    os.makedirs(AUDIT, exist_ok=True)
    with open(os.path.join(AUDIT, "l5v2_verdict.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
