"""L5v3 完成后一键验证 + 裁决（H31 B 输出验证自动化）。

用法：python l5v3_verify.py
前置：cv_outer.csv / config_used.json / preds_cv_main.csv 已出现（train 完成）
步骤：
  1. config_used 验证（必须 swa=True + marker=True，否则 BLOCK）
  2. cv_outer 数值读取（per-fold mean R2）
  3. ensemble R2（5-seed 平均）
  4. 与 RF 0.8093/0.8164 对比 → A/B 分支裁决
  5. 写 audit/l5v3_final_verdict.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
METRICS = os.path.join(ROOT, "results", "metrics")
TUNING = os.path.join(ROOT, "results", "tuning")
PREDS = os.path.join(ROOT, "results", "preds")
AUDIT = os.path.join(ROOT, "audit")
TARGET = "glass_adhesion_kpa"
RF_SINGLE = 0.8093
RF_ENSEMBLE = 0.8164


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def main():
    print("=" * 70)
    print("L5v3 H31 OUTPUT VERIFICATION")
    print("=" * 70)

    # 1. config validation
    cu_p = os.path.join(TUNING, "config_used.json")
    if not os.path.exists(cu_p):
        print("[FAIL] config_used.json missing (train not finished?)")
        return 1
    with open(cu_p, encoding="utf-8") as f:
        cu = json.load(f)
    swa_ok = cu.get("use_swa") is True
    marker_ok = cu.get("_inherited_v2") is True
    print(f"[CONFIG] use_swa={cu.get('use_swa')} marker={cu.get('_inherited_v2')} "
          f"d_model={cu.get('d_model')}")
    if not (swa_ok and marker_ok):
        print("[BLOCK] config_used does NOT show inherited config (swa=True)!"
              " -> DO NOT trust numbers, investigate first")
        return 1
    print("[OK] config_used confirms inherited config (swa=True, marker=True)")

    # 2. per-fold + ensemble
    cv_p = os.path.join(METRICS, "cv_outer.csv")
    if not os.path.exists(cv_p):
        print("[FAIL] cv_outer.csv missing")
        return 1
    cv = pd.read_csv(cv_p)
    per_fold = float(cv["R2"].mean())
    print(f"[CV] per-fold mean R2 = {per_fold:.4f} (n={len(cv)})")

    preds_p = os.path.join(PREDS, "preds_cv_main.csv")
    if not os.path.exists(preds_p):
        print("[FAIL] preds_cv_main.csv missing")
        return 1
    preds = pd.read_csv(preds_p)
    preds = preds[preds["tag"] == "main"]
    yc = f"y_true_{TARGET}"
    pc = f"y_pred_{TARGET}"
    ens = preds.groupby("sample_id").agg(yt=(yc, "first"), yp=(pc, "mean"))
    ens_r2 = r2(ens["yt"].values, ens["yp"].values)
    print(f"[ENSEMBLE] 5-seed mean R2 = {ens_r2:.4f}")
    print(f"  RF refs: single {RF_SINGLE} / ensemble {RF_ENSEMBLE}")

    # per-seed
    seeds = sorted(preds["seed"].unique())
    per_seed = {int(s): round(r2(preds[preds["seed"] == s][yc],
                                 preds[preds["seed"] == s][pc]), 4)
                for s in seeds}
    print(f"[PER-SEED] {per_seed}")

    # external
    ext_p = os.path.join(PREDS, "preds_external.csv")
    ext_r2 = None
    if os.path.exists(ext_p):
        ext = pd.read_csv(ext_p)
        eyc = [c for c in ext.columns if c.startswith("y_true")]
        epc = [c for c in ext.columns if c.startswith("y_pred")]
        if eyc and epc:
            ext_r2 = round(r2(ext[eyc[0]], ext[epc[0]]), 4)
            print(f"[EXTERNAL] R2 = {ext_r2}")

    # 3. verdict
    if ens_r2 >= RF_ENSEMBLE:
        branch = "A_full_win"
        verdict = f"INTERNAL FULL WIN: ensemble {ens_r2:.4f} >= RF ensemble {RF_ENSEMBLE}"
    elif ens_r2 > RF_SINGLE:
        branch = "A_half_win"
        verdict = f"INTERNAL PARTIAL WIN: {ens_r2:.4f} > RF single {RF_SINGLE} but < {RF_ENSEMBLE}"
    else:
        branch = "B_tie"
        verdict = f"INTERNAL TIE: {ens_r2:.4f} <= RF single {RF_SINGLE}"

    out = {
        "per_fold_r2": round(per_fold, 4),
        "ensemble_r2": round(ens_r2, 4),
        "rf_single": RF_SINGLE,
        "rf_ensemble": RF_ENSEMBLE,
        "per_seed_r2": per_seed,
        "external_r2": ext_r2,
        "config_valid": True,
        "branch": branch,
        "verdict": verdict,
    }
    os.makedirs(AUDIT, exist_ok=True)
    with open(os.path.join(AUDIT, "l5v3_final_verdict.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n>>> {verdict}")
    print(f">>> branch={branch} -> saved audit/l5v3_final_verdict.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
