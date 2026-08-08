"""L4 external evaluation: does the stacked model win prospectively too?

Internal stacking beat RF (0.8272 vs 0.8093) but baseline-only stacking is
close (0.8248). The decisive dimension is the 25-formulation prospective
cohort where SIMPLEX alone had R2=0.71 vs RF 0.56. Here we evaluate the
stacked model on the external cohort (both SIMPLEX+baseline stacking and
baseline-only stacking), with FIX-2 parity: every model contributes its
frozen external predictions.

Inputs (must exist from the current pipeline run):
  results/preds/preds_external.csv            SIMPLEX external preds (25 rows)
  results/preds/baselines_external_preds.csv  baseline external preds
Outputs:
  results/metrics/stack_external.json
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
import l4_stack_ood as L  # noqa: E402  (reuses load_xy)

TARGET = "glass_adhesion_kpa"


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def spearman(y, p):
    from scipy.stats import spearmanr
    return float(spearmanr(y, p).statistic)


def main():
    sim = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_external.csv"))
    yc = [c for c in sim.columns if c.startswith("y_true")]
    pc = [c for c in sim.columns if c.startswith("y_pred")]
    y = sim[yc[0]].values
    p_sim = sim[pc[0]].values

    base_pred = os.path.join(paths.METRICS_DIR, "baselines_external_preds.csv")
    if not os.path.exists(base_pred):
        print("MISSING baselines_external_preds.csv (FIX-2 output)")
        return
    be = pd.read_csv(base_pred)
    be = be[be["target"] == TARGET].copy()
    # external preds: one row per (model, seed) -> average per model
    pivot = be.pivot_table(index="model", values="y_pred",
                           aggfunc="mean").reset_index()

    # stacked meta: weights fitted on internal OOF (Ridge), then applied to
    # external predictions. Reuse internal fit to avoid external leakage.
    # ---- refit meta on ALL internal OOF (same scheme as l4) ----
    from sklearn.linear_model import RidgeCV

    sim_in = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_cv_main.csv"))
    sim_in = sim_in[sim_in["tag"] == "main"].copy()
    sim_in["y"] = sim_in[f"y_true_{TARGET}"]
    sim_in["p_sim"] = sim_in[f"y_pred_{TARGET}"]
    base_in = pd.read_csv(os.path.join(paths.PREDS_DIR,
                                       "preds_baselines.csv"))
    base_in = base_in[base_in["target"] == TARGET].copy()
    piv_in = base_in.pivot_table(index="sample_id", columns="model",
                                 values="y_pred", aggfunc="mean").reset_index()
    merged = sim_in.merge(piv_in, on="sample_id", how="left").dropna(
        subset=["p_sim"])

    feat = ["p_sim"] + [c for c in piv_in.columns if c != "sample_id"]
    meta = RidgeCV(alphas=np.logspace(-3, 3, 20))
    meta.fit(merged[feat].values, merged["y"].values)

    # ---- apply to external ----
    ext_feat = {"p_sim": p_sim[0]}
    # align baseline model columns: use the same column order as feat
    col_map = {}
    for m in pivot["model"]:
        col_map[m] = m
    Xext = np.zeros((len(y), len(feat)))
    for i, f in enumerate(feat):
        if f == "p_sim":
            Xext[:, i] = p_sim
        else:
            row = pivot[pivot["model"] == f]
            if len(row):
                Xext[:, i] = row["y_pred"].iloc[0]
    p_stack = meta.predict(Xext)

    # baseline-only stacking weights (same internal fit, drop p_sim)
    feat_b = [c for c in feat if c != "p_sim"]
    meta_b = RidgeCV(alphas=np.logspace(-3, 3, 20))
    meta_b.fit(merged[feat_b].values, merged["y"].values)
    Xext_b = np.zeros((len(y), len(feat_b)))
    for i, f in enumerate(feat_b):
        row = pivot[pivot["model"] == f]
        if len(row):
            Xext_b[:, i] = row["y_pred"].iloc[0]
    p_base_only = meta_b.predict(Xext_b)

    out = {
        "n_external": int(len(y)),
        "simplex_alone_r2": float(r2(y, p_sim)),
        "simplex_alone_spearman": float(spearman(y, p_sim)),
        "stack_r2": float(r2(y, p_stack)),
        "stack_spearman": float(spearman(y, p_stack)),
        "baseline_only_stack_r2": float(r2(y, p_base_only)),
        "baseline_only_stack_spearman": float(spearman(y, p_base_only)),
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(paths.METRICS_DIR, "stack_external.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
