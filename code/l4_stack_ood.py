"""L4: OOF stacking meta-model to break RF 0.8093 internal R^2.

Idea: SIMPLEX deep predictions + strongest baseline OOF predictions + raw
features -> meta-model (ridge/GBM) trained on OOF predictions only
(no leakage: meta features are out-of-fold for every sample).

Protocol: 5-fold grouped CV repeated over 5 seeds, mirroring the main
protocol. The meta-model is refit inside each train fold on the OOF
features of that fold's train portion; test fold gets the frozen meta.

Outputs:
  results/preds/preds_stack_main.csv   (OOF stacked predictions)
  results/metrics/stack_vs_rf.csv      (stack R2 per seed/fold + RF ref)
  results/metrics/stack_gate.json      (verdict vs RF 0.8093)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

TARGET = "glass_adhesion_kpa"
SEEDS = [42, 2024, 7, 1337, 20260731]
RF_REF = 0.80931  # best-in-class tree ensemble (internal, full protocol)


def load_xy():
    """Load internal 316 tensor set from processed/dataset.npz."""
    import numpy as np
    d = np.load(os.path.join(paths.PROCESSED_DIR, "dataset.npz"),
                allow_pickle=True)
    keys = list(d.keys())
    # common key names in this pipeline
    X = d["X"] if "X" in keys else d["features"]
    Y = d["Y"] if "Y" in keys else d["y"]
    ids = d["sample_ids"] if "sample_ids" in keys else (
        d["ids"] if "ids" in keys else np.arange(len(Y)))
    groups = d["groups"] if "groups" in keys else (
        d["group_ids"] if "group_ids" in keys else ids)
    print("dataset.npz keys:", keys)
    return X, Y, ids, groups


def main():
    # ---- 1. load OOF predictions ----
    sim = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_cv_main.csv"))
    sim = sim[sim["tag"] == "main"].copy()
    sim["y"] = sim[f"y_true_{TARGET}"]
    sim["p_sim"] = sim[f"y_pred_{TARGET}"]

    base = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_baselines.csv"))
    base = base[base["target"] == TARGET].copy()

    # pivot baselines to one column per model (mean over the 25 OOF fits:
    # a single stable OOF prediction per sample per baseline).
    pivot = base.pivot_table(index="sample_id", columns="model",
                             values="y_pred", aggfunc="mean").reset_index()

    merged = sim.merge(pivot, on="sample_id", how="left")
    merged = merged.dropna(subset=["p_sim"])

    # ---- 2. build OOF stacked predictions (5 seeds, shuffled grouped CV) ----
    X, Y, ids, groups = load_xy()
    id2group = {str(i): str(g) for i, g in zip(ids, groups)}
    merged["group"] = merged["sample_id"].map(id2group)
    strata = Y[:, 0]  # regression: stratified binning happens inside trainer

    feat_cols = ["p_sim"] + [c for c in pivot.columns if c != "sample_id"]
    stack_rows = []
    # continuous target -> stratified binning (mirror trainer._strata)
    ybin = pd.qcut(merged["y"].values, q=5, labels=False,
                   duplicates="drop").astype(int)
    for seed in SEEDS:
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        idx = np.arange(len(merged))
        splits = gkf.split(idx, ybin, merged["group"].values)
        for fold, (tr, te) in enumerate(splits):
            tr_df, te_df = merged.iloc[tr], merged.iloc[te]
            Xtr = tr_df[feat_cols].values
            ytr = tr_df["y"].values
            Xte = te_df[feat_cols].values

            meta = RidgeCV(alphas=np.logspace(-3, 3, 20))
            meta.fit(Xtr, ytr)
            for j, (_, row) in enumerate(te_df.iterrows()):
                stack_rows.append({
                    "seed": seed, "fold": fold,
                    "sample_id": row["sample_id"],
                    "y_true": row["y"], "y_pred": float(meta.predict(Xte[[j]])[0]),
                    "model": "SIMPLEX-STACK",
                })

    stack = pd.DataFrame(stack_rows)
    stack.to_csv(os.path.join(paths.PREDS_DIR, "preds_stack_main.csv"),
                 index=False)

    # ---- 3. metrics ----
    def r2(y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)

    agg = stack.groupby("seed").apply(
        lambda g: r2(g["y_true"].values, g["y_pred"].values), include_groups=False)
    overall = r2(stack["y_true"].values, stack["y_pred"].values)
    print(f"STACK overall R2 = {overall:.4f}")
    for s, v in agg.items():
        print(f"  seed {s}: {v:.4f}")

    # RF reference (from baselines, full protocol mean R2)
    base_r2 = pd.read_csv(paths.BASELINES_CSV)
    rf_r2 = base_r2[(base_r2["model"] == "RandomForest") &
                    (base_r2["target"] == TARGET)]["R2"].mean()
    print(f"RF ref = {rf_r2:.4f} (paper 0.8093)")

    n_pos = int((agg > RF_REF).sum())
    verdict = "PASS" if (overall > RF_REF and n_pos >= 4) else "FAIL"
    out = {
        "verdict": verdict, "stack_overall_r2": float(overall),
        "rf_ref_r2": float(rf_r2), "delta_vs_rf": float(overall - rf_r2),
        "seeds_above_rf": int(n_pos), "n_seeds": len(SEEDS),
        "per_seed_r2": {str(s): float(v) for s, v in agg.items()},
        "note": "OOF stacking: SIMPLEX + baselines OOF preds -> RidgeCV meta",
    }
    with open(os.path.join(paths.METRICS_DIR, "stack_gate.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
