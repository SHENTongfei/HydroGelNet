"""L4 variant test: pure-SIMPLEX stacking vs baseline-augmented stacking."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold
import paths
import l4_stack_ood as L

TARGET = "glass_adhesion_kpa"
SEEDS = [42, 2024, 7, 1337, 20260731]
RF_REF = 0.8093
X, Y, ids, groups = L.load_xy()

sim = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_cv_main.csv"))
sim = sim[sim["tag"] == "main"].copy()
sim["y"] = sim[f"y_true_{TARGET}"]
sim["p_sim"] = sim[f"y_pred_{TARGET}"]
id2group = {str(i): str(g) for i, g in zip(ids, groups)}
sim["group"] = sim["sample_id"].map(id2group)
ybin = pd.qcut(sim["y"].values, q=5, labels=False, duplicates="drop").astype(int)


def run_stack(use_raw, use_base, meta="ridge"):
    base = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_baselines.csv"))
    base = base[base["target"] == TARGET].copy()
    pivot = base.pivot_table(index="sample_id", columns="model",
                             values="y_pred", aggfunc="mean").reset_index()
    merged = sim.merge(pivot, on="sample_id", how="left").dropna(
        subset=["p_sim"])
    feat = ["p_sim"]
    if use_raw:
        raw_df = pd.DataFrame(X, columns=[f"x{j}" for j in range(X.shape[1])])
        raw_df["sample_id"] = [str(i) for i in ids]
        merged = merged.merge(raw_df, on="sample_id", how="left")
        feat += [f"x{j}" for j in range(X.shape[1])]
    if use_base:
        feat += [c for c in pivot.columns if c != "sample_id"]
    rows = []
    for seed in SEEDS:
        gkf = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                   random_state=seed)
        for tr, te in gkf.split(np.arange(len(merged)), ybin,
                                merged["group"].values):
            Xtr = merged.iloc[tr][feat].values
            ytr = merged.iloc[tr]["y"].values
            Xte = merged.iloc[te][feat].values
            if meta == "ridge":
                m = RidgeCV(alphas=np.logspace(-3, 3, 20))
            else:
                m = GradientBoostingRegressor(n_estimators=120, max_depth=2,
                                              learning_rate=0.06,
                                              random_state=0)
            m.fit(Xtr, ytr)
            for j, (_, row) in enumerate(merged.iloc[te].iterrows()):
                rows.append({"seed": seed, "sample_id": row["sample_id"],
                             "y_true": row["y"],
                             "y_pred": float(m.predict(Xte[[j]])[0])})
    st = pd.DataFrame(rows)

    def r2(y, p):
        return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)

    overall = r2(st["y_true"].values, st["y_pred"].values)
    per = st.groupby("seed").apply(
        lambda g: r2(g["y_true"].values, g["y_pred"].values),
        include_groups=False)
    return overall, per


if __name__ == "__main__":
    for use_raw, use_base, meta in [
        (False, True, "ridge"),   # baseline-augmented (original L4)
        (True, False, "ridge"),   # pure SIMPLEX + raw X
        (True, False, "gbr"),     # pure SIMPLEX + raw X, GBM meta
    ]:
        ov, per = run_stack(use_raw, use_base, meta)
        tag = f"raw={use_raw} base={use_base} {meta}"
        print(f"{tag:30s} overall R2={ov:.4f}  "
              f"seeds>{RF_REF}: {(per.values > RF_REF).sum()}/5  "
              f"{np.round(per.values, 4).tolist()}")
