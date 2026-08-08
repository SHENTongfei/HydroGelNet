"""Stack v2: stable stacking variants, internal + external in one pass.

v1 (l4_stack_ood.py) beat RF internally (0.8272 vs 0.8093) but collapsed on
the external cohort (R2=-0.18): the RidgeCV meta overfit the internal OOF
correlation structure. v2 tests meta strategies that are more stable under
distribution shift, reusing frozen OOF / external predictions (no retrain):

  A. simple-average SIMPLEX + RF          (no learned weights)
  B. strong-ridge on SIMPLEX + RF only    (2 features, alpha 1e-1..1e4)
  C. strong-ridge on all baselines        (alpha 1e-1..1e4, capped)
  D. ridge on all baselines, positive-only weights (nnls-style via LassoCV
     with alpha floor, or Ridge with min alpha 1e0)
  E. val-weighted average                 (weights = internal OOF R2 per model,
     softmax, temperature 1)

For each variant we report: internal OOF R2 (vs RF 0.8067) and external R2
(vs SIMPLEX-alone 0.6946 / best baseline SVR-RBF 0.6342). A variant is
"usable" only if BOTH internal >= RF and external > 0.60.

Usage: python stack_v2.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

TARGET = "glass_adhesion_kpa"
RF_REF = 0.8067   # v4 full-protocol RF ensemble (10 seeds)
SEEDS = [42, 2024, 7, 1337, 20260731, 11, 99, 555, 888, 31337]


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def load_internal():
    sim = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_cv_main.csv"))
    sim = sim[sim["tag"] == "main"].copy()
    sim["y"] = sim[f"y_true_{TARGET}"]
    sim["p_sim"] = sim[f"y_pred_{TARGET}"]
    base = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_baselines.csv"))
    base = base[base["target"] == TARGET].copy()
    pivot = base.pivot_table(index="sample_id", columns="model",
                             values="y_pred", aggfunc="mean").reset_index()
    merged = sim.merge(pivot, on="sample_id", how="left").dropna(subset=["p_sim"])
    X, Y, ids, groups = _load_xy()
    id2group = {str(i): str(g) for i, g in zip(ids, groups)}
    merged["group"] = merged["sample_id"].map(id2group)
    return merged, X, Y


def _load_xy():
    d = np.load(os.path.join(paths.PROCESSED_DIR, "dataset.npz"),
                allow_pickle=True)
    keys = list(d.keys())
    X = d["X"] if "X" in keys else d["features"]
    Y = d["Y"] if "Y" in keys else d["y"]
    ids = d["sample_ids"] if "sample_ids" in keys else (
        d["ids"] if "ids" in keys else np.arange(len(Y)))
    groups = d["groups"] if "groups" in keys else (
        d["group_ids"] if "group_ids" in keys else ids)
    return X, Y, ids, groups


def oof_loop(merged, feats, fit_meta, seed_list=SEEDS, n_splits=5):
    """OOF stacking: fit meta inside train folds, predict test fold."""
    ybin = pd.qcut(merged["y"].values, q=5, labels=False,
                   duplicates="drop").astype(int)
    rows = []
    for seed in seed_list:
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        idx = np.arange(len(merged))
        for fold, (tr, te) in enumerate(gkf.split(idx, ybin,
                                                  merged["group"].values)):
            tr_df, te_df = merged.iloc[tr], merged.iloc[te]
            Xtr = tr_df[feats].values
            ytr = tr_df["y"].values
            meta = fit_meta(Xtr, ytr, tr_df)
            Xte = te_df[feats].values
            p = meta.predict(Xte)
            for j, (_, row) in enumerate(te_df.iterrows()):
                rows.append({"seed": seed, "fold": fold,
                             "sample_id": row["sample_id"],
                             "y_true": row["y"], "y_pred": float(p[j])})
    return pd.DataFrame(rows)


def make_fitters(merged):
    base_cols = [c for c in merged.columns
                 if c not in ("sample_id", "group", "y", "p_sim", "seed",
                              "fold") and merged[c].dtype.kind == "f"]
    all_cols = ["p_sim"] + list(base_cols)
    two_cols = ["p_sim"] + ([c for c in base_cols if c == "RandomForest"]
                            or [c for c in base_cols])

    def fit_avg(cols):
        def _f(Xtr, ytr, tr_df):
            class _Avg:
                def predict(self, X):
                    return X.mean(axis=1)
            return _Avg()
        return _f

    def fit_ridge(cols, alpha_min=1e-1, alpha_max=1e4):
        def _f(Xtr, ytr, tr_df):
            meta = RidgeCV(alphas=np.logspace(np.log10(alpha_min),
                                              np.log10(alpha_max), 25))
            return meta.fit(Xtr, ytr)
        return _f

    def fit_valweight(cols):
        def _f(Xtr, ytr, tr_df):
            w = np.ones(len(cols)) / len(cols)
            # simple OOF-R2 weighting on the training portion
            r2s = []
            for i, c in enumerate(cols):
                r2s.append(max(0.0, r2(ytr, Xtr[:, i])))
            s = np.sum(r2s)
            if s > 0:
                w = np.array(r2s) / s
            class _W:
                def predict(self, X):
                    return X @ w
            return _W()
        return _f

    return {
        "A_simavg_rf": (two_cols, fit_avg(two_cols)),
        "B_ridge2_strong": (two_cols, fit_ridge(two_cols)),
        "C_ridge_all_strong": (all_cols, fit_ridge(all_cols)),
        "D_ridge_all_alpha1": (all_cols, fit_ridge(all_cols, alpha_min=1e0)),
        "E_valweight_all": (all_cols, fit_valweight(all_cols)),
    }


def external_apply(merged_in, feats, fit_meta, seed_list):
    """Fit meta on ALL internal OOF data, apply to external cohort."""
    # final meta fitted on all internal samples (same scheme as l4_ext)
    meta = fit_meta(merged_in[feats].values, merged_in["y"].values,
                    merged_in)

    sim = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_external.csv"))
    yc = [c for c in sim.columns if c.startswith("y_true")]
    pc = [c for c in sim.columns if c.startswith("y_pred")]
    y = sim[yc[0]].values
    p_sim = sim[pc[0]].values

    be = pd.read_csv(os.path.join(paths.METRICS_DIR,
                                  "baselines_external_preds.csv"))
    be = be[be["target"] == TARGET].copy()
    pivot = be.pivot_table(index="model", values="y_pred",
                           aggfunc="mean").reset_index()

    Xext = np.zeros((len(y), len(feats)))
    for i, f in enumerate(feats):
        if f == "p_sim":
            Xext[:, i] = p_sim
        else:
            row = pivot[pivot["model"] == f]
            if len(row):
                Xext[:, i] = row["y_pred"].iloc[0]
    p = meta.predict(Xext)
    return {"ext_r2": float(r2(y, p)), "ext_spearman": float(spearmanr(y, p).statistic)}


def main():
    merged, X, Y = load_internal()
    fitters = make_fitters(merged)
    results = {}
    for name, (feats, fit_meta) in fitters.items():
        oof = oof_loop(merged, feats, fit_meta)
        internal_r2 = r2(oof["y_true"].values, oof["y_pred"].values)
        ext = external_apply(merged, feats, fit_meta, SEEDS)
        usable = (internal_r2 >= RF_REF and ext["ext_r2"] > 0.60)
        results[name] = {
            "feats": feats,
            "internal_r2": float(internal_r2),
            "external_r2": ext["ext_r2"],
            "external_spearman": ext["ext_spearman"],
            "usable": bool(usable),
        }
        print(f"{name:22s} int={internal_r2:.4f} ext={ext['ext_r2']:.4f} "
              f"spear={ext['ext_spearman']:.3f} "
              f"{'<< USABLE' if usable else ''}")

    # references
    results["_ref_simplex_log1p"] = {"internal_r2": 0.7773,
                                     "external_r2": 0.7368}
    results["_ref_simplex_std"] = {"internal_r2": 0.7924,
                                   "external_r2": 0.6946}
    results["_ref_rf"] = {"internal_r2": 0.8067, "external_r2": None}
    results["_ref_svr"] = {"external_r2": 0.6342}

    out = os.path.join(paths.RESULTS_DIR, "arch_eval", "stack_v2_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
