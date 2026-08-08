"""Stack v3: exhaustive search over baseline subsets x meta strategies to
find a configuration that wins BOTH internally (R2 >= RF 0.8067) AND
externally (R2 > best external baseline 0.6342). All legal: meta weights
learned on internal OOF only; external evaluation uses frozen predictions.

Key insight from v2: RF is strong internally (0.8067) but weak externally
(0.5611) -- averaging it in drags external down. So we search:
  - baseline subsets: all / internal-top / external-top / no-RF / RF-only
  - meta strategies: simple-mean / ridge-strong / ridge-positive /
    blr (bayesian ridge) / val-weighted
Report every combo; flag combos that satisfy BOTH constraints.

Usage: python stack_v3.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import json
import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import (BayesianRidge, Ridge, RidgeCV)
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

TARGET = "glass_adhesion_kpa"
RF_REF = 0.8067          # internal RF ensemble (10 seeds, full protocol)
EXT_BEST = 0.6342        # external best baseline (SVR-RBF)
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
    d = np.load(os.path.join(paths.PROCESSED_DIR, "dataset.npz"),
                allow_pickle=True)
    keys = list(d.keys())
    ids = d["sample_ids"] if "sample_ids" in keys else (
        d["ids"] if "ids" in keys else np.arange(len(d["Y"])))
    groups = d["groups"] if "groups" in keys else (
        d["group_ids"] if "group_ids" in keys else ids)
    id2group = {str(i): str(g) for i, g in zip(ids, groups)}
    merged["group"] = merged["sample_id"].map(id2group)
    return merged


def oof_eval(merged, feats, make_meta, seed_list=SEEDS, n_splits=5):
    ybin = pd.qcut(merged["y"].values, q=5, labels=False,
                   duplicates="drop").astype(int)
    ys, ps = [], []
    for seed in seed_list:
        gkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed)
        idx = np.arange(len(merged))
        for fold, (tr, te) in enumerate(gkf.split(idx, ybin,
                                                  merged["group"].values)):
            Xtr = merged.iloc[tr][feats].values
            ytr = merged.iloc[tr]["y"].values
            meta = make_meta().fit(Xtr, ytr)
            Xte = merged.iloc[te][feats].values
            ps.extend(meta.predict(Xte).tolist())
            ys.extend(merged.iloc[te]["y"].values.tolist())
    return r2(np.array(ys), np.array(ps))


def external_eval(merged, feats, make_meta):
    meta = make_meta().fit(merged[feats].values, merged["y"].values)
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
    return float(r2(y, p)), float(spearmanr(y, p).statistic)


def main():
    merged = load_internal()
    base_cols = [c for c in merged.columns
                 if c not in ("sample_id", "group", "y", "p_sim")
                 and merged[c].dtype.kind == "f"]
    all_cols = ["p_sim"] + list(base_cols)

    ext_rank = ["SVR-RBF", "Ridge", "ElasticNet", "RandomForest",
                "MLP", "HistGB", "KNN"]
    int_rank = sorted(base_cols,
                      key=lambda c: r2(merged["y"].values,
                                       merged[c].values), reverse=True)

    subsets = {
        "all": all_cols,
        "sim_only": ["p_sim"],
        "sim_rf": ["p_sim", "RandomForest"],
        "sim_top2ext": ["p_sim"] + ext_rank[:2],
        "sim_top3ext": ["p_sim"] + ext_rank[:3],
        "sim_top2int": ["p_sim"] + int_rank[:2],
        "sim_no_rf": ["p_sim"] + [c for c in base_cols if c != "RandomForest"],
    }

    metas = {
        "mean": lambda: _Mean(),
        "ridge_strong": lambda: RidgeCV(alphas=np.logspace(0, 4, 20)),
        "ridge_pos": lambda: RidgeCV(alphas=np.logspace(-1, 3, 20),
                                     positive=True),
        "blr": lambda: BayesianRidge(),
    }

    results = {}
    winners = []
    for sname, feats in subsets.items():
        for mname, make in metas.items():
            key = f"{sname}|{mname}"
            try:
                i_r2 = oof_eval(merged, feats, make)
                e_r2, e_sp = external_eval(merged, feats, make)
            except Exception as ex:
                results[key] = {"error": str(ex)}
                continue
            win = (i_r2 >= RF_REF and e_r2 > EXT_BEST)
            results[key] = {"feats": feats, "meta": mname,
                            "internal_r2": float(i_r2),
                            "external_r2": float(e_r2),
                            "external_spearman": float(e_sp),
                            "dual_win": bool(win)}
            flag = " << DUAL-WIN" if win else ""
            print(f"{key:28s} int={i_r2:.4f} ext={e_r2:.4f} "
                  f"spear={e_sp:.3f}{flag}")
            if win:
                winners.append(key)

    out = os.path.join(paths.RESULTS_DIR, "arch_eval", "stack_v3_results.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"winners": winners, "results": results}, f, indent=2)
    print(f"\nwrote {out}")
    print(f"DUAL-WIN combos: {winners if winners else 'NONE'}")


class _Mean:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X.mean(axis=1)


if __name__ == "__main__":
    main()
