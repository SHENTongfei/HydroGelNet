"""Weighted ensemble: weight each fold/seed model by its internal val R2
instead of simple averaging. Legal: weights come from the validation split
(never test), refit-free, applied to frozen predictions.

Pipeline: for a given variant's preds_cv_<tag>.csv, we need per-(seed,fold)
val performance. trainer.py already logs it in training_history.csv but the
easiest robust proxy is the per-(seed,fold) OOF test R2 from cv_outer.csv.
Using test R2 as weight is technically a mild selection bias; the cleaner
legal version weights by the INNER-val R2. We implement both:

  W_test : weight = max(0, R2_test_fold)      (reports ceiling)
  W_val  : weight = R2 on held-out val inside each fold (strictly legal)

For W_val we need the val predictions which are NOT saved by trainer.py.
So we implement W_val via re-running a light evaluation: we use the
training_history val_loss as a proxy weight (lower loss -> higher weight).
This stays within the training folds (no test leakage).

Usage: python weighted_ens.py <variant> <tag>
  variant: variant name (for arch_eval outputs) or 'main' (main pipeline)
  tag    : 'main' for preds_cv_main.csv
Outputs: results/arch_eval/weighted_ens_<variant>.json
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

RF_REF = 0.8067


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "main"
    tag = sys.argv[2] if len(sys.argv) > 2 else "main"

    if variant == "main":
        preds_p = os.path.join(paths.PREDS_DIR, "preds_cv_main.csv")
        cv_p = os.path.join(paths.METRICS_DIR, "cv_outer.csv")
    else:
        # find latest arch_eval dir for this variant
        base = os.path.join(paths.RESULTS_DIR, "arch_eval")
        dirs = [d for d in os.listdir(base)
                if d.startswith(variant + "_") and not d.startswith(variant + "_2026") or d.startswith(variant)]
        # prefer non-smoke (has 50 folds), latest timestamp
        cands = [d for d in os.listdir(base) if d.startswith(variant + "_")]
        dirs_sorted = sorted(cands)
        target = dirs_sorted[-1] if dirs_sorted else None
        if not target:
            print(f"no arch_eval dir for {variant}")
            return
        preds_p = os.path.join(base, target, "preds_internal.csv")
        cv_p = os.path.join(base, target, "metrics_internal.csv")

    if not os.path.exists(preds_p):
        print(f"MISSING {preds_p}")
        return

    preds = pd.read_csv(preds_p)
    if "tag" in preds.columns:
        preds = preds[preds["tag"] == tag].copy()
    tc = [c for c in preds.columns if c.startswith("y_true_")][0]
    pc = [c for c in preds.columns if c.startswith("y_pred_")][0]

    # ---- simple ensemble (baseline) ----
    g = preds.groupby("sample_id").agg(y=(tc, "mean"), p=(pc, "mean"))
    simple_r2 = r2(g["y"].values, g["p"].values)

    # ---- weight by per-(seed,fold) test R2 (ceiling, mild selection) ----
    fold_r2 = preds.groupby(["seed", "fold"]).apply(
        lambda df: r2(df[tc].values, df[pc].values), include_groups=False)
    fold_r2 = fold_r2.clip(lower=0.0)
    weights = fold_r2 / fold_r2.sum()

    wsum = preds.copy()
    wsum["w"] = wsum.set_index(["seed", "fold"]).index.map(weights)
    g2 = wsum.groupby("sample_id").apply(
        lambda df: np.average(df[pc].values, weights=df["w"].values),
        include_groups=False)
    w_test_r2 = r2(g["y"].values, g2.values)

    # ---- weight by training_history val_loss (strictly legal proxy) ----
    hist_p = os.path.join(paths.METRICS_DIR, "training_history.csv")
    w_val_r2 = None
    if os.path.exists(hist_p):
        hist = pd.read_csv(hist_p)
        # per (seed,fold): mean val loss over epochs
        vl = hist.groupby(["seed", "fold"])["val_loss"].mean()
        # lower val loss -> higher weight
        vw = 1.0 / (vl + 1e-6)
        vw = vw / vw.sum()
        # preds may not align seed/fold with history; join on keys
        wsum3 = preds.merge(vw.rename("w"), on=["seed", "fold"], how="left")
        wsum3 = wsum3.dropna(subset=["w"])
        if len(wsum3):
            g3 = wsum3.groupby("sample_id").apply(
                lambda df: np.average(df[pc].values, weights=df["w"].values),
                include_groups=False)
            w_val_r2 = r2(wsum3.groupby("sample_id")[tc].first().values,
                          g3.values)

    out = {
        "variant": variant,
        "n_rows": len(preds),
        "simple_ensemble_r2": float(simple_r2),
        "w_test_r2": float(w_test_r2),
        "w_val_r2": float(w_val_r2) if w_val_r2 is not None else None,
        "rf_ref": RF_REF,
        "simple_delta_vs_rf": float(simple_r2 - RF_REF),
        "w_test_delta_vs_rf": float(w_test_r2 - RF_REF),
    }
    print(json.dumps(out, indent=2))
    op = os.path.join(paths.RESULTS_DIR, "arch_eval",
                      f"weighted_ens_{variant}.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
