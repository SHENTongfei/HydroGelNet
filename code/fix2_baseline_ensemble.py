"""FIX-2 ensemble-parity audit for external baselines.

SIMPLEX external numbers come from a 25-model ensemble (5 seeds x 5 folds),
while baseline external numbers were single-fit means. This script recomputes
each baseline's external metrics from ENSEMBLED predictions (mean across the
25 per-fit predictions per sample) so the comparison is apples-to-apples.

Reads : results/metrics/baselines_external_preds.csv  (written by baselines.py)
Writes: results/metrics/baselines_external_ensemble.csv
Prints : per-model ensemble R2/Spearman vs SIMPLEX ensemble R2 (0.71) and the
         previous single-fit means, plus a verdict on whether the SIMPLEX
         transfer advantage survives baseline ensembling.

Usage: python fix2_baseline_ensemble.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import os

import numpy as np
import pandas as pd

import paths


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def _spearman(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    ry = pd.Series(y).rank().values
    rp = pd.Series(p).rank().values
    d = ry - rp
    n = len(y)
    return float(1.0 - 6.0 * np.sum(d ** 2) / (n * (n ** 2 - 1.0)))


def main() -> int:
    paths.ensure_dirs()
    src = os.path.join(paths.METRICS_DIR, "baselines_external_preds.csv")
    if not os.path.exists(src):
        print(f"FIX-2: {src} not found. Run baselines.py first.")
        return 1
    df = pd.read_csv(src)
    print(f"loaded {len(df)} rows from {os.path.basename(src)}")

    rows = []
    for model, g in df.groupby("model"):
        # ensemble prediction = mean across (seed, fold) fits per sample
        ens = g.groupby("sample_id")["y_pred"].mean().sort_index()
        y = g.groupby("sample_id")["y_true"].first().sort_index()
        single_mean = g.groupby("sample_id")["y_pred"].mean()
        # single-fit mean R2: mean of per-fit metrics is already in
        # baselines_external.csv; here we also report mean-of-per-fit R2
        per_fit = g.groupby(["seed", "fold"]).apply(
            lambda gg: _r2(gg["y_true"].values, gg["y_pred"].values),
            include_groups=False)
        rows.append({
            "model": model,
            "n_fits": g["seed"].nunique() * g["fold"].nunique(),
            "R2_ensemble": _r2(y.values, ens.values),
            "Spearman_ensemble": _spearman(y.values, ens.values),
            "R2_singlefit_mean": float(per_fit.mean()),
        })
    out = pd.DataFrame(rows).sort_values("R2_ensemble", ascending=False)
    out.to_csv(os.path.join(paths.METRICS_DIR,
                            "baselines_external_ensemble.csv"), index=False)

    print("\n  external ensemble parity (FIX-2):")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  SIMPLEX ensemble external R2 = 0.71 (reference)")
    best_ens = out["R2_ensemble"].max()
    verdict = ("SIMPLEX SURVIVES" if 0.71015 > best_ens
               else "SIMPLEX LOST to ensembled baseline")
    print(f"  strongest ensembled baseline R2 = {best_ens:.4f} -> {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
