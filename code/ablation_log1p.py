"""Ablation for the FINAL deliverable model (log1p variant), producing a
delta-column CSV that PERF-GATE G5 can read directly.

Reuses tuner.ABLATIONS / FUSION_VARIANTS semantics but:
  - base config = results/tuning/variant_log1p.json (the deliverable)
  - budget = 3 seeds x 3 folds (reliable direction, ~15-25 min per batch)
  - output = results/ablation/ablation_results.csv with R2 + delta columns

Delta is computed vs the "full model" row (mean R2 over folds/seeds).
G5 (escalate.py) looks for a column named delta/drop/gain/diff with
abs max > 1e-4 -> we provide "delta".

Usage: python ablation_log1p.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from trainer import PRIMARY_METRIC, TrainConfig, load_dataset, run_cv
from tuner import ABLATIONS, FUSION_VARIANTS

BASE_CFG = os.path.join(paths.TUNING_DIR, "variant_log1p.json")
N_SEEDS = 3
N_FOLDS = 3


def main() -> int:
    if not os.path.exists(BASE_CFG):
        print(f"[FAIL] deliverable config missing: {BASE_CFG}")
        return 2
    with open(BASE_CFG, encoding="utf-8") as fh:
        best = TrainConfig.from_dict(json.load(fh))
    print(f"ablation base = {os.path.basename(BASE_CFG)}  "
          f"y_transform={best.y_transform} swa={best.use_swa} "
          f"ema={best.use_ema}")

    ds = load_dataset(paths.DATASET_NPZ)
    pm = PRIMARY_METRIC[ds["task_type"]]
    seeds = paths.SEEDS[:N_SEEDS]
    folds = N_FOLDS

    rows = []
    for name, patch in ABLATIONS.items():
        cfg = copy.deepcopy(best)
        for k, v in patch.items():
            setattr(cfg, k, v)
        t0 = time.time()
        metrics, _, _ = run_cv(cfg, ds, seeds, tag=name, verbose=False,
                               n_splits=folds)
        for _, r in metrics.iterrows():
            rows.append({"variant": name, "kind": "component",
                         "seed": r["seed"], "fold": r["fold"],
                         "target": r["target"],
                         "R2": r["R2"], "RMSE": r["RMSE"]})
        print(f"    {name:<32s} {pm}={metrics[pm].mean():+.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    for fus in FUSION_VARIANTS:
        if fus == best.fusion:
            continue
        cfg = copy.deepcopy(best)
        cfg.fusion = fus
        t0 = time.time()
        metrics, _, _ = run_cv(cfg, ds, seeds, tag=f"fusion={fus}",
                               verbose=False, n_splits=folds)
        for _, r in metrics.iterrows():
            rows.append({"variant": f"fusion = {fus}", "kind": "fusion",
                         "seed": r["seed"], "fold": r["fold"],
                         "target": r["target"],
                         "R2": r["R2"], "RMSE": r["RMSE"]})
        print(f"    fusion = {fus:<24s} {pm}={metrics[pm].mean():+.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    # delta column: full-model mean minus variant mean (positive = helps)
    means = df.groupby("variant")["R2"].mean()
    full = means.get("full model", np.nan)
    df["delta"] = df["variant"].map(
        lambda v: (full - means.get(v, np.nan)) if v != "full model" else 0.0)

    os.makedirs(paths.ABLATION_DIR, exist_ok=True)
    out = paths.ABLATION_CSV
    for _i in range(8):
        try:
            df.to_csv(out, index=False)
            break
        except PermissionError:
            time.sleep(5)
    print(f"\nwrote {out}  ({len(df)} rows)")
    print(df.groupby("variant")["R2"].agg(["mean", "count"]).round(4)
          .to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
