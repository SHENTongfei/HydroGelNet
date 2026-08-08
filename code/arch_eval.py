"""Escalation H4: architecture micro-variant FULL-PROTOCOL evaluation.

Runs one architecture variant (A/B/C) through the full protocol
(10 seeds x 5 folds internal CV + external evaluation) and writes all
outputs into an INDEPENDENT subdirectory so multiple variants can run
in parallel without clobbering each other or the main pipeline files.

Outputs (results/arch_eval/<variant>_<ts>/):
  metrics_internal.csv   per-fold internal CV metrics
  metrics_external.csv   external evaluation metrics
  preds_internal.csv     internal predictions
  preds_external.csv     external predictions
  summary.json           compact result for cross-variant comparison
  config_used.json       the exact config that was evaluated

Usage:
    python arch_eval.py arch_A_depth    # variant name -> results/tuning/<name>.json
    python arch_eval.py arch_B_width
    python arch_eval.py arch_C_heads
    python arch_eval.py --smoke arch_A_depth   # 1 seed x 2 folds, fast sanity
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

import paths
from trainer import (PRIMARY_METRIC, TrainConfig, evaluate_external, load_dataset,
                     run_cv)
from build_dataset import load_dataset as _bld_load  # noqa: F401  (alias guard)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant", help="arch_A_depth | arch_B_width | arch_C_heads")
    ap.add_argument("--smoke", action="store_true",
                    help="1 seed x 2 folds sanity run")
    args = ap.parse_args()

    variant = args.variant
    cfg_path = os.path.join(paths.TUNING_DIR, f"{variant}.json")
    if not os.path.exists(cfg_path):
        print(f"[FAIL] variant config not found: {cfg_path}")
        return 2

    with open(cfg_path, encoding="utf-8") as fh:
        cfg = TrainConfig.from_dict(json.load(fh))
    print(f"variant={variant}  d_model={cfg.d_model} n_blocks={cfg.n_blocks} "
          f"n_heads={cfg.n_heads} proj_dim={cfg.proj_dim} "
          f"swa={cfg.use_swa} ema={cfg.use_ema}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(paths.RESULTS_DIR, "arch_eval", f"{variant}_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    seeds = paths.SEEDS[:]
    n_splits = paths.N_OUTER_FOLDS
    if args.smoke:
        seeds = paths.SEEDS[:1]
        n_splits = 2
        print("SMOKE MODE: 1 seed x 2 folds")

    ds = load_dataset(paths.DATASET_NPZ)
    ds_ext = load_dataset(paths.EXTERNAL_NPZ)
    print(f"internal n={len(ds['Y'])}  external n={len(ds_ext['Y'])}  "
          f"task={ds['task_type']}  seeds={len(seeds)}")

    t0 = time.time()
    metrics, preds, fitted = run_cv(cfg, ds, seeds, tag=variant,
                                    verbose=True, n_splits=n_splits)
    ext_metrics, ext_preds = evaluate_external(fitted, cfg, ds_ext)

    metrics.to_csv(os.path.join(out_dir, "metrics_internal.csv"), index=False)
    preds.to_csv(os.path.join(out_dir, "preds_internal.csv"), index=False)
    ext_metrics.to_csv(os.path.join(out_dir, "metrics_external.csv"),
                       index=False)
    ext_preds.to_csv(os.path.join(out_dir, "preds_external.csv"), index=False)

    pm = PRIMARY_METRIC[ds["task_type"]]
    internal_mean = float(metrics[pm].mean())
    internal_std = float(metrics[pm].std())
    ens = ext_metrics[ext_metrics["tag"].str.endswith("ensemble")]
    external_mean = float(ens[pm].mean()) if len(ens) else float("nan")

    summary = {
        "variant": variant,
        "timestamp": ts,
        "smoke": bool(args.smoke),
        "metric": pm,
        "seeds": seeds,
        "n_splits": n_splits,
        "config": cfg.to_dict(),
        "internal_mean": internal_mean,
        "internal_std": internal_std,
        "external_mean": external_mean,
        "seconds": time.time() - t0,
    }
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(os.path.join(out_dir, "config_used.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg.to_dict(), fh, indent=2)

    print(f"\n[arch_eval] {variant} DONE in {(time.time()-t0)/60:.1f} min")
    print(f"  internal {pm} = {internal_mean:.4f} +/- {internal_std:.4f}")
    print(f"  external {pm} = {external_mean:.4f}")
    print(f"  out = {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
