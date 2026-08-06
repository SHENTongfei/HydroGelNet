"""Hyper-parameter search + ablation study + automatic trick pruning.

Three stages
------------
1. COARSE random search over the full space (default 60 candidates).
2. FINE local search around the coarse winner (default 40 candidates).
3. ABLATION: switch every trick off one at a time, plus fusion variants.
   Any trick whose removal does NOT hurt is switched off in the final config.
   "Test it, keep it if it works, drop it if it does not" is automated here,
   so the paper never claims a component that carries no weight.

Search folds are reduced (3 outer folds, 1 seed) to keep the budget sane;
the winning configuration is later re-evaluated with the full protocol in
trainer.py. The search never touches the external cohort.

Usage
-----
    python tuner.py
    python tuner.py --coarse 60 --fine 40 --search-folds 3
    python tuner.py --quick            # smoke test only
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import copy
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd

import paths
from build_dataset import load_dataset
from trainer import PRIMARY_METRIC, TrainConfig, run_cv


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
def sample_config(rng: np.random.Generator, base: TrainConfig | None = None,
                  local: bool = False) -> TrainConfig:
    """Draw one configuration. ``local=True`` perturbs around ``base``."""
    cfg = copy.deepcopy(base) if base is not None else TrainConfig()

    def pick(options):
        return options[int(rng.integers(len(options)))]

    def jitter(value, lo, hi, frac=0.35, integer=False):
        span = max(abs(value) * frac, (hi - lo) * 0.06)
        v = float(np.clip(rng.normal(value, span), lo, hi))
        return int(round(v)) if integer else v

    if local:
        cfg.d_model = int(np.clip(jitter(cfg.d_model, 32, 256, integer=True),
                                  32, 256) // 8 * 8)
        cfg.n_blocks = int(np.clip(cfg.n_blocks + rng.integers(-1, 2), 1, 5))
        cfg.dropout = jitter(cfg.dropout, 0.0, 0.5)
        cfg.lr = float(np.clip(cfg.lr * np.exp(rng.normal(0, 0.35)), 1e-4, 1e-2))
        cfg.weight_decay = float(np.clip(
            cfg.weight_decay * np.exp(rng.normal(0, 0.7)), 1e-7, 1e-1))
        cfg.mixup_alpha = jitter(cfg.mixup_alpha, 0.0, 1.0)
        cfg.attn_entropy_w = float(np.clip(
            cfg.attn_entropy_w * np.exp(rng.normal(0, 0.9)), 0.0, 1e-1))
        cfg.batch_size = pick([max(16, cfg.batch_size // 2), cfg.batch_size,
                               min(1024 if paths.DEVICE == "cuda" else 256,
                                   cfg.batch_size * 2)])
        cfg.constraint_w = jitter(cfg.constraint_w, 0.0, 0.5)
        cfg.contrastive_epochs = int(np.clip(
            jitter(cfg.contrastive_epochs, 0, 120, integer=True), 0, 120))
        return cfg

    cfg.d_model = pick([48, 64, 96, 128, 160, 192])
    cfg.n_blocks = pick([1, 2, 3, 4])
    cfg.n_heads = pick([2, 4, 8])
    cfg.n_tokens1 = pick([2, 4, 6, 8, 12])
    cfg.n_tokens2 = pick([2, 3, 4, 6])
    cfg.dropout = float(rng.uniform(0.0, 0.40))
    cfg.fusion = pick(["concat", "film", "cross", "gated"])
    cfg.lr = float(np.exp(rng.uniform(np.log(3e-4), np.log(6e-3))))
    cfg.weight_decay = float(np.exp(rng.uniform(np.log(1e-6), np.log(1e-2))))
    cfg.batch_size = pick([16, 32, 64, 128] + ([256, 512] if paths.DEVICE == "cuda" else []))
    cfg.scaler = pick(["standard", "quantile"])
    cfg.y_transform = "standard"   # log1p hurts internal fit on this dataset
    cfg.mixup_alpha = float(rng.uniform(0.05, 0.8))
    cfg.attn_entropy_w = float(np.exp(rng.uniform(np.log(1e-5), np.log(1e-1))))
    cfg.contrastive_epochs = pick([10, 20, 40, 60, 80])
    cfg.contrastive_temp = float(rng.uniform(0.05, 0.5))
    cfg.contrastive_bins = pick([3, 4, 5, 6])
    cfg.constraint_w = float(rng.uniform(0.0, 0.3))
    cfg.swa_start_frac = float(rng.uniform(0.4, 0.8))
    cfg.proj_dim = pick([16, 32, 64])
    # ---- extra tricks (all ablation switches) ----------------------------- #
    cfg.use_modality_gate = bool(rng.random() < 0.5)
    cfg.gate_sparsity_w = (float(rng.uniform(0.0, 0.05))
                           if cfg.use_modality_gate else 0.0)
    cfg.use_transformer = False   # ResBlock empirically superior on 180-sample data
    cfg.use_sam = bool(rng.random() < 0.5)
    cfg.sam_rho = float(rng.uniform(0.02, 0.1))
    cfg.use_ema = bool(rng.random() < 0.5)
    cfg.ema_decay = float(rng.uniform(0.99, 0.9999))
    cfg.use_rdrop = bool(rng.random() < 0.3)
    cfg.rdrop_w = float(rng.uniform(0.1, 1.0))
    cfg.feature_noise = float(rng.uniform(0.0, 0.15))
    cfg.label_smoothing = float(rng.uniform(0.0, 0.15))
    # self-supervised pre-training (MFM = masked-feature modelling)
    cfg.use_mfm = bool(rng.random() < 0.5)
    cfg.mfm_epochs = pick([10, 20, 30, 40])
    cfg.mfm_mask_frac = float(rng.uniform(0.15, 0.4))
    # prediction-time MC-Dropout (cheap uncertainty; usually a small boost)
    cfg.mc_samples = pick([0, 0, 5, 10])
    return cfg


def score_config(cfg: TrainConfig, ds: dict, seeds: List[int],
                 folds: int) -> float:
    """Mean primary metric over the reduced search protocol."""
    pm = PRIMARY_METRIC[ds["task_type"]]
    metrics, _, _ = run_cv(cfg, ds, seeds, tag="search", verbose=False,
                           n_splits=folds)
    if pm not in metrics.columns or metrics[pm].isna().all():
        return -1e9
    return float(metrics[pm].mean())


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
ABLATIONS: Dict[str, Dict] = {
    "full model": {},
    "w/o multimodal fusion": {"use_modality2": False},
    "w/o sparse attention": {"use_attention": False},
    "w/o attention sparsity reg.": {"attn_entropy_w": 0.0},
    "w/o FiLM conditioning": {"use_film": False},
    "w/o task-specific gating": {"use_task_gate": False},
    "w/o residual blocks": {"use_residual": False},
    "w/o contrastive pre-training": {"use_contrastive": False,
                                     "contrastive_epochs": 0},
    "w/o Mixup": {"use_mixup": False, "mixup_alpha": 0.0},
    "w/o SWA": {"use_swa": False},
    "w/o uncertainty weighting": {"use_uncertainty_weighting": False},
    "w/o domain constraint": {"use_domain_constraint": False,
                              "constraint_w": 0.0},
    "w/o modality gate": {"use_modality_gate": False, "gate_sparsity_w": 0.0},
    "w/o transformer block": {"use_transformer": False},
    "w/o SAM": {"use_sam": False},
    "w/o EMA": {"use_ema": False},
    "w/o R-Drop": {"use_rdrop": False, "rdrop_w": 0.0},
    "w/o feature noise": {"feature_noise": 0.0},
    "w/o MFM pre-training": {"use_mfm": False, "mfm_epochs": 0},
    "w/o MC-Dropout": {"mc_samples": 0},
    "w/o pretrained transfer": {"pretrained_path": ""},
}
FUSION_VARIANTS = ["concat", "film", "cross", "gated"]

# Removing one of these must not be auto-pruned even if it looks neutral,
# because they define the model family. They are still reported.
STRUCTURAL = {"w/o residual blocks"}


def run_ablation(best: TrainConfig, ds: dict, seeds: List[int], folds: int
                 ) -> pd.DataFrame:
    pm = PRIMARY_METRIC[ds["task_type"]]
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
                         "target": r["target"], **{
                             c: r[c] for c in metrics.columns
                             if c not in ("tag", "model", "seed", "fold",
                                          "target", "n_test", "epochs",
                                          "swa_used")}})
        print(f"    {name:<32s} {pm}={metrics[pm].mean():+.4f}  "
              f"({time.time() - t0:.0f}s)")

    for fus in FUSION_VARIANTS:
        if fus == best.fusion:
            continue
        cfg = copy.deepcopy(best)
        cfg.fusion = fus
        metrics, _, _ = run_cv(cfg, ds, seeds, tag=f"fusion={fus}",
                               verbose=False, n_splits=folds)
        for _, r in metrics.iterrows():
            rows.append({"variant": f"fusion = {fus}", "kind": "fusion",
                         "seed": r["seed"], "fold": r["fold"],
                         "target": r["target"], **{
                             c: r[c] for c in metrics.columns
                             if c not in ("tag", "model", "seed", "fold",
                                          "target", "n_test", "epochs",
                                          "swa_used")}})
        print(f"    fusion = {fus:<24s} {pm}={metrics[pm].mean():+.4f}")
    return pd.DataFrame(rows)


def prune_config(best: TrainConfig, abl: pd.DataFrame, pm: str,
                 tol: float = 0.002) -> tuple[TrainConfig, List[str]]:
    """Disable every component whose removal did not hurt beyond ``tol``."""
    means = abl[abl["kind"] == "component"].groupby("variant")[pm].mean()
    if "full model" not in means:
        return best, []
    full = means["full model"]
    pruned, notes = copy.deepcopy(best), []
    for name, patch in ABLATIONS.items():
        if name == "full model" or name in STRUCTURAL or name not in means:
            continue
        delta = full - means[name]          # >0 means the component helps
        if delta <= tol:
            for k, v in patch.items():
                setattr(pruned, k, v)
            notes.append(f"{name}: delta {pm} = {delta:+.4f} -> component "
                         "removed from the final model")
    return pruned, notes


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", type=int, default=60)
    ap.add_argument("--fine", type=int, default=40)
    ap.add_argument("--search-folds", type=int, default=3)
    ap.add_argument("--search-seeds", type=int, default=1)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-search", action="store_true",
                    help="reuse best_config.json and only run the ablation")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 4/9  HYPER-PARAMETER SEARCH + ABLATION")

    ds = load_dataset(paths.DATASET_NPZ)
    pm = PRIMARY_METRIC[ds["task_type"]]
    seeds = paths.SEEDS[:max(1, args.search_seeds)]
    folds = args.search_folds
    n_coarse, n_fine = args.coarse, args.fine
    if args.quick:
        n_coarse, n_fine, folds = 3, 2, 2
        print("  QUICK MODE: tiny budget, results are not publishable.")

    rng = np.random.default_rng(paths.PRIMARY_SEED)
    log: List[dict] = []
    best_cfg, best_score = TrainConfig(), -np.inf

    if args.skip_search and os.path.exists(paths.BEST_CONFIG_JSON):
        with open(paths.BEST_CONFIG_JSON, "r", encoding="utf-8") as fh:
            best_cfg = TrainConfig.from_dict(json.load(fh))
        print(f"  reusing {paths.BEST_CONFIG_JSON}")
    else:
        if args.quick:
            base = TrainConfig()
            base.max_epochs, base.patience, base.contrastive_epochs = 25, 6, 4
        else:
            base = None

        print(f"\n  -- coarse search ({n_coarse} candidates) --")
        for i in range(n_coarse):
            cfg = sample_config(rng)
            if args.quick:
                cfg.max_epochs, cfg.patience, cfg.contrastive_epochs = 25, 6, 4
            t0 = time.time()
            score = score_config(cfg, ds, seeds, folds)
            log.append({"phase": "coarse", "iter": i, "score": score,
                        "seconds": round(time.time() - t0, 1),
                        **cfg.to_dict()})
            if score > best_score:
                best_score, best_cfg = score, cfg
                print(f"    [{i:3d}] {pm}={score:+.4f}  <-- new best")
            elif i % 10 == 0:
                print(f"    [{i:3d}] {pm}={score:+.4f}")

        print(f"\n  -- fine search ({n_fine} candidates around the winner) --")
        for i in range(n_fine):
            cfg = sample_config(rng, base=best_cfg, local=True)
            t0 = time.time()
            score = score_config(cfg, ds, seeds, folds)
            log.append({"phase": "fine", "iter": i, "score": score,
                        "seconds": round(time.time() - t0, 1),
                        **cfg.to_dict()})
            if score > best_score:
                best_score, best_cfg = score, cfg
                print(f"    [{i:3d}] {pm}={score:+.4f}  <-- new best")

        pd.DataFrame(log).to_csv(paths.SEARCH_LOG_CSV, index=False)
        print(f"\n  search finished. best {pm} = {best_score:+.4f}")
        print(f"  log: {paths.SEARCH_LOG_CSV}")

    # ---------------------------- ablation ---------------------------- #
    print(f"\n  -- ablation ({len(ABLATIONS)} components "
          f"+ {len(FUSION_VARIANTS) - 1} fusion variants) --")
    abl = run_ablation(best_cfg, ds, seeds, folds)
    abl.to_csv(paths.ABLATION_CSV, index=False)

    final_cfg, notes = prune_config(best_cfg, abl, pm)
    print("\n  -- automatic component pruning --")
    if notes:
        for n in notes:
            print(f"    {n}")
    else:
        print("    every component earned its place; nothing removed.")

    with open(paths.BEST_CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(final_cfg.to_dict(), fh, indent=2)
    with open(os.path.join(paths.TUNING_DIR, "pruning_notes.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(notes) if notes else "No component was removed.\n")

    print(f"\nWrote: {paths.BEST_CONFIG_JSON}\n       {paths.ABLATION_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
