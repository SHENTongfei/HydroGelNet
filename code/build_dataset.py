"""Turn raw downloads into the canonical dataset.npz / dataset_external.npz.

Canonical schema (both files identical in structure)
----------------------------------------------------
    X              float32 (n, d)   feature matrix, RAW values (no scaling!)
    Y              float32 (n, t)   target matrix
    groups         <U64    (n,)     grouping key for GroupKFold (subject /
                                    cluster / scaffold / batch). Never leak.
    cond           int32   (n,)     CATEGORICAL condition index, e.g. cell type,
                                    material family, assay platform. Feeds the
                                    learnable embedding + FiLM. Use all-zeros
                                    with a single level if the study has none.
    cond_levels    <U64    (k,)     human-readable name of each condition index
    feature_names  <U128   (d,)
    target_names   <U64    (t,)
    modality_ends  int32   (m,)     cumulative column ends per modality;
                                    e.g. [24, 40] = modality 1 is cols 0:24,
                                    modality 2 is cols 24:40
    modality_names <U64    (m,)
    task_type      <U16    scalar   "regression" | "classification"
    sample_ids     <U64    (n,)

IMPORTANT: scaling / imputation / feature selection are NOT done here.
They happen inside each CV fold (trainer.py) to prevent leakage.

Usage
-----
    python build_dataset.py --demo          # synthetic data, for smoke tests
    python build_dataset.py                 # real build (fill in the TODOs)
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import os
import sys

import numpy as np

import paths


# --------------------------------------------------------------------------- #
# Saving / loading
# --------------------------------------------------------------------------- #
def save_dataset(
    path: str,
    X: np.ndarray,
    Y: np.ndarray,
    groups,
    feature_names,
    target_names,
    cond=None,
    cond_levels=None,
    modality_ends=None,
    modality_names=None,
    task_type: str = "regression",
    sample_ids=None,
) -> None:
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n, d = X.shape
    t = Y.shape[1]

    if cond is None:
        cond = np.zeros(n, dtype=np.int32)
        cond_levels = ["all"]
    cond = np.asarray(cond, dtype=np.int32).reshape(-1)
    if cond_levels is None:
        cond_levels = [f"level_{i}" for i in range(int(cond.max()) + 1)]
    if modality_ends is None:
        modality_ends, modality_names = [d], ["all"]
    if sample_ids is None:
        sample_ids = np.array([f"s{i:05d}" for i in range(n)])

    assert Y.shape[0] == n, "X/Y row mismatch"
    assert len(groups) == n, "groups length mismatch"
    assert len(cond) == n, "cond length mismatch"
    assert cond.min() >= 0, "cond must be non-negative indices"
    assert int(cond.max()) < len(cond_levels), "cond index out of range"
    assert len(feature_names) == d, "feature_names length mismatch"
    assert len(target_names) == t, "target_names length mismatch"
    assert int(modality_ends[-1]) == d, "modality_ends must end at d"

    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(
        path,
        X=X,
        Y=Y,
        groups=np.asarray(groups).astype("<U64"),
        cond=cond,
        cond_levels=np.asarray(cond_levels).astype("<U64"),
        feature_names=np.asarray(feature_names).astype("<U128"),
        target_names=np.asarray(target_names).astype("<U64"),
        modality_ends=np.asarray(modality_ends).astype(np.int32),
        modality_names=np.asarray(modality_names).astype("<U64"),
        task_type=np.asarray(task_type),
        sample_ids=np.asarray(sample_ids).astype("<U64"),
    )
    print(f"  saved {path}")
    print(f"    X={X.shape}  Y={Y.shape}  "
          f"groups={len(set(map(str, groups)))} unique  "
          f"cond_levels={len(cond_levels)}  task={task_type}")


def load_dataset(path: str) -> dict:
    """Load a canonical npz into a plain dict."""
    z = np.load(path, allow_pickle=False)
    out = {k: z[k] for k in z.files}
    out["task_type"] = str(out["task_type"])
    return out


def split_modalities(X: np.ndarray, modality_ends) -> tuple:
    """Return (X1, X2) using the first modality boundary. X2 may be width 0."""
    ends = [int(e) for e in modality_ends]
    if len(ends) == 1:
        return X, np.zeros((X.shape[0], 0), dtype=np.float32)
    return X[:, :ends[0]], X[:, ends[0]:ends[1]]


# --------------------------------------------------------------------------- #
# Demo generator (only for pipeline smoke tests -- NEVER for the paper)
# --------------------------------------------------------------------------- #
def make_demo(n: int, seed: int, shift: float = 0.0) -> dict:
    """Synthetic two-modality, two-task problem with genuine nonlinearity.

    A covariate shift can be injected to emulate an independent cohort.
    """
    rng = np.random.default_rng(seed)
    d1, d2, k_cond = 24, 16, 3
    m1 = rng.normal(shift * 0.35, 1.0, size=(n, d1))
    m2 = rng.normal(0.0, 1.0, size=(n, d2)) + shift * 0.2
    cond = rng.integers(0, k_cond, size=n)

    w1 = rng.normal(0, 1, d1) * (np.arange(d1) < 8)      # 8 informative
    w2 = rng.normal(0, 1, d2) * (np.arange(d2) < 5)      # 5 informative
    lin = m1 @ w1 + m2 @ w2
    inter = 0.6 * m1[:, 0] * m2[:, 1] + 0.4 * np.tanh(m1[:, 2] * 1.5)
    cond_eff = np.array([0.0, 1.2, -0.9])[cond] * m1[:, 3]

    y1 = lin + inter + cond_eff + rng.normal(0, 0.8, n)
    y2 = 0.7 * lin - 0.5 * inter + 0.4 * cond_eff + rng.normal(0, 0.9, n)

    X = np.hstack([m1, m2]).astype(np.float32)
    Y = np.column_stack([y1, y2]).astype(np.float32)
    groups = np.array([f"g{i % max(6, n // 8):03d}" for i in range(n)])
    fnames = ([f"mod1_f{i:02d}" for i in range(d1)]
              + [f"mod2_f{i:02d}" for i in range(d2)])
    return dict(
        X=X, Y=Y, groups=groups, cond=cond,
        cond_levels=[f"context_{i}" for i in range(k_cond)],
        feature_names=fnames, target_names=["target_A", "target_B"],
        modality_ends=[d1, d1 + d2], modality_names=["mod1", "mod2"],
        task_type="regression",
        sample_ids=np.array([f"{'ext' if shift else 'int'}{i:05d}"
                             for i in range(n)]),
    )


# --------------------------------------------------------------------------- #
# Real builders -- HydroGelNet: composition -> adhesion strength
# --------------------------------------------------------------------------- #
# Data: sheng-hu/hydrogels (MIT, Nature 2025).
#   df_180.csv : round-1 baseline, 180 formulas (train region, low perf)
#   df_341.csv : full dataset, 341 formulas
#   EXTERNAL   : rows in df_341 NOT in df_180 = 161 SMBO-guided high-perf
#                formulas (time-extrapolation test region).
# Features    : 6 monomer molar fractions on the simplex (sum ~ 1).
# Target      : Glass (kPa)_max -- underwater adhesion strength.
# NOTE: no scaling here; per-fold scaling happens in trainer.py to avoid leakage.
MONOMER_NAMES = [
    "Nucleophilic-HEA",
    "Hydrophobic-BA",
    "Acidic-CBEA",
    "Cationic-ATAC",
    "Aromatic-PEA",
    "Amide-AAm",
]


def _read_hydrogel_csv(path: str) -> tuple:
    """Read a hydrogel CSV, return (X, y, sample_ids)."""
    import pandas as pd
    df = pd.read_csv(path)
    # columns: index, 6 monomers, Glass (kPa)_max
    X = df[MONOMER_NAMES].to_numpy(dtype=np.float32)
    y = df["Glass (kPa)_max"].to_numpy(dtype=np.float32).reshape(-1, 1)
    sample_ids = np.array([f"hg{i:04d}" for i in range(len(df))])
    # simplex sanity: row sums close to 1
    row_sum = X.sum(axis=1)
    assert np.allclose(row_sum, 1.0, atol=0.02), "composition rows must sum ~1"
    return X, y, sample_ids


def build_internal() -> dict:
    """Internal = df_180 (round-1 baseline, 180 formulas)."""
    import os
    X, y, sample_ids = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_180.csv"))
    n = X.shape[0]

    # Modality 1: raw 6 monomer fractions (composition on the simplex).
    # Modality 2: pairwise interaction features (6 choose 2 = 15 products),
    #             capturing monomer synergy explicitly as a second modality.
    # This is a handcrafted polynomial expansion that the model can
    # additionally refine with attention.
    idx = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    X2 = np.stack([X[:, i] * X[:, j] for (i, j) in idx], axis=1).astype(np.float32)

    return dict(
        X=np.hstack([X, X2]).astype(np.float32),
        Y=y,
        groups=sample_ids,          # each formula unique -> per-sample grouping
        cond=np.zeros(n, dtype=np.int32),
        cond_levels=["all"],
        feature_names=np.array(MONOMER_NAMES + [f"pair_{i}{j}" for (i, j) in idx]),
        target_names=["glass_adhesion_kpa"],
        modality_ends=[6, 6 + len(idx)],
        modality_names=["monomer_fractions", "pairwise_synergy"],
        task_type="regression",
        sample_ids=sample_ids,
    )


def build_external() -> dict:
    """External = df_341 minus df_180 = 161 SMBO-guided high-performance
    formulas (time-extrapolation region: trained on low-performance space,
    evaluated on high-performance space discovered by the Nature SMBO loop)."""
    import os
    df180 = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_180.csv"))
    df341 = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_341.csv"))
    X180, _, _ = df180
    X341, y341, ids341 = df341

    # find rows in df_341 not present in df_180 (by exact composition match)
    import numpy as np
    key180 = set(map(tuple, np.round(X180, 4)))
    mask = np.array([tuple(np.round(r, 4)) not in key180 for r in X341])
    X = X341[mask]
    y = y341[mask]
    sample_ids = ids341[mask]
    n = X.shape[0]

    idx = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    X2 = np.stack([X[:, i] * X[:, j] for (i, j) in idx], axis=1).astype(np.float32)

    print(f"  external rows: {n} (df_341 minus df_180)")
    print(f"  external y: mean={y.mean():.1f} max={y.max():.1f} (high-perf region)")

    return dict(
        X=np.hstack([X, X2]).astype(np.float32),
        Y=y,
        groups=sample_ids,
        cond=np.zeros(n, dtype=np.int32),
        cond_levels=["all"],
        feature_names=np.array(MONOMER_NAMES + [f"pair_{i}{j}" for (i, j) in idx]),
        target_names=["glass_adhesion_kpa"],
        modality_ends=[6, 6 + len(idx)],
        modality_names=["monomer_fractions", "pairwise_synergy"],
        task_type="regression",
        sample_ids=sample_ids,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="generate synthetic data to smoke-test the pipeline")
    ap.add_argument("--n-internal", type=int, default=420)
    ap.add_argument("--n-external", type=int, default=160)
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 2/9  BUILD DATASET")

    if args.demo:
        print("!! DEMO MODE -- synthetic data. Never report these numbers. !!")
        internal = make_demo(args.n_internal, seed=paths.PRIMARY_SEED, shift=0.0)
        external = make_demo(args.n_external, seed=999, shift=1.0)
    else:
        internal = build_internal()
        external = build_external()

    save_dataset(paths.DATASET_NPZ, **internal)
    save_dataset(paths.EXTERNAL_NPZ, **external)

    a = load_dataset(paths.DATASET_NPZ)
    b = load_dataset(paths.EXTERNAL_NPZ)
    same_feats = list(a["feature_names"]) == list(b["feature_names"])
    same_targs = list(a["target_names"]) == list(b["target_names"])
    print(f"\n  feature space identical : {same_feats}")
    print(f"  target space identical  : {same_targs}")
    if not (same_feats and same_targs):
        print("  ERROR: internal and external must share feature/target space.")
        return 1
    print("\nDataset build finished.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
