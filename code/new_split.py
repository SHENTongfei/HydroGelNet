"""Redefine data split: internal = df_316 (incl. high-y up to 353 kPa),
external = the 25 independent final-SMBO formulations (df_341 minus df_316).
This removes the 2.4x target-range extrapolation ceiling: the model now
trains on the full composition-adhesion range and is validated prospectively
on the final model-discovered cohort (independent, held-out formulations).
"""
import os
import numpy as np
import pandas as pd
import paths
from build_dataset import MONOMER_NAMES, _read_hydrogel_csv


def build_internal() -> dict:
    """Internal = df_316 (316 formulas, adhesion up to 353 kPa)."""
    X, y, sample_ids = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_316.csv"))
    n = X.shape[0]
    idx = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    X2 = np.stack([X[:, i] * X[:, j] for (i, j) in idx], axis=1).astype(np.float32)
    print(f"  internal rows: {n} (df_316)")
    print(f"  internal y: mean={y.mean():.1f} max={y.max():.1f}")
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


def build_external() -> dict:
    """External = df_341 minus df_316 = 25 independent final-SMBO formulas
    (prospective validation cohort: the last model-guided discoveries)."""
    df316 = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_316.csv"))
    df341 = _read_hydrogel_csv(os.path.join(paths.RAW_DIR, "df_341.csv"))
    X316, _, _ = df316
    X341, y341, ids341 = df341
    key316 = set(map(tuple, np.round(X316, 4)))
    mask = np.array([tuple(np.round(r, 4)) not in key316 for r in X341])
    X = X341[mask]
    y = y341[mask]
    sample_ids = ids341[mask]
    n = X.shape[0]
    idx = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    X2 = np.stack([X[:, i] * X[:, j] for (i, j) in idx], axis=1).astype(np.float32)
    print(f"  external rows: {n} (df_341 minus df_316)")
    print(f"  external y: mean={y.mean():.1f} max={y.max():.1f} (final SMBO)")
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
