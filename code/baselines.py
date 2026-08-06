"""Classical baselines under EXACTLY the same protocol as the deep model.

Fairness rules enforced here:
  * identical outer folds  (trainer.make_outer_splits, same seeds)
  * identical preprocessing (trainer.Preprocessor, fitted per fold)
  * identical metrics       (trainer.compute_metrics)
  * a real inner hyper-parameter search for every baseline, with a budget
    comparable to the one spent on the proposed model. An untuned baseline is
    the fastest way to get a paper rejected.

Usage
-----
    python baselines.py                 # full protocol
    python baselines.py --n-iter 12 --seeds 1 --quick
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.svm import SVC, SVR

import paths
from build_dataset import load_dataset
from trainer import (PRIMARY_METRIC, Preprocessor, compute_metrics,
                     inner_val_split, make_outer_splits)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# Search spaces
# --------------------------------------------------------------------------- #
def baseline_zoo(task_type: str, seed: int) -> dict:
    """Return {name: (estimator, param_distribution)}."""
    if task_type == "regression":
        return {
            "Mean": (DummyRegressor(strategy="mean"), {}),
            "Ridge": (Ridge(random_state=seed),
                      {"alpha": loguniform(1e-3, 1e3)}),
            "ElasticNet": (ElasticNet(random_state=seed, max_iter=5000),
                           {"alpha": loguniform(1e-4, 1e1),
                            "l1_ratio": uniform(0.05, 0.9)}),
            "KNN": (KNeighborsRegressor(),
                    {"n_neighbors": randint(2, 30),
                     "weights": ["uniform", "distance"],
                     "p": [1, 2]}),
            "SVR-RBF": (SVR(kernel="rbf"),
                        {"C": loguniform(1e-2, 1e3),
                         "gamma": loguniform(1e-4, 1e0),
                         "epsilon": uniform(0.01, 0.4)}),
            "RandomForest": (RandomForestRegressor(random_state=seed, n_jobs=-1),
                             {"n_estimators": randint(200, 900),
                              "max_depth": [None, 6, 10, 16, 24],
                              "min_samples_leaf": randint(1, 12),
                              "max_features": ["sqrt", "log2", 0.3, 0.6, 1.0]}),
            "HistGB": (HistGradientBoostingRegressor(random_state=seed),
                       {"learning_rate": loguniform(1e-3, 3e-1),
                        "max_iter": randint(150, 800),
                        "max_leaf_nodes": randint(8, 64),
                        "min_samples_leaf": randint(3, 40),
                        "l2_regularization": loguniform(1e-6, 1e1)}),
            "MLP": (MLPRegressor(random_state=seed, max_iter=1200,
                                 early_stopping=True),
                    {"hidden_layer_sizes": [(64,), (128,), (128, 64),
                                            (256, 128), (96, 96, 48)],
                     "alpha": loguniform(1e-6, 1e-1),
                     "learning_rate_init": loguniform(1e-4, 1e-2)}),
        }
    return {
        "Majority": (DummyClassifier(strategy="most_frequent"), {}),
        "LogReg": (LogisticRegression(max_iter=4000, random_state=seed),
                   {"C": loguniform(1e-3, 1e3),
                    "penalty": ["l2"], "class_weight": [None, "balanced"]}),
        "KNN": (KNeighborsClassifier(),
                {"n_neighbors": randint(2, 30),
                 "weights": ["uniform", "distance"], "p": [1, 2]}),
        "SVC-RBF": (SVC(kernel="rbf", probability=True, random_state=seed),
                    {"C": loguniform(1e-2, 1e3),
                     "gamma": loguniform(1e-4, 1e0),
                     "class_weight": [None, "balanced"]}),
        "RandomForest": (RandomForestClassifier(random_state=seed, n_jobs=-1),
                         {"n_estimators": randint(200, 900),
                          "max_depth": [None, 6, 10, 16, 24],
                          "min_samples_leaf": randint(1, 12),
                          "max_features": ["sqrt", "log2", 0.3, 0.6],
                          "class_weight": [None, "balanced"]}),
        "HistGB": (HistGradientBoostingClassifier(random_state=seed),
                   {"learning_rate": loguniform(1e-3, 3e-1),
                    "max_iter": randint(150, 800),
                    "max_leaf_nodes": randint(8, 64),
                    "min_samples_leaf": randint(3, 40),
                    "l2_regularization": loguniform(1e-6, 1e1)}),
        "MLP": (MLPClassifier(random_state=seed, max_iter=1200,
                              early_stopping=True),
                {"hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
                 "alpha": loguniform(1e-6, 1e-1),
                 "learning_rate_init": loguniform(1e-4, 1e-2)}),
    }


def _fit_one(est, grid, Xtr, ytr, task_type, seed, n_iter, n_inner):
    """Inner random search; returns the refitted best estimator + params."""
    if not grid:
        est.fit(Xtr, ytr)
        return est, {}
    scoring = "r2" if task_type == "regression" else "roc_auc"
    search = RandomizedSearchCV(
        est, grid, n_iter=n_iter, cv=n_inner, scoring=scoring,
        random_state=seed, n_jobs=-1, refit=True, error_score=np.nan)
    search.fit(Xtr, ytr)
    return search.best_estimator_, search.best_params_


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter", type=int, default=40,
                    help="inner random-search budget per baseline per fold")
    ap.add_argument("--seeds", type=int, default=len(paths.SEEDS))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 6/9  CLASSICAL BASELINES (equally tuned)")

    n_iter = 6 if args.quick else args.n_iter
    seeds = paths.SEEDS[:1] if args.quick else paths.SEEDS[:max(1, args.seeds)]

    ds = load_dataset(paths.DATASET_NPZ)
    ds_ext = load_dataset(paths.EXTERNAL_NPZ)
    X, Y = ds["X"], ds["Y"]
    task_type = ds["task_type"]
    tnames = [str(t) for t in ds["target_names"]]
    pm = PRIMARY_METRIC[task_type]
    print(f"  budget: {n_iter} inner candidates x {paths.N_INNER_FOLDS} inner "
          f"folds x {paths.N_OUTER_FOLDS} outer folds x {len(seeds)} seed(s)")

    rows, pred_rows, ext_rows, param_rows = [], [], [], []
    t0 = time.time()

    for seed in seeds:
        zoo = baseline_zoo(task_type, seed)
        splits = make_outer_splits(ds, seed=seed)
        for fold, (tr_idx, te_idx) in enumerate(splits):
            prep = Preprocessor("standard", task_type).fit(X[tr_idx], Y[tr_idx])
            Xtr = prep.transform_x(X[tr_idx])
            Xte = prep.transform_x(X[te_idx])
            Xex = prep.transform_x(ds_ext["X"])

            for name, (est, grid) in zoo.items():
                for t, tname in enumerate(tnames):
                    ytr = Y[tr_idx, t]
                    if task_type == "classification":
                        ytr = (ytr > 0.5).astype(int)
                    try:
                        best, bp = _fit_one(est, grid, Xtr, ytr, task_type,
                                            seed, n_iter, paths.N_INNER_FOLDS)
                    except Exception as exc:               # noqa: BLE001
                        print(f"    [skip] {name}/{tname}: {exc}")
                        continue

                    if task_type == "regression":
                        p_te = best.predict(Xte)
                        p_ex = best.predict(Xex)
                    else:
                        p_te = best.predict_proba(Xte)[:, 1]
                        p_ex = best.predict_proba(Xex)[:, 1]

                    m = compute_metrics(Y[te_idx, t], p_te, task_type)
                    rows.append({"tag": "baseline", "model": name, "seed": seed,
                                 "fold": fold, "target": tname,
                                 "n_test": len(te_idx), **m})
                    me = compute_metrics(ds_ext["Y"][:, t], p_ex, task_type)
                    ext_rows.append({"tag": "external_single", "model": name,
                                     "seed": seed, "fold": fold,
                                     "target": tname,
                                     "n_test": len(ds_ext["Y"]), **me})
                    param_rows.append({"model": name, "seed": seed,
                                       "fold": fold, "target": tname,
                                       "best_params": str(bp)})
                    for j, gi in enumerate(te_idx):
                        pred_rows.append({
                            "model": name, "seed": seed, "fold": fold,
                            "target": tname,
                            "sample_id": str(ds["sample_ids"][gi]),
                            "y_true": float(Y[gi, t]),
                            "y_pred": float(p_te[j])})
            done = pd.DataFrame(rows)
            best_now = (done.groupby("model")[pm].mean().sort_values(
                ascending=False).head(1))
            print(f"    seed={seed} fold={fold} done  "
                  f"(leader: {best_now.index[0]} {pm}={best_now.iloc[0]:.4f})")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(paths.BASELINES_CSV, index=False)
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(paths.PREDS_DIR, "preds_baselines.csv"), index=False)
    pd.DataFrame(param_rows).to_csv(
        os.path.join(paths.TUNING_DIR, "baseline_best_params.csv"), index=False)
    ext = pd.DataFrame(ext_rows)
    ext.to_csv(os.path.join(paths.METRICS_DIR, "baselines_external.csv"),
               index=False)

    print("\n  internal CV summary (mean over folds/seeds/targets):")
    summ = (metrics.groupby("model")[pm]
            .agg(["mean", "std", "count"]).sort_values("mean", ascending=False))
    print(summ.to_string())
    print(f"\n  elapsed {time.time() - t0:.1f}s")
    print(f"Wrote: {paths.BASELINES_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
