"""Statistical evidence for every claim the paper will make.

Implemented
-----------
  * Cluster (group) bootstrap confidence intervals on out-of-fold predictions.
  * Nadeau-Bengio corrected resampled t-test  -- the correct test when the
    same data are reused across CV folds; a naive paired t-test is anti-
    conservative and reviewers know it.
  * Wilcoxon signed-rank test (distribution-free companion).
  * DeLong test for correlated ROC curves (classification only).
  * Label-permutation test against the null "the model learned nothing".
  * Holm and Benjamini-Hochberg multiplicity correction.
  * Cohen's d effect sizes.

Outputs
-------
    results/stats/comparisons.csv     proposed vs each baseline
    results/stats/bootstrap_ci.csv    CI for every model / target
    results/stats/ablation_stats.csv  full vs each ablated variant
    results/stats/permutation.csv     empirical p of the null model
    results/stats/stats_report.md

Usage
-----
    python stats_tests.py
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import paths
from build_dataset import load_dataset
from trainer import PRIMARY_METRIC, compute_metrics


# --------------------------------------------------------------------------- #
# Core tests
# --------------------------------------------------------------------------- #
def corrected_resampled_ttest(diffs: np.ndarray, n_train: int, n_test: int
                              ) -> Tuple[float, float]:
    """Nadeau & Bengio (2003) corrected resampled t-test.

    ``diffs`` holds the per-fold performance differences (model A - model B).
    The variance is inflated by (1/k + n_test/n_train) to account for the
    overlap between training sets across folds.
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    k = len(diffs)
    if k < 2:
        return np.nan, np.nan
    var = diffs.var(ddof=1)
    if var <= 0:
        return (np.inf if diffs.mean() > 0 else -np.inf), 0.0
    correction = (1.0 / k) + (n_test / max(n_train, 1))
    t = diffs.mean() / np.sqrt(correction * var)
    p = 2 * (1 - stats.t.cdf(abs(t), df=k - 1))
    return float(t), float(p)


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / max(len(a) + len(b) - 2, 1))
    return float((a.mean() - b.mean()) / (s + 1e-12))


def group_bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                       groups: np.ndarray, metric: str, task_type: str,
                       n_boot: int = paths.BOOTSTRAP_N, alpha: float = 0.05,
                       seed: int = 0) -> Dict[str, float]:
    """Percentile CI obtained by resampling GROUPS, not rows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx_by_group = {g: np.where(groups == g)[0] for g in uniq}
    point = compute_metrics(y_true, y_pred, task_type).get(metric, np.nan)
    vals = []
    for _ in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_group[g] for g in picked])
        m = compute_metrics(y_true[idx], y_pred[idx], task_type)
        if metric in m:
            vals.append(m[metric])
    if not vals:
        return {"point": point, "lo": np.nan, "hi": np.nan, "se": np.nan}
    vals = np.asarray(vals)
    return {"point": float(point),
            "lo": float(np.percentile(vals, 100 * alpha / 2)),
            "hi": float(np.percentile(vals, 100 * (1 - alpha / 2))),
            "se": float(vals.std(ddof=1))}


def permutation_test(y_true: np.ndarray, y_pred: np.ndarray, metric: str,
                     task_type: str, n_perm: int = 5000, seed: int = 0
                     ) -> Dict[str, float]:
    """Null = predictions carry no information about the labels."""
    rng = np.random.default_rng(seed)
    obs = compute_metrics(y_true, y_pred, task_type).get(metric, np.nan)
    null = []
    for _ in range(n_perm):
        m = compute_metrics(rng.permutation(y_true), y_pred, task_type)
        if metric in m:
            null.append(m[metric])
    null = np.asarray(null)
    p = (1 + int((null >= obs).sum())) / (1 + len(null))
    return {"observed": float(obs), "null_mean": float(null.mean()),
            "null_p95": float(np.percentile(null, 95)),
            "p_value": float(p), "n_perm": len(null)}


def delong_test(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray
                ) -> Tuple[float, float, float, float]:
    """Fast DeLong test for two correlated ROC curves. Returns (auc1, auc2, z, p)."""
    y = (np.asarray(y_true) > 0.5).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return np.nan, np.nan, np.nan, np.nan

    def structural(scores):
        x, yv = scores[pos], scores[neg]
        tx = np.empty(m)
        ty = np.empty(n)
        for i in range(m):
            tx[i] = (np.sum(yv < x[i]) + 0.5 * np.sum(yv == x[i])) / n
        for j in range(n):
            ty[j] = (np.sum(x > yv[j]) + 0.5 * np.sum(x == yv[j])) / m
        return tx, ty, tx.mean()

    v10a, v01a, auc_a = structural(np.asarray(p1, float))
    v10b, v01b, auc_b = structural(np.asarray(p2, float))
    s10 = np.cov(np.vstack([v10a, v10b]))
    s01 = np.cov(np.vstack([v01a, v01b]))
    S = s10 / m + s01 / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return auc_a, auc_b, np.nan, np.nan
    z = (auc_a - auc_b) / np.sqrt(var)
    return float(auc_a), float(auc_b), float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def adjust(pvals: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pvals, float)
    ok = np.isfinite(p)
    holm = np.full_like(p, np.nan)
    fdr = np.full_like(p, np.nan)
    if ok.sum() > 0:
        holm[ok] = multipletests(p[ok], method="holm")[1]
        fdr[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return holm, fdr


def stars(p: float) -> str:
    if not np.isfinite(p):
        return "n.a."
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


# --------------------------------------------------------------------------- #
def main() -> int:
    paths.ensure_dirs()
    paths.banner("STEP 7/9  STATISTICAL TESTING")

    ds = load_dataset(paths.DATASET_NPZ)
    task_type = ds["task_type"]
    pm = PRIMARY_METRIC[task_type]
    n_total = len(ds["Y"])
    n_test = int(round(n_total / paths.N_OUTER_FOLDS))
    n_train = n_total - n_test

    cv = pd.read_csv(paths.CV_OUTER_CSV)
    base = pd.read_csv(paths.BASELINES_CSV)
    cv["model"] = paths.MODEL_NAME
    both = pd.concat([cv, base], ignore_index=True)
    targets = sorted(both["target"].unique())

    # ------------------- 1. proposed vs each baseline ------------------- #
    rows = []
    for target in targets:
        ours = (both[(both["model"] == paths.MODEL_NAME)
                     & (both["target"] == target)]
                .sort_values(["seed", "fold"]))
        for model in sorted(base["model"].unique()):
            theirs = (base[(base["model"] == model)
                           & (base["target"] == target)]
                      .sort_values(["seed", "fold"]))
            k = min(len(ours), len(theirs))
            if k < 2:
                continue
            a = ours[pm].to_numpy()[:k]
            b = theirs[pm].to_numpy()[:k]
            d = a - b
            t, p_t = corrected_resampled_ttest(d, n_train, n_test)
            try:
                _, p_w = stats.wilcoxon(a, b)
            except Exception:                              # noqa: BLE001
                p_w = np.nan
            rows.append({
                "target": target, "reference": model,
                "metric": pm,
                "proposed_mean": float(a.mean()), "proposed_sd": float(a.std()),
                "reference_mean": float(b.mean()), "reference_sd": float(b.std()),
                "delta": float(d.mean()),
                "delta_pct": float(100 * d.mean() / (abs(b.mean()) + 1e-12)),
                "n_folds": k, "t_corrected": t, "p_corrected_t": p_t,
                "p_wilcoxon": p_w, "cohens_d": cohens_d(a, b),
            })
    comp = pd.DataFrame(rows)
    if len(comp):
        comp["p_holm"], comp["p_fdr"] = adjust(comp["p_corrected_t"].tolist())
        comp["significance"] = comp["p_holm"].map(stars)
    comp.to_csv(paths.COMPARISONS_CSV, index=False)
    print(f"  [1/5] {len(comp)} pairwise comparisons -> {paths.COMPARISONS_CSV}")

    # ---------------------- 2. bootstrap CI on OOF ---------------------- #
    ci_rows = []
    oof_path = None
    for f in os.listdir(paths.PREDS_DIR):
        if f.startswith("preds_cv_"):
            oof_path = os.path.join(paths.PREDS_DIR, f)
            break
    if oof_path:
        oof = pd.read_csv(oof_path)
        for target in targets:
            yc, pc = f"y_true_{target}", f"y_pred_{target}"
            if yc not in oof.columns:
                continue
            agg = oof.groupby("sample_id").agg(
                {yc: "mean", pc: "mean", "group": "first"}).reset_index()
            for metric in ([pm, "RMSE", "MAE"] if task_type == "regression"
                           else [pm, "AUPRC", "MCC"]):
                ci = group_bootstrap_ci(agg[yc].to_numpy(), agg[pc].to_numpy(),
                                        agg["group"].to_numpy(), metric,
                                        task_type, seed=paths.PRIMARY_SEED)
                ci_rows.append({"scope": "internal_oof", "model": paths.MODEL_NAME,
                                "target": target, "metric": metric, **ci})
    ext_pred = os.path.join(paths.PREDS_DIR, "preds_external.csv")
    if os.path.exists(ext_pred):
        ext = pd.read_csv(ext_pred)
        for target in targets:
            yc, pc = f"y_true_{target}", f"y_pred_{target}"
            if yc not in ext.columns:
                continue
            for metric in ([pm, "RMSE", "MAE"] if task_type == "regression"
                           else [pm, "AUPRC", "MCC"]):
                ci = group_bootstrap_ci(ext[yc].to_numpy(), ext[pc].to_numpy(),
                                        ext["group"].to_numpy(), metric,
                                        task_type, seed=paths.PRIMARY_SEED)
                ci_rows.append({"scope": "external", "model": paths.MODEL_NAME,
                                "target": target, "metric": metric, **ci})
    ci_df = pd.DataFrame(ci_rows)
    p_ci = os.path.join(paths.STATS_DIR, "bootstrap_ci.csv")
    ci_df.to_csv(p_ci, index=False)
    print(f"  [2/5] {len(ci_df)} bootstrap CIs -> {p_ci}")

    # ------------------------ 3. permutation test ----------------------- #
    perm_rows = []
    if oof_path:
        oof = pd.read_csv(oof_path)
        for target in targets:
            yc, pc = f"y_true_{target}", f"y_pred_{target}"
            if yc not in oof.columns:
                continue
            agg = oof.groupby("sample_id").agg({yc: "mean", pc: "mean"})
            r = permutation_test(agg[yc].to_numpy(), agg[pc].to_numpy(),
                                 pm, task_type, n_perm=5000,
                                 seed=paths.PRIMARY_SEED)
            perm_rows.append({"target": target, "metric": pm, **r,
                              "significance": stars(r["p_value"])})
    perm = pd.DataFrame(perm_rows)
    p_perm = os.path.join(paths.STATS_DIR, "permutation.csv")
    perm.to_csv(p_perm, index=False)
    print(f"  [3/5] permutation tests -> {p_perm}")

    # --------------------------- 4. ablation ---------------------------- #
    abl_rows = []
    if os.path.exists(paths.ABLATION_CSV):
        abl = pd.read_csv(paths.ABLATION_CSV)
        n_ab = int(round(n_total / max(abl["fold"].nunique(), 2)))
        for target in targets:
            sub = abl[abl["target"] == target]
            full = sub[sub["variant"] == "full model"].sort_values(
                ["seed", "fold"])
            if not len(full):
                continue
            for variant in sorted(sub["variant"].unique()):
                if variant == "full model":
                    continue
                var = sub[sub["variant"] == variant].sort_values(["seed", "fold"])
                k = min(len(full), len(var))
                if k < 2:
                    continue
                a, b = full[pm].to_numpy()[:k], var[pm].to_numpy()[:k]
                t, p_t = corrected_resampled_ttest(a - b, n_total - n_ab, n_ab)
                abl_rows.append({
                    "target": target, "variant": variant, "metric": pm,
                    "full_mean": float(a.mean()), "variant_mean": float(b.mean()),
                    "delta": float((a - b).mean()),
                    "delta_pct": float(100 * (a - b).mean()
                                       / (abs(a.mean()) + 1e-12)),
                    "t_corrected": t, "p_corrected_t": p_t,
                    "cohens_d": cohens_d(a, b),
                    "contribution": ("beneficial" if (a - b).mean() > 0
                                     else "neutral/harmful"),
                })
    abl_df = pd.DataFrame(abl_rows)
    if len(abl_df):
        abl_df["p_holm"], abl_df["p_fdr"] = adjust(abl_df["p_corrected_t"].tolist())
        # A zero (or near-zero) delta carries no testable signal: leave the
        # significance column empty for those rows instead of a spurious p.
        abl_df["significance"] = [
            "" if abs(d) < 1e-6 else stars(p)
            for d, p in zip(abl_df["delta"], abl_df["p_holm"])]
    p_abl = os.path.join(paths.STATS_DIR, "ablation_stats.csv")
    abl_df.to_csv(p_abl, index=False)
    print(f"  [4/5] ablation statistics -> {p_abl}")

    # ------------------------ 5. narrative report ----------------------- #
    lines = [f"# Statistical report -- {paths.MODEL_NAME}", "",
             f"Primary metric: **{pm}**. Outer protocol: "
             f"{paths.N_OUTER_FOLDS}-fold grouped CV repeated over "
             f"{cv['seed'].nunique()} seed(s). All p-values from the "
             "Nadeau-Bengio corrected resampled t-test, adjusted with Holm.",
             ""]
    if len(comp):
        lines += ["## Proposed model vs baselines", "",
                  comp[["target", "reference", "proposed_mean",
                        "reference_mean", "delta", "p_corrected_t", "p_holm",
                        "significance"]].round(4).to_markdown(index=False), ""]
        n_sig = int((comp["p_holm"] < 0.05).sum())
        lines.append(f"{n_sig}/{len(comp)} comparisons remain significant "
                     "after Holm correction.")
        lines.append("")
    if len(ci_df):
        lines += ["## Bootstrap confidence intervals (cluster bootstrap)", "",
                  ci_df.round(4).to_markdown(index=False), ""]
    if len(perm):
        lines += ["## Permutation test", "",
                  perm.round(5).to_markdown(index=False), ""]
    if len(abl_df):
        lines += ["## Ablation", "",
                  abl_df[["target", "variant", "full_mean", "variant_mean",
                          "delta", "p_holm", "significance",
                          "contribution"]].round(4).to_markdown(index=False), ""]
    p_md = os.path.join(paths.STATS_DIR, "stats_report.md")
    with open(p_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"  [5/5] narrative report -> {p_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
