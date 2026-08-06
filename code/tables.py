"""Publication tables: CSV + LaTeX + Markdown, ten tables.

  Table 1   Dataset and cohort summary
  Table 2   Data sources with verified links and licences
  Table 3   Search space and selected hyper-parameters
  Table 4   Internal cross-validation performance (all metrics)
  Table 5   Comparison against tuned baselines with corrected p-values
  Table 6   External cohort validation
  Table 7   Ablation study
  Table 8   Top-ranked features and candidate markers
  Table 9   Performance stratified by condition
  Table 10  Reproducibility settings

Each table is written three times: ``TableN_slug.csv`` for reuse,
``TableN_slug.tex`` (booktabs) for the manuscript, and ``TableN_slug.md``
for quick reading.

Usage
-----
    python tables.py
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import json
import os
import platform
import sys
from typing import Optional

import numpy as np
import pandas as pd

import paths
from build_dataset import load_dataset


def _csv(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if len(df) else None
    except Exception:                                      # noqa: BLE001
        pass
    return None


def emit(df: pd.DataFrame, number: int, slug: str, caption: str) -> None:
    """Write one table in three formats."""
    if df is None or not len(df):
        print(f"  [skip] Table {number} ({slug}): no data")
        return
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(4)
    stem = os.path.join(paths.TABLES_DIR, f"Table{number}_{slug}")
    df.to_csv(stem + ".csv", index=False)
    with open(stem + ".md", "w", encoding="utf-8") as fh:
        fh.write(f"**Table {number}.** {caption}\n\n")
        fh.write(df.to_markdown(index=False) + "\n")
    try:
        latex = df.to_latex(index=False, escape=True, longtable=False,
                            caption=caption, label=f"tab:{slug}")
    except TypeError:
        latex = df.to_latex(index=False, escape=True)
    with open(stem + ".tex", "w", encoding="utf-8") as fh:
        fh.write(latex)
    print(f"  Table {number:>2d}  {slug:<26s} {len(df):>4d} rows -> {stem}.csv")


def _agg(df: pd.DataFrame, by, cols) -> pd.DataFrame:
    cols = [c for c in cols if c in df.columns]
    g = df.groupby(by)[cols].agg(["mean", "std"])
    out = pd.DataFrame(index=g.index)
    for c in cols:
        out[c] = [f"{m:.3f} ± {s:.3f}" if np.isfinite(m) else "-"
                  for m, s in zip(g[(c, "mean")], g[(c, "std")])]
    return out.reset_index()


def main() -> int:
    paths.ensure_dirs()
    paths.banner("TABLES")

    ds = load_dataset(paths.DATASET_NPZ)
    ext = (load_dataset(paths.EXTERNAL_NPZ)
           if os.path.exists(paths.EXTERNAL_NPZ) else None)
    task = ds["task_type"]
    pm = "R2" if task == "regression" else "AUROC"
    reg_cols = ["R2", "RMSE", "MAE", "NRMSE", "PearsonR", "SpearmanRho", "CCC"]
    clf_cols = ["AUROC", "AUPRC", "Accuracy", "BalancedAcc", "F1", "MCC"]
    metric_cols = reg_cols if task == "regression" else clf_cols

    # ---------------------------- Table 1 ---------------------------- #
    rows = []
    for name, d in [("Internal", ds), ("External", ext)]:
        if d is None:
            continue
        rows.append({
            "Cohort": name,
            "Samples": len(d["Y"]),
            "Features": d["X"].shape[1],
            "Targets": d["Y"].shape[1],
            "Groups": len(set(map(str, d["groups"]))),
            "Conditions": len(d["cond_levels"]),
            "Missing (%)": round(float(np.isnan(d["X"]).mean() * 100), 2),
            "Modalities": ", ".join(str(m) for m in d["modality_names"]),
        })
    emit(pd.DataFrame(rows), 1, "dataset_summary",
         "Summary of the internal and external cohorts.")

    # ---------------------------- Table 2 ---------------------------- #
    src = None
    if os.path.exists(paths.DATA_SOURCES_MD):
        try:
            tables = pd.read_html(paths.DATA_SOURCES_MD)     # noqa: PD901
            src = tables[0] if tables else None
        except Exception:                                  # noqa: BLE001
            lines = open(paths.DATA_SOURCES_MD, encoding="utf-8").read().split("\n")
            rows = [l for l in lines if l.startswith("|") and "---" not in l]
            if len(rows) > 1:
                header = [h.strip() for h in rows[0].strip("|").split("|")]
                body = [[c.strip() for c in r.strip("|").split("|")]
                        for r in rows[1:]]
                body = [b for b in body if len(b) == len(header)]
                src = pd.DataFrame(body, columns=header)
    emit(src, 2, "data_sources",
         "Public data sources, licences and verified download links.")

    # ---------------------------- Table 3 ---------------------------- #
    if os.path.exists(paths.BEST_CONFIG_JSON):
        with open(paths.BEST_CONFIG_JSON, encoding="utf-8") as fh:
            best = json.load(fh)
        search = _csv(paths.SEARCH_LOG_CSV)
        rows = []
        for k, v in best.items():
            rng = "-"
            if search is not None and k in search.columns:
                col = search[k]
                if pd.api.types.is_numeric_dtype(col):
                    rng = f"[{col.min():.4g}, {col.max():.4g}]"
                else:
                    rng = ", ".join(sorted(set(map(str, col.unique())))[:5])
            rows.append({"Hyper-parameter": k, "Search range": rng,
                         "Selected value": v})
        emit(pd.DataFrame(rows), 3, "hyperparameters",
             "Search space and finally selected hyper-parameter values.")

    # ---------------------------- Table 4 ---------------------------- #
    cv = _csv(paths.CV_OUTER_CSV)
    if cv is not None:
        t4 = _agg(cv, "target", metric_cols)
        t4.insert(0, "Model", paths.MODEL_NAME)
        emit(t4, 4, "internal_cv",
             f"Internal {paths.N_OUTER_FOLDS}-fold grouped cross-validation "
             "performance (mean ± SD over folds and seeds).")

    # ---------------------------- Table 5 ---------------------------- #
    comp = _csv(paths.COMPARISONS_CSV)
    if comp is not None:
        cols = ["target", "reference", "reference_mean", "proposed_mean",
                "delta", "delta_pct", "cohens_d", "p_corrected_t", "p_holm",
                "significance"]
        cols = [c for c in cols if c in comp.columns]
        t5 = comp[cols].rename(columns={
            "target": "Target", "reference": "Baseline",
            "reference_mean": f"Baseline {pm}",
            "proposed_mean": f"{paths.MODEL_NAME} {pm}",
            "delta": "Delta", "delta_pct": "Delta (%)",
            "cohens_d": "Cohen's d", "p_corrected_t": "p (corrected t)",
            "p_holm": "p (Holm)", "significance": "Sig."})
        emit(t5, 5, "baseline_comparison",
             "Comparison against equally tuned baselines. p-values from the "
             "Nadeau-Bengio corrected resampled t-test, Holm adjusted.")

    # ---------------------------- Table 6 ---------------------------- #
    extm = _csv(paths.EXTERNAL_CSV)
    if extm is not None:
        ens = extm[extm["tag"].str.endswith("ensemble")]
        t6 = _agg(extm[extm["tag"].str.endswith("single")], "target",
                  metric_cols)
        t6.columns = ["target"] + [f"{c} (per-fold)" for c in t6.columns[1:]]
        e = ens.groupby("target")[
            [c for c in metric_cols if c in ens.columns]].mean().reset_index()
        e.columns = ["target"] + [f"{c} (ensemble)" for c in e.columns[1:]]
        t6 = t6.merge(e, on="target", how="outer")
        ci = _csv(os.path.join(paths.STATS_DIR, "bootstrap_ci.csv"))
        if ci is not None:
            c = ci[(ci["scope"] == "external") & (ci["metric"] == pm)]
            if len(c):
                c = c.assign(**{f"{pm} 95% CI": [
                    f"[{lo:.3f}, {hi:.3f}]" for lo, hi in zip(c["lo"], c["hi"])]})
                t6 = t6.merge(c[["target", f"{pm} 95% CI"]], on="target",
                              how="left")
        emit(t6, 6, "external_validation",
             "Performance on the independent external cohort.")

    # ---------------------------- Table 7 ---------------------------- #
    abl_stats = _csv(os.path.join(paths.STATS_DIR, "ablation_stats.csv"))
    if abl_stats is not None:
        cols = ["target", "variant", "variant_mean", "full_mean", "delta",
                "delta_pct", "p_holm", "significance", "contribution"]
        cols = [c for c in cols if c in abl_stats.columns]
        t7 = abl_stats[cols].rename(columns={
            "target": "Target", "variant": "Variant",
            "variant_mean": f"Variant {pm}", "full_mean": f"Full {pm}",
            "delta": "Contribution", "delta_pct": "Contribution (%)",
            "p_holm": "p (Holm)", "significance": "Sig.",
            "contribution": "Verdict"})
        emit(t7, 7, "ablation",
             "Ablation study. A positive contribution means removing the "
             "component degrades performance.")
    else:
        abl = _csv(paths.ABLATION_CSV)
        if abl is not None:
            emit(_agg(abl, "variant", metric_cols), 7, "ablation",
                 "Ablation study (descriptive).")

    # ---------------------------- Table 8 ---------------------------- #
    markers = _csv(os.path.join(paths.INTERPRET_DIR, "candidate_markers.csv"))
    if markers is not None:
        cols = ["target", "feature", "rank", "importance_mean",
                "selection_frequency", "stat", "p_fdr", "direction",
                "evidence_score", "tier"]
        cols = [c for c in cols if c in markers.columns]
        t8 = (markers[cols].sort_values(["target", "evidence_score"],
                                        ascending=[True, False])
              .groupby("target").head(15))
        t8 = t8.rename(columns={
            "target": "Target", "feature": "Feature", "rank": "Rank",
            "importance_mean": "Permutation importance",
            "selection_frequency": "Stability",
            "stat": "Univariate stat", "p_fdr": "FDR q",
            "direction": "Direction", "evidence_score": "Evidence",
            "tier": "Tier"})
        emit(t8, 8, "candidate_markers",
             "Top candidate markers ranked by combined model-based and "
             "univariate evidence.")

    # ---------------------------- Table 9 ---------------------------- #
    cond = _csv(os.path.join(paths.INTERPRET_DIR, "condition_performance.csv"))
    emit(cond, 9, "condition_performance",
         "Performance stratified by experimental condition.")

    # ---------------------------- Table 10 --------------------------- #
    try:
        import sklearn
        import torch
        versions = {"python": platform.python_version(),
                    "numpy": np.__version__, "pandas": pd.__version__,
                    "scikit-learn": sklearn.__version__,
                    "torch": torch.__version__}
    except Exception:                                      # noqa: BLE001
        versions = {"python": platform.python_version()}
    rows = [{"Setting": "Model name", "Value": paths.MODEL_NAME},
            {"Setting": "Outer folds", "Value": paths.N_OUTER_FOLDS},
            {"Setting": "Inner folds", "Value": paths.N_INNER_FOLDS},
            {"Setting": "Random seeds", "Value": str(paths.SEEDS)},
            {"Setting": "Bootstrap resamples", "Value": paths.BOOTSTRAP_N},
            {"Setting": "Device", "Value": paths.DEVICE},
            {"Setting": "Operating system", "Value": platform.platform()}]
    rows += [{"Setting": f"{k} version", "Value": v}
             for k, v in versions.items()]
    emit(pd.DataFrame(rows), 10, "reproducibility",
         "Software environment and protocol settings for reproducibility.")

    n = len([f for f in os.listdir(paths.TABLES_DIR) if f.endswith(".csv")])
    print(f"\n  {n} table CSV file(s) in {paths.TABLES_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
