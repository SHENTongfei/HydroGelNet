"""Interpretation and domain discovery.

This is where the study stops being a benchmark and starts being science.
Produced evidence:

  1. Permutation importance, computed on every OUTER TEST fold, so the ranking
     is itself cross-validated instead of fitted on the training data.
  2. Stability selection -- how often a feature enters the top-k across folds
     and seeds. Unstable rankings are the classic small-n trap.
  3. Attention attribution -- the CLS token's attention over feature-block
     tokens, averaged over samples, plus per-condition attention profiles.
  4. Latent geometry -- PCA (always) and UMAP (if installed) of the learned
     representation, coloured by target and by condition.
  5. One-dimensional partial dependence for the top features.
  6. A candidate-marker table combining model importance with a univariate
     association test (FDR corrected) and the sign of the effect.
  7. Per-condition performance, to show the model is not carried by one
     subgroup.

Outputs
-------
    results/interpret/importance.csv
    results/interpret/stability.csv
    results/interpret/attention.csv
    results/interpret/attention_by_condition.csv
    results/interpret/latent.npz
    results/interpret/embedding.csv
    results/interpret/pdp.csv
    results/interpret/candidate_markers.csv
    results/interpret/condition_performance.csv

Usage
-----
    python interpret.py
    python interpret.py --seeds 2 --top-k 20 --quick
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import multipletests

import paths
from build_dataset import load_dataset, split_modalities
from trainer import (PRIMARY_METRIC, TrainConfig, compute_metrics, predict,
                     run_cv)


# --------------------------------------------------------------------------- #
def permutation_importance_fold(model, cfg, prep, ends, X_test, Y_test, C_test,
                                n_cond, task_type, metric, n_repeats, rng
                                ) -> np.ndarray:
    """Drop in ``metric`` when each raw feature column is shuffled."""
    Xs = prep.transform_x(X_test)
    X1, X2 = split_modalities(Xs, ends)
    base_pred, _ = predict(model, cfg, X1, X2, C_test, n_cond, task_type)
    if task_type == "regression":
        base_pred = prep.inverse_y(base_pred)
    base = np.mean([compute_metrics(Y_test[:, t], base_pred[:, t],
                                    task_type).get(metric, np.nan)
                    for t in range(Y_test.shape[1])])

    d = X_test.shape[1]
    drops = np.zeros(d)
    for j in range(d):
        vals = []
        for _ in range(n_repeats):
            Xp = X_test.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            Xps = prep.transform_x(Xp)
            P1, P2 = split_modalities(Xps, ends)
            p, _ = predict(model, cfg, P1, P2, C_test, n_cond, task_type)
            if task_type == "regression":
                p = prep.inverse_y(p)
            vals.append(np.mean([compute_metrics(Y_test[:, t], p[:, t],
                                                 task_type).get(metric, np.nan)
                                 for t in range(Y_test.shape[1])]))
        drops[j] = base - np.nanmean(vals)
    return drops


def partial_dependence(model, cfg, prep, ends, X, C, n_cond, task_type,
                       col: int, grid_n: int = 20) -> pd.DataFrame:
    """1-D partial dependence for one raw feature column."""
    lo, hi = np.nanpercentile(X[:, col], [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.DataFrame()
    grid = np.linspace(lo, hi, grid_n)
    rows = []
    for g in grid:
        Xg = X.copy()
        Xg[:, col] = g
        Xs = prep.transform_x(Xg)
        G1, G2 = split_modalities(Xs, ends)
        p, _ = predict(model, cfg, G1, G2, C, n_cond, task_type)
        if task_type == "regression":
            p = prep.inverse_y(p)
        for t in range(p.shape[1]):
            rows.append({"grid_value": float(g), "target_index": t,
                         "pd_mean": float(p[:, t].mean()),
                         "pd_sd": float(p[:, t].std())})
    return pd.DataFrame(rows)


def univariate_association(X: np.ndarray, y: np.ndarray, task_type: str
                           ) -> pd.DataFrame:
    """Spearman (regression) or Mann-Whitney (classification) per feature."""
    rows = []
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 8 or np.std(x[ok]) == 0:
            rows.append({"stat": np.nan, "p": np.nan, "direction": 0.0})
            continue
        if task_type == "regression":
            r, p = stats.spearmanr(x[ok], y[ok])
            rows.append({"stat": float(r), "p": float(p),
                         "direction": float(np.sign(r))})
        else:
            pos, neg = y[ok] > 0.5, y[ok] <= 0.5
            if pos.sum() < 3 or neg.sum() < 3:
                rows.append({"stat": np.nan, "p": np.nan, "direction": 0.0})
                continue
            u, p = stats.mannwhitneyu(x[ok][pos], x[ok][neg])
            auc = u / (pos.sum() * neg.sum())
            rows.append({"stat": float(auc), "p": float(p),
                         "direction": float(np.sign(auc - 0.5))})
    df = pd.DataFrame(rows)
    ok = df["p"].notna()
    df["p_fdr"] = np.nan
    if ok.sum() > 0:
        df.loc[ok, "p_fdr"] = multipletests(df.loc[ok, "p"],
                                            method="fdr_bh")[1]
    return df


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--perm-repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    paths.ensure_dirs()
    paths.banner("STEP 8/9  INTERPRETATION AND DOMAIN DISCOVERY")

    cfg = TrainConfig()
    if os.path.exists(paths.BEST_CONFIG_JSON):
        with open(paths.BEST_CONFIG_JSON, "r", encoding="utf-8") as fh:
            cfg = TrainConfig.from_dict(json.load(fh))
    seeds = paths.SEEDS[:max(1, args.seeds)]
    n_repeats = args.perm_repeats
    if args.quick:
        cfg.max_epochs, cfg.patience, cfg.contrastive_epochs = 30, 8, 5
        seeds, n_repeats = paths.SEEDS[:1], 2

    ds = load_dataset(paths.DATASET_NPZ)
    X, Y = ds["X"], ds["Y"]
    C = np.asarray(ds["cond"]).astype(int)
    n_cond = int(len(ds["cond_levels"]))
    cond_levels = [str(c) for c in ds["cond_levels"]]
    task_type = ds["task_type"]
    fnames = [str(f) for f in ds["feature_names"]]
    tnames = [str(t) for t in ds["target_names"]]
    modality_names = [str(m) for m in ds["modality_names"]]
    metric = PRIMARY_METRIC[task_type]

    print(f"  refitting CV models for interpretation (seeds={seeds}) ...")
    _, preds_df, fitted = run_cv(cfg, ds, seeds, tag="interpret", verbose=False)
    from trainer import make_outer_splits
    print(f"  {len(fitted)} fold-models available")

    rng = np.random.default_rng(paths.PRIMARY_SEED)

    # ---------------- 1+2. permutation importance & stability -------------- #
    imp_rows, stab_counter = [], {f: 0 for f in fnames}
    n_models = 0
    for item in fitted:
        seed, fold = item["seed"], item["fold"]
        splits = make_outer_splits(ds, seed=seed)
        _, te_idx = splits[fold]
        drops = permutation_importance_fold(
            item["model"], cfg, item["prep"], item["ends"],
            X[te_idx], Y[te_idx], C[te_idx], n_cond, task_type,
            metric, n_repeats, rng)
        for j, f in enumerate(fnames):
            imp_rows.append({"seed": seed, "fold": fold, "feature": f,
                             "importance": float(drops[j])})
        for j in np.argsort(-drops)[:args.top_k]:
            stab_counter[fnames[j]] += 1
        n_models += 1
        print(f"    permutation importance: seed={seed} fold={fold} done")

    imp_long = pd.DataFrame(imp_rows)
    imp = (imp_long.groupby("feature")["importance"]
           .agg(["mean", "std", "count"]).reset_index()
           .rename(columns={"mean": "importance_mean",
                            "std": "importance_sd",
                            "count": "n_folds"}))
    imp["importance_se"] = imp["importance_sd"] / np.sqrt(imp["n_folds"])
    imp = imp.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(paths.IMPORTANCE_CSV, index=False)

    stab = pd.DataFrame([{"feature": f, "times_in_topk": c,
                          "selection_frequency": c / max(n_models, 1)}
                         for f, c in stab_counter.items()])
    stab = stab.sort_values("selection_frequency", ascending=False)
    p_stab = os.path.join(paths.INTERPRET_DIR, "stability.csv")
    stab.to_csv(p_stab, index=False)
    print(f"  [1/7] importance -> {paths.IMPORTANCE_CSV}")
    print(f"  [2/7] stability  -> {p_stab}")

    # ------------------- 2b. modality-gate importance ---------------------- #
    # If the final config uses the learnable ModalityGate, report the mean
    # gate per modality across all fold-models: the model's own answer to
    # "which feature block do I actually rely on".
    gate_rows = []
    for item in fitted:
        model = item["model"]
        gate = getattr(getattr(model, "mod_gate", None), "last_gates", None)
        if gate is None:
            continue
        for mi, mname in enumerate(modality_names):
            gate_rows.append({"seed": item["seed"], "fold": item["fold"],
                              "modality": mname,
                              "gate": float(np.mean(gate[:, mi]))})
    if gate_rows:
        gdf = (pd.DataFrame(gate_rows).groupby("modality")["gate"]
               .agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"mean": "gate_mean", "std": "gate_sd",
                                "count": "n_folds"}))
        gdf = gdf.sort_values("gate_mean", ascending=False)
        p_gate = os.path.join(paths.INTERPRET_DIR, "modality_gate.csv")
        gdf.to_csv(p_gate, index=False)
        print(f"  [2b] modality gates -> {p_gate}")
        for _, r in gdf.iterrows():
            print(f"       {r['modality']:<10s} gate={r['gate_mean']:.3f} "
                  f"+- {r['gate_sd']:.3f}")
    else:
        print("  [2b] modality gates skipped (ModalityGate not in final model)")

    # --------------------- 3. attention attribution ------------------------ #
    attn_rows, attn_cond_rows = [], []
    ref = fitted[0]
    Xs = ref["prep"].transform_x(X)
    A1, A2 = split_modalities(Xs, ref["ends"])
    tok_names = ref["model"].token_names()
    with torch.no_grad():
        amap = ref["model"].attention_map(
            torch.as_tensor(A1, dtype=torch.float32, device=paths.DEVICE),
            (torch.as_tensor(A2, dtype=torch.float32, device=paths.DEVICE)
             if A2.shape[1] else None),
            (torch.as_tensor(C, dtype=torch.long, device=paths.DEVICE)
             if n_cond > 0 and cfg.use_film else None))
    if amap is not None:
        amap = amap.cpu().numpy()
        bounds = ref["model"].token_bounds()
        keep = np.where(ref["prep"].keep_cols)[0]
        for i, tname in enumerate(tok_names[:amap.shape[1]]):
            lo_hi = bounds.get(tname)
            members = ([fnames[keep[c]] for c in range(*lo_hi)
                        if c < len(keep)] if lo_hi else [])
            attn_rows.append({
                "token": tname,
                "attention_mean": float(amap[:, i].mean()),
                "attention_sd": float(amap[:, i].std()),
                "n_features": len(members),
                "features": "; ".join(members[:12]),
            })
            for k, lvl in enumerate(cond_levels):
                sel = C == k
                if sel.sum() >= 3:
                    attn_cond_rows.append({
                        "token": tname, "condition": lvl,
                        "n": int(sel.sum()),
                        "attention_mean": float(amap[sel, i].mean())})
        ent = -(np.clip(amap, 1e-9, 1) * np.log(np.clip(amap, 1e-9, 1))).sum(1)
        print(f"  [3/7] attention entropy: {ent.mean():.3f} "
              f"(uniform would be {np.log(amap.shape[1]):.3f})")
    p_attn = os.path.join(paths.INTERPRET_DIR, "attention.csv")
    pd.DataFrame(attn_rows).to_csv(p_attn, index=False)
    pd.DataFrame(attn_cond_rows).to_csv(
        os.path.join(paths.INTERPRET_DIR, "attention_by_condition.csv"),
        index=False)

    # ------------------------- 4. latent geometry -------------------------- #
    with torch.no_grad():
        _, latent = predict(ref["model"], cfg, A1, A2, C, n_cond, task_type)
    np.savez_compressed(paths.LATENT_NPZ, latent=latent, Y=Y, cond=C,
                        sample_ids=ds["sample_ids"])
    pca = PCA(n_components=min(10, latent.shape[1]), random_state=0)
    pcs = pca.fit_transform(latent)
    emb = pd.DataFrame({"sample_id": [str(s) for s in ds["sample_ids"]],
                        "condition": [cond_levels[c] for c in C],
                        "PC1": pcs[:, 0], "PC2": pcs[:, 1]})
    for t, tn in enumerate(tnames):
        emb[f"y_{tn}"] = Y[:, t]
    emb["method"] = "PCA"
    try:
        import umap                                        # noqa: PLC0415
        um = umap.UMAP(n_neighbors=min(15, len(latent) - 1), min_dist=0.1,
                       random_state=paths.PRIMARY_SEED).fit_transform(latent)
        emb["UMAP1"], emb["UMAP2"] = um[:, 0], um[:, 1]
        emb["method"] = "PCA+UMAP"
        print("  [4/7] latent geometry: PCA + UMAP")
    except Exception:                                      # noqa: BLE001
        emb["UMAP1"], emb["UMAP2"] = np.nan, np.nan
        print("  [4/7] latent geometry: PCA only (umap-learn not installed)")
    emb.to_csv(os.path.join(paths.INTERPRET_DIR, "embedding.csv"), index=False)
    pd.DataFrame({"component": np.arange(1, len(pca.explained_variance_ratio_) + 1),
                  "explained_variance_ratio": pca.explained_variance_ratio_}
                 ).to_csv(os.path.join(paths.INTERPRET_DIR, "pca_variance.csv"),
                          index=False)

    # --------------------------- 5. partial dependence --------------------- #
    top_feats = imp.head(min(6, len(imp)))["feature"].tolist()
    pdp_all = []
    for f in top_feats:
        col = fnames.index(f)
        d = partial_dependence(ref["model"], cfg, ref["prep"], ref["ends"],
                               X, C, n_cond, task_type, col)
        if len(d):
            d["feature"] = f
            d["target"] = d["target_index"].map(lambda i: tnames[int(i)])
            pdp_all.append(d)
    pdp = pd.concat(pdp_all) if pdp_all else pd.DataFrame()
    pdp.to_csv(os.path.join(paths.INTERPRET_DIR, "pdp.csv"), index=False)
    print(f"  [5/7] partial dependence for {len(top_feats)} feature(s)")

    # -------------------------- 6. candidate markers ----------------------- #
    marker_frames = []
    for t, tn in enumerate(tnames):
        uni = univariate_association(X, Y[:, t], task_type)
        uni["feature"] = fnames
        uni["target"] = tn
        marker_frames.append(uni)
    markers = pd.concat(marker_frames)
    markers = markers.merge(imp[["feature", "importance_mean", "rank"]],
                            on="feature", how="left")
    markers = markers.merge(stab[["feature", "selection_frequency"]],
                            on="feature", how="left")
    markers["evidence_score"] = (
        markers["selection_frequency"].fillna(0) * 0.4
        + (1 - markers["rank"].fillna(len(imp)) / max(len(imp), 1)) * 0.3
        + (markers["p_fdr"].fillna(1) < 0.05).astype(float) * 0.3)
    markers = markers.sort_values(["target", "evidence_score"],
                                  ascending=[True, False])
    markers["tier"] = np.where(markers["evidence_score"] >= 0.7, "high",
                               np.where(markers["evidence_score"] >= 0.45,
                                        "moderate", "low"))
    p_mark = os.path.join(paths.INTERPRET_DIR, "candidate_markers.csv")
    markers.to_csv(p_mark, index=False)
    n_high = int((markers["tier"] == "high").sum())
    print(f"  [6/7] candidate markers -> {p_mark} ({n_high} high-evidence)")

    # ----------------------- 7. per-condition performance ------------------ #
    cond_rows = []
    if len(preds_df):
        cond_map = {i: lv for i, lv in enumerate(cond_levels)}
        for cval, grp in preds_df.groupby("cond"):
            for tn in tnames:
                yc, pc = f"y_true_{tn}", f"y_pred_{tn}"
                if yc not in grp.columns or len(grp) < 5:
                    continue
                m = compute_metrics(grp[yc].to_numpy(), grp[pc].to_numpy(),
                                    task_type)
                cond_rows.append({"condition": cond_map.get(int(cval), str(cval)),
                                  "target": tn, "n": len(grp), **m})
    cond_df = pd.DataFrame(cond_rows)
    p_cond = os.path.join(paths.INTERPRET_DIR, "condition_performance.csv")
    cond_df.to_csv(p_cond, index=False)
    print(f"  [7/7] per-condition performance -> {p_cond}")

    print("\n  Top-10 features by cross-validated permutation importance:")
    print(imp.head(10)[["rank", "feature", "importance_mean",
                        "importance_se"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
