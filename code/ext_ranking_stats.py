"""External ranking significance: sample-level bootstrap CI for Spearman.
Fixes P0-1 (statistical support for 'SIMPLEX ranks first externally').
Compares SIMPLEX ensemble vs ElasticNet vs RF on the 161-sample external set."""
import numpy as np
import pandas as pd
import os
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import paths
from build_dataset import load_dataset

ds = load_dataset(paths.DATASET_NPZ)
ext = load_dataset(paths.EXTERNAL_NPZ)
Xtr, ytr = ds["X"], ds["Y"].ravel()
Xte, yte = ext["X"], ext["Y"].ravel()
n = len(yte)

# SIMPLEX 25-model ensemble predictions
preds_ext = pd.read_csv(os.path.join(paths.PREDS_DIR, "preds_external.csv"))
p_simplex = preds_ext["y_pred_glass_adhesion_kpa"].values
assert len(p_simplex) == n

# RF and ElasticNet external predictions (5 seeds each)
SEEDS = [42, 2024, 7, 1337, 20260731]
p_rf, p_el = [], []
for s in SEEDS:
    rf = RandomForestRegressor(n_estimators=500, random_state=s)
    rf.fit(Xtr, ytr)
    p_rf.append(rf.predict(Xte))
    el = make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, random_state=s))
    el.fit(Xtr, ytr)
    p_el.append(el.predict(Xte))
p_rf = np.mean(p_rf, axis=0)
p_el = np.mean(p_el, axis=0)

def bootstrap_spearman(pred, y, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(spearmanr(y[idx], pred[idx])[0])
    vals = np.array(vals)
    return vals.mean(), np.percentile(vals, [2.5, 97.5])

def spearman_diff_boot(p1, p2, y, n_boot=2000, seed=42):
    """Bootstrap distribution of rho(p1)-rho(p2) on resampled samples (paired)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(spearmanr(y[idx], p1[idx])[0] - spearmanr(y[idx], p2[idx])[0])
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, [2.5, 97.5]), (diffs > 0).mean()

print("=== External Spearman rho with 95% bootstrap CI (n=161) ===")
rho_s = spearmanr(yte, p_simplex)[0]
rho_rf = spearmanr(yte, p_rf)[0]
rho_el = spearmanr(yte, p_el)[0]
for name, p in [("SIMPLEX", p_simplex), ("RF", p_rf), ("ElasticNet", p_el)]:
    m, ci = bootstrap_spearman(p, yte)
    print(f"  {name:<10}: rho={spearmanr(yte,p)[0]:.3f}  bootstrap mean={m:.3f}  95% CI=[{ci[0]:.3f}, {ci[1]:.3f}]")

print("\n=== Paired bootstrap: rho(SIMPLEX) - rho(baseline) ===")
for name, p in [("RF", p_rf), ("ElasticNet", p_el)]:
    m, ci, frac_pos = spearman_diff_boot(p_simplex, p, yte)
    sig = "significant" if ci[0] > 0 else ("significant (negative)" if ci[1] < 0 else "NOT significant")
    print(f"  SIMPLEX - {name:<10}: mean diff={m:+.3f}  95% CI=[{ci[0]:+.3f}, {ci[1]:+.3f}]  P(diff>0)={frac_pos:.3f}  -> {sig}")

# Top-k screening precision
def topk_prec(pred, k):
    topk_pred = np.argsort(pred)[-k:]
    topk_true = set(np.argsort(yte)[-k:])
    return len(set(topk_pred) & topk_true) / k
print("\n=== Top-k screening precision (external) ===")
for k in [10, 20, 30]:
    print(f"  k={k}: SIMPLEX={topk_prec(p_simplex,k):.2f}  RF={topk_prec(p_rf,k):.2f}  ElasticNet={topk_prec(p_el,k):.2f}")

# Prediction-range compression check (P0-3: does SIMPLEX compress less?)
print("\n=== Prediction range (kPa) on external set ===")
for name, p in [("SIMPLEX", p_simplex), ("RF", p_rf), ("ElasticNet", p_el)]:
    print(f"  {name:<10}: pred range=[{p.min():.0f}, {p.max():.0f}]  std={p.std():.0f}  | true range=[{yte.min():.0f}, {yte.max():.0f}] std={yte.std():.0f}")
