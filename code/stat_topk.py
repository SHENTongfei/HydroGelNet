"""External Top-k screening precision: paired bootstrap significance tests.
SIMPLEX vs each baseline (and vs baseline mean) on TopK20/TopK30."""
import numpy as np
import json

d = np.load("C:/Users/TS/WorkBuddy/HydroGelNet/results/preds/external_v3_all.npz",
            allow_pickle=True)
y = d["y"]
preds = {
    "SIMPLEX": d["simplex"],
    "RandomForest": d["rf"],
    "ElasticNet": d["en"],
    "Ridge": d["ridge"],
    "SVR-RBF": d["svr"],
}

def topk(y, p, k):
    return len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])) / k

rng = np.random.default_rng(42)
n_boot = 2000

out = {}
for k in (20, 30):
    rows = {}
    for name, p in preds.items():
        rows[name] = float(topk(y, p, k))
    base_mean = np.mean([rows[n] for n in preds if n != "SIMPLEX"])
    # paired bootstrap on sample-level resampling
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(len(y), size=len(y), replace=True)
        yd, pd_ = y[idx], {n: preds[n][idx] for n in preds}
        t_sim = topk(yd, pd_["SIMPLEX"], k)
        t_others = [topk(yd, pd_[n], k) for n in preds if n != "SIMPLEX"]
        diffs.append(t_sim - np.mean(t_others))
    diffs = np.array(diffs)
    out[f"TopK{k}"] = {
        "SIMPLEX": rows["SIMPLEX"],
        "baseline_mean": float(base_mean),
        "baselines": rows,
        "delta_vs_mean": float(np.mean(diffs)),
        "ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p_gt0": float(np.mean(diffs > 0)),
    }
    print(f"TopK{k}: SIMPLEX={rows['SIMPLEX']:.3f} vs mean={base_mean:.3f} "
          f"delta={np.mean(diffs):+.3f} CI={out[f'TopK{k}']['ci95']} "
          f"P(diff>0)={np.mean(diffs>0):.3f}")
    print(f"  baselines: { {n: round(v,3) for n, v in rows.items()} }")

with open("C:/Users/TS/WorkBuddy/HydroGelNet/results/stats/topk_stats.json", "w") as f:
    json.dump(out, f, indent=2)
print("saved topk_stats.json")
