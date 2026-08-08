"""L5 gate verdict reader: decide branch A (internal win) vs B (tie+external).

Run after perf_gate.json is fresh (L5). Prints verdict + key numbers and
writes a small JSON decision file used by the delivery chain.
"""
import json
import os
import sys

import pandas as pd

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
METRICS = os.path.join(ROOT, "results", "metrics")


def main():
    gate_p = os.path.join(METRICS, "perf_gate.json")
    if not os.path.exists(gate_p):
        print("NO_GATE_YET")
        return 1
    with open(gate_p, encoding="utf-8") as f:
        gate = json.load(f)

    cv = pd.read_csv(os.path.join(METRICS, "cv_outer.csv"))
    ours = cv[cv["model"] == "SIMPLEX"]
    r2 = ours["R2"].mean()
    seeds = ours["seed"].nunique()

    # ensemble R2 from preds_cv_main (5-seed average per sample)
    preds = pd.read_csv(os.path.join(ROOT, "results", "preds",
                                     "preds_cv_main.csv"))
    yc = [c for c in preds.columns if c.startswith("y_true")]
    pc = [c for c in preds.columns if c.startswith("y_pred")]
    ens = preds.groupby("sample_id").agg(
        yt=(yc[0], "first"), yp=(pc[0], "mean"))
    y, p = ens["yt"].values, ens["yp"].values
    ens_r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    # per-seed direction vs RF (from gate)
    g3 = gate.get("checks", {}).get("G3_seed_stability", {})
    g4 = gate.get("checks", {}).get("G4_external", {})
    g1 = gate.get("checks", {}).get("G1_beats_best_baseline", {})

    verdict = gate.get("verdict", "?")
    delta = gate.get("delta", float("nan"))
    rf = gate.get("best_baseline_mean", float("nan"))
    ext_r2 = gate.get("proposed_external", None)

    branch = "A" if (r2 > rf or delta > 0) else "B"
    dec = {
        "gate_verdict": verdict,
        "branch": branch,
        "internal_r2_perfold": float(r2),
        "internal_r2_ensemble": float(ens_r2),
        "rf_ref": float(rf),
        "delta": float(delta),
        "n_seeds": int(seeds),
        "external_r2": ext_r2,
        "g1": g1.get("detail"),
        "g3": g3.get("detail"),
        "g4": g4.get("detail"),
    }
    with open(os.path.join(ROOT, "audit", "l5_decision.json"), "w",
              encoding="utf-8") as f:
        json.dump(dec, f, indent=2, ensure_ascii=False)
    print(json.dumps(dec, indent=2))


if __name__ == "__main__":
    main()
