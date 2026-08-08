"""Escalation H4: architecture micro-variants evaluated with FULL protocol.

Lesson learned: 3-fold x 1-seed search picked fine,37 (score 0.8010) that
failed on full protocol (0.7783). Therefore H4 evaluates 2-3 candidate
architectures DIRECTLY on the full protocol (5-fold x 5-seed), no search.

Candidates (each one changes exactly ONE axis from the inherited config):
  A. depth:   n_blocks 1 -> 2   (deeper residual refinement)
  B. width:   d_model 152 -> 192 (wider embeddings, heads 8 -> 12)
  C. heads:   n_heads 8 -> 12    (more attention heads, same width)

Usage:
    python make_arch_variants.py   # writes candidate JSON files
Then launch each with trainer.py --config results/tuning/arch_A.json --tag main
"""
import json
import os

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
BASE = os.path.join(ROOT, "results", "tuning", "best_config_final.json")


def main():
    with open(BASE, encoding="utf-8") as f:
        base = json.load(f)

    variants = {
        "arch_A_depth": dict(base, n_blocks=2),
        "arch_B_width": dict(base, d_model=192, n_heads=12, proj_dim=48),
        "arch_C_heads": dict(base, n_heads=12),
    }
    for name, cfg in variants.items():
        p = os.path.join(ROOT, "results", "tuning", f"{name}.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        print(f"WROTE {name}: d_model={cfg['d_model']} n_blocks={cfg['n_blocks']} "
              f"n_heads={cfg['n_heads']} dropout={cfg['dropout']}")


if __name__ == "__main__":
    main()
