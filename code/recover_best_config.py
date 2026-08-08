"""Recover best coarse+fine config from search_log.csv into best_config.json.

The tune run completed coarse+fine (search_log.csv written) but crashed on the
ablation write (PermissionError), so best_config.json was never updated. This
script selects the best-scoring FINE candidate and writes it as best_config.json
so that `tuner.py --skip-search` can re-run only the ablation + pruning.

Usage: python recover_best_config.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import json
import os

import numpy as np
import pandas as pd

import paths

META = {"phase", "iter", "score", "seconds"}


def _to_py(v):
    if pd.isna(v):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, (bool, int, float)):
        return v
    s = str(v).strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("none", "nan"):
        return None
    return s


def main() -> int:
    src = paths.SEARCH_LOG_CSV
    if not os.path.exists(src):
        print(f"search log not found: {src}")
        return 1
    df = pd.read_csv(src)
    fine = df[df["phase"] == "fine"].copy()
    if fine.empty:
        print("no fine rows in search log")
        return 1
    best = fine.sort_values("score", ascending=False).iloc[0]
    cfg = {k: (None if pd.isna(v) else _to_py(v)) for k, v in best.items()
           if k not in META}
    with open(paths.BEST_CONFIG_JSON, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"wrote {paths.BEST_CONFIG_JSON}")
    print(f"  best fine config: score={best['score']:.4f} "
          f"d_model={cfg.get('d_model')} n_blocks={cfg.get('n_blocks')} "
          f"dropout={cfg.get('dropout')} fusion={cfg.get('fusion')} "
          f"lr={cfg.get('lr')} batch={cfg.get('batch_size')} "
          f"epochs={cfg.get('max_epochs')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
