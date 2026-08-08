"""Recover the PRUNED best config after tuner crashed on the final write.

The ablation completed and ablation_results.csv was saved, but the tuner then
crashed writing best_config.json (transient Windows file lock). This script
re-runs ONLY the pruning step (prune_config) on the saved ablation results and
writes the pruned config to paths.BEST_CONFIG_JSON (fresh filename) so the
train stage can proceed without recomputing the ~50 min ablation.

Inputs : results/tuning/best_config.json        (raw search winner, fine,37)
         results/ablation/ablation_results.csv  (completed ablation)
Output : results/tuning/best_config_final.json  (pruned) + pruning_notes.txt
Usage  : python recover_pruned_config.py
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402
from trainer import TrainConfig  # noqa: E402
from tuner import prune_config  # noqa: E402

RAW_CONFIG = os.path.join(paths.TUNING_DIR, "best_config.json")
PM = "R2"


def _write_text_retry(text: str, path: str, tries: int = 12,
                      delay: float = 6.0) -> None:
    for _i in range(tries):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
            return
        except PermissionError:
            time.sleep(delay)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def main() -> int:
    if not os.path.exists(RAW_CONFIG):
        print(f"raw config not found: {RAW_CONFIG}")
        return 1
    if not os.path.exists(paths.ABLATION_CSV):
        print(f"ablation results not found: {paths.ABLATION_CSV}")
        return 1
    with open(RAW_CONFIG, encoding="utf-8") as fh:
        cfg = TrainConfig.from_dict(json.load(fh))
    abl = pd.read_csv(paths.ABLATION_CSV)
    final_cfg, notes = prune_config(cfg, abl, PM)

    _write_text_retry(json.dumps(final_cfg.to_dict(), indent=2),
                      paths.BEST_CONFIG_JSON)
    notes_txt = "\n".join(notes) if notes else "No component was removed.\n"
    _write_text_retry(notes_txt,
                      os.path.join(paths.TUNING_DIR, "pruning_notes.txt"))

    print(f"PRUNED_CONFIG_WRITTEN -> {paths.BEST_CONFIG_JSON}")
    print(f"pruned {len(notes)} neutral component(s):")
    for n in notes:
        print(f"  {n}")
    d = final_cfg.to_dict()
    print(f"kept: fusion={d.get('fusion')} use_attention={d.get('use_attention')} "
          f"use_mixup={d.get('use_mixup')} use_contrastive={d.get('use_contrastive')} "
          f"use_film={d.get('use_film')} use_residual={d.get('use_residual')} "
          f"use_modality_gate={d.get('use_modality_gate')} "
          f"use_swa={d.get('use_swa')} use_sam={d.get('use_sam')} "
          f"use_ema={d.get('use_ema')} use_domain_constraint={d.get('use_domain_constraint')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
