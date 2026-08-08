"""Escalation H2: seed expansion for stronger ensemble (5 -> 10 seeds).

Trigger: L5 inherited-config ensemble R2 is close to but below RF ensemble
(0.8164). More seeds -> ensemble variance reduction -> usually +0.003-0.008.

Runs trainer.py with --seeds 10 (paths.SEEDS + 5 extra). This is a full
retrain of 10 seeds x 5 folds = 50 model fits (~60 min). Only launch when
L5 ensemble is within 0.01 of the RF ensemble reference.

Usage (detached):
    python launch_detached.py --from train --log opt_train_seeds10.log
    # then set SEEDS via a patch before launching, OR pass --seeds 10
    # to run_all.py (it forwards to trainer.py)
"""
import os

# Extra seeds to append (deterministic, fixed list)
EXTRA_SEEDS = [123, 4567, 20251, 314159, 271828]

PATCH = f"""
# Escalation H2: extended seeds for stronger ensemble
SEEDS = [42, 2024, 7, 1337, 20260731] + {EXTRA_SEEDS}
"""


def main():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paths.py")
    with open(p, encoding="utf-8") as f:
        src = f.read()
    old_line = "SEEDS = [42, 2024, 7, 1337, 20260731]"
    if "EXTRA_SEEDS" not in src:
        src = src.replace(old_line, PATCH.strip())
        with open(p, "w", encoding="utf-8") as f:
            f.write(src)
        print("PATCHED paths.SEEDS -> 10 seeds")
    else:
        print("ALREADY PATCHED")


if __name__ == "__main__":
    main()
