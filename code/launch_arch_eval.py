"""Launch H4 architecture evaluation: 3 variants in parallel (detached).

Each variant writes to its own results/arch_eval/<variant>_<ts>/ dir.
Logs to results/arch_eval_logs/<variant>.log / .err.
Writes results/arch_eval/pids.json with the launched PIDs for monitoring.

Usage:
    python launch_arch_eval.py            # launch all 3 (full protocol)
    python launch_arch_eval.py --smoke    # smoke test each variant
"""
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = r"C:\Users\TS\WorkBuddy\HydroGelNet"
CODE = os.path.join(ROOT, "code")
RESULTS = os.path.join(ROOT, "results")
LOG_DIR = os.path.join(RESULTS, "arch_eval_logs")
PY = r"C:\Users\TS\.conda\envs\HydroGelNet\python.exe"

VARIANTS = ["arch_A_depth", "arch_B_width", "arch_C_heads"]


def main():
    smoke = "--smoke" in sys.argv
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    pids = {}
    ts = datetime.now().strftime("%H%M%S")
    for v in VARIANTS:
        log_path = os.path.join(LOG_DIR, f"{v}.log")
        err_path = os.path.join(LOG_DIR, f"{v}.err")
        cmd = [PY, "arch_eval.py"] + ([ "--smoke"] if smoke else []) + [v]
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        with open(log_path, "w", encoding="utf-8") as out, \
             open(err_path, "w", encoding="utf-8") as err:
            p = subprocess.Popen(cmd, cwd=CODE, env=env,
                                 stdout=out, stderr=err,
                                 creationflags=flags, close_fds=True)
        pids[v] = p.pid
        print(f"[OK] {v} pid={p.pid} log={log_path}")

    snap = os.path.join(RESULTS, "arch_eval", "pids.json")
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    with open(snap, "w", encoding="utf-8") as fh:
        json.dump({"launched_at": ts, "smoke": smoke, "pids": pids},
                  fh, indent=2)
    print(f"snapshot: {snap}")
    print("monitor:  nvidia-smi / py-spy dump --pid <pid>")


if __name__ == "__main__":
    main()
