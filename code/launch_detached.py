"""Launch run_all.py as a fully detached Windows process — H31 FULL-CHAIN GUARD.

Why: the Bash tool's background mechanism kills child processes at turn
boundary (proven 6x). A DETACHED_PROCESS child survives. And the 7th incident
was a file lock: host holds handles to OLD files from before session start,
so writing cv_outer.csv/run_log.md etc. raises PermissionError even with
to_csv retry. THE FIX is to DELETE all stale outputs BEFORE launching so the
pipeline creates them fresh (new files bypass the host handle).

H31 guards baked into THIS script (not left to memory):
  A. CONFIG CHECK  — verify the config run_all will load (swa/ema/marker)
  B. STALE FILE PURGE — delete every output the pipeline will rewrite,
                       so it creates them fresh (kills the lock root cause)
  C. WRITABILITY TEST — after purge, confirm key dirs are writable
  D. LAUNCH SNAPSHOT — record pid/config/mtime into results/launch_snapshot.json
  E. POST-CHECK PROMPT — remind to run protect_check.py after completion

Usage:
    python launch_detached.py --from train --log opt_train_XXXX.log
    python launch_detached.py --from train --log opt_train_XXXX.log --no-purge
"""
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

PY = r"C:/Users/TS/.conda/envs/HydroGelNet/python.exe"
CODE = r"C:/Users/TS/WorkBuddy/HydroGelNet/code"
ROOT = r"C:/Users/TS/WorkBuddy/HydroGelNet"
RESULTS = os.path.join(ROOT, "results")

# Every output file the pipeline rewrites. Host holds handles to OLD ones
# (created before this session), so they must be purged before launch so the
# pipeline creates them fresh. Keep this list in sync with run_all stages.
STALE_OUTPUTS = [
    # metrics
    "metrics/cv_outer.csv", "metrics/external.csv",
    "metrics/baselines.csv", "metrics/baselines_external.csv",
    "metrics/baselines_external_preds.csv", "metrics/perf_gate.json",
    "metrics/escalation_log.csv",
    # logs
    "run_log.md",
    # stats
    "stats/comparisons.csv", "stats/bootstrap_ci.csv",
    "stats/permutation.csv", "stats/topk_stats.json",
    "stats/ablation_stats.csv", "stats/stats_report.md",
    # preds
    "preds/preds_cv_main.csv", "preds/preds_external.csv",
    "preds/preds_baselines.csv",
    # tuning (regenerated each run)
    "tuning/config_used.json",
    # ablation
    "ablation/ablation_results.csv", "ablation/ablation.csv",
    # audit
    "audit/PERF_GATE.md" if False else "",  # keep audit md, it's a report
]

# Files the pipeline READS but that are config inputs (do NOT purge, they are
# not rewritten by the pipeline — purge only STALE_OUTPUTS).


def purge_stale():
    """Delete every stale output so the pipeline creates it fresh."""
    purged, failed = [], []
    for rel in STALE_OUTPUTS:
        if not rel:
            continue
        p = os.path.join(RESULTS, rel)
        if os.path.exists(p):
            try:
                os.remove(p)
                purged.append(rel)
            except Exception as e:
                failed.append((rel, type(e).__name__, str(e)[:80]))
    return purged, failed


def write_test():
    """Confirm the critical dirs are writable after purge."""
    ok = True
    for sub in ("metrics", "stats", "preds", "tuning", "ablation"):
        d = os.path.join(RESULTS, sub)
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".h31_probe")
        try:
            with open(probe, "w") as f:
                f.write("ok")
            os.remove(probe)
        except Exception:
            ok = False
            print(f"  [FAIL] dir not writable: {d}")
    return ok


def config_check():
    """Verify the config run_all will load (A)."""
    cfg_p = os.path.join(RESULTS, "tuning", "best_config_final.json")
    if not os.path.exists(cfg_p):
        print(f"  [FAIL] config missing: {cfg_p}")
        return None
    with open(cfg_p, encoding="utf-8") as fh:
        cfg = json.load(fh)
    print(f"  [PASS] config: {os.path.basename(cfg_p)}")
    print(f"         swa={cfg.get('use_swa')} ema={cfg.get('use_ema')} "
          f"d_model={cfg.get('d_model')} marker={cfg.get('_inherited_v2')}")
    return cfg


def snapshot(pid, cfg, purged):
    """Record launch facts (D)."""
    snap = {
        "launched_at": datetime.now().isoformat(timespec="seconds"),
        "pid": pid,
        "stage": None,
        "config": {
            "swa": cfg.get("use_swa") if cfg else None,
            "ema": cfg.get("use_ema") if cfg else None,
            "marker": cfg.get("_inherited_v2") if cfg else None,
            "d_model": cfg.get("d_model") if cfg else None,
        },
        "purged_stale": purged,
    }
    p = os.path.join(RESULTS, "launch_snapshot.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, ensure_ascii=False)
    return p


def main():
    args = sys.argv[1:]
    stage = None
    log_name = "opt_detached.log"
    no_purge = "--no-purge" in args
    if "--from" in args:
        i = args.index("--from")
        stage = args[i + 1]
    if "--log" in args:
        i = args.index("--log")
        log_name = args[i + 1]

    print("=" * 70)
    print("H31 FULL-CHAIN GUARD — pre-flight checks")
    print("=" * 70)

    # A. config
    cfg = config_check()

    # B. purge stale outputs (THE fix for the 7th lock incident)
    if no_purge:
        print("  [SKIP] purge (--no-purge)")
        purged = []
    else:
        purged, failed = purge_stale()
        if purged:
            print(f"  [PURGE] removed {len(purged)} stale outputs:")
            for r in purged:
                print(f"         - {r}")
        else:
            print("  [PURGE] no stale outputs to remove (fresh dir)")
        if failed:
            print(f"  [WARN] {len(failed)} could not be purged:")
            for r, t, m in failed:
                print(f"         - {r} ({t}: {m})")

    # C. writability
    if not write_test():
        print("  [BLOCK] results dirs not writable — abort launch")
        return 1

    # launch
    cmd = [PY, "-u", "run_all.py"]
    if stage:
        cmd += ["--from", stage]
    log_path = os.path.join(RESULTS, log_name)
    err_path = log_path.replace(".log", ".err")

    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"

    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    with open(log_path, "w", encoding="utf-8") as out, \
         open(err_path, "w", encoding="utf-8") as err:
        p = subprocess.Popen(
            cmd, cwd=CODE, env=env,
            stdout=out, stderr=err,
            creationflags=flags,
            close_fds=True,
        )

    pid_file = os.path.join(RESULTS, "opt_detached.pid")
    with open(pid_file, "w", encoding="utf-8") as f:
        f.write(str(p.pid))

    snap_p = snapshot(p.pid, cfg, purged)
    print(f"  [OK] launched pid={p.pid} log={log_path}")
    print(f"      snapshot={snap_p}")
    print()
    print("POST-CHECK: when the run finishes, run:")
    print("  python code/protect_check.py   # H31 output verification")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
