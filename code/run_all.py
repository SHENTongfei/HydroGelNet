"""One-command orchestrator for the whole study.

Runs every stage in the correct order, in separate processes, so that a crash
in one stage cannot corrupt the state of another. Timing, exit codes and the
exact command line of every stage are recorded in ``results/run_log.md``.

Stage order
-----------
    1  download   download_data.py   verify links, fetch raw files
    2  build      build_dataset.py   raw -> dataset.npz + dataset_external.npz
    3  qc         data_qc.py         quality control, leakage and shift audit
    4  tune       tuner.py           hyper-parameter search + ablation + pruning
    5  train      trainer.py         final nested CV with the pruned config
    6  baselines  baselines.py       equally tuned classical baselines
    7  stats      stats_tests.py     corrected tests, bootstrap, permutation
       gate       escalate.py        PERF-GATE: did we actually win? (no-fail)
    8  interpret  interpret.py       importance, attention, markers
    9  figures    figures.py         8 publication figures
       tables     tables.py          10 publication tables
       paper      paper_pdf.py       manuscript Markdown + PDF + self-check

Usage
-----
    python run_all.py --all
    python run_all.py --all --demo --quick          # synthetic smoke test
    python run_all.py --only figures tables paper   # re-render deliverables
    python run_all.py --from stats                  # resume after a crash
    python run_all.py --all --skip download         # data already present
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import paths

CODE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE_ORDER = ["download", "build", "qc", "tune", "train", "baselines",
               "stats", "gate", "interpret", "figures", "tables", "paper"]

SCRIPTS = {
    "download": "download_data.py",
    "build": "build_dataset.py",
    "qc": "data_qc.py",
    "tune": "tuner.py",
    "train": "trainer.py",
    "baselines": "baselines.py",
    "stats": "stats_tests.py",
    "gate": "escalate.py",
    "interpret": "interpret.py",
    "figures": "figures.py",
    "tables": "tables.py",
    "paper": "paper_pdf.py",
}

LABELS = {
    "download": "STEP 1/9  download and verify data",
    "build": "STEP 2/9  build the canonical dataset",
    "qc": "STEP 3/9  quality control and leakage audit",
    "tune": "STEP 4/9  hyper-parameter search and ablation",
    "train": "STEP 5/9  cross-validated training",
    "baselines": "STEP 6/9  classical baselines",
    "stats": "STEP 7/9  statistical testing",
    "gate": "          PERF-GATE (no-fail protocol)",
    "interpret": "STEP 8/9  interpretation and discovery",
    "figures": "STEP 9/9  publication figures",
    "tables": "          publication tables",
    "paper": "          manuscript PDF",
}

# Stages that may legitimately fail without stopping the run
SOFT_FAIL = {"download"}


def python_exe() -> str:
    """Return the interpreter for every sub-stage.

    Priority: $SCI_PYTHON_EXE -> paths.PYTHON_EXE (the py311 CUDA env forced
    at scaffold time) -> the interpreter that launched us -> plain 'python'.
    This guarantees a fresh conversation never silently falls back to a
    CPU-only interpreter just because run_all.py was started with it.
    """
    if os.environ.get("SCI_PYTHON_EXE"):
        return os.environ["SCI_PYTHON_EXE"]
    if paths.PYTHON_EXE and os.path.exists(paths.PYTHON_EXE):
        return paths.PYTHON_EXE
    if sys.executable and os.path.exists(sys.executable):
        return sys.executable
    return "python"


def stage_args(stage: str, a: argparse.Namespace) -> List[str]:
    """Translate the global CLI options into per-script arguments."""
    out: List[str] = []
    if stage == "download":
        if a.verify_only:
            out += ["--verify-only"]
        if a.force:
            out += ["--force"]
    elif stage == "build":
        if a.demo:
            out += ["--demo", "--n-internal", str(a.n_internal),
                    "--n-external", str(a.n_external)]
    elif stage == "tune":
        if a.quick:
            out += ["--quick"]
        else:
            out += ["--coarse", str(a.coarse), "--fine", str(a.fine)]
        if a.skip_search:
            out += ["--skip-search"]
    elif stage == "train":
        if os.path.exists(paths.BEST_CONFIG_JSON):
            out += ["--config", paths.BEST_CONFIG_JSON]
        out += ["--tag", "main"]
        if a.quick:
            out += ["--quick"]
        else:
            out += ["--seeds", str(a.seeds)]
    elif stage == "baselines":
        if a.quick:
            out += ["--quick"]
        else:
            out += ["--seeds", str(a.seeds), "--n-iter", str(a.n_iter)]
    elif stage == "interpret":
        if a.quick:
            out += ["--quick"]
        else:
            out += ["--seeds", str(min(a.seeds, 2))]
    elif stage == "gate":
        # In a real run the gate is a hard stop: a model that does not beat
        # the baselines is not allowed to become a manuscript.
        if a.demo or a.quick:
            out += ["--allow-demo"]
        else:
            out += ["--strict"]
    elif stage == "paper":
        if a.topic:
            out += ["--topic", a.topic]
        if a.strict:
            out += ["--strict"]
    return out


def run_stage(stage: str, a: argparse.Namespace) -> Dict[str, object]:
    script = os.path.join(CODE_DIR, SCRIPTS[stage])
    cmd = [python_exe(), script] + stage_args(stage, a)
    print("\n" + "#" * 78)
    print(f"# {LABELS[stage]}")
    print(f"# {' '.join(cmd)}")
    print("#" * 78, flush=True)

    if not os.path.exists(script):
        print(f"  MISSING SCRIPT: {script}")
        return {"stage": stage, "code": 127, "seconds": 0.0,
                "cmd": " ".join(cmd)}

    t0 = time.time()
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.run(cmd, cwd=CODE_DIR, env=env)
    dt = time.time() - t0
    status = "ok" if proc.returncode == 0 else f"FAILED ({proc.returncode})"
    print(f"\n  [{stage}] {status} in {dt:.1f}s", flush=True)
    return {"stage": stage, "code": int(proc.returncode), "seconds": dt,
            "cmd": " ".join(cmd)}


def write_log(records: List[Dict[str, object]], a: argparse.Namespace) -> str:
    path = os.path.join(paths.RESULTS_DIR, "run_log.md")
    total = sum(float(r["seconds"]) for r in records)
    lines = [f"# Run log -- {paths.MODEL_NAME}", "",
             f"- started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"- python  : {python_exe()}",
             f"- root    : {paths.PROJECT_ROOT}",
             f"- mode    : {'demo ' if a.demo else ''}"
             f"{'quick ' if a.quick else ''}".strip() or "full",
             f"- total   : {total/60.0:.1f} min", "",
             "| stage | status | seconds | command |",
             "|---|---|---|---|"]
    for r in records:
        st = "ok" if r["code"] == 0 else f"FAIL {r['code']}"
        lines.append(f"| {r['stage']} | {st} | {float(r['seconds']):.1f} | "
                     f"`{r['cmd']}` |")
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    for _i in range(8):
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            break
        except PermissionError:
            import time as _t
            _t.sleep(4)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return path


def inventory() -> List[str]:
    """List the deliverables that actually exist on disk."""
    items = []
    for label, path in [
        ("dataset", paths.DATASET_NPZ),
        ("external dataset", paths.EXTERNAL_NPZ),
        ("best config", paths.BEST_CONFIG_JSON),
        ("internal CV metrics", paths.CV_OUTER_CSV),
        ("baseline metrics", paths.BASELINES_CSV),
        ("external metrics", paths.EXTERNAL_CSV),
        ("statistics", paths.COMPARISONS_CSV),
        ("perf gate", os.path.join(paths.METRICS_DIR, "perf_gate.json")),
        ("importance", paths.IMPORTANCE_CSV),
        ("manuscript (md)", paths.MANUSCRIPT_MD),
        ("manuscript (pdf)", paths.MANUSCRIPT_PDF),
    ]:
        mark = "OK " if os.path.exists(path) else "-- "
        items.append(f"  {mark} {label:<22s} {path}")
    n_fig = len([f for f in os.listdir(paths.FIGURES_DIR)
                 if f.endswith(".png")]) if os.path.isdir(paths.FIGURES_DIR) else 0
    n_tab = len([f for f in os.listdir(paths.TABLES_DIR)
                 if f.endswith(".csv")]) if os.path.isdir(paths.TABLES_DIR) else 0
    items.append(f"  {'OK ' if n_fig >= 8 else '-- '} figures (png)"
                 f"{'':<9s} {n_fig} in {paths.FIGURES_DIR}")
    items.append(f"  {'OK ' if n_tab >= 8 else '-- '} tables (csv)"
                 f"{'':<10s} {n_tab} in {paths.TABLES_DIR}")
    return items


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    sel = ap.add_argument_group("stage selection")
    sel.add_argument("--all", action="store_true", help="run every stage")
    sel.add_argument("--only", nargs="*", default=None,
                     metavar="STAGE", help=f"run only these: {STAGE_ORDER}")
    sel.add_argument("--from", dest="from_stage", default=None,
                     metavar="STAGE", help="run from this stage to the end")
    sel.add_argument("--skip", nargs="*", default=[], metavar="STAGE")

    mode = ap.add_argument_group("mode")
    mode.add_argument("--demo", action="store_true",
                      help="synthetic data (smoke test, never publishable)")
    mode.add_argument("--quick", action="store_true",
                      help="tiny budgets everywhere (smoke test)")
    mode.add_argument("--continue-on-error", action="store_true")
    mode.add_argument("--dry-run", action="store_true")

    bud = ap.add_argument_group("budgets")
    bud.add_argument("--seeds", type=int, default=len(paths.SEEDS))
    bud.add_argument("--coarse", type=int, default=60)
    bud.add_argument("--fine", type=int, default=40)
    bud.add_argument("--n-iter", type=int, default=40,
                     help="randomised-search budget per baseline")
    bud.add_argument("--n-internal", type=int, default=420)
    bud.add_argument("--n-external", type=int, default=160)
    bud.add_argument("--skip-search", action="store_true")

    misc = ap.add_argument_group("misc")
    misc.add_argument("--topic", default="")
    misc.add_argument("--strict", action="store_true",
                      help="manuscript self-check failures end the run")
    misc.add_argument("--verify-only", action="store_true")
    misc.add_argument("--force", action="store_true")

    a = ap.parse_args()

    # ------------------------- resolve the plan ------------------------ #
    if a.only:
        plan = [s for s in STAGE_ORDER if s in set(a.only)]
        unknown = set(a.only) - set(STAGE_ORDER)
        if unknown:
            raise SystemExit(f"unknown stage(s): {sorted(unknown)}")
    elif a.from_stage:
        if a.from_stage not in STAGE_ORDER:
            raise SystemExit(f"unknown stage: {a.from_stage}")
        plan = STAGE_ORDER[STAGE_ORDER.index(a.from_stage):]
    elif a.all:
        plan = list(STAGE_ORDER)
    else:
        ap.print_help()
        print("\nNothing to do: pass --all, --only or --from.")
        return 0

    if a.demo and "download" in plan:
        plan.remove("download")          # nothing to download for synthetic data
    plan = [s for s in plan if s not in set(a.skip)]

    paths.ensure_dirs()
    paths.banner(f"RUN ALL  --  {paths.MODEL_NAME}")
    print(f"  project : {paths.PROJECT_ROOT}")
    print(f"  python  : {python_exe()}")
    print(f"  stages  : {' -> '.join(plan)}")
    if a.demo:
        print("  MODE    : DEMO (synthetic data; results are NOT publishable)")
    if a.quick:
        print("  MODE    : QUICK (reduced budgets; results are NOT publishable)")

    if a.dry_run:
        for s in plan:
            print(f"    {python_exe()} {SCRIPTS[s]} "
                  f"{' '.join(stage_args(s, a))}")
        return 0

    # ----------------------------- execute ----------------------------- #
    records: List[Dict[str, object]] = []
    t0 = time.time()
    failed: Optional[str] = None
    for stage in plan:
        rec = run_stage(stage, a)
        records.append(rec)
        if rec["code"] != 0:
            if stage in SOFT_FAIL or a.continue_on_error:
                print(f"  continuing despite failure in '{stage}'")
                continue
            failed = stage
            break

    log = write_log(records, a)
    paths.banner("SUMMARY")
    for r in records:
        st = "ok  " if r["code"] == 0 else f"FAIL"
        print(f"  {st}  {r['stage']:<10s} {float(r['seconds']):>7.1f}s")
    print(f"\n  total {((time.time() - t0) / 60.0):.1f} min")
    print(f"  log   {log}\n")
    print("  deliverables:")
    for line in inventory():
        print(line)

    if failed == "gate":
        gate_md = os.path.join(paths.PROJECT_ROOT, "audit", "PERF_GATE.md")
        print("\n" + "=" * 78)
        print("  PERF-GATE CLOSED -- the model does not beat the baselines "
              "yet.")
        print("  Writing the manuscript now is NOT an option. Open the "
              "report,")
        print("  apply the escalation level it prints, then re-run:")
        print(f"    {gate_md}")
        print(f"    python run_all.py --from tune")
        print("=" * 78)
        return 1

    if failed:
        print(f"\n  RUN STOPPED at stage '{failed}'. "
              f"Fix it, then resume with:\n"
              f"    python run_all.py --from {failed}")
        return 1
    print("\n  all requested stages completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
