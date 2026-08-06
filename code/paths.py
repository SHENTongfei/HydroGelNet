"""Central path and constant registry.

This is the SINGLE source of truth for every path in the project.
All other scripts must import from here. Never hardcode a path elsewhere,
and never use relative paths.

To move the project to another machine, edit PROJECT_ROOT only.
"""

from __future__ import annotations

import _runtime_guard  # noqa: F401  (must be first)
import os
import sys

# --------------------------------------------------------------------------- #
# Core identity
# --------------------------------------------------------------------------- #
PROJECT_ROOT = r"C:/Users/TS/WorkBuddy/HydroGelNet"
MODEL_NAME = "SIMPLEX"
MODEL_SLUG = MODEL_NAME.lower().replace("-", "_")

# The Python interpreter that runs this project. Priority:
#   1. $SCI_PYTHON_EXE              (explicit user override)
#   2. the py311 env injected by new_project.py -- this machine's ONLY
#      ready-made CUDA environment (torch 2.14.0+cu130, RTX 5080). It is
#      FORCED so that a fresh conversation always lands on GPU, never on a
#      CPU-only interpreter.
#   3. sys.executable               (whatever interpreter imported this file)
PYTHON_EXE = (os.environ.get("SCI_PYTHON_EXE")
              or (r"C:/Users/TS/.conda/envs/HydroGelNet/python.exe"
                  if os.path.exists(r"C:/Users/TS/.conda/envs/HydroGelNet/python.exe") else None)
              or sys.executable
              or "python")


def _detect_device() -> str:
    """Return 'cuda' when a usable GPU is present, else 'cpu'.

    Override at runtime with the environment variable SCI_DEVICE=cuda|cpu.
    The import of torch happens exactly once, at module load, so every script
    shares the same decision (and prints it in its banner).
    """
    override = os.environ.get("SCI_DEVICE", "").strip().lower()
    if override in ("cuda", "gpu"):
        return "cuda"
    if override in ("cpu",):
        return "cpu"
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


DEVICE = _detect_device()

# --------------------------------------------------------------------------- #
# Directory layout (absolute paths only)
# --------------------------------------------------------------------------- #
CODE_DIR = os.path.join(PROJECT_ROOT, "code")

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
INTERIM_DIR = os.path.join(DATA_DIR, "interim")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR = os.path.join(DATA_DIR, "external")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
PREDS_DIR = os.path.join(RESULTS_DIR, "preds")
ABLATION_DIR = os.path.join(RESULTS_DIR, "ablation")
TUNING_DIR = os.path.join(RESULTS_DIR, "tuning")
INTERPRET_DIR = os.path.join(RESULTS_DIR, "interpret")
STATS_DIR = os.path.join(RESULTS_DIR, "stats")

FIGURES_DIR = os.path.join(PROJECT_ROOT, "figures")
TABLES_DIR = os.path.join(PROJECT_ROOT, "tables")
PAPER_DIR = os.path.join(PROJECT_ROOT, "paper")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

ALL_DIRS = [
    CODE_DIR, DATA_DIR, RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EXTERNAL_DIR,
    RESULTS_DIR, METRICS_DIR, PREDS_DIR, ABLATION_DIR, TUNING_DIR,
    INTERPRET_DIR, STATS_DIR, FIGURES_DIR, TABLES_DIR, PAPER_DIR, MODELS_DIR,
]

# --------------------------------------------------------------------------- #
# Canonical file names
# --------------------------------------------------------------------------- #
DATASET_NPZ = os.path.join(PROCESSED_DIR, "dataset.npz")
EXTERNAL_NPZ = os.path.join(EXTERNAL_DIR, "dataset_external.npz")

CV_OUTER_CSV = os.path.join(METRICS_DIR, "cv_outer.csv")
BASELINES_CSV = os.path.join(METRICS_DIR, "baselines.csv")
EXTERNAL_CSV = os.path.join(METRICS_DIR, "external.csv")
ABLATION_CSV = os.path.join(ABLATION_DIR, "ablation.csv")
SEARCH_LOG_CSV = os.path.join(TUNING_DIR, "search_log.csv")
BEST_CONFIG_JSON = os.path.join(TUNING_DIR, "best_config.json")
COMPARISONS_CSV = os.path.join(STATS_DIR, "comparisons.csv")
IMPORTANCE_CSV = os.path.join(INTERPRET_DIR, "importance.csv")
LATENT_NPZ = os.path.join(INTERPRET_DIR, "latent.npz")
DATA_SOURCES_MD = os.path.join(PROJECT_ROOT, "DATA_SOURCES.md")
MANUSCRIPT_MD = os.path.join(PAPER_DIR, "manuscript.md")
MANUSCRIPT_PDF = os.path.join(PAPER_DIR, MODEL_NAME + "_manuscript.pdf")

# --------------------------------------------------------------------------- #
# Global experiment constants
# --------------------------------------------------------------------------- #
SEEDS = [42, 2024, 7, 1337, 20260731]
PRIMARY_SEED = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
N_REPEATS = 3
BOOTSTRAP_N = 2000
DEVICE = _detect_device()  # 'cuda' if a GPU is present, else 'cpu'; override with $SCI_DEVICE

# Figure export settings
FIG_DPI = 600
FIG_FORMATS = ("png", "pdf")


def ensure_dirs() -> None:
    """Create every project directory if it does not exist."""
    for directory in ALL_DIRS:
        os.makedirs(directory, exist_ok=True)


def banner(title: str) -> None:
    """Print a visually distinct section header to stdout."""
    line = "=" * 78
    print("\n" + line)
    print(title)
    print(line, flush=True)


if __name__ == "__main__":
    ensure_dirs()
    banner("PATH REGISTRY")
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("MODEL_NAME   :", MODEL_NAME)
    print("PYTHON_EXE   :", PYTHON_EXE)
    print("DEVICE       :", DEVICE)
    print("python       :", sys.version.split()[0])
    for d in ALL_DIRS:
        print("  [dir]", d, "OK" if os.path.isdir(d) else "MISSING")
