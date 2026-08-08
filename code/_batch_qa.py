"""Batch qwen36 vision QA on all 8 figures."""
import _runtime_guard  # noqa
import subprocess
import sys

figs = [
    (3, "dataset"),
    (4, "internal_cv"),
    (5, "benchmark"),
    (6, "external"),
    (7, "ablation"),
    (8, "interpretation"),
]

for i, name in figs:
    print(f"\n{'='*60}\n=== Figure{i} {name} ===\n{'='*60}")
    try:
        out = subprocess.run(
            ["C:/Users/TS/.conda/envs/HydroGelNet/python.exe",
             "-u", "qwen_vision_qa.py",
             f"C:/Users/TS/WorkBuddy/HydroGelNet/figures/Figure{i}_{name}.png",
             "--focus", "overlap"],
            cwd="C:/Users/TS/WorkBuddy/HydroGelNet/code",
            capture_output=True, text=True, timeout=180)
        lines = out.stdout.strip().split("\n")
        # find OVERALL verdict and severity counts
        verdict = next((l for l in lines if l.startswith("OVERALL")), "(no verdict)")
        high = sum(1 for l in lines if "HIGH" in l)
        med = sum(1 for l in lines if "MED" in l)
        print(f"HIGH issues: {high}, MED issues: {med}")
        print(verdict)
        if high + med > 0:
            # print last 10 lines that contain issues
            for l in lines:
                if "HIGH" in l or "MED" in l:
                    print(" ", l.strip()[:130])
    except Exception as e:
        print(f"ERROR: {e}")