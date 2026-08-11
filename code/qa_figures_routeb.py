"""qwen3.6 vision QA for the 8 Route-B figures.

For each figure (1..8):
  * downscaled preview (already in results/figures_qc/preview/figN.png, ≤1100px)
  * POST to local Ollama qwen36:latest with a vision prompt that asks about
    panel-letter placement, text/broken-axis overlap, Morandi gradient and
    colour distinguishability.
  * collect verdict text and append to results/figures_qc/ollama_vision_review.md

The review is the single source of truth for the per-figure QA verdict
required by the team-lead's R1-4 task.
"""
from __future__ import annotations

import _runtime_guard  # noqa: F401
import base64
import json
import os
import sys
import time
import urllib.request

PROJECT_ROOT = r"C:/Users/TS/WorkBuddy/HydroGelNet"
PREVIEW_DIR = os.path.join(PROJECT_ROOT, "results", "figures_qc", "preview")
REVIEW_PATH = os.path.join(PROJECT_ROOT, "results", "figures_qc",
                           "ollama_vision_review.md")
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen36:latest"

FIGURES = [
    (1, "Fig1_pipeline",
     "Fig 1 (Route-B schematic): 5-column A-E pipeline (Data → Training → "
     "SIMPLEX → BO-acquired batch evaluation → Screening & insight)."),
    (2, "Fig2_architecture",
     "Fig 2 (Route-B architecture): SIMPLEX d=152 / 1 ResBlock / 8 heads / "
     "370,327 params; regularisation band at bottom."),
    (3, "Fig3_dataset",
     "Fig 3 (cohort characterisation): 6 panels; panel A must read "
     "'BO-acquired batch' (not 'Prospective')."),
    (4, "Fig4_internal_cv",
     "Fig 4 (internal 10x5 grouped CV): 9 panels; panel B must show r=0.8944 "
     "and n=316; RMSE 33.145 / MAE 23.748 must appear in panel G."),
    (5, "Fig5_benchmark",
     "Fig 5 (benchmark vs 8 baselines): 6 panels; panel E axis must NOT "
     "contain the word 'best' (must say 'top-ranked'); bar order must be "
     "RF / SVR / SIMPLEX / KNN / Ridge / Enet / HistGB / MLP."),
    (6, "Fig6_external",
     "Fig 6 (BO-acquired batch / external): 9 panels A-I; panel B and C "
     "must show broken-axis bars (// markers) with the dummy long tail "
     "truncated; panel H is a slope chart showing Ridge overtaking SIMPLEX "
     "in Spearman; panel A must show single R²=0.6712 / ρ=0.8031."),
    (7, "Fig7_ablation",
     "Fig 7 (ablation): 9 panels; panels A/B must show the three-category "
     "grouping (10 positive / 10 bit-identical / 3 removal-improves) with "
     "full R²=0.768460; panel H must NOT contain 75% (replaced by LOCO "
     "p=0.805)."),
    (8, "Fig8_interpretation",
     "Fig 8 (interpretation): 8 panels; panel A must show pair_14 "
     "importance 0.0631±0.0361; panel D must NOT have a truncated legend "
     "(the 'conditi...'/'fused_m...' glitch)."),
]

PROMPT = (
    "You are reviewing a scientific figure for an SCI submission. "
    "Check the image carefully against these criteria and reply with "
    "ONE short paragraph (≤ 6 sentences) covering all of:\n"
    "  (a) panel letters (A, B, C, ...) — are they present and placed "
    "      OUTSIDE the axes top-left (no overlap with titles/ticks)?\n"
    "  (b) text overlap — any text labels colliding with other labels, "
    "      panel borders, data marks, or the panel title?\n"
    "  (c) broken-axis markers — for the panels expected to have them, "
    "      are the // marks visible and the axis clearly discontinuous?\n"
    "  (d) Morandi colour distinguishability — is the palette muted "
    "      (blues/lilac/peach/coral/terracotta), and is 'ours' "
      "      (SIMPLEX) clearly distinguishable from baselines?\n"
    "  (e) any other obvious rendering defect (truncated labels, axes "
    "      overflow, missing annotations)?\n"
    "If a criterion PASSES, say so briefly. If it FAILS, point at the "
    "exact panel / text. Do not invent issues."
)


def _b64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("ascii")


def _query(image_path: str, figure_descr: str, retries: int = 3) -> str:
    body = json.dumps({
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": PROMPT,
            "images": [_b64(image_path)],
        }],
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body,
                                 headers={"Content-Type": "application/json"})
    last = ""
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                data = json.loads(r.read().decode("utf-8"))
                return (data.get("message") or {}).get("content", "") \
                    or json.dumps(data)[:1500]
        except Exception as exc:  # noqa: BLE001
            last = "{0}: {1}".format(type(exc).__name__, exc)
            time.sleep(2 + i * 3)
    return "(qa unavailable: {0})".format(last)


def main(argv=None) -> int:
    argv = argv or sys.argv[1:]
    only = None
    if argv:
        only = set(int(x) for x in argv)

    os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
    lines = ["# Ollama qwen3.6 vision QA — Route-B figure redraw (R1-4)\n",
             "Model: `{0}`  ·  Source: local Ollama @ {1}\n"
             .format(MODEL, OLLAMA_URL),
             "Criterion legend: (a) panel letters · (b) text overlap · "
             "(c) broken-axis markers · (d) Morandi palette · "
             "(e) other defects.\n\n---\n"]

    for k, slug, descr in FIGURES:
        if only and k not in only:
            continue
        img = os.path.join(PREVIEW_DIR, "fig{0}.png".format(k))
        if not os.path.exists(img):
            lines.append("\n## Figure {0} ({1}) — preview MISSING\n"
                         .format(k, slug))
            continue
        print("[qa] Figure {0} ...".format(k), flush=True)
        verdict = _query(img, descr)
        lines.append("\n## Figure {0} — {1}\n\n"
                     "Context: {2}\n\n"
                     "Verdict:\n\n> {3}\n".format(k, slug, descr,
                                                verdict.replace("\n",
                                                                "\n> ")))

    with open(REVIEW_PATH, "w", encoding="utf-8") as fh:
        fh.write("".join(lines))
    print("[qa] wrote", REVIEW_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())