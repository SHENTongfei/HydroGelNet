import _runtime_guard  # noqa
import base64
import json
import io
import os
import sys
import urllib.request
from PIL import Image

OLLAMA = "http://localhost:11434/api/chat"
MODEL = "qwen36:latest"

QA_PROMPT = (
    "Audit this scientific figure for a Nature-level journal. Strict checklist:\n"
    "1. TEXT-OVERLAP (text on text): list panel + HIGH/MED/LOW.\n"
    "2. TEXT-DATA-OVERLAP (text on data points/bars/lines): list panel + severity.\n"
    "3. PANEL-LETTER position: letters A-I outside axes top-left, no collision? PASS/FAIL.\n"
    "4. CROPPED text at figure edge? list.\n"
    "5. STYLE palette: elegant consistent (warm orange ours, cool blue baseline)? Rate 1-10.\n"
    "6. CHART-VARIETY: advanced types (lollipop/heatmap/violin/slope/forest) vs plain bars?\n"
    "End: OVERALL: PASS or FAIL, then TOP-3 FIXES. Keep each answer under 25 words."
)


def qa_one(path):
    img = Image.open(path)
    img.thumbnail((1400, 1400))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": QA_PROMPT, "images": [b64]}],
        "think": False,
        "stream": False,
    }
    req = urllib.request.Request(OLLAMA, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=600))
    return resp.get("message", {}).get("content", "(empty)")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    FIGDIR = r"C:/Users/TS/WorkBuddy/HydroGelNet/figures"
    figs = [
        "Figure3_dataset", "Figure4_internal_cv", "Figure5_benchmark",
        "Figure6_external", "Figure7_ablation", "Figure8_interpretation",
    ]
    if which != "all":
        figs = [f for f in figs if which in f]
    for name in figs:
        p = os.path.join(FIGDIR, name + ".png")
        print(f"\n### {name}", flush=True)
        try:
            out = qa_one(p)
            print(out[:1200], flush=True)
        except Exception as e:
            print(f"ERROR: {e}", flush=True)