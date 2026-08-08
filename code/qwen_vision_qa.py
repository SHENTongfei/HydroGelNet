"""qwen36 vision QA helper - sends a figure PNG to local Ollama qwen36 for audit.

Usage:
    python qwen_vision_qa.py <image_path> [--focus overlap|info|style|all]
"""
import _runtime_guard  # noqa: F401
import argparse
import base64
import json
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "qwen36:latest"

QA_TEMPLATES = {
    "overlap": (
        "You are a ruthless scientific figure quality auditor. Examine this figure "
        "panel-by-panel. Report STRICTLY:\n"
        "1. TEXT-OVERLAP: any text overlapping other text (labels, legends, ticks, "
        "panel letters, titles colliding) - list panel letter and exact collision.\n"
        "2. TEXT-DATA-OVERLAP: any text sitting on top of plotted data, bars, lines, "
        "heatmap cells making it unreadable.\n"
        "3. PANEL-LETTER position: are the bold panel letters (A,B,C...) OUTSIDE the "
        "axes top-left corner, not colliding with titles or adjacent panels?\n"
        "4. CROPPED/CLIPPED text: any tick label, axis label or legend cut off at "
        "figure edge.\n"
        "For each finding give panel letter + severity (HIGH/MED/LOW). "
        "End with OVERALL: PASS or FAIL."
    ),
    "info": (
        "You are a ruthless scientific figure auditor. For EACH subplot panel in this "
        "multi-panel figure, answer:\n"
        "1. What information does this panel convey? (one sentence)\n"
        "2. Is the message CLEAR and UNAMBIGUOUS, or is it confusing/meaningless?\n"
        "3. Does the panel clearly show a comparison/contrast, or do bars/lines look "
        "nearly identical making differences invisible?\n"
        "4. Is the chart type appropriate (bar/heatmap/violin/line/scatter) for the data, "
        "or would a different type show the story better?\n"
        "List per panel A,B,C,... End with PANELS_WITH_WEAK_MESSAGE: <list> and "
        "SUGGESTED_UPGRADES: <chart types>."
    ),
    "style": (
        "You are a Nature/Science-level figure style auditor. Assess this figure:\n"
        "1. COLOR: is the palette elegant, professional, high-end journal quality? "
        "Any garish/vivid/ugly colors? Are our-model colors clearly distinguished?\n"
        "2. DENSITY: is it too plain/bland (too much white space, sparse) or too busy?\n"
        "3. SPACING: are subplot gaps even? Any panel cramped or oversized?\n"
        "4. Overall: does this look like a top-tier international journal figure? "
        "Rate 1-10 and list specific improvements."
    ),
}

def qa(image_path: str, focus: str = "all", max_img_px: int = 1600) -> str:
    with open(image_path, "rb") as fh:
        raw = fh.read()
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw))
        img.thumbnail((max_img_px, max_img_px))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        raw = buf.getvalue()
    except Exception:
        pass
    b64 = base64.b64encode(raw).decode()
    prompt = "\n\n".join([QA_TEMPLATES[f] for f in
                          (["overlap", "info", "style"] if focus == "all" else [focus])])
    payload = {
        "model": MODEL,
        "messages": [{
            "role": "user",
            "content": prompt,
            "images": [b64],
        }],
        "think": False,
        "stream": False,
    }
    req = urllib.request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=300))
    return resp.get("message", {}).get("content", "(empty)")

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--focus", default="all",
                    choices=["overlap", "info", "style", "all"])
    ap.add_argument("--max-px", type=int, default=1600)
    args = ap.parse_args()
    print(qa(args.image, args.focus, args.max_px), flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
