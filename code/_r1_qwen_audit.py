# -*- coding: utf-8 -*-
"""R1 Final QA: qwen36 vision audit of all 8 main figures (H35 Phase 14.5).
Append mode; smaller images (720px) for qwen36 reliability."""
import base64, json, sys, time, urllib.request, os

OLLAMA = "http://localhost:11434/api/chat"

def qa_one(path):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": "qwen36:latest",
        "messages": [{
            "role": "user",
            "content": (
                "You are a final scientific figure auditor. Check each panel: "
                "1) panel letters A-I positioned OUTSIDE the top-left of each subplot axis "
                "(in the whitespace above the axis, NOT inside the plotting area); "
                "2) any text-label/data-point overlap or text-text overlap; "
                "3) clipped labels/dots outside axes; "
                "4) unreadable tiny text; "
                "5) empty/blank panels. "
                "Reply with OVERALL: PASS or OVERALL: FAIL, then list [HIGH]/[MED]/[LOW] "
                "with panel letter and one-line description. Be strict."
            ),
            "images": [b64],
        }],
        "think": False,
        "num_predict": 400,
        "temperature": 0.1,
        "keep_alive": "10m",
    }
    req = urllib.request.Request(
        OLLAMA, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        raw = resp.read().decode()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        texts = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line).get("message", {}).get("content", ""))
            except Exception:
                pass
        return "\n".join(texts).strip()
    return data.get("message", {}).get("content", "").strip()

if __name__ == "__main__":
    figs_dir = r"C:\Users\TS\WorkBuddy\HydroGelNet\results\audit\_r1_imgs"
    names = {1: "f1.png", 2: "f2.png", 3: "f3.png", 4: "f4.png",
             5: "f5.png", 6: "f6.png", 7: "f7.png", 8: "f8.png"}
    outdir = r"C:\Users\TS\WorkBuddy\HydroGelNet\results\audit"
    os.makedirs(outdir, exist_ok=True)
    only = sys.argv[1:] if len(sys.argv) > 1 else [str(k) for k in names]
    report_path = os.path.join(outdir, "qwen_final_audit.md")
    with open(report_path, "a", encoding="utf-8") as rp:
        for k in only:
            p = os.path.join(figs_dir, names[int(k)])
            print(f"### Figure{k} ###", flush=True)
            try:
                out = qa_one(p)
            except Exception as e:
                out = f"ERROR {e}"
            print(out, flush=True)
            rp.write(f"## Figure {k} ({names[int(k)]})\n\n")
            rp.write(out.replace("\n", "  \n") + "\n\n---\n\n")
            time.sleep(1)
    print(f"\nreport -> {report_path}")
