# -*- coding: utf-8 -*-
"""qwen36 visual QA on the 6 re-rendered figures (anti-overlap focus)."""
import base64, json, sys, time, urllib.request, os

OLLAMA = "http://localhost:11434/api/chat"

def qa_one(path, focus="overlap"):
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = {
        "model": "qwen36:latest",
        "messages": [{
            "role": "user",
            "content": (
                "You are a scientific figure QA inspector. Focus ONLY on: "
                "(1) text-label/data-point overlap or text-text overlap; "
                "(2) labels or dots clipped outside axes; "
                "(3) titles/axis labels too long and colliding with panel letters or other panels; "
                "(4) unreadable tiny text. "
                "Reply with a verdict line starting OVERALL: PASS or OVERALL: FAIL, "
                "then list issues as [HIGH]/[MED]/[LOW] with panel letter and one-line description. "
                "Be strict: any visible overlap is at least MED."
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
    # ollama may stream multiple JSON lines; join them
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
    figs_dir = r"C:\Users\TS\WorkBuddy\HydroGelNet\figures\_qa2"
    names = {
        3: "f3.png", 4: "f4.png",
        5: "f5.png", 6: "f6.png",
        7: "f7.png", 8: "f8.png",
    }
    only = sys.argv[1:] if len(sys.argv) > 1 else [str(k) for k in names]
    for k in only:
        p = os.path.join(figs_dir, names[int(k)])
        print(f"### Figure{k} ###", flush=True)
        try:
            out = qa_one(p)
            print(out, flush=True)
        except Exception as e:
            print(f"ERROR {e}", flush=True)
        time.sleep(1)
