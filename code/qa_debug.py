import base64, json, urllib.request, traceback, time
img = r"C:/Users/TS/WorkBuddy/HydroGelNet/results/figures_qc/preview/fig1.png"
print("opening", img)
try:
    with open(img, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    print("b64 len:", len(b64))
except Exception:
    traceback.print_exc(); raise

prompt = "Look at this figure. In 2 short sentences: (1) are panel letters A B C D E visible OUTSIDE the axes top-left? (2) is any text overlapping other labels or panel titles?"
body = json.dumps({
    "model": "qwen36:latest",
    "messages": [{"role": "user", "content": prompt, "images": [b64]}],
    "stream": False,
}).encode()
print("POST", len(body), "bytes")
t0 = time.time()
req = urllib.request.Request("http://localhost:11434/api/chat", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=300) as r:
    raw = r.read().decode()
print("got response in", round(time.time()-t0, 1), "s, len", len(raw))
data = json.loads(raw)
print("keys:", list(data.keys()))
content = (data.get("message") or {}).get("content", "")
print("CONTENT:")
print(content)